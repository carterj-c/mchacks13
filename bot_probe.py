"""
Diagnostic Probe Bot
====================
A specialized tool to 'grope around in the dark' and map the hidden liquidity
structure of the exchange simulator.

It cycles through various pricing strategies (offsets) to determine:
1. Is the tick size truly 0.25?
2. Can we front-run bots by 0.01?
3. Where does the 'Noise' trader volume actually hit?
"""

import json
import websocket
import threading
import argparse
import time
import requests
import ssl
import urllib3
import math
import csv
from typing import Dict, Optional, List, Tuple

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class ProbeBot:
    def __init__(
        self,
        student_id: str,
        host: str,
        scenario: str,
        password: str = None,
        secure: bool = False,
    ):
        self.student_id = student_id
        self.host = host
        self.scenario = scenario
        self.password = password
        self.secure = secure

        self.http_proto = "https" if secure else "http"
        self.ws_proto = "wss" if secure else "ws"

        self.token = None
        self.run_id = None

        # Trading state
        self.inventory = 0
        self.cash_flow = 0.0
        self.pnl = 0.0
        self.current_step = 0
        self.orders_sent = 0

        # Market data
        self.last_bid = 0.0
        self.last_ask = 0.0
        self.last_mid = 0.0

        # Connection state
        self.market_ws = None
        self.order_ws = None
        self.running = True

        # ------------------------------------------------------------------
        # DIAGNOSTIC STATE
        # ------------------------------------------------------------------
        # Probes to test: (Description, Offset from Bid)
        # 0.00 = Join the Queue
        # 0.01 = Penny Jump
        # 0.05 = Nickel Jump
        # 0.12 = Mid-ish
        # 0.25 = Full Tick Jump
        self.probe_schedule = [0.35, 0.4, 0.45, 0.5, 0.75, 1.0]
        self.current_probe_idx = 0

        self.active_probe = None  # {type: "BUY", price: float, offset: float, start_step: int, order_id: str}
        self.probe_results = []  # Log results here
        self.last_known_inventory = 0
        self.order_limit_max = 100  # Override limit

    # =========================================================================
    # CORE INFRASTRUCTURE (Registration & Connection)
    # =========================================================================

    def register(self) -> bool:
        print(f"[{self.student_id}] Registering for scenario '{self.scenario}'...")
        try:
            # Add delay to allow server to warm up if needed, though usually not for local
            url = f"{self.http_proto}://{self.host}/api/replays/{self.scenario}/start"
            headers = {"Authorization": f"Bearer {self.student_id}"}
            if self.password:
                headers["X-Team-Password"] = self.password

            resp = requests.get(
                url, headers=headers, timeout=10, verify=not self.secure
            )

            if resp.status_code != 200:
                print(f"[{self.student_id}] Registration FAILED: {resp.text}")
                return False

            data = resp.json()
            self.token = data.get("token")
            self.run_id = data.get("run_id")

            if not self.token or not self.run_id:
                print(f"[{self.student_id}] Missing token or run_id")
                return False

            print(f"[{self.student_id}] Registered! Run ID: {self.run_id}")
            return True
        except Exception as e:
            print(f"[{self.student_id}] Registration error: {e}")
            return False

    def connect(self) -> bool:
        try:
            sslopt = {"cert_reqs": ssl.CERT_NONE} if self.secure else None

            market_url = (
                f"{self.ws_proto}://{self.host}/api/ws/market?run_id={self.run_id}"
            )
            self.market_ws = websocket.WebSocketApp(
                market_url,
                on_message=self._on_market_data,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=lambda ws: print(
                    f"[{self.student_id}] Market stream connected"
                ),
            )

            order_url = f"{self.ws_proto}://{self.host}/api/ws/orders?token={self.token}&run_id={self.run_id}"
            self.order_ws = websocket.WebSocketApp(
                order_url,
                on_message=self._on_order_response,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=lambda ws: print(f"[{self.student_id}] Order stream connected"),
            )

            threading.Thread(
                target=lambda: self.market_ws.run_forever(sslopt=sslopt), daemon=True
            ).start()
            threading.Thread(
                target=lambda: self.order_ws.run_forever(sslopt=sslopt), daemon=True
            ).start()
            time.sleep(1)
            return True
        except Exception as e:
            print(f"[{self.student_id}] Connection error: {e}")
            return False

    # =========================================================================
    # THE BRAIN: PROBE LOGIC
    # =========================================================================

    def decide_order(self, bid: float, ask: float, mid: float) -> Optional[Dict]:
        """
        Executes the 'Grope in the Dark' strategy.
        1. Unwind if inventory gets too high (reset).
        2. If no probe active, launch next probe from schedule.
        3. If probe active, check timeout.
        """
        # 0. Safety Checks
        if bid <= 0 or ask <= 0:
            return None

        # 1. Unwind Phase (Reset to 0)
        # If we successfully probed and got filled, we have inventory.
        # Dump it fast so we can test the next offset clean.
        if abs(self.inventory) >= 1:  # Unwind immediately
            if self.inventory > 0:
                # Sell at Bid (Market Sell)
                return {"side": "SELL", "price": bid, "qty": abs(self.inventory)}
            else:
                # Buy at Ask (Market Buy)
                return {"side": "BUY", "price": ask, "qty": abs(self.inventory)}

        # 2. Check Active Probe Status
        if self.active_probe:
            # Check for timeout (e.g., 50 steps without fill)
            steps_waited = self.current_step - self.active_probe["start_step"]
            if steps_waited > 50:
                print(
                    f"[{self.student_id}] PROBE FAILED (Timeout): Offset +{self.active_probe['offset']:.2f} | Price {self.active_probe['price']:.2f}"
                )
                self.probe_results.append(
                    {
                        "offset": self.active_probe["offset"],
                        "result": "TIMEOUT",
                        "steps": steps_waited,
                    }
                )
                # Move to next probe
                self.current_probe_idx = (self.current_probe_idx + 1) % len(
                    self.probe_schedule
                )
                self.active_probe = None

            # If not timed out, we just wait. The _on_order_response handles the Success case.
            return None

        # 3. Launch New Probe
        offset = self.probe_schedule[self.current_probe_idx]

        # Logic: Test BUY side.
        # Price = Bid + Offset
        probe_price = round(bid + offset, 2)

        # Don't cross the spread (unless offset is huge, which shouldn't happen in this test)
        if probe_price >= ask:
            probe_price = ask - 0.01  # Cap at just under Ask

        qty = 100  # Standard lot size

        print(
            f"[{self.student_id}] LAUNCHING PROBE: Offset +{offset:.2f} | Bid {bid:.2f} -> Order {probe_price:.2f}"
        )

        self.active_probe = {
            "type": "BUY",
            "price": probe_price,
            "offset": offset,
            "start_step": self.current_step,
            "bid_at_start": bid,
        }

        return {"side": "BUY", "price": probe_price, "qty": qty}

    # =========================================================================
    # HANDLERS
    # =========================================================================

    def _on_market_data(self, ws, message: str):
        try:
            data = json.loads(message)
            if data.get("type") == "CONNECTED":
                return

            self.current_step = data.get("step", 0)
            self.last_bid = data.get("bid", 0.0)
            self.last_ask = data.get("ask", 0.0)
            self.last_mid = (
                (self.last_bid + self.last_ask) / 2
                if (self.last_bid and self.last_ask)
                else 0
            )

            # Log Market Data occasionally
            if self.current_step % 100 == 0:
                print(
                    f"Step {self.current_step}: {self.last_bid:.2f} / {self.last_ask:.2f} | Spread: {self.last_ask - self.last_bid:.2f}"
                )

            order = self.decide_order(self.last_bid, self.last_ask, self.last_mid)
            if order:
                self._send_order(order)
            self._send_done()

        except Exception as e:
            print(f"Market Msg Error: {e}")

    def _send_order(self, order: Dict):
        order_id = f"PRB_{self.current_step}_{self.orders_sent}"
        msg = {
            "order_id": order_id,
            "side": order["side"],
            "price": order["price"],
            "qty": order["qty"],
        }
        try:
            self.order_ws.send(json.dumps(msg))
            self.orders_sent += 1
        except Exception as e:
            print(f"Order Send Error: {e}")

    def _send_done(self):
        try:
            self.order_ws.send(json.dumps({"action": "DONE"}))
        except:
            pass

    def _on_order_response(self, ws, message: str):
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "FILL":
                qty = data.get("qty", 0)
                price = data.get("price", 0)
                side = data.get("side", "")

                # Update State
                if side == "BUY":
                    self.inventory += qty
                else:
                    self.inventory -= qty

                print(
                    f"[{self.student_id}] >>> FILL DETECTED: {side} {qty} @ {price:.2f}"
                )

                # ANALYZE PROBE RESULT
                if self.active_probe and side == self.active_probe["type"]:
                    # Calculate how long it took
                    duration = self.current_step - self.active_probe["start_step"]
                    print(f"[{self.student_id}] *** PROBE SUCCESS ***")
                    print(f"    Offset: +{self.active_probe['offset']:.2f}")
                    print(f"    Price:  {price:.2f}")
                    print(f"    Wait:   {duration} steps")

                    self.probe_results.append(
                        {
                            "offset": self.active_probe["offset"],
                            "result": "SUCCESS",
                            "steps": duration,
                            "price": price,
                        }
                    )

                    # Move to next probe
                    self.current_probe_idx = (self.current_probe_idx + 1) % len(
                        self.probe_schedule
                    )
                    self.active_probe = None  # Reset

            elif msg_type == "ERROR":
                print(f"[{self.student_id}] EXCHANGE ERROR: {data.get('message')}")

        except Exception as e:
            print(f"Order Response Error: {e}")

    def _on_error(self, ws, error):
        if self.running:
            print(f"WS Error: {error}")

    def _on_close(self, ws, status, msg):
        self.running = False
        print(f"Closed: {status}")

    def run(self):
        if not self.register():
            return
        if not self.connect():
            return

        print(
            f"[{self.student_id}] PROBE STARTED. Cycling offsets: {self.probe_schedule}"
        )
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.running = False
            if self.market_ws:
                self.market_ws.close()
            if self.order_ws:
                self.order_ws.close()

            # Print Summary Report
            print("\n" + "=" * 40)
            print("       LIQUIDITY PROBE REPORT       ")
            print("=" * 40)
            print(f"{'OFFSET':<10} | {'RESULT':<10} | {'STEPS':<10} | {'PRICE':<10}")
            print("-" * 46)

            # Save to CSV
            filename = f"probe_results_2{self.scenario}.csv"
            try:
                with open(filename, mode="w", newline="") as file:
                    writer = csv.DictWriter(
                        file, fieldnames=["offset", "result", "steps", "price"]
                    )
                    writer.writeheader()
                    for res in self.probe_results:
                        p_str = f"{res.get('price', 0):.2f}"
                        print(
                            f"+{res['offset']:<9.2f} | {res['result']:<10} | {res['steps']:<10} | {p_str:<10}"
                        )
                        writer.writerow(res)
                print("=" * 40)
                print(f"Results saved to {filename}")
            except Exception as e:
                print(f"Error saving results: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--scenario", default="normal_market")
    parser.add_argument("--host", default="localhost:8080")
    parser.add_argument("--secure", action="store_true")
    args = parser.parse_args()

    bot = ProbeBot(args.name, args.host, args.scenario, args.password, args.secure)
    bot.run()
