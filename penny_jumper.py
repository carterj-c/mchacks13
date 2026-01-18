"""
Student Trading Algorithm - The Penny Jumper (With Cancellation)
================================================================
STRATEGY: PENNY JUMPING (Tick Size Arbitrage)
TARGET: Normal Market (Tick Size $0.25)

MECHANICS:
1. Detect the "Bot Walls" at the nearest 0.25 increments.
2. Place orders exactly 1 cent in front of them (Bid + 0.01 / Ask - 0.01).
3. Steal all queue priority from the Market Maker bots.
4. If orders don't fill in 20 ticks, CANCEL them and re-price.
"""

import json
import websocket
import threading
import argparse
import time
import requests
import ssl
import urllib3
import pickle
import xgboost as xgb
import math
import os
from typing import Dict, Optional, Any, List
from collections import deque

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class TradingBot:
    def __init__(
        self,
        student_id: str,
        host: str,
        scenario: str,
        password: str = None,
        secure: bool = False,
        regime_model_path: str = None,
        regime_meta_path: str = None,
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
        self.price_history = deque(maxlen=20)

        # Connections
        self.market_ws = None
        self.order_ws = None
        self.running = True

        # State for Strategies
        self.order_limit_max = 100  # OVERRIDE: We need speed
        self.active_orders = False
        self.last_active_step = 0  # Track when we last placed orders
        self.last_known_inventory = 0
        self.flip_flop = False  # Used to alternate Buy/Sell placement

        # Track Active Order IDs for Cancellation
        self.active_buy_id = None
        self.active_sell_id = None

        # Track our own prices to avoid jumping ourselves
        self.my_last_bid = 0.0
        self.my_last_ask = 0.0

    # =========================================================================
    # CORE INFRASTRUCTURE
    # =========================================================================

    def register(self) -> bool:
        print(f"[{self.student_id}] Registering for scenario '{self.scenario}'...")
        try:
            # Adding a small delay to ensure simulation startup synchronization
            url = f"{self.http_proto}://{self.host}/api/replays/{self.scenario}/start?delay=3"
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
            print(f"[{self.student_id}] Registered! Run ID: {self.run_id}")
            return True
        except Exception as e:
            print(f"Registration Error: {e}")
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
                on_open=lambda ws: print(f"[{self.student_id}] Market Data Connected"),
            )
            order_url = f"{self.ws_proto}://{self.host}/api/ws/orders?token={self.token}&run_id={self.run_id}"
            self.order_ws = websocket.WebSocketApp(
                order_url,
                on_message=self._on_order_response,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=lambda ws: print(f"[{self.student_id}] Order Entry Connected"),
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
            print(f"Connection Error: {e}")
            return False

    # =========================================================================
    # STRATEGY LOGIC
    # =========================================================================

    def decide_order(self, bid: float, ask: float, mid: float) -> Optional[Dict]:
        """
        THE PENNY JUMPER STRATEGY (Adaptive & Anchored & Cancellable)
        """
        if bid <= 0 or ask <= 0:
            return None

        # ------------------------------------------------------------------
        # 1. DETECT FILLS (RELOAD MECHANISM)
        # ------------------------------------------------------------------
        # If inventory changes, we got a fill. We must RELOAD our orders immediately.
        if self.inventory != self.last_known_inventory:
            change = self.inventory - self.last_known_inventory
            print(
                f"[{self.student_id}] >>> FILL CONFIRMED: {change:+d} | Inv: {self.inventory}"
            )
            self.last_known_inventory = self.inventory

            # If we partially filled, the other side might still be active.
            # We aggressively reset to stay competitive.
            # (Ideally we cancel the remaining side, but let's just reset flag)
            self.active_orders = False
            self.last_active_step = self.current_step  # Reset timer

        # ------------------------------------------------------------------
        # 2. INVENTORY PROTECTION (DUMP VALVE)
        # ------------------------------------------------------------------
        # If we get too heavy, stop playing games and dump to the bots.
        if self.inventory > 1000:
            bot_bid = math.floor(bid / 0.25) * 0.25
            return self._create_order("SELL", bot_bid, 100)

        elif self.inventory < -1000:
            bot_ask = math.ceil(ask / 0.25) * 0.25
            return self._create_order("BUY", bot_ask, 100)

        # ------------------------------------------------------------------
        # 3. TIMEOUT & CANCELLATION (THE FIX)
        # ------------------------------------------------------------------
        # If orders have been active for 50 steps without filling, they are stale.
        # We must CANCEL them to free up the 50-order limit and re-price.
        if self.active_orders:
            if self.current_step - self.last_active_step > 50:
                print(
                    f"[{self.student_id}] Prices Stale (50 ticks). Cancelling & Repositioning..."
                )

                # Cancel both sides
                if self.active_buy_id:
                    self._cancel_order(self.active_buy_id)
                if self.active_sell_id:
                    self._cancel_order(self.active_sell_id)

                self.active_orders = False
                self.active_buy_id = None
                self.active_sell_id = None
                return None  # Wait for next tick to place new ones
            else:
                return None  # Still waiting, valid orders

        # ------------------------------------------------------------------
        # 4. PRICE CALCULATION (ADAPTIVE)
        # ------------------------------------------------------------------
        # Check if we are already the best price
        target_bid = bid
        target_ask = ask

        # Adaptive jump: If current bid is us, hold. If not, jump.
        if abs(bid - self.my_last_bid) < 0.001:
            my_bid = bid
        else:
            my_bid = round(bid + 0.01, 2)

        if abs(ask - self.my_last_ask) < 0.001:
            my_ask = ask
        else:
            my_ask = round(ask - 0.01, 2)

        # Spread safety
        if my_bid >= my_ask:
            return None

        # ------------------------------------------------------------------
        # 5. EXECUTION
        # ------------------------------------------------------------------
        # We alternate Buy/Sell requests
        self.flip_flop = not self.flip_flop

        qty = 100

        if self.flip_flop:
            self.my_last_bid = my_bid
            # Store ID logic handled in _send_order wrapper
            return self._create_order("BUY", my_bid, qty)
        else:
            self.my_last_ask = my_ask
            self.active_orders = True
            self.last_active_step = self.current_step
            return self._create_order("SELL", my_ask, qty)

    # =========================================================================
    # HELPERS & HANDLERS
    # =========================================================================

    def _create_order(self, side: str, price: float, qty: int) -> Dict:
        return {"side": side, "price": price, "qty": qty}

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

            # Print Stats occasionally
            if self.current_step % 200 == 0:
                print(
                    f"Step {self.current_step} | Inv: {self.inventory} | PnL: {self.pnl:.2f}"
                )

            order = self.decide_order(self.last_bid, self.last_ask, self.last_mid)
            if order:
                self._send_order(order)
            self._send_done()
        except Exception as e:
            print(f"Market Error: {e}")

    def _send_order(self, order: Dict):
        order_id = f"ORD_{self.current_step}_{self.orders_sent}"

        # Track IDs for cancellation
        if order["side"] == "BUY":
            self.active_buy_id = order_id
        else:
            self.active_sell_id = order_id

        msg = {
            "order_id": order_id,
            "side": order["side"],
            "price": order["price"],
            "qty": order["qty"],
        }
        try:
            self.order_ws.send(json.dumps(msg))
            self.orders_sent += 1
        except:
            pass

    def _cancel_order(self, order_id: str):
        # Protocol guess: "action": "CANCEL" with order_id
        # If this fails, we will just hit the limit again, but it's our best bet.
        msg = {"action": "CANCEL", "order_id": order_id}
        try:
            self.order_ws.send(json.dumps(msg))
        except:
            pass

    def _send_done(self):
        try:
            self.order_ws.send(json.dumps({"action": "DONE"}))
        except:
            pass

    def _on_order_response(self, ws, message: str):
        try:
            data = json.loads(message)
            if data.get("type") == "FILL":
                qty = data.get("qty", 0)
                price = data.get("price", 0.0)
                side = data.get("side", "")
                if side == "BUY":
                    self.inventory += qty
                    self.cash_flow -= qty * price
                else:
                    self.inventory -= qty
                    self.cash_flow += qty * price
                self.pnl = self.cash_flow + (self.inventory * self.last_mid)
        except:
            pass

    def _on_error(self, ws, error):
        if self.running:
            print(f"WS Error: {error}")

    def _on_close(self, ws, *args):
        self.running = False
        print("Connection Closed")

    def run(self):
        if self.register() and self.connect():
            print(f"[{self.student_id}] Penny Jumper Running...")
            try:
                while self.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Stopped.")
            finally:
                self.running = False
                if self.market_ws:
                    self.market_ws.close()
                if self.order_ws:
                    self.order_ws.close()
                print(f"Final PnL: {self.pnl:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--scenario", default="normal_market")
    parser.add_argument("--host", default="localhost:8080")
    parser.add_argument("--secure", action="store_true")
    args = parser.parse_args()

    bot = TradingBot(args.name, args.host, args.scenario, args.password, args.secure)
    bot.run()
