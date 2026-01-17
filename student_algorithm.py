"""
Student Trading Algorithm Template
===================================
Connect to the exchange simulator, receive market data, and submit orders.

    python student_algorithm.py --host ip:host --scenario normal_market --name your_name --password your_password --secure

YOUR TASK:
    Modify the `decide_order()` method to implement your trading strategy.
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
from typing import Dict, Optional
from collections import deque

# Suppress SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

REGIME_SCENARIOS = (
    "normal_market",
    "stressed_market",
    "hft_dominated",
)

REGIME_ID_MAP = {
    0: "normal_market",
    1: "stressed_market",
    3: "hft_dominated",
}

INVENTORY_LIMIT = 5000
RISK_UNWIND_THRESHOLD = 10
RISK_UNWIND_MAX = 150


class TradingBot:
    """
    A trading bot that connects to the exchange simulator.
    
    Students should modify the `decide_order()` method to implement their strategy.
    """
    
    def __init__(
        self,
        student_id: str,
        host: str,
        scenario: str,
        password: str = None,
        secure: bool = False,
        regime_model_path: Optional[str] = None,
    ):
        self.student_id = student_id
        self.host = host
        self.scenario = scenario
        self.password = password
        self.secure = secure
        
        # Protocol configuration
        self.http_proto = "https" if secure else "http"
        self.ws_proto = "wss" if secure else "ws"
        
        # Session info (set after registration)
        self.token = None
        self.run_id = None
        
        # Trading state - track your position
        self.inventory = 0      # Current position (positive = long, negative = short)
        self.cash_flow = 0.0    # Cumulative cash from trades (negative when buying)
        self.pnl = 0.0          # Mark-to-market PnL (cash_flow + inventory * mid_price)
        self.current_step = 0   # Current simulation step
        self.orders_sent = 0    # Number of orders sent
        
        # Market data
        self.last_bid = 0.0
        self.last_ask = 0.0
        self.last_mid = 0.0
        self.price_history = deque(maxlen=5)
        
        # WebSocket connections
        self.market_ws = None
        self.order_ws = None
        self.running = True
        
        # Latency measurement
        self.last_done_time = None          # When we sent DONE
        self.step_latencies = []            # Time between DONE and next market data
        self.order_send_times = {}          # order_id -> time sent
        self.fill_latencies = []            # Time between order and fill
        # Track order rate compliance
        self.order_history = deque()
        self.order_limit_window = 50
        self.order_limit_max = 1
        self.regime_model = self._load_regime_model(regime_model_path)
        self.regime_strategy_map = {
            "normal_market": self._strategy_normal_market,
            "stressed_market": self._strategy_stressed_market,
            "hft_dominated": self._strategy_hft_dominated,
        }
    
    # =========================================================================
    # REGISTRATION - Get a token to start trading
    # =========================================================================
    
    def register(self) -> bool:
        """Register with the server and get an auth token."""
        print(f"[{self.student_id}] Registering for scenario '{self.scenario}'...")
        try:
            url = f"{self.http_proto}://{self.host}/api/replays/{self.scenario}/start"
            headers = {"Authorization": f"Bearer {self.student_id}"}
            if self.password:
                headers["X-Team-Password"] = self.password
            resp = requests.get(
                url,
                headers=headers,
                timeout=10,
                verify=not self.secure  # Disable SSL verification for self-signed certs
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
    
    # =========================================================================
    # CONNECTION - Connect to WebSocket streams
    # =========================================================================
    
    def connect(self) -> bool:
        """Connect to market data and order entry WebSockets."""
        try:
            # SSL options for self-signed certificates
            sslopt = {"cert_reqs": ssl.CERT_NONE} if self.secure else None
            
            # Market Data WebSocket
            market_url = f"{self.ws_proto}://{self.host}/api/ws/market?run_id={self.run_id}"
            self.market_ws = websocket.WebSocketApp(
                market_url,
                on_message=self._on_market_data,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=lambda ws: print(f"[{self.student_id}] Market data connected")
            )
            
            # Order Entry WebSocket
            order_url = f"{self.ws_proto}://{self.host}/api/ws/orders?token={self.token}&run_id={self.run_id}"
            self.order_ws = websocket.WebSocketApp(
                order_url,
                on_message=self._on_order_response,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=lambda ws: print(f"[{self.student_id}] Order entry connected")
            )
            
            # Start WebSocket threads
            threading.Thread(
                target=lambda: self.market_ws.run_forever(sslopt=sslopt),
                daemon=True
            ).start()
            
            threading.Thread(
                target=lambda: self.order_ws.run_forever(sslopt=sslopt),
                daemon=True
            ).start()
            
            # Wait for connections
            time.sleep(1)
            return True
            
        except Exception as e:
            print(f"[{self.student_id}] Connection error: {e}")
            return False
    
    # =========================================================================
    # MARKET DATA HANDLER - Called when new market data arrives
    # =========================================================================
    
    def _on_market_data(self, ws, message: str):
        """Handle incoming market data snapshot."""
        try:
            recv_time = time.time()
            data = json.loads(message)
            
            # Skip connection confirmation messages
            if data.get("type") == "CONNECTED":
                return
            
            # Measure step latency (time since we sent DONE)
            if self.last_done_time is not None:
                step_latency = (recv_time - self.last_done_time) * 1000  # ms
                self.step_latencies.append(step_latency)
            
            # Extract market data
            self.current_step = data.get("step", 0)
            self.last_bid = data.get("bid", 0.0)
            self.last_ask = data.get("ask", 0.0)
            
            # Log progress every 500 steps with latency stats
            if self.current_step % 500 == 0 and self.step_latencies:
                avg_lat = sum(self.step_latencies[-100:]) / min(len(self.step_latencies), 100)
                print(f"[{self.student_id}] Step {self.current_step} | Orders: {self.orders_sent} | Inv: {self.inventory} | Avg Latency: {avg_lat:.1f}ms")
            
            # Calculate mid price
            if self.last_bid > 0 and self.last_ask > 0:
                self.last_mid = (self.last_bid + self.last_ask) / 2
            elif self.last_bid > 0:
                self.last_mid = self.last_bid
            elif self.last_ask > 0:
                self.last_mid = self.last_ask
            else:
                self.last_mid = 0

            if self.last_mid > 0:
                self.price_history.append(self.last_mid)
            
            # =============================================
            # YOUR STRATEGY LOGIC GOES HERE
            # =============================================
            order = self.decide_order(self.last_bid, self.last_ask, self.last_mid)
            
            #ADDED
            if order and self.order_ws and self.order_ws.sock and self._can_send_order():
                self._send_order(order)
            
            # Signal DONE to advance to next step
            self._send_done()
            
        except Exception as e:
            print(f"[{self.student_id}] Market data error: {e}")
    
    def _load_regime_model(self, path: Optional[str]):
        """Attempt to load a pre-trained model for regime classification."""
        if not path:
            return None
        try:
            booster = xgb.Booster()
            booster.load_model(path)
            print(f"[{self.student_id}] Loaded XGBoost model from {path}")
            return booster
        except Exception:
            pass

        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            print(f"[{self.student_id}] Loaded pickled model from {path}")
            return model
        except Exception as exc:
            print(f"[{self.student_id}] Regime model load failed: {exc}")
        return None

    def _recent_momentum(self) -> float:
        if len(self.price_history) >= 2:
            return self.price_history[-1] - self.price_history[-2]
        return 0.0

    def _build_regime_features(self, bid: float, ask: float, mid: float):
        spread = max(ask - bid, 0.0)
        momentum = self._recent_momentum()
        volatility = 0.0
        if len(self.price_history) >= 2:
            diffs = [
                abs(self.price_history[i + 1] - self.price_history[i])
                for i in range(len(self.price_history) - 1)
            ]
            if diffs:
                volatility = sum(diffs) / len(diffs)
        return [bid, ask, mid, spread, momentum, volatility]

    def _fallback_regime_guess(self, features):
        mid = features[2]
        spread = features[3]
        momentum = features[4]
        if mid <= 0:
            return "normal_market"
        if spread > 0.05 * mid:
            return "stressed_market"
        if abs(momentum) > 0.015 * mid:
            return "flash_crash" if momentum < 0 else "hft_dominated"
        if spread > 0.02 * mid:
            return "stressed_market"
        return "mini_flash_crash" if abs(momentum) > 0.005 * mid else "normal_market"

    def _resolve_regime_label(self, prediction):
        label = str(prediction).strip()
        if label in self.regime_strategy_map:
            return label
        try:
            idx = int(float(prediction))
            return REGIME_ID_MAP.get(idx, "normal_market")
        except Exception:
            return "normal_market"

    def _predict_regime(self, features):
        model = self.regime_model
        if isinstance(model, xgb.Booster):
            dmatrix = xgb.DMatrix([features])
            return model.predict(dmatrix)[0]
        predict_fn = getattr(model, "predict", None)
        if callable(predict_fn):
            return predict_fn([features])[0]
        raise TypeError("Unsupported regime model type")

    def classify_regime(self, bid: float, ask: float, mid: float) -> str:
        features = self._build_regime_features(bid, ask, mid)
        if self.regime_model:
            try:
                prediction = self._predict_regime(features)
                return self._resolve_regime_label(prediction)
            except Exception as exc:
                print(f"[{self.student_id}] Regime classification failed: {exc}")
        return self._fallback_regime_guess(features)
    
    # =========================================================================
    # YOUR STRATEGY - MODIFY THIS METHOD!
    # =========================================================================
    
    def decide_order(self, bid: float, ask: float, mid: float) -> Optional[Dict]:
        if mid <= 0 or bid <= 0 or ask <= 0:
            return None

        if abs(self.inventory) > INVENTORY_LIMIT:
            print("too much inventory hit")
            return None

        risk_order = self._risk_manage_inventory(bid, ask, mid)
        if risk_order:
            return risk_order

        regime = self.classify_regime(bid, ask, mid)
        handler = self.regime_strategy_map.get(regime, self._strategy_normal_market)
        return handler(bid, ask, mid, regime)

    def _create_order(self, side: str, price: float, qty: int) -> Dict:
        """Normalize order price and quantity for submission."""
        return {"side": side, "price": round(max(price, 0.01), 2), "qty": qty}

    def _risk_manage_inventory(
        self, bid: float, ask: float, mid: float
    ) -> Optional[Dict]:
        exposure = self.inventory
        if abs(exposure) <= RISK_UNWIND_THRESHOLD:
            return None

        qty = min(abs(exposure) - RISK_UNWIND_THRESHOLD, RISK_UNWIND_MAX)
        qty = max(qty, 1)

        if exposure > 0:
            price = max(bid - 0.01, 0.01)
            return self._create_order("SELL", price, qty)

        price = min(ask + 0.01, ask + 0.03)
        return self._create_order("BUY", price, qty)

    def _strategy_normal_market(
        self, bid: float, ask: float, mid: float, regime: str
    ) -> Optional[Dict]:
        momentum = self._recent_momentum()
        threshold = 0.0025 * mid
        if abs(momentum) < threshold:
            return None

        small_tick = max(0.01, mid * 0.0005)
        qty = 100

        if momentum > 0:
            price = min(ask + small_tick, ask + 0.05)
            return self._create_order("BUY", price, qty)

        price = max(bid - small_tick, bid - 0.05)
        return self._create_order("SELL", price, qty)

    def _strategy_stressed_market(
        self, bid: float, ask: float, mid: float, regime: str
    ) -> Optional[Dict]:
        momentum = self._recent_momentum()
        if abs(momentum) < 0.01 * mid:
            return None

        small_tick = max(0.01, mid * 0.001)
        qty = 60

        if momentum > 0:
            price = min(ask + small_tick, ask + 0.03)
            return self._create_order("BUY", price, qty)

        price = max(bid - small_tick, bid - 0.03)
        return self._create_order("SELL", price, qty)

    def _strategy_hft_dominated(
        self, bid: float, ask: float, mid: float, regime: str
    ) -> Optional[Dict]:
        momentum = self._recent_momentum()
        if abs(momentum) < 0.0015 * mid:
            return None

        small_tick = max(0.005, mid * 0.0003)
        qty = 40

        if momentum > 0:
            price = min(ask + small_tick, ask + 0.02)
            return self._create_order("BUY", price, qty)

        price = max(bid - small_tick, bid - 0.02)
        return self._create_order("SELL", price, qty)

    # =========================================================================
    # ORDER HANDLING
    # =========================================================================
    
    def _send_order(self, order: Dict):
        """Send an order to the exchange."""
        order_id = f"ORD_{self.student_id}_{self.current_step}_{self.orders_sent}"
        
        msg = {
            "order_id": order_id,
            "side": order["side"],
            "price": order["price"],
            "qty": order["qty"]
        }
        
        try:
            self.order_send_times[order_id] = time.time()  # Track send time
            self.order_ws.send(json.dumps(msg))
            self.orders_sent += 1
            self.order_history.append(self.current_step)
        except Exception as e:
            print(f"[{self.student_id}] Send order error: {e}")

    def _can_send_order(self) -> bool:
        window_start = self.current_step - self.order_limit_window
        while self.order_history and self.order_history[0] <= window_start:
            self.order_history.popleft()
        allowed = len(self.order_history) < self.order_limit_max
        print(f"[{self.student_id}] Can send order? {allowed} | recent={len(self.order_history)} window={self.order_limit_window}")
        return allowed
    
    def _send_done(self):
        """Signal DONE to advance to the next simulation step."""
        try:
            self.order_ws.send(json.dumps({"action": "DONE"}))
            self.last_done_time = time.time()  # Track when we sent DONE
        except:
            pass
    
    def _on_order_response(self, ws, message: str):
        """Handle order responses and fills."""
        try:
            recv_time = time.time()
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "AUTHENTICATED":
                print(f"[{self.student_id}] Authenticated - ready to trade!")
            
            elif msg_type == "FILL":
                qty = data.get("qty", 0)
                price = data.get("price", 0)
                side = data.get("side", "")
                order_id = data.get("order_id", "")
                
                # Measure fill latency
                if order_id in self.order_send_times:
                    fill_latency = (recv_time - self.order_send_times[order_id]) * 1000  # ms
                    self.fill_latencies.append(fill_latency)
                    del self.order_send_times[order_id]
                
                # Update inventory and cash flow
                if side == "BUY":
                    self.inventory += qty
                    self.cash_flow -= qty * price  # Spent cash to buy
                else:
                    self.inventory -= qty
                    self.cash_flow += qty * price  # Received cash from selling
                
                # Calculate mark-to-market PnL using mid price
                self.pnl = self.cash_flow + self.inventory * self.last_mid
                
                print(f"[{self.student_id}] FILL: {side} {qty} @ {price:.2f} | Inventory: {self.inventory} | PnL: {self.pnl:.2f}")
            
            elif msg_type == "ERROR":
                print(f"[{self.student_id}] ERROR: {data.get('message')}")
                
        except Exception as e:
            print(f"[{self.student_id}] Order response error: {e}")
    
    # =========================================================================
    # ERROR HANDLING
    # =========================================================================
    
    def _on_error(self, ws, error):
        if self.running:
            print(f"[{self.student_id}] WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        self.running = False
        print(f"[{self.student_id}] Connection closed (status: {close_status_code})")
    
    # =========================================================================
    # MAIN RUN LOOP
    # =========================================================================
    
    def run(self):
        """Main entry point - register, connect, and run."""
        # Step 1: Register
        if not self.register():
            return
        
        # Step 2: Connect
        if not self.connect():
            return
        
        # Step 3: Run until complete
        print(f"[{self.student_id}] Running... Press Ctrl+C to stop")
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n[{self.student_id}] Stopped by user")
        finally:
            self.running = False
            if self.market_ws:
                self.market_ws.close()
            if self.order_ws:
                self.order_ws.close()
            
            print(f"\n[{self.student_id}] Final Results:")
            print(f"  Orders Sent: {self.orders_sent}")
            print(f"  Inventory: {self.inventory}")
            print(f"  PnL: {self.pnl:.2f}")
            
            # Print latency statistics
            if self.step_latencies:
                print(f"\n  Step Latency (ms):")
                print(f"    Min: {min(self.step_latencies):.1f}")
                print(f"    Max: {max(self.step_latencies):.1f}")
                print(f"    Avg: {sum(self.step_latencies)/len(self.step_latencies):.1f}")
            
            if self.fill_latencies:
                print(f"\n  Fill Latency (ms):")
                print(f"    Min: {min(self.fill_latencies):.1f}")
                print(f"    Max: {max(self.fill_latencies):.1f}")
                print(f"    Avg: {sum(self.fill_latencies)/len(self.fill_latencies):.1f}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Student Trading Algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Local server:
    python student_algorithm.py --name team_alpha --password secret123 --scenario normal_market
    
  Deployed server (HTTPS):
    python student_algorithm.py --name team_alpha --password secret123 --scenario normal_market --host 3.98.52.120:8433 --secure
        """
    )
    
    parser.add_argument("--name", required=True, help="Your team name")
    parser.add_argument("--password", required=True, help="Your team password")
    parser.add_argument("--scenario", default="normal_market", help="Scenario to run")
    parser.add_argument("--host", default="localhost:8080", help="Server host:port")
    parser.add_argument("--secure", action="store_true", help="Use HTTPS/WSS (for deployed servers)")
    parser.add_argument("--regime-model", help="Path to a pretrained regime classification model")
    args = parser.parse_args()
    
    bot = TradingBot(
        student_id=args.name,
        host=args.host,
        scenario=args.scenario,
        password=args.password,
        secure=args.secure,
        regime_model_path=args.regime_model
    )
    
    bot.run()
