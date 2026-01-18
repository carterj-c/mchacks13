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
from typing import Dict, Optional, Any, List
from collections import deque
import math
import random
import os

# Suppress SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

REGIME_SCENARIOS = (
    "normal_market",
    "stressed_market",
    "hft_dominated",
)

INVENTORY_LIMIT = 5000
RISK_UNWIND_THRESHOLD = 10
RISK_UNWIND_MAX = 150

# NEW: Max number of open/resting orders allowed by exchange (your error was 50)
MAX_OPEN_ORDERS = 50

# Manual trader shows ticker field
DEFAULT_TICKER = "SYM"


class TradingBot:
    """
    A trading bot that connects to the exchange simulator.
    """

    def __init__(
        self,
        student_id: str,
        host: str,
        scenario: str,
        password: str = None,
        secure: bool = False,
        regime_model_path: Optional[str] = "market_classifier_crator.pkl",
        regime_meta_path: Optional[str] = None,
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
        self.price_history = deque(maxlen=5)

        # WebSocket connections
        self.market_ws = None
        self.order_ws = None
        self.running = True

        # Latency measurement
        self.last_done_time = None
        self.step_latencies = []
        self.order_send_times = {}
        self.fill_latencies = []

        # Track order rate compliance (keep this!)
        self.order_history = deque()
        self.order_limit_window = 50
        self.order_limit_max = 1

        # NEW: Open order tracking so we can cancel oldest
        self.open_order_queue = deque()  # oldest -> newest order_ids
        self.open_order_set = set()      # fast membership check

        # Load model + metadata
        self.regime_model = self._load_regime_model(regime_model_path)
        self._label_classes: Optional[List[str]] = None
        self._feature_names: Optional[List[str]] = None

        # If meta path not supplied, try common defaults next to model path
        if regime_meta_path is None and regime_model_path:
            candidates = []
            base, _ = os.path.splitext(regime_model_path)
            candidates.append(base + "_meta.pkl")
            candidates.append(base + ".meta.pkl")
            candidates.append("market_classifier_meta.pkl")
            for c in candidates:
                if os.path.exists(c):
                    regime_meta_path = c
                    break

        self._load_regime_metadata(regime_meta_path)

        self.regime_strategy_map = {
            "normal_market": self._strategy_normal_market,
            "stressed_market": self._strategy_stressed_market,
            "hft_dominated": self._strategy_hft_dominated,
        }

        # EXACT feature-engineering state (rolling/EMA)
        self._mid_hist = deque(maxlen=60)         # supports rolling(50), rolling(20)
        self._spread_rel_hist = deque(maxlen=60)  # supports rolling mean(50)
        self._velocity_ema = None                # EMA(span=20, adjust=False)

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    def register(self) -> bool:
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
                verify=not self.secure
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
    # CONNECTION
    # =========================================================================

    def connect(self) -> bool:
        try:
            sslopt = {"cert_reqs": ssl.CERT_NONE} if self.secure else None

            market_url = f"{self.ws_proto}://{self.host}/api/ws/market?run_id={self.run_id}"
            self.market_ws = websocket.WebSocketApp(
                market_url,
                on_message=self._on_market_data,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=lambda ws: print(f"[{self.student_id}] Market data connected")
            )

            order_url = f"{self.ws_proto}://{self.host}/api/ws/orders?token={self.token}&run_id={self.run_id}"
            self.order_ws = websocket.WebSocketApp(
                order_url,
                on_message=self._on_order_response,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=lambda ws: print(f"[{self.student_id}] Order entry connected")
            )

            threading.Thread(target=lambda: self.market_ws.run_forever(sslopt=sslopt), daemon=True).start()
            threading.Thread(target=lambda: self.order_ws.run_forever(sslopt=sslopt), daemon=True).start()

            time.sleep(1)
            return True

        except Exception as e:
            print(f"[{self.student_id}] Connection error: {e}")
            return False

    # =========================================================================
    # MARKET DATA HANDLER
    # =========================================================================

    def _on_market_data(self, ws, message: str):
        try:
            recv_time = time.time()
            data = json.loads(message)

            if data.get("type") == "CONNECTED":
                return

            if self.last_done_time is not None:
                step_latency = (recv_time - self.last_done_time) * 1000
                self.step_latencies.append(step_latency)

            self.current_step = data.get("step", 0)
            self.last_bid = data.get("bid", 0.0)
            self.last_ask = data.get("ask", 0.0)

            if self.current_step % 500 == 0 and self.step_latencies:
                avg_lat = sum(self.step_latencies[-100:]) / min(len(self.step_latencies), 100)
                print(f"[{self.student_id}] Step {self.current_step} | Orders: {self.orders_sent} | Open: {len(self.open_order_set)} | Inv: {self.inventory} | Avg Latency: {avg_lat:.1f}ms")

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

            order = self.decide_order(self.last_bid, self.last_ask, self.last_mid)

            # If we want to send an order, cancel oldest orders first if needed
            if order and self.order_ws and self.order_ws.sock and self._can_send_order():
                self._ensure_open_order_capacity(extra_needed=1)
                self._send_order(order)

            self._send_done()

        except Exception as e:
            print(f"[{self.student_id}] Market data error: {e}")

    # =========================================================================
    # MODEL + META LOADING
    # =========================================================================

    def _load_regime_model(self, path: Optional[str]):
        if not path:
            print(f"[{self.student_id}] No regime model path provided.")
            return None

        # Try loading as Booster model
        try:
            booster = xgb.Booster()
            booster.load_model(path)
            print(f"[{self.student_id}] Loaded XGBoost Booster model from {path}")
            return booster
        except Exception:
            pass

        # Try pickled sklearn model (XGBClassifier or similar)
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            print(f"[{self.student_id}] Loaded pickled model from {path}")
            # If it has classes_, use it
            classes_ = getattr(model, "classes_", None)
            if classes_ is not None:
                try:
                    self._label_classes = [str(x) for x in list(classes_)]
                except Exception:
                    pass
            return model
        except Exception as exc:
            print(f"[{self.student_id}] Regime model load failed: {exc}")
            return None

    def _load_regime_metadata(self, meta_path: Optional[str]) -> None:
        """
        Recommended: save this from training:
            meta = {"classes": list(le.classes_), "feature_names": list(X_train.columns)}
            pickle.dump(meta, open("market_classifier_meta.pkl","wb"))
        """
        if not meta_path:
            self._label_classes = self._label_classes or sorted(REGIME_SCENARIOS)
            self._feature_names = self._feature_names or [
                "spread_rel",
                "vol_20",
                "vol_50",
                "mid_change_abs",
                "velocity_ema",
                "spread_ma_50",
                "spread_ratio",
            ]
            print(f"[{self.student_id}] No meta file found. Using fallback label order: {self._label_classes}")
            return

        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            classes = meta.get("classes")
            feats = meta.get("feature_names")
            if classes and isinstance(classes, (list, tuple)):
                self._label_classes = [str(x) for x in list(classes)]
            if feats and isinstance(feats, (list, tuple)):
                self._feature_names = [str(x) for x in list(feats)]

            self._label_classes = self._label_classes or sorted(REGIME_SCENARIOS)
            self._feature_names = self._feature_names or [
                "spread_rel",
                "vol_20",
                "vol_50",
                "mid_change_abs",
                "velocity_ema",
                "spread_ma_50",
                "spread_ratio",
            ]

            print(f"[{self.student_id}] Loaded metadata from {meta_path}")
            print(f"[{self.student_id}] Label classes (id->name): {self._label_classes}")
            print(f"[{self.student_id}] Feature order: {self._feature_names}")

        except Exception as exc:
            print(f"[{self.student_id}] Meta load failed ({meta_path}): {exc}")
            self._label_classes = self._label_classes or sorted(REGIME_SCENARIOS)
            self._feature_names = self._feature_names or [
                "spread_rel",
                "vol_20",
                "vol_50",
                "mid_change_abs",
                "velocity_ema",
                "spread_ma_50",
                "spread_ratio",
            ]

    # =========================================================================
    # BASIC HELPERS
    # =========================================================================

    def _recent_momentum(self) -> float:
        if len(self.price_history) >= 2:
            return self.price_history[-1] - self.price_history[-2]
        return 0.0

    def _fallback_regime_guess(self, bid: float, ask: float, mid: float) -> str:
        spread = max(ask - bid, 0.0)
        momentum = self._recent_momentum()
        if mid <= 0:
            return "normal_market"
        if spread > 0.05 * mid:
            return "stressed_market"
        if abs(momentum) > 0.015 * mid:
            return "hft_dominated" if momentum > 0 else "stressed_market"
        if spread > 0.02 * mid:
            return "stressed_market"
        return "normal_market"

    # =========================================================================
    # EXACT FEATURE ENGINEERING (online)
    # ==========================================================================

    def _rolling_std(self, values, window: int) -> Optional[float]:
        if len(values) < window:
            return None
        arr = list(values)[-window:]
        n = len(arr)
        if n <= 1:
            return 0.0
        mean = sum(arr) / n
        var = sum((x - mean) ** 2 for x in arr) / (n - 1)
        return math.sqrt(var)

    def _rolling_mean(self, values, window: int) -> Optional[float]:
        if len(values) < window:
            return None
        arr = list(values)[-window:]
        return sum(arr) / window

    def _update_engineered_state_and_get_features(
        self, bid: float, ask: float, mid: float
    ) -> Optional[Dict[str, float]]:
        if mid <= 0:
            return None

        spread_rel = (ask - bid) / mid

        prev_mid = self._mid_hist[-1] if self._mid_hist else None
        self._mid_hist.append(mid)
        self._spread_rel_hist.append(spread_rel)

        mid_change_abs = 0.0
        if prev_mid is not None:
            mid_change_abs = abs(mid - prev_mid)

        alpha = 2.0 / (20.0 + 1.0)
        if self._velocity_ema is None:
            self._velocity_ema = mid_change_abs
        else:
            self._velocity_ema = alpha * mid_change_abs + (1.0 - alpha) * self._velocity_ema

        vol_20 = self._rolling_std(self._mid_hist, 20)
        vol_50 = self._rolling_std(self._mid_hist, 50)
        spread_ma_50 = self._rolling_mean(self._spread_rel_hist, 50)

        if vol_20 is None or vol_50 is None or spread_ma_50 is None:
            return None
        if spread_ma_50 == 0:
            return None

        spread_ratio = spread_rel / spread_ma_50

        return {
            "spread_rel": float(spread_rel),
            "vol_20": float(vol_20),
            "vol_50": float(vol_50),
            "mid_change_abs": float(mid_change_abs),
            "velocity_ema": float(self._velocity_ema),
            "spread_ma_50": float(spread_ma_50),
            "spread_ratio": float(spread_ratio),
        }

    # =========================================================================
    # MODEL CALL (robust) + LABEL RESOLUTION (FIXED)
    # =========================================================================

    def _to_scalar(self, pred: Any) -> float:
        try:
            while isinstance(pred, (list, tuple)) and len(pred) == 1:
                pred = pred[0]
        except Exception:
            pass
        try:
            if hasattr(pred, "shape") and hasattr(pred, "item") and getattr(pred, "shape", ()) != ():
                if getattr(pred, "size", 1) == 1:
                    return float(pred.item())
        except Exception:
            pass
        try:
            return float(pred)
        except Exception:
            return float("nan")

    def _predict_regime_from_engineered_features(self, feats: Dict[str, float]) -> Any:
        feature_names = self._feature_names or [
            "spread_rel",
            "vol_20",
            "vol_50",
            "mid_change_abs",
            "velocity_ema",
            "spread_ma_50",
            "spread_ratio",
        ]
        vec = [feats[name] for name in feature_names]

        model = self.regime_model
        if model is None:
            raise RuntimeError("No model loaded")

        if isinstance(model, xgb.Booster):
            dmatrix = xgb.DMatrix([vec], feature_names=feature_names)
            pred = model.predict(dmatrix)
            return pred[0] if hasattr(pred, "__len__") else pred

        predict_fn = getattr(model, "predict", None)
        if callable(predict_fn):
            return predict_fn([vec])[0]

        raise TypeError("Unsupported regime model type")

    def _resolve_regime_label(self, prediction: Any) -> str:
        classes = self._label_classes or sorted(REGIME_SCENARIOS)

        s = str(prediction).strip()
        if s in self.regime_strategy_map:
            return s

        idx = int(self._to_scalar(prediction))
        if 0 <= idx < len(classes):
            label = classes[idx]
            if label in self.regime_strategy_map:
                return label

        return "normal_market"

    # =========================================================================
    # YOUR STRATEGY
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

        engineered = self._update_engineered_state_and_get_features(bid, ask, mid)

        if engineered is None:
            regime = self._fallback_regime_guess(bid, ask, mid)
            handler = self.regime_strategy_map.get(regime, self._strategy_normal_market)
            return handler(bid, ask, mid, regime)

        regime = None
        if self.regime_model is not None:
            try:
                pred = self._predict_regime_from_engineered_features(engineered)
                regime = self._resolve_regime_label(pred)
            except Exception as exc:
                print(f"[{self.student_id}] Regime classification failed: {exc}")

        if not regime:
            regime = self._fallback_regime_guess(bid, ask, mid)

        handler = self.regime_strategy_map.get(regime, self._strategy_normal_market)

        if self.current_step % 200 == 0:
            print(f"[{self.student_id}] Regime: {regime}")

        return handler(bid, ask, mid, regime)

    # =========================================================================
    # ORDER CREATION + RISK MGMT + STRATEGIES
    # =========================================================================

    def _create_order(self, side: str, price: float, qty: int) -> Dict:
        return {"side": side, "price": round(max(price, 0.01), 2), "qty": qty}

    def _risk_manage_inventory(self, bid: float, ask: float, mid: float) -> Optional[Dict]:
        exposure = self.inventory

        if abs(exposure) <= RISK_UNWIND_THRESHOLD:
            return None

        desired_qty = min(abs(exposure) - RISK_UNWIND_THRESHOLD, RISK_UNWIND_MAX)
        qty = (desired_qty // 100) * 100

        if qty < 100:
            return None

        if exposure > 0:
            price = max(bid - 0.01, 0.01)
            return self._create_order("SELL", price, qty)

        price = min(ask + 0.01, ask + 0.03)
        return self._create_order("BUY", price, qty)

    def _strategy_normal_market(
        self, bid: float, ask: float, mid: float, regime: str
    ) -> Optional[Dict]:
        """
        Implementation of SPRAY AND PRAY
        """

        # 2. Price Calculation (Adaptive Anti-Spiral)
        # Check if we are already the top of book. If so, reinforce. If not, jump.
        if abs(bid - self.my_last_bid) < 0.001:
            my_bid = bid
        else:
            my_bid = round(bid + 0.01, 2)

        if abs(ask - self.my_last_ask) < 0.001:
            my_ask = ask
        else:
            my_ask = round(ask - 0.01, 2)

        # Spread Safety
        if my_bid >= my_ask:
            return None

        # 3. Execution (Alternating Fire)
        self.flip_flop = not self.flip_flop
        qty = 100

        if self.flip_flop:
            self.my_last_bid = my_bid
            return self._create_order("BUY", my_bid, qty)
        else:
            self.my_last_ask = my_ask
            return self._create_order("SELL", my_ask, qty)

    def _strategy_stressed_market(self, bid: float, ask: float, mid: float, regime: str) -> Optional[Dict]:
        # Trade rarely in stressed mode
        if self.current_step % 300 != 0:
            return None

        qty = 100
        offset = mid * random.uniform(0.005, 0.02)

        if self.inventory > 0:
            side = "SELL"
            price = ask + offset
        elif self.inventory < 0:
            side = "BUY"
            price = bid - offset
        else:
            if random.random() > 0.5:
                side = "BUY"
                price = bid - offset
            else:
                side = "SELL"
                price = ask + offset

        return self._create_order(side, round(price, 2), qty)

    def _strategy_hft_dominated(self, bid: float, ask: float, mid: float, regime: str) -> Optional[Dict]:
        return self._create_order("BUY", ask, 2000)

    # =========================================================================
    # ORDER CANCELLATION (NEW)
    # =========================================================================

    def _send_cancel(self, order_id: str) -> None:
        """
        Manual trader uses:
          {"type": "CANCEL_ORDER", "order_id": "..."}
        """
        try:
            if not (self.order_ws and self.order_ws.sock):
                return
            msg = {"type": "CANCEL_ORDER", "order_id": order_id}
            self.order_ws.send(json.dumps(msg))
        except Exception as e:
            print(f"[{self.student_id}] Cancel send error: {e}")

    def _ensure_open_order_capacity(self, extra_needed: int = 1) -> None:
        """
        If sending a new order would exceed MAX_OPEN_ORDERS,
        cancel oldest orders until there is room.
        """
        target_max = MAX_OPEN_ORDERS - extra_needed
        while len(self.open_order_set) > target_max and self.open_order_queue:
            oldest_id = self.open_order_queue.popleft()
            if oldest_id in self.open_order_set:
                self.open_order_set.remove(oldest_id)
                self._send_cancel(oldest_id)

    def _mark_order_open(self, order_id: str) -> None:
        if order_id not in self.open_order_set:
            self.open_order_set.add(order_id)
            self.open_order_queue.append(order_id)

    def _mark_order_closed(self, order_id: str) -> None:
        # remove from set; queue removal is lazy (we skip non-members later)
        if order_id in self.open_order_set:
            self.open_order_set.remove(order_id)

    # =========================================================================
    # ORDER HANDLING
    # =========================================================================

    def _send_order(self, order: Dict):
        """
        Sends order in the SAME schema as manual trader:
          {"type":"NEW_ORDER","order_id":...,"ticker":"SYM",...}
        Tracks the order_id as "open" so we can cancel oldest later.
        """
        order_id = f"ORD_{self.student_id}_{self.current_step}_{self.orders_sent}"
        msg = {
            "type": "NEW_ORDER",
            "order_id": order_id,
            "ticker": DEFAULT_TICKER,
            "side": order["side"],
            "price": order["price"],
            "qty": order["qty"],
        }

        try:
            self.order_send_times[order_id] = time.time()
            self.order_ws.send(json.dumps(msg))

            self.orders_sent += 1
            self.order_history.append(self.current_step)

            # Mark as open immediately; if it fills instantly, we'll remove it on FILL
            self._mark_order_open(order_id)

        except Exception as e:
            print(f"[{self.student_id}] Send order error: {e}")

    def _can_send_order(self) -> bool:
        window_start = self.current_step - self.order_limit_window
        while self.order_history and self.order_history[0] <= window_start:
            self.order_history.popleft()
        return len(self.order_history) < self.order_limit_max

    def _send_done(self):
        try:
            self.order_ws.send(json.dumps({"action": "DONE"}))
            self.last_done_time = time.time()
        except Exception:
            pass

    def _on_order_response(self, ws, message: str):
        try:
            recv_time = time.time()
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "AUTHENTICATED":
                print(f"[{self.student_id}] Authenticated - ready to trade!")

            elif msg_type == "FILL":
                qty = data.get("qty", 0)
                price = data.get("price", 0.0)
                side = data.get("side", "")
                order_id = data.get("order_id", "")

                # If it filled, it is no longer open/resting
                if order_id:
                    self._mark_order_closed(order_id)

                if order_id in self.order_send_times:
                    fill_latency = (recv_time - self.order_send_times[order_id]) * 1000
                    self.fill_latencies.append(fill_latency)
                    del self.order_send_times[order_id]

                if side == "BUY":
                    self.inventory += qty
                    self.cash_flow -= qty * price
                else:
                    self.inventory -= qty
                    self.cash_flow += qty * price

                self.pnl = self.cash_flow + self.inventory * self.last_mid

                print(f"[{self.student_id}] FILL: {side} {qty} @ {price:.2f} | Open: {len(self.open_order_set)} | Inventory: {self.inventory} | PnL: {self.pnl:.2f}")

            # Many sims emit a cancel confirmation type - handle a few common ones
            elif msg_type in ("CANCELLED", "CANCELED", "ORDER_CANCELLED", "CANCEL_CONFIRM"):
                order_id = data.get("order_id", "")
                if order_id:
                    self._mark_order_closed(order_id)

            elif msg_type == "ERROR":
                msg = data.get("message")
                print(f"[{self.student_id}] ERROR: {msg}")

                # If error indicates too many open orders, cancel a few immediately
                # (This is defensive in case our tracking desyncs.)
                if msg and "open orders" in str(msg).lower():
                    self._ensure_open_order_capacity(extra_needed=1)

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
        if not self.register():
            return
        if not self.connect():
            return

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
            print(f"  Open Orders (tracked): {len(self.open_order_set)}")
            print(f"  Inventory: {self.inventory}")
            print(f"  PnL: {self.pnl:.2f}")

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
    parser.add_argument("--regime-model", default="market_classifier_crator.pkl", help="Path to a pretrained regime classification model")
    parser.add_argument("--regime-meta", default=None, help="Path to metadata pickle (classes + feature_names) from training")
    args = parser.parse_args()

    bot = TradingBot(
        student_id=args.name,
        host=args.host,
        scenario=args.scenario,
        password=args.password,
        secure=args.secure,
        regime_model_path=args.regime_model,
        regime_meta_path=args.regime_meta,
    )

    bot.run()
