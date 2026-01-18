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
from collections import deque


# Suppress SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

REGIME_SCENARIOS = (
    "normal_market",
    "stressed_market",
    "hft_dominated",
)

INVENTORY_LIMIT = 5000
RISK_UNWIND_THRESHOLD = 1000
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

        # Track our last quoted prices + alternating order direction (needed by _strategy_normal_market)
        self.my_last_bid = 0.0
        self.my_last_ask = 0.0
        self.flip_flop = False

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

        # Track last 3 spreads for z-score detection in HFT dominated regime
        self._spread_hist3 = deque(maxlen=3)

        # NEW: Open order tracking so we can cancel oldest
        self.open_order_queue = deque()  # oldest -> newest order_ids
        self.open_order_set = set()  # fast membership check

        # NEW: Track cancels that have been sent but not confirmed yet
        self.cancel_pending_set = set()

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
        self._mid_hist = deque(maxlen=60)  # supports rolling(50), rolling(20)
        self._spread_rel_hist = deque(maxlen=60)  # supports rolling mean(50)
        self._velocity_ema = None  # EMA(span=20, adjust=False)

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

    # =========================================================================
    # CONNECTION
    # =========================================================================

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
                on_open=lambda ws: print(f"[{self.student_id}] Market data connected"),
            )

            order_url = f"{self.ws_proto}://{self.host}/api/ws/orders?token={self.token}&run_id={self.run_id}"
            self.order_ws = websocket.WebSocketApp(
                order_url,
                on_message=self._on_order_response,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=lambda ws: print(f"[{self.student_id}] Order entry connected"),
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
                avg_lat = sum(self.step_latencies[-100:]) / min(
                    len(self.step_latencies), 100
                )
                print(
                    f"[{self.student_id}] Step {self.current_step} | Orders: {self.orders_sent} | Open: {len(self.open_order_set)} | Inv: {self.inventory} | Avg Latency: {avg_lat:.1f}ms"
                )

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
            if (
                order
                and self.order_ws
                and self.order_ws.sock
                and self._can_send_order()
            ):
                self._ensure_open_order_capacity(extra_needed=1)

                # IMPORTANT FIX:
                # Do NOT send a new order unless we are strictly under MAX_OPEN_ORDERS
                # because cancels are async and exchange may still count them as open.
                if len(self.open_order_set) < MAX_OPEN_ORDERS:
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
            print(
                f"[{self.student_id}] No meta file found. Using fallback label order: {self._label_classes}"
            )
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
            print(
                f"[{self.student_id}] Label classes (id->name): {self._label_classes}"
            )
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
            self._velocity_ema = (
                alpha * mid_change_abs + (1.0 - alpha) * self._velocity_ema
            )

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
            if (
                hasattr(pred, "shape")
                and hasattr(pred, "item")
                and getattr(pred, "shape", ()) != ()
            ):
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

    def _risk_manage_inventory(
        self, bid: float, ask: float, mid: float
    ) -> Optional[Dict]:
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
        Normal market strategy:
        Alternate forever:
        1) SELL at (ask - 0.01)
        2) BUY  at (bid + 0.01)
        """

        TICK = 0.01
        qty = 100

        # Safety check
        if bid <= 0 or ask <= 0:
            return None

        sell_price = round(ask - TICK, 2)
        buy_price = round(bid + TICK, 2)

        # Prevent crossing / invalid spread
        # If spread is too tight, these would cross (buy >= sell), so do nothing.
        if buy_price >= sell_price:
            return None

        # Alternate order side every time function is called
        self.flip_flop = not self.flip_flop

        if self.flip_flop:
            # SELL at ask - 0.01
            return self._create_order("SELL", sell_price, qty)
        else:
            # BUY at bid + 0.01
            return self._create_order("BUY", buy_price, qty)

    def _strategy_stressed_market(self, bid: float, ask: float, mid: float, regime: str) -> Optional[Dict]:
        """
        Stressed market strategy that matches your plot:
        - Long drift + occasional violent swings.
        So we use:
        - EMA fair value (tracks drift)
        - ATR-like band sizing (tracks volatility)
        - Mean reversion ONLY when move is slowing/reversing
        - Ladder of resting orders (hold inventory) but with "stop-add" protection
        - Near-touch exits so you actually get out when it reverts
        """

        TICK = 0.01
        BASE_QTY = 100

        # --- parameters you can tune ---
        EMA_ALPHA = 2.0 / (40.0 + 1.0)   # EMA(40) fair value (follows drift)
        ATR_WIN = 25                     # volatility window
        ENTRY_K = 2.2                    # entry band width (bigger = fewer knife catches)
        EXIT_K  = 0.7                    # exit band width
        STOP_ADD_K = 3.4                 # if deviation too extreme, stop averaging down
        COOLDOWN = 80                    # rebuild ladder at most every N steps

        MAX_LEVELS = 10                  # ladder depth (open orders are limited anyway)
        MIN_LEVELS = 4

        # spacing control
        SPACING_FRAC_ATR = 0.55          # how wide levels are spaced in stressed tape
        FIRST_LEVEL_MULT = 1.2           # start ladder deeper away from touch

        # protect inventory
        MAX_INV_TO_ADD = int(0.75 * INVENTORY_LIMIT)  # don't keep adding if already loaded

        # --- lazy init ---
        if not hasattr(self, "_sm_ema"):
            self._sm_ema = mid
            self._sm_atr_hist = deque(maxlen=ATR_WIN)
            self._mr_ladder = deque()
            self._mr_last_build_step = -999999
            self._mr_last_z = 0.0

        spread = ask - bid
        if mid <= 0 or bid <= 0 or ask <= 0 or spread <= 0:
            return None

        # ---------------------------
        # 1) Update EMA fair value
        # ---------------------------
        self._sm_ema = EMA_ALPHA * mid + (1.0 - EMA_ALPHA) * self._sm_ema
        fair = self._sm_ema

        # ---------------------------
        # 2) Update ATR-like volatility estimate
        # ---------------------------
        if len(self.price_history) >= 2:
            dm = abs(self.price_history[-1] - self.price_history[-2])
            self._sm_atr_hist.append(dm)

        if len(self._sm_atr_hist) < max(5, ATR_WIN // 2):
            return None

        atr = sum(self._sm_atr_hist) / len(self._sm_atr_hist)
        atr = max(atr, 2 * TICK)  # floor so we don't get microscopic bands

        # ---------------------------
        # 3) Compute deviation "z-ish"
        # ---------------------------
        dev = mid - fair
        z = dev / atr

        # momentum (to avoid fading a freight train)
        mom = 0.0
        if len(self.price_history) >= 3:
            mom = self.price_history[-1] - self.price_history[-2]

        last_z = self._mr_last_z
        self._mr_last_z = z

        def round_px(x: float) -> float:
            return round(max(x, 0.01), 2)

        # ---------------------------
        # 4) Exit logic (when it reverts back toward EMA)
        #    Use near-touch exits so you actually get filled.
        # ---------------------------
        if abs(z) < EXIT_K and abs(self.inventory) >= 100:
            qty = min(abs(self.inventory), 300)
            qty = (qty // 100) * 100
            if qty < 100:
                return None

            if self.inventory > 0:
                # long -> sell near touch
                px = max(bid + TICK, ask - TICK)
                return self._create_order("SELL", round_px(px), qty)
            else:
                # short -> buy near touch
                px = min(ask - TICK, bid + TICK)
                return self._create_order("BUY", round_px(px), qty)

        # ---------------------------
        # 5) If ladder queued, keep placing it (hold orders / build inventory)
        # ---------------------------
        if self._mr_ladder:
            side, px, lvl = self._mr_ladder.popleft()

            # small early size, scale gently deeper
            qty = BASE_QTY * (1 + (lvl // 6))
            room = INVENTORY_LIMIT - abs(self.inventory)
            if room < 100:
                return None
            qty = min(qty, room)
            qty = (qty // 100) * 100
            if qty < 100:
                return None

            return self._create_order(side, px, qty)

        # ---------------------------
        # 6) Entry logic: build ladder ONLY at extremes
        #    AND only if the move is slowing/reversing.
        # ---------------------------
        if abs(z) < ENTRY_K:
            return None

        # cooldown so we don't rebuild constantly
        if self.current_step - self._mr_last_build_step < COOLDOWN:
            return None

        # don't average down forever in a trend
        if abs(self.inventory) > MAX_INV_TO_ADD:
            return None

        # stop-add if it is an extreme runaway move
        if abs(z) > STOP_ADD_K:
            return None

        # ✅ Reversal filter:
        # If price is below fair (z < 0): only BUY if momentum stops going down OR z improves
        # If price is above fair (z > 0): only SELL if momentum stops going up OR z improves
        if z < 0:
            reversal_ok = (mom >= 0) or (abs(z) < abs(last_z))
            if not reversal_ok:
                return None
            side = "BUY"
        else:
            reversal_ok = (mom <= 0) or (abs(z) < abs(last_z))
            if not reversal_ok:
                return None
            side = "SELL"

        # ---------------------------
        # 7) Build ladder prices (deeper first level!)
        # ---------------------------
        z_mag = min(abs(z), 4.0)
        levels = int(2 + z_mag * 2.5)          # more extreme -> deeper ladder
        levels = max(MIN_LEVELS, min(MAX_LEVELS, levels))

        spacing = max(2 * TICK, SPACING_FRAC_ATR * atr, 0.75 * spread)
        first_offset = FIRST_LEVEL_MULT * spacing

        self._mr_ladder.clear()
        for lvl in range(1, levels + 1):
            if side == "BUY":
                px = bid - (first_offset + (lvl - 1) * spacing)
            else:
                px = ask + (first_offset + (lvl - 1) * spacing)

            self._mr_ladder.append((side, round_px(px), lvl))

        self._mr_last_build_step = self.current_step

        # Send first order immediately
        side, px, lvl = self._mr_ladder.popleft()
        qty = BASE_QTY

        room = INVENTORY_LIMIT - abs(self.inventory)
        if room < 100:
            return None
        qty = min(qty, room)
        qty = (qty // 100) * 100
        if qty < 100:
            return None

        return self._create_order(side, px, qty)


    def _strategy_hft_dominated(
        self, bid: float, ask: float, mid: float, regime: str
    ) -> Optional[Dict]:
        """
        HFT-dominated strategy:
        - Detect when spread becomes unusually wide via z-score vs last 3 spreads
        - Bet spread will tighten (mean-revert)
        - Immediately place tighter order inside spread:
            BUY  @ bid + expected_spread_move   (if market likely up)
            SELL @ ask - expected_spread_move  (if market likely down)
        """

        TICK = 0.01
        Z_THRESH = 1.5  # "unusually high spread" threshold
        REVERT_K = 0.75  # fraction of (spread - mean_spread) you expect to revert
        MAX_MOVE = 0.05  # cap the expected spread move so you don't do insane jumps
        QTY = 100

        # Basic checks
        spread = ask - bid
        if bid <= 0 or ask <= 0 or mid <= 0 or spread <= 0:
            return None

        # Need last 3 spreads to compute baseline
        if len(self._spread_hist3) < 3:
            self._spread_hist3.append(spread)
            return None

        # Compute z-score of current spread relative to last 3 spreads
        mean_sp = sum(self._spread_hist3) / 3.0
        var_sp = sum((x - mean_sp) ** 2 for x in self._spread_hist3) / 3.0
        std_sp = math.sqrt(var_sp)

        # If std is ~0, z-score is not meaningful
        if std_sp < 1e-9:
            z = 0.0
        else:
            z = (spread - mean_sp) / std_sp

        # Update spread history AFTER computing z-score baseline
        self._spread_hist3.append(spread)

        # If spread is not unusually wide, do nothing
        if z < Z_THRESH:
            return None

        # ---------------------------
        # Predict market direction
        # ---------------------------
        direction = 0  # +1 up, -1 down
        if len(self.price_history) >= 3:
            y0, y1, y2 = (
                self.price_history[-3],
                self.price_history[-2],
                self.price_history[-1],
            )

            # Small, stable slope proxy + momentum
            mean_y = (y0 + y1 + y2) / 3.0
            cov = (-1.0) * (y0 - mean_y) + 0.0 * (y1 - mean_y) + (1.0) * (y2 - mean_y)
            slope = cov / 2.0
            momentum = y2 - y1

            combined = slope + momentum

            if combined > 0:
                direction = 1
            elif combined < 0:
                direction = -1
            else:
                direction = 0

        elif len(self.price_history) >= 2:
            momentum = self.price_history[-1] - self.price_history[-2]
            if momentum > 0:
                direction = 1
            elif momentum < 0:
                direction = -1

        # If no directional belief, you can either skip or pick a random side
        if direction == 0:
            return None

        # ---------------------------
        # Expected spread tightening
        # ---------------------------
        # How much wider than normal is the spread?
        excess = max(spread - mean_sp, 0.0)

        # Expected tightening amount (your "expected spread move")
        expected_move = REVERT_K * excess

        # Hard caps + floors (must be at least 1 tick to matter)
        expected_move = max(TICK, min(expected_move, MAX_MOVE))

        # Also can't tighten *past* crossing the book
        # You must leave at least 1 tick between bid and ask.
        max_inside = max(spread - TICK, TICK)
        expected_move = min(expected_move, max_inside)

        # ---------------------------
        # Place tighter order NOW
        # ---------------------------
        if direction > 0:
            # BUY inside the spread at bid + expected_move
            price = bid + expected_move

            # Ensure we don't cross the ask
            price = min(price, ask - TICK)

            if price <= bid:
                return None

            return self._create_order("BUY", round(price, 2), QTY)

        else:
            # SELL inside the spread at ask - expected_move
            price = ask - expected_move

            # Ensure we don't cross the bid
            price = max(price, bid + TICK)

            if price >= ask:
                return None

            return self._create_order("SELL", round(price, 2), QTY)

    # =========================================================================
    # ORDER CANCELLATION (FIXED)
    # =========================================================================

    def _send_cancel(self, order_id: str) -> None:
        """
        Manual trader uses:
          {"type": "CANCEL_ORDER", "order_id": "..."}
        """
        try:
            if not (self.order_ws and self.order_ws.sock):
                return
            # Mark cancel as pending (do NOT remove from open set yet)
            self.cancel_pending_set.add(order_id)
            msg = {"type": "CANCEL_ORDER", "order_id": order_id}
            self.order_ws.send(json.dumps(msg))
        except Exception as e:
            print(f"[{self.student_id}] Cancel send error: {e}")

    def _ensure_open_order_capacity(self, extra_needed: int = 1) -> None:
        """
        FIXED:
        Cancels are async, so we cannot "free" capacity instantly.
        We issue cancel requests for enough oldest orders to make room,
        BUT we only remove orders from open tracking when we receive CANCELLED/FILL.
        """
        # How many orders must be removed (by the exchange) before sending "extra_needed" more
        need_to_free = (len(self.open_order_set) + extra_needed) - MAX_OPEN_ORDERS
        if need_to_free <= 0:
            return

        cancels_sent = 0
        while cancels_sent < need_to_free and self.open_order_queue:
            oldest_id = self.open_order_queue.popleft()

            # If it's not actually open anymore, skip it
            if oldest_id not in self.open_order_set:
                continue

            # If we already requested cancel, don't spam
            if oldest_id in self.cancel_pending_set:
                continue

            self._send_cancel(oldest_id)
            cancels_sent += 1

    def _mark_order_open(self, order_id: str) -> None:
        if order_id not in self.open_order_set:
            self.open_order_set.add(order_id)
            self.open_order_queue.append(order_id)

    def _mark_order_closed(self, order_id: str) -> None:
        # remove from set; queue removal is lazy (we skip non-members later)
        if order_id in self.open_order_set:
            self.open_order_set.remove(order_id)
        if order_id in self.cancel_pending_set:
            self.cancel_pending_set.remove(order_id)

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

                print(
                    f"[{self.student_id}] FILL: {side} {qty} @ {price:.2f} | Open: {len(self.open_order_set)} | Inventory: {self.inventory} | PnL: {self.pnl:.2f}"
                )

            # Many sims emit a cancel confirmation type - handle a few common ones
            elif msg_type in (
                "CANCELLED",
                "CANCELED",
                "ORDER_CANCELLED",
                "CANCEL_CONFIRM",
            ):
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
                print(
                    f"    Avg: {sum(self.step_latencies) / len(self.step_latencies):.1f}"
                )

            if self.fill_latencies:
                print(f"\n  Fill Latency (ms):")
                print(f"    Min: {min(self.fill_latencies):.1f}")
                print(f"    Max: {max(self.fill_latencies):.1f}")
                print(
                    f"    Avg: {sum(self.fill_latencies) / len(self.fill_latencies):.1f}"
                )


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
        """,
    )

    parser.add_argument("--name", required=True, help="Your team name")
    parser.add_argument("--password", required=True, help="Your team password")
    parser.add_argument("--scenario", default="normal_market", help="Scenario to run")
    parser.add_argument("--host", default="localhost:8080", help="Server host:port")
    parser.add_argument(
        "--secure", action="store_true", help="Use HTTPS/WSS (for deployed servers)"
    )
    parser.add_argument(
        "--regime-model",
        default="market_classifier_crator.pkl",
        help="Path to a pretrained regime classification model",
    )
    parser.add_argument(
        "--regime-meta",
        default=None,
        help="Path to metadata pickle (classes + feature_names) from training",
    )
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
