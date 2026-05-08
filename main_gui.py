from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
import json
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any
from urllib import request

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

from avellaneda_stoikov.model import AvellanedaStoikovModel, AvellanedaStoikovParameters, MarketSnapshot
from microstructure.fill_hazard import ExponentialFillHazardModel
from microstructure.kyle_lambda import fit_kyle_lambda
from microstructure.microprice.L1microprice import (
    GUI_POLL_MS,
    HYPERLIQUID_MAINNET_WS_URL,
    _default_websocket_factory,
    build_live_payload_from_message,
    build_subscription_message,
    load_streaming_model,
)
from microstructure.microprice.calibration import FittedMicropriceModel
from microstructure.microprice.multilevel_calibration import FittedMultilevelMicropriceModel
from microstructure.ofi import OnlineOFI
from microstructure.reservation_price_adjuster import LinearReservationPriceAdjuster
from microstructure.spread import quoted_spread
from microstructure.trades import MarketTrade
from vpin import OnlineVPIN

StreamingModel = FittedMicropriceModel | FittedMultilevelMicropriceModel
DEFAULT_MODEL_PATH = ROOT / "models" / "btc_l1_model_20260122_20260221.json"
AUTO_ADV_LOOKBACK_DAYS = 7
TIME_SALES_ROWS = 40


def _fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _fmt_timestamp(timestamp_seconds: float | None) -> str:
    if timestamp_seconds is None:
        return "-"
    return datetime.fromtimestamp(float(timestamp_seconds), UTC).strftime("%H:%M:%S.%f")[:-3] + " UTC"


def _trade_side_label(side: Any) -> str:
    if side in (1, "1", "+1"):
        return "Buy"
    if side in (-1, "-1"):
        return "Sell"
    if isinstance(side, str):
        text = side.strip().lower()
        if text in {"b", "buy", "bid", "buyer"}:
            return "Buy"
        if text in {"s", "sell", "ask", "seller", "a"}:
            return "Sell"
    return "-"


def _trade_side_tag(side: Any) -> str:
    label = _trade_side_label(side)
    if label == "Buy":
        return "buy"
    if label == "Sell":
        return "sell"
    return "neutral"


def _trade_subscription_message(coin: str) -> dict[str, Any]:
    return {
        "method": "subscribe",
        "subscription": {
            "type": "trades",
            "coin": coin,
        },
    }


def _info_endpoint_from_ws_url(url: str) -> str:
    normalized = url.strip()
    if normalized == "wss://api.hyperliquid.xyz/ws":
        return "https://api.hyperliquid.xyz/info"
    if normalized == "wss://api.hyperliquid-testnet.xyz/ws":
        return "https://api.hyperliquid-testnet.xyz/info"
    if normalized.endswith("/ws"):
        if normalized.startswith("wss://"):
            return "https://" + normalized[len("wss://") : -3] + "/info"
        if normalized.startswith("ws://"):
            return "http://" + normalized[len("ws://") : -3] + "/info"
    raise ValueError("Unable to infer Hyperliquid info endpoint from websocket URL.")


def _fetch_recent_adv_from_candles(
    *,
    coin: str,
    ws_url: str,
    lookback_days: int = AUTO_ADV_LOOKBACK_DAYS,
    timeout_seconds: float = 10.0,
) -> float:
    info_url = _info_endpoint_from_ws_url(ws_url)
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - lookback_days * 24 * 60 * 60 * 1000
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": "1d",
            "startTime": start_ms,
            "endTime": now_ms,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        info_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:
        raw = response.read()
    decoded = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, list) or not decoded:
        raise ValueError(f"No daily candle data returned for {coin}.")
    volumes: list[float] = []
    for candle in decoded:
        if not isinstance(candle, dict):
            continue
        try:
            volumes.append(float(candle["v"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not volumes:
        raise ValueError(f"Daily candle response for {coin} did not contain usable volume data.")
    return sum(volumes) / len(volumes)


class MainMarketGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Microstructure Dashboard")
        self.root.geometry("1680x980")
        self.root.minsize(1420, 860)

        self.coin_var = tk.StringVar(value="BTC")
        self.url_var = tk.StringVar(value=HYPERLIQUID_MAINNET_WS_URL)
        self.model_var = tk.StringVar(value=str(DEFAULT_MODEL_PATH))
        self.estimator_var = tk.StringVar(value="G1")
        self.max_points_var = tk.StringVar(value="300")

        self.ofi_window_var = tk.StringVar(value="50")
        self.kyle_window_var = tk.StringVar(value="50")
        self.vpin_buckets_var = tk.StringVar(value="50")
        self.vpin_support_var = tk.StringVar(value="50")
        self.vpin_adv_var = tk.StringVar(value="")
        self.vpin_bucket_volume_var = tk.StringVar(value="")

        self.hazard_scale_var = tk.StringVar(value="1.4")
        self.hazard_decay_var = tk.StringVar(value="4.0")
        self.hazard_horizon_var = tk.StringVar(value="0.25")

        self.mm_gamma_var = tk.StringVar(value="0.1")
        self.mm_sigma_var = tk.StringVar(value="0.015")
        self.mm_arrival_scale_var = tk.StringVar(value="1.25")
        self.mm_arrival_decay_var = tk.StringVar(value="4.0")
        self.mm_horizon_var = tk.StringVar(value="30.0")
        self.mm_tick_var = tk.StringVar(value="0.5")
        self.mm_inventory_var = tk.StringVar(value="0.0")

        self.adjust_micro_w_var = tk.StringVar(value="1.0")
        self.adjust_obi_w_var = tk.StringVar(value="0.0")
        self.adjust_ofi_w_var = tk.StringVar(value="0.0")
        self.adjust_toxicity_w_var = tk.StringVar(value="0.0")
        self.adjust_impact_w_var = tk.StringVar(value="1.0")
        self.adjust_max_var = tk.StringVar(value="")

        self.status_var = tk.StringVar(value="Ready")
        self.time_var = tk.StringVar(value="Book Time: -")
        self.bidask_var = tk.StringVar(value="Bid / Ask: -")
        self.mid_var = tk.StringVar(value="Mid: -")
        self.spread_var = tk.StringVar(value="Spread: -")
        self.microprice_var = tk.StringVar(value="Microprice: -")
        self.obi_var = tk.StringVar(value="OBI: -")
        self.raw_obi_var = tk.StringVar(value="Raw OBI: -")
        self.ofi_var = tk.StringVar(value="Rolling OFI: -")
        self.vpin_var = tk.StringVar(value="VPIN: -")
        self.kyle_var = tk.StringVar(value="Kyle Lambda: -")
        self.adjustment_var = tk.StringVar(value="Adjustment: -")
        self.adjusted_ref_var = tk.StringVar(value="Adjusted Ref: -")
        self.quote_var = tk.StringVar(value="MM Quote: -")
        self.hazard_var = tk.StringVar(value="Fill Hazard: -")
        self.trade_status_var = tk.StringVar(value="Last Trade: -")

        self.data_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.active_connection: Any | None = None
        self.model_cache: StreamingModel | None = None

        self.latest_vpin: float | None = None
        self.latest_kyle_lambda: float | None = None
        self.latest_signed_flow: float | None = None

        self.x_series: deque[int] = deque(maxlen=300)
        self.mid_series: deque[float] = deque(maxlen=300)
        self.microprice_series: deque[float] = deque(maxlen=300)
        self.adjusted_ref_series: deque[float] = deque(maxlen=300)
        self.ofi_series: deque[float] = deque(maxlen=300)
        self.raw_obi_series: deque[float] = deque(maxlen=300)
        self.vpin_series: deque[float] = deque(maxlen=300)
        self.kyle_series: deque[float] = deque(maxlen=300)
        self.point_counter = 0
        self.time_sales: deque[dict[str, Any]] = deque(maxlen=TIME_SALES_ROWS)

        self._build_ui()
        self._reload_model()
        self.root.after(GUI_POLL_MS, self._poll_queue)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=0, minsize=410)
        container.columnconfigure(1, weight=1)
        container.columnconfigure(2, weight=0, minsize=360)
        container.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(container)
        sidebar.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        sidebar.columnconfigure(0, weight=1)

        charts = ttk.Frame(container)
        charts.grid(row=0, column=1, sticky="nsew")
        charts.columnconfigure(0, weight=1)
        charts.rowconfigure(0, weight=1)

        tape_panel = ttk.Frame(container)
        tape_panel.grid(row=0, column=2, sticky="nsew", padx=(10, 0))
        tape_panel.columnconfigure(0, weight=1)
        tape_panel.rowconfigure(1, weight=1)

        self._build_sidebar(sidebar)
        self._build_charts(charts)
        self._build_tape_panel(tape_panel)

    def _build_sidebar(self, parent: ttk.Frame) -> None:
        session = ttk.LabelFrame(parent, text="Session", padding=10)
        session.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        session.columnconfigure(1, weight=1)
        ttk.Label(session, text="Coin").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(session, textvariable=self.coin_var).grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Label(session, text="Websocket URL").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(session, textvariable=self.url_var).grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Label(session, text="Microprice Model").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(session, textvariable=self.model_var).grid(row=2, column=1, sticky="ew", pady=3)
        buttons = ttk.Frame(session)
        buttons.grid(row=3, column=1, sticky="w", pady=(3, 0))
        ttk.Button(buttons, text="Browse", command=self._browse_model).pack(side="left")
        ttk.Button(buttons, text="Reload", command=self._reload_model).pack(side="left", padx=(6, 0))
        ttk.Label(session, text="Estimator").grid(row=4, column=0, sticky="w", pady=3)
        self.estimator_combo = ttk.Combobox(
            session,
            textvariable=self.estimator_var,
            values=("G1", "G_star"),
            state="readonly",
        )
        self.estimator_combo.grid(row=4, column=1, sticky="ew", pady=3)
        ttk.Label(session, text="Chart Points").grid(row=5, column=0, sticky="w", pady=3)
        ttk.Entry(session, textvariable=self.max_points_var).grid(row=5, column=1, sticky="ew", pady=3)

        signal_frame = ttk.LabelFrame(parent, text="Signal Windows", padding=10)
        signal_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        signal_frame.columnconfigure(1, weight=1)
        ttk.Label(signal_frame, text="OFI Window").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(signal_frame, textvariable=self.ofi_window_var).grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Label(signal_frame, text="Kyle Window").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(signal_frame, textvariable=self.kyle_window_var).grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Label(signal_frame, text="VPIN Buckets / Day").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(signal_frame, textvariable=self.vpin_buckets_var).grid(row=2, column=1, sticky="ew", pady=3)
        ttk.Label(signal_frame, text="VPIN Support").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Entry(signal_frame, textvariable=self.vpin_support_var).grid(row=3, column=1, sticky="ew", pady=3)
        ttk.Label(signal_frame, text="ADV").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Entry(signal_frame, textvariable=self.vpin_adv_var).grid(row=4, column=1, sticky="ew", pady=3)
        ttk.Label(signal_frame, text="Bucket Volume").grid(row=5, column=0, sticky="w", pady=3)
        ttk.Entry(signal_frame, textvariable=self.vpin_bucket_volume_var).grid(row=5, column=1, sticky="ew", pady=3)

        hazard_frame = ttk.LabelFrame(parent, text="Fill Hazard", padding=10)
        hazard_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        hazard_frame.columnconfigure(1, weight=1)
        ttk.Label(hazard_frame, text="Intensity Scale").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(hazard_frame, textvariable=self.hazard_scale_var).grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Label(hazard_frame, text="Decay").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(hazard_frame, textvariable=self.hazard_decay_var).grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Label(hazard_frame, text="Horizon Sec").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(hazard_frame, textvariable=self.hazard_horizon_var).grid(row=2, column=1, sticky="ew", pady=3)

        adjust_frame = ttk.LabelFrame(parent, text="Reservation Adjuster", padding=10)
        adjust_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        adjust_frame.columnconfigure(1, weight=1)
        rows = [
            ("Micro W", self.adjust_micro_w_var),
            ("OBI W", self.adjust_obi_w_var),
            ("OFI W", self.adjust_ofi_w_var),
            ("Toxicity W", self.adjust_toxicity_w_var),
            ("Impact W", self.adjust_impact_w_var),
            ("Max Adj", self.adjust_max_var),
        ]
        for row, (label, variable) in enumerate(rows):
            ttk.Label(adjust_frame, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(adjust_frame, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=3)

        mm_frame = ttk.LabelFrame(parent, text="MM Quote", padding=10)
        mm_frame.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        mm_frame.columnconfigure(1, weight=1)
        rows = [
            ("Gamma", self.mm_gamma_var),
            ("Sigma", self.mm_sigma_var),
            ("Arrival A", self.mm_arrival_scale_var),
            ("Arrival k", self.mm_arrival_decay_var),
            ("Horizon", self.mm_horizon_var),
            ("Tick Size", self.mm_tick_var),
            ("Inventory", self.mm_inventory_var),
        ]
        for row, (label, variable) in enumerate(rows):
            ttk.Label(mm_frame, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(mm_frame, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=3)

        controls = ttk.Frame(parent)
        controls.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        controls.columnconfigure((0, 1, 2), weight=1)
        self.start_button = ttk.Button(controls, text="Start", command=self._start)
        self.start_button.grid(row=0, column=0, sticky="ew")
        self.stop_button = ttk.Button(controls, text="Stop", command=self._stop, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(controls, text="Clear", command=self._clear_state).grid(row=0, column=2, sticky="ew")

        stats = ttk.LabelFrame(parent, text="Combined View", padding=10)
        stats.grid(row=6, column=0, sticky="nsew")
        stats.columnconfigure(0, weight=1)
        labels = [
            self.status_var,
            self.time_var,
            self.bidask_var,
            self.mid_var,
            self.spread_var,
            self.microprice_var,
            self.obi_var,
            self.raw_obi_var,
            self.ofi_var,
            self.vpin_var,
            self.kyle_var,
            self.adjustment_var,
            self.adjusted_ref_var,
            self.quote_var,
            self.hazard_var,
            self.trade_status_var,
        ]
        for row, variable in enumerate(labels):
            ttk.Label(stats, textvariable=variable, wraplength=380, justify="left").grid(
                row=row, column=0, sticky="w", pady=1
            )

    def _build_charts(self, parent: ttk.Frame) -> None:
        figure = Figure(figsize=(11.5, 8.8), dpi=100, constrained_layout=True)
        grid = figure.add_gridspec(3, 1, height_ratios=(3.0, 1.6, 1.4))
        self.price_axis = figure.add_subplot(grid[0])
        self.flow_axis = figure.add_subplot(grid[1], sharex=self.price_axis)
        self.tox_axis = figure.add_subplot(grid[2], sharex=self.price_axis)

        (self.mid_line,) = self.price_axis.plot([], [], color="#1f77b4", linewidth=1.7, label="Mid")
        (self.micro_line,) = self.price_axis.plot([], [], color="#d62728", linewidth=1.8, label="Microprice")
        (self.adjusted_ref_line,) = self.price_axis.plot([], [], color="#2ca02c", linewidth=1.6, linestyle="--", label="Adj Ref")

        (self.ofi_line,) = self.flow_axis.plot([], [], color="#ff7f0e", linewidth=1.5, label="Rolling OFI")
        (self.raw_obi_line,) = self.flow_axis.plot([], [], color="#9467bd", linewidth=1.4, label="Raw OBI")

        (self.vpin_line,) = self.tox_axis.plot([], [], color="#8c564b", linewidth=1.5, label="VPIN")
        (self.kyle_line,) = self.tox_axis.plot([], [], color="#e377c2", linewidth=1.4, label="Kyle Lambda")

        self.price_axis.set_title("Market and Adjusted Reference")
        self.price_axis.set_ylabel("Price")
        self.flow_axis.set_title("Order-Flow Signals")
        self.flow_axis.set_ylabel("Flow")
        self.tox_axis.set_title("Toxicity and Impact")
        self.tox_axis.set_ylabel("Metric")
        self.tox_axis.set_xlabel("Observation")
        for axis in (self.price_axis, self.flow_axis, self.tox_axis):
            axis.grid(True, alpha=0.25)
            axis.legend(loc="upper left")

        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        self.price_axis.yaxis.set_major_formatter(formatter)

        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas = canvas

    def _build_tape_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Time & Sales", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 6))
        columns = ("time", "side", "price", "size")
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=TIME_SALES_ROWS)
        tree.heading("time", text="Time")
        tree.heading("side", text="Side")
        tree.heading("price", text="Price")
        tree.heading("size", text="Size")
        tree.column("time", width=110, anchor="w")
        tree.column("side", width=60, anchor="center")
        tree.column("price", width=90, anchor="e")
        tree.column("size", width=90, anchor="e")
        tree.tag_configure("buy", foreground="#15803d")
        tree.tag_configure("sell", foreground="#b91c1c")
        tree.tag_configure("neutral", foreground="#444444")
        tree.grid(row=1, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        tree.configure(yscrollcommand=scrollbar.set)
        self.tape_tree = tree

    def _browse_model(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select Microprice Model JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(ROOT / "models"),
        )
        if selected:
            self.model_var.set(selected)
            self._reload_model()

    def _reload_model(self) -> None:
        path = self.model_var.get().strip()
        if not path:
            self.model_cache = None
            return
        try:
            model = load_streaming_model(path)
        except Exception as exc:
            self.model_cache = None
            messagebox.showerror("Model Error", str(exc))
            return
        self.model_cache = model
        if isinstance(model, FittedMultilevelMicropriceModel):
            self.estimator_combo.configure(values=("multilevel",), state="disabled")
            self.estimator_var.set("multilevel")
        else:
            values = model.available_estimators()
            self.estimator_combo.configure(values=values, state="readonly")
            if self.estimator_var.get().strip() not in values:
                self.estimator_var.set(values[0])

    def _get_positive_int(self, value: str, name: str, minimum: int = 1) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer.") from exc
        if parsed < minimum:
            raise ValueError(f"{name} must be at least {minimum}.")
        return parsed

    def _get_float(self, value: str, name: str) -> float:
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be numeric.") from exc
        if not np.isfinite(parsed):
            raise ValueError(f"{name} must be finite.")
        return parsed

    def _collect_settings(self) -> dict[str, Any]:
        coin = self.coin_var.get().strip().upper()
        if not coin:
            raise ValueError("Coin is required.")
        model = self.model_cache or load_streaming_model(self.model_var.get().strip())
        max_points = self._get_positive_int(self.max_points_var.get().strip(), "Chart Points", minimum=20)
        self._resize_series(max_points)

        bucket_volume_text = self.vpin_bucket_volume_var.get().strip()
        adv_text = self.vpin_adv_var.get().strip()
        bucket_volume = None if not bucket_volume_text else self._get_float(bucket_volume_text, "Bucket Volume")
        average_daily_volume = None if not adv_text else self._get_float(adv_text, "ADV")
        buckets_per_day = self._get_positive_int(self.vpin_buckets_var.get().strip(), "VPIN Buckets / Day")
        support_buckets = self._get_positive_int(self.vpin_support_var.get().strip(), "VPIN Support")
        if bucket_volume is None and average_daily_volume is not None:
            bucket_volume = average_daily_volume / buckets_per_day

        adjuster = LinearReservationPriceAdjuster(
            microprice_weight=self._get_float(self.adjust_micro_w_var.get().strip(), "Micro W"),
            obi_weight=self._get_float(self.adjust_obi_w_var.get().strip(), "OBI W"),
            ofi_weight=self._get_float(self.adjust_ofi_w_var.get().strip(), "OFI W"),
            toxicity_weight=self._get_float(self.adjust_toxicity_w_var.get().strip(), "Toxicity W"),
            impact_weight=self._get_float(self.adjust_impact_w_var.get().strip(), "Impact W"),
            max_adjustment=None
            if not self.adjust_max_var.get().strip()
            else self._get_float(self.adjust_max_var.get().strip(), "Max Adj"),
        )
        hazard_model = ExponentialFillHazardModel(
            intensity_scale=self._get_float(self.hazard_scale_var.get().strip(), "Hazard Scale"),
            decay=self._get_float(self.hazard_decay_var.get().strip(), "Hazard Decay"),
        )
        hazard_horizon = self._get_float(self.hazard_horizon_var.get().strip(), "Hazard Horizon")
        mm_params = AvellanedaStoikovParameters(
            gamma=self._get_float(self.mm_gamma_var.get().strip(), "Gamma"),
            sigma=max(self._get_float(self.mm_sigma_var.get().strip(), "Sigma"), 1e-12),
            arrival_rate_scale=max(self._get_float(self.mm_arrival_scale_var.get().strip(), "Arrival A"), 1e-12),
            arrival_rate_decay=max(self._get_float(self.mm_arrival_decay_var.get().strip(), "Arrival k"), 1e-12),
            horizon_seconds=max(self._get_float(self.mm_horizon_var.get().strip(), "MM Horizon"), 1e-9),
            tick_size=max(self._get_float(self.mm_tick_var.get().strip(), "MM Tick Size"), 1e-12),
        )
        return {
            "coin": coin,
            "url": self.url_var.get().strip() or HYPERLIQUID_MAINNET_WS_URL,
            "model": model,
            "estimator": self.estimator_var.get().strip() or "G1",
            "ofi_window": self._get_positive_int(self.ofi_window_var.get().strip(), "OFI Window"),
            "kyle_window": self._get_positive_int(self.kyle_window_var.get().strip(), "Kyle Window", minimum=2),
            "bucket_volume": bucket_volume,
            "average_daily_volume": average_daily_volume,
            "buckets_per_day": buckets_per_day,
            "support_buckets": support_buckets,
            "adjuster": adjuster,
            "hazard_model": hazard_model,
            "hazard_horizon": hazard_horizon,
            "mm_model": AvellanedaStoikovModel(mm_params),
            "inventory": self._get_float(self.mm_inventory_var.get().strip(), "Inventory"),
        }

    def _resize_series(self, max_points: int) -> None:
        self.x_series = deque(self.x_series, maxlen=max_points)
        self.mid_series = deque(self.mid_series, maxlen=max_points)
        self.microprice_series = deque(self.microprice_series, maxlen=max_points)
        self.adjusted_ref_series = deque(self.adjusted_ref_series, maxlen=max_points)
        self.ofi_series = deque(self.ofi_series, maxlen=max_points)
        self.raw_obi_series = deque(self.raw_obi_series, maxlen=max_points)
        self.vpin_series = deque(self.vpin_series, maxlen=max_points)
        self.kyle_series = deque(self.kyle_series, maxlen=max_points)

    def _clear_state(self) -> None:
        max_points = max(self.x_series.maxlen or 300, 20)
        self.x_series = deque(maxlen=max_points)
        self.mid_series = deque(maxlen=max_points)
        self.microprice_series = deque(maxlen=max_points)
        self.adjusted_ref_series = deque(maxlen=max_points)
        self.ofi_series = deque(maxlen=max_points)
        self.raw_obi_series = deque(maxlen=max_points)
        self.vpin_series = deque(maxlen=max_points)
        self.kyle_series = deque(maxlen=max_points)
        self.time_sales.clear()
        self.point_counter = 0
        self.latest_vpin = None
        self.latest_kyle_lambda = None
        self.latest_signed_flow = None
        for item in self.tape_tree.get_children():
            self.tape_tree.delete(item)
        self._refresh_plot()

    def _start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo("Already Running", "The combined dashboard stream is already running.")
            return
        try:
            settings = self._collect_settings()
        except Exception as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return
        self._clear_state()
        self.stop_event.clear()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Connecting...")
        self.worker = threading.Thread(target=self._run_stream, args=(settings,), daemon=True)
        self.worker.start()

    def _stop(self) -> None:
        self.stop_event.set()
        self.status_var.set("Stopping...")
        self.stop_button.configure(state="disabled")
        if self.active_connection is not None:
            try:
                self.active_connection.close()
            except Exception:
                pass

    def _build_vpin_engine(self, settings: dict[str, Any]) -> OnlineVPIN:
        bucket_volume = settings["bucket_volume"]
        if bucket_volume is None:
            try:
                adv = _fetch_recent_adv_from_candles(coin=settings["coin"], ws_url=settings["url"])
            except Exception as exc:
                raise RuntimeError(
                    "Combined GUI needs either ADV / Bucket Volume or a successful historical ADV fetch."
                ) from exc
            settings["average_daily_volume"] = adv
            bucket_volume = adv / settings["buckets_per_day"]
            settings["bucket_volume"] = bucket_volume
        return OnlineVPIN(
            bucket_volume=float(bucket_volume),
            support_buckets=int(settings["support_buckets"]),
            classification="aggressor",
        )

    def _run_stream(self, settings: dict[str, Any]) -> None:
        connection: Any | None = None
        try:
            model: StreamingModel = settings["model"]
            estimator = "multilevel" if isinstance(model, FittedMultilevelMicropriceModel) else settings["estimator"]
            ofi_engine = OnlineOFI(window=int(settings["ofi_window"]))
            vpin_engine = self._build_vpin_engine(settings)
            mm_model: AvellanedaStoikovModel = settings["mm_model"]
            hazard_model: ExponentialFillHazardModel = settings["hazard_model"]
            kyle_window = int(settings["kyle_window"])
            signed_flow_window: deque[float] = deque(maxlen=kyle_window)
            price_change_window: deque[float] = deque(maxlen=kyle_window)
            last_trade_price: float | None = None
            last_trade_time: float | None = None
            last_trade_side: int | None = None
            latest_vpin: float | None = None
            latest_kyle: float | None = None
            latest_signed_flow: float | None = None

            connection = _default_websocket_factory(settings["url"])
            self.active_connection = connection
            connection.send(json.dumps(build_subscription_message(settings["coin"], "l2Book")))
            connection.send(json.dumps(_trade_subscription_message(settings["coin"])))
            self.data_queue.put(("status", f"Streaming combined dashboard for {settings['coin']}"))
            while not self.stop_event.is_set():
                raw = connection.recv()
                if raw is None:
                    break
                try:
                    parsed = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    continue
                channel = parsed.get("channel")
                if channel == "trades":
                    trades = parsed.get("data")
                    if not isinstance(trades, list):
                        continue
                    for item in trades:
                        if not isinstance(item, dict):
                            continue
                        try:
                            trade = MarketTrade(
                                timestamp=float(item["time"]),
                                price=float(item["px"]),
                                volume=float(item["sz"]),
                                side=str(item["side"]),
                            )
                        except (KeyError, TypeError, ValueError):
                            continue
                        snapshot = vpin_engine.update(trade)
                        latest_vpin = None if snapshot.current_vpin is None else float(snapshot.current_vpin)
                        if trade.side is not None:
                            latest_signed_flow = float(trade.side) * float(trade.volume)
                            if last_trade_price is not None:
                                signed_flow_window.append(latest_signed_flow)
                                price_change_window.append(float(trade.price) - float(last_trade_price))
                                if len(signed_flow_window) >= 2:
                                    try:
                                        latest_kyle = fit_kyle_lambda(
                                            np.asarray(signed_flow_window, dtype=np.float64),
                                            np.asarray(price_change_window, dtype=np.float64),
                                        ).lambda_
                                    except Exception:
                                        latest_kyle = None
                        last_trade_price = float(trade.price)
                        last_trade_time = trade.timestamp.timestamp()
                        last_trade_side = trade.side
                        self.data_queue.put(
                            (
                                "trade",
                                {
                                    "time": last_trade_time,
                                    "side": last_trade_side,
                                    "price": float(trade.price),
                                    "size": float(trade.volume),
                                    "vpin": latest_vpin,
                                    "kyle_lambda": latest_kyle,
                                    "signed_flow": latest_signed_flow,
                                },
                            )
                        )
                    continue

                if channel != "l2Book":
                    continue
                payload = build_live_payload_from_message(parsed, model, estimator=estimator)
                if payload is None:
                    continue
                ofi_snapshot = ofi_engine.update(
                    bid=float(payload["bid"]),
                    bid_size=float(payload["bid_size"]),
                    ask=float(payload["ask"]),
                    ask_size=float(payload["ask_size"]),
                )
                latest_obi = float(payload["imbalance"])
                microprice = float(payload["microprice"])
                reference_price = float(payload["midprice"])
                adjustment = settings["adjuster"].compute(
                    reference_price=reference_price,
                    microprice=microprice,
                    obi=latest_obi,
                    ofi=ofi_snapshot.rolling_ofi,
                    toxicity=latest_vpin,
                    kyle_lambda=latest_kyle,
                    signed_flow=latest_signed_flow,
                )
                snapshot = MarketSnapshot(
                    timestamp=float(payload["timestamp"]) / 1000.0,
                    best_bid=float(payload["bid"]),
                    best_ask=float(payload["ask"]),
                    bid_size=float(payload["bid_size"]),
                    ask_size=float(payload["ask_size"]),
                )
                quote = mm_model.compute_quote(
                    snapshot,
                    inventory=float(settings["inventory"]),
                    reference_price=reference_price,
                    reservation_price_adjustment=adjustment.total_adjustment,
                    sigma=mm_model.params.sigma,
                    dt_seconds=float(settings["hazard_horizon"]),
                )
                bid_distance = max(adjustment.adjusted_reference_price - quote.bid_price, 0.0)
                ask_distance = max(quote.ask_price - adjustment.adjusted_reference_price, 0.0)
                hazard_payload = {
                    "bid_fill_probability": hazard_model.fill_probability(bid_distance, float(settings["hazard_horizon"])),
                    "ask_fill_probability": hazard_model.fill_probability(ask_distance, float(settings["hazard_horizon"])),
                }
                self.data_queue.put(
                    (
                        "book",
                        {
                            "timestamp": float(payload["timestamp"]) / 1000.0,
                            "bid": float(payload["bid"]),
                            "ask": float(payload["ask"]),
                            "midprice": reference_price,
                            "spread": quoted_spread(payload["bid"], payload["ask"]),
                            "microprice": microprice,
                            "obi": latest_obi,
                            "rolling_ofi": float(ofi_snapshot.rolling_ofi),
                            "vpin": latest_vpin,
                            "kyle_lambda": latest_kyle,
                            "adjustment": adjustment.total_adjustment,
                            "adjusted_reference": adjustment.adjusted_reference_price,
                            "quote": quote,
                            "hazard": hazard_payload,
                            "last_trade_price": last_trade_price,
                            "last_trade_time": last_trade_time,
                            "last_trade_side": last_trade_side,
                        },
                    )
                )
        except Exception as exc:
            self.data_queue.put(("error", str(exc)))
        finally:
            self.active_connection = None
            try:
                if connection is not None:
                    connection.close()
            except Exception:
                pass
            self.data_queue.put(("stopped", "Stopped"))

    def _apply_trade_update(self, payload: dict[str, Any]) -> None:
        self.latest_vpin = payload.get("vpin")
        self.latest_kyle_lambda = payload.get("kyle_lambda")
        self.latest_signed_flow = payload.get("signed_flow")
        side = payload.get("side")
        side_text = _trade_side_label(side)
        self.trade_status_var.set(
            f"Last Trade: {_fmt_timestamp(payload.get('time'))} | {side_text} | px={_fmt_float(payload.get('price'))} | sz={_fmt_float(payload.get('size'))}"
        )
        self.time_sales.appendleft(payload)
        self._refresh_tape()

    def _apply_book_update(self, payload: dict[str, Any]) -> None:
        self.point_counter += 1
        self.x_series.append(self.point_counter)
        self.mid_series.append(float(payload["midprice"]))
        self.microprice_series.append(float(payload["microprice"]))
        self.adjusted_ref_series.append(float(payload["adjusted_reference"]))
        self.ofi_series.append(float(payload["rolling_ofi"]))
        self.raw_obi_series.append(float(payload["obi"]))
        self.vpin_series.append(float("nan") if payload.get("vpin") is None else float(payload["vpin"]))
        self.kyle_series.append(float("nan") if payload.get("kyle_lambda") is None else float(payload["kyle_lambda"]))

        self.time_var.set(f"Book Time: {_fmt_timestamp(payload.get('timestamp'))}")
        self.bidask_var.set(f"Bid / Ask: {_fmt_float(payload['bid'])} / {_fmt_float(payload['ask'])}")
        self.mid_var.set(f"Mid: {_fmt_float(payload['midprice'])}")
        self.spread_var.set(f"Spread: {_fmt_float(payload['spread'])}")
        self.microprice_var.set(f"Microprice: {_fmt_float(payload['microprice'])}")
        self.obi_var.set(f"OBI: {_fmt_float(payload['obi'])}")
        self.raw_obi_var.set(f"Raw OBI: {_fmt_float(payload['obi'])}")
        self.ofi_var.set(f"Rolling OFI: {_fmt_float(payload['rolling_ofi'])}")
        self.vpin_var.set(
            "VPIN: warming up" if payload.get("vpin") is None else f"VPIN: {_fmt_float(payload['vpin'])}"
        )
        self.kyle_var.set(
            "Kyle Lambda: warming up"
            if payload.get("kyle_lambda") is None
            else f"Kyle Lambda: {_fmt_float(payload['kyle_lambda'])}"
        )
        self.adjustment_var.set(f"Adjustment: {_fmt_float(payload['adjustment'])}")
        self.adjusted_ref_var.set(f"Adjusted Ref: {_fmt_float(payload['adjusted_reference'])}")
        quote = payload["quote"]
        self.quote_var.set(
            f"MM Quote: bid={_fmt_float(quote.bid_price)} | ask={_fmt_float(quote.ask_price)} | reservation={_fmt_float(quote.reservation_price)}"
        )
        hazard = payload["hazard"]
        self.hazard_var.set(
            f"Fill Hazard: bid={_fmt_float(hazard['bid_fill_probability'])} | ask={_fmt_float(hazard['ask_fill_probability'])}"
        )
        self.status_var.set(f"Streaming combined dashboard for {self.coin_var.get().strip().upper()}")
        self._refresh_plot()

    def _refresh_tape(self) -> None:
        for item in self.tape_tree.get_children():
            self.tape_tree.delete(item)
        for trade in self.time_sales:
            side = trade.get("side")
            side_text = _trade_side_label(side)
            side_tag = _trade_side_tag(side)
            self.tape_tree.insert(
                "",
                "end",
                values=(
                    _fmt_timestamp(trade.get("time")).replace(" UTC", ""),
                    side_text,
                    _fmt_float(trade.get("price")),
                    _fmt_float(trade.get("size")),
                ),
                tags=(side_tag,),
            )

    def _refresh_plot(self) -> None:
        if not self.x_series:
            self.mid_line.set_data([], [])
            self.micro_line.set_data([], [])
            self.adjusted_ref_line.set_data([], [])
            self.ofi_line.set_data([], [])
            self.raw_obi_line.set_data([], [])
            self.vpin_line.set_data([], [])
            self.kyle_line.set_data([], [])
            self.canvas.draw_idle()
            return

        x_values = list(self.x_series)
        self.mid_line.set_data(x_values, list(self.mid_series))
        self.micro_line.set_data(x_values, list(self.microprice_series))
        self.adjusted_ref_line.set_data(x_values, list(self.adjusted_ref_series))
        self.ofi_line.set_data(x_values, list(self.ofi_series))
        self.raw_obi_line.set_data(x_values, list(self.raw_obi_series))
        self.vpin_line.set_data(x_values, list(self.vpin_series))
        self.kyle_line.set_data(x_values, list(self.kyle_series))

        x_min = x_values[0]
        x_max = x_values[-1] if x_values[-1] > x_min else x_min + 1
        for axis in (self.price_axis, self.flow_axis, self.tox_axis):
            axis.set_xlim(x_min, x_max)

        def _set_axis_limits(axis: Any, arrays: list[deque[float]]) -> None:
            values = []
            for series in arrays:
                arr = np.asarray(series, dtype=np.float64)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    values.append(arr)
            if not values:
                axis.set_ylim(-1.0, 1.0)
                return
            joined = np.concatenate(values)
            low = float(joined.min())
            high = float(joined.max())
            pad = max(1e-6, 0.1 * (high - low)) if high > low else max(1e-6, 0.05 * max(abs(low), 1.0))
            axis.set_ylim(low - pad, high + pad)

        _set_axis_limits(self.price_axis, [self.mid_series, self.microprice_series, self.adjusted_ref_series])
        _set_axis_limits(self.flow_axis, [self.ofi_series, self.raw_obi_series])
        _set_axis_limits(self.tox_axis, [self.vpin_series, self.kyle_series])
        self.canvas.draw_idle()

    def _poll_queue(self) -> None:
        try:
            event_type, payload = self.data_queue.get_nowait()
        except queue.Empty:
            self.root.after(GUI_POLL_MS, self._poll_queue)
            return

        if event_type == "trade":
            self._apply_trade_update(payload)
        elif event_type == "book":
            self._apply_book_update(payload)
        elif event_type == "status":
            self.status_var.set(str(payload))
        elif event_type == "error":
            self.status_var.set("Failed")
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.worker = None
            messagebox.showerror("Stream Error", str(payload))
        elif event_type == "stopped":
            self.status_var.set(str(payload))
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.worker = None

        next_delay = 0 if not self.data_queue.empty() else GUI_POLL_MS
        self.root.after(next_delay, self._poll_queue)


def launch_gui() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    MainMarketGui(root)
    root.mainloop()


def main() -> None:
    launch_gui()


if __name__ == "__main__":
    main()
