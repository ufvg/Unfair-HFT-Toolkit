from __future__ import annotations

from collections import deque
from pathlib import Path
import queue
import sys
import threading
from typing import Any
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

from microprice.L1microprice import (
    ESTIMATOR_CHOICES,
    GUI_POLL_MS,
    HYPERLIQUID_MAINNET_WS_URL,
    _default_websocket_factory,
    load_streaming_model,
    stream_hyperliquid_microprice,
)
from microprice.calibration import FittedMicropriceModel
from microprice.multilevel_calibration import FittedMultilevelMicropriceModel

try:
    from local_strategies.adjustment_threshold_strategy import AdjustmentThresholdStrategy, StrategySnapshot
except ImportError:
    AdjustmentThresholdStrategy = None
    StrategySnapshot = Any

StreamingModel = FittedMicropriceModel | FittedMultilevelMicropriceModel

DEFAULT_PRIMARY_MODEL = ROOT / "models" / "btc_l1_model_20260122_20260221.json"


def _fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _model_family(model: StreamingModel) -> str:
    return "Multilevel" if isinstance(model, FittedMultilevelMicropriceModel) else "L1"


class MicropriceGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Microprice Monitor")
        self.root.geometry("1280x860")
        self.root.minsize(1080, 720)

        self.coin_var = tk.StringVar(value="BTC")
        self.model_var = tk.StringVar(value=str(DEFAULT_PRIMARY_MODEL))
        self.subscription_var = tk.StringVar(value="l2Book")
        self.estimator_var = tk.StringVar(value="G1")
        self.url_var = tk.StringVar(value=HYPERLIQUID_MAINNET_WS_URL)
        self.max_points_var = tk.StringVar(value="300")
        self.run_strategy_var = tk.BooleanVar(value=False)

        self.status_var = tk.StringVar(value="Ready")
        self.primary_model_info_var = tk.StringVar(value="Primary model: not loaded")
        self.usage_note_var = tk.StringVar(value="Load an L1 or multilevel model to stream live microprice.")
        self.last_feed_var = tk.StringVar(value="Feed: -")
        self.last_bidask_var = tk.StringVar(value="Bid / Ask: -")
        self.last_spread_var = tk.StringVar(value="Spread: -")
        self.last_mid_var = tk.StringVar(value="Mid: -")
        self.last_primary_micro_var = tk.StringVar(value="Primary Micro: -")
        self.last_primary_adjustment_var = tk.StringVar(value="Primary Adj: -")
        self.last_adjustment_ticks_var = tk.StringVar(value="Signal Ticks: -")
        self.last_imbalance_var = tk.StringVar(value="Imbalance: -")
        self.last_state_var = tk.StringVar(value="Model State: -")
        self.strategy_status_var = tk.StringVar(value="Strategy: unavailable")
        self.strategy_position_var = tk.StringVar(value="Position: -")
        self.strategy_pnl_var = tk.StringVar(value="PnL: -")
        self.strategy_trade_count_var = tk.StringVar(value="Closed Trades: -")

        self.data_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.active_connection: Any | None = None
        self.stop_event = threading.Event()
        self.primary_model_cache: StreamingModel | None = None
        self.show_adjustment_ticks_line = True
        self.strategy_engine = AdjustmentThresholdStrategy() if AdjustmentThresholdStrategy is not None else None

        self.mid_series: deque[float] = deque(maxlen=300)
        self.primary_micro_series: deque[float] = deque(maxlen=300)
        self.primary_adjustment_series: deque[float] = deque(maxlen=300)
        self.adjustment_ticks_series: deque[float] = deque(maxlen=300)
        self.strategy_pnl_series: deque[float] = deque(maxlen=300)
        self.x_series: deque[int] = deque(maxlen=300)
        self.point_counter = 0

        self._build_ui()
        self._reload_model_metadata()
        self.root.after(GUI_POLL_MS, self._poll_queue)

    def _reset_live_text(self) -> None:
        self.last_feed_var.set("Feed: -")
        self.last_bidask_var.set("Bid / Ask: -")
        self.last_spread_var.set("Spread: -")
        self.last_mid_var.set("Mid: -")
        self.last_primary_micro_var.set("Primary Micro: -")
        self.last_primary_adjustment_var.set("Primary Adj: -")
        self.last_adjustment_ticks_var.set("Signal Ticks: -")
        self.last_imbalance_var.set("Imbalance: -")
        self.last_state_var.set("Model State: -")

    def _resize_series(self, max_points: int) -> None:
        self.mid_series = deque(self.mid_series, maxlen=max_points)
        self.primary_micro_series = deque(self.primary_micro_series, maxlen=max_points)
        self.primary_adjustment_series = deque(self.primary_adjustment_series, maxlen=max_points)
        self.adjustment_ticks_series = deque(self.adjustment_ticks_series, maxlen=max_points)
        self.strategy_pnl_series = deque(self.strategy_pnl_series, maxlen=max_points)
        self.x_series = deque(self.x_series, maxlen=max_points)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=0, minsize=350)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(container)
        sidebar.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        sidebar.columnconfigure(0, weight=1)

        charts = ttk.Frame(container)
        charts.grid(row=0, column=1, sticky="nsew")
        charts.columnconfigure(0, weight=1)
        charts.rowconfigure(1, weight=1)

        model_frame = ttk.LabelFrame(sidebar, text="Model", padding=10)
        model_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        model_frame.columnconfigure(0, weight=1)
        ttk.Entry(model_frame, textvariable=self.model_var, width=40).grid(row=0, column=0, sticky="ew", pady=(0, 6))
        model_buttons = ttk.Frame(model_frame)
        model_buttons.grid(row=1, column=0, sticky="w", pady=(0, 6))
        ttk.Button(model_buttons, text="Browse", command=self._browse_model).pack(side="left")
        ttk.Button(model_buttons, text="Reload", command=self._reload_model_metadata).pack(side="left", padx=(6, 0))
        ttk.Label(model_frame, textvariable=self.primary_model_info_var, wraplength=320, justify="left").grid(
            row=2, column=0, sticky="w"
        )
        ttk.Label(
            model_frame,
            textvariable=self.usage_note_var,
            wraplength=320,
            justify="left",
            foreground="#555555",
        ).grid(row=3, column=0, sticky="w", pady=(4, 0))

        session_frame = ttk.LabelFrame(sidebar, text="Session", padding=10)
        session_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        session_frame.columnconfigure(1, weight=1)
        ttk.Label(session_frame, text="Coin").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(session_frame, textvariable=self.coin_var, width=18).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(session_frame, text="Subscription").grid(row=1, column=0, sticky="w", pady=4)
        self.subscription_combo = ttk.Combobox(
            session_frame,
            textvariable=self.subscription_var,
            values=("l2Book", "bbo"),
            state="readonly",
            width=16,
        )
        self.subscription_combo.grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Label(session_frame, text="Estimator").grid(row=2, column=0, sticky="w", pady=4)
        self.estimator_combo = ttk.Combobox(
            session_frame,
            textvariable=self.estimator_var,
            values=ESTIMATOR_CHOICES,
            state="readonly",
            width=16,
        )
        self.estimator_combo.grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Label(session_frame, text="Chart Points").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Spinbox(session_frame, from_=50, to=5000, textvariable=self.max_points_var, width=10).grid(
            row=3, column=1, sticky="w", pady=4
        )
        ttk.Checkbutton(session_frame, text="Run Strategy", variable=self.run_strategy_var).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=4
        )
        ttk.Label(session_frame, text="Websocket URL").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Entry(session_frame, textvariable=self.url_var).grid(row=5, column=1, sticky="ew", pady=4)

        controls = ttk.Frame(sidebar)
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        controls.columnconfigure((0, 1, 2), weight=1)
        self.start_button = ttk.Button(controls, text="Start Stream", command=self._start_stream)
        self.start_button.grid(row=0, column=0, sticky="ew")
        self.stop_button = ttk.Button(controls, text="Stop", command=self._stop_stream, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(controls, text="Clear Chart", command=self._clear_chart).grid(row=0, column=2, sticky="ew")

        info_frame = ttk.LabelFrame(sidebar, text="Live Stats", padding=10)
        info_frame.grid(row=3, column=0, sticky="nsew")
        info_frame.columnconfigure(0, weight=1)
        labels = [
            self.status_var,
            self.last_feed_var,
            self.last_bidask_var,
            self.last_spread_var,
            self.last_mid_var,
            self.last_primary_micro_var,
            self.last_primary_adjustment_var,
            self.last_adjustment_ticks_var,
            self.last_imbalance_var,
            self.last_state_var,
            self.strategy_status_var,
            self.strategy_position_var,
            self.strategy_pnl_var,
            self.strategy_trade_count_var,
        ]
        for row, variable in enumerate(labels):
            ttk.Label(info_frame, textvariable=variable, wraplength=320, justify="left").grid(
                row=row, column=0, sticky="w", pady=1
            )

        header = ttk.Frame(charts)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header,
            text="Asset Price and Microprice",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")

        figure = Figure(figsize=(12.5, 8.4), dpi=100, constrained_layout=True)
        grid = figure.add_gridspec(3, 1, height_ratios=(3.0, 1.4, 1.2))
        self.axis = figure.add_subplot(grid[0])
        self.adjustment_axis = figure.add_subplot(grid[1], sharex=self.axis)
        self.strategy_axis = figure.add_subplot(grid[2], sharex=self.axis)
        self.axis.set_title("Microprice vs Midprice")
        self.axis.set_ylabel("Price")
        self.axis.grid(True, alpha=0.25)
        price_formatter = ScalarFormatter(useOffset=False)
        price_formatter.set_scientific(False)
        self.axis.yaxis.set_major_formatter(price_formatter)
        self.adjustment_axis.set_title("Microprice Adjustment")
        self.adjustment_axis.set_ylabel("Adjustment")
        self.adjustment_axis.grid(True, alpha=0.25)
        self.strategy_axis.set_title("Strategy Equity")
        self.strategy_axis.set_xlabel("Observation")
        self.strategy_axis.set_ylabel("PnL")
        self.strategy_axis.grid(True, alpha=0.25)

        (self.mid_line,) = self.axis.plot([], [], color="#1f77b4", label="Mid", linewidth=1.8)
        (self.primary_micro_line,) = self.axis.plot([], [], color="#d62728", label="Microprice", linewidth=1.9)
        (self.primary_adjustment_line,) = self.adjustment_axis.plot(
            [], [], color="#2ca02c", label="Adjustment", linewidth=1.6
        )
        (self.adjustment_ticks_line,) = self.adjustment_axis.plot(
            [], [], color="#9467bd", label="Signal (ticks)", linewidth=1.5, linestyle="--"
        )
        (self.strategy_pnl_line,) = self.strategy_axis.plot(
            [], [], color="#ff7f0e", label="Total PnL", linewidth=1.6
        )
        self._refresh_legend()

        canvas = FigureCanvasTkAgg(figure, master=charts)
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self.canvas = canvas

    def _model_picker_dir(self) -> str:
        return str(ROOT / "models")

    def _browse_model(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select Primary Model JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=self._model_picker_dir(),
        )
        if selected:
            self.model_var.set(selected)
            self._reload_model_metadata()

    def _describe_model(self, model: StreamingModel, path: str) -> str:
        name = Path(path).name
        if isinstance(model, FittedMultilevelMicropriceModel):
            return (
                f"{name} | family=Multilevel | levels={model.levels} | decay={model.decay_lambda:.4g} "
                f"| dt={model.dt} | tick={model.tick_size:.6g}"
            )
        estimators = ", ".join(model.available_estimators())
        return (
            f"{name} | family=L1 | n_imb={model.n_imb} | n_spread={model.n_spread} "
            f"| dt={model.dt} | tick={model.tick_size:.6g} | estimators={estimators}"
        )

    def _set_text_var(self, value: str, *names: str) -> None:
        for name in names:
            variable = getattr(self, name, None)
            if variable is not None:
                variable.set(value)

    def _set_line_labels(self, model: StreamingModel | None) -> None:
        primary_label = "Microprice" if model is None else _model_family(model)
        self.primary_micro_line.set_label(f"{primary_label} Microprice")
        self.primary_adjustment_line.set_label(f"{primary_label} Adjustment")
        if hasattr(self.adjustment_ticks_line, "set_label"):
            self.adjustment_ticks_line.set_label(f"{primary_label} Signal (ticks)")
        self._refresh_legend()

    def _apply_model_controls(self, model: StreamingModel | None) -> None:
        if isinstance(model, FittedMultilevelMicropriceModel):
            self.estimator_var.set("multilevel")
            self.estimator_combo.configure(values=("multilevel",), state="disabled")
            self.subscription_var.set("l2Book")
            self.subscription_combo.configure(state="disabled")
            self.usage_note_var.set("Multilevel models require l2Book because they consume top-of-book depth levels.")
        elif isinstance(model, FittedMicropriceModel):
            estimator_values = model.available_estimators()
            self.estimator_combo.configure(values=estimator_values, state="readonly")
            if self.estimator_var.get().strip() not in estimator_values:
                self.estimator_var.set(estimator_values[0])
            self.subscription_combo.configure(state="readonly")
            self.usage_note_var.set("L1 models can stream from bbo or l2Book.")
        else:
            self.estimator_combo.configure(values=ESTIMATOR_CHOICES, state="readonly")
            self.subscription_combo.configure(state="readonly")
            self.usage_note_var.set("Load a model to inspect runtime requirements.")

        self._set_line_labels(model)

    def _reload_model_metadata(self, show_errors: bool = False) -> None:
        model_path = self.model_var.get().strip()
        model: StreamingModel | None = None
        if not model_path:
            self.primary_model_info_var.set("Primary model: required")
        else:
            try:
                model = load_streaming_model(model_path)
                self.primary_model_info_var.set(self._describe_model(model, model_path))
            except Exception as exc:
                self.primary_model_info_var.set(f"Primary model error: {exc}")
                if show_errors:
                    messagebox.showerror("Model Error", str(exc))
        self.primary_model_cache = model
        self._apply_model_controls(model)

    def _refresh_legend(self) -> None:
        price_handles = [self.mid_line, self.primary_micro_line]
        adjustment_handles = [self.primary_adjustment_line]
        if self.show_adjustment_ticks_line:
            adjustment_handles.append(self.adjustment_ticks_line)

        price_legend = self.axis.get_legend()
        if price_legend is not None:
            price_legend.remove()
        self.axis.legend(handles=price_handles, loc="upper left")

        adjustment_legend = self.adjustment_axis.get_legend()
        if adjustment_legend is not None:
            adjustment_legend.remove()
        self.adjustment_axis.legend(handles=adjustment_handles, loc="upper left")

        strategy_legend = self.strategy_axis.get_legend()
        if strategy_legend is not None:
            strategy_legend.remove()
        self.strategy_axis.legend(handles=[self.strategy_pnl_line], loc="upper left")

    def _set_adjustment_ticks_visibility(self, tick_size: float) -> None:
        self.show_adjustment_ticks_line = not np.isclose(float(tick_size), 1.0)
        self.adjustment_ticks_line.set_visible(self.show_adjustment_ticks_line)
        self._refresh_legend()

    def _get_max_points(self) -> int:
        try:
            value = int(self.max_points_var.get())
        except ValueError as exc:
            raise ValueError("Chart Points must be an integer.") from exc
        if value < 10:
            raise ValueError("Chart Points must be at least 10.")
        return value

    def _clear_chart(self) -> None:
        try:
            max_points = self._get_max_points()
        except ValueError as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return
        self.mid_series = deque(maxlen=max_points)
        self.primary_micro_series = deque(maxlen=max_points)
        self.primary_adjustment_series = deque(maxlen=max_points)
        self.adjustment_ticks_series = deque(maxlen=max_points)
        self.strategy_pnl_series = deque(maxlen=max_points)
        self.x_series = deque(maxlen=max_points)
        self.point_counter = 0
        if self.strategy_engine is not None:
            self.strategy_engine.reset()
        self._reset_live_text()
        self._update_strategy_labels(None)
        self._refresh_plot()

    def _validate_inputs(self) -> tuple[StreamingModel, dict[str, Any]]:
        coin = self.coin_var.get().strip().upper()
        if not coin:
            raise ValueError("Coin is required.")

        model_path = self.model_var.get().strip()
        if not model_path:
            raise ValueError("Primary model path is required.")
        model = load_streaming_model(model_path)

        max_points = self._get_max_points()
        self._resize_series(max_points)

        subscription_type = self.subscription_var.get().strip() or "l2Book"
        if isinstance(model, FittedMultilevelMicropriceModel):
            if subscription_type != "l2Book":
                raise ValueError("Multilevel models require the l2Book subscription.")
            estimator = "multilevel"
        else:
            estimator = self.estimator_var.get().strip() or "G1"
            if estimator not in model.available_estimators():
                raise ValueError(
                    f"Estimator {estimator} is not available in this model. "
                    f"Available estimators: {', '.join(model.available_estimators())}."
                )

        self._set_adjustment_ticks_visibility(model.tick_size)
        self._set_line_labels(model)

        return model, {
            "coin": coin,
            "url": self.url_var.get().strip() or HYPERLIQUID_MAINNET_WS_URL,
            "subscription_type": subscription_type,
            "estimator": estimator,
            "run_strategy": bool(self.run_strategy_var.get()),
        }

    def _start_stream(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo("Already Running", "The live stream is already running.")
            return
        try:
            model, settings = self._validate_inputs()
        except Exception as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        self.stop_event.clear()
        self._clear_chart()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Connecting...")
        self.worker = threading.Thread(target=self._run_stream, args=(model, settings), daemon=True)
        self.worker.start()

    def _stop_stream(self) -> None:
        self.stop_event.set()
        self.status_var.set("Stopping...")
        self.stop_button.configure(state="disabled")
        if self.active_connection is not None:
            try:
                self.active_connection.close()
            except Exception:
                pass

    def _websocket_factory(self, url: str) -> Any:
        connection = _default_websocket_factory(url)
        self.active_connection = connection
        return connection

    def _run_stream(self, model: StreamingModel, settings: dict[str, Any]) -> None:
        def on_update(payload: dict[str, Any]) -> None:
            if self.stop_event.is_set():
                raise StopIteration
            self.data_queue.put(("update", dict(payload)))

        try:
            stream_hyperliquid_microprice(
                coin=settings["coin"],
                model=model,
                on_update=on_update,
                url=settings["url"],
                subscription_type=settings["subscription_type"],
                estimator=settings["estimator"],
                websocket_factory=self._websocket_factory,
            )
            self.data_queue.put(("status", "Stream ended"))
        except StopIteration:
            self.data_queue.put(("status", "Stopped"))
        except Exception as exc:
            self.data_queue.put(("error", str(exc)))
        finally:
            self.active_connection = None

    def _refresh_plot(self) -> None:
        if not self.x_series:
            self.mid_line.set_data([], [])
            self.primary_micro_line.set_data([], [])
            self.primary_adjustment_line.set_data([], [])
            self.adjustment_ticks_line.set_data([], [])
            self.strategy_pnl_line.set_data([], [])
            self.canvas.draw_idle()
            return

        x_values = list(self.x_series)
        mid_values = list(self.mid_series)
        primary_micro_values = list(self.primary_micro_series)
        primary_adjustment_values = list(self.primary_adjustment_series)
        adjustment_ticks_values = list(self.adjustment_ticks_series)
        strategy_pnl_values = list(self.strategy_pnl_series)

        self.mid_line.set_data(x_values, mid_values)
        self.primary_micro_line.set_data(x_values, primary_micro_values)
        self.primary_adjustment_line.set_data(x_values, primary_adjustment_values)
        if self.show_adjustment_ticks_line:
            self.adjustment_ticks_line.set_data(x_values, adjustment_ticks_values)
        else:
            self.adjustment_ticks_line.set_data([], [])
        self.strategy_pnl_line.set_data(x_values, strategy_pnl_values)

        x_max = x_values[-1] if x_values[-1] > x_values[0] else x_values[0] + 1
        self.axis.set_xlim(x_values[0], x_max)
        self.adjustment_axis.set_xlim(x_values[0], x_max)
        self.strategy_axis.set_xlim(x_values[0], x_max)

        price_arrays = [np.asarray(mid_values, dtype=float), np.asarray(primary_micro_values, dtype=float)]
        finite_prices = np.concatenate([array[np.isfinite(array)] for array in price_arrays if array.size > 0])
        y_min = float(finite_prices.min())
        y_max = float(finite_prices.max())
        y_pad = max(abs(y_min) * 0.001, 1e-6) if y_max <= y_min else (y_max - y_min) * 0.05
        self.axis.set_ylim(y_min - y_pad, y_max + y_pad)

        adjustment_sources = list(primary_adjustment_values)
        if self.show_adjustment_ticks_line:
            adjustment_sources.extend(adjustment_ticks_values)
        finite_adjustments = np.asarray(adjustment_sources, dtype=float)
        finite_adjustments = finite_adjustments[np.isfinite(finite_adjustments)]
        if finite_adjustments.size == 0:
            self.adjustment_axis.set_ylim(-1.0, 1.0)
        else:
            adj_min = float(finite_adjustments.min())
            adj_max = float(finite_adjustments.max())
            adj_pad = max(abs(adj_min) * 0.05, 1e-6) if adj_max <= adj_min else (adj_max - adj_min) * 0.1
            self.adjustment_axis.set_ylim(adj_min - adj_pad, adj_max + adj_pad)

        finite_pnl = np.asarray(strategy_pnl_values, dtype=float)
        finite_pnl = finite_pnl[np.isfinite(finite_pnl)]
        if finite_pnl.size == 0:
            self.strategy_axis.set_ylim(-1.0, 1.0)
        else:
            pnl_min = float(finite_pnl.min())
            pnl_max = float(finite_pnl.max())
            pnl_pad = max(abs(pnl_min) * 0.05, 1e-6) if pnl_max <= pnl_min else (pnl_max - pnl_min) * 0.1
            self.strategy_axis.set_ylim(pnl_min - pnl_pad, pnl_max + pnl_pad)

        self.canvas.draw_idle()

    def _apply_update(self, data: dict[str, Any]) -> None:
        self.point_counter += 1
        self.x_series.append(self.point_counter)
        self.mid_series.append(float(data["midprice"]))
        self.primary_micro_series.append(float(data["microprice"]))
        self.primary_adjustment_series.append(float(data["adjustment"]))
        adjustment_ticks = data.get("adjustment_ticks")
        self.adjustment_ticks_series.append(float("nan") if adjustment_ticks is None else float(adjustment_ticks))
        strategy_snapshot = self._update_strategy_state(
            midprice=float(data["midprice"]),
            adjustment=float(data["adjustment"]),
        )
        self.strategy_pnl_series.append(
            float("nan") if strategy_snapshot is None else float(strategy_snapshot.total_pnl)
        )

        primary_label = "Multilevel" if data.get("model_kind") == "multilevel" else "L1"
        self._set_text_var(
            f"Feed: {self.coin_var.get().strip().upper()} | {self.subscription_var.get().strip()} | {str(data.get('estimator', '-'))}",
            "last_feed_var",
        )
        bid = data.get("bid")
        ask = data.get("ask")
        self._set_text_var(f"Bid / Ask: {_fmt_float(bid)} / {_fmt_float(ask)}", "last_bidask_var")
        if bid is None or ask is None:
            self._set_text_var("Spread: -", "last_spread_var")
        else:
            self._set_text_var(f"Spread: {_fmt_float(float(ask) - float(bid))}", "last_spread_var")
        self._set_text_var(f"Mid: {_fmt_float(data.get('midprice'))}", "last_mid_var")
        self._set_text_var(f"{primary_label} Micro: {_fmt_float(data.get('microprice'))}", "last_primary_micro_var", "last_micro_var")
        self._set_text_var(f"{primary_label} Adj: {_fmt_float(data.get('adjustment'))}", "last_primary_adjustment_var", "last_adjustment_var")
        self._set_text_var(
            "Signal Ticks: -" if adjustment_ticks is None else f"Signal Ticks: {_fmt_float(adjustment_ticks)}",
            "last_adjustment_ticks_var",
        )
        self._set_text_var(f"Imbalance: {_fmt_float(data.get('imbalance'))}", "last_imbalance_var")
        if isinstance(data.get("state"), dict):
            state = data["state"]
            if state.get("state_index") is None:
                self._set_text_var("Model State: inactive", "last_state_var")
            else:
                self._set_text_var(
                    "Model State: "
                    f"idx={state['state_index']} | spread_bucket={state['spread_bucket']} | imb_bucket={state['imbalance_bucket']}",
                    "last_state_var",
                )
        else:
            model_levels = data.get("model_levels")
            decay_lambda = data.get("model_decay_lambda")
            if model_levels is None:
                self._set_text_var("Model State: -", "last_state_var")
            else:
                self._set_text_var(
                    f"Model Params: levels={int(model_levels)} | decay={float(decay_lambda):.4g}",
                    "last_state_var",
                )

        self._update_strategy_labels(strategy_snapshot)
        status_label = str(data.get("estimator", self.estimator_var.get().strip()))
        self.status_var.set(
            f"Streaming {self.coin_var.get().strip().upper()} via {self.subscription_var.get().strip()} ({status_label})"
        )
        self._refresh_plot()

    def _update_strategy_state(self, midprice: float, adjustment: float) -> StrategySnapshot | None:
        if self.strategy_engine is None:
            return None
        return self.strategy_engine.update(
            midprice=midprice,
            adjustment=adjustment,
            enabled=bool(self.run_strategy_var.get()),
        )

    def _update_strategy_labels(self, snapshot: StrategySnapshot | None) -> None:
        if self.strategy_engine is None:
            self.strategy_status_var.set("Strategy: local strategy module not found")
            self.strategy_position_var.set("Position: -")
            self.strategy_pnl_var.set("PnL: -")
            self.strategy_trade_count_var.set("Closed Trades: -")
            return
        if snapshot is None:
            if self.run_strategy_var.get():
                self.strategy_status_var.set("Strategy: armed")
            else:
                self.strategy_status_var.set("Strategy: disabled")
            self.strategy_position_var.set("Position: Flat")
            self.strategy_pnl_var.set("PnL: realized=0.000000 | total=0.000000")
            self.strategy_trade_count_var.set("Closed Trades: 0")
            return

        position_name = "Long" if snapshot.position > 0 else "Short" if snapshot.position < 0 else "Flat"
        self.strategy_status_var.set(f"Strategy: {snapshot.last_action}")
        self.strategy_position_var.set(
            f"Position: {position_name} | hold_left={snapshot.events_remaining} | entry={_fmt_float(snapshot.entry_price)}"
            f" | entry_adj={_fmt_float(snapshot.entry_adjustment)}"
        )
        self.strategy_pnl_var.set(
            f"PnL: realized={snapshot.realized_pnl:.6f} | unrealized={snapshot.unrealized_pnl:.6f} | total={snapshot.total_pnl:.6f}"
        )
        self.strategy_trade_count_var.set(
            f"Closed Trades: {snapshot.trades_closed} | last_exit_event={snapshot.exit_event_index or '-'}"
        )

    def _poll_queue(self) -> None:
        try:
            event_type, payload = self.data_queue.get_nowait()
        except queue.Empty:
            self.root.after(GUI_POLL_MS, self._poll_queue)
            return

        if event_type == "update":
            self._apply_update(payload)
        elif event_type == "status":
            self.status_var.set(str(payload))
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.worker = None
        elif event_type == "error":
            self.status_var.set("Failed")
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.worker = None
            messagebox.showerror("Stream Error", str(payload))

        next_delay = 0 if not self.data_queue.empty() else GUI_POLL_MS
        self.root.after(next_delay, self._poll_queue)


def launch_gui() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    MicropriceGui(root)
    root.mainloop()


def main() -> None:
    launch_gui()


if __name__ == "__main__":
    main()
