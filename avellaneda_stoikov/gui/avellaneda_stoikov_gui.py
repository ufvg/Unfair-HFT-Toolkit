from __future__ import annotations

from collections import deque
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

from avellaneda_stoikov.engine import AvellanedaStoikovEngine
from avellaneda_stoikov.hyperliquid import HyperliquidBookFeed
from avellaneda_stoikov.model import AvellanedaStoikovModel, AvellanedaStoikovParameters, MarketSnapshot
from avellaneda_stoikov.simulator import SimulationConfig, iter_simulation

GUI_POLL_MS = 25


def _fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


class AvellanedaStoikovGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Avellaneda-Stoikov Monitor")
        self.root.geometry("1320x880")
        self.root.minsize(1120, 760)

        self.mode_var = tk.StringVar(value="demo")
        self.coin_var = tk.StringVar(value="BTC")
        self.gamma_var = tk.StringVar(value="0.1")
        self.sigma_var = tk.StringVar(value="0.015")
        self.arrival_scale_var = tk.StringVar(value="1.25")
        self.arrival_decay_var = tk.StringVar(value="4.0")
        self.horizon_var = tk.StringVar(value="30")
        self.tick_size_var = tk.StringVar(value="0.5")
        self.order_size_var = tk.StringVar(value="0.01")
        self.inventory_limit_var = tk.StringVar(value="5")
        self.steps_var = tk.StringVar(value="500")
        self.dt_var = tk.StringVar(value="0.25")
        self.max_points_var = tk.StringVar(value="600")

        self.status_var = tk.StringVar(value="Ready")
        self.quote_var = tk.StringVar(value="Quote: -")
        self.market_var = tk.StringVar(value="Market: -")
        self.inventory_var = tk.StringVar(value="Inventory: -")
        self.pnl_var = tk.StringVar(value="PnL: -")
        self.sigma_live_var = tk.StringVar(value="Sigma: -")
        self.intensity_var = tk.StringVar(value="Intensities: -")
        self.fill_var = tk.StringVar(value="Fills: -")

        self.data_queue: queue.Queue[tuple[str, dict[str, float] | str]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()

        self.x_series: deque[int] = deque(maxlen=600)
        self.mid_series: deque[float] = deque(maxlen=600)
        self.reservation_series: deque[float] = deque(maxlen=600)
        self.bid_series: deque[float] = deque(maxlen=600)
        self.ask_series: deque[float] = deque(maxlen=600)
        self.inventory_series: deque[float] = deque(maxlen=600)
        self.pnl_series: deque[float] = deque(maxlen=600)
        self.counter = 0

        self._build_ui()
        self.root.after(GUI_POLL_MS, self._poll_queue)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=0, minsize=360)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(container)
        sidebar.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        sidebar.columnconfigure(0, weight=1)

        chart_frame = ttk.Frame(container)
        chart_frame.grid(row=0, column=1, sticky="nsew")
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)

        mode_frame = ttk.LabelFrame(sidebar, text="Mode", padding=10)
        mode_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Radiobutton(mode_frame, text="Internal Demo", value="demo", variable=self.mode_var).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(mode_frame, text="Hyperliquid Preview", value="live", variable=self.mode_var).grid(row=1, column=0, sticky="w")
        ttk.Label(
            mode_frame,
            text="Demo simulates fills and PnL. Hyperliquid preview streams live top-of-book and shows hypothetical quotes only.",
            wraplength=320,
            justify="left",
        ).grid(row=2, column=0, sticky="w", pady=(6, 0))

        param_frame = ttk.LabelFrame(sidebar, text="Model Params", padding=10)
        param_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        param_frame.columnconfigure(1, weight=1)
        rows = [
            ("Coin", self.coin_var),
            ("Gamma", self.gamma_var),
            ("Sigma", self.sigma_var),
            ("Arrival A", self.arrival_scale_var),
            ("Arrival k", self.arrival_decay_var),
            ("Horizon Sec", self.horizon_var),
            ("Tick Size", self.tick_size_var),
            ("Order Size", self.order_size_var),
            ("Inventory Limit", self.inventory_limit_var),
            ("Demo Steps", self.steps_var),
            ("dt Seconds", self.dt_var),
            ("Chart Points", self.max_points_var),
        ]
        for row, (label, variable) in enumerate(rows):
            ttk.Label(param_frame, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(param_frame, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=3)

        controls = ttk.Frame(sidebar)
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        controls.columnconfigure((0, 1, 2), weight=1)
        self.start_button = ttk.Button(controls, text="Start", command=self._start)
        self.start_button.grid(row=0, column=0, sticky="ew")
        self.stop_button = ttk.Button(controls, text="Stop", command=self._stop, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(controls, text="Clear", command=self._clear).grid(row=0, column=2, sticky="ew")

        stats = ttk.LabelFrame(sidebar, text="Live Stats", padding=10)
        stats.grid(row=3, column=0, sticky="nsew")
        stats.columnconfigure(0, weight=1)
        for row, variable in enumerate(
            [
                self.status_var,
                self.market_var,
                self.quote_var,
                self.inventory_var,
                self.pnl_var,
                self.sigma_live_var,
                self.intensity_var,
                self.fill_var,
            ]
        ):
            ttk.Label(stats, textvariable=variable, wraplength=320, justify="left").grid(
                row=row, column=0, sticky="w", pady=2
            )

        figure = Figure(figsize=(10.5, 7.8), dpi=100, constrained_layout=True)
        grid = figure.add_gridspec(3, 1, height_ratios=(3.0, 1.2, 1.2))
        self.price_axis = figure.add_subplot(grid[0])
        self.inventory_axis = figure.add_subplot(grid[1], sharex=self.price_axis)
        self.pnl_axis = figure.add_subplot(grid[2], sharex=self.price_axis)
        (self.mid_line,) = self.price_axis.plot([], [], color="#1f77b4", linewidth=1.4, label="Mid", zorder=2)
        (self.reservation_line,) = self.price_axis.plot(
            [], [], color="#2ca02c", linewidth=1.4, linestyle="--", label="Reservation", zorder=3
        )
        (self.bid_line,) = self.price_axis.plot(
            [], [], color="#ff7f0e", linewidth=2.1, label="Bid", zorder=5
        )
        (self.ask_line,) = self.price_axis.plot(
            [], [], color="#d62728", linewidth=2.1, label="Ask", zorder=5
        )
        self.quote_band = None
        (self.inventory_line,) = self.inventory_axis.plot([], [], color="#9467bd", linewidth=1.5, label="Inventory")
        (self.pnl_line,) = self.pnl_axis.plot([], [], color="#8c564b", linewidth=1.5, label="PnL")
        self.price_axis.set_title("Hypothetical Avellaneda-Stoikov Quotes")
        self.price_axis.set_ylabel("Price")
        self.inventory_axis.set_ylabel("Inventory")
        self.pnl_axis.set_ylabel("PnL")
        self.pnl_axis.set_xlabel("Update")
        price_formatter = ScalarFormatter(useOffset=False)
        price_formatter.set_scientific(False)
        self.price_axis.yaxis.set_major_formatter(price_formatter)
        self.price_axis.grid(True, alpha=0.2)
        self.inventory_axis.grid(True, alpha=0.2)
        self.pnl_axis.grid(True, alpha=0.2)
        self.price_axis.legend(loc="upper left")
        self.inventory_axis.legend(loc="upper left")
        self.pnl_axis.legend(loc="upper left")

        self.canvas = FigureCanvasTkAgg(figure, master=chart_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _build_params(self) -> tuple[AvellanedaStoikovParameters, int, float, int]:
        try:
            params = AvellanedaStoikovParameters(
                gamma=float(self.gamma_var.get()),
                sigma=float(self.sigma_var.get()),
                arrival_rate_scale=float(self.arrival_scale_var.get()),
                arrival_rate_decay=float(self.arrival_decay_var.get()),
                horizon_seconds=float(self.horizon_var.get()),
                tick_size=float(self.tick_size_var.get()),
                order_size=float(self.order_size_var.get()),
                inventory_limit=float(self.inventory_limit_var.get()),
            )
            steps = int(self.steps_var.get())
            dt = float(self.dt_var.get())
            max_points = int(self.max_points_var.get())
        except ValueError as exc:
            raise ValueError("All parameters must be numeric and valid.") from exc
        if steps < 10:
            raise ValueError("Demo Steps must be at least 10.")
        if dt <= 0.0:
            raise ValueError("dt Seconds must be positive.")
        if max_points < 50:
            raise ValueError("Chart Points must be at least 50.")
        return params, steps, dt, max_points

    def _resize_series(self, max_points: int) -> None:
        self.x_series = deque(self.x_series, maxlen=max_points)
        self.mid_series = deque(self.mid_series, maxlen=max_points)
        self.reservation_series = deque(self.reservation_series, maxlen=max_points)
        self.bid_series = deque(self.bid_series, maxlen=max_points)
        self.ask_series = deque(self.ask_series, maxlen=max_points)
        self.inventory_series = deque(self.inventory_series, maxlen=max_points)
        self.pnl_series = deque(self.pnl_series, maxlen=max_points)

    def _start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo("Already Running", "A session is already running.")
            return
        try:
            params, steps, dt, max_points = self._build_params()
        except Exception as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return
        self._resize_series(max_points)
        self._clear()
        self.stop_event.clear()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Starting...")
        if self.mode_var.get() == "demo":
            self.worker = threading.Thread(target=self._run_demo, args=(params, steps, dt), daemon=True)
        else:
            self.worker = threading.Thread(target=self._run_live_preview, args=(params, dt), daemon=True)
        self.worker.start()

    def _stop(self) -> None:
        self.stop_event.set()
        self.status_var.set("Stopping...")
        self.stop_button.configure(state="disabled")

    def _clear(self) -> None:
        self.x_series.clear()
        self.mid_series.clear()
        self.reservation_series.clear()
        self.bid_series.clear()
        self.ask_series.clear()
        self.inventory_series.clear()
        self.pnl_series.clear()
        self.counter = 0
        self._refresh_plot()

    def _run_demo(self, params: AvellanedaStoikovParameters, steps: int, dt: float) -> None:
        config = SimulationConfig(steps=steps, dt_seconds=dt, sigma=params.sigma, initial_midprice=100000.0)
        for step in iter_simulation(params, config):
            if self.stop_event.is_set():
                break
            self.data_queue.put(
                (
                    "step",
                    {
                        "timestamp": step.snapshot.timestamp,
                        "mid": step.snapshot.midprice,
                        "reservation": step.reservation_price,
                        "bid": step.bid_price,
                        "ask": step.ask_price,
                        "inventory": step.inventory,
                        "pnl": step.pnl,
                        "sigma": step.sigma,
                        "bid_fill": 1.0 if step.bid_fill else 0.0,
                        "ask_fill": 1.0 if step.ask_fill else 0.0,
                        "bid_intensity": 0.0,
                        "ask_intensity": 0.0,
                        "mode": "demo",
                    },
                )
            )
            time.sleep(min(dt, 0.2))
        self.data_queue.put(("status", "Stopped" if self.stop_event.is_set() else "Demo completed"))

    def _run_live_preview(self, params: AvellanedaStoikovParameters, dt: float) -> None:
        engine = AvellanedaStoikovEngine(AvellanedaStoikovModel(params))
        feed = HyperliquidBookFeed(coin=self.coin_var.get().strip().upper())

        def on_snapshot(snapshot: MarketSnapshot) -> None:
            if self.stop_event.is_set():
                return
            engine_snapshot = engine.on_market_snapshot(snapshot, dt_seconds=dt)
            quote = engine_snapshot.quote
            self.data_queue.put(
                (
                    "step",
                    {
                        "timestamp": snapshot.timestamp,
                        "mid": snapshot.midprice,
                        "reservation": quote.reservation_price,
                        "bid": quote.bid_price,
                        "ask": quote.ask_price,
                        "inventory": engine.state.inventory,
                        "pnl": engine.state.mark_value(snapshot.midprice),
                        "sigma": engine_snapshot.sigma,
                        "bid_fill": 0.0,
                        "ask_fill": 0.0,
                        "bid_intensity": quote.bid_intensity,
                        "ask_intensity": quote.ask_intensity,
                        "mode": "live",
                    },
                )
            )

        try:
            self.status_var.set("Connecting to Hyperliquid...")
            feed.stream(on_snapshot, stop_condition=self.stop_event.is_set)
            self.data_queue.put(("status", "Stopped"))
        except Exception as exc:
            self.data_queue.put(("status", f"Live preview failed: {exc}"))

    def _apply_step(self, payload: dict[str, float]) -> None:
        self.counter += 1
        self.x_series.append(self.counter)
        self.mid_series.append(float(payload["mid"]))
        self.reservation_series.append(float(payload["reservation"]))
        self.bid_series.append(float(payload["bid"]))
        self.ask_series.append(float(payload["ask"]))
        self.inventory_series.append(float(payload["inventory"]))
        self.pnl_series.append(float(payload["pnl"]))

        self.market_var.set(
            f"Market: mid={_fmt(payload['mid'], 4)} | reservation={_fmt(payload['reservation'], 4)}"
        )
        self.quote_var.set(f"Quote: bid={_fmt(payload['bid'], 4)} | ask={_fmt(payload['ask'], 4)}")
        self.inventory_var.set(f"Inventory: {_fmt(payload['inventory'], 4)}")
        self.pnl_var.set(f"PnL: {_fmt(payload['pnl'], 4)}")
        self.sigma_live_var.set(f"Sigma: {_fmt(payload['sigma'], 6)}")
        self.intensity_var.set(
            f"Intensities: bid={_fmt(payload['bid_intensity'], 4)} | ask={_fmt(payload['ask_intensity'], 4)}"
        )
        self.fill_var.set(
            f"Fills: bid={'yes' if payload['bid_fill'] else 'no'} | ask={'yes' if payload['ask_fill'] else 'no'}"
        )
        mode = str(payload.get("mode", "demo"))
        self.status_var.set("Streaming live theoretical quotes" if mode == "live" else "Running internal demo")
        self._refresh_plot()

    def _refresh_plot(self) -> None:
        x = list(self.x_series)
        bid_values = list(self.bid_series)
        ask_values = list(self.ask_series)
        self.mid_line.set_data(x, list(self.mid_series))
        self.reservation_line.set_data(x, list(self.reservation_series))
        self.bid_line.set_data(x, bid_values)
        self.ask_line.set_data(x, ask_values)
        self.inventory_line.set_data(x, list(self.inventory_series))
        self.pnl_line.set_data(x, list(self.pnl_series))

        if self.quote_band is not None:
            self.quote_band.remove()
            self.quote_band = None
        if x and bid_values and ask_values:
            self.quote_band = self.price_axis.fill_between(
                x,
                bid_values,
                ask_values,
                color="#ffbb78",
                alpha=0.2,
                label="_nolegend_",
                zorder=1,
            )

        for axis in (self.price_axis, self.inventory_axis, self.pnl_axis):
            axis.relim()
            axis.autoscale_view()
        self.canvas.draw_idle()

    def _poll_queue(self) -> None:
        try:
            event_type, payload = self.data_queue.get_nowait()
        except queue.Empty:
            self.root.after(GUI_POLL_MS, self._poll_queue)
            return

        if event_type == "step":
            self._apply_step(payload)  # type: ignore[arg-type]
        elif event_type == "status":
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
    AvellanedaStoikovGui(root)
    root.mainloop()


def main() -> None:
    launch_gui()


if __name__ == "__main__":
    main()
