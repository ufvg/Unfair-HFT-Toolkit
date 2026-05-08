from __future__ import annotations

import json
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any
from urllib import request

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from vpin import OnlineVPIN, TradePrint

HYPERLIQUID_MAINNET_WS_URL = "wss://api.hyperliquid.xyz/ws"
GUI_POLL_MS = 25
HISTORICAL_ADV_LOOKBACK_DAYS = 7


def _fmt_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _default_websocket_factory(url: str) -> Any:
    try:
        import websocket
    except ImportError as exc:
        raise RuntimeError(
            "Missing websocket client dependency. Install `websocket-client` in the project venv."
        ) from exc

    create_connection = getattr(websocket, "create_connection", None)
    if create_connection is None:
        raise RuntimeError("The installed `websocket` module is not `websocket-client`.")
    return create_connection(url)


def _build_subscription_message(coin: str) -> dict[str, Any]:
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
    lookback_days: int = HISTORICAL_ADV_LOOKBACK_DAYS,
    timeout_seconds: float = 10.0,
) -> float:
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")
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


class VpinGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Realtime VPIN Monitor")
        self.root.geometry("1320x860")
        self.root.minsize(1120, 720)

        self.coin_var = tk.StringVar(value="BTC")
        self.url_var = tk.StringVar(value=HYPERLIQUID_MAINNET_WS_URL)
        self.classification_var = tk.StringVar(value="aggressor")
        self.buckets_per_day_var = tk.StringVar(value="50")
        self.support_buckets_var = tk.StringVar(value="50")
        self.average_daily_volume_var = tk.StringVar()
        self.bucket_volume_var = tk.StringVar()
        self.max_points_var = tk.StringVar(value="500")

        self.status_var = tk.StringVar(
            value="Connect to Hyperliquid trades and stream VPIN in real time."
        )
        self.feed_var = tk.StringVar(value="Feed: -")
        self.trade_var = tk.StringVar(value="Trades: -")
        self.bucket_var = tk.StringVar(value="Buckets: -")
        self.latest_vpin_var = tk.StringVar(value="Latest VPIN: -")
        self.last_price_var = tk.StringVar(value="Last Price: -")
        self.last_side_var = tk.StringVar(value="Last Side: -")
        self.last_time_var = tk.StringVar(value="Last Trade Time: -")
        self.bootstrap_var = tk.StringVar(value="Bootstrap: -")

        self.data_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.active_connection: Any | None = None

        self.x_series: list[int] = []
        self.price_series: list[float] = []
        self.imbalance_series: list[float] = []
        self.vpin_series: list[float] = []
        self.point_counter = 0

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

        feed_frame = ttk.LabelFrame(sidebar, text="Trade Feed", padding=10)
        feed_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        feed_frame.columnconfigure(1, weight=1)
        ttk.Label(feed_frame, text="Coin").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(feed_frame, textvariable=self.coin_var).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(feed_frame, text="Websocket URL").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(feed_frame, textvariable=self.url_var).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Label(feed_frame, text="Chart Points").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(feed_frame, textvariable=self.max_points_var).grid(row=2, column=1, sticky="ew", pady=4)

        param_frame = ttk.LabelFrame(sidebar, text="Online VPIN", padding=10)
        param_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        param_frame.columnconfigure(1, weight=1)
        ttk.Label(param_frame, text="Classification").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Combobox(
            param_frame,
            textvariable=self.classification_var,
            values=("aggressor", "tick"),
            state="readonly",
            width=14,
        ).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(param_frame, text="Buckets / Day").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(param_frame, textvariable=self.buckets_per_day_var).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Label(param_frame, text="Support Buckets").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(param_frame, textvariable=self.support_buckets_var).grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Label(param_frame, text="Average Daily Volume").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(param_frame, textvariable=self.average_daily_volume_var).grid(row=3, column=1, sticky="ew", pady=4)
        ttk.Label(param_frame, text="Manual Bucket Volume").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(param_frame, textvariable=self.bucket_volume_var).grid(row=4, column=1, sticky="ew", pady=4)
        ttk.Label(
            param_frame,
            text=(
                "For realtime monitoring, use `aggressor` because Hyperliquid trades include a true side flag. "
                "If both ADV and manual bucket volume are set, manual bucket volume wins."
            ),
            wraplength=320,
            justify="left",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))

        controls = ttk.Frame(sidebar)
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        controls.columnconfigure((0, 1, 2), weight=1)
        self.start_button = ttk.Button(controls, text="Start Stream", command=self._start_stream)
        self.start_button.grid(row=0, column=0, sticky="ew")
        self.stop_button = ttk.Button(controls, text="Stop", command=self._stop_stream, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(controls, text="Clear Chart", command=self._clear_chart).grid(row=0, column=2, sticky="ew")

        stats_frame = ttk.LabelFrame(sidebar, text="Live Stats", padding=10)
        stats_frame.grid(row=3, column=0, sticky="nsew")
        stats_frame.columnconfigure(0, weight=1)
        for row, variable in enumerate(
            [
                self.status_var,
                self.feed_var,
                self.trade_var,
                self.bucket_var,
                self.latest_vpin_var,
                self.last_price_var,
                self.last_side_var,
                self.last_time_var,
                self.bootstrap_var,
            ]
        ):
            ttk.Label(stats_frame, textvariable=variable, wraplength=320, justify="left").grid(
                row=row, column=0, sticky="w", pady=3
            )

        figure = Figure(figsize=(9.4, 7.2), dpi=100)
        self.price_axis = figure.add_subplot(311)
        self.imbalance_axis = figure.add_subplot(312, sharex=self.price_axis)
        self.vpin_axis = figure.add_subplot(313, sharex=self.price_axis)
        self.figure = figure

        (self.price_line,) = self.price_axis.plot([], [], color="#1f77b4", linewidth=1.6, label="Last Trade Price")
        (self.imbalance_line,) = self.imbalance_axis.plot(
            [], [], color="#ff7f0e", linewidth=1.4, label="Last Bucket Imbalance"
        )
        (self.vpin_line,) = self.vpin_axis.plot([], [], color="#d62728", linewidth=1.6, label="Current VPIN")

        self.price_axis.set_ylabel("Price")
        self.imbalance_axis.set_ylabel("Imbalance")
        self.vpin_axis.set_ylabel("VPIN")
        self.vpin_axis.set_xlabel("Trade Update")
        self.price_axis.legend(loc="upper left")
        self.imbalance_axis.legend(loc="upper left")
        self.vpin_axis.legend(loc="upper left")
        self.figure.tight_layout()

        canvas = FigureCanvasTkAgg(figure, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas = canvas

    def _get_max_points(self) -> int:
        try:
            value = int(self.max_points_var.get().strip())
        except ValueError as exc:
            raise ValueError("Chart Points must be an integer.") from exc
        if value < 50:
            raise ValueError("Chart Points must be at least 50.")
        return value

    def _clear_chart(self) -> None:
        self.x_series = []
        self.price_series = []
        self.imbalance_series = []
        self.vpin_series = []
        self.point_counter = 0
        self.price_line.set_data([], [])
        self.imbalance_line.set_data([], [])
        self.vpin_line.set_data([], [])
        self.canvas.draw_idle()

    def _build_online_settings(self) -> dict[str, Any]:
        classification = self.classification_var.get().strip() or "aggressor"
        if classification not in {"aggressor", "tick"}:
            raise ValueError("Realtime GUI supports aggressor or tick classification.")
        try:
            support_buckets = int(self.support_buckets_var.get().strip())
            buckets_per_day = int(self.buckets_per_day_var.get().strip())
        except ValueError as exc:
            raise ValueError("Support Buckets and Buckets / Day must be integers.") from exc
        if support_buckets <= 0 or buckets_per_day <= 0:
            raise ValueError("Support Buckets and Buckets / Day must be positive.")

        bucket_text = self.bucket_volume_var.get().strip()
        adv_text = self.average_daily_volume_var.get().strip()
        bucket_volume = None
        average_daily_volume = None
        source = "auto_live_rate"
        if bucket_text:
            try:
                bucket_volume = float(bucket_text)
            except ValueError as exc:
                raise ValueError("Manual Bucket Volume must be numeric.") from exc
            source = "manual_bucket_volume"
        elif adv_text:
            try:
                average_daily_volume = float(adv_text)
                bucket_volume = average_daily_volume / buckets_per_day
            except ValueError as exc:
                raise ValueError("Average Daily Volume must be numeric.") from exc
            source = "manual_adv"

        if bucket_volume is not None and (not np.isfinite(bucket_volume) or bucket_volume <= 0.0):
            raise ValueError("Derived bucket volume must be positive and finite.")

        return {
            "classification": classification,
            "support_buckets": support_buckets,
            "buckets_per_day": buckets_per_day,
            "bucket_volume": bucket_volume,
            "average_daily_volume": average_daily_volume,
            "source": source,
        }

    def _start_stream(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo("Already Running", "The realtime VPIN stream is already running.")
            return
        coin = self.coin_var.get().strip().upper()
        if not coin:
            messagebox.showerror("Missing Coin", "Coin is required.")
            return
        try:
            self._get_max_points()
            settings = self._build_online_settings()
        except Exception as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        self._clear_chart()
        self.stop_event.clear()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Connecting to realtime trades...")
        url = self.url_var.get().strip() or HYPERLIQUID_MAINNET_WS_URL
        if settings["bucket_volume"] is None:
            try:
                estimated_adv = _fetch_recent_adv_from_candles(coin=coin, ws_url=url)
                bucket_volume = estimated_adv / int(settings["buckets_per_day"])
                settings["average_daily_volume"] = estimated_adv
                settings["bucket_volume"] = bucket_volume
                settings["source"] = "historical_daily_candles"
                self.bootstrap_var.set(
                    f"Bootstrap: ADV={_fmt_float(estimated_adv, 2)} from last {HISTORICAL_ADV_LOOKBACK_DAYS} daily candles"
                )
                self.status_var.set("Fetched recent historical daily volume and connecting to realtime trades...")
            except Exception as exc:
                self.start_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                messagebox.showerror(
                    "ADV Fetch Failed",
                    "Unable to fetch ADV from recent historical daily bars. "
                    "Set Average Daily Volume or Manual Bucket Volume, or try again.\n\n"
                    f"Details: {exc}",
                )
                return

        self.worker = threading.Thread(
            target=self._run_stream,
            args=(coin, url, settings),
            daemon=True,
        )
        self.worker.start()

    def _stop_stream(self) -> None:
        self.stop_event.set()
        self.status_var.set("Stopping...")
        self.bootstrap_var.set("Bootstrap: stopping and releasing in-memory historical context")
        self.stop_button.configure(state="disabled")
        if self.active_connection is not None:
            try:
                self.active_connection.close()
            except Exception:
                pass

    def _run_stream(self, coin: str, url: str, settings: dict[str, Any]) -> None:
        reconnect_delay_seconds = 1.0
        engine = OnlineVPIN(
            bucket_volume=float(settings["bucket_volume"]),
            support_buckets=int(settings["support_buckets"]),
            classification=str(settings["classification"]),
        )
        while not self.stop_event.is_set():
            try:
                connection = _default_websocket_factory(url)
                self.active_connection = connection
                connection.send(json.dumps(_build_subscription_message(coin)))
                self.data_queue.put(("status", f"Streaming realtime trades for {coin}"))
                while not self.stop_event.is_set():
                    raw_message = connection.recv()
                    if raw_message is None:
                        break
                    try:
                        message = json.loads(raw_message)
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if message.get("channel") != "trades":
                        continue
                    trades = message.get("data")
                    if not isinstance(trades, list) or not trades:
                        continue

                    last_trade: TradePrint | None = None
                    snapshot = None
                    batch_size = 0
                    for item in trades:
                        if not isinstance(item, dict):
                            continue
                        try:
                            trade = TradePrint(
                                timestamp=float(item["time"]),
                                price=float(item["px"]),
                                volume=float(item["sz"]),
                                side=str(item["side"]),
                            )
                        except (KeyError, TypeError, ValueError):
                            continue
                        snapshot = engine.update(trade)
                        last_trade = trade
                        batch_size += 1
                    if snapshot is None or last_trade is None or batch_size == 0:
                        continue
                    self.data_queue.put(
                        (
                            "trade_update",
                            {
                                "coin": coin,
                                "classification": engine.classification,
                                "bucket_volume": float(engine.bucket_volume),
                                "support_buckets": int(engine.support_buckets),
                                "adv_source": str(settings["source"]) if settings["bucket_volume"] is not None else "auto_live_rate",
                                "average_daily_volume": None if settings["average_daily_volume"] is None and settings["bucket_volume"] is not None else settings["average_daily_volume"],
                                "batch_size": batch_size,
                                "price": float(last_trade.price),
                                "side": last_trade.side,
                                "timestamp": last_trade.timestamp.timestamp(),
                                "completed_buckets": int(snapshot.completed_buckets),
                                "current_bucket_fill": float(snapshot.current_bucket_fill),
                                "current_vpin": (
                                    None if snapshot.current_vpin is None else float(snapshot.current_vpin)
                                ),
                                "last_bucket_imbalance": (
                                    None
                                    if snapshot.last_bucket_imbalance is None
                                    else float(snapshot.last_bucket_imbalance)
                                ),
                            },
                        )
                    )
            except Exception as exc:
                if self.stop_event.is_set():
                    break
                self.data_queue.put(("reconnect", str(exc)))
                time.sleep(reconnect_delay_seconds)
            finally:
                if self.active_connection is not None:
                    try:
                        self.active_connection.close()
                    except Exception:
                        pass
                self.active_connection = None

        self.data_queue.put(("stopped", "Stopped"))

    def _apply_trade_update(self, payload: dict[str, Any]) -> None:
        max_points = self._get_max_points()
        self.point_counter += 1
        self.x_series.append(self.point_counter)
        self.price_series.append(float(payload["price"]))
        last_imbalance = payload.get("last_bucket_imbalance")
        current_vpin = payload.get("current_vpin")
        self.imbalance_series.append(float("nan") if last_imbalance is None else float(last_imbalance))
        self.vpin_series.append(float("nan") if current_vpin is None else float(current_vpin))

        if len(self.x_series) > max_points:
            self.x_series = self.x_series[-max_points:]
            self.price_series = self.price_series[-max_points:]
            self.imbalance_series = self.imbalance_series[-max_points:]
            self.vpin_series = self.vpin_series[-max_points:]

        self.feed_var.set(
            f"Feed: {payload['coin']} | trades | mode={payload['classification']} | bucket_vol={_fmt_float(payload['bucket_volume'], 2)}"
        )
        self.trade_var.set(
            f"Trades: updates={self.point_counter} | last_batch={payload['batch_size']} | support={payload['support_buckets']}"
        )
        self.bucket_var.set(
            f"Buckets: completed={payload['completed_buckets']} | fill={_fmt_float(payload['current_bucket_fill'], 3)}"
        )
        self.latest_vpin_var.set(
            "Latest VPIN: warming up"
            if current_vpin is None
            else f"Latest VPIN: {_fmt_float(current_vpin, 6)}"
        )
        self.last_price_var.set(f"Last Price: {_fmt_float(payload['price'], 6)}")
        side = payload.get("side")
        side_text = "Buy" if side == 1 else "Sell" if side == -1 else "-"
        self.last_side_var.set(f"Last Side: {side_text}")
        self.last_time_var.set(f"Last Trade Time: {_fmt_float(payload['timestamp'], 3)}")
        self.status_var.set(f"Streaming realtime VPIN for {payload['coin']}")
        self.bootstrap_var.set(
            "Bootstrap: "
            f"source={payload.get('adv_source', '-')} | ADV={_fmt_float(payload.get('average_daily_volume'), 2)}"
        )

        self.price_line.set_data(self.x_series, self.price_series)
        self.imbalance_line.set_data(self.x_series, self.imbalance_series)
        self.vpin_line.set_data(self.x_series, self.vpin_series)

        if self.x_series:
            x_min = self.x_series[0]
            x_max = self.x_series[-1] if self.x_series[-1] > x_min else x_min + 1
            self.price_axis.set_xlim(x_min, x_max)
            self.imbalance_axis.set_xlim(x_min, x_max)
            self.vpin_axis.set_xlim(x_min, x_max)

        self.price_axis.relim()
        self.price_axis.autoscale_view()

        finite_imbalance = np.asarray(self.imbalance_series, dtype=float)
        finite_imbalance = finite_imbalance[np.isfinite(finite_imbalance)]
        if finite_imbalance.size == 0:
            self.imbalance_axis.set_ylim(0.0, 1.0)
        else:
            self.imbalance_axis.set_ylim(0.0, min(1.0, float(finite_imbalance.max()) + 0.05))

        finite_vpin = np.asarray(self.vpin_series, dtype=float)
        finite_vpin = finite_vpin[np.isfinite(finite_vpin)]
        if finite_vpin.size == 0:
            self.vpin_axis.set_ylim(0.0, 1.0)
        else:
            self.vpin_axis.set_ylim(0.0, min(1.0, float(finite_vpin.max()) + 0.05))

        self.canvas.draw_idle()

    def _poll_queue(self) -> None:
        try:
            event_type, payload = self.data_queue.get_nowait()
        except queue.Empty:
            self.root.after(GUI_POLL_MS, self._poll_queue)
            return

        if event_type == "trade_update":
            self._apply_trade_update(payload)
        elif event_type == "bootstrap_status":
            self.bootstrap_var.set(str(payload))
        elif event_type == "status":
            self.status_var.set(str(payload))
        elif event_type == "reconnect":
            self.status_var.set(f"Disconnected, reconnecting: {payload}")
        elif event_type == "stopped":
            self.status_var.set(str(payload))
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.worker = None

        next_delay = 0 if not self.data_queue.empty() else GUI_POLL_MS
        self.root.after(next_delay, self._poll_queue)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    VpinGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
