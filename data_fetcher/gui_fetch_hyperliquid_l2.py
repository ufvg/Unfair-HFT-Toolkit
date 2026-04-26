from __future__ import annotations

import contextlib
import io
import queue
import threading
import tkinter as tk
from pathlib import Path
import sys
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_fetcher.fetch_hyperliquid_l2 import (
    decompress_archive_object,
    decompressed_output_path,
    fetch_historical_l2,
    parse_yyyymmdd,
)


class QueueWriter(io.TextIOBase):
    def __init__(self, output_queue: queue.Queue[tuple[str, object]]) -> None:
        self.output_queue = output_queue
        self._buffer = ""

    def write(self, s: str) -> int:
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self.output_queue.put(("log", line))
        return len(s)

    def flush(self) -> None:
        if self._buffer:
            self.output_queue.put(("log", self._buffer))
            self._buffer = ""


class FetchGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Hyperliquid Historical L2 Fetcher")
        self.root.geometry("920x680")

        self.output_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None

        self.coin_var = tk.StringVar(value="BTC")
        self.date_from_var = tk.StringVar(value="20260122")
        self.date_to_var = tk.StringVar(value="20260221")
        self.out_dir_var = tk.StringVar(value=str(Path.cwd() / "Hyperliquid_Historical"))
        self.request_payer_var = tk.StringVar(value="requester")
        self.overwrite_var = tk.BooleanVar(value=False)
        self.dry_run_var = tk.BooleanVar(value=False)
        self.auto_decompress_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._build_ui()
        self.root.after(100, self._poll_queue)

    def _set_run_state(self, *, running: bool) -> None:
        self.run_button.configure(state="disabled" if running else "normal")

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Coin").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.coin_var, width=18).grid(row=0, column=1, sticky="ew", pady=4)

        ttk.Label(frame, text="Date From (YYYYMMDD)").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.date_from_var, width=18).grid(row=1, column=1, sticky="ew", pady=4)

        ttk.Label(frame, text="Date To (YYYYMMDD)").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.date_to_var, width=18).grid(row=2, column=1, sticky="ew", pady=4)

        ttk.Label(frame, text="Coverage").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Label(frame, text="Always full day (00:00-23:00)").grid(row=3, column=1, sticky="w", pady=4)

        ttk.Label(frame, text="Output Folder").grid(row=4, column=0, sticky="w", pady=4)
        out_dir_frame = ttk.Frame(frame)
        out_dir_frame.grid(row=4, column=1, sticky="ew", pady=4)
        out_dir_frame.columnconfigure(0, weight=1)
        ttk.Entry(out_dir_frame, textvariable=self.out_dir_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(out_dir_frame, text="Browse", command=self._browse_out_dir).grid(row=0, column=1, padx=(8, 0))

        ttk.Label(frame, text="Request Payer").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.request_payer_var, width=18).grid(row=5, column=1, sticky="ew", pady=4)

        options_frame = ttk.Frame(frame)
        options_frame.grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 8))
        ttk.Checkbutton(options_frame, text="Overwrite Existing Files", variable=self.overwrite_var).pack(
            side="left", padx=(0, 16)
        )
        ttk.Checkbutton(options_frame, text="Dry Run", variable=self.dry_run_var).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(
            options_frame,
            text="Auto-Decompress to .jsonl",
            variable=self.auto_decompress_var,
        ).pack(side="left")

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=7, column=0, columnspan=2, sticky="w", pady=(4, 10))
        self.run_button = ttk.Button(button_frame, text="Start Fetch", command=self._start_fetch)
        self.run_button.pack(side="left")
        ttk.Button(button_frame, text="Clear Log", command=self._clear_log).pack(side="left", padx=(8, 0))

        self.progress_bar = ttk.Progressbar(
            frame,
            orient="horizontal",
            mode="determinate",
            maximum=100,
            variable=self.progress_var,
        )
        self.progress_bar.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        ttk.Label(frame, textvariable=self.status_var).grid(row=9, column=0, columnspan=2, sticky="w", pady=(0, 8))

        self.log_widget = tk.Text(frame, wrap="word", height=24)
        self.log_widget.grid(row=10, column=0, columnspan=2, sticky="nsew")
        frame.rowconfigure(10, weight=1)

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.log_widget.yview)
        scrollbar.grid(row=10, column=2, sticky="ns")
        self.log_widget.configure(yscrollcommand=scrollbar.set)

    def _browse_out_dir(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.out_dir_var.get() or str(Path.cwd()))
        if chosen:
            self.out_dir_var.set(chosen)

    def _clear_log(self) -> None:
        self.log_widget.delete("1.0", "end")

    def _append_log(self, message: str) -> None:
        self.log_widget.insert("end", message + "\n")
        self.log_widget.see("end")

    def _validate_inputs(self) -> dict[str, object]:
        coin = self.coin_var.get().strip().upper()
        if not coin:
            raise ValueError("Coin is required.")

        date_from = parse_yyyymmdd(self.date_from_var.get().strip())
        date_to = parse_yyyymmdd(self.date_to_var.get().strip())
        out_dir = self.out_dir_var.get().strip()
        if not out_dir:
            raise ValueError("Output folder is required.")

        request_payer = self.request_payer_var.get().strip() or "requester"

        return {
            "coin": coin,
            "date_from": date_from,
            "date_to": date_to,
            "out_dir": out_dir,
            "hour_from": 0,
            "hour_to": 23,
            "overwrite": self.overwrite_var.get(),
            "dry_run": self.dry_run_var.get(),
            "request_payer": request_payer,
            "auto_decompress": self.auto_decompress_var.get(),
        }

    def _start_fetch(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo("Fetch Running", "A fetch is already running.")
            return

        try:
            kwargs = self._validate_inputs()
        except Exception as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        self._set_run_state(running=True)
        self.progress_var.set(0)
        self.status_var.set("Fetching...")
        self._append_log(
            "Starting fetch:"
            f" coin={kwargs['coin']}"
            f" date_from={self.date_from_var.get().strip()}"
            f" date_to={self.date_to_var.get().strip()}"
            " coverage=24/7"
            f" auto_decompress={kwargs['auto_decompress']}"
        )
        self.worker = threading.Thread(target=self._run_fetch, args=(kwargs,), daemon=True)
        self.worker.start()

    def _progress_callback(self, payload: dict[str, object]) -> None:
        self.output_queue.put(("progress", payload))

    def _run_fetch(self, kwargs: dict[str, object]) -> None:
        writer = QueueWriter(self.output_queue)
        auto_decompress = bool(kwargs.pop("auto_decompress"))
        try:
            with contextlib.redirect_stdout(writer):
                stats = fetch_historical_l2(**kwargs, progress_callback=self._progress_callback)
            writer.flush()

            decompressed_count = 0
            if auto_decompress and not kwargs["dry_run"]:
                paths = stats.local_paths or []
                existing_paths = [path for path in paths if Path(path).exists()]
                total = len(existing_paths)
                for index, archive_path in enumerate(existing_paths, start=1):
                    output_path = decompress_archive_object(
                        archive_path,
                        overwrite=bool(kwargs["overwrite"]),
                        remove_source=True,
                    )
                    decompressed_count += 1
                    self.output_queue.put(
                        (
                            "decompress_progress",
                            {
                                "current": index,
                                "planned": total,
                                "archive_path": str(archive_path),
                                "output_path": str(output_path),
                            },
                        )
                    )

            summary = (
                f"Fetch summary: planned={stats.planned} downloaded={stats.downloaded} "
                f"skipped={stats.skipped} missing={stats.missing} bytes_downloaded={stats.bytes_downloaded}"
            )
            if auto_decompress and not kwargs["dry_run"]:
                summary += f" decompressed={decompressed_count}"
            self.output_queue.put(("done", summary))
        except Exception as exc:
            writer.flush()
            self.output_queue.put(("error", str(exc)))

    def _poll_queue(self) -> None:
        try:
            while True:
                event_type, payload = self.output_queue.get_nowait()
                if event_type == "log":
                    self._append_log(str(payload))
                elif event_type == "progress":
                    info = payload  # type: ignore[assignment]
                    current = int(info["current"])
                    planned = max(int(info["planned"]), 1)
                    self.progress_var.set(current / planned * 100.0)
                    self.status_var.set(
                        f"Fetching {current}/{planned}: {info['date']} {info['hour']}:00 ({info['status']})"
                    )
                elif event_type == "decompress_progress":
                    info = payload  # type: ignore[assignment]
                    current = int(info["current"])
                    planned = max(int(info["planned"]), 1)
                    self.progress_var.set(current / planned * 100.0)
                    self.status_var.set(f"Decompressing {current}/{planned}")
                    self._append_log(f"Decompressed {info['archive_path']} -> {info['output_path']}")
                elif event_type == "done":
                    self._append_log(str(payload))
                    self.progress_var.set(100.0)
                    self.status_var.set("Completed")
                    self._set_run_state(running=False)
                elif event_type == "error":
                    self._append_log(f"Error: {payload}")
                    self.status_var.set("Failed")
                    self._set_run_state(running=False)
                    messagebox.showerror("Fetch Failed", str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    FetchGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
