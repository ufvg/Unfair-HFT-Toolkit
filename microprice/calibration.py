from __future__ import annotations

import argparse
import json
import math
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lz4.frame
import numpy as np
import pandas as pd

from .validation import _validate_finite as _validate_finite_scalar
from .validation import _validate_nonnegative as _validate_nonnegative_scalar

REQUIRED_COLUMNS = ("time", "bid", "ask", "bs", "as")
L2_TENSOR_WIDTH = 81
TIMESTAMP_COLUMN = 0
BEST_BID_PRICE_COLUMN = 1
BEST_BID_SIZE_COLUMN = 2
BEST_ASK_PRICE_COLUMN = 41
BEST_ASK_SIZE_COLUMN = 42
DEFAULT_THRESHOLD_PERCENTILES = (80, 90, 95)


def _as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def raw_l1_microprice(
    bid: float,
    ask: float,
    bid_size: float,
    ask_size: float,
) -> tuple[float, float]:
    bid_price = _validate_finite_scalar(bid, "bid")
    ask_price = _validate_finite_scalar(ask, "ask")
    bid_qty = _validate_nonnegative_scalar(bid_size, "bid_size")
    ask_qty = _validate_nonnegative_scalar(ask_size, "ask_size")
    if bid_price > ask_price:
        raise ValueError("best bid must not exceed best ask.")
    mid = 0.5 * (bid_price + ask_price)
    denominator = bid_qty + ask_qty
    if denominator <= 0.0:
        return mid, mid
    return mid, float((ask_price * bid_qty + bid_price * ask_qty) / denominator)


@dataclass(frozen=True, slots=True)
class L1Microprice:
    """Raw top-of-book cross-side weighted microprice."""

    def __call__(
        self,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
    ) -> float:
        return self.compute(bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size)

    def compute(
        self,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
    ) -> float:
        _, microprice = raw_l1_microprice(bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size)
        return microprice


@dataclass
class FittedMicropriceModel:
    tick_size: float
    n_imb: int
    n_spread: int
    dt: int
    imbalance_edges: np.ndarray
    G1: np.ndarray
    B: np.ndarray
    G_star: np.ndarray
    Q: np.ndarray
    Q2: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    move_values: np.ndarray
    finite_horizons: dict[str, Any] | None = None
    G6: np.ndarray | None = None

    def __post_init__(self) -> None:
        finite_horizons_provided = self.finite_horizons is not None
        legacy_g6 = None if self.G6 is None else _as_float_array(self.G6)
        self.tick_size = float(self.tick_size)
        self.n_imb = int(self.n_imb)
        self.n_spread = int(self.n_spread)
        self.dt = int(self.dt)
        self.imbalance_edges = _as_float_array(self.imbalance_edges)
        self.G1 = _as_float_array(self.G1)
        self.B = _as_float_array(self.B)
        self.G_star = _as_float_array(self.G_star)
        self.Q = _as_float_array(self.Q)
        self.Q2 = _as_float_array(self.Q2)
        self.R1 = _as_float_array(self.R1)
        self.R2 = _as_float_array(self.R2)
        self.move_values = _as_float_array(self.move_values)
        if not np.isfinite(self.tick_size) or self.tick_size <= 0.0:
            raise ValueError("tick_size must be positive and finite.")
        if self.n_imb <= 0:
            raise ValueError("n_imb must be positive.")
        if self.n_spread <= 0:
            raise ValueError("n_spread must be positive.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        finite_horizons: dict[int, np.ndarray] = {}
        if self.finite_horizons is not None:
            for key, values in self.finite_horizons.items():
                horizon = int(key)
                if horizon <= 0:
                    raise ValueError("Finite horizons must be positive integers.")
                finite_horizons[horizon] = _as_float_array(values)
        finite_horizons.setdefault(1, self.G1)
        if legacy_g6 is not None:
            finite_horizons.setdefault(6, legacy_g6)
        target_horizon = max(finite_horizons)
        if not finite_horizons_provided:
            target_horizon = max(target_horizon, 6)
        for horizon in range(1, target_horizon + 1):
            if horizon not in finite_horizons:
                finite_horizons[horizon] = _propagate_adjustment_steps(self.G1, self.B, steps=horizon)
        self.finite_horizons = {str(horizon): finite_horizons[horizon] for horizon in sorted(finite_horizons)}
        self.G1 = self.finite_horizons["1"]
        self.G6 = self.finite_horizons.get("6")

    def _midprice(self, bid: float, ask: float) -> float:
        bid_price = _validate_finite_scalar(bid, "bid")
        ask_price = _validate_finite_scalar(ask, "ask")
        return (bid_price + ask_price) / 2.0

    def _imbalance(self, bid_size: float, ask_size: float) -> float:
        bid_qty = _validate_nonnegative_scalar(bid_size, "bid_size")
        ask_qty = _validate_nonnegative_scalar(ask_size, "ask_size")
        total = bid_qty + ask_qty
        if total <= 0:
            return 0.5
        return bid_qty / total

    def _spread_ticks(self, bid: float, ask: float) -> int | None:
        bid_price = _validate_finite_scalar(bid, "bid")
        ask_price = _validate_finite_scalar(ask, "ask")
        spread = ask_price - bid_price
        if spread <= 0 or self.tick_size <= 0:
            return None
        ticks = int(np.rint(spread / self.tick_size))
        if ticks < 1:
            return None
        return min(max(ticks, 1), self.n_spread)

    def _imbalance_bucket(self, imbalance: float) -> int:
        clipped = min(max(float(imbalance), 0.0), 1.0)
        bucket = int(np.searchsorted(self.imbalance_edges[1:-1], clipped, side="right"))
        return min(max(bucket, 0), self.n_imb - 1)

    def state_from_l1(
        self,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
    ) -> dict[str, Any]:
        if float(bid) > float(ask):
            raise ValueError("best bid must not exceed best ask.")
        imbalance = self._imbalance(bid_size, ask_size)
        spread_ticks = self._spread_ticks(bid, ask)
        if spread_ticks is None:
            return {
                "imbalance": imbalance,
                "imbalance_bucket": self._imbalance_bucket(imbalance),
                "spread_ticks": None,
                "spread_bucket": None,
                "state_index": None,
            }
        imbalance_bucket = self._imbalance_bucket(imbalance)
        spread_bucket = spread_ticks - 1
        state_index = spread_bucket * self.n_imb + imbalance_bucket
        return {
            "imbalance": imbalance,
            "imbalance_bucket": imbalance_bucket,
            "spread_ticks": spread_ticks,
            "spread_bucket": spread_bucket,
            "state_index": state_index,
        }

    def adjustment_from_l1(
        self,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
        estimator: str | int = "G_star",
    ) -> float:
        state = self.state_from_l1(bid, ask, bid_size, ask_size)
        if state["state_index"] is None:
            return 0.0
        adjustment_vector = self._adjustment_vector(estimator)
        return float(adjustment_vector[state["state_index"]])

    def raw_microprice_from_l1(
        self,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
    ) -> float:
        return L1Microprice().compute(bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size)

    def microprice_from_l1(
        self,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
        estimator: str | int = "G_star",
    ) -> float:
        return self._midprice(bid, ask) + self.adjustment_from_l1(
            bid,
            ask,
            bid_size,
            ask_size,
            estimator=estimator,
        )

    def raw_microprice_from_book(
        self,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
    ) -> float:
        return self.raw_microprice_from_l1(bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size)

    def adjustment_from_book(
        self,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
        estimator: str | int = "G_star",
    ) -> float:
        return self.adjustment_from_l1(
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            estimator=estimator,
        )

    def microprice_from_book(
        self,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
        estimator: str | int = "G_star",
    ) -> float:
        return self.microprice_from_l1(
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            estimator=estimator,
        )

    def available_horizons(self) -> tuple[int, ...]:
        return tuple(sorted(int(key) for key in self.finite_horizons))

    def available_estimators(self) -> tuple[str, ...]:
        return tuple([f"G{horizon}" for horizon in self.available_horizons()] + ["G_star"])

    def _adjustment_vector(self, estimator: str | int) -> np.ndarray:
        if estimator == "G_star":
            return self.G_star
        if isinstance(estimator, int):
            horizon_key = str(int(estimator))
        else:
            match = re.fullmatch(r"G(\d+)", str(estimator))
            if match is None:
                raise ValueError(
                    f"Unsupported estimator '{estimator}'. Expected 'G<k>' for finite horizons or 'G_star'."
                )
            horizon_key = match.group(1)
        adjustment_vector = self.finite_horizons.get(horizon_key)
        if adjustment_vector is None:
            raise ValueError(
                f"Unsupported estimator '{estimator}'. Available finite horizons are {self.available_horizons()} and G_star."
            )
        return adjustment_vector

    def adjustment_vector(self, estimator: str | int = "G_star") -> np.ndarray:
        return np.array(self._adjustment_vector(estimator), copy=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_kind": "l1",
            "tick_size": self.tick_size,
            "calibration_units": "ticks",
            "calibration_horizon": self.dt,
            "n_imb": self.n_imb,
            "n_spread": self.n_spread,
            "dt": self.dt,
            "imbalance_edges": self.imbalance_edges.tolist(),
            "G1": self.G1.tolist(),
            "B": self.B.tolist(),
            "G6": None if self.G6 is None else self.G6.tolist(),
            "G_star": self.G_star.tolist(),
            "Q": self.Q.tolist(),
            "Q2": self.Q2.tolist(),
            "R1": self.R1.tolist(),
            "R2": self.R2.tolist(),
            "move_values": self.move_values.tolist(),
            "finite_horizons": {
                horizon: values.tolist() for horizon, values in self.finite_horizons.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FittedMicropriceModel":
        normalized = dict(payload)
        normalized.pop("model_kind", None)
        normalized.pop("calibration_units", None)
        normalized.pop("calibration_horizon", None)
        return cls(**normalized)

    def save_model(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def save_model(model: FittedMicropriceModel, path: str | Path) -> None:
    model.save_model(path)


def load_model(path: str | Path) -> FittedMicropriceModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return FittedMicropriceModel.from_dict(payload)


def _extract_date_from_path(path: Path) -> str | None:
    match = re.search(r"(\d{8})", path.stem)
    if match is None:
        return None
    return match.group(1)


def _date_in_range(date_str: str | None, date_from: str | None, date_to: str | None) -> bool:
    if date_str is None:
        return True
    if date_from is not None and date_str < date_from:
        return False
    if date_to is not None and date_str > date_to:
        return False
    return True


def _iter_l2_tensor_paths(
    raw_dir: str | Path,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[Path]:
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"Raw tensor directory does not exist: {root}")
    paths = []
    for path in sorted(root.rglob("*.npy")):
        if _date_in_range(_extract_date_from_path(path), date_from, date_to):
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No .npy tensors found in {root} for the requested date range.")
    return paths


def _iter_hyperliquid_jsonl_paths(
    raw_dir: str | Path,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[Path]:
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"Raw archive directory does not exist: {root}")
    paths = []
    for path in sorted(root.rglob("*.jsonl")):
        parent_date = path.parent.name if re.fullmatch(r"\d{8}", path.parent.name) else None
        date_str = parent_date or _extract_date_from_path(path)
        if _date_in_range(date_str, date_from, date_to):
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No .jsonl files found in {root} for the requested date range.")
    return paths


def _iter_npz_paths(
    raw_dir: str | Path,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[Path]:
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {root}")
    paths = []
    for path in sorted(root.rglob("*.npz")):
        parent_date = path.parent.name if re.fullmatch(r"\d{8}", path.parent.name) else None
        date_str = parent_date or _extract_date_from_path(path)
        if _date_in_range(date_str, date_from, date_to):
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {root} for the requested date range.")
    return paths


def _iter_csv_paths(
    raw_dir: str | Path,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[Path]:
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {root}")
    paths = []
    for path in sorted(root.rglob("*.csv")):
        parent_date = path.parent.name if re.fullmatch(r"\d{8}", path.parent.name) else None
        date_str = parent_date or _extract_date_from_path(path)
        if _date_in_range(date_str, date_from, date_to):
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No .csv files found in {root} for the requested date range.")
    return paths


def _ensure_parquet_engine() -> None:
    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Parquet support requires `pyarrow`. Install it with `pip install pyarrow`."
        ) from exc


def _print_file_progress(label: str, index: int, total: int, path: Path) -> None:
    print(f"{label} {index}/{total}: {path}")


def _print_stage(label: str) -> None:
    print(f"[stage] {label}")


def extract_l1_from_l2_tensor(path: str | Path) -> pd.DataFrame:
    tensor_path = Path(path)
    tensor = np.load(tensor_path, mmap_mode="r")
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D tensor in {tensor_path}, got shape {tensor.shape}.")
    if tensor.shape[1] != L2_TENSOR_WIDTH:
        raise ValueError(
            f"Unexpected tensor width in {tensor_path}: expected {L2_TENSOR_WIDTH}, got {tensor.shape[1]}."
        )

    frame = pd.DataFrame(
        {
            "time": tensor[:, TIMESTAMP_COLUMN],
            "bid": tensor[:, BEST_BID_PRICE_COLUMN],
            "bs": tensor[:, BEST_BID_SIZE_COLUMN],
            "ask": tensor[:, BEST_ASK_PRICE_COLUMN],
            "as": tensor[:, BEST_ASK_SIZE_COLUMN],
        }
    )
    date_str = _extract_date_from_path(tensor_path)
    if date_str is not None:
        frame["date"] = date_str
    return frame


def _extract_l1_record_from_message(message: dict[str, Any]) -> dict[str, Any] | None:
    raw = message.get("raw")
    if not isinstance(raw, dict):
        return None
    if raw.get("channel") != "l2Book":
        return None
    data = raw.get("data")
    if not isinstance(data, dict):
        return None
    levels = data.get("levels")
    if not isinstance(levels, list) or len(levels) < 2:
        return None
    bids = levels[0] or []
    asks = levels[1] or []
    if not bids or not asks:
        return None
    best_bid = bids[0]
    best_ask = asks[0]
    return {
        "time": float(data["time"]),
        "bid": float(best_bid["px"]),
        "bs": float(best_bid["sz"]),
        "ask": float(best_ask["px"]),
        "as": float(best_ask["sz"]),
    }


def extract_l1_from_hyperliquid_jsonl(path: str | Path) -> pd.DataFrame:
    jsonl_path = Path(path)
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                message = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {jsonl_path} at line {line_number}.") from exc
            record = _extract_l1_record_from_message(message)
            if record is not None:
                rows.append(record)

    if not rows:
        raise ValueError(f"No valid l2Book records found in {jsonl_path}.")

    frame = pd.DataFrame(rows)
    parent_date = jsonl_path.parent.name if re.fullmatch(r"\d{8}", jsonl_path.parent.name) else None
    date_str = parent_date or _extract_date_from_path(jsonl_path)
    if date_str is not None:
        frame["date"] = date_str
    return frame


def extract_l1_from_hyperliquid_lz4(path: str | Path) -> pd.DataFrame:
    archive_path = Path(path)
    try:
        decompressed = lz4.frame.decompress(archive_path.read_bytes()).decode("utf-8")
    except Exception as exc:
        raise ValueError(f"Unable to decompress Hyperliquid archive file {archive_path}.") from exc

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(decompressed.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            message = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {archive_path} at line {line_number}.") from exc
        record = _extract_l1_record_from_message(message)
        if record is not None:
            rows.append(record)

    if not rows:
        raise ValueError(f"No valid l2Book records found in {archive_path}.")

    frame = pd.DataFrame(rows)
    parent_date = archive_path.parent.name if re.fullmatch(r"\d{8}", archive_path.parent.name) else None
    date_str = parent_date or _extract_date_from_path(archive_path)
    if date_str is not None:
        frame["date"] = date_str
    return frame


def extract_l1_from_npz(path: str | Path) -> pd.DataFrame:
    npz_path = Path(path)
    with np.load(npz_path) as archive:
        if not archive.files:
            raise ValueError(f"No arrays found in {npz_path}.")
        tensor = archive[archive.files[0]]

    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D array in {npz_path}, got shape {tensor.shape}.")
    if tensor.shape[1] != L2_TENSOR_WIDTH:
        raise ValueError(
            f"Unexpected tensor width in {npz_path}: expected {L2_TENSOR_WIDTH}, got {tensor.shape[1]}."
        )

    frame = pd.DataFrame(
        {
            "time": tensor[:, TIMESTAMP_COLUMN],
            "bid": tensor[:, BEST_BID_PRICE_COLUMN],
            "bs": tensor[:, BEST_BID_SIZE_COLUMN],
            "ask": tensor[:, BEST_ASK_PRICE_COLUMN],
            "as": tensor[:, BEST_ASK_SIZE_COLUMN],
        }
    )
    parent_date = npz_path.parent.name if re.fullmatch(r"\d{8}", npz_path.parent.name) else None
    date_str = parent_date or _extract_date_from_path(npz_path)
    if date_str is not None:
        frame["date"] = date_str
    return frame


def extract_l1_from_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    frame = _normalize_input(frame)
    parent_date = csv_path.parent.name if re.fullmatch(r"\d{8}", csv_path.parent.name) else None
    date_str = parent_date or _extract_date_from_path(csv_path)
    if date_str is not None:
        frame["date"] = date_str
    return frame


def _detect_raw_source_type(raw_dir: str | Path) -> str:
    root = Path(raw_dir)
    if any(root.rglob("*.jsonl")):
        return "jsonl"
    if any(root.rglob("*.npy")):
        return "tensor"
    if any(root.rglob("*.npz")):
        return "npz"
    if any(root.rglob("*.csv")):
        return "csv"
    raise FileNotFoundError(f"No supported raw files found in {root}. Expected .jsonl, .npy, .npz, or .csv.")


def build_l1_cache(
    raw_dir: str | Path,
    parquet_dir: str | Path,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[Path]:
    _print_stage("building parquet L1 cache")
    _ensure_parquet_engine()
    output_dir = Path(parquet_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []

    source_type = _detect_raw_source_type(raw_dir)
    if source_type == "tensor":
        source_paths = _iter_l2_tensor_paths(raw_dir, date_from=date_from, date_to=date_to)
        extractor = extract_l1_from_l2_tensor
    elif source_type == "jsonl":
        source_paths = _iter_hyperliquid_jsonl_paths(raw_dir, date_from=date_from, date_to=date_to)
        extractor = extract_l1_from_hyperliquid_jsonl
    elif source_type == "npz":
        source_paths = _iter_npz_paths(raw_dir, date_from=date_from, date_to=date_to)
        extractor = extract_l1_from_npz
    elif source_type == "csv":
        source_paths = _iter_csv_paths(raw_dir, date_from=date_from, date_to=date_to)
        extractor = extract_l1_from_csv
    else:
        raise ValueError(f"Unsupported raw source type: {source_type}")

    total = len(source_paths)
    for index, source_path in enumerate(source_paths, start=1):
        _print_file_progress("Caching", index, total, source_path)
        frame = extractor(source_path)
        target_path = output_dir / f"{source_path.stem}.parquet"
        frame.to_parquet(target_path, index=False)
        written_paths.append(target_path)

    return written_paths


def load_l1_cache(
    parquet_dir: str | Path,
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    _print_stage("loading parquet cache")
    _ensure_parquet_engine()
    root = Path(parquet_dir)
    if not root.exists():
        raise FileNotFoundError(f"Parquet cache directory does not exist: {root}")

    parquet_paths = [
        path
        for path in sorted(root.rglob("*.parquet"))
        if _date_in_range(_extract_date_from_path(path), date_from, date_to)
    ]
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found in {root} for the requested date range.")

    total = len(parquet_paths)
    frames = []
    for index, path in enumerate(parquet_paths, start=1):
        _print_file_progress("Loading parquet", index, total, path)
        frames.append(pd.read_parquet(path))
    combined = pd.concat(frames, ignore_index=True)
    return combined


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _normalize_input(
    df: pd.DataFrame,
    drop_invalid_rows: bool = True,
) -> pd.DataFrame:
    _validate_columns(df)
    frame = df.loc[:, REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().sort_values("time", kind="mergesort").reset_index(drop=True)
    if frame.empty:
        raise ValueError("Input dataframe has no valid rows after numeric coercion.")
    valid = (
        np.isfinite(frame["bid"].to_numpy(dtype=float))
        & np.isfinite(frame["ask"].to_numpy(dtype=float))
        & np.isfinite(frame["bs"].to_numpy(dtype=float))
        & np.isfinite(frame["as"].to_numpy(dtype=float))
        & (frame["bs"].to_numpy(dtype=float) >= 0.0)
        & (frame["as"].to_numpy(dtype=float) >= 0.0)
        & (frame["bid"].to_numpy(dtype=float) <= frame["ask"].to_numpy(dtype=float))
    )
    if not drop_invalid_rows and not bool(valid.all()):
        raise ValueError("Input dataframe contains invalid L1 rows.")
    frame = frame.loc[valid].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid L1 rows remain after filtering invalid states.")
    return frame


def _infer_tick_size(spread: pd.Series) -> float:
    positive_spreads = spread[spread > 0]
    if positive_spreads.empty:
        raise ValueError("Unable to infer tick size: no positive spreads found.")
    return float(positive_spreads.min())


def _resolve_tick_size(frame: pd.DataFrame, tick_size: float | None = None) -> float:
    if tick_size is not None:
        resolved_tick_size = float(tick_size)
        if not np.isfinite(resolved_tick_size) or resolved_tick_size <= 0.0:
            raise ValueError("tick_size must be positive and finite.")
        return resolved_tick_size
    return _infer_tick_size((frame["ask"] - frame["bid"]).astype(float))


def _smallest_positive_change(series: pd.Series) -> float | None:
    diffs = series.astype(float).diff().abs()
    positive = diffs[diffs > 1e-9]
    if positive.empty:
        return None
    return float(positive.min())


def summarize_l1_diagnostics(df: pd.DataFrame, n_spread: int | None = None) -> dict[str, Any]:
    frame = _normalize_input(df)
    spread = (frame["ask"] - frame["bid"]).astype(float)
    positive_spread = spread[spread > 0]
    if positive_spread.empty:
        raise ValueError("Unable to diagnose L1 data: no positive spreads found.")

    inferred_tick_size = _infer_tick_size(spread)
    bid_change = _smallest_positive_change(frame["bid"])
    ask_change = _smallest_positive_change(frame["ask"])
    retained_fraction = None
    if n_spread is not None:
        spread_ticks = np.rint(spread / inferred_tick_size)
        retained_fraction = float(((spread_ticks >= 1) & (spread_ticks <= n_spread)).mean())

    top_spreads = positive_spread.value_counts().head(5)
    warnings: list[str] = []
    finest_quote_change = None
    positive_quote_changes = [value for value in (bid_change, ask_change) if value is not None]
    if positive_quote_changes:
        finest_quote_change = float(min(positive_quote_changes))
        if finest_quote_change < inferred_tick_size:
            warnings.append(
                "Inferred tick size is coarser than the smallest positive bid/ask change."
            )
    else:
        warnings.append("No positive bid/ask changes found in the selected data.")

    if retained_fraction is not None and retained_fraction < 0.5:
        warnings.append("Less than 50% of rows are retained by the configured n_spread filter.")

    if inferred_tick_size > float(positive_spread.min()):
        warnings.append("Inferred tick size exceeds the minimum positive spread.")

    return {
        "row_count": int(len(frame)),
        "min_positive_spread": float(positive_spread.min()),
        "most_common_positive_spreads": [
            {"spread": float(index), "count": int(count)} for index, count in top_spreads.items()
        ],
        "smallest_positive_bid_change": bid_change,
        "smallest_positive_ask_change": ask_change,
        "inferred_tick_size": inferred_tick_size,
        "retained_fraction_at_n_spread": retained_fraction,
        "warnings": warnings,
    }


def diagnose_raw_hyperliquid(
    raw_dir: str | Path,
    date_from: str | None = None,
    date_to: str | None = None,
    n_spread: int | None = None,
) -> dict[str, Any]:
    _print_stage("diagnosing raw input")
    source_type = _detect_raw_source_type(raw_dir)
    if source_type == "tensor":
        source_paths = _iter_l2_tensor_paths(raw_dir, date_from=date_from, date_to=date_to)
        extractor = extract_l1_from_l2_tensor
    elif source_type == "jsonl":
        source_paths = _iter_hyperliquid_jsonl_paths(raw_dir, date_from=date_from, date_to=date_to)
        extractor = extract_l1_from_hyperliquid_jsonl
    elif source_type == "npz":
        source_paths = _iter_npz_paths(raw_dir, date_from=date_from, date_to=date_to)
        extractor = extract_l1_from_npz
    elif source_type == "csv":
        source_paths = _iter_csv_paths(raw_dir, date_from=date_from, date_to=date_to)
        extractor = extract_l1_from_csv
    else:
        raise ValueError(f"Unsupported raw source type: {source_type}")

    frames = []
    total = len(source_paths)
    for index, path in enumerate(source_paths, start=1):
        _print_file_progress("Diagnosing", index, total, path)
        frames.append(extractor(path))
    combined = pd.concat(frames, ignore_index=True)
    summary = summarize_l1_diagnostics(combined, n_spread=n_spread)
    summary["source"] = source_type
    summary["file_count"] = len(source_paths)
    return summary


def _load_raw_l1_frame(
    raw_dir: str | Path,
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    source_type = _detect_raw_source_type(raw_dir)
    if source_type == "tensor":
        source_paths = _iter_l2_tensor_paths(raw_dir, date_from=date_from, date_to=date_to)
        extractor = extract_l1_from_l2_tensor
    elif source_type == "jsonl":
        source_paths = _iter_hyperliquid_jsonl_paths(raw_dir, date_from=date_from, date_to=date_to)
        extractor = extract_l1_from_hyperliquid_jsonl
    elif source_type == "npz":
        source_paths = _iter_npz_paths(raw_dir, date_from=date_from, date_to=date_to)
        extractor = extract_l1_from_npz
    elif source_type == "csv":
        source_paths = _iter_csv_paths(raw_dir, date_from=date_from, date_to=date_to)
        extractor = extract_l1_from_csv
    else:
        raise ValueError(f"Unsupported raw source type: {source_type}")

    frames = []
    total = len(source_paths)
    for index, path in enumerate(source_paths, start=1):
        _print_file_progress("Loading raw", index, total, path)
        frames.append(extractor(path))
    return pd.concat(frames, ignore_index=True)


def load_l1_source_frame(
    source: str,
    raw_dir: str | Path | None = None,
    parquet_dir: str | Path | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    if source == "parquet":
        path = Path("l1_cache_btc") if parquet_dir is None else Path(parquet_dir)
        return load_l1_cache(path, date_from=date_from, date_to=date_to)
    if source == "raw":
        if raw_dir is None:
            raise ValueError("raw_dir is required when loading raw input.")
        return _load_raw_l1_frame(raw_dir, date_from=date_from, date_to=date_to)
    raise ValueError(f"Unsupported source '{source}'. Expected 'raw' or 'parquet'.")


def chronological_train_test_split(
    df: pd.DataFrame,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _normalize_input(df).sort_values("time", kind="mergesort").reset_index(drop=True)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1.")
    split_index = int(math.floor(len(frame) * train_fraction))
    if split_index <= 0 or split_index >= len(frame):
        raise ValueError("train_fraction leaves an empty train or test split.")
    train_df = frame.iloc[:split_index].reset_index(drop=True)
    test_df = frame.iloc[split_index:].reset_index(drop=True)
    return train_df, test_df


def prepare_alpha_evaluation_frame(
    df: pd.DataFrame,
    horizon: int = 4,
) -> pd.DataFrame:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    frame = _normalize_input(df).sort_values("time", kind="mergesort").reset_index(drop=True)
    frame["mid"] = (frame["bid"] + frame["ask"]) / 2.0
    total_size = frame["bs"] + frame["as"]
    frame["imbalance"] = np.where(total_size > 0.0, frame["bs"] / total_size, 0.5)
    frame["weighted_mid"] = np.where(
        total_size > 0.0,
        (frame["ask"] * frame["bs"] + frame["bid"] * frame["as"]) / total_size,
        frame["mid"],
    )
    frame["weighted_mid_adjustment"] = frame["weighted_mid"] - frame["mid"]
    frame["imbalance_signal"] = frame["imbalance"] - 0.5
    frame["future_mid"] = frame["mid"].shift(-horizon)
    frame["future_return"] = frame["future_mid"] - frame["mid"]
    return frame


def _model_adjustment_series(
    frame: pd.DataFrame,
    model: FittedMicropriceModel,
    estimator: str | int,
) -> pd.Series:
    total_size = frame["bs"] + frame["as"]
    spread = frame["ask"] - frame["bid"]
    spread_ticks_raw = np.rint(spread / model.tick_size)
    valid = (
        (spread > 0.0)
        & (total_size > 0.0)
        & np.isfinite(spread_ticks_raw)
        & (spread_ticks_raw >= 1)
        & (spread_ticks_raw <= model.n_spread)
    )
    adjustment = pd.Series(np.nan, index=frame.index, dtype=float)
    if not bool(valid.any()):
        return adjustment

    imbalance = np.clip(frame["imbalance"].to_numpy(dtype=float), 0.0, 1.0)
    imbalance_bucket = np.searchsorted(model.imbalance_edges[1:-1], imbalance, side="right")
    imbalance_bucket = np.clip(imbalance_bucket, 0, model.n_imb - 1)
    spread_ticks = spread_ticks_raw[valid].astype(int)
    valid_indices = np.flatnonzero(valid.to_numpy(dtype=bool))
    state_indices = (spread_ticks - 1) * model.n_imb + imbalance_bucket[valid_indices]
    adjustment_vector = model._adjustment_vector(estimator)
    adjustment.iloc[valid_indices] = adjustment_vector[state_indices]
    return adjustment


def build_alpha_signal_frame(
    frame: pd.DataFrame,
    model: FittedMicropriceModel,
) -> pd.DataFrame:
    if "future_return" not in frame.columns or "imbalance" not in frame.columns:
        raise ValueError("frame must be produced by prepare_alpha_evaluation_frame.")
    signals: dict[str, pd.Series] = {
        "weighted_mid_adjustment": frame["weighted_mid_adjustment"].astype(float),
        "imbalance_signal": frame["imbalance_signal"].astype(float),
    }
    for horizon in model.available_horizons():
        signals[f"G{horizon}"] = _model_adjustment_series(frame, model, horizon)
    signals["G_star"] = _model_adjustment_series(frame, model, "G_star")
    return pd.DataFrame(signals)


def _bucket_mean_future_returns(
    signal: pd.Series,
    target: pd.Series,
    buckets: int = 10,
) -> list[dict[str, Any]]:
    valid = signal.notna() & target.notna()
    if not bool(valid.any()):
        return []
    signal_values = signal.loc[valid].astype(float)
    target_values = target.loc[valid].astype(float)
    bucket_count = min(int(buckets), len(signal_values))
    ranked = signal_values.rank(method="first")
    bucket_ids = pd.qcut(ranked, bucket_count, labels=False) + 1
    summary = (
        pd.DataFrame({"bucket": bucket_ids.astype(int), "future_return": target_values.to_numpy()})
        .groupby("bucket", sort=True)["future_return"]
        .agg(["mean", "count"])
    )
    return [
        {
            "bucket": int(bucket),
            "mean_future_return": float(row["mean"]),
            "count": int(row["count"]),
        }
        for bucket, row in summary.iterrows()
    ]


def summarize_alpha_signal(
    signal: pd.Series,
    target: pd.Series,
    buckets: int = 10,
) -> dict[str, Any]:
    target_ready = int(target.notna().sum())
    valid = signal.notna() & target.notna()
    row_count = int(valid.sum())
    if row_count == 0:
        return {
            "coverage": 0.0 if target_ready > 0 else 0.0,
            "row_count": 0,
            "pearson_correlation": None,
            "spearman_correlation": None,
            "directional_accuracy": None,
            "deciles": [],
        }

    signal_values = signal.loc[valid].astype(float)
    target_values = target.loc[valid].astype(float)
    pearson = signal_values.corr(target_values)
    spearman = signal_values.rank(method="average").corr(target_values.rank(method="average"))
    nonzero_target = target_values != 0.0
    directional_accuracy = None
    if bool(nonzero_target.any()):
        directional_accuracy = float(
            (np.sign(signal_values.loc[nonzero_target]) == np.sign(target_values.loc[nonzero_target])).mean()
        )

    return {
        "coverage": 0.0 if target_ready == 0 else float(row_count / target_ready),
        "row_count": row_count,
        "pearson_correlation": None if pd.isna(pearson) else float(pearson),
        "spearman_correlation": None if pd.isna(spearman) else float(spearman),
        "directional_accuracy": directional_accuracy,
        "deciles": _bucket_mean_future_returns(signal_values, target_values, buckets=buckets),
    }


def _decile_spread(deciles: list[dict[str, Any]]) -> float | None:
    if len(deciles) < 2:
        return None
    return float(deciles[-1]["mean_future_return"] - deciles[0]["mean_future_return"])


def _thresholded_signal_metrics(
    signal: pd.Series,
    target: pd.Series,
    tick_size: float,
    threshold_percentiles: tuple[int | float, ...] = DEFAULT_THRESHOLD_PERCENTILES,
) -> list[dict[str, Any]]:
    valid = signal.notna() & target.notna()
    target_ready = int(target.notna().sum())
    if not bool(valid.any()):
        return []
    signal_values = signal.loc[valid].astype(float)
    target_values = target.loc[valid].astype(float)
    absolute_signal = signal_values.abs()
    results: list[dict[str, Any]] = []
    for percentile in threshold_percentiles:
        percentile_value = float(percentile)
        if not 0.0 < percentile_value < 100.0:
            raise ValueError("threshold percentiles must be strictly between 0 and 100.")
        threshold = float(np.nanpercentile(absolute_signal.to_numpy(dtype=np.float64), percentile_value))
        selected = absolute_signal >= threshold
        row_count = int(selected.sum())
        if row_count == 0:
            results.append(
                {
                    "percentile": percentile_value,
                    "threshold": threshold,
                    "coverage": 0.0,
                    "row_count": 0,
                    "hit_rate": None,
                    "mean_future_return_ticks": None,
                }
            )
            continue
        selected_signal = signal_values.loc[selected]
        selected_target = target_values.loc[selected]
        nonzero_target = selected_target != 0.0
        hit_rate = None
        if bool(nonzero_target.any()):
            hit_rate = float(
                (np.sign(selected_signal.loc[nonzero_target]) == np.sign(selected_target.loc[nonzero_target])).mean()
            )
        results.append(
            {
                "percentile": percentile_value,
                "threshold": threshold,
                "coverage": 0.0 if target_ready == 0 else float(row_count / target_ready),
                "row_count": row_count,
                "hit_rate": hit_rate,
                "mean_future_return_ticks": float((np.sign(selected_signal) * selected_target / tick_size).mean()),
            }
        )
    return results


def summarize_signal_performance(
    signal: pd.Series,
    target: pd.Series,
    tick_size: float,
    buckets: int = 10,
    threshold_percentiles: tuple[int | float, ...] = DEFAULT_THRESHOLD_PERCENTILES,
) -> dict[str, Any]:
    summary = summarize_alpha_signal(signal, target, buckets=buckets)
    valid = signal.notna() & target.notna()
    if not bool(valid.any()):
        summary["mse_ticks"] = None
        summary["mae_ticks"] = None
        summary["rmse_ticks"] = None
        summary["top_bottom_decile_spread_ticks"] = None
        summary["threshold_metrics"] = []
        return summary
    signal_ticks = signal.loc[valid].astype(float).to_numpy(dtype=np.float64) / tick_size
    target_ticks = target.loc[valid].astype(float).to_numpy(dtype=np.float64) / tick_size
    errors = signal_ticks - target_ticks
    summary["mse_ticks"] = float(np.mean(errors * errors))
    summary["mae_ticks"] = float(np.mean(np.abs(errors)))
    summary["rmse_ticks"] = float(np.sqrt(np.mean(errors * errors)))
    summary["top_bottom_decile_spread_ticks"] = (
        None if len(summary["deciles"]) < 2 else _decile_spread(summary["deciles"]) / tick_size
    )
    summary["threshold_metrics"] = _thresholded_signal_metrics(
        signal=signal,
        target=target,
        tick_size=tick_size,
        threshold_percentiles=threshold_percentiles,
    )
    return summary


def evaluate_alpha_signals(
    frame: pd.DataFrame,
    signals: pd.DataFrame,
    buckets: int = 10,
) -> list[dict[str, Any]]:
    if "future_return" not in frame.columns:
        raise ValueError("frame must include a future_return column.")
    results = []
    for name in signals.columns:
        summary = summarize_alpha_signal(signals[name], frame["future_return"], buckets=buckets)
        summary["name"] = name
        results.append(summary)
    return results


def run_alpha_evaluation(
    source: str = "parquet",
    raw_dir: str | Path | None = None,
    parquet_dir: str | Path | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    train_fraction: float = 0.8,
    horizon: int = 4,
    n_imb: int = 10,
    n_spread: int = 2,
    dt: int = 1,
    max_horizon: int = 6,
    model_path: str | Path | None = None,
    buckets: int = 10,
) -> dict[str, Any]:
    frame = load_l1_source_frame(
        source=source,
        raw_dir=raw_dir,
        parquet_dir=parquet_dir,
        date_from=date_from,
        date_to=date_to,
    )
    train_df, test_df = chronological_train_test_split(frame, train_fraction=train_fraction)
    if model_path is None:
        model = fit_from_dataframe(train_df, n_imb=n_imb, n_spread=n_spread, dt=dt, max_horizon=max_horizon)
        mode = "refit"
    else:
        model = load_model(model_path)
        mode = "score-existing"

    evaluation_frame = prepare_alpha_evaluation_frame(test_df, horizon=horizon)
    signals = build_alpha_signal_frame(evaluation_frame, model)
    summaries = evaluate_alpha_signals(evaluation_frame, signals, buckets=buckets)
    target_row_count = int(evaluation_frame["future_return"].notna().sum())

    return {
        "source": source,
        "mode": mode,
        "raw_dir": None if raw_dir is None else str(raw_dir),
        "parquet_dir": None if parquet_dir is None else str(parquet_dir),
        "date_from": date_from,
        "date_to": date_to,
        "train_fraction": float(train_fraction),
        "horizon": int(horizon),
        "train_row_count": int(len(train_df)),
        "test_row_count": int(len(test_df)),
        "target_row_count": target_row_count,
        "model": {
            "tick_size": float(model.tick_size),
            "n_imb": int(model.n_imb),
            "n_spread": int(model.n_spread),
            "dt": int(model.dt),
            "max_horizon": int(max(model.available_horizons())),
        },
        "signals": summaries,
    }


def run_l1_microprice_evaluation(
    df: pd.DataFrame,
    model: FittedMicropriceModel | None = None,
    train_fraction: float = 0.8,
    horizon: int = 1,
    n_imb: int = 10,
    n_spread: int = 2,
    dt: int = 1,
    max_horizon: int = 6,
    tick_size: float | None = None,
    buckets: int = 10,
    threshold_percentiles: tuple[int | float, ...] = DEFAULT_THRESHOLD_PERCENTILES,
    purge_rows: int = 0,
) -> dict[str, Any]:
    frame = _normalize_input(df)
    if purge_rows < 0:
        raise ValueError("purge_rows must be nonnegative.")
    split_index = int(math.floor(len(frame) * train_fraction))
    train_end = split_index - purge_rows
    validation_start = split_index + purge_rows
    if train_end <= 0 or validation_start >= len(frame):
        raise ValueError("train_fraction and purge_rows leave an empty train or validation split.")
    train_df = frame.iloc[:train_end].reset_index(drop=True)
    validation_df = frame.iloc[validation_start:].reset_index(drop=True)
    fitted_model = model or fit_from_dataframe(
        train_df,
        n_imb=n_imb,
        n_spread=n_spread,
        dt=dt,
        max_horizon=max_horizon,
        tick_size=tick_size,
    )
    evaluation_frame = prepare_alpha_evaluation_frame(validation_df, horizon=horizon)
    raw_signal = evaluation_frame["weighted_mid_adjustment"].astype(float)
    fitted_signals = build_alpha_signal_frame(evaluation_frame, fitted_model)
    signal_summaries = [
        {
            "name": "raw_l1_microprice",
            **summarize_signal_performance(
                raw_signal,
                evaluation_frame["future_return"],
                tick_size=fitted_model.tick_size,
                buckets=buckets,
                threshold_percentiles=threshold_percentiles,
            ),
        }
    ]
    for name in fitted_signals.columns:
        signal_summaries.append(
            {
                "name": name,
                **summarize_signal_performance(
                    fitted_signals[name],
                    evaluation_frame["future_return"],
                    tick_size=fitted_model.tick_size,
                    buckets=buckets,
                    threshold_percentiles=threshold_percentiles,
                ),
            }
        )
    return {
        "model_kind": "l1",
        "tick_size": float(fitted_model.tick_size),
        "horizon": int(horizon),
        "train_row_count": int(len(train_df)),
        "validation_row_count": int(len(validation_df)),
        "target_row_count": int(evaluation_frame["future_return"].notna().sum()),
        "signals": signal_summaries,
    }


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def print_alpha_evaluation_report(report: dict[str, Any]) -> None:
    print(
        "Alpha evaluation:"
        f" source={report['source']}"
        f", mode={report['mode']}"
        f", horizon={report['horizon']}"
        f", train_fraction={report['train_fraction']:.2f}"
    )
    print(
        "Dataset:"
        f" train_rows={report['train_row_count']}"
        f", test_rows={report['test_row_count']}"
        f", target_rows={report['target_row_count']}"
    )
    model = report["model"]
    print(
        "Model:"
        f" tick_size={model['tick_size']}"
        f", n_imb={model['n_imb']}"
        f", n_spread={model['n_spread']}"
        f", dt={model['dt']}"
        f", max_horizon={model['max_horizon']}"
    )
    print("Signal scores:")
    for signal in report["signals"]:
        print(
            f"  {signal['name']}:"
            f" coverage={signal['coverage']:.2%}"
            f", rows={signal['row_count']}"
            f", pearson={_format_metric(signal['pearson_correlation'])}"
            f", spearman={_format_metric(signal['spearman_correlation'])}"
            f", directional_accuracy={_format_metric(signal['directional_accuracy'])}"
        )
        print("    deciles:")
        for bucket in signal["deciles"]:
            print(
                f"      {bucket['bucket']}:"
                f" mean_future_return={bucket['mean_future_return']:.6f}"
                f", count={bucket['count']}"
            )


def diagnose_parquet_cache(
    parquet_dir: str | Path,
    date_from: str | None = None,
    date_to: str | None = None,
    n_spread: int | None = None,
) -> dict[str, Any]:
    _print_stage("diagnosing parquet cache")
    frame = load_l1_cache(parquet_dir, date_from=date_from, date_to=date_to)
    summary = summarize_l1_diagnostics(frame, n_spread=n_spread)
    summary["source"] = "parquet"
    summary["file_count"] = len(
        [
            path
            for path in sorted(Path(parquet_dir).rglob("*.parquet"))
            if _date_in_range(_extract_date_from_path(path), date_from, date_to)
        ]
    )
    return summary


def _format_float(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value:.10g}"


def _print_diagnostics(summary: dict[str, Any], verbose: bool) -> None:
    print(
        "L1 diagnostics:"
        f" source={summary.get('source', 'dataframe')}"
        f", files={summary.get('file_count', 'n/a')}"
        f", rows={summary['row_count']}"
        f", inferred_tick_size={_format_float(summary['inferred_tick_size'])}"
        f", min_positive_spread={_format_float(summary['min_positive_spread'])}"
        f", smallest_positive_bid_change={_format_float(summary['smallest_positive_bid_change'])}"
        f", smallest_positive_ask_change={_format_float(summary['smallest_positive_ask_change'])}"
    )
    retained_fraction = summary.get("retained_fraction_at_n_spread")
    if retained_fraction is not None:
        print(f"Rows retained by n_spread filter: {retained_fraction:.2%}")
    if verbose:
        print("Most common positive spreads:")
        for item in summary["most_common_positive_spreads"]:
            print(f"  spread={_format_float(item['spread'])}, count={item['count']}")
    if summary["warnings"]:
        print("Warnings:")
        for warning in summary["warnings"]:
            print(f"  - {warning}")


def _qcut_edges(values: pd.Series, n_imb: int) -> tuple[pd.Series, np.ndarray]:
    buckets, edges = pd.qcut(values, n_imb, labels=False, retbins=True, duplicates="drop")
    if len(edges) - 1 != n_imb:
        raise ValueError(f"Unable to create {n_imb} imbalance buckets from the calibration data.")
    return buckets.astype(int), np.asarray(edges, dtype=float)


def _propagate_adjustment(
    G1: np.ndarray,
    B: np.ndarray,
    max_steps: int = 100,
    tol: float = 1e-12,
) -> np.ndarray:
    cumulative = np.array(G1, dtype=float)
    contribution = np.array(G1, dtype=float)
    converged = False
    for _ in range(max_steps - 1):
        contribution = B @ contribution
        cumulative = cumulative + contribution
        if np.max(np.abs(contribution)) < tol:
            converged = True
            break
    if not converged:
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(B))))
        warnings.warn(
            f"G_star Neumann series did not converge in {max_steps} steps. "
            f"Spectral radius of B = {spectral_radius:.6f}. "
            f"Results may be inaccurate if rho(B) >= 1.",
            stacklevel=2,
        )
    return cumulative


def _propagate_adjustment_steps(
    G1: np.ndarray,
    B: np.ndarray,
    steps: int,
) -> np.ndarray:
    if steps <= 0:
        raise ValueError("steps must be positive.")
    cumulative = np.array(G1, dtype=float)
    contribution = np.array(G1, dtype=float)
    for _ in range(steps - 1):
        contribution = B @ contribution
        cumulative = cumulative + contribution
    return cumulative


def _safe_solve(system: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Transition matrix (I - Q) is singular; falling back to pseudo-inverse. "
            "This typically indicates absorbing states in the Markov chain.",
            stacklevel=2,
        )
        return np.linalg.pinv(system) @ rhs


def prepare_training_data(
    df: pd.DataFrame,
    n_imb: int,
    dt: int,
    n_spread: int,
    tick_size: float | None = None,
    drop_invalid_rows: bool = True,
    mirror: bool = True,
) -> tuple[pd.DataFrame, float]:
    _print_stage("preparing training data")
    frame = _normalize_input(df, drop_invalid_rows=drop_invalid_rows)
    frame["spread"] = frame["ask"] - frame["bid"]
    tick_size = _resolve_tick_size(frame, tick_size=tick_size)
    frame["spread_ticks"] = np.rint(frame["spread"] / tick_size).astype(int)
    frame["spread"] = frame["spread_ticks"] * tick_size
    frame["mid"] = (frame["bid"] + frame["ask"]) / 2.0
    frame = frame.loc[(frame["spread_ticks"] >= 1) & (frame["spread_ticks"] <= n_spread)].copy()
    total_size = frame["bs"] + frame["as"]
    frame = frame.loc[total_size > 0].copy()
    frame["imb"] = frame["bs"] / total_size.loc[frame.index]
    frame["imb_bucket"], imbalance_edges = _qcut_edges(frame["imb"], n_imb)
    frame["next_mid"] = frame["mid"].shift(-dt)
    frame["next_spread_ticks"] = frame["spread_ticks"].shift(-dt)
    frame["next_spread"] = frame["spread"].shift(-dt)
    frame["next_time"] = frame["time"].shift(-dt)
    frame["next_imb_bucket"] = frame["imb_bucket"].shift(-dt)
    frame["dM"] = np.rint(((frame["next_mid"] - frame["mid"]) / tick_size) * 2.0) * tick_size / 2.0
    frame = frame.dropna().copy()
    frame["next_spread_ticks"] = frame["next_spread_ticks"].astype(int)
    frame["next_imb_bucket"] = frame["next_imb_bucket"].astype(int)
    pre_filter_count = len(frame)
    frame = frame.loc[
        (frame["dM"] <= tick_size * 1.1)
        & (frame["dM"] >= -tick_size * 1.1)
        & (frame["next_spread_ticks"] >= 1)
        & (frame["next_spread_ticks"] <= n_spread)
    ].copy()
    discarded = pre_filter_count - len(frame)
    if discarded > 0 and pre_filter_count > 0:
        pct = 100.0 * discarded / pre_filter_count
        if pct > 5.0:
            warnings.warn(
                f"Large-move filter discarded {discarded} rows ({pct:.1f}% of data). "
                f"This may indicate tick_size mismatch or volatile periods being excluded.",
                stacklevel=2,
            )

    if mirror:
        mirrored = frame.copy(deep=True)
        mirrored["imb"] = 1.0 - mirrored["imb"]
        mirrored["imb_bucket"] = n_imb - 1 - mirrored["imb_bucket"]
        mirrored["next_imb_bucket"] = n_imb - 1 - mirrored["next_imb_bucket"]
        mirrored["dM"] = -mirrored["dM"]
        mirrored["mid"] = -mirrored["mid"]
        prepared = pd.concat([frame, mirrored], ignore_index=True)
    else:
        prepared = frame.reset_index(drop=True)

    prepared.attrs["imbalance_edges"] = imbalance_edges.tolist()
    prepared.attrs["dt"] = dt
    return prepared, tick_size


def _state_index(spread_ticks: int, imb_bucket: int, n_imb: int) -> int:
    return (int(spread_ticks) - 1) * n_imb + int(imb_bucket)


def fit_microprice_model(
    prepared_df: pd.DataFrame,
    n_imb: int,
    n_spread: int,
    tick_size: float,
    max_horizon: int = 6,
) -> FittedMicropriceModel:
    _print_stage("estimating microprice model")
    if prepared_df.empty:
        raise ValueError("Prepared dataframe is empty.")
    if max_horizon <= 0:
        raise ValueError("max_horizon must be positive.")

    imbalance_edges = np.asarray(prepared_df.attrs.get("imbalance_edges"), dtype=float)
    if imbalance_edges.size != n_imb + 1:
        raise ValueError("Prepared dataframe is missing imbalance bucket edges.")

    dt = int(prepared_df.attrs.get("dt", 1))
    state_count = n_imb * n_spread
    move_values = np.array([-tick_size, -tick_size / 2.0, tick_size / 2.0, tick_size], dtype=float)
    move_lookup = {round(float(value), 12): idx for idx, value in enumerate(move_values)}

    Q_counts = np.zeros((state_count, state_count), dtype=float)
    R1_counts = np.zeros((state_count, len(move_values)), dtype=float)
    R2_counts = np.zeros((state_count, state_count), dtype=float)

    spread_ticks_arr = prepared_df["spread_ticks"].to_numpy(dtype=np.intp)
    imb_bucket_arr = prepared_df["imb_bucket"].to_numpy(dtype=np.intp)
    next_spread_arr = prepared_df["next_spread_ticks"].to_numpy(dtype=np.intp)
    next_imb_arr = prepared_df["next_imb_bucket"].to_numpy(dtype=np.intp)
    dM_arr = prepared_df["dM"].to_numpy(dtype=np.float64)

    current_state_arr = (spread_ticks_arr - 1) * n_imb + imb_bucket_arr
    next_state_arr = (next_spread_arr - 1) * n_imb + next_imb_arr
    rounded_dM = np.round(dM_arr, 12)

    is_zero = np.abs(rounded_dM) < 1e-12
    np.add.at(Q_counts, (current_state_arr[is_zero], next_state_arr[is_zero]), 1.0)

    nonzero_mask = ~is_zero
    nz_dM = rounded_dM[nonzero_mask]
    nz_cur = current_state_arr[nonzero_mask]
    nz_nxt = next_state_arr[nonzero_mask]

    move_indices = np.full(len(nz_dM), -1, dtype=np.intp)
    for mv_val, mv_idx in move_lookup.items():
        move_indices[np.abs(nz_dM - mv_val) < 1e-12] = mv_idx

    valid_moves = move_indices >= 0
    np.add.at(R1_counts, (nz_cur[valid_moves], move_indices[valid_moves]), 1.0)
    np.add.at(R2_counts, (nz_cur[valid_moves], nz_nxt[valid_moves]), 1.0)

    def normalize_rows(matrix: np.ndarray) -> np.ndarray:
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = matrix.copy().astype(float)
        nonzero = row_sums[:, 0] > 0
        normalized[nonzero] = normalized[nonzero] / row_sums[nonzero]
        return normalized

    T1 = normalize_rows(np.concatenate([Q_counts, R1_counts], axis=1))
    T2 = normalize_rows(np.concatenate([Q_counts, R2_counts], axis=1))
    Q = T1[:, :state_count]
    R1 = T1[:, state_count:]
    Q2 = T2[:, :state_count]
    R2 = T2[:, state_count:]
    identity = np.eye(state_count)
    G1 = _safe_solve(identity - Q, R1 @ move_values)
    B = _safe_solve(identity - Q, R2)
    finite_horizons = {
        str(horizon): _propagate_adjustment_steps(G1, B, steps=horizon) for horizon in range(1, max_horizon + 1)
    }
    G6 = finite_horizons.get("6")
    G_star = _propagate_adjustment(G1, B)

    return FittedMicropriceModel(
        tick_size=tick_size,
        n_imb=n_imb,
        n_spread=n_spread,
        dt=dt,
        imbalance_edges=imbalance_edges,
        G1=G1,
        B=B,
        G6=G6,
        G_star=G_star,
        Q=Q,
        Q2=Q2,
        R1=R1,
        R2=R2,
        move_values=move_values,
        finite_horizons=finite_horizons,
    )


def fit_from_dataframe(
    df: pd.DataFrame,
    n_imb: int = 10,
    n_spread: int = 2,
    dt: int = 1,
    max_horizon: int = 6,
    tick_size: float | None = None,
    drop_invalid_rows: bool = True,
    mirror: bool = True,
) -> FittedMicropriceModel:
    _print_stage("fitting from dataframe")
    prepared_df, tick_size = prepare_training_data(
        df,
        n_imb=n_imb,
        dt=dt,
        n_spread=n_spread,
        tick_size=tick_size,
        drop_invalid_rows=drop_invalid_rows,
        mirror=mirror,
    )
    return fit_microprice_model(
        prepared_df,
        n_imb=n_imb,
        n_spread=n_spread,
        tick_size=tick_size,
        max_horizon=max_horizon,
    )


def fit_from_parquet(
    parquet_dir: str | Path,
    n_imb: int = 10,
    n_spread: int = 2,
    dt: int = 1,
    max_horizon: int = 6,
    tick_size: float | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> FittedMicropriceModel:
    start = time.perf_counter()
    frame = load_l1_cache(parquet_dir, date_from=date_from, date_to=date_to)
    print(f"Loaded parquet rows={len(frame)} in {time.perf_counter() - start:.2f}s")
    fit_start = time.perf_counter()
    model = fit_from_dataframe(
        frame,
        n_imb=n_imb,
        n_spread=n_spread,
        dt=dt,
        max_horizon=max_horizon,
        tick_size=tick_size,
    )
    print(f"Fit completed in {time.perf_counter() - fit_start:.2f}s")
    return model


def fit_from_raw_hyperliquid(
    raw_dir: str | Path,
    parquet_dir: str | Path | None = None,
    n_imb: int = 10,
    n_spread: int = 2,
    dt: int = 1,
    max_horizon: int = 6,
    tick_size: float | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> FittedMicropriceModel:
    if parquet_dir is not None:
        cache_start = time.perf_counter()
        build_l1_cache(raw_dir, parquet_dir, date_from=date_from, date_to=date_to)
        print(f"Cache build completed in {time.perf_counter() - cache_start:.2f}s")
        return fit_from_parquet(
            parquet_dir,
            n_imb=n_imb,
            n_spread=n_spread,
            dt=dt,
            max_horizon=max_horizon,
            tick_size=tick_size,
            date_from=date_from,
            date_to=date_to,
        )

    source_type = _detect_raw_source_type(raw_dir)
    if source_type == "tensor":
        source_paths = _iter_l2_tensor_paths(raw_dir, date_from=date_from, date_to=date_to)
        extractor = extract_l1_from_l2_tensor
    elif source_type == "jsonl":
        source_paths = _iter_hyperliquid_jsonl_paths(raw_dir, date_from=date_from, date_to=date_to)
        extractor = extract_l1_from_hyperliquid_jsonl
    elif source_type == "npz":
        source_paths = _iter_npz_paths(raw_dir, date_from=date_from, date_to=date_to)
        extractor = extract_l1_from_npz
    elif source_type == "csv":
        source_paths = _iter_csv_paths(raw_dir, date_from=date_from, date_to=date_to)
        extractor = extract_l1_from_csv
    else:
        raise ValueError(f"Unsupported raw source type: {source_type}")

    parse_start = time.perf_counter()
    frames = []
    total = len(source_paths)
    for index, path in enumerate(source_paths, start=1):
        _print_file_progress("Parsing", index, total, path)
        frames.append(extractor(path))
    combined = pd.concat(frames, ignore_index=True)
    print(f"Parsed raw rows={len(combined)} in {time.perf_counter() - parse_start:.2f}s")
    fit_start = time.perf_counter()
    model = fit_from_dataframe(
        combined,
        n_imb=n_imb,
        n_spread=n_spread,
        dt=dt,
        max_horizon=max_horizon,
        tick_size=tick_size,
    )
    print(f"Fit completed in {time.perf_counter() - fit_start:.2f}s")
    return model


def run_calibration_sweep(
    df: pd.DataFrame,
    n_imb_values: list[int],
    n_spread_values: list[int],
    dt_values: list[int],
    max_horizon: int = 6,
    tick_size: float | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = len(n_imb_values) * len(n_spread_values) * len(dt_values)
    completed = 0
    for n_imb in n_imb_values:
        for n_spread in n_spread_values:
            for dt in dt_values:
                completed += 1
                print(
                    f"Sweep fit {completed}/{total}: "
                    f"n_imb={n_imb}, n_spread={n_spread}, dt={dt}"
                )
                started = time.perf_counter()
                model = fit_from_dataframe(
                    df,
                    n_imb=n_imb,
                    n_spread=n_spread,
                    dt=dt,
                    max_horizon=max_horizon,
                    tick_size=tick_size,
                )
                results.append(
                    {
                        "n_imb": int(n_imb),
                        "n_spread": int(n_spread),
                        "dt": int(dt),
                        "tick_size": float(model.tick_size),
                        "fit_seconds": float(time.perf_counter() - started),
                        "model": model,
                    }
                )
    return results


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an L1 cache and calibrate a Stoikov microprice model from Hyperliquid raw data or parquet."
    )
    parser.add_argument("-r", "--raw-dir", help="Directory containing raw data (.jsonl, .npy, .npz, or .csv).")
    parser.add_argument("-c", "--cache-dir", help="Directory for parquet L1 cache files.")
    parser.add_argument("-o", "--model-out", help="Output path for the calibrated model JSON.")
    parser.add_argument("--n-imb", type=int, default=10, help="Number of imbalance buckets.")
    parser.add_argument("--n-spread", type=int, default=2, help="Maximum spread in ticks used for calibration.")
    parser.add_argument("--dt", type=int, default=1, help="Forward step used for the Stoikov transition fit.")
    parser.add_argument("--tick-size", type=float, help="Explicit tick size. Preferred over inference.")
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=6,
        help="Largest finite propagated horizon Gk to store in the calibrated model. Defaults to 6.",
    )
    parser.add_argument("-f", "--date-from", help="Lower date bound in YYYYMMDD.")
    parser.add_argument("-t", "--date-to", help="Upper date bound in YYYYMMDD.")
    parser.add_argument(
        "--source",
        choices=("auto", "raw", "parquet"),
        default="auto",
        help="Calibration source. Default 'auto' uses parquet if --cache-dir exists, otherwise raw.",
    )
    parser.add_argument(
        "--skip-cache-build",
        action="store_true",
        help="When using --source raw, fit directly from raw files instead of writing parquet cache first.",
    )
    parser.add_argument(
        "--diagnose-only",
        action="store_true",
        help="Inspect L1 quote granularity and filtering without fitting or writing a model.",
    )
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()

    source = args.source
    if source == "auto":
        if args.cache_dir and Path(args.cache_dir).exists():
            source = "parquet"
        else:
            source = "raw"

    if source == "raw" and not args.raw_dir:
        parser.error("--raw-dir is required when using raw input.")
    if source == "parquet" and not args.cache_dir:
        parser.error("--cache-dir is required when using parquet input.")
    if not args.diagnose_only and not args.model_out:
        parser.error("--model-out is required unless --diagnose-only is used.")

    print(f"Calibration source={source}")

    if source == "parquet":
        diagnostics = diagnose_parquet_cache(
            args.cache_dir,
            date_from=args.date_from,
            date_to=args.date_to,
            n_spread=args.n_spread,
        )
    else:
        diagnostics = diagnose_raw_hyperliquid(
            args.raw_dir,
            date_from=args.date_from,
            date_to=args.date_to,
            n_spread=args.n_spread,
        )

    _print_diagnostics(diagnostics, verbose=args.diagnose_only)
    if args.diagnose_only:
        return

    if source == "parquet":
        model = fit_from_parquet(
            args.cache_dir,
            n_imb=args.n_imb,
            n_spread=args.n_spread,
            dt=args.dt,
            max_horizon=args.max_horizon,
            tick_size=args.tick_size,
            date_from=args.date_from,
            date_to=args.date_to,
        )
    else:
        parquet_dir = None if args.skip_cache_build else args.cache_dir
        model = fit_from_raw_hyperliquid(
            args.raw_dir,
            parquet_dir=parquet_dir,
            n_imb=args.n_imb,
            n_spread=args.n_spread,
            dt=args.dt,
            max_horizon=args.max_horizon,
            tick_size=args.tick_size,
            date_from=args.date_from,
            date_to=args.date_to,
        )

    save_model(model, args.model_out)
    print(f"Saved calibrated model to {args.model_out}")
    print(
        "Calibration summary:"
        f" tick_size={model.tick_size}"
        f", n_imb={model.n_imb}"
        f", n_spread={model.n_spread}"
        f", dt={model.dt}"
    )


if __name__ == "__main__":
    main()
