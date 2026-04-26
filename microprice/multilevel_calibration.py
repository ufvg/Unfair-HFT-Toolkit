from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from .calibration import (
    L1Microprice,
    L2_TENSOR_WIDTH,
    TIMESTAMP_COLUMN,
    build_alpha_signal_frame,
    evaluate_alpha_signals,
    fit_from_dataframe,
    prepare_alpha_evaluation_frame,
    run_l1_microprice_evaluation,
    summarize_signal_performance,
    summarize_alpha_signal,
)
from .multilevel_microprice import MultilevelMicroprice, raw_multilevel_microprice, raw_multilevel_microprice_batch

MAX_BOOK_LEVELS = 20
LEVEL_STRIDE = 2
BID_LEVEL_OFFSET = 1
ASK_LEVEL_OFFSET = 41
DEFAULT_CALIBRATION_HORIZON = 1
DEFAULT_ALPHA_HORIZON = 8
DEFAULT_DECILE_BUCKETS = 10
DEFAULT_THRESHOLD_PERCENTILES = (80, 90, 95)
DEFAULT_RIDGE_ALPHA_CANDIDATES = (0.0, 0.1, 1.0, 10.0, 100.0)
CalibrationUnits = Literal["price", "ticks"]


def _level_columns(levels: int) -> list[str]:
    if levels <= 0:
        raise ValueError("levels must be positive.")
    columns = ["time"]
    for level in range(1, levels + 1):
        columns.extend(
            [
                f"bid_px_{level}",
                f"bid_sz_{level}",
                f"ask_px_{level}",
                f"ask_sz_{level}",
            ]
        )
    return columns


def _price_columns(side: str, levels: int) -> list[str]:
    return [f"{side}_px_{level}" for level in range(1, levels + 1)]


def _size_columns(side: str, levels: int) -> list[str]:
    return [f"{side}_sz_{level}" for level in range(1, levels + 1)]


def _tensor_level_columns(levels: int, offset: int) -> tuple[list[int], list[int]]:
    if not 1 <= levels <= MAX_BOOK_LEVELS:
        raise ValueError(f"levels must be between 1 and {MAX_BOOK_LEVELS}.")
    price_columns = [offset + LEVEL_STRIDE * level for level in range(levels)]
    size_columns = [column + 1 for column in price_columns]
    return price_columns, size_columns


def _extract_date_from_path(path: Path) -> str | None:
    direct_match = re.search(r"(\d{8})", str(path))
    return None if direct_match is None else direct_match.group(1)


def extract_multilevel_from_l2_tensor(path: str | Path, levels: int = 5) -> pd.DataFrame:
    """Load the top `levels` bid and ask levels from an L2 tensor into a wide dataframe."""
    tensor_path = Path(path)
    tensor = np.load(tensor_path, mmap_mode="r")
    if tensor.ndim != 2:
        raise ValueError(f"Expected a 2D tensor in {tensor_path}, got shape {tensor.shape}.")
    if tensor.shape[1] != L2_TENSOR_WIDTH:
        raise ValueError(
            f"Unexpected tensor width in {tensor_path}: expected {L2_TENSOR_WIDTH}, got {tensor.shape[1]}."
        )

    bid_price_columns, bid_size_columns = _tensor_level_columns(levels, BID_LEVEL_OFFSET)
    ask_price_columns, ask_size_columns = _tensor_level_columns(levels, ASK_LEVEL_OFFSET)
    frame_dict: dict[str, Any] = {"time": tensor[:, TIMESTAMP_COLUMN]}
    for level in range(1, levels + 1):
        frame_dict[f"bid_px_{level}"] = tensor[:, bid_price_columns[level - 1]]
        frame_dict[f"bid_sz_{level}"] = tensor[:, bid_size_columns[level - 1]]
        frame_dict[f"ask_px_{level}"] = tensor[:, ask_price_columns[level - 1]]
        frame_dict[f"ask_sz_{level}"] = tensor[:, ask_size_columns[level - 1]]

    frame = pd.DataFrame(frame_dict)
    date_str = _extract_date_from_path(tensor_path)
    if date_str is not None:
        frame["date"] = date_str
    return frame


def _extract_multilevel_record_from_message(message: dict[str, Any], levels: int) -> dict[str, float] | None:
    raw = message.get("raw")
    if not isinstance(raw, dict) or raw.get("channel") != "l2Book":
        return None
    data = raw.get("data")
    if not isinstance(data, dict):
        return None
    book_levels = data.get("levels")
    if not isinstance(book_levels, list) or len(book_levels) < 2:
        return None
    bids = book_levels[0] or []
    asks = book_levels[1] or []
    if len(bids) < levels or len(asks) < levels:
        return None

    record: dict[str, float] = {"time": float(data["time"])}
    try:
        for level in range(levels):
            record[f"bid_px_{level + 1}"] = float(bids[level]["px"])
            record[f"bid_sz_{level + 1}"] = float(bids[level]["sz"])
            record[f"ask_px_{level + 1}"] = float(asks[level]["px"])
            record[f"ask_sz_{level + 1}"] = float(asks[level]["sz"])
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    return record


def extract_multilevel_from_hyperliquid_jsonl(path: str | Path, levels: int = 5) -> pd.DataFrame:
    """Load the top `levels` bid and ask levels from a Hyperliquid raw jsonl file."""
    jsonl_path = Path(path)
    rows: list[dict[str, float]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                message = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {jsonl_path} at line {line_number}.") from exc
            record = _extract_multilevel_record_from_message(message, levels=levels)
            if record is not None:
                rows.append(record)

    if not rows:
        raise ValueError(f"No valid multilevel l2Book records found in {jsonl_path}.")

    frame = pd.DataFrame(rows)
    parent_date = jsonl_path.parent.name if re.fullmatch(r"\d{8}", jsonl_path.parent.name) else None
    date_str = parent_date or _extract_date_from_path(jsonl_path)
    if date_str is not None:
        frame["date"] = date_str
    return frame


def _row_validity_mask(frame: pd.DataFrame, levels: int) -> np.ndarray:
    bid_prices = frame.loc[:, _price_columns("bid", levels)].to_numpy(dtype=np.float64, copy=False)
    bid_sizes = frame.loc[:, _size_columns("bid", levels)].to_numpy(dtype=np.float64, copy=False)
    ask_prices = frame.loc[:, _price_columns("ask", levels)].to_numpy(dtype=np.float64, copy=False)
    ask_sizes = frame.loc[:, _size_columns("ask", levels)].to_numpy(dtype=np.float64, copy=False)

    valid = np.ones(len(frame), dtype=bool)
    valid &= np.all(np.isfinite(bid_prices), axis=1)
    valid &= np.all(np.isfinite(ask_prices), axis=1)
    valid &= np.all(np.isfinite(bid_sizes), axis=1)
    valid &= np.all(np.isfinite(ask_sizes), axis=1)
    valid &= np.all(bid_sizes >= 0.0, axis=1)
    valid &= np.all(ask_sizes >= 0.0, axis=1)
    valid &= bid_prices[:, 0] <= ask_prices[:, 0]
    if levels > 1:
        valid &= np.all(np.diff(bid_prices, axis=1) <= 0.0, axis=1)
        valid &= np.all(np.diff(ask_prices, axis=1) >= 0.0, axis=1)
    return valid


def _normalize_multilevel_input(
    df: pd.DataFrame,
    levels: int,
    drop_invalid_rows: bool = True,
) -> pd.DataFrame:
    required_columns = _level_columns(levels)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required multilevel columns: {missing}")

    frame = df.loc[:, required_columns].copy()
    for column in required_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().sort_values("time", kind="mergesort").reset_index(drop=True)
    if frame.empty:
        raise ValueError("Input dataframe has no valid multilevel rows after numeric coercion.")

    valid_mask = _row_validity_mask(frame, levels=levels)
    if not drop_invalid_rows and not bool(valid_mask.all()):
        raise ValueError("Input dataframe contains invalid multilevel book rows.")
    frame = frame.loc[valid_mask].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid multilevel rows remain after filtering invalid book states.")
    return frame


def _validate_explicit_tick_size(tick_size: float) -> float:
    resolved_tick_size = float(tick_size)
    if not np.isfinite(resolved_tick_size) or resolved_tick_size <= 0.0:
        raise ValueError("tick_size must be positive and finite.")
    return resolved_tick_size


def _infer_tick_size_from_price_grid(frame: pd.DataFrame, levels: int) -> float:
    """Infer tick size from the observed price grid, not from the spread."""
    candidates: list[float] = []
    for column in _price_columns("bid", levels) + _price_columns("ask", levels):
        values = np.asarray(frame[column], dtype=np.float64)
        unique_values = np.unique(np.round(values[np.isfinite(values)], decimals=12))
        if unique_values.size < 2:
            continue
        diffs = np.diff(unique_values)
        positive_diffs = diffs[diffs > 0.0]
        if positive_diffs.size:
            candidates.append(float(positive_diffs.min()))
    if not candidates:
        raise ValueError("Unable to infer tick size from the observed price grid. Pass tick_size explicitly.")
    return _validate_explicit_tick_size(min(candidates))


def _resolve_tick_size(frame: pd.DataFrame, levels: int, tick_size: float | None) -> float:
    if tick_size is not None:
        return _validate_explicit_tick_size(tick_size)
    return _infer_tick_size_from_price_grid(frame, levels=levels)


def _split_train_validation(
    frame: pd.DataFrame,
    train_fraction: float,
    purge_rows: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1.")
    if purge_rows < 0:
        raise ValueError("purge_rows must be nonnegative.")
    split_index = int(np.floor(len(frame) * train_fraction))
    train_end = split_index - purge_rows
    validation_start = split_index + purge_rows
    if train_end <= 0 or validation_start >= len(frame):
        raise ValueError("train_fraction and purge_rows leave an empty train or validation split.")
    train_frame = frame.iloc[:train_end].reset_index(drop=True)
    validation_frame = frame.iloc[validation_start:].reset_index(drop=True)
    return train_frame, validation_frame


def _compute_raw_microprice_series(
    frame: pd.DataFrame,
    levels: int,
    tick_size: float,
    decay_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    bid_prices = frame.loc[:, _price_columns("bid", levels)].to_numpy(dtype=np.float64, copy=False)
    bid_sizes = frame.loc[:, _size_columns("bid", levels)].to_numpy(dtype=np.float64, copy=False)
    ask_prices = frame.loc[:, _price_columns("ask", levels)].to_numpy(dtype=np.float64, copy=False)
    ask_sizes = frame.loc[:, _size_columns("ask", levels)].to_numpy(dtype=np.float64, copy=False)
    return raw_multilevel_microprice_batch(
        bid_prices=bid_prices,
        bid_sizes=bid_sizes,
        ask_prices=ask_prices,
        ask_sizes=ask_sizes,
        tick_size=tick_size,
        decay_lambda=decay_lambda,
    )


def prepare_multilevel_feature_frame(
    df: pd.DataFrame,
    levels: int,
    decay_lambda: float,
    dt: int = DEFAULT_CALIBRATION_HORIZON,
    tick_size: float | None = None,
    drop_invalid_rows: bool = True,
) -> tuple[pd.DataFrame, float]:
    """Build a feature frame for multilevel microprice calibration."""
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if not np.isfinite(float(decay_lambda)) or float(decay_lambda) < 0.0:
        raise ValueError("decay_lambda must be finite and nonnegative.")

    frame = _normalize_multilevel_input(df, levels=levels, drop_invalid_rows=drop_invalid_rows)
    resolved_tick_size = _resolve_tick_size(frame, levels=levels, tick_size=tick_size)
    mid, raw_microprice = _compute_raw_microprice_series(
        frame=frame,
        levels=levels,
        tick_size=resolved_tick_size,
        decay_lambda=float(decay_lambda),
    )

    horizon_suffix = f"h{dt}"
    future_mid = pd.Series(mid, index=frame.index).shift(-dt)
    future_return = future_mid - mid
    future_return_ticks = future_return / resolved_tick_size

    feature_frame = frame.copy()
    feature_frame["mid"] = mid
    feature_frame["raw_microprice"] = raw_microprice
    feature_frame["raw_adjustment"] = feature_frame["raw_microprice"] - feature_frame["mid"]
    feature_frame["raw_adjustment_ticks"] = feature_frame["raw_adjustment"] / resolved_tick_size
    feature_frame["future_mid"] = future_mid
    feature_frame["future_return"] = future_return
    feature_frame["future_return_ticks"] = future_return_ticks
    feature_frame[f"future_mid_{horizon_suffix}"] = future_mid
    feature_frame[f"future_return_{horizon_suffix}"] = future_return
    feature_frame[f"future_return_{horizon_suffix}_ticks"] = future_return_ticks
    feature_frame = feature_frame.dropna().reset_index(drop=True)
    if feature_frame.empty:
        raise ValueError("No usable rows remain after applying dt to the multilevel feature frame.")
    return feature_frame, resolved_tick_size


def _calibration_arrays(
    feature_frame: pd.DataFrame,
    tick_size: float,
    calibration_units: CalibrationUnits,
) -> tuple[np.ndarray, np.ndarray]:
    raw_adjustment = feature_frame["raw_adjustment"].to_numpy(dtype=np.float64)
    future_return = feature_frame["future_return"].to_numpy(dtype=np.float64)
    if calibration_units == "price":
        return raw_adjustment, future_return
    if calibration_units == "ticks":
        return raw_adjustment / tick_size, future_return / tick_size
    raise ValueError("calibration_units must be 'price' or 'ticks'.")


def _fit_linear_adjustment(
    x: np.ndarray,
    y: np.ndarray,
    fit_intercept: bool = True,
) -> tuple[float, float]:
    """Fit `y ~= intercept + slope * x` with an explicit OLS baseline."""
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if x.ndim != 1:
        raise ValueError("x and y must be one-dimensional.")
    if x.size == 0:
        raise ValueError("x and y must not be empty.")

    if not fit_intercept:
        denominator = float(np.dot(x, x))
        if denominator <= 0.0:
            return 0.0, 0.0
        return 0.0, float(np.dot(x, y) / denominator)

    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    centered_x = x - x_mean
    variance = float(np.dot(centered_x, centered_x))
    if variance <= 0.0:
        return y_mean, 0.0
    covariance = float(np.dot(centered_x, y - y_mean))
    slope = covariance / variance
    intercept = y_mean - slope * x_mean
    return intercept, slope


def _predicted_adjustment_price(
    raw_adjustment: pd.Series,
    tick_size: float,
    intercept: float,
    slope: float,
    calibration_units: CalibrationUnits,
) -> pd.Series:
    raw_adjustment_values = raw_adjustment.astype(float)
    if calibration_units == "ticks":
        return (intercept + slope * (raw_adjustment_values / tick_size)) * tick_size
    return intercept + slope * raw_adjustment_values


def _decile_spread(deciles: list[dict[str, Any]]) -> float | None:
    if len(deciles) < 2:
        return None
    return float(deciles[-1]["mean_future_return"] - deciles[0]["mean_future_return"])


def _decile_monotonicity_score(deciles: list[dict[str, Any]]) -> float | None:
    if len(deciles) < 2:
        return None
    mean_values = np.asarray([float(item["mean_future_return"]) for item in deciles], dtype=np.float64)
    return float(np.mean(np.diff(mean_values) >= 0.0))


def _thresholded_signal_metrics(
    signal: pd.Series,
    target: pd.Series,
    threshold_percentiles: Sequence[int | float] = DEFAULT_THRESHOLD_PERCENTILES,
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
        signed_future_return = float((np.sign(selected_signal) * selected_target).mean())
        results.append(
            {
                "percentile": percentile_value,
                "threshold": threshold,
                "coverage": 0.0 if target_ready == 0 else float(row_count / target_ready),
                "row_count": row_count,
                "hit_rate": hit_rate,
                "mean_future_return_ticks": signed_future_return,
            }
        )
    return results


def summarize_multilevel_alpha_signal(
    signal_ticks: pd.Series,
    future_return_ticks: pd.Series,
    buckets: int = DEFAULT_DECILE_BUCKETS,
    threshold_percentiles: Sequence[int | float] = DEFAULT_THRESHOLD_PERCENTILES,
) -> dict[str, Any]:
    """Return alpha diagnostics for a signal scored against future return in ticks."""
    summary = summarize_alpha_signal(signal_ticks, future_return_ticks, buckets=buckets)
    deciles = summary["deciles"]
    summary["top_bottom_decile_spread_ticks"] = _decile_spread(deciles)
    summary["decile_monotonicity_score"] = _decile_monotonicity_score(deciles)
    summary["threshold_metrics"] = _thresholded_signal_metrics(
        signal_ticks,
        future_return_ticks,
        threshold_percentiles=threshold_percentiles,
    )
    return summary


def _evaluate_calibrated_signal(
    feature_frame: pd.DataFrame,
    tick_size: float,
    intercept: float,
    slope: float,
    calibration_units: CalibrationUnits,
    buckets: int = DEFAULT_DECILE_BUCKETS,
    threshold_percentiles: Sequence[int | float] = DEFAULT_THRESHOLD_PERCENTILES,
) -> dict[str, Any]:
    predicted_price = _predicted_adjustment_price(
        feature_frame["raw_adjustment"],
        tick_size=tick_size,
        intercept=intercept,
        slope=slope,
        calibration_units=calibration_units,
    )
    predicted_ticks = predicted_price / tick_size
    target_price = feature_frame["future_return"].astype(float)
    target_ticks = feature_frame["future_return_ticks"].astype(float)

    price_errors = predicted_price.to_numpy(dtype=np.float64) - target_price.to_numpy(dtype=np.float64)
    tick_errors = predicted_ticks.to_numpy(dtype=np.float64) - target_ticks.to_numpy(dtype=np.float64)
    regression_errors = tick_errors if calibration_units == "ticks" else price_errors
    alpha_summary = summarize_multilevel_alpha_signal(
        predicted_ticks,
        target_ticks,
        buckets=buckets,
        threshold_percentiles=threshold_percentiles,
    )
    return {
        "mse": float(np.mean(regression_errors * regression_errors)),
        "correlation": alpha_summary["pearson_correlation"],
        "mse_ticks": float(np.mean(tick_errors * tick_errors)),
        "mae_ticks": float(np.mean(np.abs(tick_errors))),
        "rmse_ticks": float(np.sqrt(np.mean(tick_errors * tick_errors))),
        "pearson": alpha_summary["pearson_correlation"],
        "spearman": alpha_summary["spearman_correlation"],
        "directional_accuracy": alpha_summary["directional_accuracy"],
        "coverage": alpha_summary["coverage"],
        "row_count": alpha_summary["row_count"],
        "deciles": alpha_summary["deciles"],
        "top_bottom_decile_spread_ticks": alpha_summary["top_bottom_decile_spread_ticks"],
        "decile_monotonicity_score": alpha_summary["decile_monotonicity_score"],
        "threshold_metrics": alpha_summary["threshold_metrics"],
    }


def _state_conditioning_series(
    feature_frame: pd.DataFrame,
    levels: int,
    tick_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    spread_ticks = np.rint(
        (
            feature_frame["ask_px_1"].to_numpy(dtype=np.float64, copy=False)
            - feature_frame["bid_px_1"].to_numpy(dtype=np.float64, copy=False)
        )
        / tick_size
    ).astype(np.int64)
    bid_sizes = feature_frame.loc[:, _size_columns("bid", levels)].to_numpy(dtype=np.float64, copy=False)
    ask_sizes = feature_frame.loc[:, _size_columns("ask", levels)].to_numpy(dtype=np.float64, copy=False)
    depth_imbalance = _safe_ratio(
        np.sum(bid_sizes, axis=1) - np.sum(ask_sizes, axis=1),
        np.sum(bid_sizes, axis=1) + np.sum(ask_sizes, axis=1),
    )
    return spread_ticks, depth_imbalance


def _imbalance_bucket_edges(values: np.ndarray, n_imb: int) -> np.ndarray:
    if n_imb <= 0:
        raise ValueError("n_imb must be positive.")
    clipped = np.clip(np.asarray(values, dtype=np.float64), -1.0, 1.0)
    quantiles = np.linspace(0.0, 1.0, n_imb + 1, dtype=np.float64)
    edges = np.quantile(clipped, quantiles)
    edges = np.asarray(edges, dtype=np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _state_index_arrays(
    spread_ticks: np.ndarray,
    depth_imbalance: np.ndarray,
    n_spread: int,
    imbalance_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_spread <= 0:
        raise ValueError("n_spread must be positive.")
    if imbalance_edges.ndim != 1 or imbalance_edges.size < 2:
        raise ValueError("imbalance_edges must be one-dimensional with at least two entries.")
    spread_bucket = np.clip(np.asarray(spread_ticks, dtype=np.int64), 1, n_spread) - 1
    imbalance_bucket = np.searchsorted(
        imbalance_edges[1:-1],
        np.clip(np.asarray(depth_imbalance, dtype=np.float64), -1.0, 1.0),
        side="right",
    )
    imbalance_bucket = np.clip(imbalance_bucket, 0, imbalance_edges.size - 2)
    state_index = spread_bucket * (imbalance_edges.size - 1) + imbalance_bucket
    return spread_bucket, imbalance_bucket, state_index.astype(np.int64, copy=False)


def _evaluate_predicted_adjustment(
    feature_frame: pd.DataFrame,
    predicted_price: pd.Series,
    tick_size: float,
    buckets: int = DEFAULT_DECILE_BUCKETS,
    threshold_percentiles: Sequence[int | float] = DEFAULT_THRESHOLD_PERCENTILES,
) -> dict[str, Any]:
    predicted_ticks = predicted_price.astype(float) / tick_size
    target_price = feature_frame["future_return"].astype(float)
    target_ticks = feature_frame["future_return_ticks"].astype(float)
    tick_errors = predicted_ticks.to_numpy(dtype=np.float64) - target_ticks.to_numpy(dtype=np.float64)
    alpha_summary = summarize_multilevel_alpha_signal(
        predicted_ticks,
        target_ticks,
        buckets=buckets,
        threshold_percentiles=threshold_percentiles,
    )
    return {
        "mse": float(np.mean(tick_errors * tick_errors) * (tick_size * tick_size)),
        "correlation": alpha_summary["pearson_correlation"],
        "mse_ticks": float(np.mean(tick_errors * tick_errors)),
        "mae_ticks": float(np.mean(np.abs(tick_errors))),
        "rmse_ticks": float(np.sqrt(np.mean(tick_errors * tick_errors))),
        "pearson": alpha_summary["pearson_correlation"],
        "spearman": alpha_summary["spearman_correlation"],
        "directional_accuracy": alpha_summary["directional_accuracy"],
        "coverage": alpha_summary["coverage"],
        "row_count": alpha_summary["row_count"],
        "deciles": alpha_summary["deciles"],
        "top_bottom_decile_spread_ticks": alpha_summary["top_bottom_decile_spread_ticks"],
        "decile_monotonicity_score": alpha_summary["decile_monotonicity_score"],
        "threshold_metrics": alpha_summary["threshold_metrics"],
    }


def _normalize_level_candidates(level_candidates: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(sorted({int(level) for level in level_candidates}))
    if not normalized:
        raise ValueError("level_candidates must not be empty.")
    if normalized[0] <= 0 or normalized[-1] > MAX_BOOK_LEVELS:
        raise ValueError(f"level_candidates must be between 1 and {MAX_BOOK_LEVELS}.")
    return normalized


def _normalize_decay_candidates(decay_lambda_candidates: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in decay_lambda_candidates)
    if not normalized:
        raise ValueError("decay_lambda_candidates must not be empty.")
    if not np.isfinite(np.asarray(normalized, dtype=np.float64)).all():
        raise ValueError("decay_lambda_candidates must contain only finite values.")
    if any(value < 0.0 for value in normalized):
        raise ValueError("decay_lambda_candidates must contain only nonnegative values.")
    return normalized


def _metric_or_default(value: float | None, default: float) -> float:
    return default if value is None or not np.isfinite(value) else float(value)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full_like(numerator, np.nan, dtype=np.float64)
    valid = np.isfinite(denominator) & (denominator > 0.0)
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def _depth_slope(level_sizes: np.ndarray) -> np.ndarray:
    if level_sizes.ndim != 2:
        raise ValueError("level_sizes must be a two-dimensional array.")
    if level_sizes.shape[1] <= 1:
        return np.zeros(level_sizes.shape[0], dtype=np.float64)
    levels = np.arange(1, level_sizes.shape[1] + 1, dtype=np.float64)
    centered_levels = levels - float(levels.mean())
    denominator = float(np.dot(centered_levels, centered_levels))
    return (level_sizes @ centered_levels) / denominator


def _multilevel_snapshot_feature_names(levels: int) -> list[str]:
    if levels <= 0:
        raise ValueError("levels must be positive.")
    names = ["raw_adjustment_ticks", "spread_ticks"]
    names.extend([f"imbalance_{level}" for level in range(1, levels + 1)])
    names.extend([f"cum_imbalance_{level}" for level in range(1, levels + 1)])
    if levels > 1:
        names.extend([f"bid_gap_{level}_ticks" for level in range(1, levels)])
        names.extend([f"ask_gap_{level}_ticks" for level in range(1, levels)])
    names.extend(
        [
            f"total_depth_imbalance_{levels}",
            "depth_slope_bid",
            "depth_slope_ask",
            "depth_slope_net",
        ]
    )
    return names


def prepare_multilevel_snapshot_feature_frame(
    df: pd.DataFrame,
    levels: int,
    decay_lambda: float,
    alpha_horizon: int = DEFAULT_ALPHA_HORIZON,
    tick_size: float | None = None,
    drop_invalid_rows: bool = True,
) -> tuple[pd.DataFrame, float]:
    """Build an explicit multilevel snapshot feature frame for linear alpha modeling."""
    feature_frame, resolved_tick_size = prepare_multilevel_feature_frame(
        df,
        levels=levels,
        decay_lambda=decay_lambda,
        dt=alpha_horizon,
        tick_size=tick_size,
        drop_invalid_rows=drop_invalid_rows,
    )

    bid_prices = feature_frame.loc[:, _price_columns("bid", levels)].to_numpy(dtype=np.float64, copy=False)
    bid_sizes = feature_frame.loc[:, _size_columns("bid", levels)].to_numpy(dtype=np.float64, copy=False)
    ask_prices = feature_frame.loc[:, _price_columns("ask", levels)].to_numpy(dtype=np.float64, copy=False)
    ask_sizes = feature_frame.loc[:, _size_columns("ask", levels)].to_numpy(dtype=np.float64, copy=False)

    spread_ticks = (ask_prices[:, 0] - bid_prices[:, 0]) / resolved_tick_size
    level_imbalance = _safe_ratio(bid_sizes - ask_sizes, bid_sizes + ask_sizes)
    cumulative_bid_sizes = np.cumsum(bid_sizes, axis=1)
    cumulative_ask_sizes = np.cumsum(ask_sizes, axis=1)
    cumulative_imbalance = _safe_ratio(
        cumulative_bid_sizes - cumulative_ask_sizes,
        cumulative_bid_sizes + cumulative_ask_sizes,
    )
    total_depth_imbalance = cumulative_imbalance[:, -1]

    feature_frame["spread_ticks"] = spread_ticks
    for level in range(1, levels + 1):
        feature_frame[f"imbalance_{level}"] = level_imbalance[:, level - 1]
        feature_frame[f"cum_imbalance_{level}"] = cumulative_imbalance[:, level - 1]
    if levels > 1:
        bid_gaps = (bid_prices[:, :-1] - bid_prices[:, 1:]) / resolved_tick_size
        ask_gaps = (ask_prices[:, 1:] - ask_prices[:, :-1]) / resolved_tick_size
        for level in range(1, levels):
            feature_frame[f"bid_gap_{level}_ticks"] = bid_gaps[:, level - 1]
            feature_frame[f"ask_gap_{level}_ticks"] = ask_gaps[:, level - 1]
    feature_frame[f"total_depth_imbalance_{levels}"] = total_depth_imbalance
    feature_frame["depth_slope_bid"] = _depth_slope(bid_sizes)
    feature_frame["depth_slope_ask"] = _depth_slope(ask_sizes)
    feature_frame["depth_slope_net"] = (
        feature_frame["depth_slope_bid"] - feature_frame["depth_slope_ask"]
    )
    return feature_frame, resolved_tick_size


def _prepare_feature_matrices(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[float], list[float]]:
    train_matrix = train_frame.loc[:, feature_names].to_numpy(dtype=np.float64, copy=False)
    validation_matrix = validation_frame.loc[:, feature_names].to_numpy(dtype=np.float64, copy=False)
    train_means = train_matrix.mean(axis=0)
    train_stds = train_matrix.std(axis=0)
    keep_mask = np.isfinite(train_means) & np.isfinite(train_stds) & (train_stds > 0.0)
    kept_names = [str(name) for name, keep in zip(feature_names, keep_mask) if keep]
    dropped_names = [str(name) for name, keep in zip(feature_names, keep_mask) if not keep]

    if bool(keep_mask.any()):
        kept_means = train_means[keep_mask]
        kept_stds = train_stds[keep_mask]
        train_matrix = (train_matrix[:, keep_mask] - kept_means) / kept_stds
        validation_matrix = (validation_matrix[:, keep_mask] - kept_means) / kept_stds
        return (
            train_matrix,
            validation_matrix,
            kept_names,
            dropped_names,
            kept_means.tolist(),
            kept_stds.tolist(),
        )

    return (
        np.zeros((len(train_frame), 0), dtype=np.float64),
        np.zeros((len(validation_frame), 0), dtype=np.float64),
        [],
        list(feature_names),
        [],
        [],
    )


def _fit_linear_feature_model(
    x: np.ndarray,
    y: np.ndarray,
    ridge_alpha: float = 0.0,
    fit_intercept: bool = True,
) -> tuple[float, np.ndarray]:
    if x.ndim != 2:
        raise ValueError("x must be two-dimensional.")
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows.")
    if x.shape[0] == 0:
        raise ValueError("x and y must not be empty.")
    alpha = float(ridge_alpha)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("ridge_alpha must be finite and nonnegative.")

    if x.shape[1] == 0:
        return (float(y.mean()) if fit_intercept else 0.0), np.zeros(0, dtype=np.float64)

    if fit_intercept:
        x_mean = x.mean(axis=0)
        y_mean = float(y.mean())
        x_centered = x - x_mean
        y_centered = y - y_mean
        gram = x_centered.T @ x_centered
        if alpha > 0.0:
            gram = gram + alpha * np.eye(x.shape[1], dtype=np.float64)
        rhs = x_centered.T @ y_centered
        if x.shape[1] > 0:
            try:
                coefficients = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                coefficients = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        else:
            coefficients = np.zeros(0, dtype=np.float64)
        intercept = y_mean - float(x_mean @ coefficients)
        return intercept, coefficients

    gram = x.T @ x
    if alpha > 0.0:
        gram = gram + alpha * np.eye(x.shape[1], dtype=np.float64)
    rhs = x.T @ y
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return 0.0, coefficients


def _predict_linear_feature_model(
    x: np.ndarray,
    intercept: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("x must be two-dimensional.")
    if x.shape[1] != coefficients.shape[0]:
        raise ValueError("x column count must match coefficient count.")
    return intercept + (x @ coefficients)


def _evaluate_tick_predictions(
    predicted_ticks: np.ndarray,
    target_ticks: np.ndarray,
    buckets: int = DEFAULT_DECILE_BUCKETS,
    threshold_percentiles: Sequence[int | float] = DEFAULT_THRESHOLD_PERCENTILES,
) -> dict[str, Any]:
    errors = predicted_ticks - target_ticks
    alpha_summary = summarize_multilevel_alpha_signal(
        pd.Series(predicted_ticks, dtype=float),
        pd.Series(target_ticks, dtype=float),
        buckets=buckets,
        threshold_percentiles=threshold_percentiles,
    )
    return {
        "mse_ticks": float(np.mean(errors * errors)),
        "mae_ticks": float(np.mean(np.abs(errors))),
        "rmse_ticks": float(np.sqrt(np.mean(errors * errors))),
        "pearson": alpha_summary["pearson_correlation"],
        "spearman": alpha_summary["spearman_correlation"],
        "directional_accuracy": alpha_summary["directional_accuracy"],
        "coverage": alpha_summary["coverage"],
        "row_count": alpha_summary["row_count"],
        "deciles": alpha_summary["deciles"],
        "top_bottom_decile_spread_ticks": alpha_summary["top_bottom_decile_spread_ticks"],
        "decile_monotonicity_score": alpha_summary["decile_monotonicity_score"],
        "threshold_metrics": alpha_summary["threshold_metrics"],
    }


def _multilevel_to_l1_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": frame["time"].to_numpy(dtype=np.float64, copy=False),
            "bid": frame["bid_px_1"].to_numpy(dtype=np.float64, copy=False),
            "ask": frame["ask_px_1"].to_numpy(dtype=np.float64, copy=False),
            "bs": frame["bid_sz_1"].to_numpy(dtype=np.float64, copy=False),
            "as": frame["ask_sz_1"].to_numpy(dtype=np.float64, copy=False),
        }
    )


def _l1_summary_with_tick_error(
    signal: pd.Series,
    future_return: pd.Series,
    tick_size: float,
    buckets: int,
) -> dict[str, Any]:
    summary = summarize_alpha_signal(signal, future_return, buckets=buckets)
    valid = signal.notna() & future_return.notna()
    signal_ticks = signal.loc[valid].astype(float).to_numpy(dtype=np.float64) / tick_size
    target_ticks = future_return.loc[valid].astype(float).to_numpy(dtype=np.float64) / tick_size
    if signal_ticks.size == 0:
        summary["rmse_ticks"] = None
        summary["mae_ticks"] = None
        summary["top_bottom_decile_spread_ticks"] = None
        return summary
    errors = signal_ticks - target_ticks
    summary["rmse_ticks"] = float(np.sqrt(np.mean(errors * errors)))
    summary["mae_ticks"] = float(np.mean(np.abs(errors)))
    summary["top_bottom_decile_spread_ticks"] = _decile_spread(summary["deciles"])
    return summary


def _evaluate_l1_baselines_from_multilevel(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    alpha_horizon: int,
    tick_size: float,
    buckets: int = DEFAULT_DECILE_BUCKETS,
    dt: int = 1,
    max_horizon: int = 6,
) -> dict[str, Any]:
    l1_train = _multilevel_to_l1_frame(train_frame)
    l1_validation = _multilevel_to_l1_frame(validation_frame)
    model = fit_from_dataframe(l1_train, dt=dt, max_horizon=max_horizon)
    evaluation_frame = prepare_alpha_evaluation_frame(l1_validation, horizon=alpha_horizon)
    signals = build_alpha_signal_frame(evaluation_frame, model)
    summaries = []
    for column in signals.columns:
        summary = _l1_summary_with_tick_error(
            signals[column],
            evaluation_frame["future_return"],
            tick_size=tick_size,
            buckets=buckets,
        )
        summary["name"] = column
        summaries.append(summary)

    best_spearman = max((item for item in summaries if item["spearman_correlation"] is not None), key=lambda item: item["spearman_correlation"])
    best_decile_spread = max(
        (item for item in summaries if item["top_bottom_decile_spread_ticks"] is not None),
        key=lambda item: item["top_bottom_decile_spread_ticks"],
    )
    best_rmse = min((item for item in summaries if item["rmse_ticks"] is not None), key=lambda item: item["rmse_ticks"])
    return {
        "signal_summaries": summaries,
        "best_spearman_signal": best_spearman["name"],
        "best_spearman": float(best_spearman["spearman_correlation"]),
        "best_decile_spread_signal": best_decile_spread["name"],
        "best_decile_spread_ticks": float(best_decile_spread["top_bottom_decile_spread_ticks"]),
        "best_rmse_signal": best_rmse["name"],
        "best_rmse_ticks": float(best_rmse["rmse_ticks"]),
    }


@dataclass
class FittedMultilevelMicropriceModel:
    """Calibrated multilevel microprice model for alpha-style midpoint forecasting."""

    tick_size: float
    levels: int
    decay_lambda: float
    dt: int
    intercept: float
    slope: float
    train_mse: float
    train_correlation: float
    validation_mse: float
    validation_correlation: float
    calibration_units: CalibrationUnits = "ticks"
    fit_intercept: bool = True
    alpha_horizon: int | None = None
    train_mse_ticks: float | None = None
    train_mae_ticks: float | None = None
    train_rmse_ticks: float | None = None
    train_pearson: float | None = None
    train_spearman: float | None = None
    train_directional_accuracy: float | None = None
    validation_mse_ticks: float | None = None
    validation_mae_ticks: float | None = None
    validation_rmse_ticks: float | None = None
    validation_pearson: float | None = None
    validation_spearman: float | None = None
    validation_directional_accuracy: float | None = None
    validation_deciles: list[dict[str, Any]] = field(default_factory=list)
    validation_threshold_metrics: list[dict[str, Any]] = field(default_factory=list)
    top_bottom_decile_spread_ticks: float | None = None
    decile_monotonicity_score: float | None = None
    _estimator: MultilevelMicroprice = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.tick_size = _validate_explicit_tick_size(self.tick_size)
        self.levels = int(self.levels)
        self.decay_lambda = float(self.decay_lambda)
        self.dt = int(self.dt)
        self.intercept = float(self.intercept)
        self.slope = float(self.slope)
        self.train_mse = float(self.train_mse)
        self.train_correlation = float(self.train_correlation)
        self.validation_mse = float(self.validation_mse)
        self.validation_correlation = float(self.validation_correlation)
        self.fit_intercept = bool(self.fit_intercept)
        self.alpha_horizon = self.dt if self.alpha_horizon is None else int(self.alpha_horizon)
        if self.levels <= 0:
            raise ValueError("levels must be positive.")
        if not np.isfinite(self.decay_lambda) or self.decay_lambda < 0.0:
            raise ValueError("decay_lambda must be finite and nonnegative.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if self.alpha_horizon <= 0:
            raise ValueError("alpha_horizon must be positive.")
        if self.calibration_units not in ("price", "ticks"):
            raise ValueError("calibration_units must be 'price' or 'ticks'.")

        self.train_mse_ticks = self.train_mse if self.train_mse_ticks is None else float(self.train_mse_ticks)
        self.train_mae_ticks = None if self.train_mae_ticks is None else float(self.train_mae_ticks)
        self.train_rmse_ticks = None if self.train_rmse_ticks is None else float(self.train_rmse_ticks)
        self.train_pearson = self.train_correlation if self.train_pearson is None else float(self.train_pearson)
        self.train_spearman = None if self.train_spearman is None else float(self.train_spearman)
        self.train_directional_accuracy = (
            None if self.train_directional_accuracy is None else float(self.train_directional_accuracy)
        )
        self.validation_mse_ticks = (
            self.validation_mse if self.validation_mse_ticks is None else float(self.validation_mse_ticks)
        )
        self.validation_mae_ticks = None if self.validation_mae_ticks is None else float(self.validation_mae_ticks)
        self.validation_rmse_ticks = (
            None if self.validation_rmse_ticks is None else float(self.validation_rmse_ticks)
        )
        self.validation_pearson = (
            self.validation_correlation if self.validation_pearson is None else float(self.validation_pearson)
        )
        self.validation_spearman = (
            None if self.validation_spearman is None else float(self.validation_spearman)
        )
        self.validation_directional_accuracy = (
            None if self.validation_directional_accuracy is None else float(self.validation_directional_accuracy)
        )
        self.top_bottom_decile_spread_ticks = (
            None if self.top_bottom_decile_spread_ticks is None else float(self.top_bottom_decile_spread_ticks)
        )
        self.decile_monotonicity_score = (
            None if self.decile_monotonicity_score is None else float(self.decile_monotonicity_score)
        )
        self._estimator = MultilevelMicroprice(tick_size=self.tick_size, decay_lambda=self.decay_lambda)

    def raw_microprice_from_book(
        self,
        bid_prices: Sequence[float],
        bid_sizes: Sequence[float],
        ask_prices: Sequence[float],
        ask_sizes: Sequence[float],
    ) -> float:
        return self._estimator(
            bid_prices=np.asarray(bid_prices, dtype=np.float64)[: self.levels],
            bid_sizes=np.asarray(bid_sizes, dtype=np.float64)[: self.levels],
            ask_prices=np.asarray(ask_prices, dtype=np.float64)[: self.levels],
            ask_sizes=np.asarray(ask_sizes, dtype=np.float64)[: self.levels],
        )

    def adjustment_from_book(
        self,
        bid_prices: Sequence[float],
        bid_sizes: Sequence[float],
        ask_prices: Sequence[float],
        ask_sizes: Sequence[float],
    ) -> float:
        bid_array = np.asarray(bid_prices, dtype=np.float64)[: self.levels]
        ask_array = np.asarray(ask_prices, dtype=np.float64)[: self.levels]
        mid = 0.5 * (bid_array[0] + ask_array[0])
        raw_adjustment = self.raw_microprice_from_book(
            bid_prices=bid_prices,
            bid_sizes=bid_sizes,
            ask_prices=ask_prices,
            ask_sizes=ask_sizes,
        ) - mid
        if self.calibration_units == "ticks":
            calibrated_ticks = self.intercept + self.slope * (raw_adjustment / self.tick_size)
            return calibrated_ticks * self.tick_size
        return self.intercept + self.slope * raw_adjustment

    def microprice_from_book(
        self,
        bid_prices: Sequence[float],
        bid_sizes: Sequence[float],
        ask_prices: Sequence[float],
        ask_sizes: Sequence[float],
    ) -> float:
        bid_array = np.asarray(bid_prices, dtype=np.float64)[: self.levels]
        ask_array = np.asarray(ask_prices, dtype=np.float64)[: self.levels]
        mid = 0.5 * (bid_array[0] + ask_array[0])
        return mid + self.adjustment_from_book(
            bid_prices=bid_prices,
            bid_sizes=bid_sizes,
            ask_prices=ask_prices,
            ask_sizes=ask_sizes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_kind": "multilevel",
            "tick_size": self.tick_size,
            "calibration_horizon": self.alpha_horizon,
            "levels": self.levels,
            "decay_lambda": self.decay_lambda,
            "dt": self.dt,
            "intercept": self.intercept,
            "slope": self.slope,
            "train_mse": self.train_mse,
            "train_correlation": self.train_correlation,
            "validation_mse": self.validation_mse,
            "validation_correlation": self.validation_correlation,
            "calibration_units": self.calibration_units,
            "fit_intercept": self.fit_intercept,
            "alpha_horizon": self.alpha_horizon,
            "train_mse_ticks": self.train_mse_ticks,
            "train_mae_ticks": self.train_mae_ticks,
            "train_rmse_ticks": self.train_rmse_ticks,
            "train_pearson": self.train_pearson,
            "train_spearman": self.train_spearman,
            "train_directional_accuracy": self.train_directional_accuracy,
            "validation_mse_ticks": self.validation_mse_ticks,
            "validation_mae_ticks": self.validation_mae_ticks,
            "validation_rmse_ticks": self.validation_rmse_ticks,
            "validation_pearson": self.validation_pearson,
            "validation_spearman": self.validation_spearman,
            "validation_directional_accuracy": self.validation_directional_accuracy,
            "validation_deciles": self.validation_deciles,
            "validation_threshold_metrics": self.validation_threshold_metrics,
            "top_bottom_decile_spread_ticks": self.top_bottom_decile_spread_ticks,
            "decile_monotonicity_score": self.decile_monotonicity_score,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FittedMultilevelMicropriceModel":
        normalized = dict(payload)
        normalized.pop("model_kind", None)
        normalized.pop("calibration_horizon", None)
        return cls(**normalized)

    def save_model(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def load_multilevel_model(path: str | Path) -> FittedMultilevelMicropriceModel:
    return FittedMultilevelMicropriceModel.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_multilevel_microprice_model(
    df: pd.DataFrame,
    level_candidates: Sequence[int],
    decay_lambda_candidates: Sequence[float],
    dt: int = DEFAULT_CALIBRATION_HORIZON,
    train_fraction: float = 0.8,
    tick_size: float | None = None,
    calibration_units: CalibrationUnits = "ticks",
    fit_intercept: bool = True,
    purge_rows: int | None = None,
    drop_invalid_rows: bool = True,
    buckets: int = DEFAULT_DECILE_BUCKETS,
    threshold_percentiles: Sequence[int | float] = DEFAULT_THRESHOLD_PERCENTILES,
) -> FittedMultilevelMicropriceModel:
    """Fit a calibrated multilevel microprice model for alpha use."""
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if calibration_units not in ("price", "ticks"):
        raise ValueError("calibration_units must be 'price' or 'ticks'.")

    levels_grid = _normalize_level_candidates(level_candidates)
    decay_grid = _normalize_decay_candidates(decay_lambda_candidates)
    max_levels = max(levels_grid)
    normalized = _normalize_multilevel_input(df, levels=max_levels, drop_invalid_rows=drop_invalid_rows)
    resolved_tick_size = _resolve_tick_size(normalized, levels=max_levels, tick_size=tick_size)
    resolved_purge_rows = dt if purge_rows is None else int(purge_rows)
    train_frame, validation_frame = _split_train_validation(
        normalized,
        train_fraction=train_fraction,
        purge_rows=resolved_purge_rows,
    )
    best_result: dict[str, Any] | None = None
    skipped_nonpositive_spearman = 0

    for levels in levels_grid:
        train_view = train_frame.loc[:, _level_columns(levels)].copy()
        validation_view = validation_frame.loc[:, _level_columns(levels)].copy()
        for decay_lambda in decay_grid:
            prepared_train, _ = prepare_multilevel_feature_frame(
                train_view,
                levels=levels,
                decay_lambda=decay_lambda,
                dt=dt,
                tick_size=resolved_tick_size,
                drop_invalid_rows=drop_invalid_rows,
            )
            prepared_validation, _ = prepare_multilevel_feature_frame(
                validation_view,
                levels=levels,
                decay_lambda=decay_lambda,
                dt=dt,
                tick_size=resolved_tick_size,
                drop_invalid_rows=drop_invalid_rows,
            )
            train_x, train_y = _calibration_arrays(
                prepared_train,
                tick_size=resolved_tick_size,
                calibration_units=calibration_units,
            )
            intercept, slope = _fit_linear_adjustment(train_x, train_y, fit_intercept=fit_intercept)
            train_metrics = _evaluate_calibrated_signal(
                prepared_train,
                tick_size=resolved_tick_size,
                intercept=intercept,
                slope=slope,
                calibration_units=calibration_units,
                buckets=buckets,
                threshold_percentiles=threshold_percentiles,
            )
            validation_metrics = _evaluate_calibrated_signal(
                prepared_validation,
                tick_size=resolved_tick_size,
                intercept=intercept,
                slope=slope,
                calibration_units=calibration_units,
                buckets=buckets,
                threshold_percentiles=threshold_percentiles,
            )
            validation_spearman = validation_metrics["spearman"]
            if validation_spearman is None or validation_spearman <= 0.0:
                skipped_nonpositive_spearman += 1
                continue

            result = {
                "levels": levels,
                "decay_lambda": float(decay_lambda),
                "intercept": float(intercept),
                "slope": float(slope),
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
            }
            if best_result is None:
                best_result = result
                continue

            current_key = (
                -_metric_or_default(validation_metrics["spearman"], float("-inf")),
                -_metric_or_default(validation_metrics["top_bottom_decile_spread_ticks"], float("-inf")),
                -_metric_or_default(validation_metrics["directional_accuracy"], float("-inf")),
                float(validation_metrics["rmse_ticks"]),
                result["levels"],
                result["decay_lambda"],
            )
            best_validation_metrics = best_result["validation_metrics"]
            best_key = (
                -_metric_or_default(best_validation_metrics["spearman"], float("-inf")),
                -_metric_or_default(best_validation_metrics["top_bottom_decile_spread_ticks"], float("-inf")),
                -_metric_or_default(best_validation_metrics["directional_accuracy"], float("-inf")),
                float(best_validation_metrics["rmse_ticks"]),
                best_result["levels"],
                best_result["decay_lambda"],
            )
            if current_key < best_key:
                best_result = result

    if best_result is None:
        raise ValueError(
            "Unable to fit a multilevel calibration model with positive validation Spearman. "
            f"Skipped {skipped_nonpositive_spearman} non-predictive candidate configurations."
        )

    full_view = normalized.loc[:, _level_columns(int(best_result["levels"]))].copy()
    prepared_full, _ = prepare_multilevel_feature_frame(
        full_view,
        levels=int(best_result["levels"]),
        decay_lambda=float(best_result["decay_lambda"]),
        dt=dt,
        tick_size=resolved_tick_size,
        drop_invalid_rows=drop_invalid_rows,
    )
    full_x, full_y = _calibration_arrays(
        prepared_full,
        tick_size=resolved_tick_size,
        calibration_units=calibration_units,
    )
    intercept, slope = _fit_linear_adjustment(full_x, full_y, fit_intercept=fit_intercept)

    train_metrics = best_result["train_metrics"]
    validation_metrics = best_result["validation_metrics"]
    return FittedMultilevelMicropriceModel(
        tick_size=resolved_tick_size,
        levels=int(best_result["levels"]),
        decay_lambda=float(best_result["decay_lambda"]),
        dt=dt,
        intercept=intercept,
        slope=slope,
        train_mse=float(train_metrics["mse"]),
        train_correlation=float(train_metrics["pearson"] or 0.0),
        validation_mse=float(validation_metrics["mse"]),
        validation_correlation=float(validation_metrics["pearson"] or 0.0),
        calibration_units=calibration_units,
        fit_intercept=fit_intercept,
        alpha_horizon=dt,
        train_mse_ticks=float(train_metrics["mse_ticks"]),
        train_mae_ticks=float(train_metrics["mae_ticks"]),
        train_rmse_ticks=float(train_metrics["rmse_ticks"]),
        train_pearson=train_metrics["pearson"],
        train_spearman=train_metrics["spearman"],
        train_directional_accuracy=train_metrics["directional_accuracy"],
        validation_mse_ticks=float(validation_metrics["mse_ticks"]),
        validation_mae_ticks=float(validation_metrics["mae_ticks"]),
        validation_rmse_ticks=float(validation_metrics["rmse_ticks"]),
        validation_pearson=validation_metrics["pearson"],
        validation_spearman=validation_metrics["spearman"],
        validation_directional_accuracy=validation_metrics["directional_accuracy"],
        validation_deciles=list(validation_metrics["deciles"]),
        validation_threshold_metrics=list(validation_metrics["threshold_metrics"]),
        top_bottom_decile_spread_ticks=validation_metrics["top_bottom_decile_spread_ticks"],
        decile_monotonicity_score=validation_metrics["decile_monotonicity_score"],
    )


def evaluate_multilevel_microprice_model(
    df: pd.DataFrame,
    model: FittedMultilevelMicropriceModel | None = None,
    level_candidates: Sequence[int] = (1, 3, 5),
    decay_lambda_candidates: Sequence[float] = (0.0, 0.1, 0.3, 1.0, 3.0, 5.0),
    train_fraction: float = 0.8,
    horizon: int = DEFAULT_CALIBRATION_HORIZON,
    tick_size: float | None = None,
    calibration_units: CalibrationUnits = "ticks",
    fit_intercept: bool = True,
    purge_rows: int | None = None,
    drop_invalid_rows: bool = True,
    buckets: int = DEFAULT_DECILE_BUCKETS,
    threshold_percentiles: Sequence[int | float] = DEFAULT_THRESHOLD_PERCENTILES,
) -> dict[str, Any]:
    max_levels = max(_normalize_level_candidates(level_candidates)) if model is None else int(model.levels)
    normalized = _normalize_multilevel_input(df, levels=max_levels, drop_invalid_rows=drop_invalid_rows)
    resolved_tick_size = _resolve_tick_size(normalized, levels=max_levels, tick_size=tick_size)
    resolved_purge_rows = horizon if purge_rows is None else int(purge_rows)
    train_frame, validation_frame = _split_train_validation(
        normalized,
        train_fraction=train_fraction,
        purge_rows=resolved_purge_rows,
    )
    fitted_model = model or fit_multilevel_microprice_model(
        train_frame,
        level_candidates=level_candidates,
        decay_lambda_candidates=decay_lambda_candidates,
        dt=horizon,
        train_fraction=0.8,
        tick_size=resolved_tick_size,
        calibration_units=calibration_units,
        fit_intercept=fit_intercept,
        purge_rows=0,
        drop_invalid_rows=drop_invalid_rows,
        buckets=buckets,
        threshold_percentiles=threshold_percentiles,
    )
    prepared_validation, _ = prepare_multilevel_feature_frame(
        validation_frame.loc[:, _level_columns(int(fitted_model.levels))].copy(),
        levels=int(fitted_model.levels),
        decay_lambda=float(fitted_model.decay_lambda),
        dt=horizon,
        tick_size=resolved_tick_size,
        drop_invalid_rows=drop_invalid_rows,
    )
    raw_signal = prepared_validation["raw_adjustment"].astype(float)
    _bid_px = prepared_validation[_price_columns("bid", fitted_model.levels)].to_numpy(dtype=np.float64)
    _bid_sz = prepared_validation[_size_columns("bid", fitted_model.levels)].to_numpy(dtype=np.float64)
    _ask_px = prepared_validation[_price_columns("ask", fitted_model.levels)].to_numpy(dtype=np.float64)
    _ask_sz = prepared_validation[_size_columns("ask", fitted_model.levels)].to_numpy(dtype=np.float64)
    _mids_batch, _raw_batch = raw_multilevel_microprice_batch(
        bid_prices=_bid_px, bid_sizes=_bid_sz,
        ask_prices=_ask_px, ask_sizes=_ask_sz,
        tick_size=fitted_model.tick_size, decay_lambda=fitted_model.decay_lambda,
    )
    _raw_adj = _raw_batch - _mids_batch
    if fitted_model.calibration_units == "ticks":
        _calibrated = (fitted_model.intercept + fitted_model.slope * (_raw_adj / fitted_model.tick_size)) * fitted_model.tick_size
    else:
        _calibrated = fitted_model.intercept + fitted_model.slope * _raw_adj
    calibrated_signal = pd.Series(_calibrated, index=prepared_validation.index, dtype=float)
    target = prepared_validation["future_return"].astype(float)
    return {
        "model_kind": "multilevel",
        "tick_size": float(fitted_model.tick_size),
        "horizon": int(horizon),
        "train_row_count": int(len(train_frame)),
        "validation_row_count": int(len(validation_frame)),
        "target_row_count": int(target.notna().sum()),
        "signals": [
            {
                "name": "raw_multilevel_microprice",
                **summarize_signal_performance(
                    raw_signal,
                    target,
                    tick_size=fitted_model.tick_size,
                    buckets=buckets,
                    threshold_percentiles=tuple(threshold_percentiles),
                ),
            },
            {
                "name": "calibrated_multilevel_microprice",
                **summarize_signal_performance(
                    calibrated_signal,
                    target,
                    tick_size=fitted_model.tick_size,
                    buckets=buckets,
                    threshold_percentiles=tuple(threshold_percentiles),
                ),
            },
        ],
        "model": fitted_model.to_dict(),
    }


def compare_l1_and_multilevel_microprices(
    df: pd.DataFrame,
    l1_model: Any | None = None,
    multilevel_model: FittedMultilevelMicropriceModel | None = None,
    train_fraction: float = 0.8,
    horizon: int = DEFAULT_CALIBRATION_HORIZON,
    n_imb: int = 10,
    n_spread: int = 2,
    dt: int = 1,
    max_horizon: int = 6,
    level_candidates: Sequence[int] = (1, 3, 5),
    decay_lambda_candidates: Sequence[float] = (0.0, 0.1, 0.3, 1.0, 3.0, 5.0),
    tick_size: float | None = None,
    buckets: int = DEFAULT_DECILE_BUCKETS,
    threshold_percentiles: Sequence[int | float] = DEFAULT_THRESHOLD_PERCENTILES,
    purge_rows: int | None = None,
    drop_invalid_rows: bool = True,
) -> dict[str, Any]:
    max_levels = max(_normalize_level_candidates(level_candidates)) if multilevel_model is None else int(multilevel_model.levels)
    normalized = _normalize_multilevel_input(df, levels=max_levels, drop_invalid_rows=drop_invalid_rows)
    resolved_tick_size = _resolve_tick_size(normalized, levels=max_levels, tick_size=tick_size)
    l1_frame = _multilevel_to_l1_frame(normalized)
    l1_report = run_l1_microprice_evaluation(
        l1_frame,
        model=l1_model,
        train_fraction=train_fraction,
        horizon=horizon,
        n_imb=n_imb,
        n_spread=n_spread,
        dt=dt,
        max_horizon=max_horizon,
        tick_size=resolved_tick_size,
        buckets=buckets,
        threshold_percentiles=tuple(threshold_percentiles),
        purge_rows=horizon if purge_rows is None else int(purge_rows),
    )
    multilevel_report = evaluate_multilevel_microprice_model(
        normalized,
        model=multilevel_model,
        level_candidates=level_candidates,
        decay_lambda_candidates=decay_lambda_candidates,
        train_fraction=train_fraction,
        horizon=horizon,
        tick_size=resolved_tick_size,
        calibration_units="ticks",
        fit_intercept=True,
        purge_rows=horizon if purge_rows is None else int(purge_rows),
        drop_invalid_rows=drop_invalid_rows,
        buckets=buckets,
        threshold_percentiles=threshold_percentiles,
    )
    return {
        "tick_size": float(resolved_tick_size),
        "horizon": int(horizon),
        "l1": l1_report,
        "multilevel": multilevel_report,
    }


def _fit_state_conditioned_parameters(
    x: np.ndarray,
    y: np.ndarray,
    state_index: np.ndarray,
    state_count: int,
    fit_intercept: bool,
    min_state_rows: int,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    fallback_intercept, fallback_slope = _fit_linear_adjustment(x, y, fit_intercept=fit_intercept)
    state_intercepts = np.full(state_count, fallback_intercept, dtype=np.float64)
    state_slopes = np.full(state_count, fallback_slope, dtype=np.float64)
    state_rows = np.zeros(state_count, dtype=np.int64)
    for current_state in range(state_count):
        mask = state_index == current_state
        row_count = int(mask.sum())
        state_rows[current_state] = row_count
        if row_count < min_state_rows:
            continue
        intercept, slope = _fit_linear_adjustment(x[mask], y[mask], fit_intercept=fit_intercept)
        state_intercepts[current_state] = intercept
        state_slopes[current_state] = slope
    return fallback_intercept, fallback_slope, state_intercepts, state_slopes, state_rows


def _predict_state_conditioned_adjustment(
    x: np.ndarray,
    state_index: np.ndarray,
    state_intercepts: np.ndarray,
    state_slopes: np.ndarray,
) -> np.ndarray:
    return state_intercepts[state_index] + state_slopes[state_index] * x


@dataclass
class FittedStateConditionedMultilevelMicropriceModel:
    tick_size: float
    levels: int
    decay_lambda: float
    dt: int
    n_imb: int
    n_spread: int
    imbalance_edges: list[float]
    state_intercepts: list[float]
    state_slopes: list[float]
    state_row_counts: list[int]
    fallback_intercept: float
    fallback_slope: float
    min_state_rows: int
    train_mse: float
    train_correlation: float
    validation_mse: float
    validation_correlation: float
    calibration_units: CalibrationUnits = "ticks"
    fit_intercept: bool = True
    alpha_horizon: int | None = None
    train_mse_ticks: float | None = None
    train_mae_ticks: float | None = None
    train_rmse_ticks: float | None = None
    train_pearson: float | None = None
    train_spearman: float | None = None
    train_directional_accuracy: float | None = None
    validation_mse_ticks: float | None = None
    validation_mae_ticks: float | None = None
    validation_rmse_ticks: float | None = None
    validation_pearson: float | None = None
    validation_spearman: float | None = None
    validation_directional_accuracy: float | None = None
    validation_deciles: list[dict[str, Any]] = field(default_factory=list)
    validation_threshold_metrics: list[dict[str, Any]] = field(default_factory=list)
    top_bottom_decile_spread_ticks: float | None = None
    decile_monotonicity_score: float | None = None
    _estimator: MultilevelMicroprice = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.tick_size = _validate_explicit_tick_size(self.tick_size)
        self.levels = int(self.levels)
        self.decay_lambda = float(self.decay_lambda)
        self.dt = int(self.dt)
        self.n_imb = int(self.n_imb)
        self.n_spread = int(self.n_spread)
        self.imbalance_edges = [float(value) for value in self.imbalance_edges]
        self.state_intercepts = [float(value) for value in self.state_intercepts]
        self.state_slopes = [float(value) for value in self.state_slopes]
        self.state_row_counts = [int(value) for value in self.state_row_counts]
        self.fallback_intercept = float(self.fallback_intercept)
        self.fallback_slope = float(self.fallback_slope)
        self.min_state_rows = int(self.min_state_rows)
        self.train_mse = float(self.train_mse)
        self.train_correlation = float(self.train_correlation)
        self.validation_mse = float(self.validation_mse)
        self.validation_correlation = float(self.validation_correlation)
        self.fit_intercept = bool(self.fit_intercept)
        self.alpha_horizon = self.dt if self.alpha_horizon is None else int(self.alpha_horizon)
        if self.levels <= 0:
            raise ValueError("levels must be positive.")
        if not np.isfinite(self.decay_lambda) or self.decay_lambda < 0.0:
            raise ValueError("decay_lambda must be finite and nonnegative.")
        if self.dt <= 0:
            raise ValueError("dt must be positive.")
        if self.alpha_horizon <= 0:
            raise ValueError("alpha_horizon must be positive.")
        if self.n_imb <= 0 or self.n_spread <= 0:
            raise ValueError("n_imb and n_spread must be positive.")
        if self.min_state_rows <= 0:
            raise ValueError("min_state_rows must be positive.")
        if self.calibration_units not in ("price", "ticks"):
            raise ValueError("calibration_units must be 'price' or 'ticks'.")
        expected_state_count = self.n_imb * self.n_spread
        if len(self.imbalance_edges) != self.n_imb + 1:
            raise ValueError("imbalance_edges must contain n_imb + 1 entries.")
        if len(self.state_intercepts) != expected_state_count:
            raise ValueError("state_intercepts must contain n_imb * n_spread entries.")
        if len(self.state_slopes) != expected_state_count:
            raise ValueError("state_slopes must contain n_imb * n_spread entries.")
        if len(self.state_row_counts) != expected_state_count:
            raise ValueError("state_row_counts must contain n_imb * n_spread entries.")
        self.train_mse_ticks = self.train_mse if self.train_mse_ticks is None else float(self.train_mse_ticks)
        self.train_mae_ticks = None if self.train_mae_ticks is None else float(self.train_mae_ticks)
        self.train_rmse_ticks = None if self.train_rmse_ticks is None else float(self.train_rmse_ticks)
        self.train_pearson = self.train_correlation if self.train_pearson is None else float(self.train_pearson)
        self.train_spearman = None if self.train_spearman is None else float(self.train_spearman)
        self.train_directional_accuracy = (
            None if self.train_directional_accuracy is None else float(self.train_directional_accuracy)
        )
        self.validation_mse_ticks = (
            self.validation_mse if self.validation_mse_ticks is None else float(self.validation_mse_ticks)
        )
        self.validation_mae_ticks = None if self.validation_mae_ticks is None else float(self.validation_mae_ticks)
        self.validation_rmse_ticks = (
            None if self.validation_rmse_ticks is None else float(self.validation_rmse_ticks)
        )
        self.validation_pearson = (
            self.validation_correlation if self.validation_pearson is None else float(self.validation_pearson)
        )
        self.validation_spearman = (
            None if self.validation_spearman is None else float(self.validation_spearman)
        )
        self.validation_directional_accuracy = (
            None if self.validation_directional_accuracy is None else float(self.validation_directional_accuracy)
        )
        self.top_bottom_decile_spread_ticks = (
            None if self.top_bottom_decile_spread_ticks is None else float(self.top_bottom_decile_spread_ticks)
        )
        self.decile_monotonicity_score = (
            None if self.decile_monotonicity_score is None else float(self.decile_monotonicity_score)
        )
        self._estimator = MultilevelMicroprice(tick_size=self.tick_size, decay_lambda=self.decay_lambda)

    def _state_index_from_book(
        self,
        bid_prices: np.ndarray,
        bid_sizes: np.ndarray,
        ask_prices: np.ndarray,
        ask_sizes: np.ndarray,
    ) -> int:
        spread_ticks = int(np.rint((ask_prices[0] - bid_prices[0]) / self.tick_size))
        total_bid = float(np.sum(bid_sizes))
        total_ask = float(np.sum(ask_sizes))
        denominator = total_bid + total_ask
        depth_imbalance = 0.0 if denominator <= 0.0 else float((total_bid - total_ask) / denominator)
        _, _, state_index = _state_index_arrays(
            np.asarray([spread_ticks], dtype=np.int64),
            np.asarray([depth_imbalance], dtype=np.float64),
            n_spread=self.n_spread,
            imbalance_edges=np.asarray(self.imbalance_edges, dtype=np.float64),
        )
        return int(state_index[0])

    def raw_microprice_from_book(
        self,
        bid_prices: Sequence[float],
        bid_sizes: Sequence[float],
        ask_prices: Sequence[float],
        ask_sizes: Sequence[float],
    ) -> float:
        return self._estimator(
            bid_prices=np.asarray(bid_prices, dtype=np.float64)[: self.levels],
            bid_sizes=np.asarray(bid_sizes, dtype=np.float64)[: self.levels],
            ask_prices=np.asarray(ask_prices, dtype=np.float64)[: self.levels],
            ask_sizes=np.asarray(ask_sizes, dtype=np.float64)[: self.levels],
        )

    def adjustment_from_book(
        self,
        bid_prices: Sequence[float],
        bid_sizes: Sequence[float],
        ask_prices: Sequence[float],
        ask_sizes: Sequence[float],
    ) -> float:
        bid_array = np.asarray(bid_prices, dtype=np.float64)[: self.levels]
        bid_size_array = np.asarray(bid_sizes, dtype=np.float64)[: self.levels]
        ask_array = np.asarray(ask_prices, dtype=np.float64)[: self.levels]
        ask_size_array = np.asarray(ask_sizes, dtype=np.float64)[: self.levels]
        mid = 0.5 * (bid_array[0] + ask_array[0])
        raw_adjustment = self.raw_microprice_from_book(
            bid_prices=bid_array,
            bid_sizes=bid_size_array,
            ask_prices=ask_array,
            ask_sizes=ask_size_array,
        ) - mid
        raw_value = raw_adjustment / self.tick_size if self.calibration_units == "ticks" else raw_adjustment
        state_index = self._state_index_from_book(bid_array, bid_size_array, ask_array, ask_size_array)
        predicted = self.state_intercepts[state_index] + self.state_slopes[state_index] * raw_value
        if self.calibration_units == "ticks":
            return predicted * self.tick_size
        return predicted

    def microprice_from_book(
        self,
        bid_prices: Sequence[float],
        bid_sizes: Sequence[float],
        ask_prices: Sequence[float],
        ask_sizes: Sequence[float],
    ) -> float:
        bid_array = np.asarray(bid_prices, dtype=np.float64)[: self.levels]
        ask_array = np.asarray(ask_prices, dtype=np.float64)[: self.levels]
        mid = 0.5 * (bid_array[0] + ask_array[0])
        return mid + self.adjustment_from_book(
            bid_prices=bid_prices,
            bid_sizes=bid_sizes,
            ask_prices=ask_prices,
            ask_sizes=ask_sizes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_kind": "multilevel_state_conditioned",
            "tick_size": self.tick_size,
            "calibration_horizon": self.alpha_horizon,
            "levels": self.levels,
            "decay_lambda": self.decay_lambda,
            "dt": self.dt,
            "n_imb": self.n_imb,
            "n_spread": self.n_spread,
            "imbalance_edges": self.imbalance_edges,
            "state_intercepts": self.state_intercepts,
            "state_slopes": self.state_slopes,
            "state_row_counts": self.state_row_counts,
            "fallback_intercept": self.fallback_intercept,
            "fallback_slope": self.fallback_slope,
            "min_state_rows": self.min_state_rows,
            "train_mse": self.train_mse,
            "train_correlation": self.train_correlation,
            "validation_mse": self.validation_mse,
            "validation_correlation": self.validation_correlation,
            "calibration_units": self.calibration_units,
            "fit_intercept": self.fit_intercept,
            "alpha_horizon": self.alpha_horizon,
            "train_mse_ticks": self.train_mse_ticks,
            "train_mae_ticks": self.train_mae_ticks,
            "train_rmse_ticks": self.train_rmse_ticks,
            "train_pearson": self.train_pearson,
            "train_spearman": self.train_spearman,
            "train_directional_accuracy": self.train_directional_accuracy,
            "validation_mse_ticks": self.validation_mse_ticks,
            "validation_mae_ticks": self.validation_mae_ticks,
            "validation_rmse_ticks": self.validation_rmse_ticks,
            "validation_pearson": self.validation_pearson,
            "validation_spearman": self.validation_spearman,
            "validation_directional_accuracy": self.validation_directional_accuracy,
            "validation_deciles": self.validation_deciles,
            "validation_threshold_metrics": self.validation_threshold_metrics,
            "top_bottom_decile_spread_ticks": self.top_bottom_decile_spread_ticks,
            "decile_monotonicity_score": self.decile_monotonicity_score,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FittedStateConditionedMultilevelMicropriceModel":
        normalized = dict(payload)
        normalized.pop("model_kind", None)
        normalized.pop("calibration_horizon", None)
        return cls(**normalized)

    def save_model(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def load_state_conditioned_multilevel_model(path: str | Path) -> FittedStateConditionedMultilevelMicropriceModel:
    return FittedStateConditionedMultilevelMicropriceModel.from_dict(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def fit_state_conditioned_multilevel_microprice_model(
    df: pd.DataFrame,
    level_candidates: Sequence[int],
    decay_lambda_candidates: Sequence[float],
    dt: int = DEFAULT_CALIBRATION_HORIZON,
    train_fraction: float = 0.8,
    tick_size: float | None = None,
    calibration_units: CalibrationUnits = "ticks",
    fit_intercept: bool = True,
    purge_rows: int | None = None,
    drop_invalid_rows: bool = True,
    buckets: int = DEFAULT_DECILE_BUCKETS,
    threshold_percentiles: Sequence[int | float] = DEFAULT_THRESHOLD_PERCENTILES,
    n_imb_candidates: Sequence[int] = (6, 8, 10),
    n_spread_candidates: Sequence[int] = (2, 3),
    min_state_rows: int = 200,
) -> FittedStateConditionedMultilevelMicropriceModel:
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if calibration_units not in ("price", "ticks"):
        raise ValueError("calibration_units must be 'price' or 'ticks'.")
    if min_state_rows <= 0:
        raise ValueError("min_state_rows must be positive.")

    levels_grid = _normalize_level_candidates(level_candidates)
    decay_grid = _normalize_decay_candidates(decay_lambda_candidates)
    n_imb_grid = tuple(sorted({int(value) for value in n_imb_candidates}))
    n_spread_grid = tuple(sorted({int(value) for value in n_spread_candidates}))
    if not n_imb_grid or min(n_imb_grid) <= 0:
        raise ValueError("n_imb_candidates must contain positive values.")
    if not n_spread_grid or min(n_spread_grid) <= 0:
        raise ValueError("n_spread_candidates must contain positive values.")

    max_levels = max(levels_grid)
    normalized = _normalize_multilevel_input(df, levels=max_levels, drop_invalid_rows=drop_invalid_rows)
    resolved_tick_size = _resolve_tick_size(normalized, levels=max_levels, tick_size=tick_size)
    resolved_purge_rows = dt if purge_rows is None else int(purge_rows)
    train_frame, validation_frame = _split_train_validation(
        normalized,
        train_fraction=train_fraction,
        purge_rows=resolved_purge_rows,
    )
    best_result: dict[str, Any] | None = None
    skipped_nonpositive_spearman = 0

    for levels in levels_grid:
        train_view = train_frame.loc[:, _level_columns(levels)].copy()
        validation_view = validation_frame.loc[:, _level_columns(levels)].copy()
        for decay_lambda in decay_grid:
            prepared_train, _ = prepare_multilevel_feature_frame(
                train_view,
                levels=levels,
                decay_lambda=decay_lambda,
                dt=dt,
                tick_size=resolved_tick_size,
                drop_invalid_rows=drop_invalid_rows,
            )
            prepared_validation, _ = prepare_multilevel_feature_frame(
                validation_view,
                levels=levels,
                decay_lambda=decay_lambda,
                dt=dt,
                tick_size=resolved_tick_size,
                drop_invalid_rows=drop_invalid_rows,
            )
            train_x, train_y = _calibration_arrays(
                prepared_train,
                tick_size=resolved_tick_size,
                calibration_units=calibration_units,
            )
            validation_x, _ = _calibration_arrays(
                prepared_validation,
                tick_size=resolved_tick_size,
                calibration_units=calibration_units,
            )
            train_spread_ticks, train_depth_imbalance = _state_conditioning_series(
                prepared_train,
                levels=levels,
                tick_size=resolved_tick_size,
            )
            validation_spread_ticks, validation_depth_imbalance = _state_conditioning_series(
                prepared_validation,
                levels=levels,
                tick_size=resolved_tick_size,
            )

            for n_spread in n_spread_grid:
                for n_imb in n_imb_grid:
                    imbalance_edges = _imbalance_bucket_edges(train_depth_imbalance, n_imb=n_imb)
                    _, _, train_state_index = _state_index_arrays(
                        train_spread_ticks,
                        train_depth_imbalance,
                        n_spread=n_spread,
                        imbalance_edges=imbalance_edges,
                    )
                    _, _, validation_state_index = _state_index_arrays(
                        validation_spread_ticks,
                        validation_depth_imbalance,
                        n_spread=n_spread,
                        imbalance_edges=imbalance_edges,
                    )
                    (
                        fallback_intercept,
                        fallback_slope,
                        state_intercepts,
                        state_slopes,
                        state_row_counts,
                    ) = _fit_state_conditioned_parameters(
                        train_x,
                        train_y,
                        train_state_index,
                        state_count=n_imb * n_spread,
                        fit_intercept=fit_intercept,
                        min_state_rows=min_state_rows,
                    )
                    predicted_train_values = _predict_state_conditioned_adjustment(
                        train_x,
                        train_state_index,
                        state_intercepts,
                        state_slopes,
                    )
                    predicted_validation_values = _predict_state_conditioned_adjustment(
                        validation_x,
                        validation_state_index,
                        state_intercepts,
                        state_slopes,
                    )
                    train_predicted_price = (
                        pd.Series(predicted_train_values, index=prepared_train.index, dtype=float) * resolved_tick_size
                        if calibration_units == "ticks"
                        else pd.Series(predicted_train_values, index=prepared_train.index, dtype=float)
                    )
                    validation_predicted_price = (
                        pd.Series(predicted_validation_values, index=prepared_validation.index, dtype=float)
                        * resolved_tick_size
                        if calibration_units == "ticks"
                        else pd.Series(predicted_validation_values, index=prepared_validation.index, dtype=float)
                    )
                    train_metrics = _evaluate_predicted_adjustment(
                        prepared_train,
                        train_predicted_price,
                        tick_size=resolved_tick_size,
                        buckets=buckets,
                        threshold_percentiles=threshold_percentiles,
                    )
                    validation_metrics = _evaluate_predicted_adjustment(
                        prepared_validation,
                        validation_predicted_price,
                        tick_size=resolved_tick_size,
                        buckets=buckets,
                        threshold_percentiles=threshold_percentiles,
                    )
                    validation_spearman = validation_metrics["spearman"]
                    if validation_spearman is None or validation_spearman <= 0.0:
                        skipped_nonpositive_spearman += 1
                        continue

                    result = {
                        "levels": levels,
                        "decay_lambda": float(decay_lambda),
                        "n_imb": n_imb,
                        "n_spread": n_spread,
                        "imbalance_edges": imbalance_edges.tolist(),
                        "fallback_intercept": float(fallback_intercept),
                        "fallback_slope": float(fallback_slope),
                        "state_intercepts": state_intercepts.tolist(),
                        "state_slopes": state_slopes.tolist(),
                        "state_row_counts": state_row_counts.astype(int).tolist(),
                        "train_metrics": train_metrics,
                        "validation_metrics": validation_metrics,
                    }
                    if best_result is None:
                        best_result = result
                        continue
                    current_key = (
                        -_metric_or_default(validation_metrics["spearman"], float("-inf")),
                        -_metric_or_default(validation_metrics["top_bottom_decile_spread_ticks"], float("-inf")),
                        -_metric_or_default(validation_metrics["directional_accuracy"], float("-inf")),
                        float(validation_metrics["rmse_ticks"]),
                        result["levels"],
                        result["decay_lambda"],
                        result["n_spread"],
                        result["n_imb"],
                    )
                    best_validation_metrics = best_result["validation_metrics"]
                    best_key = (
                        -_metric_or_default(best_validation_metrics["spearman"], float("-inf")),
                        -_metric_or_default(best_validation_metrics["top_bottom_decile_spread_ticks"], float("-inf")),
                        -_metric_or_default(best_validation_metrics["directional_accuracy"], float("-inf")),
                        float(best_validation_metrics["rmse_ticks"]),
                        best_result["levels"],
                        best_result["decay_lambda"],
                        best_result["n_spread"],
                        best_result["n_imb"],
                    )
                    if current_key < best_key:
                        best_result = result

    if best_result is None:
        raise ValueError(
            "Unable to fit a state-conditioned multilevel calibration model with positive validation Spearman. "
            f"Skipped {skipped_nonpositive_spearman} non-predictive candidate configurations."
        )

    full_view = normalized.loc[:, _level_columns(int(best_result["levels"]))].copy()
    prepared_full, _ = prepare_multilevel_feature_frame(
        full_view,
        levels=int(best_result["levels"]),
        decay_lambda=float(best_result["decay_lambda"]),
        dt=dt,
        tick_size=resolved_tick_size,
        drop_invalid_rows=drop_invalid_rows,
    )
    full_x, full_y = _calibration_arrays(
        prepared_full,
        tick_size=resolved_tick_size,
        calibration_units=calibration_units,
    )
    full_spread_ticks, full_depth_imbalance = _state_conditioning_series(
        prepared_full,
        levels=int(best_result["levels"]),
        tick_size=resolved_tick_size,
    )
    _, _, full_state_index = _state_index_arrays(
        full_spread_ticks,
        full_depth_imbalance,
        n_spread=int(best_result["n_spread"]),
        imbalance_edges=np.asarray(best_result["imbalance_edges"], dtype=np.float64),
    )
    (
        fallback_intercept,
        fallback_slope,
        state_intercepts,
        state_slopes,
        state_row_counts,
    ) = _fit_state_conditioned_parameters(
        full_x,
        full_y,
        full_state_index,
        state_count=int(best_result["n_imb"]) * int(best_result["n_spread"]),
        fit_intercept=fit_intercept,
        min_state_rows=min_state_rows,
    )

    train_metrics = best_result["train_metrics"]
    validation_metrics = best_result["validation_metrics"]
    return FittedStateConditionedMultilevelMicropriceModel(
        tick_size=resolved_tick_size,
        levels=int(best_result["levels"]),
        decay_lambda=float(best_result["decay_lambda"]),
        dt=dt,
        n_imb=int(best_result["n_imb"]),
        n_spread=int(best_result["n_spread"]),
        imbalance_edges=list(best_result["imbalance_edges"]),
        state_intercepts=state_intercepts.tolist(),
        state_slopes=state_slopes.tolist(),
        state_row_counts=state_row_counts.astype(int).tolist(),
        fallback_intercept=float(fallback_intercept),
        fallback_slope=float(fallback_slope),
        min_state_rows=int(min_state_rows),
        train_mse=float(train_metrics["mse"]),
        train_correlation=float(train_metrics["pearson"] or 0.0),
        validation_mse=float(validation_metrics["mse"]),
        validation_correlation=float(validation_metrics["pearson"] or 0.0),
        calibration_units=calibration_units,
        fit_intercept=fit_intercept,
        alpha_horizon=dt,
        train_mse_ticks=float(train_metrics["mse_ticks"]),
        train_mae_ticks=float(train_metrics["mae_ticks"]),
        train_rmse_ticks=float(train_metrics["rmse_ticks"]),
        train_pearson=train_metrics["pearson"],
        train_spearman=train_metrics["spearman"],
        train_directional_accuracy=train_metrics["directional_accuracy"],
        validation_mse_ticks=float(validation_metrics["mse_ticks"]),
        validation_mae_ticks=float(validation_metrics["mae_ticks"]),
        validation_rmse_ticks=float(validation_metrics["rmse_ticks"]),
        validation_pearson=validation_metrics["pearson"],
        validation_spearman=validation_metrics["spearman"],
        validation_directional_accuracy=validation_metrics["directional_accuracy"],
        validation_deciles=list(validation_metrics["deciles"]),
        validation_threshold_metrics=list(validation_metrics["threshold_metrics"]),
        top_bottom_decile_spread_ticks=validation_metrics["top_bottom_decile_spread_ticks"],
        decile_monotonicity_score=validation_metrics["decile_monotonicity_score"],
    )


def evaluate_state_conditioned_multilevel_microprice_model(
    df: pd.DataFrame,
    model: FittedStateConditionedMultilevelMicropriceModel | None = None,
    level_candidates: Sequence[int] = (1, 3, 5),
    decay_lambda_candidates: Sequence[float] = (0.0, 0.1, 0.3, 1.0, 3.0, 5.0),
    n_imb_candidates: Sequence[int] = (6, 8, 10),
    n_spread_candidates: Sequence[int] = (2, 3),
    min_state_rows: int = 200,
    train_fraction: float = 0.8,
    horizon: int = DEFAULT_CALIBRATION_HORIZON,
    tick_size: float | None = None,
    calibration_units: CalibrationUnits = "ticks",
    fit_intercept: bool = True,
    purge_rows: int | None = None,
    drop_invalid_rows: bool = True,
    buckets: int = DEFAULT_DECILE_BUCKETS,
    threshold_percentiles: Sequence[int | float] = DEFAULT_THRESHOLD_PERCENTILES,
) -> dict[str, Any]:
    max_levels = max(_normalize_level_candidates(level_candidates)) if model is None else int(model.levels)
    normalized = _normalize_multilevel_input(df, levels=max_levels, drop_invalid_rows=drop_invalid_rows)
    resolved_tick_size = _resolve_tick_size(normalized, levels=max_levels, tick_size=tick_size)
    resolved_purge_rows = horizon if purge_rows is None else int(purge_rows)
    train_frame, validation_frame = _split_train_validation(
        normalized,
        train_fraction=train_fraction,
        purge_rows=resolved_purge_rows,
    )
    fitted_model = model or fit_state_conditioned_multilevel_microprice_model(
        train_frame,
        level_candidates=level_candidates,
        decay_lambda_candidates=decay_lambda_candidates,
        dt=horizon,
        train_fraction=0.8,
        tick_size=resolved_tick_size,
        calibration_units=calibration_units,
        fit_intercept=fit_intercept,
        purge_rows=0,
        drop_invalid_rows=drop_invalid_rows,
        buckets=buckets,
        threshold_percentiles=threshold_percentiles,
        n_imb_candidates=n_imb_candidates,
        n_spread_candidates=n_spread_candidates,
        min_state_rows=min_state_rows,
    )
    prepared_validation, _ = prepare_multilevel_feature_frame(
        validation_frame.loc[:, _level_columns(int(fitted_model.levels))].copy(),
        levels=int(fitted_model.levels),
        decay_lambda=float(fitted_model.decay_lambda),
        dt=horizon,
        tick_size=resolved_tick_size,
        drop_invalid_rows=drop_invalid_rows,
    )
    raw_signal = prepared_validation["raw_adjustment"].astype(float)
    _sc_bid_px = prepared_validation[_price_columns("bid", fitted_model.levels)].to_numpy(dtype=np.float64)
    _sc_bid_sz = prepared_validation[_size_columns("bid", fitted_model.levels)].to_numpy(dtype=np.float64)
    _sc_ask_px = prepared_validation[_price_columns("ask", fitted_model.levels)].to_numpy(dtype=np.float64)
    _sc_ask_sz = prepared_validation[_size_columns("ask", fitted_model.levels)].to_numpy(dtype=np.float64)
    _sc_mids, _sc_raw = raw_multilevel_microprice_batch(
        bid_prices=_sc_bid_px, bid_sizes=_sc_bid_sz,
        ask_prices=_sc_ask_px, ask_sizes=_sc_ask_sz,
        tick_size=fitted_model.tick_size, decay_lambda=fitted_model.decay_lambda,
    )
    _sc_raw_adj = _sc_raw - _sc_mids
    _sc_spread_ticks = np.rint((_sc_ask_px[:, 0] - _sc_bid_px[:, 0]) / fitted_model.tick_size).astype(np.int64)
    _sc_total_bid = np.sum(_sc_bid_sz, axis=1)
    _sc_total_ask = np.sum(_sc_ask_sz, axis=1)
    _sc_denom = _sc_total_bid + _sc_total_ask
    _sc_depth_imb = np.where(_sc_denom > 0.0, (_sc_total_bid - _sc_total_ask) / _sc_denom, 0.0)
    _, _, _sc_state_idx = _state_index_arrays(
        _sc_spread_ticks, _sc_depth_imb,
        n_spread=fitted_model.n_spread,
        imbalance_edges=np.asarray(fitted_model.imbalance_edges, dtype=np.float64),
    )
    _sc_intercepts = np.asarray(fitted_model.state_intercepts, dtype=np.float64)
    _sc_slopes = np.asarray(fitted_model.state_slopes, dtype=np.float64)
    _sc_raw_value = _sc_raw_adj / fitted_model.tick_size if fitted_model.calibration_units == "ticks" else _sc_raw_adj
    _sc_predicted = _sc_intercepts[_sc_state_idx] + _sc_slopes[_sc_state_idx] * _sc_raw_value
    if fitted_model.calibration_units == "ticks":
        _sc_predicted = _sc_predicted * fitted_model.tick_size
    calibrated_signal = pd.Series(_sc_predicted, index=prepared_validation.index, dtype=float)
    target = prepared_validation["future_return"].astype(float)
    return {
        "model_kind": "multilevel_state_conditioned",
        "tick_size": float(fitted_model.tick_size),
        "horizon": int(horizon),
        "train_row_count": int(len(train_frame)),
        "validation_row_count": int(len(validation_frame)),
        "target_row_count": int(target.notna().sum()),
        "signals": [
            {
                "name": "raw_multilevel_microprice",
                **summarize_signal_performance(
                    raw_signal,
                    target,
                    tick_size=fitted_model.tick_size,
                    buckets=buckets,
                    threshold_percentiles=tuple(threshold_percentiles),
                ),
            },
            {
                "name": "state_conditioned_multilevel_microprice",
                **summarize_signal_performance(
                    calibrated_signal,
                    target,
                    tick_size=fitted_model.tick_size,
                    buckets=buckets,
                    threshold_percentiles=tuple(threshold_percentiles),
                ),
            },
        ],
        "model": fitted_model.to_dict(),
    }


def _book_side_array(values: Sequence[float], name: str, levels: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size < levels:
        raise ValueError(f"{name} must contain at least {levels} levels.")
    return array[:levels]


@dataclass
class FittedMultilevelAlphaLinearModel:
    tick_size: float
    levels: int
    decay_lambda: float
    alpha_horizon: int
    ridge_alpha: float
    feature_names: list[str]
    dropped_feature_names: list[str]
    feature_means: list[float]
    feature_stds: list[float]
    intercept: float
    coefficients: list[float]
    fit_intercept: bool = True
    validation_mse_ticks: float | None = None
    validation_mae_ticks: float | None = None
    validation_rmse_ticks: float | None = None
    validation_pearson: float | None = None
    validation_spearman: float | None = None
    validation_directional_accuracy: float | None = None
    validation_deciles: list[dict[str, Any]] = field(default_factory=list)
    validation_threshold_metrics: list[dict[str, Any]] = field(default_factory=list)
    top_bottom_decile_spread_ticks: float | None = None
    decile_monotonicity_score: float | None = None
    l1_benchmark: dict[str, Any] = field(default_factory=dict)
    beats_best_l1_spearman: bool = False
    beats_best_l1_decile_spread: bool = False
    beats_best_l1_rmse: bool = False
    _estimator: MultilevelMicroprice = field(init=False, repr=False)
    _coefficients_array: np.ndarray = field(init=False, repr=False)
    _feature_means_array: np.ndarray = field(init=False, repr=False)
    _feature_stds_array: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.tick_size = _validate_explicit_tick_size(self.tick_size)
        self.levels = int(self.levels)
        self.decay_lambda = float(self.decay_lambda)
        self.alpha_horizon = int(self.alpha_horizon)
        self.ridge_alpha = float(self.ridge_alpha)
        self.intercept = float(self.intercept)
        self.fit_intercept = bool(self.fit_intercept)
        if self.levels <= 0:
            raise ValueError("levels must be positive.")
        if not np.isfinite(self.decay_lambda) or self.decay_lambda < 0.0:
            raise ValueError("decay_lambda must be finite and nonnegative.")
        if self.alpha_horizon <= 0:
            raise ValueError("alpha_horizon must be positive.")
        if not np.isfinite(self.ridge_alpha) or self.ridge_alpha < 0.0:
            raise ValueError("ridge_alpha must be finite and nonnegative.")
        if len(self.feature_names) != len(self.feature_means) or len(self.feature_names) != len(self.feature_stds):
            raise ValueError("feature_names, feature_means, and feature_stds must have the same length.")
        if len(self.feature_names) != len(self.coefficients):
            raise ValueError("feature_names and coefficients must have the same length.")

        self.validation_mse_ticks = (
            None if self.validation_mse_ticks is None else float(self.validation_mse_ticks)
        )
        self.validation_mae_ticks = (
            None if self.validation_mae_ticks is None else float(self.validation_mae_ticks)
        )
        self.validation_rmse_ticks = (
            None if self.validation_rmse_ticks is None else float(self.validation_rmse_ticks)
        )
        self.validation_pearson = None if self.validation_pearson is None else float(self.validation_pearson)
        self.validation_spearman = None if self.validation_spearman is None else float(self.validation_spearman)
        self.validation_directional_accuracy = (
            None if self.validation_directional_accuracy is None else float(self.validation_directional_accuracy)
        )
        self.top_bottom_decile_spread_ticks = (
            None if self.top_bottom_decile_spread_ticks is None else float(self.top_bottom_decile_spread_ticks)
        )
        self.decile_monotonicity_score = (
            None if self.decile_monotonicity_score is None else float(self.decile_monotonicity_score)
        )

        self._estimator = MultilevelMicroprice(tick_size=self.tick_size, decay_lambda=self.decay_lambda)
        self._coefficients_array = np.asarray(self.coefficients, dtype=np.float64)
        self._feature_means_array = np.asarray(self.feature_means, dtype=np.float64)
        self._feature_stds_array = np.asarray(self.feature_stds, dtype=np.float64)

    def raw_microprice_from_book(
        self,
        bid_prices: Sequence[float],
        bid_sizes: Sequence[float],
        ask_prices: Sequence[float],
        ask_sizes: Sequence[float],
    ) -> float:
        return self._estimator(
            bid_prices=_book_side_array(bid_prices, "bid_prices", self.levels),
            bid_sizes=_book_side_array(bid_sizes, "bid_sizes", self.levels),
            ask_prices=_book_side_array(ask_prices, "ask_prices", self.levels),
            ask_sizes=_book_side_array(ask_sizes, "ask_sizes", self.levels),
        )

    def feature_vector_from_book(
        self,
        bid_prices: Sequence[float],
        bid_sizes: Sequence[float],
        ask_prices: Sequence[float],
        ask_sizes: Sequence[float],
    ) -> np.ndarray:
        bid_px = _book_side_array(bid_prices, "bid_prices", self.levels)
        bid_sz = _book_side_array(bid_sizes, "bid_sizes", self.levels)
        ask_px = _book_side_array(ask_prices, "ask_prices", self.levels)
        ask_sz = _book_side_array(ask_sizes, "ask_sizes", self.levels)
        mid, raw_microprice = raw_multilevel_microprice(
            bid_px,
            bid_sz,
            ask_px,
            ask_sz,
            tick_size=self.tick_size,
            decay_lambda=self.decay_lambda,
        )

        values: dict[str, float] = {
            "raw_adjustment_ticks": (raw_microprice - mid) / self.tick_size,
            "spread_ticks": (ask_px[0] - bid_px[0]) / self.tick_size,
        }
        level_denominator = bid_sz + ask_sz
        level_imbalance = _safe_ratio(bid_sz - ask_sz, level_denominator)
        cumulative_bid = np.cumsum(bid_sz)
        cumulative_ask = np.cumsum(ask_sz)
        cumulative_imbalance = _safe_ratio(
            cumulative_bid - cumulative_ask,
            cumulative_bid + cumulative_ask,
        )
        for level in range(1, self.levels + 1):
            values[f"imbalance_{level}"] = float(level_imbalance[level - 1])
            values[f"cum_imbalance_{level}"] = float(cumulative_imbalance[level - 1])
        if self.levels > 1:
            for level in range(1, self.levels):
                values[f"bid_gap_{level}_ticks"] = float((bid_px[level - 1] - bid_px[level]) / self.tick_size)
                values[f"ask_gap_{level}_ticks"] = float((ask_px[level] - ask_px[level - 1]) / self.tick_size)
        values[f"total_depth_imbalance_{self.levels}"] = float(cumulative_imbalance[-1])
        depth_slope_bid = float(_depth_slope(bid_sz[np.newaxis, :])[0])
        depth_slope_ask = float(_depth_slope(ask_sz[np.newaxis, :])[0])
        values["depth_slope_bid"] = depth_slope_bid
        values["depth_slope_ask"] = depth_slope_ask
        values["depth_slope_net"] = depth_slope_bid - depth_slope_ask
        return np.asarray([values[name] for name in self.feature_names], dtype=np.float64)

    def adjustment_from_book(
        self,
        bid_prices: Sequence[float],
        bid_sizes: Sequence[float],
        ask_prices: Sequence[float],
        ask_sizes: Sequence[float],
    ) -> float:
        if self._coefficients_array.size == 0:
            predicted_ticks = self.intercept
        else:
            raw_vector = self.feature_vector_from_book(
                bid_prices=bid_prices,
                bid_sizes=bid_sizes,
                ask_prices=ask_prices,
                ask_sizes=ask_sizes,
            )
            standardized = (raw_vector - self._feature_means_array) / self._feature_stds_array
            predicted_ticks = float(self.intercept + standardized @ self._coefficients_array)
        return predicted_ticks * self.tick_size

    def microprice_from_book(
        self,
        bid_prices: Sequence[float],
        bid_sizes: Sequence[float],
        ask_prices: Sequence[float],
        ask_sizes: Sequence[float],
    ) -> float:
        bid_px = _book_side_array(bid_prices, "bid_prices", self.levels)
        ask_px = _book_side_array(ask_prices, "ask_prices", self.levels)
        mid = 0.5 * (bid_px[0] + ask_px[0])
        return mid + self.adjustment_from_book(
            bid_prices=bid_prices,
            bid_sizes=bid_sizes,
            ask_prices=ask_prices,
            ask_sizes=ask_sizes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick_size": self.tick_size,
            "levels": self.levels,
            "decay_lambda": self.decay_lambda,
            "alpha_horizon": self.alpha_horizon,
            "ridge_alpha": self.ridge_alpha,
            "feature_names": self.feature_names,
            "dropped_feature_names": self.dropped_feature_names,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "intercept": self.intercept,
            "coefficients": self.coefficients,
            "fit_intercept": self.fit_intercept,
            "validation_mse_ticks": self.validation_mse_ticks,
            "validation_mae_ticks": self.validation_mae_ticks,
            "validation_rmse_ticks": self.validation_rmse_ticks,
            "validation_pearson": self.validation_pearson,
            "validation_spearman": self.validation_spearman,
            "validation_directional_accuracy": self.validation_directional_accuracy,
            "validation_deciles": self.validation_deciles,
            "validation_threshold_metrics": self.validation_threshold_metrics,
            "top_bottom_decile_spread_ticks": self.top_bottom_decile_spread_ticks,
            "decile_monotonicity_score": self.decile_monotonicity_score,
            "l1_benchmark": self.l1_benchmark,
            "beats_best_l1_spearman": self.beats_best_l1_spearman,
            "beats_best_l1_decile_spread": self.beats_best_l1_decile_spread,
            "beats_best_l1_rmse": self.beats_best_l1_rmse,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FittedMultilevelAlphaLinearModel":
        return cls(**payload)

    def save_model(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def load_multilevel_alpha_linear_model(path: str | Path) -> FittedMultilevelAlphaLinearModel:
    return FittedMultilevelAlphaLinearModel.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_multilevel_alpha_linear_model(
    df: pd.DataFrame,
    level_candidates: Sequence[int] = (1, 3, 5),
    decay_lambda_candidates: Sequence[float] = (0.0, 0.1, 0.3, 1.0, 3.0, 5.0),
    ridge_alpha_candidates: Sequence[float] = DEFAULT_RIDGE_ALPHA_CANDIDATES,
    alpha_horizon: int = DEFAULT_ALPHA_HORIZON,
    train_fraction: float = 0.8,
    tick_size: float | None = None,
    fit_intercept: bool = True,
    dt: int = 1,
    purge_rows: int | None = None,
    drop_invalid_rows: bool = True,
    buckets: int = DEFAULT_DECILE_BUCKETS,
    threshold_percentiles: Sequence[int | float] = DEFAULT_THRESHOLD_PERCENTILES,
    l1_max_horizon: int = 6,
) -> FittedMultilevelAlphaLinearModel:
    if alpha_horizon <= 0:
        raise ValueError("alpha_horizon must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    levels_grid = _normalize_level_candidates(level_candidates)
    decay_grid = _normalize_decay_candidates(decay_lambda_candidates)
    ridge_grid = tuple(float(value) for value in ridge_alpha_candidates)
    if not ridge_grid:
        raise ValueError("ridge_alpha_candidates must not be empty.")
    if any((not np.isfinite(value)) or value < 0.0 for value in ridge_grid):
        raise ValueError("ridge_alpha_candidates must contain only finite nonnegative values.")

    max_levels = max(levels_grid)
    normalized = _normalize_multilevel_input(df, levels=max_levels, drop_invalid_rows=drop_invalid_rows)
    resolved_tick_size = _resolve_tick_size(normalized, levels=max_levels, tick_size=tick_size)
    resolved_purge_rows = max(dt, alpha_horizon) if purge_rows is None else int(purge_rows)
    train_frame, validation_frame = _split_train_validation(
        normalized,
        train_fraction=train_fraction,
        purge_rows=resolved_purge_rows,
    )
    l1_benchmark = _evaluate_l1_baselines_from_multilevel(
        train_frame=train_frame,
        validation_frame=validation_frame,
        alpha_horizon=alpha_horizon,
        tick_size=resolved_tick_size,
        buckets=buckets,
        dt=dt,
        max_horizon=l1_max_horizon,
    )

    best_result: dict[str, Any] | None = None
    skipped_nonpositive_spearman = 0
    for levels in levels_grid:
        train_view = train_frame.loc[:, _level_columns(levels)].copy()
        validation_view = validation_frame.loc[:, _level_columns(levels)].copy()
        for decay_lambda in decay_grid:
            prepared_train, _ = prepare_multilevel_snapshot_feature_frame(
                train_view,
                levels=levels,
                decay_lambda=decay_lambda,
                alpha_horizon=alpha_horizon,
                tick_size=resolved_tick_size,
                drop_invalid_rows=drop_invalid_rows,
            )
            prepared_validation, _ = prepare_multilevel_snapshot_feature_frame(
                validation_view,
                levels=levels,
                decay_lambda=decay_lambda,
                alpha_horizon=alpha_horizon,
                tick_size=resolved_tick_size,
                drop_invalid_rows=drop_invalid_rows,
            )
            feature_names = _multilevel_snapshot_feature_names(levels)
            train_matrix, validation_matrix, kept_names, dropped_names, means, stds = _prepare_feature_matrices(
                prepared_train,
                prepared_validation,
                feature_names=feature_names,
            )
            horizon_column = f"future_return_h{alpha_horizon}_ticks"
            train_target = prepared_train[horizon_column].to_numpy(dtype=np.float64)
            validation_target = prepared_validation[horizon_column].to_numpy(dtype=np.float64)
            for ridge_alpha in ridge_grid:
                intercept, coefficients = _fit_linear_feature_model(
                    train_matrix,
                    train_target,
                    ridge_alpha=ridge_alpha,
                    fit_intercept=fit_intercept,
                )
                train_predictions = _predict_linear_feature_model(train_matrix, intercept, coefficients)
                validation_predictions = _predict_linear_feature_model(validation_matrix, intercept, coefficients)
                train_metrics = _evaluate_tick_predictions(
                    train_predictions,
                    train_target,
                    buckets=buckets,
                    threshold_percentiles=threshold_percentiles,
                )
                validation_metrics = _evaluate_tick_predictions(
                    validation_predictions,
                    validation_target,
                    buckets=buckets,
                    threshold_percentiles=threshold_percentiles,
                )
                validation_spearman = validation_metrics["spearman"]
                if validation_spearman is None or validation_spearman <= 0.0:
                    skipped_nonpositive_spearman += 1
                    continue

                result = {
                    "levels": levels,
                    "decay_lambda": float(decay_lambda),
                    "ridge_alpha": float(ridge_alpha),
                    "feature_names": kept_names,
                    "dropped_feature_names": dropped_names,
                    "feature_means": means,
                    "feature_stds": stds,
                    "train_metrics": train_metrics,
                    "validation_metrics": validation_metrics,
                }
                if best_result is None:
                    best_result = result
                    continue

                current_key = (
                    -_metric_or_default(validation_metrics["spearman"], float("-inf")),
                    -_metric_or_default(validation_metrics["top_bottom_decile_spread_ticks"], float("-inf")),
                    -_metric_or_default(validation_metrics["directional_accuracy"], float("-inf")),
                    float(validation_metrics["rmse_ticks"]),
                    result["levels"],
                    result["decay_lambda"],
                    result["ridge_alpha"],
                )
                best_validation_metrics = best_result["validation_metrics"]
                best_key = (
                    -_metric_or_default(best_validation_metrics["spearman"], float("-inf")),
                    -_metric_or_default(best_validation_metrics["top_bottom_decile_spread_ticks"], float("-inf")),
                    -_metric_or_default(best_validation_metrics["directional_accuracy"], float("-inf")),
                    float(best_validation_metrics["rmse_ticks"]),
                    best_result["levels"],
                    best_result["decay_lambda"],
                    best_result["ridge_alpha"],
                )
                if current_key < best_key:
                    best_result = result

    if best_result is None:
        raise ValueError(
            "Unable to fit a transparent multilevel alpha model with positive validation Spearman. "
            f"Skipped {skipped_nonpositive_spearman} non-predictive candidate configurations."
        )

    full_view = normalized.loc[:, _level_columns(int(best_result["levels"]))].copy()
    prepared_full, _ = prepare_multilevel_snapshot_feature_frame(
        full_view,
        levels=int(best_result["levels"]),
        decay_lambda=float(best_result["decay_lambda"]),
        alpha_horizon=alpha_horizon,
        tick_size=resolved_tick_size,
        drop_invalid_rows=drop_invalid_rows,
    )
    full_feature_names = _multilevel_snapshot_feature_names(int(best_result["levels"]))
    full_matrix = prepared_full.loc[:, full_feature_names].to_numpy(dtype=np.float64, copy=False)
    full_means = full_matrix.mean(axis=0)
    full_stds = full_matrix.std(axis=0)
    keep_mask = np.isfinite(full_means) & np.isfinite(full_stds) & (full_stds > 0.0)
    kept_feature_names = [name for name, keep in zip(full_feature_names, keep_mask) if keep]
    dropped_feature_names = [name for name, keep in zip(full_feature_names, keep_mask) if not keep]
    if bool(keep_mask.any()):
        full_kept_means = full_means[keep_mask]
        full_kept_stds = full_stds[keep_mask]
        standardized_full = (full_matrix[:, keep_mask] - full_kept_means) / full_kept_stds
    else:
        full_kept_means = np.zeros(0, dtype=np.float64)
        full_kept_stds = np.zeros(0, dtype=np.float64)
        standardized_full = np.zeros((len(prepared_full), 0), dtype=np.float64)

    full_target = prepared_full[f"future_return_h{alpha_horizon}_ticks"].to_numpy(dtype=np.float64)
    intercept, coefficients = _fit_linear_feature_model(
        standardized_full,
        full_target,
        ridge_alpha=float(best_result["ridge_alpha"]),
        fit_intercept=fit_intercept,
    )
    validation_metrics = best_result["validation_metrics"]
    comparison = {
        **l1_benchmark,
        "delta_spearman": None
        if validation_metrics["spearman"] is None
        else float(validation_metrics["spearman"] - l1_benchmark["best_spearman"]),
        "delta_decile_spread_ticks": None
        if validation_metrics["top_bottom_decile_spread_ticks"] is None
        else float(validation_metrics["top_bottom_decile_spread_ticks"] - l1_benchmark["best_decile_spread_ticks"]),
        "delta_rmse_ticks": float(validation_metrics["rmse_ticks"] - l1_benchmark["best_rmse_ticks"]),
    }
    return FittedMultilevelAlphaLinearModel(
        tick_size=resolved_tick_size,
        levels=int(best_result["levels"]),
        decay_lambda=float(best_result["decay_lambda"]),
        alpha_horizon=alpha_horizon,
        ridge_alpha=float(best_result["ridge_alpha"]),
        feature_names=kept_feature_names,
        dropped_feature_names=dropped_feature_names,
        feature_means=full_kept_means.tolist(),
        feature_stds=full_kept_stds.tolist(),
        intercept=float(intercept),
        coefficients=coefficients.astype(np.float64).tolist(),
        fit_intercept=fit_intercept,
        validation_mse_ticks=float(validation_metrics["mse_ticks"]),
        validation_mae_ticks=float(validation_metrics["mae_ticks"]),
        validation_rmse_ticks=float(validation_metrics["rmse_ticks"]),
        validation_pearson=validation_metrics["pearson"],
        validation_spearman=validation_metrics["spearman"],
        validation_directional_accuracy=validation_metrics["directional_accuracy"],
        validation_deciles=list(validation_metrics["deciles"]),
        validation_threshold_metrics=list(validation_metrics["threshold_metrics"]),
        top_bottom_decile_spread_ticks=validation_metrics["top_bottom_decile_spread_ticks"],
        decile_monotonicity_score=validation_metrics["decile_monotonicity_score"],
        l1_benchmark=comparison,
        beats_best_l1_spearman=bool(
            validation_metrics["spearman"] is not None
            and validation_metrics["spearman"] > l1_benchmark["best_spearman"]
        ),
        beats_best_l1_decile_spread=bool(
            validation_metrics["top_bottom_decile_spread_ticks"] is not None
            and validation_metrics["top_bottom_decile_spread_ticks"] >= l1_benchmark["best_decile_spread_ticks"]
        ),
        beats_best_l1_rmse=bool(validation_metrics["rmse_ticks"] <= l1_benchmark["best_rmse_ticks"]),
    )


__all__ = [
    "DEFAULT_CALIBRATION_HORIZON",
    "DEFAULT_ALPHA_HORIZON",
    "FittedMultilevelMicropriceModel",
    "MAX_BOOK_LEVELS",
    "compare_l1_and_multilevel_microprices",
    "evaluate_multilevel_microprice_model",
    "extract_multilevel_from_hyperliquid_jsonl",
    "extract_multilevel_from_l2_tensor",
    "fit_multilevel_microprice_model",
    "load_multilevel_model",
    "prepare_multilevel_feature_frame",
    "summarize_multilevel_alpha_signal",
]
