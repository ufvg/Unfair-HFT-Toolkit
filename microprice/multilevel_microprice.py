from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .validation import _validate_nonnegative as _validate_nonnegative_scalar
from .validation import _validate_positive as _validate_positive_scalar

FloatArray = NDArray[np.float64]


def _as_float_level_array(values: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _as_nonnegative_level_array(values: ArrayLike, name: str) -> FloatArray:
    array = _as_float_level_array(values, name)
    if np.any(array < 0.0):
        raise ValueError(f"{name} must contain only nonnegative values.")
    return array


def _validate_order_book_structure(bid_prices: FloatArray, ask_prices: FloatArray) -> None:
    if np.any(np.diff(bid_prices) > 0.0):
        raise ValueError("bid_prices must be non-increasing by level.")
    if np.any(np.diff(ask_prices) < 0.0):
        raise ValueError("ask_prices must be non-decreasing by level.")
    if bid_prices[0] > ask_prices[0]:
        raise ValueError("Best bid must not exceed best ask.")


def standard_l1_microprice(
    bid_price: float,
    ask_price: float,
    bid_size: float,
    ask_size: float,
) -> float:
    """Return the standard top-of-book cross-side weighted microprice."""
    bid = float(bid_price)
    ask = float(ask_price)
    bid_qty = _validate_nonnegative_scalar(bid_size, "bid_size")
    ask_qty = _validate_nonnegative_scalar(ask_size, "ask_size")
    if not np.isfinite(bid) or not np.isfinite(ask):
        raise ValueError("bid_price and ask_price must be finite.")
    mid = 0.5 * (bid + ask)
    denominator = bid_qty + ask_qty
    if denominator <= 0.0:
        return mid
    return (ask * bid_qty + bid * ask_qty) / denominator


def _compute_distance_decayed_weights_core(
    level_prices: FloatArray,
    level_sizes: FloatArray,
    mid: FloatArray,
    tick_size: float,
    decay_lambda: float,
    side: Literal["bid", "ask"],
) -> FloatArray:
    price_distance = np.empty_like(level_prices)
    if side == "bid":
        np.subtract(np.expand_dims(mid, axis=-1), level_prices, out=price_distance)
    else:
        np.subtract(level_prices, np.expand_dims(mid, axis=-1), out=price_distance)
    price_distance /= tick_size
    price_distance *= -decay_lambda
    np.exp(price_distance, out=price_distance)
    price_distance *= level_sizes
    return price_distance


def compute_distance_decayed_weights(
    level_prices: ArrayLike,
    level_sizes: ArrayLike,
    mid: float,
    tick_size: float,
    decay_lambda: float,
    side: Literal["bid", "ask"],
) -> FloatArray:
    """Return per-level size weights after exponential distance decay."""
    prices = _as_float_level_array(level_prices, "level_prices")
    sizes = _as_nonnegative_level_array(level_sizes, "level_sizes")
    if prices.shape != sizes.shape:
        raise ValueError("level_prices and level_sizes must have the same shape.")
    center = float(mid)
    if not np.isfinite(center):
        raise ValueError("mid must be finite.")
    tick = _validate_positive_scalar(tick_size, "tick_size")
    decay = _validate_nonnegative_scalar(decay_lambda, "decay_lambda")
    if side not in ("bid", "ask"):
        raise ValueError("side must be 'bid' or 'ask'.")
    weights = _compute_distance_decayed_weights_core(
        level_prices=prices,
        level_sizes=sizes,
        mid=np.asarray(center, dtype=np.float64),
        tick_size=tick,
        decay_lambda=decay,
        side=side,
    )
    if not np.isfinite(weights).all():
        raise ValueError("Computed decayed weights must be finite.")
    return weights


def raw_multilevel_microprice(
    bid_prices: ArrayLike,
    bid_sizes: ArrayLike,
    ask_prices: ArrayLike,
    ask_sizes: ArrayLike,
    tick_size: float,
    decay_lambda: float,
) -> tuple[float, float]:
    """Return `(mid, raw_microprice)` for one validated multilevel order book.

    Assumptions:
    - `bid_prices` and `ask_prices` are top-of-book-first ladders
    - bid prices are non-increasing, ask prices are non-decreasing
    - sizes are finite and nonnegative
    - `tick_size` is positive
    - `decay_lambda >= 0`; `0` is supported as a no-decay benchmark mode
    """
    bid_px = _as_float_level_array(bid_prices, "bid_prices")
    bid_sz = _as_nonnegative_level_array(bid_sizes, "bid_sizes")
    ask_px = _as_float_level_array(ask_prices, "ask_prices")
    ask_sz = _as_nonnegative_level_array(ask_sizes, "ask_sizes")
    if bid_px.shape != bid_sz.shape:
        raise ValueError("bid_prices and bid_sizes must have the same shape.")
    if ask_px.shape != ask_sz.shape:
        raise ValueError("ask_prices and ask_sizes must have the same shape.")
    if bid_px.shape != ask_px.shape:
        raise ValueError("Bid-side and ask-side inputs must have the same length.")

    tick = _validate_positive_scalar(tick_size, "tick_size")
    decay = _validate_nonnegative_scalar(decay_lambda, "decay_lambda")
    _validate_order_book_structure(bid_px, ask_px)

    mid, raw_microprice = raw_multilevel_microprice_batch(
        bid_prices=bid_px[np.newaxis, :],
        bid_sizes=bid_sz[np.newaxis, :],
        ask_prices=ask_px[np.newaxis, :],
        ask_sizes=ask_sz[np.newaxis, :],
        tick_size=tick,
        decay_lambda=decay,
    )
    return float(mid[0]), float(raw_microprice[0])


def raw_multilevel_microprice_batch(
    bid_prices: ArrayLike,
    bid_sizes: ArrayLike,
    ask_prices: ArrayLike,
    ask_sizes: ArrayLike,
    tick_size: float,
    decay_lambda: float,
) -> tuple[FloatArray, FloatArray]:
    """Vectorized raw multilevel microprice over the last axis.

    This helper intentionally does not re-check order-book monotonicity row by row.
    Callers should validate or filter books first. It is shared by calibration and
    inference to keep the raw microprice formula consistent.
    """
    bid_px = np.asarray(bid_prices, dtype=np.float64)
    bid_sz = np.asarray(bid_sizes, dtype=np.float64)
    ask_px = np.asarray(ask_prices, dtype=np.float64)
    ask_sz = np.asarray(ask_sizes, dtype=np.float64)
    if bid_px.shape != bid_sz.shape:
        raise ValueError("bid_prices and bid_sizes must have the same shape.")
    if ask_px.shape != ask_sz.shape:
        raise ValueError("ask_prices and ask_sizes must have the same shape.")
    if bid_px.shape != ask_px.shape:
        raise ValueError("Bid-side and ask-side inputs must have the same shape.")
    if bid_px.ndim == 0:
        raise ValueError("Order book arrays must have at least one dimension.")
    if bid_px.shape[-1] == 0:
        raise ValueError("Order book arrays must contain at least one level.")
    if not np.isfinite(bid_px).all() or not np.isfinite(ask_px).all():
        raise ValueError("Price arrays must contain only finite values.")
    if not np.isfinite(bid_sz).all() or not np.isfinite(ask_sz).all():
        raise ValueError("Size arrays must contain only finite values.")
    if np.any(bid_sz < 0.0) or np.any(ask_sz < 0.0):
        raise ValueError("Size arrays must contain only nonnegative values.")

    tick = _validate_positive_scalar(tick_size, "tick_size")
    decay = _validate_nonnegative_scalar(decay_lambda, "decay_lambda")

    mid = 0.5 * (bid_px[..., 0] + ask_px[..., 0])
    bid_weights = _compute_distance_decayed_weights_core(
        level_prices=bid_px,
        level_sizes=bid_sz,
        mid=mid,
        tick_size=tick,
        decay_lambda=decay,
        side="bid",
    )
    ask_weights = _compute_distance_decayed_weights_core(
        level_prices=ask_px,
        level_sizes=ask_sz,
        mid=mid,
        tick_size=tick,
        decay_lambda=decay,
        side="ask",
    )
    numerator = np.sum(ask_px * bid_weights, axis=-1) + np.sum(bid_px * ask_weights, axis=-1)
    denominator = np.sum(bid_weights, axis=-1) + np.sum(ask_weights, axis=-1)
    raw_microprice = np.array(mid, copy=True)
    np.divide(numerator, denominator, out=raw_microprice, where=denominator > 0.0)
    return np.asarray(mid, dtype=np.float64), np.asarray(raw_microprice, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class MultilevelMicroprice:
    """Raw multilevel microprice with exponential distance decay."""

    tick_size: float
    decay_lambda: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "tick_size", _validate_positive_scalar(self.tick_size, "tick_size"))
        object.__setattr__(self, "decay_lambda", _validate_nonnegative_scalar(self.decay_lambda, "decay_lambda"))

    def __call__(
        self,
        bid_prices: ArrayLike,
        bid_sizes: ArrayLike,
        ask_prices: ArrayLike,
        ask_sizes: ArrayLike,
    ) -> float:
        return self.compute(
            bid_prices=bid_prices,
            bid_sizes=bid_sizes,
            ask_prices=ask_prices,
            ask_sizes=ask_sizes,
        )

    def compute(
        self,
        bid_prices: ArrayLike,
        bid_sizes: ArrayLike,
        ask_prices: ArrayLike,
        ask_sizes: ArrayLike,
    ) -> float:
        _, microprice = raw_multilevel_microprice(
            bid_prices=bid_prices,
            bid_sizes=bid_sizes,
            ask_prices=ask_prices,
            ask_sizes=ask_sizes,
            tick_size=self.tick_size,
            decay_lambda=self.decay_lambda,
        )
        return microprice


def benchmark_microprice_implementations(
    iterations: int = 200_000,
    levels: int = 5,
    tick_size: float = 0.01,
    decay_lambda: float = 1.0,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if levels <= 0:
        raise ValueError("levels must be positive.")

    step = tick_size * np.arange(levels, dtype=np.float64)
    bid_prices = 100.0 - step
    ask_prices = 100.01 + step
    bid_sizes = np.linspace(12.0, 12.0 + levels - 1, levels, dtype=np.float64)
    ask_sizes = np.linspace(18.0, 18.0 + levels - 1, levels, dtype=np.float64)
    multilevel = MultilevelMicroprice(tick_size=tick_size, decay_lambda=decay_lambda)

    l1_start = time.perf_counter()
    for _ in range(iterations):
        standard_l1_microprice(bid_prices[0], ask_prices[0], bid_sizes[0], ask_sizes[0])
    l1_seconds = time.perf_counter() - l1_start

    multilevel_start = time.perf_counter()
    for _ in range(iterations):
        multilevel(bid_prices, bid_sizes, ask_prices, ask_sizes)
    multilevel_seconds = time.perf_counter() - multilevel_start

    return {
        "iterations": int(iterations),
        "levels": int(levels),
        "tick_size": float(tick_size),
        "decay_lambda": float(decay_lambda),
        "l1_microprice_seconds": float(l1_seconds),
        "multilevel_microprice_seconds": float(multilevel_seconds),
        "multilevel_over_l1_ratio": float(multilevel_seconds / l1_seconds) if l1_seconds > 0.0 else np.inf,
    }


__all__ = [
    "MultilevelMicroprice",
    "benchmark_microprice_implementations",
    "compute_distance_decayed_weights",
    "raw_multilevel_microprice",
    "raw_multilevel_microprice_batch",
    "standard_l1_microprice",
]
