"""Shared order book imbalance helpers for microstructure research."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional runtime acceleration
    njit = None


def order_book_imbalance(
    bid_size: float,
    ask_size: float,
    neutral_value: float = 0.5,
) -> float:
    """Return bid share of displayed top-of-book size on ``[0, 1]``."""
    bid_qty = float(bid_size)
    ask_qty = float(ask_size)
    total = bid_qty + ask_qty
    if total <= 0.0:
        return float(neutral_value)
    return bid_qty / total


def order_book_imbalance_array(
    bid_sizes: Any,
    ask_sizes: Any,
    neutral_value: float = 0.5,
) -> np.ndarray:
    bid = np.asarray(bid_sizes, dtype=np.float64)
    ask = np.asarray(ask_sizes, dtype=np.float64)
    total = bid + ask
    result = np.full_like(total, float(neutral_value), dtype=np.float64)
    np.divide(bid, total, out=result, where=total > 0.0)
    return result


def signed_order_book_imbalance(
    bid_size: float,
    ask_size: float,
    neutral_value: float = 0.0,
) -> float:
    """Return signed imbalance on ``[-1, 1]``."""
    bid_qty = float(bid_size)
    ask_qty = float(ask_size)
    total = bid_qty + ask_qty
    if total <= 0.0:
        return float(neutral_value)
    return (bid_qty - ask_qty) / total


def signed_order_book_imbalance_array(
    bid_sizes: Any,
    ask_sizes: Any,
    neutral_value: float = 0.0,
) -> np.ndarray:
    bid = np.asarray(bid_sizes, dtype=np.float64)
    ask = np.asarray(ask_sizes, dtype=np.float64)
    total = bid + ask
    result = np.full_like(total, float(neutral_value), dtype=np.float64)
    np.divide(bid - ask, total, out=result, where=total > 0.0)
    return result


def cumulative_signed_order_book_imbalance(
    bid_sizes: Any,
    ask_sizes: Any,
    neutral_value: float = 0.0,
) -> np.ndarray:
    bid = np.cumsum(np.asarray(bid_sizes, dtype=np.float64), axis=-1)
    ask = np.cumsum(np.asarray(ask_sizes, dtype=np.float64), axis=-1)
    return signed_order_book_imbalance_array(bid, ask, neutral_value=neutral_value)


def imbalance_bucket(
    value: float,
    edges: Any,
    lower_bound: float,
    upper_bound: float,
) -> int:
    edges_array = np.asarray(edges, dtype=np.float64)
    clipped = min(max(float(value), float(lower_bound)), float(upper_bound))
    bucket = int(np.searchsorted(edges_array[1:-1], clipped, side="right"))
    return min(max(bucket, 0), edges_array.size - 2)


def bucketize_imbalance(
    values: Any,
    edges: Any,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    edges_array = np.asarray(edges, dtype=np.float64)
    clipped = np.clip(np.asarray(values, dtype=np.float64), float(lower_bound), float(upper_bound))
    buckets = np.searchsorted(edges_array[1:-1], clipped, side="right")
    return np.clip(buckets, 0, edges_array.size - 2).astype(np.int64, copy=False)


def quantile_bucket_edges(
    values: Any,
    bucket_count: int,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive.")
    clipped = np.clip(np.asarray(values, dtype=np.float64), float(lower_bound), float(upper_bound))
    quantiles = np.linspace(0.0, 1.0, bucket_count + 1, dtype=np.float64)
    edges = np.asarray(np.quantile(clipped, quantiles), dtype=np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


if njit is not None:

    @njit(cache=True)
    def _searchsorted_right(edges: np.ndarray, value: float) -> int:
        left = 0
        right = edges.shape[0]
        while left < right:
            mid = (left + right) // 2
            if value < edges[mid]:
                right = mid
            else:
                left = mid + 1
        return left


    @njit(cache=True)
    def l1_order_book_imbalance_numba(
        bid_size: float,
        ask_size: float,
        neutral_value: float = 0.5,
    ) -> float:
        total = bid_size + ask_size
        if total <= 0.0:
            return neutral_value
        return bid_size / total


    @njit(cache=True)
    def imbalance_bucket_numba(
        value: float,
        edges: np.ndarray,
        lower_bound: float,
        upper_bound: float,
    ) -> int:
        clipped = value
        if clipped < lower_bound:
            clipped = lower_bound
        elif clipped > upper_bound:
            clipped = upper_bound
        bucket = _searchsorted_right(edges[1:-1], clipped)
        if bucket < 0:
            return 0
        max_bucket = edges.shape[0] - 2
        if bucket > max_bucket:
            return max_bucket
        return bucket


else:
    l1_order_book_imbalance_numba = None
    imbalance_bucket_numba = None


__all__ = [
    "bucketize_imbalance",
    "cumulative_signed_order_book_imbalance",
    "imbalance_bucket",
    "imbalance_bucket_numba",
    "l1_order_book_imbalance_numba",
    "order_book_imbalance",
    "order_book_imbalance_array",
    "quantile_bucket_edges",
    "signed_order_book_imbalance",
    "signed_order_book_imbalance_array",
]
