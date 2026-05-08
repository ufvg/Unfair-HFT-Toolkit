"""Queue-reactive order book imbalance features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class QueueReactiveOBIResult:
    signed_imbalance: float
    bid_share: float
    weighted_bid_depth: float
    weighted_ask_depth: float


def _level_weights(level_count: int, decay: float) -> np.ndarray:
    if level_count <= 0:
        raise ValueError("level_count must be positive.")
    if decay < 0.0:
        raise ValueError("level_decay must be nonnegative.")
    levels = np.arange(level_count, dtype=np.float64)
    return np.exp(-decay * levels)


def _effective_queue_depth(sizes: np.ndarray, queue_ahead: np.ndarray | None, queue_penalty: float) -> np.ndarray:
    if queue_penalty < 0.0:
        raise ValueError("queue_penalty must be nonnegative.")
    if queue_ahead is None:
        return sizes
    if queue_ahead.shape != sizes.shape:
        raise ValueError("queue_ahead must match the shape of sizes.")
    return sizes / (1.0 + queue_penalty * np.maximum(queue_ahead, 0.0))


def queue_reactive_order_book_imbalance(
    bid_sizes: Any,
    ask_sizes: Any,
    *,
    queue_ahead_bid: Any | None = None,
    queue_ahead_ask: Any | None = None,
    level_decay: float = 0.5,
    queue_penalty: float = 1.0,
) -> QueueReactiveOBIResult:
    bid = np.asarray(bid_sizes, dtype=np.float64)
    ask = np.asarray(ask_sizes, dtype=np.float64)
    if bid.shape != ask.shape:
        raise ValueError("bid_sizes and ask_sizes must have the same shape.")
    if bid.ndim != 1:
        raise ValueError("bid_sizes and ask_sizes must be one-dimensional.")
    if bid.size == 0:
        raise ValueError("bid_sizes and ask_sizes must not be empty.")

    queue_bid = None if queue_ahead_bid is None else np.asarray(queue_ahead_bid, dtype=np.float64)
    queue_ask = None if queue_ahead_ask is None else np.asarray(queue_ahead_ask, dtype=np.float64)
    weights = _level_weights(bid.size, float(level_decay))
    effective_bid = _effective_queue_depth(bid, queue_bid, float(queue_penalty))
    effective_ask = _effective_queue_depth(ask, queue_ask, float(queue_penalty))
    weighted_bid = float(np.dot(weights, effective_bid))
    weighted_ask = float(np.dot(weights, effective_ask))
    total = weighted_bid + weighted_ask
    bid_share = 0.5 if total <= 0.0 else weighted_bid / total
    signed = 0.0 if total <= 0.0 else (weighted_bid - weighted_ask) / total
    return QueueReactiveOBIResult(
        signed_imbalance=signed,
        bid_share=bid_share,
        weighted_bid_depth=weighted_bid,
        weighted_ask_depth=weighted_ask,
    )


def queue_reactive_order_book_imbalance_series(
    bid_sizes: Any,
    ask_sizes: Any,
    *,
    queue_ahead_bid: Any | None = None,
    queue_ahead_ask: Any | None = None,
    level_decay: float = 0.5,
    queue_penalty: float = 1.0,
) -> np.ndarray:
    bid = np.asarray(bid_sizes, dtype=np.float64)
    ask = np.asarray(ask_sizes, dtype=np.float64)
    if bid.shape != ask.shape:
        raise ValueError("bid_sizes and ask_sizes must have the same shape.")
    if bid.ndim != 2:
        raise ValueError("bid_sizes and ask_sizes must be two-dimensional.")
    queue_bid = None if queue_ahead_bid is None else np.asarray(queue_ahead_bid, dtype=np.float64)
    queue_ask = None if queue_ahead_ask is None else np.asarray(queue_ahead_ask, dtype=np.float64)
    if queue_bid is not None and queue_bid.shape != bid.shape:
        raise ValueError("queue_ahead_bid must match bid_sizes shape.")
    if queue_ask is not None and queue_ask.shape != ask.shape:
        raise ValueError("queue_ahead_ask must match ask_sizes shape.")

    weights = _level_weights(bid.shape[1], float(level_decay))
    effective_bid = _effective_queue_depth(bid, queue_bid, float(queue_penalty))
    effective_ask = _effective_queue_depth(ask, queue_ask, float(queue_penalty))
    weighted_bid = effective_bid @ weights
    weighted_ask = effective_ask @ weights
    total = weighted_bid + weighted_ask
    result = np.zeros_like(total, dtype=np.float64)
    np.divide(weighted_bid - weighted_ask, total, out=result, where=total > 0.0)
    return result


__all__ = [
    "QueueReactiveOBIResult",
    "queue_reactive_order_book_imbalance",
    "queue_reactive_order_book_imbalance_series",
]
