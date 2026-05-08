"""Order flow imbalance helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


def top_of_book_order_flow_imbalance(
    previous_bid: float,
    previous_bid_size: float,
    previous_ask: float,
    previous_ask_size: float,
    bid: float,
    bid_size: float,
    ask: float,
    ask_size: float,
) -> float:
    """Return the Cont-style top-of-book OFI increment for one BBO update."""
    bid_contribution = (
        (1.0 if bid >= previous_bid else 0.0) * float(bid_size)
        - (1.0 if bid <= previous_bid else 0.0) * float(previous_bid_size)
    )
    ask_contribution = (
        (1.0 if ask <= previous_ask else 0.0) * float(ask_size)
        - (1.0 if ask >= previous_ask else 0.0) * float(previous_ask_size)
    )
    return bid_contribution - ask_contribution


def top_of_book_order_flow_imbalance_series(
    bids: Any,
    bid_sizes: Any,
    asks: Any,
    ask_sizes: Any,
) -> np.ndarray:
    bid_array = np.asarray(bids, dtype=np.float64)
    bid_size_array = np.asarray(bid_sizes, dtype=np.float64)
    ask_array = np.asarray(asks, dtype=np.float64)
    ask_size_array = np.asarray(ask_sizes, dtype=np.float64)
    if not (
        bid_array.shape == bid_size_array.shape == ask_array.shape == ask_size_array.shape
    ):
        raise ValueError("All OFI inputs must have the same shape.")
    if bid_array.ndim != 1:
        raise ValueError("OFI series inputs must be one-dimensional.")
    if bid_array.size == 0:
        return np.zeros(0, dtype=np.float64)

    result = np.zeros_like(bid_array, dtype=np.float64)
    for index in range(1, bid_array.size):
        result[index] = top_of_book_order_flow_imbalance(
            previous_bid=bid_array[index - 1],
            previous_bid_size=bid_size_array[index - 1],
            previous_ask=ask_array[index - 1],
            previous_ask_size=ask_size_array[index - 1],
            bid=bid_array[index],
            bid_size=bid_size_array[index],
            ask=ask_array[index],
            ask_size=ask_size_array[index],
        )
    return result


def rolling_order_flow_imbalance_series(
    bids: Any,
    bid_sizes: Any,
    asks: Any,
    ask_sizes: Any,
    *,
    window: int,
) -> np.ndarray:
    width = int(window)
    if width <= 0:
        raise ValueError("window must be positive.")
    increments = top_of_book_order_flow_imbalance_series(bids, bid_sizes, asks, ask_sizes)
    if increments.size == 0:
        return increments
    cumulative = np.cumsum(increments, dtype=np.float64)
    result = cumulative.copy()
    if width < increments.size:
        result[width:] = cumulative[width:] - cumulative[:-width]
    return result


@dataclass(frozen=True, slots=True)
class OnlineOFISnapshot:
    increment: float
    rolling_ofi: float
    observation_count: int


class OnlineOFI:
    def __init__(self, *, window: int) -> None:
        width = int(window)
        if width <= 0:
            raise ValueError("window must be positive.")
        self.window = width
        self._increments: deque[float] = deque(maxlen=width)
        self._previous: tuple[float, float, float, float] | None = None
        self._count = 0

    def update(
        self,
        *,
        bid: float,
        bid_size: float,
        ask: float,
        ask_size: float,
    ) -> OnlineOFISnapshot:
        current = (float(bid), float(bid_size), float(ask), float(ask_size))
        if self._previous is None:
            increment = 0.0
        else:
            increment = top_of_book_order_flow_imbalance(
                previous_bid=self._previous[0],
                previous_bid_size=self._previous[1],
                previous_ask=self._previous[2],
                previous_ask_size=self._previous[3],
                bid=current[0],
                bid_size=current[1],
                ask=current[2],
                ask_size=current[3],
            )
        self._previous = current
        self._increments.append(increment)
        self._count += 1
        return OnlineOFISnapshot(
            increment=increment,
            rolling_ofi=float(sum(self._increments)),
            observation_count=self._count,
        )


__all__ = [
    "OnlineOFI",
    "OnlineOFISnapshot",
    "rolling_order_flow_imbalance_series",
    "top_of_book_order_flow_imbalance",
    "top_of_book_order_flow_imbalance_series",
]
