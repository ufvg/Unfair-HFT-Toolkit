"""Spread and execution-cost helpers for market microstructure research."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

TradeSide = Literal["buy", "sell", 1, -1]


def _normalize_side(side: TradeSide) -> int:
    if side in ("buy", 1):
        return 1
    if side in ("sell", -1):
        return -1
    raise ValueError("side must be 'buy', 'sell', 1, or -1.")


def midprice(best_bid: float, best_ask: float) -> float:
    bid = float(best_bid)
    ask = float(best_ask)
    if bid > ask:
        raise ValueError("best_bid must not exceed best_ask.")
    return 0.5 * (bid + ask)


def quoted_spread(best_bid: float, best_ask: float) -> float:
    bid = float(best_bid)
    ask = float(best_ask)
    if bid > ask:
        raise ValueError("best_bid must not exceed best_ask.")
    return ask - bid


def relative_spread(best_bid: float, best_ask: float) -> float:
    mid = midprice(best_bid, best_ask)
    if mid <= 0.0:
        raise ValueError("midprice must be positive.")
    return quoted_spread(best_bid, best_ask) / mid


def spread_in_ticks(best_bid: float, best_ask: float, tick_size: float) -> float:
    tick = float(tick_size)
    if tick <= 0.0:
        raise ValueError("tick_size must be positive.")
    return quoted_spread(best_bid, best_ask) / tick


def effective_spread(execution_price: float, reference_midprice: float, side: TradeSide) -> float:
    sign = _normalize_side(side)
    return 2.0 * sign * (float(execution_price) - float(reference_midprice))


def realized_spread(execution_price: float, future_midprice: float, side: TradeSide) -> float:
    sign = _normalize_side(side)
    return 2.0 * sign * (float(execution_price) - float(future_midprice))


def adverse_selection_component(
    execution_price: float,
    reference_midprice: float,
    future_midprice: float,
    side: TradeSide,
) -> float:
    return effective_spread(execution_price, reference_midprice, side) - realized_spread(
        execution_price,
        future_midprice,
        side,
    )


def midprice_array(best_bids: Any, best_asks: Any) -> np.ndarray:
    bids = np.asarray(best_bids, dtype=np.float64)
    asks = np.asarray(best_asks, dtype=np.float64)
    if bids.shape != asks.shape:
        raise ValueError("best_bids and best_asks must have the same shape.")
    if np.any(bids > asks):
        raise ValueError("best_bids must not exceed best_asks.")
    return 0.5 * (bids + asks)


def quoted_spread_array(best_bids: Any, best_asks: Any) -> np.ndarray:
    bids = np.asarray(best_bids, dtype=np.float64)
    asks = np.asarray(best_asks, dtype=np.float64)
    if bids.shape != asks.shape:
        raise ValueError("best_bids and best_asks must have the same shape.")
    if np.any(bids > asks):
        raise ValueError("best_bids must not exceed best_asks.")
    return asks - bids


__all__ = [
    "adverse_selection_component",
    "effective_spread",
    "midprice",
    "midprice_array",
    "quoted_spread",
    "quoted_spread_array",
    "realized_spread",
    "relative_spread",
    "spread_in_ticks",
]
