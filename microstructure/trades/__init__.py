"""Shared trade-feed normalization helpers and tape-print models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import math
from typing import Literal


TradeSide = Literal[-1, 1]


def coerce_trade_timestamp(value: datetime | str | int | float) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    if isinstance(value, bool):
        raise ValueError("timestamp must not be boolean.")

    if isinstance(value, (int, float)):
        raw = float(value)
        if not math.isfinite(raw):
            raise ValueError("timestamp must be finite.")
        scale = 1.0
        absolute = abs(raw)
        if absolute >= 1e18:
            scale = 1e9
        elif absolute >= 1e15:
            scale = 1e6
        elif absolute >= 1e12:
            scale = 1e3
        return datetime.fromtimestamp(raw / scale, tz=UTC)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("timestamp string must not be empty.")
        try:
            numeric = float(text)
        except ValueError:
            normalized = text.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        return coerce_trade_timestamp(numeric)

    raise TypeError("timestamp must be a datetime, ISO string, or epoch number.")


def normalize_trade_side(value: int | float | str | None) -> TradeSide | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        if text in {"b", "buy", "bid", "+1", "1"}:
            return 1
        if text in {"a", "ask", "s", "sell", "-1"}:
            return -1
        raise ValueError("side must be one of buy/sell, bid/ask, b/a, +1/-1, or 1/-1.")
    numeric = float(value)
    if numeric > 0.0:
        return 1
    if numeric < 0.0:
        return -1
    raise ValueError("side must be positive for buy or negative for sell.")


def _validate_positive(value: float, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


@dataclass(frozen=True, slots=True)
class MarketTrade:
    timestamp: datetime
    price: float
    volume: float
    side: TradeSide | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", coerce_trade_timestamp(self.timestamp))
        object.__setattr__(self, "price", _validate_positive(self.price, "price"))
        object.__setattr__(self, "volume", _validate_positive(self.volume, "volume"))
        object.__setattr__(self, "side", normalize_trade_side(self.side))


__all__ = [
    "MarketTrade",
    "TradeSide",
    "coerce_trade_timestamp",
    "normalize_trade_side",
]
