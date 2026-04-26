from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class StrategySnapshot:
    enabled: bool
    event_index: int
    position: int
    events_remaining: int
    entry_price: float | None
    entry_adjustment: float | None
    entry_event_index: int | None
    exit_event_index: int | None
    mark_price: float | None
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    trades_closed: int
    last_action: str


@dataclass
class ClosedTrade:
    side: str
    entry_price: float
    exit_price: float
    entry_adjustment: float
    entry_event_index: int
    exit_event_index: int
    pnl: float
    exit_reason: str


class AdjustmentThresholdStrategy:
    """Long on adjustment >= 0.5, short on adjustment <= -0.5, hold for 6 events."""

    def __init__(self, long_threshold: float = 0.5, short_threshold: float = -0.5, hold_events: int = 6) -> None:
        self.long_threshold = float(long_threshold)
        self.short_threshold = float(short_threshold)
        self.hold_events = int(hold_events)
        self._validate_config()
        self.reset()

    def reset(self) -> None:
        self.event_index = 0
        self.position = 0
        self.events_remaining = 0
        self.entry_price: float | None = None
        self.entry_adjustment: float | None = None
        self.entry_event_index: int | None = None
        self.exit_event_index: int | None = None
        self.mark_price: float | None = None
        self.realized_pnl = 0.0
        self.trades_closed = 0
        self.closed_trades: list[ClosedTrade] = []
        self.last_action = "Idle"

    def _validate_config(self) -> None:
        if not math.isfinite(self.long_threshold):
            raise ValueError("long_threshold must be finite.")
        if not math.isfinite(self.short_threshold):
            raise ValueError("short_threshold must be finite.")
        if self.long_threshold <= self.short_threshold:
            raise ValueError("long_threshold must be greater than short_threshold.")
        if self.hold_events <= 0:
            raise ValueError("hold_events must be positive.")

    def _snapshot(self, *, enabled: bool) -> StrategySnapshot:
        unrealized = 0.0
        if self.position != 0 and self.entry_price is not None and self.mark_price is not None:
            unrealized = self.position * (self.mark_price - self.entry_price)
        return StrategySnapshot(
            enabled=enabled,
            event_index=self.event_index,
            position=self.position,
            events_remaining=self.events_remaining,
            entry_price=self.entry_price,
            entry_adjustment=self.entry_adjustment,
            entry_event_index=self.entry_event_index,
            exit_event_index=self.exit_event_index,
            mark_price=self.mark_price,
            unrealized_pnl=unrealized,
            realized_pnl=self.realized_pnl,
            total_pnl=self.realized_pnl + unrealized,
            trades_closed=self.trades_closed,
            last_action=self.last_action,
        )

    def _close_position(self, *, price: float, event_index: int, exit_reason: str) -> None:
        if self.position == 0 or self.entry_price is None or self.entry_adjustment is None or self.entry_event_index is None:
            return
        trade_pnl = self.position * (price - self.entry_price)
        side = "Long" if self.position > 0 else "Short"
        self.realized_pnl += trade_pnl
        self.trades_closed += 1
        self.exit_event_index = event_index
        self.closed_trades.append(
            ClosedTrade(
                side=side.lower(),
                entry_price=self.entry_price,
                exit_price=price,
                entry_adjustment=self.entry_adjustment,
                entry_event_index=self.entry_event_index,
                exit_event_index=event_index,
                pnl=trade_pnl,
                exit_reason=exit_reason,
            )
        )
        self.position = 0
        self.entry_price = None
        self.entry_adjustment = None
        self.entry_event_index = None
        self.events_remaining = 0
        self.last_action = f"{side} exit @ {price:.6f} | pnl={trade_pnl:.6f} | reason={exit_reason}"

    def update(self, midprice: float, adjustment: float, *, enabled: bool = True) -> StrategySnapshot:
        price = float(midprice)
        signal = float(adjustment)
        if not math.isfinite(price):
            raise ValueError("midprice must be finite.")
        if not math.isfinite(signal):
            raise ValueError("adjustment must be finite.")

        self.event_index += 1
        self.mark_price = price

        if not enabled:
            if self.position != 0:
                self._close_position(price=price, event_index=self.event_index, exit_reason="disabled")
            else:
                self.last_action = "Disabled"
            return self._snapshot(enabled=False)

        if self.position == 0:
            if signal >= self.long_threshold:
                self.position = 1
                self.events_remaining = self.hold_events - 1
                self.entry_price = price
                self.entry_adjustment = signal
                self.entry_event_index = self.event_index
                self.last_action = f"Long entered @ {price:.6f}"
            elif signal <= self.short_threshold:
                self.position = -1
                self.events_remaining = self.hold_events - 1
                self.entry_price = price
                self.entry_adjustment = signal
                self.entry_event_index = self.event_index
                self.last_action = f"Short entered @ {price:.6f}"
            else:
                self.last_action = "Flat"
            return self._snapshot(enabled=True)

        self.events_remaining -= 1
        if self.events_remaining <= 0:
            self._close_position(price=price, event_index=self.event_index, exit_reason="timed")
        else:
            self.last_action = f"Holding {'Long' if self.position > 0 else 'Short'} ({self.events_remaining} left)"
        return self._snapshot(enabled=True)
