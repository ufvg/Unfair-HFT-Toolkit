from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StrategySnapshot:
    enabled: bool
    position: int
    events_remaining: int
    entry_price: float | None
    mark_price: float | None
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    trades_closed: int
    last_action: str


class AdjustmentThresholdStrategy:
    """Long on adjustment >= 0.5, short on adjustment <= -0.5, hold for 6 events."""

    def __init__(self, long_threshold: float = 0.5, short_threshold: float = -0.5, hold_events: int = 6) -> None:
        self.long_threshold = float(long_threshold)
        self.short_threshold = float(short_threshold)
        self.hold_events = int(hold_events)
        if self.hold_events <= 0:
            raise ValueError("hold_events must be positive.")
        self.reset()

    def reset(self) -> None:
        self.position = 0
        self.events_remaining = 0
        self.entry_price: float | None = None
        self.mark_price: float | None = None
        self.realized_pnl = 0.0
        self.trades_closed = 0
        self.last_action = "Idle"

    def _snapshot(self, *, enabled: bool) -> StrategySnapshot:
        unrealized = 0.0
        if self.position != 0 and self.entry_price is not None and self.mark_price is not None:
            unrealized = self.position * (self.mark_price - self.entry_price)
        return StrategySnapshot(
            enabled=enabled,
            position=self.position,
            events_remaining=self.events_remaining,
            entry_price=self.entry_price,
            mark_price=self.mark_price,
            unrealized_pnl=unrealized,
            realized_pnl=self.realized_pnl,
            total_pnl=self.realized_pnl + unrealized,
            trades_closed=self.trades_closed,
            last_action=self.last_action,
        )

    def update(self, midprice: float, adjustment: float, *, enabled: bool = True) -> StrategySnapshot:
        price = float(midprice)
        signal = float(adjustment)
        self.mark_price = price

        if not enabled:
            self.last_action = "Disabled"
            return self._snapshot(enabled=False)

        if self.position == 0:
            if signal >= self.long_threshold:
                self.position = 1
                self.events_remaining = self.hold_events
                self.entry_price = price
                self.last_action = f"Long entered @ {price:.6f}"
            elif signal <= self.short_threshold:
                self.position = -1
                self.events_remaining = self.hold_events
                self.entry_price = price
                self.last_action = f"Short entered @ {price:.6f}"
            else:
                self.last_action = "Flat"
            return self._snapshot(enabled=True)

        self.events_remaining -= 1
        if self.events_remaining <= 0 and self.entry_price is not None:
            trade_pnl = self.position * (price - self.entry_price)
            side = "Long" if self.position > 0 else "Short"
            self.realized_pnl += trade_pnl
            self.trades_closed += 1
            self.position = 0
            self.entry_price = None
            self.events_remaining = 0
            self.last_action = f"{side} exit @ {price:.6f} | pnl={trade_pnl:.6f}"
        else:
            self.last_action = f"Holding {'Long' if self.position > 0 else 'Short'} ({self.events_remaining} left)"
        return self._snapshot(enabled=True)
