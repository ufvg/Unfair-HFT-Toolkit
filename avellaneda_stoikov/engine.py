from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .model import AvellanedaStoikovModel, MarketSnapshot, QuoteDecision


@dataclass(slots=True)
class Fill:
    side: str
    price: float
    size: float
    timestamp: float
    fee_paid: float = 0.0


@dataclass(slots=True)
class InventoryState:
    inventory: float = 0.0
    cash: float = 0.0
    realized_fees: float = 0.0
    fills: list[Fill] = field(default_factory=list)

    def mark_value(self, reference_price: float) -> float:
        return self.cash + self.inventory * reference_price


@dataclass(slots=True)
class EngineSnapshot:
    market: MarketSnapshot
    quote: QuoteDecision
    state: InventoryState
    mark_to_market_pnl: float
    sigma: float


class EWMAVariance:
    def __init__(self, alpha: float, initial_sigma: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        if initial_sigma <= 0.0:
            raise ValueError("initial_sigma must be positive.")
        self.alpha = alpha
        self.variance = initial_sigma * initial_sigma
        self.last_price: float | None = None
        self.last_timestamp: float | None = None

    def update(self, price: float, timestamp: float) -> float:
        if self.last_price is None or self.last_timestamp is None:
            self.last_price = price
            self.last_timestamp = timestamp
            return self.sigma
        dt = max(timestamp - self.last_timestamp, 1e-9)
        ret = (price - self.last_price) / self.last_price
        realized_var = (ret * ret) / dt
        self.variance = self.alpha * realized_var + (1.0 - self.alpha) * self.variance
        self.last_price = price
        self.last_timestamp = timestamp
        return self.sigma

    @property
    def sigma(self) -> float:
        return self.variance ** 0.5


class AvellanedaStoikovEngine:
    def __init__(
        self,
        model: AvellanedaStoikovModel,
        *,
        ewma_alpha: float = 0.08,
    ) -> None:
        self.model = model
        self.state = InventoryState()
        self.volatility = EWMAVariance(alpha=ewma_alpha, initial_sigma=model.params.sigma)
        self.last_snapshot: EngineSnapshot | None = None

    def on_market_snapshot(
        self,
        snapshot: MarketSnapshot,
        *,
        reference_price: float | None = None,
        reservation_price_adjustment: float = 0.0,
        dt_seconds: float | None = None,
    ) -> EngineSnapshot:
        sigma = self.volatility.update(snapshot.midprice, snapshot.timestamp)
        quote = self.model.compute_quote(
            snapshot,
            self.state.inventory,
            reference_price=reference_price,
            reservation_price_adjustment=reservation_price_adjustment,
            sigma=sigma,
            dt_seconds=dt_seconds,
        )
        engine_snapshot = EngineSnapshot(
            market=snapshot,
            quote=quote,
            state=self._copy_state(),
            mark_to_market_pnl=self.state.mark_value(snapshot.midprice),
            sigma=sigma,
        )
        self.last_snapshot = engine_snapshot
        return engine_snapshot

    def apply_fill(self, fill: Fill) -> InventoryState:
        notional = fill.price * fill.size
        if fill.side == "buy":
            self.state.inventory += fill.size
            self.state.cash -= notional + fill.fee_paid
        elif fill.side == "sell":
            self.state.inventory -= fill.size
            self.state.cash += notional - fill.fee_paid
        else:
            raise ValueError("fill.side must be 'buy' or 'sell'.")
        self.state.realized_fees += fill.fee_paid
        self.state.fills.append(fill)
        return self._copy_state()

    def apply_fills(self, fills: Iterable[Fill]) -> InventoryState:
        for fill in fills:
            self.apply_fill(fill)
        return self._copy_state()

    def reset(self) -> None:
        self.state = InventoryState()
        self.volatility = EWMAVariance(alpha=self.volatility.alpha, initial_sigma=self.model.params.sigma)
        self.last_snapshot = None

    def _copy_state(self) -> InventoryState:
        return InventoryState(
            inventory=self.state.inventory,
            cash=self.state.cash,
            realized_fees=self.state.realized_fees,
            fills=list(self.state.fills),
        )
