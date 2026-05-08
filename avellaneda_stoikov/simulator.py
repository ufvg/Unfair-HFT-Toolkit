from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator

import numpy as np

from .engine import AvellanedaStoikovEngine, Fill
from .model import AvellanedaStoikovModel, AvellanedaStoikovParameters, MarketSnapshot


@dataclass(slots=True)
class SimulationConfig:
    steps: int = 1000
    dt_seconds: float = 0.25
    initial_midprice: float = 100.0
    spread_ticks: int = 2
    drift_per_second: float = 0.0
    sigma: float = 0.015
    seed: int = 7
    enable_fills: bool = True


@dataclass(slots=True)
class SimulationStep:
    snapshot: MarketSnapshot
    reservation_price: float
    bid_price: float
    ask_price: float
    inventory: float
    cash: float
    pnl: float
    sigma: float
    bid_fill: bool
    ask_fill: bool


@dataclass(slots=True)
class SimulationResult:
    steps: list[SimulationStep]


def _simulation_generator(
    params: AvellanedaStoikovParameters,
    config: SimulationConfig,
) -> Iterator[SimulationStep]:
    rng = np.random.default_rng(config.seed)
    model = AvellanedaStoikovModel(params)
    engine = AvellanedaStoikovEngine(model)
    tick_size = params.tick_size
    half_spread = 0.5 * config.spread_ticks * tick_size
    midprice = config.initial_midprice
    timestamp = 0.0

    for _ in range(config.steps):
        shock = rng.normal()
        midprice += config.drift_per_second * config.dt_seconds
        midprice += config.sigma * math.sqrt(config.dt_seconds) * shock * midprice
        best_bid = midprice - half_spread
        best_ask = midprice + half_spread
        snapshot = MarketSnapshot(
            timestamp=timestamp,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_size=1.0,
            ask_size=1.0,
        )
        engine_snapshot = engine.on_market_snapshot(snapshot, dt_seconds=config.dt_seconds)
        quote = engine_snapshot.quote

        bid_fill = False
        ask_fill = False
        maker_fee_rate = params.maker_fee_bps * 1e-4
        if config.enable_fills:
            if rng.random() < quote.bid_fill_probability:
                bid_fill = True
                engine.apply_fill(
                    Fill(
                        side="buy",
                        price=quote.bid_price,
                        size=params.order_size,
                        timestamp=timestamp,
                        fee_paid=quote.bid_price * params.order_size * maker_fee_rate,
                    )
                )
            if rng.random() < quote.ask_fill_probability:
                ask_fill = True
                engine.apply_fill(
                    Fill(
                        side="sell",
                        price=quote.ask_price,
                        size=params.order_size,
                        timestamp=timestamp,
                        fee_paid=quote.ask_price * params.order_size * maker_fee_rate,
                    )
                )
        yield SimulationStep(
            snapshot=snapshot,
            reservation_price=quote.reservation_price,
            bid_price=quote.bid_price,
            ask_price=quote.ask_price,
            inventory=engine.state.inventory,
            cash=engine.state.cash,
            pnl=engine.state.mark_value(snapshot.midprice),
            sigma=engine_snapshot.sigma,
            bid_fill=bid_fill,
            ask_fill=ask_fill,
        )
        timestamp += config.dt_seconds


def simulate_avellaneda_stoikov(
    params: AvellanedaStoikovParameters,
    config: SimulationConfig | None = None,
) -> SimulationResult:
    sim_config = SimulationConfig() if config is None else config
    return SimulationResult(steps=list(_simulation_generator(params, sim_config)))


def iter_simulation(
    params: AvellanedaStoikovParameters,
    config: SimulationConfig | None = None,
) -> Iterator[SimulationStep]:
    sim_config = SimulationConfig() if config is None else config
    return _simulation_generator(params, sim_config)
