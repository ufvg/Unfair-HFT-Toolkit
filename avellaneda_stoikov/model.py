from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(slots=True)
class MarketSnapshot:
    timestamp: float
    best_bid: float
    best_ask: float
    bid_size: float
    ask_size: float

    @property
    def midprice(self) -> float:
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid


@dataclass(slots=True)
class AvellanedaStoikovParameters:
    gamma: float
    sigma: float
    arrival_rate_scale: float
    arrival_rate_decay: float
    horizon_seconds: float
    tick_size: float
    min_half_spread: float = 0.0
    max_half_spread: float | None = None
    inventory_target: float = 0.0
    inventory_limit: float = 5.0
    order_size: float = 0.01
    maker_fee_bps: float = 0.0
    quote_refresh_seconds: float = 0.25
    price_band_ticks: int = 200

    def __post_init__(self) -> None:
        if self.gamma < 0.0:
            raise ValueError("gamma must be non-negative.")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive.")
        if self.arrival_rate_scale <= 0.0:
            raise ValueError("arrival_rate_scale must be positive.")
        if self.arrival_rate_decay <= 0.0:
            raise ValueError("arrival_rate_decay must be positive.")
        if self.horizon_seconds <= 0.0:
            raise ValueError("horizon_seconds must be positive.")
        if self.tick_size <= 0.0:
            raise ValueError("tick_size must be positive.")
        if self.order_size <= 0.0:
            raise ValueError("order_size must be positive.")
        if self.inventory_limit < 0.0:
            raise ValueError("inventory_limit must be non-negative.")
        if self.price_band_ticks < 1:
            raise ValueError("price_band_ticks must be at least one tick.")


@dataclass(slots=True)
class QuoteDecision:
    timestamp: float
    reference_price: float
    reservation_price: float
    bid_price: float
    ask_price: float
    raw_bid_price: float
    raw_ask_price: float
    half_spread: float
    inventory_skew: float
    tau_seconds: float
    inventory: float
    sigma: float
    bid_intensity: float
    ask_intensity: float
    bid_fill_probability: float
    ask_fill_probability: float
    maker_fee_bps: float


class AvellanedaStoikovModel:
    """Paper-faithful quoting core with practical exchange constraints.

    The finite-horizon center and spread follow the classical Avellaneda-Stoikov
    formulation. We keep the implementation explicit and lightweight so it can be
    embedded in low-latency event loops without bringing in heavy dependencies.
    """

    def __init__(self, params: AvellanedaStoikovParameters) -> None:
        self.params = params

    @staticmethod
    def _log_term(gamma: float, decay: float) -> float:
        if gamma <= 0.0:
            return 1.0 / decay
        return math.log1p(gamma / decay) / gamma

    @staticmethod
    def _round_down(price: float, tick_size: float) -> float:
        return math.floor(price / tick_size) * tick_size

    @staticmethod
    def _round_up(price: float, tick_size: float) -> float:
        return math.ceil(price / tick_size) * tick_size

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return min(max(value, lower), upper)

    def compute_quote(
        self,
        snapshot: MarketSnapshot,
        inventory: float,
        *,
        now_timestamp: float | None = None,
        reference_price: float | None = None,
        reservation_price_adjustment: float = 0.0,
        sigma: float | None = None,
        dt_seconds: float | None = None,
    ) -> QuoteDecision:
        params = self.params
        now = snapshot.timestamp if now_timestamp is None else now_timestamp
        ref_price = snapshot.midprice if reference_price is None else float(reference_price)
        sigma_value = params.sigma if sigma is None else max(float(sigma), 1e-12)
        dt_value = params.quote_refresh_seconds if dt_seconds is None else max(float(dt_seconds), 1e-9)
        tau_seconds = max(params.horizon_seconds - max(now - snapshot.timestamp, 0.0), 0.0)

        inventory_offset = float(inventory) - params.inventory_target
        reservation_price = (
            ref_price
            - inventory_offset * params.gamma * sigma_value * sigma_value * tau_seconds
            + float(reservation_price_adjustment)
        )
        half_spread = 0.5 * params.gamma * sigma_value * sigma_value * tau_seconds + self._log_term(
            params.gamma,
            params.arrival_rate_decay,
        )
        half_spread = max(half_spread, params.min_half_spread)
        if params.max_half_spread is not None:
            half_spread = min(half_spread, params.max_half_spread)

        raw_bid = reservation_price - half_spread
        raw_ask = reservation_price + half_spread

        band = params.price_band_ticks * params.tick_size
        raw_bid = self._clamp(raw_bid, snapshot.midprice - band, snapshot.midprice)
        raw_ask = self._clamp(raw_ask, snapshot.midprice, snapshot.midprice + band)

        bid_price = self._round_down(raw_bid, params.tick_size)
        ask_price = self._round_up(raw_ask, params.tick_size)
        if ask_price <= bid_price:
            bid_price = self._round_down(reservation_price - params.tick_size, params.tick_size)
            ask_price = self._round_up(reservation_price + params.tick_size, params.tick_size)

        max_bid = snapshot.best_ask - params.tick_size
        min_ask = snapshot.best_bid + params.tick_size
        bid_price = min(bid_price, max_bid)
        ask_price = max(ask_price, min_ask)
        if ask_price <= bid_price:
            bid_price = self._round_down(snapshot.best_bid, params.tick_size)
            ask_price = self._round_up(snapshot.best_ask, params.tick_size)

        if inventory >= params.inventory_limit:
            bid_intensity = 0.0
            bid_probability = 0.0
        else:
            bid_distance = max(ref_price - bid_price, 0.0)
            bid_intensity = params.arrival_rate_scale * math.exp(-params.arrival_rate_decay * bid_distance)
            bid_probability = 1.0 - math.exp(-bid_intensity * dt_value)
        if inventory <= -params.inventory_limit:
            ask_intensity = 0.0
            ask_probability = 0.0
        else:
            ask_distance = max(ask_price - ref_price, 0.0)
            ask_intensity = params.arrival_rate_scale * math.exp(-params.arrival_rate_decay * ask_distance)
            ask_probability = 1.0 - math.exp(-ask_intensity * dt_value)

        return QuoteDecision(
            timestamp=now,
            reference_price=ref_price,
            reservation_price=reservation_price,
            bid_price=bid_price,
            ask_price=ask_price,
            raw_bid_price=raw_bid,
            raw_ask_price=raw_ask,
            half_spread=half_spread,
            inventory_skew=reservation_price - ref_price,
            tau_seconds=tau_seconds,
            inventory=inventory,
            sigma=sigma_value,
            bid_intensity=bid_intensity,
            ask_intensity=ask_intensity,
            bid_fill_probability=bid_probability,
            ask_fill_probability=ask_probability,
            maker_fee_bps=params.maker_fee_bps,
        )
