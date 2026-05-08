from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import math
from typing import Iterable, Literal

from microstructure.trades import TradeSide, coerce_trade_timestamp, normalize_trade_side

from .validation import _validate_positive, _validate_positive_int

ClassificationMethod = Literal["bvc", "tick", "aggressor"]


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


@dataclass(frozen=True, slots=True)
class TradePrint:
    timestamp: datetime
    price: float
    volume: float
    side: TradeSide | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", coerce_trade_timestamp(self.timestamp))
        object.__setattr__(self, "price", _validate_positive(self.price, "price"))
        object.__setattr__(self, "volume", _validate_positive(self.volume, "volume"))
        object.__setattr__(self, "side", normalize_trade_side(self.side))


@dataclass(frozen=True, slots=True)
class TimeBar:
    start_time: datetime
    end_time: datetime
    close_price: float
    volume: float
    price_change: float
    buy_volume: float
    sell_volume: float


@dataclass(frozen=True, slots=True)
class VolumeBucket:
    index: int
    start_time: datetime
    end_time: datetime
    volume: float
    buy_volume: float
    sell_volume: float
    close_price: float
    imbalance_ratio: float


@dataclass(frozen=True, slots=True)
class VPINPoint:
    bucket_index: int
    end_time: datetime
    bucket_imbalance: float
    vpin: float | None


@dataclass(frozen=True, slots=True)
class BucketSizeSelection:
    bucket_volume: float
    average_daily_volume: float
    buckets_per_day: int
    sample_days: int
    source: str


@dataclass(frozen=True, slots=True)
class VPINConfig:
    time_bar_seconds: int = 60
    buckets_per_day: int = 50
    support_buckets: int = 50
    bucket_volume: float | None = None
    average_daily_volume: float | None = None
    classification: ClassificationMethod = "bvc"

    def __post_init__(self) -> None:
        _validate_positive_int(self.time_bar_seconds, "time_bar_seconds")
        _validate_positive_int(self.buckets_per_day, "buckets_per_day")
        _validate_positive_int(self.support_buckets, "support_buckets")
        if self.bucket_volume is not None:
            _validate_positive(self.bucket_volume, "bucket_volume")
        if self.average_daily_volume is not None:
            _validate_positive(self.average_daily_volume, "average_daily_volume")
        if self.classification not in {"bvc", "tick", "aggressor"}:
            raise ValueError("classification must be 'bvc', 'tick', or 'aggressor'.")


@dataclass(frozen=True, slots=True)
class VPINResult:
    config: VPINConfig
    bucket_size: BucketSizeSelection
    trades: tuple[TradePrint, ...]
    time_bars: tuple[TimeBar, ...]
    volume_buckets: tuple[VolumeBucket, ...]
    points: tuple[VPINPoint, ...]

    @property
    def latest_vpin(self) -> float | None:
        for point in reversed(self.points):
            if point.vpin is not None:
                return point.vpin
        return None


@dataclass(frozen=True, slots=True)
class OnlineVPINSnapshot:
    bucket_index: int
    completed_buckets: int
    current_bucket_fill: float
    current_vpin: float | None
    last_bucket_imbalance: float | None
    latest_trade_time: datetime


def normalize_trades(
    trades: Iterable[
        TradePrint
        | tuple[datetime | str | int | float, float, float]
        | tuple[datetime | str | int | float, float, float, int | float | str | None]
    ],
) -> tuple[TradePrint, ...]:
    normalized: list[TradePrint] = []
    for record in trades:
        if isinstance(record, TradePrint):
            normalized.append(record)
            continue
        if len(record) == 3:
            timestamp, price, volume = record
            side = None
        elif len(record) == 4:
            timestamp, price, volume, side = record
        else:
            raise ValueError("trade tuples must have 3 fields (timestamp, price, volume) or 4 including side.")
        normalized.append(TradePrint(timestamp=timestamp, price=price, volume=volume, side=side))
    if not normalized:
        raise ValueError("at least one trade is required.")
    normalized.sort(key=lambda trade: trade.timestamp)
    return tuple(normalized)


def estimate_average_daily_volume(trades: Iterable[TradePrint]) -> tuple[float, dict[str, float]]:
    daily_volume: dict[str, float] = {}
    count = 0
    for trade in trades:
        count += 1
        key = trade.timestamp.date().isoformat()
        daily_volume[key] = daily_volume.get(key, 0.0) + trade.volume
    if count == 0:
        raise ValueError("at least one trade is required to estimate average daily volume.")
    average = sum(daily_volume.values()) / len(daily_volume)
    return average, daily_volume


def select_bucket_size(
    trades: Iterable[TradePrint],
    *,
    buckets_per_day: int = 50,
    average_daily_volume: float | None = None,
    bucket_volume: float | None = None,
) -> BucketSizeSelection:
    buckets = _validate_positive_int(buckets_per_day, "buckets_per_day")
    normalized_trades = tuple(trades)
    if bucket_volume is not None:
        bucket = _validate_positive(bucket_volume, "bucket_volume")
        if average_daily_volume is None:
            estimated_adv, daily = estimate_average_daily_volume(normalized_trades)
            return BucketSizeSelection(
                bucket_volume=bucket,
                average_daily_volume=estimated_adv,
                buckets_per_day=buckets,
                sample_days=len(daily),
                source="manual_bucket_volume",
            )
        return BucketSizeSelection(
            bucket_volume=bucket,
            average_daily_volume=_validate_positive(average_daily_volume, "average_daily_volume"),
            buckets_per_day=buckets,
            sample_days=0,
            source="manual_bucket_volume",
        )

    if average_daily_volume is None:
        average_daily_volume, daily = estimate_average_daily_volume(normalized_trades)
        sample_days = len(daily)
        source = "sample_average_daily_volume"
    else:
        average_daily_volume = _validate_positive(average_daily_volume, "average_daily_volume")
        sample_days = 0
        source = "provided_average_daily_volume"

    bucket = average_daily_volume / buckets
    return BucketSizeSelection(
        bucket_volume=bucket,
        average_daily_volume=average_daily_volume,
        buckets_per_day=buckets,
        sample_days=sample_days,
        source=source,
    )


def build_time_bars(trades: Iterable[TradePrint], time_bar_seconds: int = 60) -> tuple[TimeBar, ...]:
    seconds = _validate_positive_int(time_bar_seconds, "time_bar_seconds")
    trade_list = tuple(trades)
    if not trade_list:
        raise ValueError("at least one trade is required.")

    bars: list[tuple[datetime, datetime, float, float]] = []
    current_start: datetime | None = None
    current_end: datetime | None = None
    current_close = 0.0
    current_volume = 0.0

    for trade in trade_list:
        epoch_seconds = int(trade.timestamp.timestamp())
        bar_start_epoch = epoch_seconds - (epoch_seconds % seconds)
        bar_start = datetime.fromtimestamp(bar_start_epoch, tz=UTC)
        bar_end = datetime.fromtimestamp(bar_start_epoch + seconds, tz=UTC)

        if current_start is None:
            current_start = bar_start
            current_end = bar_end
        elif bar_start != current_start:
            bars.append((current_start, current_end, current_close, current_volume))
            current_start = bar_start
            current_end = bar_end
            current_volume = 0.0

        current_close = trade.price
        current_volume += trade.volume

    if current_start is None or current_end is None:
        raise ValueError("unable to build time bars from trades.")
    bars.append((current_start, current_end, current_close, current_volume))

    closes = [bar[2] for bar in bars]
    price_changes = [0.0]
    for index in range(1, len(closes)):
        price_changes.append(closes[index] - closes[index - 1])

    if len(price_changes) > 1:
        mean = sum(price_changes[1:]) / (len(price_changes) - 1)
        variance = sum((change - mean) ** 2 for change in price_changes[1:]) / max(len(price_changes) - 1, 1)
        sigma = math.sqrt(variance)
    else:
        sigma = 0.0

    time_bars: list[TimeBar] = []
    for (start_time, end_time, close_price, volume), price_change in zip(bars, price_changes, strict=True):
        if sigma <= 0.0:
            buy_fraction = 0.5
        else:
            buy_fraction = _normal_cdf(price_change / sigma)
        buy_volume = volume * buy_fraction
        sell_volume = volume - buy_volume
        time_bars.append(
            TimeBar(
                start_time=start_time,
                end_time=end_time,
                close_price=close_price,
                volume=volume,
                price_change=price_change,
                buy_volume=buy_volume,
                sell_volume=sell_volume,
            )
        )
    return tuple(time_bars)


def build_volume_buckets(
    time_bars: Iterable[TimeBar],
    bucket_volume: float,
) -> tuple[VolumeBucket, ...]:
    target_volume = _validate_positive(bucket_volume, "bucket_volume")
    epsilon = target_volume * 1e-12
    buckets: list[VolumeBucket] = []
    current_volume = 0.0
    current_buy = 0.0
    current_sell = 0.0
    current_start: datetime | None = None
    current_end: datetime | None = None

    for bar in time_bars:
        remaining_volume = bar.volume
        remaining_buy = bar.buy_volume
        remaining_sell = bar.sell_volume
        if current_start is None:
            current_start = bar.start_time

        while remaining_volume > epsilon:
            available = target_volume - current_volume
            take = min(available, remaining_volume)
            ratio = take / remaining_volume if remaining_volume > 0.0 else 0.0
            take_buy = remaining_buy * ratio
            take_sell = remaining_sell * ratio

            current_volume += take
            current_buy += take_buy
            current_sell += take_sell
            current_end = bar.end_time

            remaining_volume -= take
            remaining_buy -= take_buy
            remaining_sell -= take_sell

            if current_volume + epsilon >= target_volume:
                imbalance = abs(current_buy - current_sell) / current_volume
                buckets.append(
                    VolumeBucket(
                        index=len(buckets),
                        start_time=current_start,
                        end_time=current_end,
                        volume=current_volume,
                        buy_volume=current_buy,
                        sell_volume=current_sell,
                        close_price=bar.close_price,
                        imbalance_ratio=imbalance,
                    )
                )
                current_volume = 0.0
                current_buy = 0.0
                current_sell = 0.0
                current_start = bar.end_time
                current_end = None

    return tuple(buckets)


def _tick_rule_side(
    trade: TradePrint,
    *,
    previous_price: float | None,
    previous_nonzero_side: TradeSide | None,
) -> tuple[TradeSide | None, TradeSide | None]:
    if previous_price is None:
        return None, previous_nonzero_side
    if trade.price > previous_price:
        return 1, 1
    if trade.price < previous_price:
        return -1, -1
    return previous_nonzero_side, previous_nonzero_side


def build_trade_buckets(
    trades: Iterable[TradePrint],
    *,
    bucket_volume: float,
    classification: ClassificationMethod,
) -> tuple[VolumeBucket, ...]:
    if classification == "bvc":
        raise ValueError("build_trade_buckets does not support 'bvc'; use build_time_bars + build_volume_buckets.")

    target_volume = _validate_positive(bucket_volume, "bucket_volume")
    epsilon = target_volume * 1e-12
    buckets: list[VolumeBucket] = []
    current_volume = 0.0
    current_buy = 0.0
    current_sell = 0.0
    current_start: datetime | None = None
    current_end: datetime | None = None
    current_close = 0.0
    previous_price: float | None = None
    previous_nonzero_side: TradeSide | None = None

    for trade in trades:
        if classification == "aggressor":
            side = trade.side
            if side is None:
                raise ValueError("aggressor classification requires a side on every trade.")
        else:
            side, previous_nonzero_side = _tick_rule_side(
                trade,
                previous_price=previous_price,
                previous_nonzero_side=previous_nonzero_side,
            )

        remaining_volume = trade.volume
        if current_start is None:
            current_start = trade.timestamp

        while remaining_volume > epsilon:
            available = target_volume - current_volume
            take = min(available, remaining_volume)
            if side == 1:
                take_buy = take
                take_sell = 0.0
            elif side == -1:
                take_buy = 0.0
                take_sell = take
            else:
                take_buy = 0.5 * take
                take_sell = 0.5 * take

            current_volume += take
            current_buy += take_buy
            current_sell += take_sell
            current_end = trade.timestamp
            current_close = trade.price
            remaining_volume -= take

            if current_volume + epsilon >= target_volume:
                imbalance = abs(current_buy - current_sell) / current_volume
                buckets.append(
                    VolumeBucket(
                        index=len(buckets),
                        start_time=current_start,
                        end_time=current_end,
                        volume=current_volume,
                        buy_volume=current_buy,
                        sell_volume=current_sell,
                        close_price=current_close,
                        imbalance_ratio=imbalance,
                    )
                )
                current_volume = 0.0
                current_buy = 0.0
                current_sell = 0.0
                current_start = trade.timestamp
                current_end = None

        previous_price = trade.price

    return tuple(buckets)


def _build_points(volume_buckets: Iterable[VolumeBucket], support_buckets: int) -> tuple[VPINPoint, ...]:
    rolling: deque[float] = deque(maxlen=support_buckets)
    points: list[VPINPoint] = []
    for bucket in volume_buckets:
        rolling.append(bucket.imbalance_ratio)
        vpin = None
        if len(rolling) == support_buckets:
            vpin = sum(rolling) / support_buckets
        points.append(
            VPINPoint(
                bucket_index=bucket.index,
                end_time=bucket.end_time,
                bucket_imbalance=bucket.imbalance_ratio,
                vpin=vpin,
            )
        )
    return tuple(points)


def compute_vpin(
    trades: Iterable[
        TradePrint
        | tuple[datetime | str | int | float, float, float]
        | tuple[datetime | str | int | float, float, float, int | float | str | None]
    ],
    config: VPINConfig = VPINConfig(),
) -> VPINResult:
    normalized_trades = normalize_trades(trades)
    selection = select_bucket_size(
        normalized_trades,
        buckets_per_day=config.buckets_per_day,
        average_daily_volume=config.average_daily_volume,
        bucket_volume=config.bucket_volume,
    )

    if config.classification == "bvc":
        time_bars = build_time_bars(normalized_trades, time_bar_seconds=config.time_bar_seconds)
        volume_buckets = build_volume_buckets(time_bars, bucket_volume=selection.bucket_volume)
    else:
        time_bars = ()
        volume_buckets = build_trade_buckets(
            normalized_trades,
            bucket_volume=selection.bucket_volume,
            classification=config.classification,
        )

    if len(volume_buckets) < config.support_buckets:
        raise ValueError(
            f"need at least {config.support_buckets} completed volume buckets for VPIN; got {len(volume_buckets)}"
        )

    return VPINResult(
        config=config,
        bucket_size=selection,
        trades=normalized_trades,
        time_bars=tuple(time_bars),
        volume_buckets=tuple(volume_buckets),
        points=_build_points(volume_buckets, config.support_buckets),
    )


def extract_vpin_series(result: VPINResult) -> tuple[list[datetime], list[float]]:
    timestamps: list[datetime] = []
    values: list[float] = []
    for point in result.points:
        if point.vpin is None:
            continue
        timestamps.append(point.end_time)
        values.append(point.vpin)
    return timestamps, values


def summarize_vpin(result: VPINResult) -> dict[str, float]:
    values = [point.vpin for point in result.points if point.vpin is not None]
    if not values:
        raise ValueError("result does not contain a completed VPIN window.")
    sorted_values = sorted(values)
    percentile_index = min(max(int(round(0.95 * (len(sorted_values) - 1))), 0), len(sorted_values) - 1)
    return {
        "latest_vpin": values[-1],
        "mean_vpin": sum(values) / len(values),
        "max_vpin": max(values),
        "p95_vpin": sorted_values[percentile_index],
    }


class OnlineVPIN:
    """Incremental VPIN engine optimized for streaming HFT trade feeds.

    For low-latency applications, prefer `aggressor` when your feed provides a
    buyer/seller aggressor flag. `tick` is the lightweight fallback when only
    trade prints are available. `bvc` is intentionally excluded here because it
    requires time-bar aggregation and is less suitable for event-driven HFT.
    """

    def __init__(
        self,
        *,
        bucket_volume: float,
        support_buckets: int = 50,
        classification: ClassificationMethod = "aggressor",
    ) -> None:
        self.bucket_volume = _validate_positive(bucket_volume, "bucket_volume")
        self.support_buckets = _validate_positive_int(support_buckets, "support_buckets")
        if classification not in {"tick", "aggressor"}:
            raise ValueError("OnlineVPIN classification must be 'tick' or 'aggressor'.")
        self.classification = classification

        self._rolling: deque[float] = deque(maxlen=self.support_buckets)
        self._rolling_sum = 0.0
        self._completed_buckets = 0
        self._current_volume = 0.0
        self._current_buy = 0.0
        self._current_sell = 0.0
        self._previous_price: float | None = None
        self._previous_nonzero_side: TradeSide | None = None
        self._last_bucket_imbalance: float | None = None
        self._latest_trade_time: datetime | None = None

    @property
    def completed_buckets(self) -> int:
        return self._completed_buckets

    @property
    def current_vpin(self) -> float | None:
        if len(self._rolling) < self.support_buckets:
            return None
        return self._rolling_sum / self.support_buckets

    def _push_imbalance(self, imbalance: float) -> None:
        if len(self._rolling) == self.support_buckets:
            self._rolling_sum -= self._rolling[0]
        self._rolling.append(imbalance)
        self._rolling_sum += imbalance
        self._last_bucket_imbalance = imbalance
        self._completed_buckets += 1

    def update(self, trade: TradePrint) -> OnlineVPINSnapshot:
        self._latest_trade_time = trade.timestamp
        if self.classification == "aggressor":
            side = trade.side
            if side is None:
                raise ValueError("OnlineVPIN aggressor mode requires trade.side on every trade.")
        else:
            side, self._previous_nonzero_side = _tick_rule_side(
                trade,
                previous_price=self._previous_price,
                previous_nonzero_side=self._previous_nonzero_side,
            )

        remaining_volume = trade.volume
        epsilon = self.bucket_volume * 1e-12
        while remaining_volume > epsilon:
            available = self.bucket_volume - self._current_volume
            take = min(available, remaining_volume)
            if side == 1:
                self._current_buy += take
            elif side == -1:
                self._current_sell += take
            else:
                half = 0.5 * take
                self._current_buy += half
                self._current_sell += half
            self._current_volume += take
            remaining_volume -= take

            if self._current_volume + epsilon >= self.bucket_volume:
                imbalance = abs(self._current_buy - self._current_sell) / self._current_volume
                self._push_imbalance(imbalance)
                self._current_volume = 0.0
                self._current_buy = 0.0
                self._current_sell = 0.0

        self._previous_price = trade.price
        return self.snapshot()

    def snapshot(self) -> OnlineVPINSnapshot:
        if self._latest_trade_time is None:
            raise ValueError("OnlineVPIN has not processed any trades yet.")
        return OnlineVPINSnapshot(
            bucket_index=self._completed_buckets - 1,
            completed_buckets=self._completed_buckets,
            current_bucket_fill=self._current_volume / self.bucket_volume,
            current_vpin=self.current_vpin,
            last_bucket_imbalance=self._last_bucket_imbalance,
            latest_trade_time=self._latest_trade_time,
        )
