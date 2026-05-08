from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(slots=True)
class IntensitySample:
    distance: float
    fills: float
    exposure_seconds: float


def realized_volatility_per_sqrt_second(
    prices: Sequence[float],
    *,
    dt_seconds: float,
    log_returns: bool = True,
) -> float:
    if dt_seconds <= 0.0:
        raise ValueError("dt_seconds must be positive.")
    values = np.asarray(prices, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("prices must contain at least two observations.")
    if np.any(values <= 0.0):
        raise ValueError("prices must be strictly positive.")
    if log_returns:
        rets = np.diff(np.log(values))
    else:
        rets = np.diff(values) / values[:-1]
    variance = float(np.var(rets, ddof=1))
    return math.sqrt(max(variance, 0.0) / dt_seconds)


def fit_exponential_arrival_model(
    samples: Iterable[IntensitySample],
) -> tuple[float, float]:
    """Fit lambda(delta) = A * exp(-k * delta) from exposure-normalized fills."""

    distances: list[float] = []
    log_rates: list[float] = []
    for sample in samples:
        if sample.exposure_seconds <= 0.0:
            continue
        if sample.fills <= 0.0:
            continue
        rate = sample.fills / sample.exposure_seconds
        if rate <= 0.0:
            continue
        distances.append(float(sample.distance))
        log_rates.append(math.log(rate))
    if len(distances) < 2:
        raise ValueError("Need at least two positive-rate samples to fit A and k.")
    slope, intercept = np.polyfit(np.asarray(distances, dtype=float), np.asarray(log_rates, dtype=float), 1)
    arrival_rate_decay = float(-slope)
    arrival_rate_scale = float(math.exp(intercept))
    if not np.isfinite(arrival_rate_scale) or not np.isfinite(arrival_rate_decay):
        raise ValueError("Fitted arrival parameters are not finite.")
    if arrival_rate_scale <= 0.0 or arrival_rate_decay <= 0.0:
        raise ValueError("Fitted arrival parameters must be positive.")
    return arrival_rate_scale, arrival_rate_decay
