"""Passive fill hazard estimation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class FillHazardSample:
    distance: float
    fills: float
    exposure_seconds: float


@dataclass(frozen=True, slots=True)
class ExponentialFillHazardModel:
    intensity_scale: float
    decay: float

    def __post_init__(self) -> None:
        if self.intensity_scale <= 0.0 or not math.isfinite(self.intensity_scale):
            raise ValueError("intensity_scale must be positive and finite.")
        if self.decay <= 0.0 or not math.isfinite(self.decay):
            raise ValueError("decay must be positive and finite.")

    def hazard_rate(self, distance: float) -> float:
        delta = float(distance)
        if delta < 0.0:
            raise ValueError("distance must be nonnegative.")
        return self.intensity_scale * math.exp(-self.decay * delta)

    def survival_probability(self, distance: float, horizon_seconds: float) -> float:
        horizon = float(horizon_seconds)
        if horizon < 0.0:
            raise ValueError("horizon_seconds must be nonnegative.")
        return math.exp(-self.hazard_rate(distance) * horizon)

    def fill_probability(self, distance: float, horizon_seconds: float) -> float:
        return 1.0 - self.survival_probability(distance, horizon_seconds)


def fit_exponential_fill_hazard(
    samples: Iterable[FillHazardSample],
) -> ExponentialFillHazardModel:
    distances: list[float] = []
    log_rates: list[float] = []
    for sample in samples:
        if sample.distance < 0.0:
            raise ValueError("distance must be nonnegative.")
        if sample.exposure_seconds <= 0.0:
            continue
        if sample.fills <= 0.0:
            continue
        rate = float(sample.fills) / float(sample.exposure_seconds)
        if rate <= 0.0:
            continue
        distances.append(float(sample.distance))
        log_rates.append(math.log(rate))
    if len(distances) < 2:
        raise ValueError("Need at least two positive-rate samples to fit a fill hazard model.")
    slope, intercept = np.polyfit(np.asarray(distances, dtype=np.float64), np.asarray(log_rates, dtype=np.float64), 1)
    scale = float(math.exp(intercept))
    decay = float(-slope)
    return ExponentialFillHazardModel(intensity_scale=scale, decay=decay)


def empirical_fill_hazard(fills: float, exposure_seconds: float) -> float:
    exposure = float(exposure_seconds)
    if exposure <= 0.0:
        raise ValueError("exposure_seconds must be positive.")
    count = float(fills)
    if count < 0.0:
        raise ValueError("fills must be nonnegative.")
    return count / exposure


__all__ = [
    "ExponentialFillHazardModel",
    "FillHazardSample",
    "empirical_fill_hazard",
    "fit_exponential_fill_hazard",
]
