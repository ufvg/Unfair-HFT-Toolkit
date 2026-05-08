"""Kyle lambda estimation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class KyleLambdaEstimate:
    lambda_: float
    intercept: float
    r_squared: float | None
    sample_size: int


def fit_kyle_lambda(
    signed_flow: Any,
    price_change: Any,
    *,
    fit_intercept: bool = False,
) -> KyleLambdaEstimate:
    flow = np.asarray(signed_flow, dtype=np.float64)
    change = np.asarray(price_change, dtype=np.float64)
    if flow.shape != change.shape:
        raise ValueError("signed_flow and price_change must have the same shape.")
    if flow.ndim != 1:
        raise ValueError("signed_flow and price_change must be one-dimensional.")
    valid = np.isfinite(flow) & np.isfinite(change)
    flow = flow[valid]
    change = change[valid]
    if flow.size < 2:
        raise ValueError("Need at least two valid observations to fit Kyle lambda.")

    if fit_intercept:
        flow_mean = float(np.mean(flow))
        change_mean = float(np.mean(change))
        centered_flow = flow - flow_mean
        variance = float(np.dot(centered_flow, centered_flow))
        if variance <= 0.0:
            raise ValueError("signed_flow variance must be positive.")
        covariance = float(np.dot(centered_flow, change - change_mean))
        slope = covariance / variance
        intercept = change_mean - slope * flow_mean
        fitted = intercept + slope * flow
    else:
        variance = float(np.dot(flow, flow))
        if variance <= 0.0:
            raise ValueError("signed_flow variance must be positive.")
        slope = float(np.dot(flow, change) / variance)
        intercept = 0.0
        fitted = slope * flow

    residual = change - fitted
    total_ss = float(np.dot(change - np.mean(change), change - np.mean(change)))
    r_squared = None if total_ss <= 0.0 else float(1.0 - (np.dot(residual, residual) / total_ss))
    return KyleLambdaEstimate(
        lambda_=float(slope),
        intercept=float(intercept),
        r_squared=r_squared,
        sample_size=int(flow.size),
    )


def rolling_kyle_lambda(
    signed_flow: Any,
    price_change: Any,
    *,
    window: int,
    fit_intercept: bool = False,
) -> np.ndarray:
    width = int(window)
    if width <= 1:
        raise ValueError("window must be at least 2.")
    flow = np.asarray(signed_flow, dtype=np.float64)
    change = np.asarray(price_change, dtype=np.float64)
    if flow.shape != change.shape:
        raise ValueError("signed_flow and price_change must have the same shape.")
    if flow.ndim != 1:
        raise ValueError("signed_flow and price_change must be one-dimensional.")
    result = np.full(flow.shape, np.nan, dtype=np.float64)
    for index in range(width - 1, flow.size):
        estimate = fit_kyle_lambda(
            flow[index - width + 1 : index + 1],
            change[index - width + 1 : index + 1],
            fit_intercept=fit_intercept,
        )
        result[index] = estimate.lambda_
    return result


__all__ = [
    "KyleLambdaEstimate",
    "fit_kyle_lambda",
    "rolling_kyle_lambda",
]
