"""Shared input validation helpers for the microprice library."""

from __future__ import annotations

import math


def _validate_positive(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return v


def _validate_nonnegative(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite.")
    return v


def _validate_finite(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite.")
    return v


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    v = int(value)
    if v <= 0 or float(v) != float(value):
        raise ValueError(f"{name} must be a positive integer.")
    return v
