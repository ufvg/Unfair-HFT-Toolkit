"""Reservation price adjustment helpers for market making models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReservationPriceAdjustment:
    total_adjustment: float
    adjusted_reference_price: float
    microprice_component: float = 0.0
    obi_component: float = 0.0
    ofi_component: float = 0.0
    toxicity_component: float = 0.0
    impact_component: float = 0.0


@dataclass(frozen=True, slots=True)
class LinearReservationPriceAdjuster:
    microprice_weight: float = 1.0
    obi_weight: float = 0.0
    ofi_weight: float = 0.0
    toxicity_weight: float = 0.0
    impact_weight: float = 1.0
    max_adjustment: float | None = None

    def compute(
        self,
        *,
        reference_price: float,
        microprice: float | None = None,
        obi: float | None = None,
        ofi: float | None = None,
        toxicity: float | None = None,
        kyle_lambda: float | None = None,
        signed_flow: float | None = None,
    ) -> ReservationPriceAdjustment:
        ref = float(reference_price)
        microprice_component = 0.0 if microprice is None else self.microprice_weight * (float(microprice) - ref)

        obi_component = 0.0
        if obi is not None:
            obi_value = float(obi)
            centered_obi = obi_value - 0.5 if 0.0 <= obi_value <= 1.0 else obi_value
            obi_component = self.obi_weight * centered_obi

        ofi_component = 0.0 if ofi is None else self.ofi_weight * float(ofi)

        toxicity_component = 0.0
        if toxicity is not None:
            toxicity_component = self.toxicity_weight * (float(toxicity) - 0.5)

        impact_component = 0.0
        if kyle_lambda is not None and signed_flow is not None:
            impact_component = self.impact_weight * float(kyle_lambda) * float(signed_flow)

        total = microprice_component + obi_component + ofi_component + toxicity_component + impact_component
        if self.max_adjustment is not None:
            limit = abs(float(self.max_adjustment))
            total = min(max(total, -limit), limit)
        return ReservationPriceAdjustment(
            total_adjustment=total,
            adjusted_reference_price=ref + total,
            microprice_component=microprice_component,
            obi_component=obi_component,
            ofi_component=ofi_component,
            toxicity_component=toxicity_component,
            impact_component=impact_component,
        )


__all__ = [
    "LinearReservationPriceAdjuster",
    "ReservationPriceAdjustment",
]
