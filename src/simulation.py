"""Discrete-event approximation of rural edge/cloud review capacity."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapacityEstimate:
    patients: int
    ingestion_seconds: float
    processing_seconds: float
    review_seconds: float
    bottleneck: str


def estimate_capacity(
    patients: int = 100_000,
    ingestion_rate: float = 10.0,
    processing_rate: float = 4.0,
    review_rate: float = 0.5,
) -> CapacityEstimate:
    """Estimate stage completion times; replace with SimEvents for deployment studies."""
    values = {
        "ingestion": patients / ingestion_rate,
        "processing": patients / processing_rate,
        "review": patients / review_rate,
    }
    bottleneck = max(values, key=values.get)
    return CapacityEstimate(patients, values["ingestion"], values["processing"],
                            values["review"], bottleneck)
