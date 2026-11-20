"""Dataclasses and constants for v1 rApp scheduler policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .services import CRITICAL_SERVICES, SERVICE_NAMES


MIN_WEIGHT = 1
MAX_WEIGHT = 10

SUPPORTED_METRICS = (
    "tod_p95_latency_ms",
    "tod_dc_prr",
    "tod_violation_rate",
    "awareness_p95_latency_ms",
    "awareness_dc_prr",
    "awareness_violation_rate",
    "hdmap_throughput_degradation",
    "sensor_throughput_degradation",
    "congestion_ratio",
)

ADAPTIVE_DIRECTIONAL_METRICS = tuple(
    f"{service}_{direction}_{metric}"
    for service in ("tod", "awareness")
    for direction in ("ul", "dl")
    for metric in ("delivery_ratio", "late_ue_fraction", "p95_ue_mean_latency_ms")
)

SUPPORTED_OPERATORS = (">", ">=", "<", "<=")
SUPPORTED_ACTIONS = ("set_weight", "increase_weight", "decrease_weight")
SUPPORTED_SOURCES = ("llm_rapp", "manual", "baseline", "repaired")


@dataclass
class ValidationResult:
    valid: bool
    repaired: bool = False
    rejected: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    policy: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "repaired": self.repaired,
            "rejected": self.rejected,
            "errors": self.errors,
            "warnings": self.warnings,
            "policy": self.policy or {},
        }


def default_bounds() -> dict[str, list[int]]:
    return {
        service: [5 if service in CRITICAL_SERVICES else MIN_WEIGHT, MAX_WEIGHT]
        for service in SERVICE_NAMES
    }


def default_safety_constraints() -> dict[str, Any]:
    return {
        "min_critical_weight": 5,
        "max_background_degradation": 0.35,
        "forbid_zero_weight": True,
        "critical_services": list(CRITICAL_SERVICES),
    }


def fallback_policy_name() -> str:
    return "static_expert"
