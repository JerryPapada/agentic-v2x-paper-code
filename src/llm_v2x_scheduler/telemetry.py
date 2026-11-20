"""Telemetry model for deterministic xApp control and mock simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .services import SERVICE_AWARENESS, SERVICE_HDMAP, SERVICE_NAMES, SERVICE_SENSOR, SERVICE_TOD


@dataclass
class ServiceTelemetry:
    packet_count: int = 0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    violation_rate: float = 0.0
    throughput_mbps: float = 0.0
    dc_prr: float = 1.0
    throughput_degradation: float = 0.0
    ul_p95_latency_ms: float | None = None
    dl_p95_latency_ms: float | None = None
    ul_violation_rate: float | None = None
    dl_violation_rate: float | None = None
    ul_dc_prr: float | None = None
    dl_dc_prr: float | None = None


@dataclass
class TelemetrySnapshot:
    services: dict[str, ServiceTelemetry]
    current_weights: dict[str, int]
    congestion_ratio: float
    observed_metrics: dict[str, float | None] = field(default_factory=dict)
    critical_observed: bool = True

    def metric(self, name: str) -> float | None:
        if name in self.observed_metrics:
            return self.observed_metrics[name]
        if name == "congestion_ratio":
            return float(self.congestion_ratio)
        mapping = {
            "tod_p95_latency_ms": (SERVICE_TOD, "p95_latency_ms"),
            "tod_dc_prr": (SERVICE_TOD, "dc_prr"),
            "tod_violation_rate": (SERVICE_TOD, "violation_rate"),
            "awareness_p95_latency_ms": (SERVICE_AWARENESS, "p95_latency_ms"),
            "awareness_dc_prr": (SERVICE_AWARENESS, "dc_prr"),
            "awareness_violation_rate": (SERVICE_AWARENESS, "violation_rate"),
            "hdmap_throughput_degradation": (SERVICE_HDMAP, "throughput_degradation"),
            "sensor_throughput_degradation": (SERVICE_SENSOR, "throughput_degradation"),
        }
        if name not in mapping:
            raise KeyError(f"unsupported telemetry metric {name}")
        service, attr = mapping[name]
        return float(getattr(self.services[service], attr))

    def as_flat_dict(self) -> dict[str, float]:
        out = {"congestion_ratio": float(self.congestion_ratio)}
        for service, prefix in (
            (SERVICE_TOD, "tod"),
            (SERVICE_AWARENESS, "awareness"),
            (SERVICE_HDMAP, "hdmap"),
            (SERVICE_SENSOR, "sensor"),
        ):
            tel = self.services[service]
            out[f"{prefix}_packet_count"] = float(tel.packet_count)
            out[f"{prefix}_mean_latency_ms"] = tel.mean_latency_ms
            out[f"{prefix}_p95_latency_ms"] = tel.p95_latency_ms
            out[f"{prefix}_violation_rate"] = tel.violation_rate
            out[f"{prefix}_throughput_mbps"] = tel.throughput_mbps
            out[f"{prefix}_dc_prr"] = tel.dc_prr
            out[f"{prefix}_throughput_degradation"] = tel.throughput_degradation
        return out


def default_telemetry(weights: dict[str, int] | None = None) -> TelemetrySnapshot:
    weights = weights or {service: 1 for service in SERVICE_NAMES}
    return TelemetrySnapshot(
        services={
            SERVICE_TOD: ServiceTelemetry(1200, 12.0, 18.0, 0.01, 12.0, 0.99, 0.0),
            SERVICE_AWARENESS: ServiceTelemetry(1600, 35.0, 85.0, 0.015, 1.2, 0.985, 0.0),
            SERVICE_HDMAP: ServiceTelemetry(700, 220.0, 480.0, 0.0, 48.0, 0.97, 0.20),
            SERVICE_SENSOR: ServiceTelemetry(900, 180.0, 380.0, 0.0, 4.0, 0.98, 0.15),
        },
        current_weights=dict(weights),
        congestion_ratio=0.72,
    )


_UE_MIX: dict[int, dict[str, int]] = {
    20: {"ToD": 1, "Awareness": 5, "HDMap": 1, "Sensor": 13},
    25: {"ToD": 2, "Awareness": 6, "HDMap": 2, "Sensor": 15},
    30: {"ToD": 3, "Awareness": 7, "HDMap": 3, "Sensor": 17},
}

# Tightest deadline per service (minimum across UL/DL directions)
# AIC: UL=20ms DL=20ms → 20ms
# ToD: UL=100ms DL=20ms → 20ms (DL operator commands are the tight path)
# HDMap: UL=1000ms DL=500ms → 500ms
# RTSA: UL=1000ms DL=1000ms → 1000ms
SERVICE_DEADLINE_MS = {
    SERVICE_TOD: 20.0,
    SERVICE_AWARENESS: 20.0,
    SERVICE_HDMAP: 500.0,
    SERVICE_SENSOR: 1000.0,
}


def telemetry_summary(snapshot: TelemetrySnapshot, scenario: dict[str, Any], objective: str) -> str:
    s = snapshot.services
    density = int(scenario.get("density", 0))
    mix = _UE_MIX.get(density)
    if mix:
        mix_line = (
            f"- UE class mix: ToD={mix['ToD']}, Awareness={mix['Awareness']}, "
            f"HDMap={mix['HDMap']}, Sensor={mix['Sensor']} "
            f"(ToD DL deadline 20 ms; Awareness deadline 20 ms; ToD UL deadline 100 ms)\n"
        )
    else:
        mix_line = "- UE class mix: unknown density\n"
    return (
        "Scenario:\n"
        f"- {density} vehicles (fixed V2X application traffic models)\n"
        f"{mix_line}"
        f"- current method: {scenario.get('method', 'unknown')}\n\n"
        "Recent performance:\n"
        f"- ToD DC-PRR: {100.0 * s[SERVICE_TOD].dc_prr:.1f}%  (DL deadline 20 ms, UL deadline 100 ms)\n"
        f"- ToD p95 latency: {s[SERVICE_TOD].p95_latency_ms:.1f} ms\n"
        f"- Awareness DC-PRR: {100.0 * s[SERVICE_AWARENESS].dc_prr:.1f}%  (deadline 20 ms both directions)\n"
        f"- Awareness p95 latency: {s[SERVICE_AWARENESS].p95_latency_ms:.1f} ms\n"
        f"- HDMap throughput degradation: {100.0 * s[SERVICE_HDMAP].throughput_degradation:.1f}%\n"
        f"- Sensor throughput degradation: {100.0 * s[SERVICE_SENSOR].throughput_degradation:.1f}%\n"
        f"- congestion ratio: {snapshot.congestion_ratio:.2f}\n\n"
        f"Objective:\n{objective}\n"
    )
