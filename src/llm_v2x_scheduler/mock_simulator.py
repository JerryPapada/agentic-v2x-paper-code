"""Deterministic mock simulator for the smoke-test-first milestone."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from .services import SERVICE_AWARENESS, SERVICE_HDMAP, SERVICE_NAMES, SERVICE_SENSOR, SERVICE_TOD
from .telemetry import ServiceTelemetry, TelemetrySnapshot
from .xapp_controller import XAppController


TARGET_THROUGHPUT = {
    SERVICE_TOD: 16.0,
    SERVICE_AWARENESS: 1.5,
    SERVICE_HDMAP: 64.0,
    SERVICE_SENSOR: 8.0,
}


def run_mock_simulation(
    policy: dict[str, Any],
    out_dir: str | Path,
    scenario: dict[str, Any] | None = None,
    cycles: int = 20,
    seed: int = 1,
    adaptive: bool = True,
) -> list[dict[str, Any]]:
    """Run a simple deterministic closed-loop mock simulation.

    This is not a network model. It provides stable, reproducible telemetry so
    schema/controller/metrics/plotting can be tested before ns-3 integration.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scenario = scenario or {"density": 20, "seed": seed}
    rng = random.Random(seed)
    density = int(scenario.get("density", 20))
    # load_factor is driven solely by density; V2X application traffic is fixed by service definitions
    load_factor = 1.0 + max(0, density - 20) / 30.0
    congestion = min(0.95, 0.45 + 0.18 * load_factor)

    controller = XAppController(cooldown_cycles=2)
    weights = dict(policy["baseline_weights"])
    rows: list[dict[str, Any]] = []

    for cycle in range(cycles):
        telemetry = _telemetry_from_weights(weights, load_factor, congestion, rng)
        if adaptive:
            decision = controller.decide(policy, telemetry)
            weights = decision.weights
            adaptations = decision.adaptations
            safety = decision.safety_interventions
            fallback = int(decision.fallback_used)
            xapp_latency_ms = decision.execution_latency_ms
        else:
            adaptations = 0
            safety = 0
            fallback = 0
            xapp_latency_ms = 0.0
        for service in SERVICE_NAMES:
            tel = telemetry.services[service]
            rows.append({
                "cycle": cycle,
                "service": service,
                "weight": weights[service],
                "packet_count": tel.packet_count,
                "mean_latency_ms": round(tel.mean_latency_ms, 4),
                "p95_latency_ms": round(tel.p95_latency_ms, 4),
                "violation_rate": round(tel.violation_rate, 6),
                "dc_prr": round(tel.dc_prr, 6),
                "throughput_mbps": round(tel.throughput_mbps, 4),
                "throughput_degradation": round(tel.throughput_degradation, 6),
                "congestion_ratio": round(congestion, 6),
                "xapp_adaptations": adaptations,
                "safety_interventions": safety,
                "fallback_used": fallback,
                "xapp_latency_ms": round(xapp_latency_ms, 6),
            })

    _write_csv(out / "mock_cycle_metrics.csv", rows)
    return rows


def _telemetry_from_weights(weights: dict[str, int], load_factor: float, congestion: float, rng: random.Random) -> TelemetrySnapshot:
    total_weight = max(1, sum(weights.values()))
    services = {}
    for service in SERVICE_NAMES:
        share = weights[service] / total_weight
        demand = _demand(service) * load_factor
        capacity_share = share * 110.0
        stress = max(0.15, demand / max(capacity_share, 1.0))
        base_latency = _base_latency(service)
        latency = base_latency * (1.0 + 0.55 * stress + 0.65 * congestion)
        p95 = latency * (1.35 + 0.25 * stress)
        deadline = _deadline(service)
        violation = min(0.95, max(0.0, (p95 - deadline) / max(deadline, 1.0)) * 0.45)
        dc_prr = max(0.0, 1.0 - violation - 0.02 * congestion)
        delivered = min(demand, capacity_share * (1.0 - 0.1 * congestion))
        target = TARGET_THROUGHPUT[service]
        degradation = max(0.0, 1.0 - delivered / max(target, 1.0))
        jitter = 1.0 + rng.random() * 0.015
        services[service] = ServiceTelemetry(
            packet_count=int(900 * load_factor * (0.6 + share)),
            mean_latency_ms=latency * jitter,
            p95_latency_ms=p95 * jitter,
            violation_rate=violation,
            throughput_mbps=delivered,
            dc_prr=dc_prr,
            throughput_degradation=degradation,
        )
    return TelemetrySnapshot(services=services, current_weights=dict(weights), congestion_ratio=congestion)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _demand(service: str) -> float:
    return {SERVICE_TOD: 18.0, SERVICE_AWARENESS: 2.0, SERVICE_HDMAP: 70.0, SERVICE_SENSOR: 10.0}[service]


def _base_latency(service: str) -> float:
    return {SERVICE_TOD: 8.0, SERVICE_AWARENESS: 35.0, SERVICE_HDMAP: 180.0, SERVICE_SENSOR: 150.0}[service]


def _deadline(service: str) -> float:
    return {SERVICE_TOD: 20.0, SERVICE_AWARENESS: 100.0, SERVICE_HDMAP: 1000.0, SERVICE_SENSOR: 500.0}[service]

