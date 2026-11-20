"""Metric computation for mock and ns-3 parsed outputs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from .services import CRITICAL_SERVICES, SERVICE_HDMAP, SERVICE_NAMES, SERVICE_SENSOR


def aggregate_mock_cycle_csv(path: str | Path, method: str, policy_id: str, scenario_id: str) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    by_service: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_service[row["service"]].append(row)
    out: list[dict[str, Any]] = []
    throughputs = []
    for service in SERVICE_NAMES:
        vals = by_service.get(service, [])
        if not vals:
            continue
        throughput = _avg(vals, "throughput_mbps")
        throughputs.append(throughput)
        out.append({
            "policy_id": policy_id,
            "method": method,
            "scenario_id": scenario_id,
            "service": service,
            "dc_prr": _avg(vals, "dc_prr"),
            "violation_rate": _avg(vals, "violation_rate"),
            "mean_latency_ms": _avg(vals, "mean_latency_ms"),
            "p95_latency_ms": _avg(vals, "p95_latency_ms"),
            "throughput_mbps": throughput,
            "throughput_degradation": _avg(vals, "throughput_degradation"),
            "fairness": 0.0,
            "xapp_adaptations": _sum_last_cycle_events(vals, "xapp_adaptations"),
            "safety_interventions": _sum_last_cycle_events(vals, "safety_interventions"),
            "fallback_rate": _avg(vals, "fallback_used"),
            "xapp_execution_latency_ms": _avg(vals, "xapp_latency_ms"),
        })
    fairness = jain_fairness(throughputs)
    for row in out:
        row["fairness"] = fairness
    out.append(_aggregate_row(out, policy_id, method, scenario_id))
    return out


def write_metrics_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def jain_fairness(values: list[float]) -> float:
    if not values:
        return 0.0
    num = sum(values) ** 2
    den = len(values) * sum(v * v for v in values)
    return num / den if den else 0.0


def _aggregate_row(rows: list[dict[str, Any]], policy_id: str, method: str, scenario_id: str) -> dict[str, Any]:
    service_rows = [r for r in rows if r["service"] in SERVICE_NAMES]
    crit = [r for r in service_rows if r["service"] in CRITICAL_SERVICES]
    background = [r for r in service_rows if r["service"] in (SERVICE_HDMAP, SERVICE_SENSOR)]
    return {
        "policy_id": policy_id,
        "method": method,
        "scenario_id": scenario_id,
        "service": "aggregate",
        "dc_prr": sum(r["dc_prr"] for r in crit) / max(1, len(crit)),
        "violation_rate": sum(r["violation_rate"] for r in crit) / max(1, len(crit)),
        "mean_latency_ms": sum(r["mean_latency_ms"] for r in crit) / max(1, len(crit)),
        "p95_latency_ms": sum(r["p95_latency_ms"] for r in crit) / max(1, len(crit)),
        "throughput_mbps": sum(r["throughput_mbps"] for r in service_rows),
        "throughput_degradation": sum(r["throughput_degradation"] for r in background) / max(1, len(background)),
        "fairness": service_rows[0]["fairness"] if service_rows else 0.0,
        "xapp_adaptations": sum(r["xapp_adaptations"] for r in service_rows),
        "safety_interventions": sum(r["safety_interventions"] for r in service_rows),
        "fallback_rate": sum(r["fallback_rate"] for r in service_rows) / max(1, len(service_rows)),
        "xapp_execution_latency_ms": sum(r["xapp_execution_latency_ms"] for r in service_rows) / max(1, len(service_rows)),
    }


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def _avg(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(r[key]) for r in rows) / max(1, len(rows))


def _sum_last_cycle_events(rows: list[dict[str, str]], key: str) -> int:
    seen = set()
    total = 0
    for row in rows:
        cycle = row["cycle"]
        if cycle in seen:
            continue
        seen.add(cycle)
        total += int(float(row[key]))
    return total

