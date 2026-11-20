#!/usr/bin/env python3
"""Deterministic xApp executor for real ns3-ai LLM-rApp scenarios."""

from __future__ import annotations

import argparse
import csv
import copy
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ORAN_TOOLS = ROOT.parents[0] / "oran_xapp" / "tools"
for path in (SRC, ORAN_TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm_v2x_scheduler.llm_rapp_runner import generate_rapp_policy
from llm_v2x_scheduler.parser import dump_policy, load_policy_file, parse_policy_text
from llm_v2x_scheduler.rapp_prompt_builder import build_rapp_prompt
from llm_v2x_scheduler.services import SERVICE_AWARENESS, SERVICE_HDMAP, SERVICE_NAMES, SERVICE_SENSOR, SERVICE_TOD
from llm_v2x_scheduler.telemetry import ServiceTelemetry, TelemetrySnapshot
from llm_v2x_scheduler.utils import write_json
from llm_v2x_scheduler.validator import validate_policy
from llm_v2x_scheduler.xapp_controller import XAppController
from shm_interface import MAGIC_STATE_VALID, SharedMemoryInterface

TARGET_MBPS_PER_UE = {
    SERVICE_TOD: 16.048,
    SERVICE_AWARENESS: 0.064,
    SERVICE_HDMAP: 16.008,
    SERVICE_SENSOR: 0.096,
}

# UE class mix per density — must match MakeAssignByMix() in the .cc scenario files
UE_MIX = {
    20: {"hdm": 1, "awareness": 5, "tod": 1, "sensor": 13},
    25: {"hdm": 2, "awareness": 6, "tod": 2, "sensor": 15},
    30: {"hdm": 3, "awareness": 7, "tod": 3, "sensor": 17},
}

OBJECTIVE = (
    "Protect ToD (DL deadline 20 ms, UL deadline 100 ms) and Awareness (deadline 20 ms both directions) at all times. "
    "These are safety-of-life services. Deprioritise HDMap and Sensor under congestion. "
    "Priority order under congestion: ToD > Awareness > HDMap ≈ Sensor."
)


def main() -> int:
    args = parse_args()
    raw_policy = load_policy_file(args.policy)
    validation = validate_policy(raw_policy, attempt_repair=True)
    if not validation.policy:
        raise RuntimeError(f"no executable policy: {validation.errors}")

    policy_holder = {
        "policy": validation.policy,
        "version": 0,
        "last_update_step": None,  # None = trigger immediately at step 0
    }
    policy_lock = threading.Lock()
    rapp_worker: threading.Thread | None = None
    update_index = 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    density = args.density
    mix = UE_MIX.get(density, UE_MIX[30])
    service_by_ue = service_assignment(density, mix)

    scenario = {
        "scenario_id": raw_policy.get("scenario_id", f"real_{density}veh"),
        "density": density,
        "method": "llm_rapp_xapp",
        "ue_mix": {
            "ToD": mix["tod"],
            "Awareness": mix["awareness"],
            "HDMap": mix["hdm"],
            "Sensor": mix["sensor"],
        },
    }

    controller = XAppController(cooldown_cycles=args.cooldown_cycles)
    shm = SharedMemoryInterface(timeout_s=args.timeout_s)
    shm.connect(expected_mode=4)

    period_steps = max(1, int(round((args.rapp_period_s * 1000.0) / max(args.ai_cycle_ms, 1.0))))

    log_path = out_dir / "llm_v2x_xapp_log.csv"
    with log_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "sim_time_s",
                "tod_weight",
                "awareness_weight",
                "hdmap_weight",
                "sensor_weight",
                "policy_version",
                "policy_id",
                "adaptations",
                "safety_interventions",
                "safety_intervention",
                "fallback_used",
                "xapp_latency_ms",
                "congestion_ratio",
                "tod_p95_ms",
                "tod_p95_latency_ms",
                "awareness_p95_ms",
                "awareness_p95_latency_ms",
                "tod_dc_prr",
                "tod_ul_dc_prr",
                "tod_dl_dc_prr",
                "awareness_dc_prr",
                "awareness_ul_dc_prr",
                "awareness_dl_dc_prr",
                "hdmap_degradation",
                "hdmap_throughput_degradation",
                "sensor_degradation",
                "sensor_throughput_degradation",
                "service_counts",
                "ue_services",
                "ul_weights_per_ue",
                "dl_weights_per_ue",
                "reasons",
                "adaptation_reason",
            ],
        )
        writer.writeheader()
        last_step = -1
        cycles = 0
        try:
            while cycles < args.max_cycles:
                state = shm.read_state()
                if state.nodeId != MAGIC_STATE_VALID or state.stepCount == last_step:
                    time.sleep(0.0001)
                    continue
                last_step = state.stepCount
                cycles += 1

                with policy_lock:
                    policy = copy.deepcopy(policy_holder["policy"])
                    policy_version = int(policy_holder["version"])
                    last_update_step = policy_holder["last_update_step"]

                telemetry = telemetry_from_state(state, service_by_ue, policy["baseline_weights"])

                # Trigger LLM rApp worker: at step 0 and every rapp_period_s thereafter
                if args.llm_rapp and _should_update(state.stepCount, last_update_step, period_steps, rapp_worker):
                    policy_holder["last_update_step"] = state.stepCount
                    update_index += 1
                    rapp_worker = threading.Thread(
                        target=rapp_update_worker,
                        kwargs={
                            "update_index": update_index,
                            "step": state.stepCount,
                            "sim_time_s": state.stepCount * args.ai_cycle_ms / 1000.0,
                            "policy_snapshot": copy.deepcopy(policy),
                            "telemetry_snapshot": copy.deepcopy(telemetry),
                            "scenario": scenario,
                            "objective_id": args.objective_id,
                            "model": args.model,
                            "backend": args.backend,
                            "out_dir": out_dir,
                            "policy_holder": policy_holder,
                            "policy_lock": policy_lock,
                        },
                        daemon=True,
                    )
                    rapp_worker.start()

                decision = controller.decide(policy, telemetry)
                ul_weights, dl_weights = ue_weight_arrays(
                    service_by_ue,
                    decision.weights,
                    decision.directional_weights,
                )
                shm.write_action(
                    step=state.stepCount,
                    mode=4,
                    profile=0,
                    raw_profile=0,
                    shield_reason=0,
                    ul_weights=ul_weights,
                    dl_weights=dl_weights,
                    applied_critical_weight=max(decision.weights[SERVICE_TOD], decision.weights[SERVICE_AWARENESS]),
                    applied_background_weight=max(decision.weights[SERVICE_HDMAP], decision.weights[SERVICE_SENSOR]),
                    applied_background_cap=1.0,
                )
                writer.writerow({
                    "step": state.stepCount,
                    "sim_time_s": f"{state.stepCount * args.ai_cycle_ms / 1000.0:.1f}",
                    "tod_weight": decision.weights[SERVICE_TOD],
                    "awareness_weight": decision.weights[SERVICE_AWARENESS],
                    "hdmap_weight": decision.weights[SERVICE_HDMAP],
                    "sensor_weight": decision.weights[SERVICE_SENSOR],
                    "policy_version": policy_version,
                    "policy_id": decision.policy_id or policy.get("policy_id", ""),
                    "adaptations": decision.adaptations,
                    "safety_interventions": decision.safety_interventions,
                    "safety_intervention": int(decision.safety_interventions > 0),
                    "fallback_used": int(decision.fallback_used),
                    "xapp_latency_ms": f"{decision.execution_latency_ms:.6f}",
                    "congestion_ratio": f"{telemetry.congestion_ratio:.4f}",
                    "tod_p95_ms": f"{telemetry.services[SERVICE_TOD].p95_latency_ms:.2f}",
                    "tod_p95_latency_ms": f"{telemetry.services[SERVICE_TOD].p95_latency_ms:.2f}",
                    "awareness_p95_ms": f"{telemetry.services[SERVICE_AWARENESS].p95_latency_ms:.2f}",
                    "awareness_p95_latency_ms": f"{telemetry.services[SERVICE_AWARENESS].p95_latency_ms:.2f}",
                    "tod_dc_prr": f"{telemetry.services[SERVICE_TOD].dc_prr:.4f}",
                    "tod_ul_dc_prr": _format_optional(telemetry.services[SERVICE_TOD].ul_dc_prr),
                    "tod_dl_dc_prr": _format_optional(telemetry.services[SERVICE_TOD].dl_dc_prr),
                    "awareness_dc_prr": f"{telemetry.services[SERVICE_AWARENESS].dc_prr:.4f}",
                    "awareness_ul_dc_prr": _format_optional(telemetry.services[SERVICE_AWARENESS].ul_dc_prr),
                    "awareness_dl_dc_prr": _format_optional(telemetry.services[SERVICE_AWARENESS].dl_dc_prr),
                    "hdmap_degradation": f"{telemetry.services[SERVICE_HDMAP].throughput_degradation:.4f}",
                    "hdmap_throughput_degradation": f"{telemetry.services[SERVICE_HDMAP].throughput_degradation:.4f}",
                    "sensor_degradation": f"{telemetry.services[SERVICE_SENSOR].throughput_degradation:.4f}",
                    "sensor_throughput_degradation": f"{telemetry.services[SERVICE_SENSOR].throughput_degradation:.4f}",
                    "service_counts": service_counts_string(service_by_ue),
                    "ue_services": ";".join(service_by_ue),
                    "ul_weights_per_ue": ";".join(str(v) for v in ul_weights[:len(service_by_ue)]),
                    "dl_weights_per_ue": ";".join(str(v) for v in dl_weights[:len(service_by_ue)]),
                    "reasons": ";".join(decision.reasons),
                    "adaptation_reason": ";".join(decision.reasons),
                })
                f.flush()
        finally:
            if rapp_worker and rapp_worker.is_alive():
                rapp_worker.join(timeout=2.0)
            try:
                shm.write_action(last_step + 1, 4, 0, 0, 0, stop=True)
            except Exception:
                pass
            shm.close()
    print(f"[llm-v2x-xapp] wrote {log_path}")
    return 0


def _should_update(step: int, last_update_step: int | None, period_steps: int, worker: threading.Thread | None) -> bool:
    if worker and worker.is_alive():
        return False
    if last_update_step is None:
        return True
    return (step - last_update_step) >= period_steps


def rapp_update_worker(
    update_index: int,
    step: int,
    sim_time_s: float,
    policy_snapshot: dict,
    telemetry_snapshot: TelemetrySnapshot,
    scenario: dict,
    objective_id: str,
    model: str,
    backend: str,
    out_dir: Path,
    policy_holder: dict,
    policy_lock: threading.Lock,
) -> None:
    started = time.perf_counter()
    rapp_dir = out_dir / "rapp_updates"
    raw_dir = rapp_dir / "raw"
    validation_dir = rapp_dir / "validation"
    policy_dir = rapp_dir / "executed"
    for path in (raw_dir, validation_dir, policy_dir):
        path.mkdir(parents=True, exist_ok=True)

    label = f"update_{update_index:03d}_step_{step}_t{sim_time_s:.0f}s"

    # Build context-aware objective: show the LLM current weights and live telemetry
    tel = telemetry_snapshot.services
    current_weights = policy_snapshot.get("baseline_weights", {})
    objective = (
        f"{OBJECTIVE}\n\n"
        f"Current policy: {policy_snapshot.get('policy_id', 'unknown')} "
        f"(version {update_index - 1}, t={sim_time_s:.0f}s).\n"
        f"Current baseline weights — "
        f"ToD: {current_weights.get('ToD', '?')}, "
        f"Awareness: {current_weights.get('Awareness', '?')}, "
        f"HDMap: {current_weights.get('HDMap', '?')}, "
        f"Sensor: {current_weights.get('Sensor', '?')}.\n"
        f"Live performance — "
        f"ToD p95: {tel[SERVICE_TOD].p95_latency_ms:.1f} ms (deadline 20 ms), "
        f"ToD DC-PRR: {tel[SERVICE_TOD].dc_prr:.3f}; "
        f"Awareness p95: {tel[SERVICE_AWARENESS].p95_latency_ms:.1f} ms (deadline 20 ms), "
        f"Awareness DC-PRR: {tel[SERVICE_AWARENESS].dc_prr:.3f}; "
        f"HDMap degradation: {tel[SERVICE_HDMAP].throughput_degradation:.2%}; "
        f"Sensor degradation: {tel[SERVICE_SENSOR].throughput_degradation:.2%}; "
        f"Congestion: {telemetry_snapshot.congestion_ratio:.2f}.\n"
        f"Generate an improved policy if the live telemetry shows deadline violations or high congestion. "
        f"Otherwise generate a conservative equivalent policy."
    )

    prompt = build_rapp_prompt(
        scenario,
        objective,
        telemetry_snapshot,
        model_name=model,
        objective_id=objective_id,
    )

    report = {
        "update_index": update_index,
        "step": step,
        "sim_time_s": sim_time_s,
        "model": model,
        "accepted": False,
        "rejected": True,
        "repaired": False,
        "old_policy_id": policy_snapshot.get("policy_id"),
        "new_policy_id": None,
        "latency_s": None,
        "errors": [],
        "warnings": [],
    }
    raw = ""
    try:
        raw = generate_rapp_policy(model, prompt, backend=backend, temperature=0.1, max_tokens=900)
        (raw_dir / f"{label}.txt").write_text(raw)
        candidate = parse_policy_text(raw)
        candidate["scenario_id"] = scenario.get("scenario_id", candidate.get("scenario_id", "unknown"))
        candidate["objective_id"] = objective_id
        validation = validate_policy(candidate, attempt_repair=True)
        dump_policy(validation_dir / f"{label}.yaml", validation.as_dict())
        report["rejected"] = bool(validation.rejected)
        report["repaired"] = bool(validation.repaired)
        report["errors"] = list(validation.errors)
        report["warnings"] = list(validation.warnings)
        if validation.policy and validation.valid and not validation.rejected:
            new_policy = validation.policy
            new_policy["metadata"]["model_name"] = model
            dump_policy(policy_dir / f"{label}.yaml", new_policy)
            with policy_lock:
                policy_holder["policy"] = new_policy
                policy_holder["version"] = int(policy_holder["version"]) + 1
            report["accepted"] = True
            report["new_policy_id"] = new_policy.get("policy_id")
    except Exception as exc:
        report["errors"].append(str(exc))
        if raw:
            (raw_dir / f"{label}_failed.txt").write_text(raw)
    finally:
        report["latency_s"] = time.perf_counter() - started
        write_json(rapp_dir / f"{label}_report.json", report)
    print(
        f"[rapp] t={sim_time_s:.0f}s update={update_index} "
        f"accepted={report['accepted']} repaired={report['repaired']} "
        f"latency={report['latency_s']:.1f}s",
        flush=True,
    )


def service_assignment(density: int, mix: dict) -> list[str]:
    services = []
    for _ in range(mix["hdm"]):
        services.append(SERVICE_HDMAP)
    for _ in range(mix["awareness"]):
        services.append(SERVICE_AWARENESS)
    for _ in range(mix["tod"]):
        services.append(SERVICE_TOD)
    for _ in range(mix["sensor"]):
        services.append(SERVICE_SENSOR)
    return services


def telemetry_from_state(state, service_by_ue: list[str], current_weights: dict[str, int]) -> TelemetrySnapshot:
    services = {}
    congestion = float(max(0.0, min(1.0, state.bufferLoad)))
    for service in SERVICE_NAMES:
        indices = [i for i, svc in enumerate(service_by_ue) if svc == service and i < state.numUes]
        lats = []
        ul_lats = []
        dl_lats = []
        violations = 0
        ul_violations = 0
        dl_violations = 0
        throughput = 0.0
        for i in indices:
            ul_lat = float(state.ulAverageLatency[i])
            dl_lat = float(state.dlAverageLatency[i])
            valid_ul = _valid_latency(ul_lat)
            valid_dl = _valid_latency(dl_lat)
            lat = max(valid_ul, valid_dl)
            lats.append(lat)
            ul_lats.append(valid_ul)
            dl_lats.append(valid_dl)
            ul_bad = float(state.ulMinSlack[i]) < 0.0
            dl_bad = float(state.dlMinSlack[i]) < 0.0
            if ul_bad or dl_bad:
                violations += 1
            if ul_bad:
                ul_violations += 1
            if dl_bad:
                dl_violations += 1
            throughput += max(0.0, float(state.ulServiceRate[i]) + float(state.dlServiceRate[i])) / 1e6
        mean = statistics.fmean(lats) if lats else 0.0
        p95 = percentile(lats, 0.95)
        violation_rate = violations / max(1, len(indices))
        dc_prr = max(0.0, 1.0 - violation_rate)
        ul_violation_rate = ul_violations / max(1, len(indices))
        dl_violation_rate = dl_violations / max(1, len(indices))
        target = TARGET_MBPS_PER_UE[service] * max(1, len(indices))
        degradation = max(0.0, 1.0 - throughput / target)
        services[service] = ServiceTelemetry(
            packet_count=len(indices),
            mean_latency_ms=mean,
            p95_latency_ms=p95,
            violation_rate=violation_rate,
            throughput_mbps=throughput,
            dc_prr=dc_prr,
            throughput_degradation=degradation,
            ul_p95_latency_ms=percentile(ul_lats, 0.95),
            dl_p95_latency_ms=percentile(dl_lats, 0.95),
            ul_violation_rate=ul_violation_rate,
            dl_violation_rate=dl_violation_rate,
            ul_dc_prr=max(0.0, 1.0 - ul_violation_rate),
            dl_dc_prr=max(0.0, 1.0 - dl_violation_rate),
        )
    return TelemetrySnapshot(services=services, current_weights=dict(current_weights), congestion_ratio=congestion)


def ue_weight_arrays(
    service_by_ue: list[str],
    weights: dict[str, int],
    directional_weights: dict[str, dict[str, int]] | None = None,
) -> tuple[list[int], list[int]]:
    ul_by_service = (directional_weights or {}).get("ul", weights)
    dl_by_service = (directional_weights or {}).get("dl", weights)
    ul_vals = [int(ul_by_service.get(svc, weights[svc])) for svc in service_by_ue]
    dl_vals = [int(dl_by_service.get(svc, weights[svc])) for svc in service_by_ue]
    ul_vals.extend([0] * (60 - len(ul_vals)))
    dl_vals.extend([0] * (60 - len(dl_vals)))
    return ul_vals[:60], dl_vals[:60]


def service_counts_string(service_by_ue: list[str]) -> str:
    return ";".join(f"{service}={service_by_ue.count(service)}" for service in SERVICE_NAMES)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * pct))))
    return vals[idx]


def _valid_latency(value: float) -> float:
    if value < 0 or value > 99999:
        return 0.0
    return value


def _format_optional(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--density", type=int, default=30)
    p.add_argument("--max-cycles", type=int, default=100000)
    p.add_argument("--cooldown-cycles", type=int, default=2)
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument("--ai-cycle-ms", type=float, default=100.0)
    p.add_argument("--objective-id", default="safety_background_balance")
    p.add_argument("--llm-rapp", action="store_true",
                   help="enable periodic LLM rApp policy refresh (llm_rapp_xapp method)")
    p.add_argument("--rapp-period-s", type=float, default=10.0,
                   help="LLM rApp refresh period in seconds (default 10 s)")
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--backend", default="ollama", choices=["ollama", "auto", "mock"])
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
