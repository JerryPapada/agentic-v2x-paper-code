#!/usr/bin/env python3
"""Run the versioned Adaptive LLM revision study in resumable phases."""

from __future__ import annotations

import argparse
import importlib.util
import math
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from xml.etree.ElementTree import iterparse

import _bootstrap  # noqa: F401

from adaptive_study_report import archive_audit, diagnose_legacy, packets, signature, summarize
from adaptive_study_runner import NS3_ROOT, Trial, archived, jsonl, offline_archived_actions, simulate
from llm_v2x_scheduler.adaptive_study import (
    ADAPTIVE_DIRECTIONAL_METRICS, MODEL, MODEL_OPTIONS, OBJECTIVES,
    build_prompt, generate_proposal, identifiers, model_digest, policy_content, probe_snapshot,
)
from llm_v2x_scheduler.baselines import heuristic_xapp_policy
from llm_v2x_scheduler.utils import PROJECT_ROOT, read_json, write_json
from llm_v2x_scheduler.xapp_controller import XAppController

PILOT_SEEDS = (101, 102)
MAIN_SEEDS = range(11, 31)
SHIELD_SEEDS = range(11, 19)
LOAD_GRID = (0.624, 2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0)
PROBES = ("healthy", "tod_ul_stress", "awareness_stress", "hdmap_degradation")


def pinned_model(root: Path) -> str:
    path = root / "ollama_model.json"
    current = {"model": MODEL, "digest": model_digest(), "options": MODEL_OPTIONS}
    if path.exists() and read_json(path) != current:
        raise RuntimeError(f"Local model changed since {path} was recorded")
    write_json(path, current)
    return current["digest"]


def mobility(root: Path) -> Path:
    out = root / "mobility"
    trace = out / "manhattan20_300s.tcl"
    if (out / "verification.json").exists():
        if not trace.is_file():
            raise RuntimeError(f"Verified mobility file is missing: {trace}")
        return trace
    out.mkdir(parents=True, exist_ok=True)
    generator = NS3_ROOT / "scratch/mobility_1/sumo/generate_nested_v2.py"
    spec = importlib.util.spec_from_file_location("manhattan_mobility", generator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SCRIPT_DIR = str(out)
    network = module.generate_network()
    for intermediate in (18, 27, 36):
        route = module.generate_routes_all_at_once(network, 20, 42, 301, intermediate=intermediate)
        fcd = module.run_sumo(network, route, 20, 301)
        raw = module.fcd_to_ns2(fcd, 20)
        if module.postprocess_ns2_trace(raw, trace, 20) != 20:
            continue
        selected = set()
        with Path(raw).open() as stream:
            for line in stream:
                if line.startswith("$node_(") and " set X_ " in line:
                    selected.add(int(line.split(")", 1)[0].split("(")[1]))
        selected = set(sorted(selected)[:20])
        # The exporter assigns sequential IDs to the ordered FCD appearance set.
        seen = {}
        missing = 0
        min_concurrent = math.inf
        max_dist = 0.0
        covered_steps = 0
        for _, node in iterparse(fcd, events=("end",)):
            if node.tag != "timestep":
                continue
            sim_time = float(node.attrib["time"])
            for vehicle in node:
                if vehicle.attrib["id"] not in seen:
                    seen[vehicle.attrib["id"]] = len(seen)
            if 2 <= sim_time <= 300:
                active = [vehicle for vehicle in node if seen[vehicle.attrib["id"]] in selected]
                covered_steps += 1
                min_concurrent = min(min_concurrent, len(active))
                missing += len(active) != 20
                max_dist = max(max_dist, max((math.hypot(float(v.attrib["x"]) - 300, float(v.attrib["y"]) - 300)
                                              for v in active), default=0.0))
            node.clear()
        if not missing and covered_steps == 2981 and module.validate_trace(trace, 20, sim_duration=300):
            write_json(out / "verification.json", {
                "sumo_seed": 42, "intermediate_waypoints": intermediate, "analysis_duration_s": 300,
                "ue_count": 20, "minimum_concurrent_ues": min_concurrent, "checked_timesteps": covered_steps,
                "max_horizontal_distance_m": max_dist,
                "sumo_version": subprocess.check_output(["sumo", "--version"], text=True).splitlines()[0],
            })
            return trace
    raise RuntimeError("Unable to verify all 20 mobility nodes through 300 s")


def probes() -> dict:
    healthy = {
        metric: (0.99 if metric.endswith("delivery_ratio") else 0.01 if metric.endswith("late_ue_fraction") else 12.0)
        for metric in ADAPTIVE_DIRECTIONAL_METRICS
    }
    healthy |= {"hdmap_throughput_degradation": 0.05, "sensor_throughput_degradation": 0.05, "congestion_ratio": 0.2}
    return {
        "healthy": healthy,
        "tod_ul_stress": healthy | {"tod_ul_delivery_ratio": 0.70, "tod_ul_late_ue_fraction": 0.25,
                                    "tod_ul_p95_ue_mean_latency_ms": 130.0},
        "awareness_stress": healthy | {"awareness_ul_delivery_ratio": 0.75, "awareness_ul_late_ue_fraction": 0.25,
                                       "awareness_ul_p95_ue_mean_latency_ms": 35.0},
        "hdmap_degradation": healthy | {"hdmap_throughput_degradation": 0.55},
    }


def prompt_audit(root: Path, digest: str) -> None:
    scenarios = probes()
    out = root / "prompt_audit"
    write_json(out / "telemetry_probes.json", scenarios)
    startup = heuristic_xapp_policy()
    rows = []
    for snapshot, objective, seed, example in product(PROBES, OBJECTIVES, range(1, 11), (False, True)):
        name = "worked_example" if example else "schema_only"
        path = out / snapshot / objective / f"seed_{seed:02d}" / name
        prompt = build_prompt("adaptive_v2_prompt_audit", objective, scenarios[snapshot], {
            "baseline_weights": startup["baseline_weights"], "source": startup["policy_id"],
        }, example=example)
        if (path / "ollama_response.json").is_file() and (path / "validation.json").is_file():
            if (path / "prompt.txt").read_text() != prompt:
                raise RuntimeError(f"Prompt changed for {path}. Use a new study directory.")
            report = read_json(path / "validation.json")
            policy = read_json(path / "policy.json") if report["accepted"] else None
        else:
            print(f"PROMPT {snapshot}/{objective}/seed{seed:02d}/{name}", flush=True)
            report, policy = generate_proposal(prompt, seed, path, identifiers("adaptive_v2_prompt_audit", objective))
        if "done_reason" not in report:
            raise RuntimeError(f"Model transport failed at {path}: {report['errors']}")
        actions = {}
        content = None
        if policy is not None:
            content = policy_content(policy)
            for key, metrics in scenarios.items():
                decision = XAppController(performance_shield=False).decide(policy, probe_snapshot(metrics))
                actions[key] = decision.weights
        rows.append({"snapshot": snapshot, "objective": objective, "seed": seed, "prompt": name,
                     "executable_content": content, "probe_weights": actions, **report})
    if model_digest() != digest:
        raise RuntimeError("Local model digest changed during prompt evaluation")
    cells = {}
    for snapshot, objective, name in product(PROBES, OBJECTIVES, ("schema_only", "worked_example")):
        matching = [row for row in rows if (row["snapshot"], row["objective"], row["prompt"]) == (snapshot, objective, name)]
        cells[f"{snapshot}/{objective}/{name}"] = {
            "attempts": len(matching), "strict_pass": sum(row["strict_pass"] for row in matching),
            "repaired": sum(row["repaired"] for row in matching), "rejected": sum(not row["accepted"] for row in matching),
            "distinct_executable": len({row["executable_content"] for row in matching if row["accepted"]}),
        }
    keyed = {(row["snapshot"], row["objective"], row["seed"], row["prompt"]): row for row in rows}
    sensitivity = {}
    for name in ("schema_only", "worked_example"):
        for stressed, objective, service in (
            ("tod_ul_stress", "protect_tod_ul", "ToD"), ("awareness_stress", "protect_awareness", "Awareness"),
            ("hdmap_degradation", "recover_hdmap", "HDMap"),
        ):
            comparisons = {"intent": [], "telemetry": []}
            for seed in range(1, 11):
                candidate = keyed[stressed, objective, seed, name]["probe_weights"]
                peers = {
                    "intent": keyed[stressed, "safety_background_balance", seed, name]["probe_weights"],
                    "telemetry": keyed["healthy", objective, seed, name]["probe_weights"],
                }
                for axis, peer in peers.items():
                    if candidate and peer:
                        comparisons[axis].append(candidate[stressed][service] - peer[stressed][service])
            sensitivity[f"{name}/{stressed}"] = comparisons
    write_json(out / "results.json", {"model_digest": digest, "cells": cells, "sensitivity_weight_deltas": sensitivity, "runs": rows})


def pilot_metrics(path: Path) -> dict:
    rows = jsonl(path / "actions.jsonl")
    end = rows[-1]["sim_time_s"]
    final = [row["telemetry"] for row in rows if row["sim_time_s"] > end - 40]
    if len(final) != 400:
        raise ValueError(f"Need 400 control windows for calibration: {path}")
    offered = [sum(values[f"{s}_ul_offered_mbps"] for s in ("tod", "sensor", "awareness", "hdmap")) for values in final]
    served = [sum(values[f"{s}_ul_delivered_mbps"] for s in ("tod", "sensor", "awareness", "hdmap")) for values in final]
    queues = [sum(values[f"{s}_ul_queue_bytes"] for s in ("tod", "sensor", "awareness", "hdmap")) for values in final]
    return {
        "ul_delivery_ratio": sum(served) / sum(offered),
        "ul_queue_growth_bytes": statistics.median(queues[200:]) - statistics.median(queues[:200]),
        "ul_offered_mbps": statistics.mean(offered),
        "elapsed_s": read_json(path / "run_summary.json")["elapsed_s"],
    }


def classify_pilots(runs: list[dict]) -> str:
    delivery = statistics.mean(row["ul_delivery_ratio"] for row in runs)
    growth = statistics.mean(row["ul_queue_growth_bytes"] for row in runs)
    # The growth threshold is 100 ms worth of offered UL payload.
    threshold = statistics.mean(row["ul_offered_mbps"] for row in runs) * 1e6 / 80
    if delivery >= 0.97 and growth <= threshold:
        return "stable"
    if delivery < 0.95 and growth > threshold:
        return "overloaded"
    return "transition"


def calibrate(root: Path, trace: Path, jobs: int, timeout_s: float) -> dict:
    outcomes = {}

    def measure(rate: float) -> str:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            paths = list(executor.map(lambda seed: simulate(
                root, Trial("pilot", 20, seed, 100, "startup", f"rate_{rate:g}", 0.0 if rate == LOAD_GRID[0] else rate),
                trace=trace, timeout_s=timeout_s,
            ), PILOT_SEEDS))
        runs = [pilot_metrics(path) for path in paths]
        label = classify_pilots(runs)
        outcomes[rate] = {"classification": label, "seed_metrics": dict(zip(PILOT_SEEDS, runs))}
        write_json(root / "calibration_pilots.json", outcomes)
        return label

    for rate in LOAD_GRID:
        if measure(rate) == "overloaded":
            break
    heavy = min((rate for rate, data in outcomes.items() if data["classification"] == "overloaded"), default=None)
    stable = sorted(rate for rate, data in outcomes.items() if data["classification"] == "stable")
    if heavy is None or not stable:
        raise RuntimeError("Pilot did not bracket stable and overloaded UL regimes. Inspect calibration_pilots.json.")
    if len(stable) == 1:
        lower, upper = stable[0], heavy
        for _ in range(4):
            candidate = round((lower + upper) / 2, 3)
            if measure(candidate) == "stable":
                stable.append(candidate)
                lower = candidate
            else:
                upper = candidate
    if len(stable) < 2:
        raise RuntimeError("Pilot could not establish a distinct stable near-capacity rate.")
    rates = {"light": min(stable), "near_capacity": max(stable), "overloaded": heavy}
    serial_estimate_hours = (96 * statistics.mean(
        metrics["elapsed_s"] for result in outcomes.values() for metrics in result["seed_metrics"].values()
    ) + 21 * 3000 + 3 * 2000) / 3600
    result = {"rates_mbps": rates, "pilot_seeds": PILOT_SEEDS, "serial_estimate_hours": serial_estimate_hours}
    write_json(root / "calibrated_loads.json", result)
    print(f"Calibrated rates {rates}. Estimated simulation matrix {serial_estimate_hours:.0f} serial hours.", flush=True)
    return result


def preflight(root: Path, digest: str, timeout_s: float) -> dict:
    """Rerun the two collapsed 20-vehicle seeds before the full study."""
    results = []
    for seed in (2, 7):
        archived_run = archived(20, seed)
        legacy_run = root / "legacy" / f"original_20veh_seed{seed}"
        if not (legacy_run / "run_summary.json").is_file():
            raise RuntimeError(f"Complete the archived-action legacy replay before preflight: {legacy_run}")
        replay = packets(legacy_run)
        baseline = packets(archived_run)
        current_run = simulate(
            root,
            Trial("original", 20, seed, 100, "live"),
            expected_digest=digest,
            timeout_s=timeout_s,
        )
        current = packets(current_run)
        summary = read_json(current_run / "run_summary.json")
        results.append({
            "seed": seed,
            "archived": baseline,
            "legacy_replay": replay,
            "revised_adaptive_llm": current,
            "legacy_replay_critical_delta_pp": 100 * (replay["critical_dc_prr"] - baseline["critical_dc_prr"]),
            "revised_critical_delta_pp": 100 * (current["critical_dc_prr"] - baseline["critical_dc_prr"]),
            "revised_tod_ul_delta_pp": 100 * (current["tod_ul_dc_prr"] - baseline["tod_ul_dc_prr"]),
            "run_summary": summary,
        })

    replay_consistent = all(abs(row["legacy_replay_critical_delta_pp"]) <= 1.0 for row in results)
    improved_collapsed_seeds = all(row["revised_critical_delta_pp"] >= 5.0 for row in results)
    active_proposals = all(row["run_summary"]["published_proposals"] > 0 for row in results)
    report = {
        "purpose": "short go/no-go check on the two previously collapsed 20-vehicle seeds",
        "sim_time_s": 100,
        "seeds": [2, 7],
        "checks": {
            "legacy_replay_within_1pp_of_archive": replay_consistent,
            "revised_critical_dc_prr_improves_by_at_least_5pp_on_both_seeds": improved_collapsed_seeds,
            "revised_llm_publishes_at_least_one_proposal_per_seed": active_proposals,
        },
        "gate": "pass" if replay_consistent and improved_collapsed_seeds and active_proposals else "hold",
        "runs": results,
        "interpretation": "This small pilot is a diagnostic gate, not a statistical claim.",
    }
    write_json(root / "preflight_collapsed_seeds.json", report)
    print(f"PRE-FLIGHT {report['gate'].upper()}: "
          f"critical DC-PRR deltas {[round(row['revised_critical_delta_pp'], 2) for row in results]} pp", flush=True)
    if report["gate"] != "pass":
        raise RuntimeError(f"Adaptive LLM preflight did not pass; inspect {root / 'preflight_collapsed_seeds.json'}")
    return report


def canary(root: Path, timeout_s: float) -> None:
    live = simulate(root, Trial("canary", 20, 1, 5, "canary", "async", period_s=0.5), timeout_s=timeout_s)
    actions = jsonl(live / "actions.jsonl")
    updates = [read_json(path) for path in sorted((live / "refreshes").glob("*.json"))]
    if not any(
        any(update["wall_request_s"] < row["wall_action_s"] < update["wall_complete_s"] for row in actions)
        for update in updates if "wall_request_s" in update
    ):
        raise RuntimeError("Timing canary did not observe xApp actions during delayed inference")
    proposals = read_json(live / "proposals.json")
    if not proposals:
        raise RuntimeError("Timing canary ended before any delayed policy activated")
    replay = simulate(root, Trial("canary", 20, 1, 5, "replay", "replay"), proposals=proposals, timeout_s=timeout_s)
    if signature(actions) != signature(jsonl(replay / "actions.jsonl")) or packets(live) != packets(replay):
        raise RuntimeError("Shield-on replay differs from live canary activation, actions, or packet outcomes")
    write_json(root / "canary_verified.json", {"async_progress": True, "resume_replay_parity": True})
    print("PASS ns-3 timing and replay canary", flush=True)


def publish_matrix(root: Path, rates: dict) -> None:
    write_json(root / "seed_manifest.json", {
        "failure_reproduction": {"seeds": [1, 2, 7], "densities": [20], "seconds": 100},
        "revised_original": {"seeds": list(range(1, 8)), "densities": [20, 25, 30], "seconds": 100},
        "calibrated_live": {"seeds": list(MAIN_SEEDS), "densities": [20], "seconds": 100, "loads_mbps": rates},
        "frozen_near_capacity": {"seeds": list(MAIN_SEEDS), "seconds": 100},
        "shield_replay_near_capacity": {"seeds": list(SHIELD_SEEDS), "seconds": 100, "settings": ["on", "off"]},
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "adaptive_revision_v2")
    parser.add_argument("--phase", choices=("all", "archive", "mobility", "canary", "legacy", "preflight", "prompts",
                                           "original", "calibrate", "matrix", "report"), default="all")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=21600)
    parser.add_argument("--legacy-binary", type=Path)
    args = parser.parse_args()
    if args.jobs < 1 or args.timeout_s <= 0:
        parser.error("jobs and timeout must be positive")
    root = args.output.resolve()
    if root == (PROJECT_ROOT / "results").resolve() or root.is_relative_to((PROJECT_ROOT / "results/raw").resolve()):
        parser.error("use a study directory outside results/raw")
    root.mkdir(parents=True, exist_ok=True)

    def enabled(*phases: str) -> bool:
        return args.phase in ("all", *phases)

    def run(trial: Trial, **kwargs) -> Path:
        return simulate(root, trial, timeout_s=args.timeout_s, **kwargs)

    def parallel(fn, rows) -> None:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            list(executor.map(fn, rows))

    if enabled("archive"):
        result = archive_audit(root)
        print({key: value for key, value in result.items() if key != "candidates"}, flush=True)
    if enabled("canary"):
        canary(root, args.timeout_s)
    if enabled("legacy"):
        def reproduce(seed: int) -> None:
            original = archived(20, seed)
            # Reproduce the archived controller without the newly added shield.
            result = run(Trial("legacy", 20, seed, 100, "legacy", shield="off"),
                         proposals=offline_archived_actions(original), old_binary=args.legacy_binary)
            diagnose_legacy(result, original)
        parallel(reproduce, (1, 2, 7))
    if enabled("prompts", "original", "preflight", "matrix"):
        digest = pinned_model(root)
    if enabled("preflight"):
        preflight(root, digest, args.timeout_s)
    if enabled("prompts"):
        prompt_audit(root, digest)
    if enabled("original"):
        parallel(lambda row: run(Trial("original", *row, 100, "live"), expected_digest=digest),
                 product((20, 25, 30), range(1, 8)))
    if enabled("mobility", "calibrate", "matrix"):
        trace = mobility(root)
        if args.phase == "mobility":
            print(trace)
    if enabled("calibrate"):
        calibrate(root, trace, args.jobs, args.timeout_s)
    if enabled("matrix"):
        if not (root / "canary_verified.json").is_file() or not (root / "prompt_audit/results.json").is_file():
            raise RuntimeError("Complete the timing/replay canary and prompt audit before the main sweep")
        rates = read_json(root / "calibrated_loads.json")["rates_mbps"]
        publish_matrix(root, rates)

        def main_seed(seed: int) -> None:
            live_inputs = {}
            for load in ("near_capacity", "light", "overloaded"):
                rate = 0 if rates[load] == LOAD_GRID[0] else rates[load]
                live_inputs[load] = run(Trial("live", 20, seed, 100, "live", load, rate), trace=trace, expected_digest=digest)
                if load == "near_capacity":
                    proposals = read_json(live_inputs[load] / "proposals.json")
                    run(Trial("frozen", 20, seed, 100, "replay", load, rate), trace=trace, proposals=proposals[:1])
                    if seed in SHIELD_SEEDS:
                        on = run(Trial("shield_on", 20, seed, 100, "replay", load, rate), trace=trace, proposals=proposals)
                        if signature(jsonl(on / "actions.jsonl")) != signature(jsonl(live_inputs[load] / "actions.jsonl")):
                            raise RuntimeError(f"Shield-on replay parity failed for seed {seed}")
                        run(Trial("shield_off", 20, seed, 100, "replay", load, rate, shield="off"), trace=trace, proposals=proposals)
        parallel(main_seed, MAIN_SEEDS)
    if enabled("report"):
        summarize(root)


if __name__ == "__main__":
    main()
