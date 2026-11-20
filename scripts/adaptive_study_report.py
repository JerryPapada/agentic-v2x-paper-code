"""Packet metrics, paired inference, and archive audit for Adaptive LLM."""

from __future__ import annotations

import csv
import random
import statistics
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from adaptive_study_runner import archived, jsonl
from llm_v2x_scheduler.adaptive_study import assess_response, policy_content
from llm_v2x_scheduler.utils import read_json, write_json

CRITICAL_DIRECTIONS = ("tod_ul", "tod_dl", "awareness_ul", "awareness_dl")


def packets(path: Path) -> dict:
    with (path / "class_report.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    metrics = {}
    names = {"ToD": "tod", "AIC": "awareness", "HDM": "hdmap", "RTSA": "sensor"}
    for row in rows:
        label = f"{names[row['use_case']]}_{row['direction'].lower()}"
        sent = int(row["tx_pkts"])
        on_deadline = row.get("deadline_met_pkts")
        metrics[f"{label}_dc_prr"] = (
            int(on_deadline) / sent if sent and on_deadline is not None
            else float(row["deadline_met_ratio"]) if sent else None
        )
        metrics[f"{label}_p95_ms"] = float(row["p95_ms"]) if int(row["rx_pkts"]) else None
        metrics[f"{label}_mbps"] = float(row["mbps_delivered"])
    metrics["background_mbps"] = sum(metrics[f"{name}_{direction}_mbps"] for name in ("hdmap", "sensor") for direction in ("dl", "ul"))
    critical = [metrics[f"{direction}_dc_prr"] for direction in CRITICAL_DIRECTIONS]
    if any(value is None for value in critical):
        raise ValueError(f"A critical direction has zero transmitted packets in {path}")
    metrics["critical_dc_prr"] = min(critical)
    return metrics


def archive_audit(root: Path) -> dict:
    runs = []
    archive_repairs = 0
    archive_attempts = 0
    distinct = set()
    for density in (20, 25, 30):
        for seed in range(1, 8):
            path = archived(density, seed)
            archive_reports = sorted((path / "rapp_updates").glob("*_report.json"))
            archive_repairs += sum(read_json(f).get("repaired", False) for f in archive_reports)
            archive_attempts += len(archive_reports)
            with (path / "llm_v2x_xapp_log.csv").open() as stream:
                trace = list(csv.DictReader(stream))
            activation_versions = {
                int(row["policy_version"])
                for prior, row in zip(trace, trace[1:])
                if prior["policy_id"] != row["policy_id"] and row["policy_id"] != "heuristic_xapp"
                and not row["fallback_used"] == "1"
            }
            raw_files = sorted(f for f in (path / "rapp_updates/raw").glob("*.txt") if not f.stem.endswith("_failed"))
            for update_index, file in enumerate(raw_files, 1):
                check, policy = assess_response(file.read_text())
                content = policy_content(policy) if policy else None
                if content:
                    distinct.add(content)
                runs.append({
                    "density": density, "seed": seed, "update_index": update_index,
                    "raw_file": str(file.relative_to(path)), "content": content,
                    "active_version_in_archive": update_index in activation_versions, **check,
                })
    result = {
        "archived_reported_attempts": archive_attempts, "archived_reported_repairs": archive_repairs,
        "raw_responses": len(runs), "strict_pass": sum(row["strict_pass"] for row in runs),
        "repaired": sum(row["repaired"] for row in runs), "rejected": sum(not row["accepted"] for row in runs),
        "distinct_executable_after_revalidation": len(distinct),
        "active_version_in_archive": sum(row["active_version_in_archive"] for row in runs), "candidates": runs,
    }
    write_json(root / "archive_audit.json", result)
    return result


def signature(rows: list[dict]) -> list[tuple]:
    return [(row["step"], row["active_content"], row["effective_ul"], row["effective_dl"]) for row in rows]


def diagnose_legacy(result: Path, source: Path) -> None:
    with (source / "llm_v2x_xapp_log.csv").open() as stream:
        original = {int(row["step"]): row for row in csv.DictReader(stream)}
    with (source / "step_log.csv").open() as stream:
        old_ns3 = {int(row["step"]): row for row in csv.DictReader(stream)}
    with (result / "step_log.csv").open() as stream:
        new_ns3 = {int(row["step"]): row for row in csv.DictReader(stream)}
    observed = jsonl(result / "actions.jsonl")
    divergences = {}
    for row in observed:
        step = row["step"]
        if step not in original:
            continue
        archived_row = original[step]
        keys = {
            "active_policy": (archived_row["policy_id"], row["active_revision"]),
            "ul_weights": (archived_row["ul_weights_per_ue"], ";".join(map(str, row["effective_ul"]))),
            "dl_weights": (archived_row["dl_weights_per_ue"], ";".join(map(str, row["effective_dl"]))),
            "tod_ul_delivery_proxy": (float(archived_row["tod_dc_prr"] or 1), row["legacy_tod_ul_dc_prr_proxy"]),
        }
        if step in old_ns3 and step in new_ns3:
            keys["ul_queue_bytes"] = (old_ns3[step]["ul_buffer_per_ue"], new_ns3[step]["ul_buffer_per_ue"])
        for key, (before, after) in keys.items():
            if before != after and key not in divergences:
                divergences[key] = {"step": step, "sim_time_s": row["sim_time_s"], "archived": before, "replay": after}
    write_json(result / "diagnostic.json", {
        "first_divergences": divergences,
        "packet_outcomes": {"archived": packets(source), "replay": packets(result)},
    })


def paired_effect(values: list[float]) -> dict:
    rng = random.Random(4201)
    n = len(values)
    effect = statistics.mean(values)
    observed = abs(sum(values))
    boot = sorted(statistics.mean(rng.choices(values, k=n)) for _ in range(10000))
    draws = 100000
    extremes = sum(
        abs(sum(delta if rng.getrandbits(1) else -delta for delta in values)) >= observed - 1e-12
        for _ in range(draws)
    )
    return {
        "n_pairs": n, "mean_delta": effect, "median_delta": statistics.median(values),
        "ci95": [boot[249], boot[9749]], "permutation_p": (extremes + 1) / (draws + 1),
    }


def holm(results: dict) -> None:
    previous = 0.0
    for i, key in enumerate(sorted(results, key=lambda key: results[key]["permutation_p"])):
        previous = max(previous, min(1.0, (len(results) - i) * results[key]["permutation_p"]))
        results[key]["holm_p"] = previous


def summarize(root: Path) -> None:
    rows = []
    for stage in ("legacy", "original", "live", "frozen", "shield_on", "shield_off"):
        for path in sorted((root / stage).glob("*/run_summary.json")):
            summary = read_json(path)
            if not summary["success"]:
                continue
            manifest = read_json(path.parent / "manifest.json")
            rows.append({
                **{k: manifest[k] for k in ("stage", "load", "density", "seed", "seconds", "sensor_ul_mbps", "shield")},
                **packets(path.parent), **summary,
            })
    if not rows:
        raise ValueError("No successful study trials to summarize")
    with (root / "run_metrics.csv").open("w") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    live = {row["seed"]: row for row in rows if row["stage"] == "live" and row["load"] == "near_capacity"}
    frozen = {row["seed"]: row for row in rows if row["stage"] == "frozen"}
    shield_on = {row["seed"]: row for row in rows if row["stage"] == "shield_on"}
    shield_off = {row["seed"]: row for row in rows if row["stage"] == "shield_off"}
    pairs = sorted(live.keys() & frozen.keys())
    results = {"pair_seeds": pairs, "run_counts": dict(Counter(row["stage"] for row in rows))}
    if pairs == list(range(11, 31)):
        metrics = ("critical_dc_prr", "tod_ul_dc_prr", "background_mbps")
        contrasts = {}
        for metric in metrics:
            scale = 1 if metric == "background_mbps" else 100
            contrasts[metric] = paired_effect([scale * (live[seed][metric] - frozen[seed][metric]) for seed in pairs])
        results["primary"] = contrasts.pop("critical_dc_prr")
        holm(contrasts)
        results["secondary"] = contrasts
    shield_pairs = sorted(shield_on.keys() & shield_off.keys())
    results["shield_seed_deltas_pp"] = {
        seed: 100 * (shield_on[seed]["critical_dc_prr"] - shield_off[seed]["critical_dc_prr"]) for seed in shield_pairs
    }
    write_json(root / "paired_results.json", results)
    model_dirs = [path.parent for stage in ("original", "live") for path in (root / stage).glob("*/run_summary.json")]
    directories = model_dirs or [path.parent for path in sorted((root / "canary").glob("*/run_summary.json"))]
    if directories:
        reference = next((path for path in directories if path.name == "near_capacity_20veh_seed11"), directories[0])
        config = read_json(reference / "run_config.json")
        write_json(root / "table4_settings.json", config)
    lines = ["# Adaptive LLM paper results", ""]
    for stage, count in results["run_counts"].items():
        lines.append(f"- {stage}: {count} completed runs.")
    lines.extend(["", f"Paired refresh comparison: {len(pairs)} of 20 planned pairs available."])
    if "primary" in results:
        estimate = results["primary"]
        lo, hi = estimate["ci95"]
        lines.append(f"Mean live minus frozen worst-direction critical DC-PRR was {estimate['mean_delta']:.2f} pp "
                     f"(95% CI {lo:.2f} to {hi:.2f} pp, paired permutation p = {estimate['permutation_p']:.4f}).")
    updates = [
        read_json(file) for path in model_dirs for file in (path / "refreshes").glob("*.json")
    ]
    attempted = [row for row in updates if "latency_s" in row]
    if attempted:
        lags = [row["activation_sim_time_s"] - row["scheduled_sim_time_s"] for row in attempted if "activation_sim_time_s" in row]
        lines.extend(["", f"Live inference exceeded one 10 s wall period on "
                      f"{sum(row['missed_wall_period'] for row in attempted)} of {len(attempted)} attempts. "
                      f"The xApp skipped {sum(row['status'] == 'busy_skip' for row in updates)} scheduled slots while inference was busy."])
        if lags:
            lines.append(f"Median simulated activation lag was {statistics.median(lags):.2f} s across {len(lags)} published updates.")
    (root / "paper_results.md").write_text("\n".join(lines) + "\n")
