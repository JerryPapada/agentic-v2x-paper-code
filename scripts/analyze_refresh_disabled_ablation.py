#!/usr/bin/env python3
"""Analyze verified paired refresh-disabled runs against the original cohort."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
from pathlib import Path

import _bootstrap  # noqa: F401

from adaptive_study_report import packets
from adaptive_study_runner import ORIGINAL_COHORT, REFRESH_DISABLED_LABEL, read_json, verify_refresh_disabled
from analyze_original_cohort import T_975_DF6, group_summary

DENSITIES = (20, 25, 30)
SEEDS = tuple(range(1, 8))
ENDPOINTS = (
    "critical_dc_prr_pct",
    "tod_ul_dc_prr_pct", "tod_dl_dc_prr_pct",
    "awareness_ul_dc_prr_pct", "awareness_dl_dc_prr_pct",
    "background_mbps", "background_pct_of_target",
)
PRR_ENDPOINTS = ENDPOINTS[:5]
CONDITION_ADAPTIVE = "LLM-adaptive (periodic refresh)"
CONDITION_FROZEN = REFRESH_DISABLED_LABEL


def background_target(config: dict) -> float:
    mix = config["class_mix"]
    traffic = config["traffic_spec"]
    return sum(
        int(mix[name]) * (float(traffic[name]["dl_mbps_target"]) + float(traffic[name]["ul_mbps_target"]))
        for name in ("HDM", "RTSA")
    )


def read_metrics(run: Path, *, refresh_disabled: bool) -> tuple[dict, dict]:
    summary_path = run / "run_summary.json"
    config_path = run / "run_config.json"
    if not summary_path.is_file() or not read_json(summary_path).get("success"):
        raise ValueError(f"Run did not complete successfully: {run}")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = read_json(config_path)
    if refresh_disabled:
        verify_refresh_disabled(run)
    metric = packets(run)
    target = background_target(config)
    result = {
        "critical_dc_prr_pct": 100 * metric["critical_dc_prr"],
        "tod_ul_dc_prr_pct": 100 * metric["tod_ul_dc_prr"],
        "tod_dl_dc_prr_pct": 100 * metric["tod_dl_dc_prr"],
        "awareness_ul_dc_prr_pct": 100 * metric["awareness_ul_dc_prr"],
        "awareness_dl_dc_prr_pct": 100 * metric["awareness_dl_dc_prr"],
        "background_mbps": metric["background_mbps"],
        "background_target_mbps": target,
        "background_pct_of_target": 100 * metric["background_mbps"] / target,
    }
    return result, config


def paired_counts(values: list[float]) -> dict:
    epsilon = 1e-12
    return {
        "wins": sum(value > epsilon for value in values),
        "ties": sum(abs(value) <= epsilon for value in values),
        "losses": sum(value < -epsilon for value in values),
    }


def mean_ci(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "sd": None, "ci95_mean": None}
    if len(values) == 1:
        return {"n": 1, "mean": values[0], "sd": None, "ci95_mean": None}
    stats = group_summary(values)
    # group_summary uses the original cohort's t interval for n=7. For partial
    # execution, leave an interval blank rather than applying the wrong df.
    if len(values) != 7:
        stats["ci95_mean"] = None
    return stats


def exact_sign_flip(values: list[float]) -> float:
    observed = abs(sum(values))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        extreme += abs(sum(value * sign for value, sign in zip(values, signs))) >= observed - 1e-12
        total += 1
    return extreme / total


def guard_metrics(run: Path) -> dict:
    actions = [json.loads(line) for line in (run / "actions.jsonl").read_text().splitlines() if line.strip()]
    reasons = [reason for row in actions for reason in row.get("reasons", [])]
    return {
        "control_windows": len(actions),
        "safety_guard_reason_events": sum(str(reason).startswith("safety_") for reason in reasons),
        "directional_ul": sum(bool(row.get("directional_ul", "safety_directional_dc_prr_gap_ul" in row.get("reasons", []))) for row in actions),
        "directional_dl": sum(bool(row.get("directional_dl", "safety_directional_dc_prr_gap_dl" in row.get("reasons", []))) for row in actions),
        "rollback_trigger": sum(bool(row.get("rollback_trigger", "rollback_policy_performance_degradation" in row.get("reasons", []))) for row in actions),
        "rollback_hold": sum(bool(row.get("rollback_hold", "rollback_active_holding_previous_policy" in row.get("reasons", []))) for row in actions),
    }


def make_pairs(root: Path) -> tuple[list[dict], list[dict], list[str]]:
    pairs = []
    guard_rows = []
    missing = []
    for density, seed in itertools.product(DENSITIES, SEEDS):
        adaptive = ORIGINAL_COHORT / f"original_{density}veh_seed{seed}"
        frozen = root / "refresh_disabled" / f"original_{density}veh_seed{seed}"
        if not (adaptive / "run_summary.json").is_file() or not (frozen / "run_summary.json").is_file():
            missing.append(f"{density}/{seed}")
            continue
        try:
            adaptive_values, adaptive_config = read_metrics(adaptive, refresh_disabled=False)
            frozen_values, frozen_config = read_metrics(frozen, refresh_disabled=True)
            if adaptive_config != frozen_config:
                raise ValueError(f"Matched run configs differ at {density}/{seed}")
            original_manifest = read_json(adaptive / "manifest.json")
            frozen_manifest = read_json(frozen / "manifest.json")
            for field in ("binary_sha256", "trace_sha256", "ns3_rng_seed", "ns3_rng_run"):
                if original_manifest.get(field) != frozen_manifest.get(field):
                    raise ValueError(f"Matched {field} differs at {density}/{seed}")
        except (FileNotFoundError, ValueError, KeyError) as exc:
            missing.append(f"{density}/{seed} (verification failed: {exc})")
            continue

        row = {
            "density": density,
            "seed": seed,
            "adaptive_condition": CONDITION_ADAPTIVE,
            "refresh_disabled_condition": CONDITION_FROZEN,
            "adaptive_output": str(adaptive.relative_to(ORIGINAL_COHORT.parents[1])),
            "refresh_disabled_output": str(frozen.relative_to(root)),
        }
        for endpoint in ENDPOINTS:
            row[f"adaptive_{endpoint}"] = adaptive_values[endpoint]
            row[f"refresh_disabled_{endpoint}"] = frozen_values[endpoint]
            row[f"delta_{endpoint}"] = adaptive_values[endpoint] - frozen_values[endpoint]
        pairs.append(row)
        for condition, run in ((CONDITION_ADAPTIVE, adaptive), (CONDITION_FROZEN, frozen)):
            counters = guard_metrics(run)
            guard_rows.append({"density": density, "seed": seed, "condition": condition, **counters})
    return pairs, guard_rows, missing


def build_summary(pairs: list[dict], guard_rows: list[dict], missing: list[str]) -> dict:
    by_density = {}
    win_loss = {}
    for density in DENSITIES:
        subset = [row for row in pairs if row["density"] == density]
        by_density[str(density)] = {"n_pairs": len(subset), "metrics": {}}
        for endpoint in ENDPOINTS:
            adaptive_values = [row[f"adaptive_{endpoint}"] for row in subset]
            frozen_values = [row[f"refresh_disabled_{endpoint}"] for row in subset]
            deltas = [row[f"delta_{endpoint}"] for row in subset]
            by_density[str(density)]["metrics"][endpoint] = {
                "adaptive": mean_ci(adaptive_values),
                "refresh_disabled": mean_ci(frozen_values),
                "paired_difference_adaptive_minus_refresh_disabled": mean_ci(deltas),
                "paired_wins_ties_losses": paired_counts(deltas),
            }
            win_loss[f"{density}/{endpoint}"] = paired_counts(deltas)

    seed_blocks = []
    complete_design = len(pairs) == 21 and not missing
    aggregate = None
    if complete_design:
        for seed in SEEDS:
            seed_rows = [row for row in pairs if row["seed"] == seed]
            primary_delta = statistics.mean(row["delta_critical_dc_prr_pct"] for row in seed_rows)
            seed_blocks.append({"seed": seed, "mean_delta_critical_dc_prr_pp": primary_delta})
        block_deltas = [row["mean_delta_critical_dc_prr_pp"] for row in seed_blocks]
        aggregate = {
            "label": "Separate exploratory ablation contrast",
            "estimator": "mean of three density-specific paired differences within each run index, then mean across seven run-index blocks",
            "n_seed_blocks": 7,
            "block_mean_delta_pp": statistics.mean(block_deltas),
            "seed_block_95pct_t_interval_pp": mean_ci(block_deltas)["ci95_mean"],
            "exact_two_sided_sign_flip_p": exact_sign_flip(block_deltas),
            "paired_wins_ties_losses_across_seed_blocks": paired_counts(block_deltas),
            "block_differences_pp": block_deltas,
        }

    guard_summary = {}
    for condition in (CONDITION_ADAPTIVE, CONDITION_FROZEN):
        selected = [row for row in guard_rows if row["condition"] == condition]
        guard_summary[condition] = {
            "n_runs": len(selected),
            "totals": {
                key: sum(row[key] for row in selected)
                for key in ("safety_guard_reason_events", "directional_ul", "directional_dl", "rollback_trigger", "rollback_hold")
            },
            "mean_per_run": {
                key: (statistics.mean(row[key] for row in selected) if selected else None)
                for key in ("safety_guard_reason_events", "directional_ul", "directional_dl", "rollback_trigger", "rollback_hold")
            },
            "by_density": {
                str(density): {
                    key: sum(row[key] for row in selected if row["density"] == density)
                    for key in ("safety_guard_reason_events", "directional_ul", "directional_dl", "rollback_trigger", "rollback_hold")
                }
                for density in DENSITIES
            },
        }
    return {
        "analysis": "paired exploratory refresh-disabled ablation",
        "difference_definition": "LLM-adaptive (periodic refresh) minus LLM-adaptive (refresh disabled) for every metric",
        "positive_dc_prr_difference": "favors periodic refresh",
        "primary_endpoint": "Within each run, the minimum DC-PRR across ToD UL, ToD DL, Awareness UL, and Awareness DL; differences reported in percentage points.",
        "density_results": by_density,
        "aggregate_exploratory_contrast": aggregate,
        "aggregate_run_index_blocks": seed_blocks,
        "paired_wins_ties_losses": win_loss,
        "guard_and_rollback_activity": guard_summary,
        "verified_pairs": len(pairs),
        "complete_21_pair_design": complete_design,
        "unavailable_or_failed_pairs": missing,
        "statistical_settings": {
            "within_density_interval": "two-sided 95% t interval over seven paired run-index differences, using the original cohort df=6 interval; omitted if fewer than seven pairs",
            "aggregate_interval": "two-sided 95% t interval over seven seed-block differences, df=6",
            "aggregate_p_value": "exact two-sided sign-flip over all 2^7 sign assignments of the seven seed-block differences",
            "tie_tolerance": 1e-12,
        },
        "interpretation": "Exploratory ablation contrast; original five-baseline analysis remains separate and is not modified by this report.",
    }


def fmean_ci(stats: dict, *, digits: int = 2, suffix: str = "") -> str:
    mean = stats.get("mean")
    if mean is None:
        return "—"
    sd = stats.get("sd")
    center = f"{mean:.{digits}f}" + (f" ({sd:.{digits}f})" if sd is not None else "")
    interval = stats.get("ci95_mean")
    if interval is not None:
        center += f" [{interval[0]:.{digits}f}, {interval[1]:.{digits}f}]"
    return center + suffix


def render_markdown(summary: dict, pairs: list[dict], guard_rows: list[dict]) -> str:
    lines = [
        "# LLM-adaptive refresh-disabled ablation",
        "",
        f"Verified matched pairs: {summary['verified_pairs']}/21. Every difference is periodic-refresh LLM-adaptive minus {REFRESH_DISABLED_LABEL}; positive DC-PRR differences favor periodic refresh.",
        "",
        "## Primary endpoint by density",
        "",
        "The endpoint is calculated within each run as the minimum of ToD UL, ToD DL, Awareness UL, and Awareness DL DC-PRR. Values are percentages; condition means are followed by sample SD. Paired differences are percentage points with a two-sided 95% t interval across seven run-index pairs.",
        "",
        "| Vehicles | LLM-adaptive mean (SD), % | Refresh disabled mean (SD), % | Paired difference, pp [95% CI] | Paired W/T/L |",
        "|---:|---:|---:|---:|---:|",
    ]
    for density in DENSITIES:
        item = summary["density_results"][str(density)]["metrics"]["critical_dc_prr_pct"]
        delta = item["paired_difference_adaptive_minus_refresh_disabled"]
        lines.append(
            f"| {density} | {fmean_ci(item['adaptive'])} | {fmean_ci(item['refresh_disabled'])} | "
            f"{fmean_ci(delta, digits=3)} | {item['paired_wins_ties_losses']['wins']}/{item['paired_wins_ties_losses']['ties']}/{item['paired_wins_ties_losses']['losses']} |"
        )

    lines.extend([
        "",
        "## Directional DC-PRR and background throughput",
        "",
        "Condition values are means across paired runs; differences are paired means. Background target percentage is delivered HDMap plus Sensor throughput divided by the nominal offered target from each matched run configuration.",
        "",
        "| Vehicles | Metric | LLM-adaptive | Refresh disabled | Difference | W/T/L |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    names = {
        "tod_ul_dc_prr_pct": "ToD UL DC-PRR (%)", "tod_dl_dc_prr_pct": "ToD DL DC-PRR (%)",
        "awareness_ul_dc_prr_pct": "Awareness UL DC-PRR (%)", "awareness_dl_dc_prr_pct": "Awareness DL DC-PRR (%)",
        "background_mbps": "Background throughput (Mbps)", "background_pct_of_target": "Background / target (%)",
    }
    for density in DENSITIES:
        for endpoint in ENDPOINTS[1:]:
            item = summary["density_results"][str(density)]["metrics"][endpoint]
            units = " pp" if endpoint.endswith("_pct") and endpoint not in ("background_pct_of_target",) else (" Mbps" if endpoint == "background_mbps" else " pp")
            lines.append(
                f"| {density} | {names[endpoint]} | {fmean_ci(item['adaptive'], digits=2)} | "
                f"{fmean_ci(item['refresh_disabled'], digits=2)} | "
                f"{fmean_ci(item['paired_difference_adaptive_minus_refresh_disabled'], digits=3)}{units} | "
                f"{item['paired_wins_ties_losses']['wins']}/{item['paired_wins_ties_losses']['ties']}/{item['paired_wins_ties_losses']['losses']} |"
            )

    aggregate = summary["aggregate_exploratory_contrast"]
    lines.extend(["", "## Separate exploratory ablation contrast", ""])
    if aggregate is None:
        lines.append("Not estimable from the current verified subset: all seven run-index blocks need all three density pairs.")
    else:
        low, high = aggregate["seed_block_95pct_t_interval_pp"]
        wtl = aggregate["paired_wins_ties_losses_across_seed_blocks"]
        lines.append(
            f"Averaging the three density-specific primary-endpoint differences within each run index gave "
            f"{aggregate['block_mean_delta_pp']:.3f} pp (seed-block 95% t interval {low:.3f} to {high:.3f} pp; "
            f"exact two-sided sign-flip p={aggregate['exact_two_sided_sign_flip_p']:.5f}; "
            f"block wins/ties/losses {wtl['wins']}/{wtl['ties']}/{wtl['losses']}; n=7 blocks)."
        )
        lines.append("This is labeled separately as exploratory and does not replace the paper’s five-baseline analysis.")

    lines.extend(["", "## Guard and rollback activity", ""])
    lines.append("Totals are summed across verified runs; mean is per run. Safety-guard reason events count logged controller reasons, while directional UL/DL columns count directional-guard activations. Zero rollback activity is reported as observed.")
    lines.extend([
        "",
        "| Condition | Runs | Safety-guard reason events | Directional UL | Directional DL | Rollback trigger | Rollback hold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for condition in (CONDITION_ADAPTIVE, CONDITION_FROZEN):
        item = summary["guard_and_rollback_activity"][condition]
        total = item["totals"]
        lines.append(
            f"| {condition} | {item['n_runs']} | {total['safety_guard_reason_events']} | {total['directional_ul']} | {total['directional_dl']} | "
            f"{total['rollback_trigger']} | {total['rollback_hold']} |"
        )

    if summary["unavailable_or_failed_pairs"]:
        lines.extend(["", "## Unavailable pairs", ""])
        lines.extend(f"- {item}" for item in summary["unavailable_or_failed_pairs"])

    primary_means = []
    density_effects = []
    for density in DENSITIES:
        item = summary["density_results"][str(density)]["metrics"]["critical_dc_prr_pct"]
        if item["adaptive"]["mean"] is not None and item["refresh_disabled"]["mean"] is not None:
            primary_means.append((density, item["adaptive"]["mean"], item["refresh_disabled"]["mean"]))
            density_effects.append(item["paired_difference_adaptive_minus_refresh_disabled"]["mean"])
    if primary_means:
        lead = "; ".join(f"{density}-vehicle mean Δ {delta:+.3f} pp" for (density, _, _), delta in zip(primary_means, density_effects))
        if aggregate is not None:
            interval = aggregate["seed_block_95pct_t_interval_pp"]
            pvalue = aggregate["exact_two_sided_sign_flip_p"]
            uncertainty = f"Across the three density contrasts, the seed-block mean was {aggregate['block_mean_delta_pp']:+.3f} pp (95% interval {interval[0]:+.3f} to {interval[1]:+.3f}; exact p={pvalue:.5f})."
        else:
            uncertainty = "The aggregate seed-block contrast is not estimable until all seven run-index blocks are complete."
        bg = [summary["density_results"][str(d)]["metrics"]["background_mbps"]["paired_difference_adaptive_minus_refresh_disabled"]["mean"] for d in DENSITIES]
        bg_pct = [summary["density_results"][str(d)]["metrics"]["background_pct_of_target"]["paired_difference_adaptive_minus_refresh_disabled"]["mean"] for d in DENSITIES]
        observed_bg = [(d, b, p) for d, b, p in zip(DENSITIES, bg, bg_pct) if b is not None and p is not None]
        tradeoff = "; ".join(f"{d} vehicles background Δ {b:+.3f} Mbps ({p:+.2f} target percentage points)" for d, b, p in observed_bg)
        lines.extend(["", "## Results paragraph", ""])
        lines.append(
            f"The matched refresh-disabled ablation showed {lead}. {uncertainty} "
            f"Background throughput differences were {tradeoff}. All four directional endpoints are shown above because the minimum endpoint can conceal UL/DL-specific effects."
        )

    lines.extend([
        "",
        "## Analysis notes",
        "",
        "The reference condition is the 21-run LLM-adaptive cohort in `adaptive_revision_v2_adaptive_prompt_fix/original/`; the five baseline result artifacts were not read or changed. The paired CSV retains all run-level values and differences. See `aggregate_seed_blocks.csv` for the three-density block differences when the complete design is available.",
        "",
        "A DC-PRR paired win means periodic-refresh LLM-adaptive exceeded refresh-disabled on that pair. For throughput, positive differences likewise indicate greater delivered background throughput under periodic refresh; they are not classified as a safety benefit.",
    ])
    return "\n".join(lines) + "\n"


def write_outputs(root: Path, pairs: list[dict], guard_rows: list[dict], missing: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "paired_results.csv"
    fields = list(pairs[0]) if pairs else ["density", "seed", "adaptive_condition", "refresh_disabled_condition"]
    if pairs:
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(pairs)
    else:
        csv_path.write_text(",".join(fields) + "\n")

    summary = build_summary(pairs, guard_rows, missing)
    aggregate_rows = summary["aggregate_run_index_blocks"]
    with (root / "aggregate_seed_blocks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("seed", "mean_delta_critical_dc_prr_pp"))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    with (root / "guard_activity.csv").open("w", newline="") as stream:
        fields_guard = ["density", "seed", "condition", "control_windows", "safety_guard_reason_events", "directional_ul", "directional_dl", "rollback_trigger", "rollback_hold"]
        writer = csv.DictWriter(stream, fieldnames=fields_guard)
        writer.writeheader()
        writer.writerows(guard_rows)
    (root / "paired_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (root / "results.md").write_text(render_markdown(summary, pairs, guard_rows))
    print(f"Paired results: {csv_path}", flush=True)
    print(f"Summary: {root / 'results.md'}", flush=True)
    print(f"Verified pairs: {len(pairs)}/21", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="campaign output directory")
    args = parser.parse_args()
    root = args.input.resolve()
    pairs, guard_rows, missing = make_pairs(root)
    write_outputs(root, pairs, guard_rows, missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
