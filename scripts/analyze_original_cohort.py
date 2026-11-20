"""Summarize the 21 Adaptive LLM original runs as a standalone first cohort.

This analysis deliberately reads only the new original-run outputs. It does not
load archived runs or estimate an effect against an archived baseline.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import statistics
from pathlib import Path

import _bootstrap  # noqa: F401

from adaptive_study_report import packets
from llm_v2x_scheduler.utils import PROJECT_ROOT


STUDY_ROOT = PROJECT_ROOT / "results" / "adaptive_revision_v2_adaptive_prompt_fix"
RUN_ROOT = STUDY_ROOT / "original"
OUTPUT_JSON = STUDY_ROOT / "original_cohort_analysis.json"
OUTPUT_MD = STUDY_ROOT / "original_cohort_analysis.md"
DETAILED_MD = STUDY_ROOT / "original_cohort_detailed_results.md"
OUTPUT_CSV = STUDY_ROOT / "original_cohort_runs.csv"
DENSITIES = (20, 25, 30)
SEEDS = tuple(range(1, 8))
T_975_DF6 = 2.4469118511449692
PAPER_TABLE_II_DEADLINES_MS = {
    "ToD_UL": 20,
    "ToD_DL": 20,
    "AIC_UL": 100,
    "AIC_DL": 100,
    "HDM_UL": 1000,
    "HDM_DL": 1000,
    "RTSA_UL": 500,
    "RTSA_DL": 500,
}


def exact_permutation_p(first: list[float], second: list[float]) -> float:
    """Two-sided exact randomization p-value for an unpaired mean difference."""
    values = first + second
    n_first = len(first)
    observed = abs(statistics.mean(first) - statistics.mean(second))
    extreme = 0
    total = 0
    for indexes in itertools.combinations(range(len(values)), n_first):
        selected = set(indexes)
        left = [value for index, value in enumerate(values) if index in selected]
        right = [value for index, value in enumerate(values) if index not in selected]
        extreme += abs(statistics.mean(left) - statistics.mean(right)) >= observed - 1e-12
        total += 1
    return extreme / total


def holm_adjust(contrasts: list[dict]) -> None:
    order = sorted(range(len(contrasts)), key=lambda index: contrasts[index]["permutation_p"])
    adjusted = 0.0
    count = len(order)
    for rank, index in enumerate(order):
        adjusted = max(adjusted, min(1.0, (count - rank) * contrasts[index]["permutation_p"]))
        contrasts[index]["holm_p_all_tests"] = adjusted


def hedges_g(first: list[float], second: list[float]) -> float | None:
    df = len(first) + len(second) - 2
    pooled_variance = (
        (len(first) - 1) * statistics.variance(first)
        + (len(second) - 1) * statistics.variance(second)
    ) / df
    if pooled_variance <= 0:
        return None
    correction = 1 - 3 / (4 * df - 1)
    return correction * (statistics.mean(first) - statistics.mean(second)) / math.sqrt(pooled_variance)


def group_summary(values: list[float]) -> dict:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    margin = T_975_DF6 * sd / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "sd": sd,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "ci95_mean": [mean - margin, mean + margin],
    }


def read_runs() -> tuple[list[dict], dict, dict]:
    rows = []
    validation = {key: 0 for key in ("attempts", "strict_pass", "repaired", "rejected")}
    runtime_deadlines = None
    for density in DENSITIES:
        for seed in SEEDS:
            path = RUN_ROOT / f"original_{density}veh_seed{seed}"
            summary_path = path / "run_summary.json"
            manifest_path = path / "manifest.json"
            config_path = path / "run_config.json"
            if not summary_path.is_file() or not manifest_path.is_file():
                raise FileNotFoundError(f"Missing completed run artifacts: {path}")
            summary = json.loads(summary_path.read_text())
            manifest = json.loads(manifest_path.read_text())
            config = json.loads(config_path.read_text())
            if not summary.get("success"):
                raise ValueError(f"Run is not successful: {path}")
            if (manifest.get("stage"), manifest.get("density"), manifest.get("seed"), manifest.get("seconds")) != (
                "original", density, seed, 100
            ):
                raise ValueError(f"Unexpected trial manifest: {path}")
            deadlines = config.get("deadlines_ms")
            if not deadlines:
                raise ValueError(f"Missing runtime deadline map: {config_path}")
            if runtime_deadlines is None:
                runtime_deadlines = deadlines
            elif deadlines != runtime_deadlines:
                raise ValueError(f"Runtime deadline map differs across runs: {config_path}")
            metrics = packets(path)
            dl_only = min(metrics["tod_dl_dc_prr"], metrics["awareness_dl_dc_prr"])
            four_dir = min(
                metrics["tod_ul_dc_prr"], metrics["tod_dl_dc_prr"],
                metrics["awareness_ul_dc_prr"], metrics["awareness_dl_dc_prr"],
            )
            mix = config["class_mix"]
            traffic = config["traffic_spec"]
            background_target = (
                mix["HDM"] * (traffic["HDM"]["dl_mbps_target"] + traffic["HDM"]["ul_mbps_target"])
                + mix["RTSA"] * (traffic["RTSA"]["dl_mbps_target"] + traffic["RTSA"]["ul_mbps_target"])
            )
            row = {
                "density": density,
                "seed": seed,
                "critical_dl_only_dc_prr_pct": 100 * dl_only,
                "critical_four_dir_dc_prr_pct": 100 * four_dir,
                "four_dir_limiting_direction": min((
                    ("ToD UL", metrics["tod_ul_dc_prr"]),
                    ("ToD DL", metrics["tod_dl_dc_prr"]),
                    ("Awareness UL", metrics["awareness_ul_dc_prr"]),
                    ("Awareness DL", metrics["awareness_dl_dc_prr"]),
                ), key=lambda pair: pair[1])[0],
                "background_mbps": metrics["background_mbps"],
                "background_target_mbps": background_target,
                "background_delivered_pct_of_target": 100 * metrics["background_mbps"] / background_target,
                "tod_ul_dc_prr_pct": 100 * metrics["tod_ul_dc_prr"],
                "tod_dl_dc_prr_pct": 100 * metrics["tod_dl_dc_prr"],
                "awareness_ul_dc_prr_pct": 100 * metrics["awareness_ul_dc_prr"],
                "awareness_dl_dc_prr_pct": 100 * metrics["awareness_dl_dc_prr"],
                "tod_ul_p95_ms": metrics["tod_ul_p95_ms"],
                "tod_dl_p95_ms": metrics["tod_dl_p95_ms"],
                "awareness_ul_p95_ms": metrics["awareness_ul_p95_ms"],
                "awareness_dl_p95_ms": metrics["awareness_dl_p95_ms"],
                "elapsed_min": summary["elapsed_s"] / 60,
                "refresh_attempts": summary["refresh_attempts"],
                "strict_pass": summary["strict_pass"],
                "repaired": summary["repaired"],
                "rejected": summary["rejected"],
            }
            rows.append(row)
            validation["attempts"] += summary["refresh_attempts"]
            validation["strict_pass"] += summary["strict_pass"]
            validation["repaired"] += summary["repaired"]
            validation["rejected"] += summary["rejected"]
    assert runtime_deadlines is not None
    return rows, validation, runtime_deadlines


def build_analysis(rows: list[dict], validation: dict) -> dict:
    metrics = (
        "critical_dl_only_dc_prr_pct",
        "critical_four_dir_dc_prr_pct",
        "background_delivered_pct_of_target",
    )
    summary_metrics = metrics + ("background_mbps",)
    metric_summaries = {}
    for metric in summary_metrics:
        metric_summaries[metric] = {
            str(density): group_summary([row[metric] for row in rows if row["density"] == density])
            for density in DENSITIES
        }

    contrasts = []
    for metric in metrics:
        for first_density, second_density in itertools.combinations(DENSITIES, 2):
            first = [row[metric] for row in rows if row["density"] == first_density]
            second = [row[metric] for row in rows if row["density"] == second_density]
            contrasts.append({
                "metric": metric,
                "first_density": first_density,
                "second_density": second_density,
                "mean_difference_first_minus_second": statistics.mean(first) - statistics.mean(second),
                "hedges_g": hedges_g(first, second),
                "permutation_p": exact_permutation_p(first, second),
            })
    holm_adjust(contrasts)

    runtime_deadlines = json.loads((RUN_ROOT / "original_20veh_seed1" / "run_config.json").read_text())["deadlines_ms"]

    return {
        "analysis_type": "exploratory standalone cohort analysis",
        "archived_runs_used": False,
        "prospectively_preregistered": False,
        "design": {
            "successful_runs": len(rows),
            "densities_vehicles": list(DENSITIES),
            "seeds_per_density": list(SEEDS),
            "simulation_seconds": 100,
            "group_comparisons": "unpaired by density; seed IDs are not treated as paired blocks",
        },
        "validation": validation,
        "outcomes": {
            "critical_dl_only_dc_prr_pct": "legacy aggregator definition: min(ToD DL, Awareness DL)",
            "critical_four_dir_dc_prr_pct": "adaptive study definition: min(ToD UL, ToD DL, Awareness UL, Awareness DL)",
            "background_mbps": "delivered throughput summed over HDMap and Sensor, UL and DL",
            "background_delivered_pct_of_target": "background_mbps divided by nominal HDMap plus Sensor offered rate in each run configuration",
        },
        "runtime_deadlines_ms": runtime_deadlines,
        "paper_table_ii_deadlines_ms": PAPER_TABLE_II_DEADLINES_MS,
        "runtime_deadlines_match_paper_table_ii": runtime_deadlines == PAPER_TABLE_II_DEADLINES_MS,
        "by_density": metric_summaries,
        "pairwise_density_contrasts": contrasts,
        "multiple_testing": "Holm adjustment across all nine exploratory pairwise contrasts for both KPI definitions and normalized background delivery",
        "limitations": [
            "Density contrasts are exploratory and are not causal estimates of changing density.",
            "No primary hypothesis or analysis plan was registered before these runs.",
            "This report summarizes the new cohort only and does not compare against archived outcomes.",
            "All retained deadline-constrained packet counts use the runtime deadline map, which differs from arXiv Table II.",
            "The stored run outputs do not contain packet-level delays needed to recompute deadline counts under arXiv Table II.",
        ],
    }


def render_markdown(analysis: dict) -> str:
    lines = [
        "# Adaptive LLM original cohort: exploratory analysis",
        "",
        "This treats the 21 successful 100-second runs as the first standalone result cohort. Archived runs are not used as a comparator.",
        "The analysis is exploratory and was not prospectively preregistered. Both the legacy DL-only critical KPI and the newer four-direction KPI are shown because their definitions differ.",
        "",
        "## Run and refresh accounting",
        "",
        f"- Runs: {analysis['design']['successful_runs']}/21 successful (7 seeds at each of 20, 25, and 30 vehicles).",
        f"- LLM refresh attempts: {analysis['validation']['attempts']}; strict passes: {analysis['validation']['strict_pass']}; repairs: {analysis['validation']['repaired']}; rejections: {analysis['validation']['rejected']}.",
        "",
        "## Outcomes by density",
        "",
        "Values are mean (SD), with a two-sided 95% t confidence interval for the mean; n=7 per density.",
        "",
        "| Vehicles | DL-only critical DC-PRR, % | Four-direction critical DC-PRR, % | Background, Mbps | Background delivered / target, % |",
        "|---:|---:|---:|---:|---:|",
    ]
    legacy = analysis["by_density"]["critical_dl_only_dc_prr_pct"]
    four_dir = analysis["by_density"]["critical_four_dir_dc_prr_pct"]
    background = analysis["by_density"]["background_mbps"]
    background_share = analysis["by_density"]["background_delivered_pct_of_target"]
    for density in DENSITIES:
        c = legacy[str(density)]
        f = four_dir[str(density)]
        b = background[str(density)]
        q = background_share[str(density)]
        lines.append(
            f"| {density} | {c['mean']:.2f} ({c['sd']:.2f}); CI [{c['ci95_mean'][0]:.2f}, {c['ci95_mean'][1]:.2f}] "
            f"| {f['mean']:.2f} ({f['sd']:.2f}); CI [{f['ci95_mean'][0]:.2f}, {f['ci95_mean'][1]:.2f}] "
            f"| {b['mean']:.2f} | {q['mean']:.2f} |"
        )
    lines.extend([
        "",
        "## Exploratory density contrasts",
        "",
        "Differences are first density minus second density. Exact two-sided permutation p-values use unpaired seed-level observations; Holm adjustment covers all nine contrasts. Hedges' g is the standardized mean difference.",
        "",
        "| Outcome | Contrast | Mean difference | Hedges' g | Exact p | Holm p |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in analysis["pairwise_density_contrasts"]:
        unit = "pp"
        label = {
            "critical_dl_only_dc_prr_pct": "Critical DC-PRR (DL-only)",
            "critical_four_dir_dc_prr_pct": "Critical DC-PRR (four directions)",
            "background_delivered_pct_of_target": "Background delivered / target",
        }[row["metric"]]
        lines.append(
            f"| {label} | {row['first_density']} vs {row['second_density']} | "
            f"{row['mean_difference_first_minus_second']:+.2f} {unit} | {row['hedges_g']:+.2f} | "
            f"{row['permutation_p']:.4f} | {row['holm_p_all_tests']:.4f} |"
        )
    lines.extend([
        "",
        "## KPI and deadline caveat",
        "",
        f"The run configs record `{analysis['runtime_deadlines_ms']}`. The paper's Table II states `{analysis['paper_table_ii_deadlines_ms']}`. These maps differ. The retained `class_report.csv` and `flow_metrics.csv` contain packet counts and latency summaries evaluated under the runtime map, not packet-level delay samples; exact DC-PRR under the Table II deadlines cannot be reconstructed from these outputs.",
        "",
        "The legacy aggregator defines critical DC-PRR as min(ToD DL, Awareness DL), whereas the adaptive report uses min across all four critical directions. The current report keeps both measures visible rather than selecting one as scientifically canonical.",
        "",
        "## Interpretation limits",
        "",
        "These density contrasts describe this cohort; density is a scenario condition, not a randomized treatment. They do not establish a causal density effect or a gain over an earlier controller. Resolve whether the code or Table II deadlines are intended before treating critical DC-PRR as a paper result. The archived replay mismatch remains a separate unresolved validity issue.",
        "",
        "Per-run inputs and outcomes are in [`original_cohort_runs.csv`](original_cohort_runs.csv); machine-readable summaries are in [`original_cohort_analysis.json`](original_cohort_analysis.json).",
    ])
    return "\n".join(lines) + "\n"


def render_detailed_markdown(rows: list[dict], analysis: dict) -> str:
    lines = [
        "# Adaptive LLM original runs: detailed seed results",
        "",
        "This is the first-cohort result set: 21 successful 100-second runs, seven seeds at each density. No archived outcomes are used as a comparator.",
        "",
        "## Measurement definitions and unresolved specification mismatch",
        "",
        "The run-time C++ class accumulator and each run's `run_config.json` record these deadlines: ToD UL 100 ms / DL 20 ms; Awareness (AIC) UL and DL 20 ms; HDMap UL 1000 ms / DL 500 ms; Sensor (RTSA) UL and DL 1000 ms. The run configuration is emitted by `scenario_lib.h`; its per-packet class-report accumulator applies the same map.",
        "",
        "The paper's arXiv Table II instead states: ToD UL/DL 20 ms; Awareness UL/DL 100 ms; HDMap UL/DL 1000 ms; Sensor UL/DL 500 ms. The [paper table](https://arxiv.org/html/2607.04290) and the executed run configuration therefore disagree. The prompts and telemetry descriptions match the executed code, not Table II.",
        "",
        "| Service | Direction | Executed deadline | arXiv Table II |",
        "|---|---|---:|---:|",
        "| ToD | UL | 100 ms | 20 ms |",
        "| ToD | DL | 20 ms | 20 ms |",
        "| Awareness / AIC | UL | 20 ms | 100 ms |",
        "| Awareness / AIC | DL | 20 ms | 100 ms |",
        "| HDMap / HDM | UL | 1000 ms | 1000 ms |",
        "| HDMap / HDM | DL | 500 ms | 1000 ms |",
        "| Sensor / RTSA | UL | 1000 ms | 500 ms |",
        "| Sensor / RTSA | DL | 1000 ms | 500 ms |",
        "",
        "There are two separate KPI differences. The legacy aggregation script selects `min(ToD DL, Awareness DL)`. The adaptive study report selects the minimum across ToD UL/DL and Awareness UL/DL. Both use deadline-constrained ratios from `class_report.csv`; this detailed report computes ratios from `deadline_met_pkts / tx_pkts` to avoid the four-decimal rounding in `deadline_met_ratio`.",
        "",
        "The stored artifacts do not include per-packet delay samples or histograms. `class_report.csv` and `flow_metrics.csv` retain aggregate deadline counts under the executed map plus latency summaries, which are insufficient to recalculate exact deadline PRR under Table II. The implementation's executed values are established; whether Table II is a documentation error or the intended contract is not established by these artifacts. If Table II is intended, rerun with that deadline map. If the code map is intended, correct Table II and regenerate paper aggregates. Until that decision, report both KPI definitions and do not call critical-PRR conclusions scientifically settled.",
        "",
        "## KPI and throughput summaries by density",
        "",
        "Means are over the seven seed-level runs. The first critical KPI matches the legacy aggregator's DL-only selection; the second matches the adaptive study's four-direction selection. The two use the executed deadline map above.",
        "",
        "| Vehicles | DL-only critical DC-PRR mean (95% CI) | Four-direction critical DC-PRR mean (95% CI) | Background delivered mean, Mbps | Background target, Mbps | Mean delivered / target |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    legacy = analysis["by_density"]["critical_dl_only_dc_prr_pct"]
    four = analysis["by_density"]["critical_four_dir_dc_prr_pct"]
    bg = analysis["by_density"]["background_mbps"]
    bg_pct = analysis["by_density"]["background_delivered_pct_of_target"]
    for density in DENSITIES:
        l = legacy[str(density)]; f = four[str(density)]; b = bg[str(density)]; q = bg_pct[str(density)]
        target = statistics.mean(r["background_target_mbps"] for r in rows if r["density"] == density)
        lines.append(
            f"| {density} | {l['mean']:.2f}% [{l['ci95_mean'][0]:.2f}, {l['ci95_mean'][1]:.2f}] "
            f"| {f['mean']:.2f}% [{f['ci95_mean'][0]:.2f}, {f['ci95_mean'][1]:.2f}] "
            f"| {b['mean']:.2f} | {target:.3f} | {q['mean']:.2f}% |"
        )
    lines.extend([
        "",
        "Aggregate background throughput rises with density because the configured background flow mix also grows (HDMap: 1/2/3 flows; RTSA: 13/15/17 flows at 20/25/30 vehicles). Its offered rate grows from 17.256 to 33.456 to 49.656 Mbps, so the raw Mbps values are not directly comparable as a service-efficiency KPI. The delivered/target fraction shows the load-normalized view.",
        "",
    ])
    for density in DENSITIES:
        group = [row for row in rows if row["density"] == density]
        lines.extend([
            f"## {density} vehicles",
            "",
            "DC-PRR values are percent of packets delivered by the deadline encoded in the run. The DL-only column is min(ToD DL, Awareness DL); the four-direction column is min(ToD UL, ToD DL, Awareness UL, Awareness DL). Background throughput is total delivered HDMap plus Sensor/RTSA UL and DL.",
            "",
            "| Seed | ToD UL % | ToD DL % | Awareness UL % | Awareness DL % | Critical DL-only % | Critical four-dir % | Four-dir limiter | Background target Mbps | Background delivered Mbps | Delivered / target % |",
            "|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
        ])
        for row in group:
            lines.append(
                f"| {row['seed']} | {row['tod_ul_dc_prr_pct']:.2f} | {row['tod_dl_dc_prr_pct']:.2f} "
                f"| {row['awareness_ul_dc_prr_pct']:.2f} | {row['awareness_dl_dc_prr_pct']:.2f} "
                f"| {row['critical_dl_only_dc_prr_pct']:.2f} | {row['critical_four_dir_dc_prr_pct']:.2f} "
                f"| {row['four_dir_limiting_direction']} | {row['background_target_mbps']:.3f} "
                f"| {row['background_mbps']:.3f} | {row['background_delivered_pct_of_target']:.2f} |"
            )
        lines.extend([
            "",
            "Directional p95 receive latency (ms; these are latency summaries, not deadline PRR):",
            "",
            "| Seed | ToD UL | ToD DL | Awareness UL | Awareness DL |",
            "|---:|---:|---:|---:|---:|",
        ])
        for row in group:
            lines.append(
                f"| {row['seed']} | {row['tod_ul_p95_ms']:.2f} | {row['tod_dl_p95_ms']:.2f} "
                f"| {row['awareness_ul_p95_ms']:.2f} | {row['awareness_dl_p95_ms']:.2f} |"
            )
        lines.append("")
    lines.extend([
        "## Exploratory density tests",
        "",
        "These are unpaired seed-level exact permutation tests; Holm correction spans all nine pairwise tests across the two critical KPIs and normalized background delivery. They are exploratory, were not preregistered, and do not establish a causal density effect.",
        "",
        "| Outcome | Contrast | Mean difference, first minus second | Hedges' g | Exact p | Holm p |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    label_unit = {
        "critical_dl_only_dc_prr_pct": ("DL-only critical DC-PRR", "pp"),
        "critical_four_dir_dc_prr_pct": ("Four-direction critical DC-PRR", "pp"),
        "background_delivered_pct_of_target": ("Background delivered / target", "pp"),
    }
    for result in analysis["pairwise_density_contrasts"]:
        label, unit = label_unit[result["metric"]]
        lines.append(
            f"| {label} | {result['first_density']} vs {result['second_density']} "
            f"| {result['mean_difference_first_minus_second']:+.2f} {unit} | {result['hedges_g']:+.2f} "
            f"| {result['permutation_p']:.4f} | {result['holm_p_all_tests']:.4f} |"
        )
    lines.extend([
        "",
        "## Source trail",
        "",
        "- Run-specific applied deadline map: each `original/original_<density>veh_seed<seed>/run_config.json`.",
        "- Per-packet deadline classification: [`scenario_lib.h`, deadline table](../../../oran_xapp/_common/scenario_lib.h#L687); run-config deadline serialization is in the same file near line 1410.",
        "- Shared deadline helper: [`metrics_dump.h`](../../../oran_xapp/_common/metrics_dump.h#L65).",
        "- Legacy KPI selection: [`aggregate_real_results.py`](../../scripts/aggregate_real_results.py#L203).",
        "- Four-direction KPI selection: [`adaptive_study_report.py`](../../scripts/adaptive_study_report.py#L17).",
        "- Per-run tables are also available as [`original_cohort_runs.csv`](original_cohort_runs.csv).",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    rows, validation, runtime_deadlines = read_runs()
    analysis = build_analysis(rows, validation)
    analysis["runtime_deadlines_ms"] = runtime_deadlines
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    OUTPUT_JSON.write_text(json.dumps(analysis, indent=2) + "\n")
    OUTPUT_MD.write_text(render_markdown(analysis))
    DETAILED_MD.write_text(render_detailed_markdown(rows, analysis))
    print(OUTPUT_MD)
    print(DETAILED_MD)


if __name__ == "__main__":
    main()
