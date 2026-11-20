#!/usr/bin/env python3
"""Run the matched 21-case LLM-adaptive refresh-disabled ablation.

The first case (20 vehicles, seed 1) is verified before the matrix proceeds.
Successful cases are retained and skipped on resume; each failed attempt keeps
its per-run logs and is retried into a fresh attempt directory on the next run.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from adaptive_study_runner import (
    PROJECT_ROOT, REFRESH_DISABLED_LABEL, Trial, read_json, simulate,
    verify_refresh_disabled, write_json,
)

DENSITIES = (20, 25, 30)
SEEDS = tuple(range(1, 8))
SCRIPT_DIR = Path(__file__).resolve().parent
ANALYZER = SCRIPT_DIR / "analyze_refresh_disabled_ablation.py"


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def acquire_lock(root: Path) -> Path:
    path = root / "campaign.lock"
    if path.exists():
        try:
            old_pid = int(path.read_text().strip())
            os.kill(old_pid, 0)
        except ProcessLookupError:
            path.unlink()
        except (ValueError, PermissionError):
            raise RuntimeError(f"Cannot determine whether campaign lock is stale: {path}")
        else:
            raise RuntimeError(f"Another refresh-disabled campaign is running with PID {old_pid}")
    path.write_text(f"{os.getpid()}\n")
    return path


def write_verified_manifest(root: Path, records: dict[str, dict]) -> None:
    path = root / "verified_run_manifest.csv"
    fields = [
        "density", "seed", "condition", "relative_output", "success",
        "binary_sha256", "trace_sha256", "ns3_rng_seed", "ns3_rng_run",
        "control_windows", "llm_request_submissions", "llm_generated_revisions",
        "scheduler_weight_windows_issued", "simulator_weight_windows_matched",
        "distinct_ul_weight_vectors", "distinct_dl_weight_vectors",
        "safety_interventions", "directional_ul", "directional_dl",
        "rollback_trigger", "rollback_hold",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in sorted(records, key=lambda item: tuple(map(int, item.split("/")))):
            item = records[key]
            writer.writerow({field: item.get(field, "") for field in fields})


def load_state(root: Path) -> dict:
    path = root / "campaign_manifest.json"
    if path.exists():
        state = read_json(path)
        expected = {
            "condition": REFRESH_DISABLED_LABEL,
            "densities": list(DENSITIES),
            "seeds": list(SEEDS),
            "simulation_seconds": 100,
        }
        pilot = state.get("pilot", {})
        if (any(state.get(key) != value for key, value in expected.items())
                or (pilot.get("density"), pilot.get("seed")) != (20, 1)):
            raise RuntimeError(f"Existing campaign manifest does not match this matrix: {path}")
        return state
    return {
        "condition": REFRESH_DISABLED_LABEL,
        "design": "paired by vehicle count and run index against the 21-run original Adaptive LLM cohort",
        "densities": list(DENSITIES),
        "seeds": list(SEEDS),
        "simulation_seconds": 100,
        "pilot": {"density": 20, "seed": 1, "status": "pending"},
        "completed": {},
        "failed_attempts": [],
        "status": "running",
    }


def run_one(root: Path, state: dict, density: int, seed: int, timeout_s: float) -> bool:
    trial = Trial("refresh_disabled", density, seed, 100, "refresh_disabled")
    key = f"{density}/{seed}"
    try:
        output = simulate(root, trial, timeout_s=timeout_s)
        audit = verify_refresh_disabled(output)
        summary = read_json(output / "run_summary.json")
        manifest = read_json(output / "manifest.json")
        record = {
            "density": density,
            "seed": seed,
            "condition": REFRESH_DISABLED_LABEL,
            "relative_output": str(output.relative_to(root)),
            "success": True,
            "binary_sha256": manifest["binary_sha256"],
            "trace_sha256": manifest["trace_sha256"],
            "ns3_rng_seed": manifest["ns3_rng_seed"],
            "ns3_rng_run": manifest["ns3_rng_run"],
            **{name: audit[name] for name in (
                "control_windows", "llm_request_submissions", "llm_generated_revisions",
                "scheduler_weight_windows_issued", "simulator_weight_windows_matched",
                "distinct_ul_weight_vectors", "distinct_dl_weight_vectors",
                "safety_interventions", "directional_ul", "directional_dl",
                "rollback_trigger", "rollback_hold",
            )},
            "elapsed_s": summary["elapsed_s"],
        }
        state.setdefault("completed", {})[key] = record
        state.setdefault("failed_attempts", [])
        if key in state["failed_attempts"]:
            state["failed_attempts"] = [item for item in state["failed_attempts"] if item != key]
        if (density, seed) == (20, 1):
            state["pilot"] = {"density": density, "seed": seed, "status": "verified", "output": record["relative_output"]}
        atomic_json(root / "campaign_manifest.json", state)
        write_verified_manifest(root, state["completed"])
        print(
            f"VERIFIED {key}: {audit['control_windows']} windows; "
            f"LLM requests/revisions={audit['llm_request_submissions']}/{audit['llm_generated_revisions']}; "
            f"simulator weights matched={audit['simulator_weight_windows_matched']}",
            flush=True,
        )
        return True
    except Exception as exc:
        attempt = {"density": density, "seed": seed, "error": str(exc)}
        state.setdefault("failed_attempts", []).append(key)
        state.setdefault("failure_details", []).append(attempt)
        if (density, seed) == (20, 1):
            state["pilot"] = {"density": density, "seed": seed, "status": "failed", "error": str(exc)}
        atomic_json(root / "campaign_manifest.json", state)
        print(f"FAILED {key}: {exc}", file=sys.stderr, flush=True)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "results" / "refresh_disabled_ablation",
        help="separate campaign directory (defaults under results/refresh_disabled_ablation)",
    )
    parser.add_argument("--timeout-s", type=float, default=21600)
    parser.add_argument("--pilot-only", action="store_true", help="run and verify only 20 vehicles, seed 1")
    args = parser.parse_args()
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    root = args.output.resolve()
    if root == (PROJECT_ROOT / "results").resolve() or root.is_relative_to((PROJECT_ROOT / "results/raw").resolve()):
        parser.error("choose a dedicated study directory outside results/raw")
    root.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(root)
    try:
        state = load_state(root)
        atomic_json(root / "campaign_manifest.json", state)
        schedule = list(itertools.product(DENSITIES, SEEDS))
        if args.pilot_only:
            schedule = [(20, 1)]
        elif state.get("pilot", {}).get("status") != "verified":
            schedule = [(20, 1)] + [pair for pair in schedule if pair != (20, 1)]

        failures = []
        for index, (density, seed) in enumerate(schedule, 1):
            key = f"{density}/{seed}"
            if key in state.get("completed", {}):
                output = root / "refresh_disabled" / state["completed"][key]["relative_output"].split("/", 1)[-1]
                # Re-audit resumed records rather than trusting the campaign CSV.
                verify_refresh_disabled(output)
                print(f"SKIP verified {key}", flush=True)
                continue
            print(f"[{index}/{len(schedule)}] {REFRESH_DISABLED_LABEL}: {density} vehicles, seed {seed}", flush=True)
            if not run_one(root, state, density, seed, args.timeout_s):
                failures.append(key)
                if args.pilot_only or (density, seed) == (20, 1):
                    break
            if args.pilot_only:
                break

        if args.pilot_only:
            state["status"] = "pilot_verified" if state.get("pilot", {}).get("status") == "verified" else "pilot_failed"
        else:
            state["status"] = "complete" if len(state.get("completed", {})) == 21 else "partial"
        state["last_run_failures"] = failures
        atomic_json(root / "campaign_manifest.json", state)
        if state.get("completed"):
            result = subprocess.run([sys.executable, str(ANALYZER), "--input", str(root)], cwd=PROJECT_ROOT)
            if result.returncode != 0:
                print("Analysis did not complete; see campaign_manifest.json and campaign logs.", file=sys.stderr)
                return 1
        return 0 if not failures and state["status"] not in ("pilot_failed", "partial") else 1
    finally:
        try:
            if int(lock.read_text().strip()) == os.getpid():
                lock.unlink()
        except (FileNotFoundError, ValueError):
            pass


if __name__ == "__main__":
    raise SystemExit(main())
