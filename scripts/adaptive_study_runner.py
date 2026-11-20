"""Run and resume isolated ns-3 trials for the Adaptive LLM study."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import _bootstrap  # noqa: F401

from llm_v2x_scheduler.adaptive_study import MODEL, MODEL_OPTIONS, model_digest
from llm_v2x_scheduler.adaptive_study import policy_content
from llm_v2x_scheduler.baselines import heuristic_xapp_policy, static_expert_policy
from llm_v2x_scheduler.parser import load_policy_file
from llm_v2x_scheduler.utils import PROJECT_ROOT, read_json, write_json
from llm_v2x_scheduler.validator import validate_policy

NS3_ROOT = PROJECT_ROOT.parents[1]
AGENT = PROJECT_ROOT / "scripts" / "adaptive_study_agent.py"
RAW = PROJECT_ROOT / "results" / "raw"
ORIGINAL_COHORT = PROJECT_ROOT / "results" / "adaptive_revision_v2_adaptive_prompt_fix" / "original"
REFRESH_DISABLED_LABEL = "LLM-adaptive (refresh disabled)"


@dataclass(frozen=True)
class Trial:
    stage: str
    density: int
    seed: int
    seconds: int
    mode: str
    load: str = "original"
    sensor_ul_mbps: float = 0.0
    shield: str = "on"
    period_s: float = 10.0
    delay_s: float = 12.0

    @property
    def key(self) -> str:
        return f"{self.load}_{self.density}veh_seed{self.seed}"


def archived(density: int, seed: int) -> Path:
    return RAW / f"real_100s_llm_rapp_xapp_{density}veh_seed{seed}_qwen2.5_7b"


def jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_metadata() -> dict:
    repositories = {"ns3": NS3_ROOT, "nr": NS3_ROOT / "contrib/nr", "ai": NS3_ROOT / "contrib/ai", "adaptive": PROJECT_ROOT}
    commits = {}
    for name, path in repositories.items():
        commits[name] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    paths = [AGENT, Path(__file__), NS3_ROOT / "scratch/oran_xapp/_common/scenario_lib.h",
             NS3_ROOT / "scratch/oran_xapp/tools/shm_interface.py"]
    paths += sorted((PROJECT_ROOT / "src/llm_v2x_scheduler").glob("*.py"))
    paths += sorted((PROJECT_ROOT / "scripts").glob("*adaptive_study*.py"))
    paths += [PROJECT_ROOT / "scripts" / "run_refresh_disabled_ablation.py",
              PROJECT_ROOT / "scripts" / "analyze_refresh_disabled_ablation.py"]
    paths = [path for path in paths if path.is_file()]
    return {
        "commits": commits,
        "source_sha256": {str(path.relative_to(NS3_ROOT)): sha256(path) for path in sorted(set(paths))},
    }


def binary_for(density: int) -> Path:
    folder = NS3_ROOT / "build/scratch/llm_v2x_scheduler/ns3" / f"s{density}"
    binaries = list(folder.glob(f"*scenario_llm_v2x_s{density}-optimized"))
    if len(binaries) != 1:
        raise RuntimeError(f"Build the optimized Adaptive LLM ns-3 scenario for {density} vehicles")
    return binaries[0]


def offline_archived_actions(run: Path) -> list[dict]:
    policies = sorted((run / "rapp_updates/executed").glob("*.yaml"))
    policy_by_version = {i: load_policy_file(path) for i, path in enumerate(policies, 1)}
    actions = []
    with (run / "llm_v2x_xapp_log.csv").open() as stream:
        for row in csv.DictReader(stream):
            version = int(row["policy_version"])
            actions.append({
                "step": int(row["step"]),
                "policy": policy_by_version.get(version),
                "archived_ul_weights": [int(weight) for weight in row["ul_weights_per_ue"].split(";")],
                "archived_dl_weights": [int(weight) for weight in row["dl_weights_per_ue"].split(";")],
            })
    return actions


def verify_refresh_disabled(out: Path, *, require_summary: bool = True) -> dict:
    """Verify the frozen-refresh condition and its simulator-side action trace."""
    manifest_path = out / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    density = int(manifest["density"])
    seed = int(manifest["seed"])
    if manifest.get("mode") != "refresh_disabled" or manifest.get("condition") != REFRESH_DISABLED_LABEL:
        raise ValueError(f"Run is not the requested refresh-disabled condition: {manifest_path}")

    summary_path = out / "run_summary.json"
    if require_summary:
        if not summary_path.is_file() or not read_json(summary_path).get("success"):
            raise ValueError(f"Run is not marked successful: {summary_path}")

    scenario = f"adaptive_v2_original_{density}veh_seed{seed}"
    validation = validate_policy(
        heuristic_xapp_policy(scenario), attempt_repair=False, adaptive=True,
    )
    if not validation.valid or validation.policy is None:
        raise ValueError(f"Expected startup heuristic is invalid: {validation.errors}")
    initial_path = out / "initial_policy.json"
    if not initial_path.is_file() or read_json(initial_path) != validation.policy:
        raise ValueError(f"Startup policy differs from the original heuristic: {initial_path}")
    startup_content = policy_content(validation.policy)
    fallback = static_expert_policy(scenario)
    fallback_content = policy_content(fallback)

    actions = jsonl(out / "actions.jsonl")
    if not actions:
        raise ValueError(f"No xApp control actions were recorded: {out}")
    if actions[-1]["step"] < 979:
        raise ValueError(f"Control loop ended early at step {actions[-1]['step']}: {out}")
    previous_revision = "revision_000"
    fallback_transitions = 0
    for row in actions:
        revision = row.get("active_revision")
        expected_states = {
            "revision_000": ("heuristic_xapp", startup_content),
            "static_expert": ("static_expert", fallback_content),
        }
        expected_state = expected_states.get(revision)
        if expected_state is None or (row.get("active_source_policy_id"), row.get("active_content")) != expected_state:
            raise ValueError(f"An unexpected or LLM-generated policy revision became active: {out}")
        if revision != previous_revision:
            if not (previous_revision == "revision_000" and revision == "static_expert" and row.get("rollback_trigger")):
                raise ValueError(f"Policy changed without its normal rollback trigger at step {row['step']}: {out}")
            fallback_transitions += 1
        previous_revision = revision
    if any(len(row.get("effective_ul", [])) != density or len(row.get("effective_dl", [])) != density
           for row in actions):
        raise ValueError(f"A control window is missing per-UE scheduler weights: {out}")

    refresh_files = sorted((out / "refreshes").rglob("*.json")) if (out / "refreshes").exists() else []
    if refresh_files:
        raise ValueError(f"Refresh requests were submitted in refresh-disabled run: {refresh_files[0]}")
    proposals_path = out / "proposals.json"
    proposals = read_json(proposals_path) if proposals_path.is_file() else None
    if proposals != []:
        raise ValueError(f"Unexpected policy proposal was recorded: {proposals_path}")

    step_log_path = out / "step_log.csv"
    if not step_log_path.is_file():
        raise FileNotFoundError(f"Missing simulator action log: {step_log_path}")
    with step_log_path.open(newline="") as stream:
        simulator_rows = {int(row["step"]): row for row in csv.DictReader(stream)}
    action_rows = {int(row["step"]): row for row in actions}
    if simulator_rows.keys() != action_rows.keys():
        raise ValueError(
            f"Controller and simulator logged different control windows: controller={len(action_rows)}, "
            f"simulator={len(simulator_rows)} in {out}"
        )
    for step, action in action_rows.items():
        received = simulator_rows[step]
        for direction, column in (("effective_ul", "action_ul_weight_per_ue"),
                                  ("effective_dl", "action_dl_weight_per_ue")):
            weights = [int(value) for value in received[column].split(";")]
            if weights != action[direction]:
                raise ValueError(f"Simulator did not receive {direction} weights from step {step}: {out}")

    config = read_json(out / "run_config.json")
    reference = ORIGINAL_COHORT / f"original_{density}veh_seed{seed}"
    reference_manifest = read_json(reference / "manifest.json")
    reference_config = read_json(reference / "run_config.json")
    if config != reference_config:
        differing = sorted(key for key in (set(config) | set(reference_config))
                           if config.get(key) != reference_config.get(key))
        raise ValueError(f"Run configuration differs from its matched LLM-adaptive reference ({differing}): {out}")
    for key in ("binary_sha256", "trace_sha256", "ns3_rng_run", "ns3_rng_seed"):
        if manifest.get(key) != reference_manifest.get(key):
            raise ValueError(f"Matched input {key} differs from the reference run: {out}")
    if manifest.get("commits") != reference_manifest.get("commits"):
        raise ValueError(f"Source repository revisions differ from the LLM-adaptive reference: {out}")
    for source in (
        "scratch/oran_xapp/_common/scenario_lib.h",
        "scratch/oran_xapp/tools/shm_interface.py",
        "scratch/llm_v2x_scheduler/scripts/run_adaptive_study.py",
        "scratch/llm_v2x_scheduler/src/llm_v2x_scheduler/xapp_controller.py",
        "scratch/llm_v2x_scheduler/src/llm_v2x_scheduler/baselines.py",
    ):
        if manifest.get("source_sha256", {}).get(source) != reference_manifest.get("source_sha256", {}).get(source):
            raise ValueError(f"Controller or simulator source differs from the LLM-adaptive reference ({source}): {out}")

    return {
        "passed": True,
        "condition": REFRESH_DISABLED_LABEL,
        "startup_policy_id": validation.policy["policy_id"],
        "startup_policy_content_sha256": startup_content,
        "llm_request_submissions": 0,
        "llm_generated_revisions": 0,
        "policy_revisions_observed": sorted({row["active_revision"] for row in actions}),
        "deterministic_fallback_transitions": fallback_transitions,
        "control_windows": len(actions),
        "scheduler_weight_windows_issued": len(actions),
        "simulator_weight_windows_matched": len(simulator_rows),
        "distinct_ul_weight_vectors": len({tuple(row["effective_ul"]) for row in actions}),
        "distinct_dl_weight_vectors": len({tuple(row["effective_dl"]) for row in actions}),
        "refresh_artifacts": len(refresh_files),
        "rollback_trigger": sum(bool(row["rollback_trigger"]) for row in actions),
        "rollback_hold": sum(bool(row["rollback_hold"]) for row in actions),
        "directional_ul": sum(bool(row["directional_ul"]) for row in actions),
        "directional_dl": sum(bool(row["directional_dl"]) for row in actions),
        "safety_interventions": sum(int(row.get("safety_interventions", 0)) for row in actions),
        "rule_adaptations": sum(int(row.get("rule_adaptations", 0)) for row in actions),
        "density": density,
        "seed": seed,
    }


def simulate(
    root: Path,
    trial: Trial,
    *,
    trace: Path | None = None,
    proposals: list[dict] | None = None,
    expected_digest: str | None = None,
    old_binary: Path | None = None,
    timeout_s: float = 21600,
) -> Path:
    out = root / trial.stage / trial.key
    binary = old_binary if trial.mode == "legacy" and old_binary is not None else binary_for(trial.density)
    actual_trace = trace or NS3_ROOT / f"scratch/mobility_1/ns2manhattanmobility{trial.density}.tcl"
    manifest = {
        "method": "llm_rapp_xapp", **asdict(trial),
        "condition": REFRESH_DISABLED_LABEL if trial.mode == "refresh_disabled" else trial.mode,
        "model": MODEL, "model_digest": expected_digest, "model_options": MODEL_OPTIONS,
        "ns3_rng_run": trial.seed, "ns3_rng_seed": 1,
        "binary": str(binary), "binary_sha256": sha256(binary),
        "trace": str(actual_trace), "trace_sha256": sha256(actual_trace),
        "proposals_sha256": hashlib.sha256(json.dumps(proposals, sort_keys=True).encode()).hexdigest(),
        **source_metadata(),
    }
    manifest_path = out / "manifest.json"
    summary_path = out / "run_summary.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        if summary_path.exists() and read_json(summary_path).get("success"):
            raise RuntimeError(f"Successful trial at {out} used different inputs. Use a new study directory.")
    if summary_path.exists() and read_json(summary_path).get("success"):
        return out
    if trial.mode == "live" and (not expected_digest or model_digest() != expected_digest):
        raise RuntimeError("Local Ollama model digest does not match the study manifest")
    if out.exists():
        out.rename(out.with_name(f"{out.name}.attempt_{uuid.uuid4().hex[:8]}"))
    out.mkdir(parents=True)
    write_json(manifest_path, manifest)
    scenario = f"adaptive_v2_{trial.load}_{trial.density}veh_seed{trial.seed}"
    prefix = f"adaptv2_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    env = {
        **os.environ, "JERRY_RESULTS_DIR": str(out), "JERRY_SHM_PREFIX": prefix,
        "NS_GLOBAL_VALUE": f"RngSeed=1;RngRun={trial.seed}", "JERRY_FAST_LOG": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    simulator_cmd = [str(binary), f"--simTime={trial.seconds}"]
    if trial.mode != "legacy":
        simulator_cmd += ["--adaptiveStudy=1", f"--studyMobility={actual_trace}",
                          f"--studySensorUlMbps={trial.sensor_ul_mbps}"]
    agent_cmd = [
        sys.executable, str(AGENT), "--output", str(out), "--mode", trial.mode, "--scenario", scenario,
        "--density", str(trial.density), "--seed", str(trial.seed), "--shield", trial.shield,
        "--period-s", str(trial.period_s), "--delay-s", str(trial.delay_s),
        "--sensor-ul-mbps", str(trial.sensor_ul_mbps),
    ]
    if proposals is not None:
        proposal_path = out / "input_proposals.json"
        write_json(proposal_path, proposals)
        agent_cmd += ["--proposals", str(proposal_path)]
    print(f"RUN {trial.stage}/{trial.key} ({trial.seconds}s, {trial.mode})", flush=True)
    started = time.monotonic()
    processes = []
    try:
        with (out / "ns3.log").open("w") as ns3_log, (out / "agent.log").open("w") as agent_log:
            processes.append(subprocess.Popen(simulator_cmd, cwd=NS3_ROOT, env=env, stdout=ns3_log, stderr=subprocess.STDOUT))
            processes.append(subprocess.Popen(agent_cmd, cwd=NS3_ROOT, env=env, stdout=agent_log, stderr=subprocess.STDOUT))
            while True:
                sim_rc, agent_rc = (process.poll() for process in processes)
                if sim_rc == 0 and agent_rc == 0:
                    break
                if (sim_rc not in (None, 0) or agent_rc not in (None, 0)
                        or (agent_rc == 0 and not (out / "run_config.json").exists())):
                    raise RuntimeError(f"Trial failed ns-3={sim_rc}, agent={agent_rc}. See {out}")
                if time.monotonic() - started > timeout_s:
                    raise TimeoutError(f"Trial exceeded {timeout_s} wall seconds. See {out}")
                time.sleep(0.25)
        actions = jsonl(out / "actions.jsonl")
        if not actions or actions[-1]["step"] < (trial.seconds - 2) * 10 - 1:
            raise RuntimeError(f"Incomplete control trace in {out}")
        if not (out / "class_report.csv").is_file():
            raise RuntimeError(f"Missing packet outcomes in {out}")
        verification = None
        if trial.mode == "refresh_disabled":
            verification = verify_refresh_disabled(out, require_summary=False)
            write_json(out / "run_verification.json", verification)
        updates = [read_json(path) for path in sorted((out / "refreshes").glob("*.json"))]
        if trial.mode == "live" and (model_digest() != expected_digest or not any("done_reason" in update for update in updates)):
            raise RuntimeError(f"Live run lacks verified model responses in {out}")
        write_json(summary_path, {
            "success": True, "elapsed_s": time.monotonic() - started,
            "condition": REFRESH_DISABLED_LABEL if trial.mode == "refresh_disabled" else trial.mode,
            "control_windows": len(actions), "refresh_slots": len(updates),
            "refresh_attempts": sum("latency_s" in update for update in updates),
            "wall_period_misses": sum(update.get("missed_wall_period", False) for update in updates),
            "busy_skips": sum(update["status"] == "busy_skip" for update in updates),
            "strict_pass": sum(update.get("strict_pass", False) for update in updates),
            "repaired": sum(update.get("repaired", False) for update in updates),
            "rejected": sum(update.get("accepted") is False for update in updates),
            "published_proposals": len(read_json(out / "proposals.json")),
            "policy_activations": sum(
                current["active_revision"] != previous["active_revision"]
                and current["active_revision"] == current["proposed_revision"]
                for previous, current in zip(actions, actions[1:])
            ),
            "no_ops": sum(update["status"] == "no_op" for update in updates),
            **{key: sum(row[key] for row in actions) for key in (
                "rollback_trigger", "rollback_hold", "directional_ul", "directional_dl",
                "safety_interventions", "rule_adaptations",
            )},
            **({
                "llm_request_submissions": verification["llm_request_submissions"],
                "llm_generated_revisions": verification["llm_generated_revisions"],
                "scheduler_weight_windows_issued": verification["scheduler_weight_windows_issued"],
                "simulator_weight_windows_matched": verification["simulator_weight_windows_matched"],
                "distinct_ul_weight_vectors": verification["distinct_ul_weight_vectors"],
                "distinct_dl_weight_vectors": verification["distinct_dl_weight_vectors"],
            } if verification else {}),
        })
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    print(f"OK {trial.stage}/{trial.key} ({time.monotonic() - started:.1f}s wall)", flush=True)
    return out
