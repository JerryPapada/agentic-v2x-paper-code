#!/usr/bin/env python3
"""Shared-memory xApp for the Adaptive LLM study and its within-method controls."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

from llm_v2x_xapp_agent import (
    MAGIC_STATE_VALID, UE_MIX, SharedMemoryInterface, service_assignment,
    telemetry_from_state, ue_weight_arrays,
)
from llm_v2x_scheduler.adaptive_study import (
    OBJECTIVES, PolicyRevisions, build_prompt, generate_proposal, identifiers, observe, policy_content,
)
from llm_v2x_scheduler.baselines import heuristic_xapp_policy
from llm_v2x_scheduler.utils import write_json
from llm_v2x_scheduler.validator import validate_policy
from llm_v2x_scheduler.xapp_controller import XAppController

AI_START_S = 2.0
CYCLE_S = 0.1


def delayed_proposal(policy: dict, delay: float) -> tuple[dict, dict]:
    started = time.time()
    time.sleep(delay)
    candidate = deepcopy(policy)
    candidate["policy_id"] = "timing_canary"
    candidate["baseline_weights"]["ToD"] = 8
    return {
        "parsed": True, "strict_pass": True, "accepted": True, "repaired": False,
        "wall_request_s": started, "wall_complete_s": time.time(), "latency_s": time.time() - started,
    }, candidate


def run_agent(args: argparse.Namespace) -> None:
    legacy = args.mode == "legacy"
    startup = heuristic_xapp_policy(args.scenario, args.objective)
    validation = validate_policy(startup, attempt_repair=False, adaptive=not legacy)
    if not validation.valid:
        raise RuntimeError(f"Invalid study startup policy: {validation.errors}")
    if args.mode == "refresh_disabled":
        # Keep an auditable copy of the exact heuristic policy used at startup.
        write_json(args.output / "initial_policy.json", validation.policy)
        print("CONDITION LLM-adaptive (refresh disabled): periodic LLM request submission is disabled", flush=True)
    revisions = PolicyRevisions(validation.policy)
    incoming = validation.policy if legacy else revisions.policy
    controller = XAppController(performance_shield=args.shield == "on")
    by_ue = service_assignment(args.density, UE_MIX[args.density])
    recorded = json.loads(args.proposals.read_text()) if args.proposals else []
    replay = iter(recorded)
    next_event = next(replay, None)
    events = []
    last_step = -1
    last_state_at = time.monotonic()
    period_steps = round(args.period_s / CYCLE_S)
    next_refresh = 1
    worker = None
    requested = None
    out = args.output
    shm = SharedMemoryInterface(timeout_s=60)
    shm.connect(expected_mode=4)

    def collect(step: int | None, sim_time: float | None) -> None:
        nonlocal incoming, worker
        report, policy = worker.result()
        requested.update(report)
        requested["missed_wall_period"] = report["wall_complete_s"] - requested["scheduled_wall_s"] > args.period_s
        if policy is not None and step is not None:
            if revisions.publish(policy):
                incoming = revisions.policy
                requested["status"] = "applied"
                requested["activation_step"] = step
                requested["activation_sim_time_s"] = sim_time
                requested["activation_wall_s"] = time.time()
                requested["revision"] = incoming["policy_id"]
                events.append({"step": step, "policy": policy, "content": revisions.content})
                write_json(out / "proposals.json", events)
            else:
                requested["status"] = "no_op"
        else:
            requested["status"] = "finished_after_run" if step is None else "rejected"
        write_json(out / "refreshes" / f"{requested['scheduled_step']:05d}.json", requested)
        worker = None

    try:
        with (out / "actions.jsonl").open("w", buffering=1) as log, ThreadPoolExecutor(max_workers=1) as executor:
            while True:
                state = shm.read_state()
                if state.nodeId != MAGIC_STATE_VALID or state.stepCount <= last_step or (state.stepCount == 0 and not legacy):
                    if (out / "run_config.json").is_file():
                        break
                    if time.monotonic() - last_state_at > 120:
                        raise TimeoutError("No new ns-3 xApp state for 120 seconds")
                    time.sleep(0.0002)
                    continue

                step = state.stepCount
                last_step = step
                last_state_at = time.monotonic()
                sim_time = AI_START_S + (step - 1) * CYCLE_S
                if worker is not None and worker.done():
                    collect(step, sim_time)

                if next_event is not None and next_event["step"] < step:
                    raise RuntimeError(f"Missed replay step {next_event['step']} (got {step})")
                legacy_action = None
                if next_event is not None and next_event["step"] == step:
                    proposal = next_event["policy"]
                    if legacy:
                        if proposal is not None:
                            incoming = proposal
                        legacy_action = next_event
                    elif revisions.publish(proposal):
                        incoming = revisions.policy
                        events.append(next_event)
                    next_event = next(replay, None)

                active = controller.state.active_policy or incoming
                weights = active["baseline_weights"]
                telemetry = (
                    telemetry_from_state(state, by_ue, incoming["baseline_weights"])
                    if legacy
                    else observe(state, by_ue, weights, args.sensor_ul_mbps)
                )
                decision = controller.decide(incoming, telemetry)
                if legacy_action is not None:
                    ul = legacy_action["archived_ul_weights"]
                    dl = legacy_action["archived_dl_weights"]
                else:
                    ul, dl = ue_weight_arrays(by_ue, decision.weights, decision.directional_weights)
                shm.write_action(step, 4, 0, 0, 0, ul_weights=ul, dl_weights=dl, publish_step_last=not legacy)
                now = time.time()
                active = controller.state.active_policy or incoming
                log.write(json.dumps({
                    "step": step, "sim_time_s": round(sim_time, 4), "wall_action_s": now,
                    "proposed_revision": incoming["policy_id"], "active_revision": decision.policy_id,
                    "active_source_policy_id": active.get("source_policy_id", active["policy_id"]),
                    "active_content": policy_content(active), "effective_ul": ul[:args.density],
                    "effective_dl": dl[:args.density], "telemetry": telemetry.observed_metrics,
                    "legacy_tod_ul_dc_prr_proxy": telemetry.services["ToD"].ul_dc_prr if legacy else None,
                    "rollback_trigger": "rollback_policy_performance_degradation" in decision.reasons,
                    "rollback_hold": "rollback_active_holding_previous_policy" in decision.reasons,
                    "directional_ul": "safety_directional_dc_prr_gap_ul" in decision.reasons,
                    "directional_dl": "safety_directional_dc_prr_gap_dl" in decision.reasons,
                    "safety_interventions": decision.safety_interventions,
                    "rule_adaptations": decision.adaptations,
                    "reasons": decision.reasons,
                }, allow_nan=False) + "\n")

                if args.mode in ("live", "canary") and step == next_refresh:
                    next_refresh += period_steps
                    entry = {"scheduled_step": step, "scheduled_sim_time_s": sim_time, "scheduled_wall_s": now}
                    directory = out / "refreshes" / f"{step:05d}"
                    if worker is not None:
                        entry["status"] = "busy_skip"
                    else:
                        entry["status"] = "scheduled"
                        requested = entry
                        if args.mode == "canary":
                            worker = executor.submit(delayed_proposal, active, args.delay_s)
                        else:
                            prompt = build_prompt(args.scenario, args.objective, telemetry.observed_metrics, {
                                "revision": decision.policy_id, "baseline_weights": active["baseline_weights"],
                                "source": active.get("source_policy_id", active["policy_id"]),
                            })
                            worker = executor.submit(
                                generate_proposal, prompt, args.seed * 10000 + step, directory,
                                identifiers(args.scenario, args.objective),
                            )
                    write_json(out / "refreshes" / f"{step:05d}.json", entry)

            if worker is not None:
                collect(None, None)

        if next_event is not None:
            raise RuntimeError(f"Replay ended before proposal step {next_event['step']}")
        if last_step < 1:
            raise RuntimeError("No actual ns-3 xApp control windows were observed")
        write_json(out / "proposals.json", events)
    finally:
        shm.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("live", "replay", "legacy", "startup", "canary", "refresh_disabled"), required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--density", type=int, required=True, choices=UE_MIX)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--objective", default="safety_background_balance", choices=OBJECTIVES)
    parser.add_argument("--sensor-ul-mbps", type=float, default=0.0)
    parser.add_argument("--period-s", type=float, default=10.0)
    parser.add_argument("--shield", choices=("on", "off"), default="on")
    parser.add_argument("--proposals", type=Path)
    parser.add_argument("--delay-s", type=float, default=12.0)
    args = parser.parse_args()
    if args.period_s < CYCLE_S:
        parser.error("--period-s must be at least one xApp cycle")
    if args.mode in ("legacy", "replay") and args.proposals is None:
        parser.error("legacy and replay modes require --proposals")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "refreshes").mkdir(exist_ok=True)
    return args


if __name__ == "__main__":
    run_agent(parse_args())
