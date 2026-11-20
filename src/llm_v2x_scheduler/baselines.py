"""Baseline rApp policies for v1 methods."""

from __future__ import annotations

from typing import Any

from .schema import default_bounds, default_safety_constraints
from .services import SERVICE_AWARENESS, SERVICE_HDMAP, SERVICE_SENSOR, SERVICE_TOD


CORE_METHODS = (
    "PF",
    "static_expert",
    "balanced_expert",
    "heuristic_xapp",
    "llm_static",
    "llm_rapp_xapp",
)


def pf_policy(scenario_id: str = "debug_30_seed1", objective_id: str = "safety_background_balance") -> dict[str, Any]:
    policy = _policy("PF", scenario_id, objective_id, {SERVICE_TOD: 1, SERVICE_AWARENESS: 1, SERVICE_HDMAP: 1, SERVICE_SENSOR: 1}, [])
    policy["bounds"] = {SERVICE_TOD: [1, 10], SERVICE_AWARENESS: [1, 10], SERVICE_HDMAP: [1, 10], SERVICE_SENSOR: [1, 10]}
    policy["safety_constraints"]["min_critical_weight"] = 1
    return policy


def static_expert_policy(scenario_id: str = "debug_30_seed1", objective_id: str = "safety_background_balance") -> dict[str, Any]:
    return _policy("static_expert", scenario_id, objective_id, {SERVICE_TOD: 10, SERVICE_AWARENESS: 6, SERVICE_HDMAP: 2, SERVICE_SENSOR: 1}, [])


def balanced_expert_policy(scenario_id: str = "debug_30_seed1", objective_id: str = "safety_background_balance") -> dict[str, Any]:
    return _policy("balanced_expert", scenario_id, objective_id, {SERVICE_TOD: 6, SERVICE_AWARENESS: 5, SERVICE_HDMAP: 3, SERVICE_SENSOR: 2}, [])


def heuristic_xapp_policy(scenario_id: str = "debug_30_seed1", objective_id: str = "safety_background_balance") -> dict[str, Any]:
    rules = [
        _rule("tod_latency_boost", "tod_p95_latency_ms", ">", 15.0, SERVICE_TOD, "increase_weight", 2),
        _rule("tod_prr_boost", "tod_dc_prr", "<", 0.98, SERVICE_TOD, "increase_weight", 1),
        _rule("tod_violation_boost", "tod_violation_rate", ">", 0.02, SERVICE_TOD, "increase_weight", 2),
        _rule("awareness_latency_boost", "awareness_p95_latency_ms", ">", 90.0, SERVICE_AWARENESS, "increase_weight", 1),
        _rule("awareness_prr_boost", "awareness_dc_prr", "<", 0.97, SERVICE_AWARENESS, "increase_weight", 1),
        _rule("awareness_violation_boost", "awareness_violation_rate", ">", 0.02, SERVICE_AWARENESS, "increase_weight", 1),
        _rule("congestion_background_trim", "congestion_ratio", ">", 0.80, SERVICE_SENSOR, "decrease_weight", 1),
        _rule("hdmap_recovery", "hdmap_throughput_degradation", ">", 0.30, SERVICE_HDMAP, "increase_weight", 1),
        _rule("sensor_recovery", "sensor_throughput_degradation", ">", 0.30, SERVICE_SENSOR, "increase_weight", 1),
    ]
    return _policy("heuristic_xapp", scenario_id, objective_id, {SERVICE_TOD: 6, SERVICE_AWARENESS: 5, SERVICE_HDMAP: 3, SERVICE_SENSOR: 2}, rules)


def llm_static_from_policy(policy: dict[str, Any]) -> dict[str, Any]:
    out = dict(policy)
    out["policy_id"] = f"{policy.get('policy_id', 'llm_policy')}_static"
    out["xapp_rules"] = []
    return out


def baseline_by_method(method: str, scenario_id: str, objective_id: str) -> dict[str, Any]:
    mapping = {
        "PF": pf_policy,
        "static_expert": static_expert_policy,
        "balanced_expert": balanced_expert_policy,
        "heuristic_xapp": heuristic_xapp_policy,
    }
    if method not in mapping:
        raise KeyError(f"no baseline policy for {method}")
    return mapping[method](scenario_id, objective_id)


def _policy(policy_id: str, scenario_id: str, objective_id: str, weights: dict[str, int], rules: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "policy_id": policy_id,
        "description": f"{policy_id} v1 rApp scheduler policy",
        "scenario_id": scenario_id,
        "objective_id": objective_id,
        "baseline_weights": weights,
        "bounds": default_bounds(),
        "safety_constraints": default_safety_constraints(),
        "xapp_rules": rules,
        "fallback_policy": "static_expert",
        "metadata": {"source": "baseline", "model_name": None, "generation_time": None},
    }


def _rule(name: str, metric: str, operator: str, value: float, service: str, action_type: str, action_value: int) -> dict[str, Any]:
    return {
        "name": name,
        "condition": {"metric": metric, "operator": operator, "value": value},
        "action": {"service": service, "type": action_type, "value": action_value},
    }
