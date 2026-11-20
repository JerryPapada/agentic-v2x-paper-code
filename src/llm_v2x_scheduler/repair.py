"""Safe rApp policy repair helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .schema import MAX_WEIGHT, MIN_WEIGHT, default_bounds, default_safety_constraints, fallback_policy_name
from .services import CRITICAL_SERVICES, SERVICE_AWARENESS, SERVICE_HDMAP, SERVICE_NAMES, SERVICE_SENSOR, SERVICE_TOD, canonical_service, canonicalize_service_dict


SAFE_BASELINE = {
    SERVICE_TOD: 10,
    SERVICE_AWARENESS: 6,
    SERVICE_HDMAP: 2,
    SERVICE_SENSOR: 1,
}


def repair_policy(raw: dict[str, Any], *, adaptive: bool = False) -> tuple[dict[str, Any], list[str], list[str]]:
    """Return a safely repaired policy, errors, warnings.

    Repairs are intentionally conservative. Ambiguous schema-level issues are
    returned as errors for the validator to reject.
    """

    policy = deepcopy(raw)
    errors: list[str] = []
    warnings: list[str] = []

    _ensure_identity(policy, warnings)
    _repair_weights(policy, errors, warnings)
    _repair_safety(policy, warnings)
    floor = policy["safety_constraints"]["min_critical_weight"] if adaptive else 5
    _repair_bounds(policy, errors, warnings, floor=floor)
    _repair_rules(policy, warnings, adaptive=adaptive)
    _repair_metadata(policy, warnings)

    if policy.get("fallback_policy") not in {"PF", "static_expert", "balanced_expert", "heuristic_xapp"}:
        policy["fallback_policy"] = fallback_policy_name()
        warnings.append("fallback_policy repaired to static_expert")

    return policy, errors, warnings


def clamp_weight(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(MIN_WEIGHT, min(MAX_WEIGHT, parsed))


def _ensure_identity(policy: dict[str, Any], warnings: list[str]) -> None:
    if not policy.get("policy_id"):
        policy["policy_id"] = "repaired_policy"
        warnings.append("missing policy_id repaired")
    if not policy.get("description"):
        policy["description"] = "Repaired rApp scheduler policy"
        warnings.append("missing description repaired")
    if not policy.get("scenario_id"):
        policy["scenario_id"] = "unknown_scenario"
        warnings.append("missing scenario_id repaired")
    if not policy.get("objective_id"):
        policy["objective_id"] = "default_objective"
        warnings.append("missing objective_id repaired")


def _repair_weights(policy: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    raw = policy.get("baseline_weights", {})
    if isinstance(raw, list):
        converted = {}
        for item in raw:
            if isinstance(item, dict) and ("service_name" in item or "service" in item):
                converted[item.get("service_name", item.get("service"))] = item.get("weight", item.get("value"))
        raw = converted
        warnings.append("baseline_weights list repaired to service mapping")
    if not isinstance(raw, dict):
        raw = {}
        warnings.append("baseline_weights was not a mapping; repaired from safe baseline")
    try:
        weights = canonicalize_service_dict(raw)
    except ValueError as exc:
        errors.append(str(exc))
        weights = {}
    repaired = {}
    for service in SERVICE_NAMES:
        default = SAFE_BASELINE[service]
        if service not in weights:
            warnings.append(f"missing baseline weight for {service}; repaired to {default}")
        repaired[service] = clamp_weight(weights.get(service, default), default)
    _enforce_critical_order(repaired, warnings)
    policy["baseline_weights"] = repaired


def _repair_bounds(policy: dict[str, Any], errors: list[str], warnings: list[str], *, floor: int) -> None:
    raw = policy.get("bounds", {})
    if not isinstance(raw, dict):
        raw = {}
        warnings.append("bounds was not a mapping; repaired to default bounds")
    try:
        bounds = canonicalize_service_dict(raw)
    except ValueError as exc:
        errors.append(str(exc))
        bounds = {}
    repaired = default_bounds()
    for service in SERVICE_NAMES:
        value = bounds.get(service, repaired[service])
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            warnings.append(f"invalid bounds for {service}; repaired to default")
            value = repaired[service]
        low = clamp_weight(value[0], MIN_WEIGHT)
        high = clamp_weight(value[1], MAX_WEIGHT)
        if service in CRITICAL_SERVICES and low < floor:
            low = floor
            warnings.append(f"critical lower bound for {service} raised to {floor}")
        if low > high:
            errors.append(f"lower bound exceeds upper bound for {service}")
            continue
        repaired[service] = [low, high]
    weights = policy.get("baseline_weights", {})
    for service in SERVICE_NAMES:
        low, high = repaired[service]
        weights[service] = max(low, min(high, int(weights[service])))
    _enforce_critical_order(weights, warnings)
    policy["bounds"] = repaired
    policy["baseline_weights"] = weights


def _repair_safety(policy: dict[str, Any], warnings: list[str]) -> None:
    raw = policy.get("safety_constraints", {})
    if not isinstance(raw, dict):
        raw = {}
    safety = default_safety_constraints()
    safety.update({k: v for k, v in raw.items() if k in safety})
    safety["min_critical_weight"] = max(5, clamp_weight(safety.get("min_critical_weight", 5), 5))
    try:
        safety["max_background_degradation"] = float(safety.get("max_background_degradation", 0.35))
    except Exception:
        safety["max_background_degradation"] = 0.35
        warnings.append("invalid max_background_degradation repaired to 0.35")
    safety["forbid_zero_weight"] = True
    crit = []
    services = safety.get("critical_services")
    for service in services if isinstance(services, list) else []:
        try:
            crit.append(canonical_service(service))
        except ValueError:
            warnings.append(f"removed unsupported critical service {service!r}")
    for service in CRITICAL_SERVICES:
        if service not in crit:
            crit.append(service)
            warnings.append(f"added missing critical service {service}")
    safety["critical_services"] = crit
    policy["safety_constraints"] = safety


def _repair_rules(policy: dict[str, Any], warnings: list[str], *, adaptive: bool) -> None:
    from .schema import ADAPTIVE_DIRECTIONAL_METRICS, SUPPORTED_ACTIONS, SUPPORTED_METRICS, SUPPORTED_OPERATORS

    rules = policy.get("xapp_rules", [])
    if not isinstance(rules, list):
        policy["xapp_rules"] = []
        warnings.append("xapp_rules was not a list; removed")
        return
    kept = []
    for rule in rules:
        if not isinstance(rule, dict):
            warnings.append("removed non-mapping xApp rule")
            continue
        condition = rule.get("condition", {})
        action = rule.get("action", {})
        if not isinstance(condition, dict) or not isinstance(action, dict):
            warnings.append(f"removed malformed xApp rule {rule.get('name', '<unnamed>')}")
            continue
        metric = condition.get("metric")
        operator = condition.get("operator")
        action_type = action.get("type")
        directional = adaptive and metric in ADAPTIVE_DIRECTIONAL_METRICS
        if (metric not in SUPPORTED_METRICS and not directional) or operator not in SUPPORTED_OPERATORS or action_type not in SUPPORTED_ACTIONS:
            warnings.append(f"removed unsupported xApp rule {rule.get('name', '<unnamed>')}")
            continue
        try:
            service = canonical_service(action.get("service"))
        except ValueError:
            warnings.append(f"removed xApp rule with unsupported service {action.get('service')!r}")
            continue
        try:
            condition_value = float(condition.get("value"))
            action_value = int(action.get("value"))
        except Exception:
            warnings.append(f"removed xApp rule with non-numeric value {rule.get('name', '<unnamed>')}")
            continue
        ratio_metric = metric in (
            "tod_dc_prr",
            "tod_violation_rate",
            "awareness_dc_prr",
            "awareness_violation_rate",
            "hdmap_throughput_degradation",
            "sensor_throughput_degradation",
            "congestion_ratio",
        ) or (directional and not metric.endswith("_ms"))
        if ratio_metric and 1.0 < condition_value <= 100.0:
            condition_value = condition_value / 100.0
            warnings.append(f"normalized percentage threshold in xApp rule {rule.get('name', '<unnamed>')}")
        kept.append({
            "name": str(rule.get("name") or f"rule_{len(kept) + 1}"),
            "condition": {"metric": metric, "operator": operator, "value": condition_value},
            "action": {"service": service, "type": action_type, "value": action_value},
        })
    policy["xapp_rules"] = kept


def _repair_metadata(policy: dict[str, Any], warnings: list[str]) -> None:
    raw = policy.get("metadata", {})
    if not isinstance(raw, dict):
        raw = {}
    source = raw.get("source", "repaired")
    if source not in ("llm_rapp", "manual", "baseline", "repaired"):
        source = "repaired"
        warnings.append("metadata source repaired")
    policy["metadata"] = {
        "source": source,
        "model_name": raw.get("model_name"),
        "generation_time": raw.get("generation_time"),
    }


def _enforce_critical_order(weights: dict[str, int], warnings: list[str]) -> None:
    sensor = weights[SERVICE_SENSOR]
    if weights[SERVICE_AWARENESS] < sensor:
        weights[SERVICE_AWARENESS] = sensor
        warnings.append("raised Awareness weight to be at least Sensor")
    if weights[SERVICE_TOD] < weights[SERVICE_AWARENESS]:
        weights[SERVICE_TOD] = weights[SERVICE_AWARENESS]
        warnings.append("raised ToD weight to be at least Awareness")
    if weights[SERVICE_TOD] < sensor:
        weights[SERVICE_TOD] = sensor
        warnings.append("raised ToD weight to be at least Sensor")
