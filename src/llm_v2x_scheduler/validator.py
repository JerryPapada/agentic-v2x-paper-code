"""Strict validation for v1 LLM-rApp scheduler policies."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from pathlib import Path
from typing import Any

from .parser import dump_policy, parse_policy_text
from .repair import repair_policy
from .schema import ADAPTIVE_DIRECTIONAL_METRICS, MAX_WEIGHT, MIN_WEIGHT, SUPPORTED_ACTIONS, SUPPORTED_METRICS, SUPPORTED_OPERATORS, SUPPORTED_SOURCES, ValidationResult, fallback_policy_name
from .services import CRITICAL_SERVICES, SERVICE_AWARENESS, SERVICE_HDMAP, SERVICE_NAMES, SERVICE_SENSOR, SERVICE_TOD, canonical_service, canonicalize_service_dict


FALLBACK_POLICY_REGISTRY = {
    "PF",
    "static_expert",
    "balanced_expert",
    "heuristic_xapp",
}


def validate_policy_text(text: str, attempt_repair: bool = True, *, adaptive: bool = False) -> ValidationResult:
    try:
        policy = parse_policy_text(text)
    except Exception as exc:
        return ValidationResult(valid=False, rejected=True, errors=[str(exc)])
    return validate_policy(policy, attempt_repair=attempt_repair, adaptive=adaptive)


def validate_policy(policy: dict[str, Any], attempt_repair: bool = True, *, adaptive: bool = False) -> ValidationResult:
    normalized, norm_errors, norm_warnings = _normalize(policy)
    errors = _validate(normalized, adaptive=adaptive)
    warnings = norm_warnings[:]
    if norm_errors:
        errors.extend(norm_errors)

    if not errors:
        return ValidationResult(valid=True, policy=normalized, warnings=warnings)

    if not attempt_repair:
        return ValidationResult(valid=False, rejected=True, errors=errors, warnings=warnings, policy=normalized)

    repaired, repair_errors, repair_warnings = repair_policy(policy, adaptive=adaptive)
    repaired_norm, repaired_norm_errors, repaired_norm_warnings = _normalize(repaired)
    repaired_errors = _validate(repaired_norm, adaptive=adaptive)
    all_warnings = warnings + repair_warnings + repaired_norm_warnings
    all_errors = repair_errors + repaired_norm_errors + repaired_errors
    if all_errors:
        fallback = _fallback_executed_policy(policy)
        return ValidationResult(
            valid=False,
            repaired=bool(repair_warnings),
            rejected=True,
            errors=errors + all_errors,
            warnings=all_warnings + ["unsafe repair rejected; fallback policy selected"],
            policy=fallback,
        )

    repaired_norm["metadata"]["source"] = "repaired"
    return ValidationResult(
        valid=True,
        repaired=True,
        rejected=False,
        errors=[],
        warnings=all_warnings,
        policy=repaired_norm,
    )


def save_validation_report(path: str | Path, result: ValidationResult) -> None:
    dump_policy(path, result.as_dict())


def _normalize(policy: dict[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    out = deepcopy(policy)
    errors: list[str] = []
    warnings: list[str] = []

    for key in ("baseline_weights", "bounds"):
        if key in out and isinstance(out[key], dict):
            try:
                out[key] = canonicalize_service_dict(out[key])
            except ValueError as exc:
                errors.append(str(exc))
    if isinstance(out.get("safety_constraints"), dict):
        crit = []
        critical_services = out["safety_constraints"].get("critical_services")
        if not isinstance(critical_services, list):
            errors.append("critical_services must be a list")
            critical_services = []
        for service in critical_services:
            try:
                crit.append(canonical_service(service))
            except ValueError as exc:
                errors.append(str(exc))
        out["safety_constraints"]["critical_services"] = crit
    if isinstance(out.get("xapp_rules"), list):
        for rule in out["xapp_rules"]:
            if isinstance(rule, dict) and isinstance(rule.get("action"), dict) and "service" in rule["action"]:
                try:
                    rule["action"]["service"] = canonical_service(rule["action"]["service"])
                except ValueError as exc:
                    errors.append(str(exc))
    return out, errors, warnings


def _validate(policy: dict[str, Any], *, adaptive: bool) -> list[str]:
    errors: list[str] = []
    for key in ("policy_id", "description", "scenario_id", "objective_id", "baseline_weights", "bounds", "safety_constraints", "xapp_rules", "fallback_policy", "metadata"):
        if key not in policy:
            errors.append(f"missing required field {key}")

    weights = policy.get("baseline_weights")
    if not isinstance(weights, dict):
        errors.append("baseline_weights must be a mapping")
    else:
        _validate_weights(weights, "baseline_weights", errors)

    bounds = policy.get("bounds")
    if not isinstance(bounds, dict):
        errors.append("bounds must be a mapping")
    else:
        _validate_bounds(bounds, errors)
        if isinstance(weights, dict):
            for service in SERVICE_NAMES:
                if _is_int(weights.get(service)) and _valid_bound(bounds.get(service)):
                    low, high = bounds[service]
                    if int(weights[service]) < int(low) or int(weights[service]) > int(high):
                        errors.append(f"baseline weight for {service} lies outside bounds")

    safety = policy.get("safety_constraints")
    if not isinstance(safety, dict):
        errors.append("safety_constraints must be a mapping")
    else:
        crit = safety.get("critical_services")
        if not isinstance(crit, list):
            errors.append("critical_services must be a list")
        else:
            for service in CRITICAL_SERVICES:
                if service not in crit:
                    errors.append(f"{service} must be listed as critical")
        if not _is_int(safety.get("min_critical_weight")) or int(safety.get("min_critical_weight", 0)) < MIN_WEIGHT:
            errors.append("min_critical_weight must be an integer in 1..10")
        if adaptive:
            _validate_adaptive_floor(safety.get("min_critical_weight"), bounds, weights, errors)
        try:
            degradation = float(safety.get("max_background_degradation"))
            if adaptive and not 0.0 <= degradation <= 1.0:
                errors.append("max_background_degradation must be in 0..1")
        except (TypeError, ValueError):
            errors.append("max_background_degradation must be numeric")
        if safety.get("forbid_zero_weight") is not True:
            errors.append("forbid_zero_weight must be true")

    if isinstance(bounds, dict) and all(_valid_bound(bounds.get(service)) for service in SERVICE_NAMES):
        tod_max = int(bounds[SERVICE_TOD][1])
        if tod_max < int(bounds.get(SERVICE_HDMAP, [1, 1])[1]) or tod_max < int(bounds.get(SERVICE_SENSOR, [1, 1])[1]):
            errors.append("ToD must not have lower maximum priority than HDMap or Sensor")

    if isinstance(weights, dict) and all(_is_int(weights.get(service)) for service in SERVICE_NAMES):
        if int(weights.get(SERVICE_TOD, 0)) < int(weights.get(SERVICE_AWARENESS, 0)):
            errors.append("ToD baseline weight must not be below Awareness")
        if int(weights.get(SERVICE_TOD, 0)) < int(weights.get(SERVICE_SENSOR, 0)):
            errors.append("ToD baseline weight must not be below Sensor")
        if int(weights.get(SERVICE_AWARENESS, 0)) < int(weights.get(SERVICE_SENSOR, 0)):
            errors.append("Awareness baseline weight must not be below Sensor")

    _validate_rules(policy.get("xapp_rules"), bounds, errors, adaptive=adaptive)

    fallback = policy.get("fallback_policy")
    if not isinstance(fallback, str) or fallback not in FALLBACK_POLICY_REGISTRY:
        errors.append("fallback_policy must exist in fallback registry")

    metadata = policy.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata must be a mapping")
    elif metadata.get("source") not in SUPPORTED_SOURCES:
        errors.append("metadata.source is unsupported")

    return errors


def _valid_bound(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_is_int(side) for side in value)


def _validate_adaptive_floor(floor: Any, bounds: Any, weights: Any, errors: list[str]) -> None:
    if not _is_int(floor) or not 5 <= floor <= MAX_WEIGHT:
        errors.append("Adaptive LLM min_critical_weight must be an integer in 5..10")
        return
    for service in CRITICAL_SERVICES:
        if isinstance(bounds, dict) and _valid_bound(bounds.get(service)) and bounds[service][0] < floor:
            errors.append(f"{service} lower bound is below the critical floor {floor}")
        if isinstance(weights, dict) and _is_int(weights.get(service)) and weights[service] < floor:
            errors.append(f"{service} baseline is below the critical floor {floor}")


def _validate_weights(weights: dict[str, Any], context: str, errors: list[str]) -> None:
    for service in SERVICE_NAMES:
        if service not in weights:
            errors.append(f"{context} missing service {service}")
            continue
        if not _is_int(weights[service]):
            errors.append(f"{context}.{service} must be an integer")
            continue
        value = int(weights[service])
        if value < MIN_WEIGHT or value > MAX_WEIGHT:
            errors.append(f"{context}.{service} must be in 1..10")
    for service in weights:
        if service not in SERVICE_NAMES:
            errors.append(f"{context} has unknown service {service}")


def _validate_bounds(bounds: dict[str, Any], errors: list[str]) -> None:
    for service in SERVICE_NAMES:
        if service not in bounds:
            errors.append(f"bounds missing service {service}")
            continue
        value = bounds[service]
        if not isinstance(value, list) or len(value) != 2:
            errors.append(f"bounds.{service} must be [min,max]")
            continue
        low, high = value
        if not _is_int(low) or not _is_int(high):
            errors.append(f"bounds.{service} must contain integers")
            continue
        if int(low) < MIN_WEIGHT or int(low) > MAX_WEIGHT or int(high) < MIN_WEIGHT or int(high) > MAX_WEIGHT:
            errors.append(f"bounds.{service} values must be in 1..10")
        if int(low) > int(high):
            errors.append(f"bounds.{service} lower bound cannot exceed upper bound")
    for service in bounds:
        if service not in SERVICE_NAMES:
            errors.append(f"bounds has unknown service {service}")


def _validate_rules(rules: Any, bounds: Any, errors: list[str], *, adaptive: bool) -> None:
    if not isinstance(rules, list):
        errors.append("xapp_rules must be a list")
        return
    for idx, rule in enumerate(rules):
        if not isinstance(rule, dict):
            errors.append(f"xapp_rules[{idx}] must be a mapping")
            continue
        condition = rule.get("condition")
        action = rule.get("action")
        if not isinstance(condition, dict) or not isinstance(action, dict):
            errors.append(f"xapp_rules[{idx}] must have condition and action mappings")
            continue
        metric = condition.get("metric")
        directional = adaptive and metric in ADAPTIVE_DIRECTIONAL_METRICS
        if metric not in SUPPORTED_METRICS and not directional:
            errors.append(f"xapp_rules[{idx}] uses unsupported telemetry metric")
        if condition.get("operator") not in SUPPORTED_OPERATORS:
            errors.append(f"xapp_rules[{idx}] uses unsupported operator")
        try:
            condition_value = float(condition.get("value"))
        except Exception:
            errors.append(f"xapp_rules[{idx}] condition value must be numeric")
            condition_value = None
        if adaptive and condition_value is not None and (not isfinite(condition_value) or condition_value < 0):
            errors.append(f"xapp_rules[{idx}] condition value must be finite and nonnegative")
        ratio_metric = metric in (
            "tod_dc_prr",
            "tod_violation_rate",
            "awareness_dc_prr",
            "awareness_violation_rate",
            "hdmap_throughput_degradation",
            "sensor_throughput_degradation",
            "congestion_ratio",
        ) or (directional and not metric.endswith("_ms"))
        if condition_value is not None and ratio_metric and not (0.0 <= condition_value <= 1.0):
            errors.append(f"xapp_rules[{idx}] ratio metric threshold must be in 0..1")
        service = action.get("service")
        if service not in SERVICE_NAMES:
            errors.append(f"xapp_rules[{idx}] action service is unsupported")
        if action.get("type") not in SUPPORTED_ACTIONS:
            errors.append(f"xapp_rules[{idx}] action type is unsupported")
        if not _is_int(action.get("value")):
            errors.append(f"xapp_rules[{idx}] action value must be integer")
        elif isinstance(bounds, dict) and service in SERVICE_NAMES and _valid_bound(bounds.get(service)):
            value = int(action.get("value"))
            low, high = bounds.get(service, [MIN_WEIGHT, MAX_WEIGHT])
            if action.get("type") == "set_weight" and not (int(low) <= value <= int(high)):
                errors.append(f"xapp_rules[{idx}] set_weight exceeds validated bounds")
            if action.get("type") in ("increase_weight", "decrease_weight") and value < 0:
                errors.append(f"xapp_rules[{idx}] adjustment must be non-negative")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _fallback_executed_policy(policy: dict[str, Any]) -> dict[str, Any]:
    from .baselines import static_expert_policy

    fallback = static_expert_policy(
        scenario_id=str(policy.get("scenario_id", "unknown_scenario")),
        objective_id=str(policy.get("objective_id", "default_objective")),
    )
    fallback["policy_id"] = f"{policy.get('policy_id', 'rejected')}_fallback_{fallback_policy_name()}"
    return fallback
