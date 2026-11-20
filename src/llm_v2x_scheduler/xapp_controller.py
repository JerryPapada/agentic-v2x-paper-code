"""Deterministic xApp-like fast executor for validated rApp policies."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any

from .baselines import static_expert_policy
from .services import CRITICAL_SERVICES, SERVICE_AWARENESS, SERVICE_HDMAP, SERVICE_NAMES, SERVICE_SENSOR, SERVICE_TOD
from .telemetry import TelemetrySnapshot


@dataclass
class XAppState:
    previous_weights: dict[str, int] | None = None
    cooldown_remaining: int = 0
    stable_cycles: int = 0
    active_policy: dict[str, Any] | None = None
    active_policy_id: str | None = None
    previous_validated_policy: dict[str, Any] | None = None
    last_input_policy_id: str | None = None
    rolled_back_input_policy_id: str | None = None
    rollback_baseline: dict[str, float] | None = None
    degraded_windows: int = 0


@dataclass
class XAppDecision:
    weights: dict[str, int]
    adaptations: int
    safety_interventions: int
    fallback_used: bool
    execution_latency_ms: float
    reasons: list[str] = field(default_factory=list)
    directional_weights: dict[str, dict[str, int]] = field(default_factory=dict)
    policy_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights,
            "adaptations": self.adaptations,
            "safety_interventions": self.safety_interventions,
            "fallback_used": self.fallback_used,
            "execution_latency_ms": self.execution_latency_ms,
            "reasons": self.reasons,
            "directional_weights": self.directional_weights,
            "policy_id": self.policy_id,
        }


class XAppController:
    def __init__(
        self,
        cooldown_cycles: int = 2,
        hysteresis_margin: float = 0.05,
        rollback_windows: int = 3,
        dc_prr_degradation_margin: float = 0.01,
        violation_degradation_ratio: float = 0.05,
        latency_degradation_ratio: float = 0.05,
        direction_gap_threshold: float = 0.05,
        performance_shield: bool = True,
    ):
        self.cooldown_cycles = cooldown_cycles
        self.hysteresis_margin = hysteresis_margin
        self.rollback_windows = rollback_windows
        self.dc_prr_degradation_margin = dc_prr_degradation_margin
        self.violation_degradation_ratio = violation_degradation_ratio
        self.latency_degradation_ratio = latency_degradation_ratio
        self.direction_gap_threshold = direction_gap_threshold
        self.performance_shield = performance_shield
        self.state = XAppState()

    def decide(self, policy: dict[str, Any], telemetry: TelemetrySnapshot) -> XAppDecision:
        start = time.perf_counter()
        fallback_used = False
        reasons: list[str] = []
        adaptations = 0
        safety_interventions = 0

        policy, rollback_fallback = self._policy_for_window(policy, telemetry, reasons)
        fallback_used = fallback_used or rollback_fallback

        try:
            weights = {service: int(policy["baseline_weights"][service]) for service in SERVICE_NAMES}
            bounds = {service: [int(policy["bounds"][service][0]), int(policy["bounds"][service][1])] for service in SERVICE_NAMES}
            safety = policy["safety_constraints"]
            rules = list(policy.get("xapp_rules", []))
        except Exception as exc:
            fallback_used = True
            reasons.append(f"policy read failed: {exc}")
            policy = static_expert_policy(str(policy.get("scenario_id", "unknown")), str(policy.get("objective_id", "default_objective"))) if isinstance(policy, dict) else static_expert_policy()
            weights = dict(policy["baseline_weights"])
            bounds = {service: [1, 10] for service in SERVICE_NAMES}
            safety = policy["safety_constraints"]
            rules = []

        if self.state.cooldown_remaining > 0:
            self.state.cooldown_remaining -= 1
            rules_to_apply = []
            reasons.append("cooldown active; holding baseline policy weights")
        else:
            rules_to_apply = rules

        for rule in rules_to_apply:
            if _condition_matches(rule["condition"], telemetry, self.hysteresis_margin):
                service = rule["action"]["service"]
                before = weights[service]
                weights[service] = _apply_action(before, rule["action"])
                low, high = bounds[service]
                weights[service] = max(low, min(high, weights[service]))
                if weights[service] != before:
                    adaptations += 1
                    reasons.append(rule["name"])

        safety_interventions += self._enforce_safety(weights, bounds, safety, telemetry, reasons)
        directional_weights = {"ul": dict(weights), "dl": dict(weights)}
        if self.performance_shield:
            safety_interventions += self._enforce_directional_safety(directional_weights, bounds, telemetry, reasons)

        if adaptations or safety_interventions:
            self.state.cooldown_remaining = self.cooldown_cycles
            self.state.stable_cycles = 0
        else:
            self.state.stable_cycles += 1
        self.state.previous_weights = dict(weights)

        elapsed = (time.perf_counter() - start) * 1000.0
        return XAppDecision(
            weights,
            adaptations,
            safety_interventions,
            fallback_used,
            elapsed,
            reasons,
            directional_weights,
            _policy_id(policy),
        )

    def _policy_for_window(self, policy: dict[str, Any], telemetry: TelemetrySnapshot, reasons: list[str]) -> tuple[dict[str, Any], bool]:
        fallback_used = False
        incoming_policy_id = _policy_id(policy)

        if self.state.active_policy is None:
            self._accept_active_policy(policy, incoming_policy_id, telemetry)
            return copy.deepcopy(self.state.active_policy), fallback_used

        if (
            incoming_policy_id != self.state.last_input_policy_id
            and incoming_policy_id != self.state.rolled_back_input_policy_id
        ):
            self.state.previous_validated_policy = copy.deepcopy(self.state.active_policy)
            self._accept_active_policy(policy, incoming_policy_id, telemetry)
            reasons.append(f"accepted_policy_{incoming_policy_id}")
            return copy.deepcopy(self.state.active_policy), fallback_used

        if incoming_policy_id == self.state.rolled_back_input_policy_id:
            fallback_used = True
            reasons.append("rollback_active_holding_previous_policy")
            return copy.deepcopy(self.state.active_policy), fallback_used

        if not self.performance_shield or not telemetry.critical_observed:
            self.state.degraded_windows = 0
            return copy.deepcopy(self.state.active_policy), fallback_used
        if self.state.rollback_baseline is None:
            self.state.rollback_baseline = _critical_metrics(telemetry)
            return copy.deepcopy(self.state.active_policy), fallback_used

        if self._performance_degraded(telemetry):
            self.state.degraded_windows += 1
            reasons.append(f"rollback_degraded_window_{self.state.degraded_windows}")
        else:
            self.state.degraded_windows = 0

        if self.state.degraded_windows >= self.rollback_windows:
            rollback_target = self.state.previous_validated_policy or static_expert_policy(
                str(policy.get("scenario_id", "unknown")),
                str(policy.get("objective_id", "default_objective")),
            )
            self.state.active_policy = copy.deepcopy(rollback_target)
            self.state.active_policy_id = _policy_id(rollback_target)
            self.state.rolled_back_input_policy_id = incoming_policy_id
            self.state.rollback_baseline = _critical_metrics(telemetry)
            self.state.degraded_windows = 0
            fallback_used = True
            reasons.append("rollback_policy_performance_degradation")
            return copy.deepcopy(self.state.active_policy), fallback_used

        return copy.deepcopy(self.state.active_policy), fallback_used

    def _accept_active_policy(self, policy: dict[str, Any], policy_id: str, telemetry: TelemetrySnapshot) -> None:
        self.state.active_policy = copy.deepcopy(policy)
        self.state.active_policy_id = policy_id
        self.state.last_input_policy_id = policy_id
        self.state.rolled_back_input_policy_id = None
        self.state.rollback_baseline = _critical_metrics(telemetry) if telemetry.critical_observed else None
        self.state.degraded_windows = 0

    def _performance_degraded(self, telemetry: TelemetrySnapshot) -> bool:
        baseline = self.state.rollback_baseline
        if not baseline:
            return False
        current = _critical_metrics(telemetry)
        dc_prr_bad = current["critical_dc_prr"] < baseline["critical_dc_prr"] - self.dc_prr_degradation_margin
        violation_limit = baseline["critical_violation_rate"] * (1.0 + self.violation_degradation_ratio)
        latency_limit = baseline["critical_p95_latency_ms"] * (1.0 + self.latency_degradation_ratio)
        violation_bad = current["critical_violation_rate"] > violation_limit
        latency_bad = current["critical_p95_latency_ms"] > latency_limit
        return dc_prr_bad or violation_bad or latency_bad

    def _enforce_safety(self, weights: dict[str, int], bounds: dict[str, list[int]], safety: dict[str, Any], telemetry: TelemetrySnapshot, reasons: list[str]) -> int:
        interventions = 0
        min_crit = int(safety.get("min_critical_weight", 5))
        for service in CRITICAL_SERVICES:
            before = weights[service]
            weights[service] = max(weights[service], min_crit, bounds[service][0])
            weights[service] = min(weights[service], bounds[service][1])
            if weights[service] != before:
                interventions += 1
                reasons.append(f"safety_min_{service}")
        if weights[SERVICE_TOD] < weights[SERVICE_AWARENESS]:
            weights[SERVICE_TOD] = min(bounds[SERVICE_TOD][1], weights[SERVICE_AWARENESS])
            interventions += 1
            reasons.append("safety_tod_not_below_awareness")
        if weights[SERVICE_TOD] < weights[SERVICE_SENSOR]:
            weights[SERVICE_TOD] = min(bounds[SERVICE_TOD][1], weights[SERVICE_SENSOR])
            interventions += 1
            reasons.append("safety_tod_not_below_sensor")
        if weights[SERVICE_AWARENESS] < weights[SERVICE_SENSOR]:
            weights[SERVICE_AWARENESS] = min(bounds[SERVICE_AWARENESS][1], weights[SERVICE_SENSOR])
            interventions += 1
            reasons.append("safety_awareness_not_below_sensor")
        if self.performance_shield and telemetry.congestion_ratio > 0.90:
            for service in (SERVICE_HDMAP, SERVICE_SENSOR):
                before = weights[service]
                weights[service] = max(bounds[service][0], weights[service] - 1)
                if weights[service] != before:
                    interventions += 1
                    reasons.append(f"safety_congestion_trim_{service}")
        return interventions

    def _enforce_directional_safety(
        self,
        directional_weights: dict[str, dict[str, int]],
        bounds: dict[str, list[int]],
        telemetry: TelemetrySnapshot,
        reasons: list[str],
    ) -> int:
        ul_dc_prr = _avg_directional_metric(telemetry, "ul_dc_prr")
        dl_dc_prr = _avg_directional_metric(telemetry, "dl_dc_prr")
        if ul_dc_prr is None or dl_dc_prr is None:
            return 0
        gap = abs(ul_dc_prr - dl_dc_prr)
        if gap <= self.direction_gap_threshold:
            return 0
        direction = "ul" if ul_dc_prr < dl_dc_prr else "dl"
        interventions = 0
        for service in CRITICAL_SERVICES:
            before = directional_weights[direction][service]
            directional_weights[direction][service] = min(bounds[service][1], before + 1)
            if directional_weights[direction][service] != before:
                interventions += 1
        if interventions:
            reasons.append(f"safety_directional_dc_prr_gap_{direction}")
        return interventions


def _condition_matches(condition: dict[str, Any], telemetry: TelemetrySnapshot, margin: float) -> bool:
    observed = telemetry.metric(condition["metric"])
    if observed is None:
        return False
    target = float(condition["value"])
    op = condition["operator"]
    if op == ">":
        return observed > target * (1.0 + margin)
    if op == ">=":
        return observed >= target
    if op == "<":
        return observed < target * (1.0 - margin)
    if op == "<=":
        return observed <= target
    return False


def _apply_action(current: int, action: dict[str, Any]) -> int:
    kind = action["type"]
    value = int(action["value"])
    if kind == "set_weight":
        return value
    if kind == "increase_weight":
        return current + value
    if kind == "decrease_weight":
        return current - value
    return current


def _policy_id(policy: dict[str, Any]) -> str:
    if isinstance(policy, dict):
        return str(policy.get("policy_id", "unknown_policy"))
    return "invalid_policy"


def _critical_metrics(telemetry: TelemetrySnapshot) -> dict[str, float]:
    critical = [telemetry.services[service] for service in CRITICAL_SERVICES]
    count = max(1, len(critical))
    return {
        "critical_dc_prr": sum(t.dc_prr for t in critical) / count,
        "critical_violation_rate": sum(t.violation_rate for t in critical) / count,
        "critical_p95_latency_ms": sum(t.p95_latency_ms for t in critical) / count,
    }


def _avg_directional_metric(telemetry: TelemetrySnapshot, attr: str) -> float | None:
    values = []
    for service in CRITICAL_SERVICES:
        value = getattr(telemetry.services[service], attr, None)
        if value is None:
            return None
        values.append(float(value))
    return sum(values) / max(1, len(values))
