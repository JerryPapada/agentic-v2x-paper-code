"""Policy generation and telemetry for the Adaptive LLM revision study."""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path
from urllib.request import Request, urlopen

import yaml

from .parser import parse_policy_text
from .schema import ADAPTIVE_DIRECTIONAL_METRICS
from .services import CRITICAL_SERVICES, SERVICE_NAMES
from .telemetry import ServiceTelemetry, TelemetrySnapshot
from .utils import PROJECT_ROOT
from .validator import validate_policy

MODEL = "qwen2.5:7b"
OLLAMA = "http://127.0.0.1:11434"
MODEL_OPTIONS = {"temperature": 0.1, "num_predict": 900}
OBJECTIVES = {
    "safety_background_balance": "Protect ToD and Awareness while preserving background throughput when possible.",
    "protect_tod_ul": "Give first priority to restoring teleoperated-driving uplink delivery within its deadline. Preserve Awareness and all hard safety limits.",
    "protect_awareness": "Give first priority to restoring cooperative awareness delivery in either direction, while protecting ToD and respecting all hard safety limits.",
    "recover_hdmap": "Where the critical services are healthy, prioritize recovery of HDMap downlink throughput. Preserve ToD and Awareness and all hard safety limits.",
}
PAYLOADS = {
    "ToD": {"ul": 1400, "dl": 200},
    "Awareness": {"ul": 400, "dl": 400},
    "HDMap": {"ul": 500, "dl": 1400},
    "Sensor": {"ul": 1200, "dl": 1200},
}
DIRECTIONS = ("ul", "dl")
SERVICE_METRIC_IDS = {"ToD": "tod", "Awareness": "awareness"}
PROMPT_SCHEMA = """Generate one scheduler policy for the supplied operator intent and telemetry.
Return exactly one JSON object and no Markdown fences, preamble, or trailing text. JSON is a YAML-compatible mapping.
The deterministic xApp executes ordered simple rules.
ToD means teleoperated driving. Sensor/video uplink is the high-load path (100 ms deadline);
operator command downlink has a 20 ms deadline. Awareness has a 20 ms deadline in both directions.
HDMap is background map download traffic. Sensor is background sensor traffic.
Set min_critical_weight to exactly 5. ToD and Awareness baseline weights and lower bounds
must each be at least 5. All weights and bounds are integers in 1..10.
Each baseline weight must lie between its service's lower and upper bounds, inclusive.
ToD baseline weight must be at least Awareness. Never assign zero weight.
max_background_degradation is an operator target, which does not guarantee throughput.
Directional delivery ratio is window received packets divided by transmitted packets, capped at 1.
It is a delivery proxy, not deadline-constrained packet reception. Late UE fraction is the share
of observed UEs whose received packets include one that missed its direction's deadline.
Latency is the p95 across UE mean receive latencies for the direction, in milliseconds.
Null means the measurement is unavailable. The congestion ratio expresses queue bytes
relative to one second of offered traffic, capped at 1. Other ratio metrics are in 0..1.
Use only these rule metrics:
{metrics}

Required policy schema (values describe types):
policy_id: string
description: string
scenario_id: string
objective_id: string
baseline_weights: {{ToD: integer, Awareness: integer, HDMap: integer, Sensor: integer}}
bounds: {{ToD: [integer, integer], Awareness: [integer, integer], HDMap: [integer, integer], Sensor: [integer, integer]}}
safety_constraints:
  min_critical_weight: integer
  max_background_degradation: number
  forbid_zero_weight: true
  critical_services: [ToD, Awareness]
xapp_rules:
  - name: string
    condition: {{metric: allowed metric, operator: one of ">" ">=" "<" "<=", value: number}}
    action: {{service: ToD or Awareness or HDMap or Sensor, type: set_weight or increase_weight or decrease_weight, value: integer}}
fallback_policy: static_expert
metadata: {{source: llm_rapp, model_name: string, generation_time: null}}
"""


def policy_content(policy: dict) -> str:
    content = {key: policy[key] for key in ("baseline_weights", "bounds", "safety_constraints", "fallback_policy")}
    content["safety_constraints"] = dict(content["safety_constraints"])
    content["safety_constraints"]["critical_services"] = sorted(content["safety_constraints"]["critical_services"])
    content["xapp_rules"] = [
        {"condition": rule["condition"], "action": rule["action"]} for rule in policy["xapp_rules"]
    ]
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


class PolicyRevisions:
    def __init__(self, startup: dict):
        self.revision = 0
        self.content = policy_content(startup)
        self.policy = self._with_revision(startup)

    def publish(self, proposal: dict) -> bool:
        content = policy_content(proposal)
        if self.content == content:
            return False
        self.revision += 1
        self.content = content
        self.policy = self._with_revision(proposal)
        return True

    def _with_revision(self, policy: dict) -> dict:
        revised = deepcopy(policy)
        revised["source_policy_id"] = policy["policy_id"]
        revised["policy_id"] = f"revision_{self.revision:03d}"
        return revised


def ollama_json(path: str, payload: dict | None = None, *, timeout: int = 180) -> dict:
    request = Request(
        OLLAMA + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def model_digest() -> str:
    for model in ollama_json("/api/tags", timeout=15)["models"]:
        if model["name"] == MODEL:
            return model["digest"]
    raise RuntimeError(f"Local Ollama does not contain {MODEL}")


def identifiers(scenario_id: str, objective: str) -> dict:
    return {
        "scenario_id": scenario_id,
        "objective_id": objective,
        "model_name": MODEL,
        "policy_id": f"qwen2_5_7b_{scenario_id}_rapp_policy",
    }


def build_prompt(scenario_id: str, objective: str, metrics: dict, active: dict, *, example: bool = False) -> str:
    ids = identifiers(scenario_id, objective)
    allowed = (*ADAPTIVE_DIRECTIONAL_METRICS, "hdmap_throughput_degradation", "sensor_throughput_degradation", "congestion_ratio")
    prompt = PROMPT_SCHEMA.format(metrics="\n".join(f"- {metric}" for metric in allowed))
    prompt += "\nUse these identifiers exactly:\n" + json.dumps(ids, indent=2)
    prompt += "\nOperator intent:\n" + OBJECTIVES[objective]
    prompt += "\nActive policy weights and source:\n" + json.dumps(active, sort_keys=True)
    prompt += "\nRecent telemetry:\n" + json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if example:
        template = (PROJECT_ROOT / "prompts/rapp_policy_prompt_template.txt").read_text()
        worked = template[template.index("A valid conservative example shape is:") :].format(**ids)
        for old, revised in {
            "tod_p95_latency_ms": "tod_ul_p95_ue_mean_latency_ms",
            "awareness_violation_rate": "awareness_ul_late_ue_fraction",
        }.items():
            worked = worked.replace(old, revised)
        prompt += "\n" + worked
    return prompt


def assess_response(raw: str, expected_ids: dict | None = None) -> tuple[dict, dict | None]:
    try:
        candidate = parse_policy_text(raw)
    except (ValueError, TypeError, OverflowError) as exc:
        return {"parsed": False, "strict_pass": False, "accepted": False, "repaired": False, "errors": [str(exc)]}, None
    # Remove the old runner's trailing timing annotation before checking format.
    plain = raw.partition("\n\n<!-- llm_generation_latency_s=")[0]
    try:
        direct_mapping = isinstance(yaml.safe_load(plain), dict)
    except yaml.YAMLError:
        direct_mapping = False
    try:
        strict = validate_policy(candidate, attempt_repair=False, adaptive=True)
        identity_errors = []
        if expected_ids is not None:
            expected = {k: v for k, v in expected_ids.items() if k != "model_name"}
            identity_errors = [f"Unexpected {key}" for key, value in expected.items() if candidate.get(key) != value]
            candidate.update(expected)
            if isinstance(candidate.get("metadata"), dict) and candidate["metadata"].get("model_name") != expected_ids["model_name"]:
                identity_errors.append("Unexpected metadata.model_name")
                candidate["metadata"]["model_name"] = expected_ids["model_name"]
        result = strict if strict.valid and not identity_errors else validate_policy(candidate, attempt_repair=True, adaptive=True)
        return {
            "parsed": True,
            "strict_format": direct_mapping,
            "strict_pass": strict.valid and direct_mapping and not identity_errors,
            "accepted": result.valid and not result.rejected,
            "repaired": result.valid and (result.repaired or not direct_mapping or bool(identity_errors)),
            "errors": strict.errors + identity_errors + ([] if direct_mapping else ["Response needed extraction of a YAML mapping"]),
            "warnings": result.warnings,
            "repair_errors": result.errors if result.rejected else [],
        }, result.policy if result.valid and not result.rejected else None
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        return {"parsed": True, "strict_pass": False, "accepted": False, "repaired": False, "errors": [str(exc)]}, None


def generate_proposal(prompt: str, seed: int, output: Path, expected_ids: dict) -> tuple[dict, dict | None]:
    output.mkdir(parents=True, exist_ok=True)
    (output / "prompt.txt").write_text(prompt)
    started = time.time()
    policy = None
    try:
        generation = ollama_json("/api/generate", {
            "model": MODEL, "prompt": prompt, "stream": False,
            "format": "json",
            "options": {**MODEL_OPTIONS, "seed": seed},
        })
        raw = generation["response"]
        (output / "response.txt").write_text(raw)
        (output / "ollama_response.json").write_text(json.dumps(generation, indent=2) + "\n")
        report, policy = assess_response(raw, expected_ids)
        report["done_reason"] = generation.get("done_reason")
    except (OSError, ValueError, KeyError) as exc:
        report = {"parsed": False, "strict_pass": False, "accepted": False, "repaired": False, "errors": [str(exc)]}
    report["wall_request_s"] = started
    report["wall_complete_s"] = time.time()
    report["latency_s"] = report["wall_complete_s"] - started
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if policy is not None:
        (output / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    return report, policy


def observe(state, by_ue: list[str], weights: dict, sensor_ul_mbps: float = 0) -> TelemetrySnapshot:
    metrics = {}
    services = {}
    offered_bytes_s = 0.0
    queue_bytes = 0.0
    sensor_count = by_ue.count("Sensor")
    for service in SERVICE_NAMES:
        stats = {}
        indices = [idx for idx, assigned in enumerate(by_ue) if assigned == service]
        for direction in DIRECTIONS:
            arrival = getattr(state, direction + "ArrivalRate")
            receive = getattr(state, direction + "ServiceRate")
            latency = getattr(state, direction + "AverageLatency")
            slack = getattr(state, direction + "MinSlack")
            backlog = getattr(state, direction + "BufferBytes")
            active = [idx for idx in indices if receive[idx] > 0]
            rx_rate = sum(receive[idx] for idx in indices)
            tx_rate = sum(arrival[idx] for idx in indices)
            payload = PAYLOADS[service][direction]
            offered_bytes_s += tx_rate * payload
            queued = sum(backlog[idx] for idx in indices)
            queue_bytes += queued
            means = sorted(latency[idx] for idx in active)
            percentile_index = max(0, (95 * len(means) + 99) // 100 - 1)
            values = {
                "delivery_ratio": min(1.0, rx_rate / tx_rate) if tx_rate > 0 else None,
                "late_ue_fraction": sum(slack[idx] < 0 for idx in active) / len(active) if active else None,
                "p95_ue_mean_latency_ms": means[percentile_index] if means else None,
                "offered_mbps": tx_rate * payload * 8 / 1e6,
                "delivered_mbps": rx_rate * payload * 8 / 1e6,
                "queue_bytes": queued,
            }
            prefix = SERVICE_METRIC_IDS.get(service, service.lower())
            metrics.update({f"{prefix}_{direction}_{name}": value for name, value in values.items()})
            stats[direction] = values

        ul, dl = (stats[direction] for direction in DIRECTIONS)
        delivered = ul["delivered_mbps"] + dl["delivered_mbps"]
        offered = ul["offered_mbps"] + dl["offered_mbps"]
        if service == "HDMap":
            offered = len(indices) * (16.0 + 0.008)
        if service == "Sensor":
            offered = sensor_count * 0.048 + (sensor_ul_mbps or sensor_count * 0.048)
        degradation = max(0.0, 1.0 - delivered / offered) if offered else 0.0
        services[service] = ServiceTelemetry(
            dc_prr=min((val for val in (ul["delivery_ratio"], dl["delivery_ratio"]) if val is not None), default=1.0),
            violation_rate=max(ul["late_ue_fraction"] or 0.0, dl["late_ue_fraction"] or 0.0),
            p95_latency_ms=max(ul["p95_ue_mean_latency_ms"] or 0.0, dl["p95_ue_mean_latency_ms"] or 0.0),
            ul_dc_prr=ul["delivery_ratio"], dl_dc_prr=dl["delivery_ratio"],
            throughput_mbps=delivered, throughput_degradation=degradation,
        )
        if service in CRITICAL_SERVICES:
            for old_name, new_name, aggregate in (
                ("dc_prr", "delivery_ratio", min),
                ("violation_rate", "late_ue_fraction", max),
                ("p95_latency_ms", "p95_ue_mean_latency_ms", max),
            ):
                observed = [v for v in (ul[new_name], dl[new_name]) if v is not None]
                metrics[f"{prefix}_{old_name}"] = aggregate(observed) if observed else None
        else:
            metrics[f"{prefix}_throughput_degradation"] = degradation

    congestion = min(1.0, queue_bytes / max(1.0, offered_bytes_s))
    metrics["congestion_ratio"] = congestion
    observed = all(
        services[service].ul_dc_prr is not None and services[service].dl_dc_prr is not None
        for service in CRITICAL_SERVICES
    )
    return TelemetrySnapshot(services, dict(weights), congestion, metrics, observed)


def probe_snapshot(metrics: dict) -> TelemetrySnapshot:
    return TelemetrySnapshot({
        service: ServiceTelemetry(
            ul_dc_prr=metrics.get(f"{service.lower()}_ul_delivery_ratio"),
            dl_dc_prr=metrics.get(f"{service.lower()}_dl_delivery_ratio"),
        ) for service in SERVICE_NAMES
    }, {}, metrics["congestion_ratio"], metrics, False)
