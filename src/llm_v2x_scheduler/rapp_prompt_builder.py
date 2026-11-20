"""Prompt construction for the slow LLM-rApp policy creator."""

from __future__ import annotations

from .utils import PROJECT_ROOT
from .telemetry import TelemetrySnapshot, telemetry_summary


def build_rapp_prompt(
    scenario: dict,
    objective: str,
    telemetry: TelemetrySnapshot,
    model_name: str = "unknown_model",
    objective_id: str = "safety_background_balance",
) -> str:
    scenario_id = str(scenario.get("scenario_id", "unknown_scenario"))
    policy_id = f"{_safe_id(model_name)}_{scenario_id}_rapp_policy"
    system_prompt = _read_prompt("rapp_system_prompt.txt")
    template = _read_prompt("rapp_policy_prompt_template.txt")
    user_prompt = template.format(
        telemetry_summary=telemetry_summary(telemetry, scenario, objective),
        scenario_id=scenario_id,
        objective_id=objective_id,
        model_name=model_name,
        policy_id=policy_id,
    )
    return system_prompt.rstrip() + "\n\n" + user_prompt.rstrip() + "\n"


def _read_prompt(name: str) -> str:
    return (PROJECT_ROOT / "prompts" / name).read_text()


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower()
