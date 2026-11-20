"""Local LLM runner for slow rApp policy generation."""

from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .baselines import heuristic_xapp_policy
from .parser import dump_policy
from .utils import project_path


def generate_rapp_policy(model_name: str, prompt: str, temperature: float = 0.2, max_tokens: int = 1024, backend: str = "auto") -> str:
    """Generate raw rApp policy text.

    v1 uses Ollama when available and mock generation otherwise. The returned
    value is raw text and must still pass parser/validator before execution.
    """

    if backend in ("auto", "ollama"):
        try:
            return _ollama_generate(model_name, prompt, temperature, max_tokens)
        except Exception:
            if backend == "ollama":
                raise
    return mock_rapp_policy_text(model_name)


def save_raw_llm_output(raw_text: str, label: str) -> Path:
    path = project_path("results", "llm_outputs", f"{label}.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw_text)
    return path


def mock_rapp_policy_text(model_name: str = "mock-rapp") -> str:
    policy = heuristic_xapp_policy()
    policy["policy_id"] = "mock_llm_rapp_policy"
    policy["description"] = "Mock LLM-rApp policy for smoke testing deterministic xApp execution"
    policy["metadata"] = {
        "source": "llm_rapp",
        "model_name": model_name,
        "generation_time": datetime.now(timezone.utc).isoformat(),
    }
    path = project_path("policies", "generated_rapp", "mock_llm_rapp_policy.yaml")
    dump_policy(path, policy)
    return path.read_text()


def _ollama_generate(model_name: str, prompt: str, temperature: float, max_tokens: int) -> str:
    cmd = ["ollama", "run", model_name]
    started = time.perf_counter()
    full_prompt = (
        f"temperature={temperature}\n"
        f"max_tokens={max_tokens}\n\n"
        f"{prompt}"
    )
    proc = subprocess.run(cmd, input=full_prompt, text=True, capture_output=True, check=True, timeout=180)
    elapsed = time.perf_counter() - started
    return proc.stdout + f"\n\n<!-- llm_generation_latency_s={elapsed:.6f} -->\n"
