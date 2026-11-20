"""Telegram notification wrapper using the existing thesis bot."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from .utils import PROJECT_ROOT


def notify_experiment(message: str, silent: bool = False) -> bool:
    """Send a Telegram message through scratch/thesis/tools/notify.py."""
    notifier = _load_thesis_notifier()
    if notifier is None:
        print("  [notify] thesis notifier not found")
        return False
    try:
        return bool(notifier.notify(message, silent=silent))
    except Exception as exc:
        print(f"  [notify] Telegram failed: {exc}")
        return False


def _load_thesis_notifier():
    ns3_root = PROJECT_ROOT.parents[1]
    path = ns3_root / "scratch" / "thesis" / "tools" / "notify.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("thesis_notify", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def format_run_message(summary: dict) -> str:
    status = "OK" if summary.get("success") else "FAILED"
    method = summary.get("method", "unknown")
    density = summary.get("density", "?")
    seed = summary.get("seed", "?")
    sim_time = summary.get("sim_time_s", summary.get("sim_time", "?"))
    model = summary.get("model_name") or summary.get("model") or "n/a"
    policy = summary.get("executed_policy_id") or summary.get("policy_id") or "n/a"
    elapsed = float(summary.get("duration_s", summary.get("elapsed_s", 0.0)) or 0.0)
    out_dir = summary.get("output_dir", "")
    return (
        f"*llm-v2x experiment {status}*\n"
        f"method: `{method}`\n"
        f"density: `{density}`  seed: `{seed}`  sim: `{sim_time}s`\n"
        f"model: `{model}`\n"
        f"policy: `{policy}`\n"
        f"duration: `{elapsed:.1f}s`\n"
        f"out: `{out_dir}`"
    )
