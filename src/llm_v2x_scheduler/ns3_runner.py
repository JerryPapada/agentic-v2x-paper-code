"""Configurable ns-3 runner for mock and real command-template execution."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from .mock_simulator import run_mock_simulation
from .parser import dump_policy
from .utils import PROJECT_ROOT


def run_experiment(policy: dict[str, Any], scenario: dict[str, Any], out_dir: str | Path, mode: str = "mock", force: bool = False) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = out / "run_complete.json"
    if done.exists() and not force:
        return {"success": True, "skipped": True, "metadata": str(done)}
    if mode == "mock":
        rows = run_mock_simulation(
            policy=policy,
            out_dir=out,
            scenario=scenario,
            cycles=10 if scenario.get("duration") == "short" else 30,
            seed=int(scenario.get("seed", 1)),
            adaptive=bool(policy.get("xapp_rules")),
        )
        result = {"success": True, "mode": "mock", "rows": len(rows), "skipped": False}
        done.write_text(json.dumps(result, indent=2) + "\n")
        return result
    if mode == "real":
        return _run_real_command(policy, scenario, out)
    raise ValueError("mode must be mock or real")


def _run_real_command(policy: dict[str, Any], scenario: dict[str, Any], out: Path) -> dict[str, Any]:
    config = _load_runner_config()
    policy_path = out / "executed_policy.yaml"
    dump_policy(policy_path, policy)
    # traffic_load is not part of the v1 experiment matrix; kept for backward compatibility
    # with any existing command_template that references {traffic_load}
    command = str(config.get("command_template", "")).format(
        policy_path=policy_path,
        density=scenario.get("density"),
        seed=scenario.get("seed"),
        traffic_load=scenario.get("traffic_load", scenario.get("load", "normal")),
        output_dir=out,
    )
    if not command:
        raise ValueError("configs/ns3_runner.yaml has no command_template")
    env = dict(os.environ)
    env["JERRY_RESULTS_DIR"] = str(out)
    env["NS_GLOBAL_VALUE"] = f"RngRun={scenario.get('seed', 1)}"
    started = time.perf_counter()
    proc = subprocess.run(
        shlex.split(command),
        cwd=PROJECT_ROOT.parents[1],
        env=env,
        text=True,
        capture_output=True,
    )
    (out / "ns3_stdout.log").write_text(proc.stdout)
    (out / "ns3_stderr.log").write_text(proc.stderr)
    result = {
        "success": proc.returncode == 0,
        "mode": "real",
        "returncode": proc.returncode,
        "command": command,
        "elapsed_s": time.perf_counter() - started,
        "skipped": False,
    }
    (out / "run_complete.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _load_runner_config() -> dict[str, Any]:
    path = PROJECT_ROOT / "configs" / "ns3_runner.yaml"
    try:
        import yaml
        data = yaml.safe_load(path.read_text())
        return data or {}
    except Exception:
        return {"command_template": ""}
