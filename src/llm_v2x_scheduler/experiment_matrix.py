"""Debug and paper-mode experiment matrix construction."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .baselines import CORE_METHODS


def build_matrix(
    mode: str = "debug",
    *,
    densities_override: list[int] | None = None,
    seeds_override: list[int] | None = None,
) -> list[dict[str, Any]]:
    if mode == "debug":
        densities = [30]
        seeds = [1]
        duration = "short"
    elif mode == "paper":
        densities = [20, 25, 30]
        seeds = [1, 2, 3]
        duration = "paper"
    else:
        raise ValueError("mode must be debug or paper")
    if densities_override is not None:
        densities = densities_override
    if seeds_override is not None:
        seeds = seeds_override
    rows = []
    for density in densities:
        for seed in seeds:
            for method in CORE_METHODS:
                scenario_id = f"{density}veh_seed{seed}"
                rows.append({
                    "scenario_id": scenario_id,
                    "density": density,
                    "seed": seed,
                    "duration": duration,
                    "method": method,
                    "objective_id": "safety_background_balance",
                })
    return rows


def write_matrix(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
