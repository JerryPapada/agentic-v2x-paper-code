"""Small aggregation helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def combine_csvs(paths: list[str | Path], out_path: str | Path) -> None:
    combined: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open() as f:
            combined.extend(csv.DictReader(f))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not combined:
        return
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(combined[0].keys()))
        writer.writeheader()
        writer.writerows(combined)

