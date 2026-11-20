"""Plotting helpers. Matplotlib is preferred; SVG fallback keeps smoke runnable."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def plot_critical_dc_prr(metrics_csv: str | Path, out_path: str | Path) -> Path:
    rows = _read_csv(metrics_csv)
    rows = [r for r in rows if r["service"] == "aggregate"]
    labels = [r["method"] for r in rows]
    values = [float(r["dc_prr"]) for r in rows]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, values, color="#2f6fbb")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Critical DC-PRR")
        ax.set_title("Critical DC-PRR by Method")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        return out
    except Exception:
        svg_path = out.with_suffix(".svg")
        _write_svg_bar(svg_path, labels, values, "Critical DC-PRR by Method")
        return svg_path


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def _write_svg_bar(path: Path, labels: list[str], values: list[float], title: str) -> None:
    width, height = 760, 420
    margin = 60
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    n = max(1, len(values))
    bar_w = plot_w / n * 0.65
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#333"/>',
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        x = margin + (i + 0.5) * plot_w / n - bar_w / 2
        h = max(0.0, min(1.0, value)) * plot_h
        y = height - margin - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#2f6fbb"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.3f}</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{height - margin + 22}" text-anchor="middle" font-family="sans-serif" font-size="11">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")

