"""Policy text parsing helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - environment dependent
    yaml = None


def parse_policy_text(text: str) -> dict[str, Any]:
    """Parse JSON/YAML from raw LLM output.

    The function first tries the full text, then fenced code blocks, then the
    first JSON object span. It never executes code.
    """

    text = _strip_instrumentation(_strip_ansi(text))
    candidates = [text.strip()]
    candidates.extend(_fenced_blocks(text))
    obj = _first_json_object(text)
    if obj:
        candidates.append(obj)

    errors: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            errors.append(f"json: {exc}")
        if yaml is not None:
            try:
                parsed = yaml.safe_load(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception as exc:
                errors.append(f"yaml: {exc}")
    raise ValueError("policy text is not parseable JSON/YAML: " + "; ".join(errors[-3:]))


def load_policy_file(path: str | Path) -> dict[str, Any]:
    return parse_policy_text(Path(path).read_text())


def dump_policy(path: str | Path, policy: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None and path.suffix.lower() in (".yaml", ".yml"):
        path.write_text(yaml.safe_dump(policy, sort_keys=False))
    else:
        path.write_text(json.dumps(policy, indent=2, sort_keys=False) + "\n")


def _fenced_blocks(text: str) -> list[str]:
    blocks = []
    for match in re.finditer(r"```(?:json|yaml|yml)?\s*(.*?)```", text, re.DOTALL | re.I):
        blocks.append(match.group(1).strip())
    return blocks


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)


def _strip_instrumentation(text: str) -> str:
    return re.sub(r"\n?\s*<!--\s*llm_generation_latency_s=.*?-->\s*$", "", text, flags=re.DOTALL)


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None
