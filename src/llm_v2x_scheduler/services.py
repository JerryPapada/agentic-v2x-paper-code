"""Central service-name mapping for the LLM V2X scheduler pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


SERVICE_TOD = "ToD"
SERVICE_AWARENESS = "Awareness"
SERVICE_HDMAP = "HDMap"
SERVICE_SENSOR = "Sensor"

SERVICE_NAMES = (SERVICE_TOD, SERVICE_AWARENESS, SERVICE_HDMAP, SERVICE_SENSOR)
CRITICAL_SERVICES = (SERVICE_TOD, SERVICE_AWARENESS)
BACKGROUND_SERVICES = (SERVICE_HDMAP, SERVICE_SENSOR)

ALIASES = {
    "tod": SERVICE_TOD,
    "to d": SERVICE_TOD,
    "to_d": SERVICE_TOD,
    "tod.": SERVICE_TOD,
    "awareness": SERVICE_AWARENESS,
    "aic": SERVICE_AWARENESS,
    "hdmap": SERVICE_HDMAP,
    "hdm": SERVICE_HDMAP,
    "hd_map": SERVICE_HDMAP,
    "sensor": SERVICE_SENSOR,
    "rtsa": SERVICE_SENSOR,
}

NS3_LABELS = {
    SERVICE_TOD: ("ToD", "TOD"),
    SERVICE_AWARENESS: ("AIC",),
    SERVICE_HDMAP: ("HDM",),
    SERVICE_SENSOR: ("RTSA",),
}


@dataclass(frozen=True)
class ServiceWeights:
    """Canonical per-service scheduler weights."""

    ToD: int
    Awareness: int
    HDMap: int
    Sensor: int

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in SERVICE_NAMES}


def canonical_service(name: str) -> str:
    key = str(name).strip()
    if key in SERVICE_NAMES:
        return key
    lowered = key.lower().replace("-", "_")
    if lowered in ALIASES:
        return ALIASES[lowered]
    raise ValueError(f"unsupported service: {name!r}")


def canonicalize_service_dict(values: dict) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in values.items():
        service = canonical_service(key)
        if service in out:
            raise ValueError(f"duplicate service after canonicalization: {service}")
        out[service] = value
    return out


def ensure_all_services(values: dict, context: str) -> list[str]:
    missing = [service for service in SERVICE_NAMES if service not in values]
    return [f"{context} missing service {service}" for service in missing]


def service_from_ns3_label(label: str) -> str:
    raw = str(label).strip()
    try:
        return canonical_service(raw)
    except ValueError:
        pass
    upper = raw.upper()
    for service, labels in NS3_LABELS.items():
        if upper in {item.upper() for item in labels}:
            return service
    raise ValueError(f"unsupported ns-3 service label: {label!r}")


def empty_service_map(default: object = 0) -> dict[str, object]:
    return {service: default for service in SERVICE_NAMES}


def ordered_values(values: dict[str, object]) -> list[object]:
    return [values[service] for service in SERVICE_NAMES]


def validate_service_names(names: Iterable[str]) -> list[str]:
    errors: list[str] = []
    for name in names:
        try:
            canonical_service(name)
        except ValueError as exc:
            errors.append(str(exc))
    return errors

