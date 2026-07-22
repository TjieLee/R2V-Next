"""Canonical source contract shared by all online R2V adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class CanonicalR2VSource:
    dataset_name: str
    adapter_name: str
    source_record_id: str

    video_path: str
    caption: str
    reference_paths: list[str]

    crop_xyxy: list[float] | None
    clip_start_frame: int
    clip_end_frame: int | None

    metadata: dict[str, Any]


class AdapterReject(ValueError):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class R2VSourceAdapter(Protocol):
    name: str

    def normalize(
        self,
        row: Mapping[str, Any],
        *,
        row_id: str,
        dataset_name: str,
        data_root: str | Path | None,
        config: Mapping[str, Any],
    ) -> CanonicalR2VSource: ...


def get_nested_field(row: Mapping[str, Any], field_path: str) -> Any:
    current: Any = row
    for component in field_path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return None
        current = current[component]
    return current


def resolve_source_path(value: Any, data_root: str | Path | None) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and data_root is not None:
        path = Path(data_root).expanduser() / path
    return str(path.resolve())


def parse_path_values(value: Any, *, field: str) -> list[str]:
    if isinstance(value, (list, tuple)):
        values = [str(item).strip() for item in value]
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            import json  # noqa: PLC0415

            return parse_path_values(json.loads(stripped), field=field)
        values = [part.strip() for part in stripped.replace(";", "|").split("|")]
    else:
        raise AdapterReject("r2v_schema_mismatch", f"Field {field!r} must contain a path list")
    return [value for value in values if value]


def parse_four_floats(value: Any, *, field: str) -> list[float]:
    if isinstance(value, str):
        import json  # noqa: PLC0415

        value = json.loads(value)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise AdapterReject("invalid_crop", f"Field {field!r} must contain four numeric values")
    return [float(item) for item in value]


def parse_two_ints(value: Any, *, field: str) -> list[int]:
    if isinstance(value, str):
        import json  # noqa: PLC0415

        value = json.loads(value)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise AdapterReject("invalid_clip", f"Field {field!r} must contain two integer values")
    return [int(item) for item in value]
