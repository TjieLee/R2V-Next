"""Phantom dict/list annotation adapter."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ltx_trainer.online_data.adapters.base import (
    AdapterReject,
    CanonicalR2VSource,
    get_nested_field,
    parse_four_floats,
    parse_path_values,
    resolve_source_path,
)


class PhantomAdapter:
    name = "phantom"

    def normalize(
        self,
        row: Mapping[str, Any],
        *,
        row_id: str,
        dataset_name: str,
        data_root: str | Path | None,
        config: Mapping[str, Any],
    ) -> CanonicalR2VSource:
        video_field = str(config.get("video_field", "video_path"))
        caption_field = str(config.get("caption_field", "metadata.video_caption"))
        reference_field = str(config.get("reference_field", "cropped_ref_paths"))
        video_value = get_nested_field(row, video_field)
        caption = str(get_nested_field(row, caption_field) or "").strip()
        if not caption:
            raise AdapterReject("phantom_empty_caption", f"Phantom caption is empty for row {row_id}")
        if video_value is None or not str(video_value).strip():
            raise AdapterReject("phantom_missing_target", f"Phantom target path is empty for row {row_id}")
        if bool(config.get("require_cross_pair", False)) and row.get("cross_pair") is None:
            raise AdapterReject(
                "phantom_missing_cross_pair",
                f"Phantom row {row_id} has no cross_pair while require_cross_pair=true",
            )
        try:
            raw_references = parse_path_values(get_nested_field(row, reference_field), field=reference_field)
        except AdapterReject as exc:
            raise AdapterReject(
                "phantom_missing_reference", f"Phantom row {row_id} has no valid reference list"
            ) from exc
        resolved: list[str] = []
        seen: set[str] = set()
        for value in raw_references:
            path = resolve_source_path(value, data_root)
            if path and path not in seen:
                seen.add(path)
                resolved.append(path)
        if not resolved:
            raise AdapterReject("phantom_missing_reference", f"Phantom row {row_id} has no reference images")

        maximum = int(config.get("max_reference_images", 4))
        if maximum <= 0:
            raise AdapterReject("phantom_missing_reference", "max_reference_images must be positive")
        selected = resolved
        if len(resolved) > maximum:
            manifest_seed = int(config.get("manifest_seed", 0))
            scored = [
                (
                    hashlib.sha256(
                        f"{dataset_name}|{row_id}|{manifest_seed}|{index}|{path}".encode("utf-8")
                    ).hexdigest(),
                    index,
                    path,
                )
                for index, path in enumerate(resolved)
            ]
            selected_indices = sorted(index for _score, index, _path in sorted(scored)[:maximum])
            selected = [resolved[index] for index in selected_indices]

        crop = None
        crop_field = config.get("crop_field")
        if crop_field:
            crop_value = get_nested_field(row, str(crop_field))
            if crop_value is not None:
                crop = parse_four_floats(crop_value, field=str(crop_field))
        clip_start = 0
        clip_start_field = config.get("clip_start_field")
        if clip_start_field:
            value = get_nested_field(row, str(clip_start_field))
            if value is not None:
                clip_start = int(value)
        clip_end = None
        clip_end_field = config.get("clip_end_field")
        if clip_end_field:
            value = get_nested_field(row, str(clip_end_field))
            if value is not None:
                clip_end = int(value)
        if clip_start < 0 or (clip_end is not None and clip_end <= clip_start):
            raise AdapterReject("invalid_clip", f"Invalid Phantom clip [{clip_start}, {clip_end})")

        return CanonicalR2VSource(
            dataset_name=dataset_name,
            adapter_name=self.name,
            source_record_id=str(row_id),
            video_path=resolve_source_path(video_value, data_root),
            caption=caption,
            reference_paths=selected,
            crop_xyxy=crop,
            clip_start_frame=clip_start,
            clip_end_frame=clip_end,
            metadata={
                "cross_pair_present": row.get("cross_pair") is not None,
                "raw_reference_count": len(resolved),
                "selected_reference_count": len(selected),
            },
        )
