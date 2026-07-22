"""OpenS2V source-field adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ltx_trainer.online_data.adapters.base import (
    AdapterReject,
    CanonicalR2VSource,
    parse_four_floats,
    parse_path_values,
    parse_two_ints,
    resolve_source_path,
)


class OpenS2VAdapter:
    name = "opens2v"

    def normalize(
        self,
        row: Mapping[str, Any],
        *,
        row_id: str,
        dataset_name: str,
        data_root: str | Path | None,
        config: Mapping[str, Any],
    ) -> CanonicalR2VSource:
        del config
        required = {"video_path", "text", "crop", "face_cut", "ref_images"}
        missing = sorted(required - row.keys())
        if missing:
            raise AdapterReject(
                "r2v_schema_mismatch",
                f"OpenS2V fields are missing: {missing}. Available columns: {sorted(row)}",
            )
        caption = str(row["text"] or "").strip()
        if not caption:
            raise AdapterReject("empty_caption", "OpenS2V caption is empty")
        crop = parse_four_floats(row["crop"], field="crop")
        if not all(math.isfinite(value) for value in crop):
            raise AdapterReject("invalid_crop", f"OpenS2V crop contains non-finite values: {crop}")
        if crop[1] <= crop[0] or crop[3] <= crop[2]:
            raise AdapterReject("invalid_crop", f"OpenS2V crop has non-positive extent: {crop}")
        clip = parse_two_ints(row["face_cut"], field="face_cut")
        if clip[0] < 0 or clip[1] <= clip[0]:
            raise AdapterReject("invalid_face_cut", f"Invalid OpenS2V face_cut={clip}")
        references = [
            resolve_source_path(path, data_root)
            for path in parse_path_values(row["ref_images"], field="ref_images")
        ]
        return CanonicalR2VSource(
            dataset_name=dataset_name,
            adapter_name=self.name,
            source_record_id=str(row_id),
            video_path=resolve_source_path(row["video_path"], data_root),
            caption=caption,
            reference_paths=references,
            crop_xyxy=[crop[0], crop[2], crop[1], crop[3]],
            clip_start_frame=clip[0],
            clip_end_frame=clip[1],
            metadata={"source_schema": "OpenS2VDataset"},
        )
