import json
from pathlib import Path

from ltx_trainer.online_data.manifest import (
    build_canonical_r2v_record,
    iter_annotation_items,
    normalize_r2v_source,
)
from ltx_trainer.online_data.manifest_schema import validate_manifest_record


def _phantom_row() -> dict:
    return {
        "video_path": "target.mp4",
        "metadata": {"video_caption": "the reference subject turns toward camera"},
        "cropped_ref_paths": ["ref.png"],
        "cross_pair": {"valid": True},
    }


def test_json_dict_and_list_preserve_source_record_ids(tmp_path: Path) -> None:
    dict_path = tmp_path / "dict.json"
    dict_path.write_text(json.dumps({"dict-id": _phantom_row()}), encoding="utf-8")
    list_path = tmp_path / "list.json"
    list_row = {**_phantom_row(), "source_record_id": "list-id"}
    list_path.write_text(json.dumps([list_row]), encoding="utf-8")

    assert next(iter_annotation_items(dict_path))[0] == "dict-id"
    assert next(iter_annotation_items(list_path))[0] == "list-id"


def test_generic_phantom_builder_plans_121_frames_and_12_anchors(tmp_path: Path) -> None:
    canonical = normalize_r2v_source(
        _phantom_row(),
        row_id="phantom-1",
        dataset_name="r2v_phantom",
        dataset_type="PhantomDataset",
        data_root=tmp_path,
        adapter_config={"require_cross_pair": True},
        manifest_seed=42,
    )
    record = build_canonical_r2v_record(
        canonical,
        manifest_seed=42,
        anchor_frame_ratio=0.10,
        video_header={"fps": 30.0, "frame_count": 600, "width": 1280, "height": 720},
        image_validator=lambda _path: (640, 640),
        target_path_validated=True,
    )

    assert record["adapter_name"] == "phantom"
    assert record["source_record_id"] == "phantom-1"
    assert record["clip_start_frame"] == 0
    assert record["clip_end_frame"] == 600
    assert len(record["target_source_frame_indices"]) == 121
    assert len(record["semantic_anchor_target_indices"]) == 12
    assert record["semantic_anchor_target_indices"][0] == 0
    assert record["semantic_anchor_target_indices"][-1] == 120
    assert record["semantic_anchor_source_indices"] == [
        record["target_source_frame_indices"][index]
        for index in record["semantic_anchor_target_indices"]
    ]
    assert "vlm_target_frame_indices" not in record
    assert "vlm_source_frame_indices" not in record
    validate_manifest_record(record, 0)


def test_manifest_plan_is_deterministic_for_seed(tmp_path: Path) -> None:
    canonical = normalize_r2v_source(
        _phantom_row(),
        row_id="phantom-1",
        dataset_name="r2v_phantom",
        dataset_type="PhantomDataset",
        data_root=tmp_path,
        adapter_config={},
        manifest_seed=7,
    )
    kwargs = {
        "manifest_seed": 7,
        "video_header": {"fps": 30.0, "frame_count": 600, "width": 1280, "height": 720},
        "image_validator": lambda _path: (640, 640),
        "target_path_validated": True,
    }
    assert build_canonical_r2v_record(canonical, **kwargs) == build_canonical_r2v_record(
        canonical, **kwargs
    )
