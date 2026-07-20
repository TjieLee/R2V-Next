from __future__ import annotations

from pathlib import Path

import pytest

from ltx_trainer.online_inference.external_eval_schema import (
    center_crop_risk,
    normalize_external_dataset,
    stable_sample_seed,
)


def test_three_external_schemas_preserve_reference_order(tmp_path: Path) -> None:
    videox, errors = normalize_external_dataset(
        dataset_schema="videoxfun_test",
        input_json=tmp_path / "videox.json",
        payload=[
            {
                "id": "complex_1",
                "prompt": "move left",
                "ref_images": ["/refs/b.png", "/refs/a.png"],
            }
        ],
    )
    assert not errors
    assert videox[0]["reference_paths"] == ["/refs/b.png", "/refs/a.png"]

    custom, errors = normalize_external_dataset(
        dataset_schema="custom64",
        input_json=tmp_path / "custom.json",
        payload=[
            {
                "video_name": "CASE24_0000.mp4",
                "caption": "turn around",
                "ref_image_paths": ["/refs/one.jpg"],
            }
        ],
    )
    assert not errors
    assert custom[0]["source_record_id"] == "CASE24_0000"
    assert custom[0]["dataset_metadata"]["video_name"] == "CASE24_0000.mp4"

    opens2v, errors = normalize_external_dataset(
        dataset_schema="opens2v_open_domain",
        input_json=tmp_path / "OpenS2V" / "Open-Domain_Eval.json",
        payload={
            "singleobj_1": {
                "img_paths": ["Images/cup/52.jpg"],
                "prompt": "a cup rotates",
                "synthesis_flag": "False",
                "class_label": ["Cup"],
            }
        },
    )
    assert not errors
    assert opens2v[0]["reference_paths"] == [
        str((tmp_path / "OpenS2V" / "Images/cup/52.jpg").resolve())
    ]
    assert opens2v[0]["original_reference_paths"] == ["Images/cup/52.jpg"]
    assert opens2v[0]["dataset_metadata"]["synthesis_flag"] == "False"
    assert opens2v[0]["dataset_metadata"]["class_label"] == ["Cup"]


@pytest.mark.parametrize("reference_count", [1, 2, 3, 4])
def test_reference_count_one_through_four_is_valid(tmp_path: Path, reference_count: int) -> None:
    records, errors = normalize_external_dataset(
        dataset_schema="videoxfun_test",
        input_json=tmp_path / "test.json",
        payload=[
            {
                "id": f"sample_{reference_count}",
                "prompt": "valid",
                "ref_images": [f"/refs/{index}.png" for index in range(reference_count)],
            }
        ],
    )
    assert not errors
    assert len(records[0]["reference_paths"]) == reference_count


@pytest.mark.parametrize("reference_count", [0, 5])
def test_reference_count_outside_supported_range_is_rejected(
    tmp_path: Path,
    reference_count: int,
) -> None:
    records, errors = normalize_external_dataset(
        dataset_schema="videoxfun_test",
        input_json=tmp_path / "test.json",
        payload=[
            {
                "id": "sample",
                "prompt": "valid",
                "ref_images": [f"/refs/{index}.png" for index in range(reference_count)],
            }
        ],
    )
    assert not records
    assert errors


@pytest.mark.parametrize("source_id", ["../escape", "a/b", "a\\b", "bad\x00id"])
def test_output_id_traversal_and_control_characters_are_rejected(
    tmp_path: Path,
    source_id: str,
) -> None:
    records, errors = normalize_external_dataset(
        dataset_schema="videoxfun_test",
        input_json=tmp_path / "test.json",
        payload=[{"id": source_id, "prompt": "valid", "ref_images": ["/r.png"]}],
    )
    assert not records
    assert errors


def test_empty_prompt_and_normalized_output_collision_are_reported(tmp_path: Path) -> None:
    records, errors = normalize_external_dataset(
        dataset_schema="videoxfun_test",
        input_json=tmp_path / "test.json",
        payload=[
            {"id": "empty", "prompt": " ", "ref_images": ["/r.png"]},
            {"id": "a b", "prompt": "one", "ref_images": ["/a.png"]},
            {"id": "a@b", "prompt": "two", "ref_images": ["/b.png"]},
        ],
    )
    assert len(records) == 2
    assert {error["error_type"] for error in errors} == {
        "ExternalEvalSchemaError",
        "NormalizedOutputIdCollision",
    }


def test_stable_seed_is_invariant_to_manifest_order_and_dataset_specific() -> None:
    first = stable_sample_seed(42, "videoxfun_test", "same")
    reordered = stable_sample_seed(42, "videoxfun_test", "same")
    other_dataset = stable_sample_seed(42, "custom64", "same")
    assert first == reordered
    assert first != other_dataset


def test_center_crop_risk_matches_current_transform_geometry() -> None:
    square = center_crop_risk(1024, 1024)
    portrait = center_crop_risk(720, 1280)
    widescreen = center_crop_risk(1664, 960)
    assert square["vertical_crop_risk"]
    assert square["square_to_wide_risk"]
    assert portrait["severe_crop_risk"]
    assert portrait["portrait_to_wide_risk"]
    assert not widescreen["vertical_crop_risk"]
    assert not widescreen["horizontal_crop_risk"]
