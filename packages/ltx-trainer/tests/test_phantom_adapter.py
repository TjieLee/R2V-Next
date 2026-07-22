from pathlib import Path

import pytest

from ltx_trainer.online_data.adapters import AdapterReject, PhantomAdapter


def _row(reference_paths: list[str]) -> dict:
    return {
        "video_path": "videos/target.mp4",
        "metadata": {"video_caption": "a subject walks through a room"},
        "cropped_ref_paths": reference_paths,
        "cross_pair": {"id": "pair-1"},
    }


def _config(**overrides: object) -> dict:
    values = {
        "video_field": "video_path",
        "caption_field": "metadata.video_caption",
        "reference_field": "cropped_ref_paths",
        "require_cross_pair": True,
        "max_reference_images": 4,
        "manifest_seed": 17,
    }
    values.update(overrides)
    return values


def test_phantom_nested_fields_defaults_and_ordered_dedup(tmp_path: Path) -> None:
    adapter = PhantomAdapter()
    canonical = adapter.normalize(
        _row(["a.png", "b.png", "a.png"]),
        row_id="sample-9",
        dataset_name="r2v_phantom",
        data_root=tmp_path,
        config=_config(),
    )

    assert canonical.adapter_name == "phantom"
    assert canonical.source_record_id == "sample-9"
    assert canonical.video_path == str((tmp_path / "videos/target.mp4").resolve())
    assert canonical.reference_paths == [
        str((tmp_path / "a.png").resolve()),
        str((tmp_path / "b.png").resolve()),
    ]
    assert canonical.crop_xyxy is None
    assert canonical.clip_start_frame == 0
    assert canonical.clip_end_frame is None


def test_phantom_reference_selection_is_capped_and_seeded(tmp_path: Path) -> None:
    adapter = PhantomAdapter()
    row = _row([f"ref-{index}.png" for index in range(9)])
    first = adapter.normalize(
        row,
        row_id="stable-id",
        dataset_name="r2v_phantom",
        data_root=tmp_path,
        config=_config(manifest_seed=99),
    )
    second = adapter.normalize(
        row,
        row_id="stable-id",
        dataset_name="r2v_phantom",
        data_root=tmp_path,
        config=_config(manifest_seed=99),
    )

    assert first.reference_paths == second.reference_paths
    assert len(first.reference_paths) == 4
    original_positions = [row["cropped_ref_paths"].index(Path(path).name) for path in first.reference_paths]
    assert original_positions == sorted(original_positions)
    assert first.metadata["raw_reference_count"] == 9
    assert first.metadata["selected_reference_count"] == 4


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda row: row.update({"video_path": ""}), "phantom_missing_target"),
        (lambda row: row["metadata"].update({"video_caption": ""}), "phantom_empty_caption"),
        (lambda row: row.update({"cropped_ref_paths": []}), "phantom_missing_reference"),
        (lambda row: row.update({"cross_pair": None}), "phantom_missing_cross_pair"),
    ],
)
def test_phantom_rejections_have_stable_reasons(mutation, reason: str, tmp_path: Path) -> None:
    row = _row(["ref.png"])
    mutation(row)
    with pytest.raises(AdapterReject) as error:
        PhantomAdapter().normalize(
            row,
            row_id="bad-row",
            dataset_name="r2v_phantom",
            data_root=tmp_path,
            config=_config(),
        )
    assert error.value.reason == reason
