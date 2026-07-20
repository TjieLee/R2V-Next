from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
import torch

from ltx_trainer.online_inference.read_only_sources import ReadOnlySourcePolicy


def test_read_only_policy_allows_only_declared_sources_and_owned_writes(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    writable_root = tmp_path / "owned"
    source_root.mkdir()
    listed = source_root / "listed.json"
    listed.write_text('{"ok": true}', encoding="utf-8")
    undeclared = tmp_path / "peer" / "hidden.json"
    undeclared.parent.mkdir()
    undeclared.write_text("{}", encoding="utf-8")
    policy = ReadOnlySourcePolicy(
        writable_root=writable_root,
        allowed_files=frozenset({listed}),
    )
    assert policy.read_json(listed) == {"ok": True}
    with pytest.raises(PermissionError, match="not allowlisted"):
        policy.read_json(undeclared)
    with pytest.raises(PermissionError, match="writes must stay"):
        policy.atomic_write_text(source_root / "forbidden.txt", "no")
    output = policy.atomic_write_json(writable_root / "reports" / "ok.json", {"ok": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}
    assert not (source_root / "forbidden.txt").exists()


def test_open_s2v_root_read_does_not_make_the_root_writable(tmp_path: Path) -> None:
    source_root = tmp_path / "OpenS2V"
    writable_root = tmp_path / "owned"
    image = source_root / "Images" / "one.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    policy = ReadOnlySourcePolicy(
        writable_root=writable_root,
        allowed_roots=(source_root,),
    )
    assert policy.assert_read_path(image) == image.resolve()
    with pytest.raises(PermissionError):
        policy.ensure_directory(source_root / "Results")
    assert not (source_root / "Results").exists()


def test_flat_export_rejects_links_from_source_tree(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    writable_root = tmp_path / "owned"
    source_root.mkdir()
    writable_root.mkdir()
    source_video = source_root / "reference.mp4"
    source_video.write_bytes(b"source")
    generated = writable_root / "samples" / "id" / "generated.mp4"
    generated.parent.mkdir(parents=True)
    generated.write_bytes(b"generated")
    policy = ReadOnlySourcePolicy(
        writable_root=writable_root,
        allowed_roots=(source_root,),
    )
    destination = writable_root / "Generated_Videos" / "id.mp4"
    with pytest.raises(PermissionError):
        policy.link_or_copy(source_video, destination)
    mode = policy.link_or_copy(generated, destination)
    assert mode in {"hardlink", "copy"}
    assert destination.read_bytes() == b"generated"


def test_write_primitives_are_confined_to_writable_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "sources"
    writable_root = tmp_path / "owned"
    source_root.mkdir()
    writable_root.mkdir()
    generated = writable_root / "generated.mp4"
    generated.write_bytes(b"video")
    policy = ReadOnlySourcePolicy(
        writable_root=writable_root,
        allowed_roots=(source_root,),
    )
    observed_writes: list[Path] = []
    original_open = Path.open
    original_mkdir = Path.mkdir
    original_replace = os.replace
    original_copy2 = shutil.copy2

    def assert_owned(value: str | Path) -> Path:
        path = Path(value).resolve()
        assert path == writable_root.resolve() or path.is_relative_to(writable_root.resolve())
        observed_writes.append(path)
        return path

    def guarded_open(path: Path, mode: str = "r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            assert_owned(path)
        return original_open(path, mode, *args, **kwargs)

    def guarded_mkdir(path: Path, *args, **kwargs):
        assert_owned(path)
        return original_mkdir(path, *args, **kwargs)

    def guarded_replace(source, destination):
        assert_owned(source)
        assert_owned(destination)
        return original_replace(source, destination)

    def force_copy_link(source, destination):
        assert_owned(source)
        assert_owned(destination)
        raise OSError("exercise copy fallback")

    def guarded_copy2(source, destination, *args, **kwargs):
        assert_owned(source)
        assert_owned(destination)
        return original_copy2(source, destination, *args, **kwargs)

    def guarded_torch_save(value, destination, *args, **kwargs):
        assert_owned(destination)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "mkdir", guarded_mkdir)
    monkeypatch.setattr(os, "replace", guarded_replace)
    monkeypatch.setattr(os, "link", force_copy_link)
    monkeypatch.setattr(shutil, "copy2", guarded_copy2)
    monkeypatch.setattr(torch, "save", guarded_torch_save)

    policy.atomic_write_json(writable_root / "report.json", {"ok": True})
    policy.link_or_copy(generated, writable_root / "Generated_Videos" / "id.mp4")
    assert observed_writes
    assert not list(source_root.iterdir())
