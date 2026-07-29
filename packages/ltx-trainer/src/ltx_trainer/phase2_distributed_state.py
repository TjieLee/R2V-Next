from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

PHASE2_DISTRIBUTED_STATE_FORMAT_VERSION = 2
PHASE2_OPTIMIZER_GROUP_NAMES = (
    "dit_semantic",
    "conditioning_bridge",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not read Phase 2 distributed-state manifest: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Phase 2 distributed-state manifest must be a JSON object: {path}"
        )
    return payload


def _load_rank_payload(path: Path) -> dict[str, Any]:
    try:
        try:
            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except TypeError:
            payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Phase 2 distributed optimizer state: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Phase 2 distributed optimizer state must be a mapping: {path}"
        )
    return payload


def _optimizer_group_names(
    optimizer_state_dict: dict[str, Any],
) -> list[str]:
    param_groups = optimizer_state_dict.get("param_groups")
    if not isinstance(param_groups, list):
        return []
    return [
        str(group.get("name", ""))
        for group in param_groups
        if isinstance(group, dict)
    ]


def validate_phase2_rank_payload(
    payload: dict[str, Any],
    *,
    path: Path,
    expected_step: int,
    expected_rank: int,
    expected_world_size: int,
) -> None:
    required_values = {
        "format_version": PHASE2_DISTRIBUTED_STATE_FORMAT_VERSION,
        "global_step": expected_step,
        "rank": expected_rank,
        "world_size": expected_world_size,
    }
    mismatches = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in required_values.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"Phase 2 distributed optimizer payload metadata mismatch "
            f"for {path}: {mismatches}"
        )
    group_names = payload.get("optimizer_group_names")
    if group_names != list(PHASE2_OPTIMIZER_GROUP_NAMES):
        raise RuntimeError(
            "Phase 2 distributed optimizer payload has invalid group names: "
            f"{group_names!r}"
        )
    optimizer_state_dict = payload.get("optimizer_state_dict")
    if not isinstance(optimizer_state_dict, dict):
        raise RuntimeError(
            f"Phase 2 distributed optimizer payload is missing optimizer state: {path}"
        )
    state_group_names = _optimizer_group_names(optimizer_state_dict)
    if state_group_names != list(PHASE2_OPTIMIZER_GROUP_NAMES):
        raise RuntimeError(
            "Phase 2 distributed optimizer state_dict has invalid group names: "
            f"{state_group_names!r}"
        )
    torch_rng_state = payload.get("torch_rng_state")
    if not isinstance(torch_rng_state, torch.Tensor):
        raise RuntimeError(
            f"Phase 2 distributed optimizer payload is missing torch RNG state: {path}"
        )
    cuda_rng_state = payload.get("cuda_rng_state")
    if cuda_rng_state is not None and not isinstance(cuda_rng_state, torch.Tensor):
        raise RuntimeError(
            f"Phase 2 distributed optimizer payload has invalid CUDA RNG state: {path}"
        )
    if "python_rng_state" not in payload:
        raise RuntimeError(
            f"Phase 2 distributed optimizer payload is missing Python RNG state: {path}"
        )
    if "numpy_rng_state" not in payload:
        raise RuntimeError(
            f"Phase 2 distributed optimizer payload is missing NumPy RNG state: {path}"
        )
    if "grad_scaler_state" not in payload:
        raise RuntimeError(
            f"Phase 2 distributed optimizer payload is missing GradScaler state: {path}"
        )


def validate_phase2_distributed_state(
    state_path: Path,
    *,
    expected_step: int,
    expected_world_size: int | None = None,
    validate_rank_payloads: bool,
    rank_to_validate: int | None = None,
) -> dict[str, Any]:
    state_path = state_path.expanduser().resolve()
    if not state_path.is_dir():
        raise RuntimeError(
            f"Phase 2 distributed optimizer state is missing: {state_path}"
        )
    manifest_path = state_path / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Phase 2 distributed optimizer manifest is missing: {manifest_path}"
        )
    manifest = _load_json_object(manifest_path)
    if manifest.get("format_version") != PHASE2_DISTRIBUTED_STATE_FORMAT_VERSION:
        raise RuntimeError(
            "Phase 2 distributed optimizer manifest has unsupported format_version: "
            f"{manifest.get('format_version')!r}"
        )
    if manifest.get("global_step") != expected_step:
        raise RuntimeError(
            "Phase 2 distributed optimizer manifest global_step mismatch: "
            f"expected={expected_step}, actual={manifest.get('global_step')!r}"
        )
    world_size = manifest.get("world_size")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise RuntimeError(
            "Phase 2 distributed optimizer manifest has invalid world_size: "
            f"{world_size!r}"
        )
    if expected_world_size is not None and world_size != expected_world_size:
        raise RuntimeError(
            "Phase 2 distributed optimizer world_size mismatch: "
            f"checkpoint={world_size}, runtime={expected_world_size}"
        )
    if rank_to_validate is not None and not 0 <= rank_to_validate < world_size:
        raise RuntimeError(
            "Phase 2 distributed optimizer validation rank is out of range: "
            f"rank={rank_to_validate}, world_size={world_size}"
        )
    if manifest.get("optimizer_group_names") != list(
        PHASE2_OPTIMIZER_GROUP_NAMES
    ):
        raise RuntimeError(
            "Phase 2 distributed optimizer manifest has invalid group names: "
            f"{manifest.get('optimizer_group_names')!r}"
        )

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RuntimeError(
            "Phase 2 distributed optimizer manifest is missing files"
        )
    expected_keys = {str(rank) for rank in range(world_size)}
    if set(files) != expected_keys:
        raise RuntimeError(
            "Phase 2 distributed optimizer manifest rank set mismatch: "
            f"expected={sorted(expected_keys)}, actual={sorted(files)}"
        )

    for rank in range(world_size):
        entry = files[str(rank)]
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"Phase 2 distributed optimizer file entry {rank} is invalid"
            )
        expected_name = f"optimizer_rank_{rank:05d}.pt"
        if entry.get("path") != expected_name:
            raise RuntimeError(
                "Phase 2 distributed optimizer file path mismatch: "
                f"rank={rank}, expected={expected_name}, actual={entry.get('path')!r}"
            )
        rank_path = state_path / expected_name
        if not rank_path.is_file() or rank_path.stat().st_size <= 0:
            raise RuntimeError(
                f"Phase 2 distributed optimizer rank file is missing or empty: {rank_path}"
            )
        expected_size = entry.get("size_bytes")
        if expected_size is not None and expected_size != rank_path.stat().st_size:
            raise RuntimeError(
                f"Phase 2 distributed optimizer rank file size mismatch: {rank_path}"
            )
        validate_contents = (
            rank_to_validate is None or rank == rank_to_validate
        )
        if validate_contents:
            actual_sha = sha256_file(rank_path)
            if entry.get("sha256") != actual_sha:
                raise RuntimeError(
                    "Phase 2 distributed optimizer rank file SHA256 mismatch: "
                    f"{rank_path}"
                )
        if validate_rank_payloads and validate_contents:
            payload = _load_rank_payload(rank_path)
            validate_phase2_rank_payload(
                payload,
                path=rank_path,
                expected_step=expected_step,
                expected_rank=rank,
                expected_world_size=world_size,
            )

    return manifest


def load_phase2_rank_state(
    state_path: Path,
    *,
    expected_step: int,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    validate_phase2_distributed_state(
        state_path,
        expected_step=expected_step,
        expected_world_size=world_size,
        validate_rank_payloads=False,
        rank_to_validate=rank,
    )
    rank_path = state_path.expanduser().resolve() / (
        f"optimizer_rank_{rank:05d}.pt"
    )
    payload = _load_rank_payload(rank_path)
    validate_phase2_rank_payload(
        payload,
        path=rank_path,
        expected_step=expected_step,
        expected_rank=rank,
        expected_world_size=world_size,
    )
    return payload
