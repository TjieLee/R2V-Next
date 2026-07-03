from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CFGModeBatch:
    """Per-sample condition dropout decisions for factorized CFG experiments."""

    mode_id: torch.Tensor
    drop_text: torch.Tensor
    drop_ref: torch.Tensor
    drop_planner: torch.Tensor
    keep_full: torch.Tensor


_DEFAULT_PROBS = {
    "full": 0.7,
    "drop_text": 0.1,
    "drop_ref": 0.1,
    "drop_planner": 0.1,
}


def sample_cfg_modes(
    batch_size: int,
    probs: dict[str, float] | None = None,
    device: torch.device | None = None,
) -> CFGModeBatch:
    """Sample text/reference/planner dropout modes for a batch.

    This helper does not change the base LTX CFG formula. It only centralizes
    the condition-drop masks needed by future factorized CFG training/inference
    branches.
    """

    if probs is None:
        probs = _DEFAULT_PROBS

    names = ["full", "drop_text", "drop_ref", "drop_planner"]
    weights = torch.tensor([float(probs.get(name, 0.0)) for name in names], device=device)
    if torch.any(weights < 0):
        raise ValueError(f"CFG probabilities must be non-negative, got {probs}")
    total = weights.sum()
    if total <= 0:
        raise ValueError(f"At least one CFG probability must be positive, got {probs}")
    weights = weights / total

    mode_id = torch.multinomial(weights, batch_size, replacement=True)
    drop_text = mode_id == names.index("drop_text")
    drop_ref = mode_id == names.index("drop_ref")
    drop_planner = mode_id == names.index("drop_planner")
    keep_full = mode_id == names.index("full")

    return CFGModeBatch(
        mode_id=mode_id,
        drop_text=drop_text,
        drop_ref=drop_ref,
        drop_planner=drop_planner,
        keep_full=keep_full,
    )
