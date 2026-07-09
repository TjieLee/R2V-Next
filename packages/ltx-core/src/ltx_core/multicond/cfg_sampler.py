from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CFGModeBatch:
    """Per-sample condition dropout decisions for factorized CFG experiments.

    Multi-reference strategies use mutually-exclusive modes: ``full``,
    ``drop_text``, ``drop_siglip``, ``drop_ref_latents``, and
    ``drop_all``/``null``. The old ``drop_ref`` name is kept only as a
    compatibility property and means either visual branch or reference-latent
    dropout was selected.
    """

    mode_id: torch.Tensor
    drop_text: torch.Tensor
    drop_siglip: torch.Tensor
    drop_ref_latents: torch.Tensor
    drop_all: torch.Tensor
    keep_full: torch.Tensor

    @property
    def drop_ref(self) -> torch.Tensor:
        """Legacy compatibility for callers that still ask for drop_ref."""

        return self.drop_siglip | self.drop_ref_latents

    @property
    def drop_planner(self) -> torch.Tensor:
        """Deprecated alias for the null/drop_all branch."""

        return self.drop_all


_DEFAULT_PROBS = {
    "full": 0.7,
    "drop_text": 0.1,
    "drop_siglip": 0.1,
    "drop_ref_latents": 0.0,
    "drop_all": 0.1,
}


def sample_cfg_modes(
    batch_size: int,
    probs: dict[str, float] | None = None,
    device: torch.device | None = None,
) -> CFGModeBatch:
    """Sample split CFG dropout modes for a batch.

    ``drop_ref`` from older configs is mapped to ``drop_siglip`` only. It no
    longer also drops DiT reference latents; callers that want both branches
    should set both ``drop_siglip`` and ``drop_ref_latents`` explicitly.
    """

    if probs is None:
        probs = _DEFAULT_PROBS
    probs = dict(probs)

    drop_all = float(probs.get("drop_all", probs.get("null", probs.get("drop_planner", 0.0))))
    drop_siglip = float(probs.get("drop_siglip", probs.get("drop_ref", 0.0)))
    drop_ref_latents = float(probs.get("drop_ref_latents", 0.0))

    names = ["full", "drop_text", "drop_siglip", "drop_ref_latents", "drop_all"]
    weights = torch.tensor(
        [
            float(probs.get("full", 0.0)),
            float(probs.get("drop_text", 0.0)),
            drop_siglip,
            drop_ref_latents,
            drop_all,
        ],
        device=device,
    )
    if torch.any(weights < 0):
        raise ValueError(f"CFG probabilities must be non-negative, got {probs}")
    total = weights.sum()
    if total <= 0:
        raise ValueError(f"At least one CFG probability must be positive, got {probs}")
    weights = weights / total

    mode_id = torch.multinomial(weights, batch_size, replacement=True)
    drop_text = mode_id == names.index("drop_text")
    drop_siglip_mask = mode_id == names.index("drop_siglip")
    drop_ref_latents_mask = mode_id == names.index("drop_ref_latents")
    drop_all_mask = mode_id == names.index("drop_all")
    keep_full = mode_id == names.index("full")

    return CFGModeBatch(
        mode_id=mode_id,
        drop_text=drop_text,
        drop_siglip=drop_siglip_mask,
        drop_ref_latents=drop_ref_latents_mask,
        drop_all=drop_all_mask,
        keep_full=keep_full,
    )
