import torch
from torch import Tensor


def combine_factorized_cfg(
    *,
    full: Tensor,
    drop_text: Tensor | None = None,
    drop_ref: Tensor | None = None,
    drop_planner: Tensor | None = None,
    text_scale: float = 1.0,
    ref_scale: float = 1.0,
    planner_scale: float = 1.0,
) -> Tensor:
    """Combine optional factorized CFG branches without changing LTX semantics.

    ``full`` is the fully conditioned prediction. Each ``drop_*`` tensor is the
    matching prediction with one condition family removed. When a branch is not
    supplied, that factor is skipped.
    """

    out = full
    if drop_text is not None:
        out = out + text_scale * (full - drop_text)
    if drop_ref is not None:
        out = out + ref_scale * (full - drop_ref)
    if drop_planner is not None:
        out = out + planner_scale * (full - drop_planner)
    return out
