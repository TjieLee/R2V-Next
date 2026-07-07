import torch
from torch import Tensor


def combine_factorized_cfg(
    *,
    full: Tensor,
    drop_text: Tensor | None = None,
    drop_ref: Tensor | None = None,
    drop_all: Tensor | None = None,
    drop_planner: Tensor | None = None,
    text_scale: float = 1.0,
    ref_scale: float = 1.0,
    all_scale: float | None = None,
    planner_scale: float = 1.0,
) -> Tensor:
    """Combine optional factorized CFG branches without changing LTX semantics.

    ``full`` is the fully conditioned prediction. ``drop_text`` removes text,
    ``drop_ref`` removes image/visual conditions, and ``drop_all`` is the
    null/unconditional branch. ``drop_planner`` is accepted only as a legacy
    alias for ``drop_all``; it is not an independent planner-only branch.
    """

    out = full
    if drop_text is not None:
        out = out + text_scale * (full - drop_text)
    if drop_ref is not None:
        out = out + ref_scale * (full - drop_ref)
    if drop_all is None:
        drop_all = drop_planner
    if all_scale is None:
        all_scale = planner_scale
    if drop_all is not None:
        out = out + all_scale * (full - drop_all)
    return out
