"""R2V source adapter registry."""

from ltx_trainer.online_data.adapters.base import (
    AdapterReject,
    CanonicalR2VSource,
    R2VSourceAdapter,
    get_nested_field,
)
from ltx_trainer.online_data.adapters.opens2v import OpenS2VAdapter
from ltx_trainer.online_data.adapters.phantom import PhantomAdapter

R2V_ADAPTERS: dict[str, R2VSourceAdapter] = {
    "OpenS2VDataset": OpenS2VAdapter(),
    "PhantomDataset": PhantomAdapter(),
}


def get_r2v_adapter(dataset_type: str) -> R2VSourceAdapter:
    try:
        return R2V_ADAPTERS[dataset_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported R2V dataset_type={dataset_type!r}; expected one of {sorted(R2V_ADAPTERS)}"
        ) from exc


__all__ = [
    "R2V_ADAPTERS",
    "AdapterReject",
    "CanonicalR2VSource",
    "OpenS2VAdapter",
    "PhantomAdapter",
    "R2VSourceAdapter",
    "get_nested_field",
    "get_r2v_adapter",
]
