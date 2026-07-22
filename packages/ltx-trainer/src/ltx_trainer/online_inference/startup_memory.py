"""Host-memory estimation and observation for semantic-flow startup."""

from __future__ import annotations

import math
import os
import resource
import sys
from pathlib import Path
from typing import Any

from safetensors import safe_open


class SemanticFlowStartupMemoryError(RuntimeError):
    """Semantic-flow startup would exceed the configured host-memory safety margin."""


def model_parameter_count(path: str | Path) -> int:
    resolved = Path(path).expanduser().resolve()
    with safe_open(resolved, framework="pt", device="cpu") as handle:
        return sum(math.prod(handle.get_slice(key).get_shape()) for key in handle.keys())


def available_host_ram_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    return page_size * available_pages


def host_memory_snapshot() -> dict[str, int]:
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak_rss *= 1024
    current_rss = peak_rss
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                current_rss = int(line.split()[1]) * 1024
                break
    return {
        "available_host_ram_bytes": available_host_ram_bytes(),
        "process_rss_bytes": current_rss,
        "process_peak_rss_bytes": peak_rss,
    }


def build_startup_host_ram_report(
    *,
    parameter_count: int,
    num_processes: int,
    training_dtype: str,
    available_bytes: int | None = None,
) -> dict[str, Any]:
    dtype_bytes = {
        "bf16": 2,
        "bfloat16": 2,
        "fp16": 2,
        "float16": 2,
        "fp32": 4,
        "float32": 4,
    }.get(training_dtype.lower())
    if dtype_bytes is None:
        raise SemanticFlowStartupMemoryError(f"Unsupported training dtype for RAM estimate: {training_dtype}")
    if parameter_count < 1 or num_processes < 1:
        raise SemanticFlowStartupMemoryError("Parameter count and num_processes must be positive")
    available = available_host_ram_bytes() if available_bytes is None else int(available_bytes)
    estimated = math.ceil(parameter_count * num_processes * (dtype_bytes + 4) * 1.2)
    margin_ratio = (available - estimated) / available if available > 0 else float("-inf")
    safe = available > 0 and estimated <= available * 0.9
    return {
        "model_parameter_count": parameter_count,
        "num_processes": num_processes,
        "training_dtype": training_dtype,
        "per_parameter_training_dtype_bytes": dtype_bytes,
        "per_parameter_fp32_conversion_bytes": 4,
        "loading_buffer_ratio": 0.2,
        "estimated_startup_host_ram_bytes": estimated,
        "available_host_ram_bytes": available,
        "startup_ram_margin_ratio": margin_ratio,
        "startup_ram_safe": safe,
    }


def enforce_startup_host_ram_report(report: dict[str, Any]) -> None:
    if report.get("startup_ram_safe") is not True:
        raise SemanticFlowStartupMemoryError(
            "Estimated semantic-flow startup RAM exceeds 90% of available host RAM; "
            "validate cpu_ram_efficient_loading=true with rank0 load plus meta initialization before retrying"
        )


__all__ = [
    "SemanticFlowStartupMemoryError",
    "available_host_ram_bytes",
    "build_startup_host_ram_report",
    "enforce_startup_host_ram_report",
    "host_memory_snapshot",
    "model_parameter_count",
]
