"""VRAM monitoring utilities using torch.cuda."""

import torch


def has_cuda() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def get_vram_info() -> dict:
    """Return dict with total, used, free VRAM in GB for the current device."""
    if not has_cuda():
        return {"total": 0.0, "used": 0.0, "free": 0.0}

    total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    free = total - reserved

    return {
        "total": round(total, 2),
        "used": round(allocated, 2),
        "free": round(free, 2),
    }


def get_vram_usage_string() -> str:
    """Return human-readable VRAM usage string like '8.2 / 24.0 GB'."""
    if not has_cuda():
        return "No CUDA device"

    info = get_vram_info()
    return f"{info['used']:.1f} / {info['total']:.1f} GB"
