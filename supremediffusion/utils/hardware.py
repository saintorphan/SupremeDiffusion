"""Hardware detection + auto-tuned settings recommendations.

This module is import-safe in any environment: every optional import
(torch, xformers, sageattention, flash_attn) is guarded so detection never
raises. On a box with no CUDA / no torch it returns zeros/false and the
recommend_settings() fallback (smallest tier) kicks in.

``detect_hardware()`` returns a plain dict so it can be JSON-serialised and
passed across process boundaries. ``recommend_settings()`` maps the detected
hardware onto a dict keyed by ``GlobalConfig`` field names — consumed by the
Settings "Detect & Apply Recommended" button.
"""

from __future__ import annotations

import importlib.util


def _has_module(name: str) -> bool:
    """True if ``name`` is importable, without importing it. Never raises."""
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def detect_hardware() -> dict:
    """Probe the local machine. Never raises — returns safe defaults instead.

    Returns a dict with:
        vram_gb (float)      -- primary GPU VRAM in GB (0.0 if no CUDA)
        ram_gb (float)       -- system RAM in GB (0.0 if undetectable)
        compute_cap (str)    -- CUDA compute capability "X.Y" ("" if none)
        has_xformers (bool)
        has_sage (bool)
        has_flash (bool)
        bf16 (bool)          -- whether the GPU supports bf16
        gpu_name (str)
        cpu_count (int)      -- logical CPU count
    """
    has_xformers = _has_module("xformers")
    has_sage = _has_module("sageattention")
    has_flash = _has_module("flash_attn")

    vram_gb = 0.0
    ram_gb = 0.0
    compute_cap = ""
    bf16 = False
    gpu_name = ""
    cpu_count = 0

    # Logical CPU count (cheap, stdlib, never raises).
    try:
        import os

        cpu_count = os.cpu_count() or 0
    except Exception:
        cpu_count = 0

    # System RAM — prefer psutil, fall back to os.sysconf on POSIX.
    try:
        import psutil  # type: ignore

        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        try:
            import os

            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            ram_gb = round((pages * page_size) / (1024 ** 3), 1)
        except Exception:
            ram_gb = 0.0

    # GPU — torch + CUDA, all guarded.
    try:
        import torch

        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                vram_gb = round(props.total_memory / (1024 ** 3), 1)
                gpu_name = props.name
            except Exception:
                pass
            try:
                major, minor = torch.cuda.get_device_capability(0)
                compute_cap = f"{major}.{minor}"
            except Exception:
                major = 0
            try:
                bf16 = bool(torch.cuda.is_bf16_supported())
            except Exception:
                # bf16 is supported on Ampere (8.0+) and newer.
                try:
                    bf16 = compute_cap != "" and float(compute_cap) >= 8.0
                except Exception:
                    bf16 = False
    except Exception:
        pass

    return {
        "vram_gb": vram_gb,
        "ram_gb": ram_gb,
        "compute_cap": compute_cap,
        "has_xformers": has_xformers,
        "has_sage": has_sage,
        "has_flash": has_flash,
        "bf16": bf16,
        "gpu_name": gpu_name,
        "cpu_count": cpu_count,
    }


def _pick_attention(hw: dict) -> str:
    """Choose the best installed attention backend for this GPU.

    Never returns "flash" unless flash_attn is importable. Prefers sage on
    Ampere (8.6) when sageattention is present, then xformers, then sdpa.
    """
    if hw.get("has_sage"):
        return "sage"
    if hw.get("has_xformers"):
        return "xformers"
    return "sdpa"


def recommend_settings(hw: dict | None = None) -> dict:
    """Recommend GlobalConfig settings tuned to the detected hardware.

    Returns a dict keyed by GlobalConfig field names:
        memory_profile (int 1-5)
        attention_mode (str)
        transformer_quantization (str)
        text_encoder_quantization (str)
        vae_precision (str "16"/"32")
        vae_tiling (str "Auto"/"Disabled"/"256"/"128")
        boost (int)
        preload_in_vram (int MB)
        mixed_precision (str "0"/"1")
    """
    if hw is None:
        hw = detect_hardware()

    vram = float(hw.get("vram_gb") or 0.0)
    bf16 = bool(hw.get("bf16"))
    attention = _pick_attention(hw)

    if vram <= 8:
        return {
            "memory_profile": 5,
            "attention_mode": attention,
            "transformer_quantization": "int8",
            "text_encoder_quantization": "int8",
            "vae_precision": "16",
            "vae_tiling": "256",
            "boost": 2,
            "preload_in_vram": 0,
            "mixed_precision": "0",
        }
    if vram <= 13:
        # THIS BOX — RTX 3080 Ti 12GB, Ampere 8.6.
        return {
            "memory_profile": 4,
            "attention_mode": attention,
            "transformer_quantization": "int8",
            "text_encoder_quantization": "int8",
            "vae_precision": "16",
            "vae_tiling": "Auto",
            "boost": 2,
            "preload_in_vram": 0,
            "mixed_precision": "0",
        }
    if vram <= 17:
        return {
            "memory_profile": 3,
            "attention_mode": attention,
            "transformer_quantization": "bf16" if bf16 else "int8",
            "text_encoder_quantization": "bf16",
            "vae_precision": "16",
            "vae_tiling": "Auto",
            "boost": 1,
            "preload_in_vram": 2000,
            "mixed_precision": "0",
        }
    if vram <= 25:
        return {
            "memory_profile": 2,
            "attention_mode": attention,
            "transformer_quantization": "bf16",
            "text_encoder_quantization": "bf16",
            "vae_precision": "32",
            "vae_tiling": "Disabled",
            "boost": 1,
            "preload_in_vram": 6000,
            "mixed_precision": "0",
        }
    return {
        "memory_profile": 1,
        "attention_mode": attention,
        "transformer_quantization": "bf16",
        "text_encoder_quantization": "bf16",
        "vae_precision": "32",
        "vae_tiling": "Disabled",
        "boost": 1,
        "preload_in_vram": 12000,
        "mixed_precision": "0",
    }
