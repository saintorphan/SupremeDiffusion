"""Sampler/scheduler mapping from A1111-style names to diffusers scheduler classes."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lazy imports — these are only needed when actually creating schedulers
_SAMPLER_MAP: dict[str, tuple[str, dict[str, Any]]] = {
    "Euler": ("EulerDiscreteScheduler", {}),
    "Euler a": ("EulerAncestralDiscreteScheduler", {}),
    "LMS": ("LMSDiscreteScheduler", {}),
    "Heun": ("HeunDiscreteScheduler", {}),
    "DPM2": ("KDPM2DiscreteScheduler", {}),
    "DPM2 a": ("KDPM2AncestralDiscreteScheduler", {}),
    "DPM++ 2S a": ("DPMSolverSinglestepScheduler", {}),
    "DPM++ 2M": ("DPMSolverMultistepScheduler", {}),
    "DPM++ SDE": ("DPMSolverSDEScheduler", {}),
    "DPM++ 2M SDE": ("DPMSolverMultistepScheduler", {"algorithm_type": "sde-dpmsolver++"}),
    "DPM++ 3M SDE": ("DPMSolverMultistepScheduler", {"algorithm_type": "sde-dpmsolver++", "solver_order": 3}),
    "DDIM": ("DDIMScheduler", {}),
    "DDPM": ("DDPMScheduler", {}),
    "UniPC": ("UniPCMultistepScheduler", {}),
    "PNDM": ("PNDMScheduler", {}),
    "LCM": ("LCMScheduler", {}),
}

_SCHEDULER_OVERRIDES: dict[str, dict[str, Any]] = {
    "Automatic": {},
    "Karras": {"use_karras_sigmas": True},
    "Exponential": {"use_exponential_sigmas": True},
}

# Every sigma-spacing key any override can set. We strip these from an inherited
# scheduler config before re-applying the current selection, otherwise a prior
# "Karras"/"Exponential" gen leaves its flag set when the user switches back to
# "Automatic" (the empty override would never reset it).
_SCHEDULER_OVERRIDE_KEYS: set[str] = {
    k for overrides in _SCHEDULER_OVERRIDES.values() for k in overrides
}


def _import_scheduler_class(class_name: str):
    """Dynamically import a scheduler class from diffusers."""
    import diffusers
    cls = getattr(diffusers, class_name, None)
    if cls is None:
        raise ImportError(f"diffusers does not have scheduler class '{class_name}'")
    return cls


def list_samplers() -> list[str]:
    """Return available sampler names."""
    return list(_SAMPLER_MAP.keys())


def list_schedulers() -> list[str]:
    """Return available scheduler variant names."""
    return list(_SCHEDULER_OVERRIDES.keys())


def create_scheduler(sampler_name: str, scheduler_name: str, config: dict | None = None) -> Any:
    """Create a diffusers scheduler from A1111-style sampler + scheduler names.

    Parameters
    ----------
    sampler_name:
        One of the keys in SAMPLER_MAP (e.g. "DPM++ 2M").
    scheduler_name:
        One of "Automatic", "Karras", "Exponential".
    config:
        Optional base scheduler config dict (from pipe.scheduler.config).
    """
    if sampler_name not in _SAMPLER_MAP:
        logger.warning("Unknown sampler '%s', falling back to 'Euler'", sampler_name)
        sampler_name = "Euler"

    if scheduler_name not in _SCHEDULER_OVERRIDES:
        logger.warning("Unknown scheduler '%s', falling back to 'Automatic'", scheduler_name)
        scheduler_name = "Automatic"

    class_name, extra_kwargs = _SAMPLER_MAP[sampler_name]
    scheduler_overrides = _SCHEDULER_OVERRIDES[scheduler_name]

    cls = _import_scheduler_class(class_name)

    # Build kwargs: start from pipeline's existing scheduler config if available
    kwargs: dict[str, Any] = {}
    if config:
        kwargs.update(config)

    # Drop any sigma-spacing flags inherited from a previous gen's scheduler so a
    # non-matching selection (e.g. "Automatic") resets them instead of leaking.
    for k in _SCHEDULER_OVERRIDE_KEYS:
        kwargs.pop(k, None)

    # Remove keys that don't apply to this scheduler class
    # (e.g. use_karras_sigmas on schedulers that don't support it)
    kwargs.update(extra_kwargs)
    kwargs.update(scheduler_overrides)

    # Clean up incompatible keys
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    # Also check from_config if available
    if hasattr(cls, "from_config"):
        try:
            sig2 = inspect.signature(cls.from_config)
            valid_params |= set(sig2.parameters.keys()) - {"self", "cls"}
        except (ValueError, TypeError):
            pass

    # If we have a config dict, use from_config pattern
    if config:
        clean_config = dict(config)
        # Reset inherited sigma-spacing flags before applying the current overrides.
        for k in _SCHEDULER_OVERRIDE_KEYS:
            clean_config.pop(k, None)
        clean_config.update(extra_kwargs)
        clean_config.update(scheduler_overrides)
        # Remove _class_name if present
        clean_config.pop("_class_name", None)
        clean_config.pop("_diffusers_version", None)
        try:
            return cls.from_config(clean_config)
        except Exception:
            pass

    # Fallback: direct instantiation with only valid params
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}
    return cls(**filtered)
