"""Wan scheduler dispatch — maps UI sample_solver strings to scheduler instances.

Before this module existed, ``wan_i2v.py`` hardcoded ``FlowUniPCMultistepScheduler``
regardless of what the UI dropdown said. This dispatch fixes that wiring and
also adds the Lightning-recommended samplers (euler/beta, euler_a/beta) plus
several variants that work with Wan's flow-matching architecture.

Key supported names (case-insensitive):
- ``unipc`` (default) — local FlowUniPCMultistepScheduler, fast + reliable
- ``euler`` — FlowMatchEulerDiscreteScheduler default
- ``euler/beta`` — Euler + use_beta_sigmas. **Lightning-recommended.**
- ``euler/karras`` — Euler + Karras sigmas
- ``euler_a`` — Euler + stochastic_sampling (ancestral)
- ``euler_a/beta`` — Euler ancestral + beta sigmas. **Lightning-recommended alt.**
- ``heun`` — FlowMatchHeunDiscreteScheduler
- ``lcm`` — FlowMatchLCMScheduler (for AnimateLCM-style 4-step)
- ``lcm/beta`` — LCM + beta sigmas
- ``sa_solver`` / ``sa_solver/beta`` — SASolverScheduler in flow-matching mode
   (``use_flow_sigmas`` + the Wan ``flow_shift``); included for compatibility with
   Lightning recommendations. Use as an experiment, not a default.

Each scheduler takes the same Wan ``flow_shift`` parameter where applicable.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Display names → internal key. Used by the UI dropdown.
SAMPLER_CHOICES: tuple[str, ...] = (
    "unipc",
    "euler",
    "euler/beta",
    "euler/karras",
    "euler_a",
    "euler_a/beta",
    "heun",
    "lcm",
    "lcm/beta",
    "sa_solver",
    "sa_solver/beta",
)

DEFAULT_SAMPLER = "unipc"

LIGHTNING_RECOMMENDED = ("euler/beta", "euler_a/beta", "sa_solver/beta")


def build_scheduler(name: str, flow_shift: float = 1.0) -> Any:
    """Return a scheduler instance for ``name``.

    Falls back to UniPC if name is unrecognized.
    """
    key = (name or "").strip().lower()

    # --- Local UniPC (default) -----------------------------------------
    if key in ("", "unipc"):
        from supremediffusion.models.fm_solvers_unipc import FlowUniPCMultistepScheduler
        return FlowUniPCMultistepScheduler(shift=float(flow_shift))

    # --- FlowMatchEulerDiscreteScheduler family ------------------------
    if key in (
        "euler", "euler/beta", "euler/karras",
        "euler_a", "euler_a/beta",
    ):
        from diffusers import FlowMatchEulerDiscreteScheduler
        kwargs: dict[str, Any] = {"shift": float(flow_shift)}
        if "beta" in key:
            kwargs["use_beta_sigmas"] = True
        elif "karras" in key:
            kwargs["use_karras_sigmas"] = True
        if "euler_a" in key:
            # Stochastic sampling = ancestral-style — adds noise on each step.
            kwargs["stochastic_sampling"] = True
        return FlowMatchEulerDiscreteScheduler(**kwargs)

    # --- Heun --------------------------------------------------------
    if key == "heun":
        from diffusers import FlowMatchHeunDiscreteScheduler
        return FlowMatchHeunDiscreteScheduler(shift=float(flow_shift))

    # --- LCM ---------------------------------------------------------
    if key in ("lcm", "lcm/beta"):
        from diffusers import FlowMatchLCMScheduler
        kwargs = {"shift": float(flow_shift)}
        if "beta" in key:
            kwargs["use_beta_sigmas"] = True
        return FlowMatchLCMScheduler(**kwargs)

    # --- SA-Solver (experimental for Wan) -----------------------------
    # SASolver enables flow-matching via use_flow_sigmas + flow_shift (the same
    # shift Wan's other schedulers honor); plain prediction_type strings don't
    # cover flow models, so we route through use_flow_sigmas instead.
    if key in ("sa_solver", "sa_solver/beta"):
        from diffusers import SASolverScheduler
        kwargs = {"use_flow_sigmas": True, "flow_shift": float(flow_shift)}
        if "beta" in key:
            kwargs["use_beta_sigmas"] = True
        try:
            return SASolverScheduler(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SASolverScheduler init failed (%s) — falling back to UniPC", exc
            )

    # --- Fallback ----------------------------------------------------
    logger.warning("Unknown sample_solver %r — falling back to UniPC", name)
    from supremediffusion.models.fm_solvers_unipc import FlowUniPCMultistepScheduler
    return FlowUniPCMultistepScheduler(shift=float(flow_shift))
