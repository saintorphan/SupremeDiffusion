"""LoRA weight loading and management for Supreme Diffusion.

Supports phase-aware multiplier scheduling (matching wan2gp's
``loras_multipliers`` format) for Lightning models that swap LoRAs
between high-noise and low-noise denoising phases.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Multiplier parsing (wan2gp-compatible format)
# ----------------------------------------------------------------------


def parse_phase_multipliers(
    multiplier_str: str,
    num_loras: int,
    num_steps: int,
    num_phases: int = 1,
    switch_step: Optional[int] = None,
    switch_step2: Optional[int] = None,
) -> list[list[float]]:
    """Parse a wan2gp-style multiplier string into per-step scale lists.

    Format:
        Space-separated entries, one per LoRA.
        Each entry may contain semicolons to separate phase values.

        ``"1;0 0;1"`` → LoRA 0: [1.0 in phase1, 0.0 in phase2],
                          LoRA 1: [0.0 in phase1, 1.0 in phase2]

        ``"1.2"``     → LoRA 0: [1.2 for all steps]

    Returns
    -------
    List of per-step multiplier lists, one list per LoRA.
    Each inner list has ``num_steps`` elements.
    """
    if switch_step is None:
        switch_step = num_steps
    if switch_step2 is None:
        switch_step2 = num_steps

    # Parse entries — space-separated for per-LoRA, semicolons for per-phase
    entries = multiplier_str.strip().split() if multiplier_str.strip() else []

    # Fall back to comma-separated if no spaces (simple format: "1.0,0.8")
    if len(entries) <= 1 and "," in multiplier_str and ";" not in multiplier_str:
        entries = [e.strip() for e in multiplier_str.split(",") if e.strip()]

    # Pad/trim to match num_loras
    while len(entries) < num_loras:
        entries.append("1.0")
    entries = entries[:num_loras]

    result = []
    for entry in entries:
        phases = entry.split(";")
        if len(phases) == 1:
            # Constant across all steps
            try:
                val = float(phases[0])
            except ValueError:
                val = 1.0
            result.append([val] * num_steps)
        else:
            # Pad phases to match num_phases
            while len(phases) < num_phases:
                phases.append(phases[-1])
            phases = phases[:num_phases]

            phase_vals = []
            for p in phases:
                try:
                    phase_vals.append(float(p))
                except ValueError:
                    phase_vals.append(1.0)

            # Expand into per-step list
            per_step = []
            if num_phases >= 2:
                # Phase 1: steps 0 to switch_step-1
                per_step.extend([phase_vals[0]] * switch_step)
                if num_phases >= 3 and switch_step2 < num_steps:
                    # Phase 2: switch_step to switch_step2-1
                    per_step.extend([phase_vals[1]] * (switch_step2 - switch_step))
                    # Phase 3: switch_step2 to end
                    per_step.extend([phase_vals[2]] * (num_steps - switch_step2))
                else:
                    # Phase 2: switch_step to end
                    per_step.extend([phase_vals[1]] * (num_steps - switch_step))
            else:
                per_step = [phase_vals[0]] * num_steps

            # Ensure exact length
            per_step = per_step[:num_steps]
            while len(per_step) < num_steps:
                per_step.append(per_step[-1] if per_step else 1.0)
            result.append(per_step)

    return result


def compute_switch_step(
    timesteps: list[float],
    switch_threshold: float,
) -> int:
    """Find the step index where the timestep first drops to or below the threshold.

    Returns ``len(timesteps)`` if no timestep is at or below the threshold.
    """
    for i, t in enumerate(timesteps):
        if t <= switch_threshold:
            return i
    return len(timesteps)


class LoRAManager:
    """Discover, apply, and remove LoRA weights from a diffusers pipeline.

    Supports phase-aware multiplier scheduling via ``apply_loras_phased``.

    Parameters
    ----------
    lora_dir:
        Directory containing ``.safetensors`` LoRA files.
    """

    def __init__(self, lora_dir: str) -> None:
        self.lora_dir: Path = Path(lora_dir)
        self.loaded_loras: list[str] = []
        self._adapter_names: list[str] = []
        self._step_scales: list[list[float]] | None = None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_available(self) -> list[str]:
        """Return filenames of LoRA safetensors in *lora_dir* (recursive)."""
        if not self.lora_dir.is_dir():
            logger.warning("LoRA directory does not exist: %s", self.lora_dir)
            return []

        loras = sorted(
            f.name for f in self.lora_dir.rglob("*.safetensors")
            if f.is_file()
        )
        logger.info("Found %d LoRA file(s) in %s", len(loras), self.lora_dir)
        return loras

    # ------------------------------------------------------------------
    # Application (simple — constant multipliers, fused)
    # ------------------------------------------------------------------

    def apply_loras(
        self,
        pipe: Any,
        lora_names: list[str],
        multipliers: list[float],
    ) -> None:
        """Load and fuse one or more LoRAs into *pipe* with constant multipliers."""
        if len(lora_names) != len(multipliers):
            raise ValueError(
                f"lora_names ({len(lora_names)}) and multipliers "
                f"({len(multipliers)}) must have the same length."
            )

        has_load = hasattr(pipe, "load_lora_weights")
        has_fuse = hasattr(pipe, "fuse_lora")
        if not has_load or not has_fuse:
            logger.error(
                "LoRA check failed: type=%s, load_lora_weights=%s, fuse_lora=%s",
                type(pipe).__name__, has_load, has_fuse,
            )
            raise RuntimeError(
                f"The provided pipeline ({type(pipe).__name__}) does not support LoRA operations "
                f"(load_lora_weights={has_load}, fuse_lora={has_fuse}). "
                f"Ensure 'peft' is installed: pip install peft"
            )

        loaded_adapters: list[str] = []
        loaded_scales: list[float] = []

        for name, scale in zip(lora_names, multipliers):
            # Resolve file — may be in a subdirectory
            lora_path = self.lora_dir / name
            if not lora_path.is_file():
                matches = list(self.lora_dir.rglob(name))
                if matches:
                    lora_path = matches[0]

            if not lora_path.is_file():
                raise ValueError(
                    f"LoRA file not found: {name} in {self.lora_dir}. "
                    f"Available: {self.list_available()}"
                )

            try:
                logger.info("Loading LoRA '%s' (scale=%.3f) ...", name, scale)
                adapter_name = Path(name).stem.replace(".", "_").replace(" ", "_")
                pipe.load_lora_weights(
                    str(lora_path.parent),
                    weight_name=lora_path.name,
                    adapter_name=adapter_name,
                )
                loaded_adapters.append(adapter_name)
                loaded_scales.append(scale)
                self.loaded_loras.append(name)
                logger.info("LoRA '%s' loaded as adapter '%s'.", name, adapter_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to apply LoRA '{name}': {exc}"
                ) from exc

        if not loaded_adapters:
            return

        # Move only peft's LoRA adapter weights to the execution device.
        # mmgp manages base model offloading — we must NOT touch base weights
        # or mmgp's bookkeeping breaks.  Only move lora_A/lora_B params.
        device = pipe._execution_device
        moved = 0
        for attr in ("transformer", "transformer_2"):
            model = getattr(pipe, attr, None)
            if model is None:
                continue
            for name, param in model.named_parameters():
                if "lora_" in name and param.device != device:
                    param.data = param.data.to(device)
                    moved += 1
        if moved:
            logger.info("Moved %d LoRA parameters to %s", moved, device)

        self._adapter_names = loaded_adapters
        try:
            pipe.set_adapters(loaded_adapters, loaded_scales)
            logger.info(
                "Set %d LoRA adapter(s) (unfused): %s",
                len(loaded_adapters),
                list(zip(loaded_adapters, loaded_scales)),
            )
        except Exception as exc:
            logger.warning("set_adapters failed: %s — trying fuse fallback", exc)
            self._adapter_names = []
            try:
                pipe.fuse_lora(lora_scale=loaded_scales[0] if loaded_scales else 1.0)
            except Exception:
                logger.warning("Fuse fallback also failed — LoRAs may not take effect")

    # ------------------------------------------------------------------
    # Application (phased — per-step multipliers, unfused adapters)
    # ------------------------------------------------------------------

    def apply_loras_phased(
        self,
        pipe: Any,
        lora_names: list[str],
        step_scales: list[list[float]],
    ) -> None:
        """Load LoRAs as named adapters with per-step multiplier schedules.

        Unlike ``apply_loras``, this does NOT fuse weights — it loads them
        as PEFT adapters so multipliers can be changed each step via
        ``set_step_scales``.

        Parameters
        ----------
        pipe:
            A diffusers pipeline.
        lora_names:
            LoRA filenames to load.
        step_scales:
            Per-LoRA per-step scales from ``parse_phase_multipliers``.
        """
        if not hasattr(pipe, "load_lora_weights"):
            raise RuntimeError("Pipeline does not support load_lora_weights.")

        self._adapter_names = []
        self._step_scales = step_scales

        for i, name in enumerate(lora_names):
            lora_path = self.lora_dir / name
            if not lora_path.is_file():
                matches = list(self.lora_dir.rglob(name))
                if matches:
                    lora_path = matches[0]
            if not lora_path.is_file():
                raise ValueError(f"LoRA file not found: {name} in {self.lora_dir}")

            adapter_name = str(i)
            try:
                logger.info("Loading LoRA '%s' as adapter '%s' ...", name, adapter_name)
                pipe.load_lora_weights(
                    str(lora_path.parent),
                    weight_name=lora_path.name,
                    adapter_name=adapter_name,
                )
                self._adapter_names.append(adapter_name)
                self.loaded_loras.append(name)
                logger.info("LoRA '%s' loaded as adapter '%s'.", name, adapter_name)
            except Exception as exc:
                raise RuntimeError(f"Failed to load LoRA '{name}': {exc}") from exc

        # Move PEFT-attached LoRA params to the execution device. mmgp manages
        # base-model offloading — when it pulls a layer to GPU on-demand, the
        # PEFT-attached lora_A / lora_B modules don't follow, leaving them on
        # CPU and causing "Expected all tensors to be on the same device"
        # crashes inside the LoRA forward. Mirrors the same fix in
        # ``apply_loras``. Only touch lora_* params; do NOT move base weights
        # or mmgp's offload bookkeeping breaks.
        if self._adapter_names:
            device = pipe._execution_device
            moved = 0
            for attr in ("transformer", "transformer_2"):
                model = getattr(pipe, attr, None)
                if model is None:
                    continue
                for pname, param in model.named_parameters():
                    if "lora_" in pname and param.device != device:
                        param.data = param.data.to(device)
                        moved += 1
            if moved:
                logger.info("Moved %d phased-LoRA parameters to %s", moved, device)

        # Set initial adapter weights (step 0)
        self.set_step_scales(pipe, 0)

    def set_step_scales(self, pipe: Any, step: int) -> None:
        """Update adapter weights for the given denoising step."""
        if not self._adapter_names or self._step_scales is None:
            return

        weights = []
        for lora_idx, adapter in enumerate(self._adapter_names):
            scales = self._step_scales[lora_idx]
            s = step if step < len(scales) else len(scales) - 1
            weights.append(scales[s])

        try:
            pipe.set_adapters(self._adapter_names, adapter_weights=weights)
            logger.debug("Step %d: adapter weights = %s", step, weights)
        except Exception as exc:
            logger.warning("Failed to set adapter weights at step %d: %s", step, exc)

    @property
    def is_phased(self) -> bool:
        """Whether this manager is using phase-aware adapters."""
        return self._step_scales is not None and len(self._adapter_names) > 0

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def remove_loras(self, pipe: Any) -> None:
        """Un-fuse and unload all previously applied LoRAs from *pipe*."""
        if not self.loaded_loras:
            logger.info("No LoRAs to remove.")
            return

        try:
            if self._adapter_names:
                # Phased mode — delete adapters
                if hasattr(pipe, "delete_adapters"):
                    pipe.delete_adapters(self._adapter_names)
                elif hasattr(pipe, "unload_lora_weights"):
                    pipe.unload_lora_weights()
            else:
                # Fused mode
                if hasattr(pipe, "unfuse_lora"):
                    pipe.unfuse_lora()
                if hasattr(pipe, "unload_lora_weights"):
                    pipe.unload_lora_weights()

            removed = list(self.loaded_loras)
            self.loaded_loras.clear()
            self._adapter_names.clear()
            self._step_scales = None
            logger.info("Removed LoRA(s): %s", ", ".join(removed))

        except Exception as exc:
            logger.error("Error removing LoRAs: %s", exc)
            raise RuntimeError(f"Failed to remove LoRAs: {exc}") from exc
