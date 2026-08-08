"""Worker for Wan 2.2 video LoRA training via ai-toolkit."""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path

import yaml

from .base import BaseWorker

logger = logging.getLogger(__name__)

# Search paths for ai-toolkit installation
_SEARCH_PATHS = [
    "third_party/ai-toolkit",
    "~/ai-toolkit",
    "~/Projects/ai-toolkit",
]


def find_ai_toolkit() -> str | None:
    """Find the ai-toolkit installation directory."""
    for p in _SEARCH_PATHS:
        d = Path(p).expanduser()
        if (d / "run.py").is_file():
            return str(d)
    return None


def _build_config(
    *,
    name: str,
    model_path: str,
    dataset_dir: str,
    output_dir: str,
    rank: int = 64,
    alpha: int = 32,
    lr: float = 1e-4,
    steps: int = 1000,
    batch_size: int = 1,
    grad_accum: int = 4,
    num_frames: int = 49,
    resolution: list[int] | None = None,
    train_high_noise: bool = True,
    train_low_noise: bool = True,
    sample_prompts: list[str] | None = None,
    fps: int = 16,
    low_vram: bool = True,
) -> dict:
    """Build an ai-toolkit YAML config dict for Wan 2.2 video LoRA training.

    ``low_vram=True`` quantizes (uint4 transformer + float8 TE) and offloads for
    12 GB cards; ``low_vram=False`` trains at higher precision (needs more VRAM).
    """
    if resolution is None:
        resolution = [480, 832]

    config = {
        "job": "extension",
        "config": {
            "name": name,
            "process": [{
                "type": "sd_trainer",
                "training_folder": output_dir,
                "device": "cuda:0",
                "network": {
                    "type": "lora",
                    "linear": rank,
                    "linear_alpha": alpha,
                },
                "save": {
                    "dtype": "float16",
                    "save_every": max(steps // 4, 100),
                    "max_step_saves_to_keep": 2,
                },
                "datasets": [{
                    "folder_path": dataset_dir,
                    "caption_ext": "txt",
                    "caption_dropout_rate": 0.05,
                    "num_frames": num_frames,
                    "resolution": resolution,
                }],
                "train": {
                    "batch_size": batch_size,
                    "steps": steps,
                    "gradient_accumulation": grad_accum,
                    "train_unet": True,
                    "train_text_encoder": False,
                    "gradient_checkpointing": True,
                    "noise_scheduler": "flowmatch",
                    "timestep_type": "linear",
                    "optimizer": "adamw8bit",
                    "lr": lr,
                    "optimizer_params": {
                        "weight_decay": 1e-4,
                    },
                    "switch_boundary_every": 10,
                    "cache_text_embeddings": True,
                    "dtype": "bf16",
                },
                "model": {
                    "name_or_path": model_path,
                    "arch": "wan22_14b",
                    "quantize": low_vram,
                    "qtype": ("uint4|ostris/accuracy_recovery_adapters/wan22_14b_t2i_torchao_uint4.safetensors"
                              if low_vram else "bf16"),
                    "quantize_te": low_vram,
                    "qtype_te": "qfloat8" if low_vram else "bf16",
                    "low_vram": low_vram,
                    "model_kwargs": {
                        "train_high_noise": train_high_noise,
                        "train_low_noise": train_low_noise,
                    },
                },
                "sample": {
                    "sampler": "flowmatch",
                    "sample_every": max(steps // 4, 100),
                    "width": resolution[1] if len(resolution) > 1 else 832,
                    "height": resolution[0],
                    "num_frames": num_frames,
                    "fps": fps,
                    "prompts": sample_prompts or [
                        f"{name} person talking naturally",
                        f"{name} person turning head",
                    ],
                    "neg": "",
                    "seed": 42,
                    "walk_seed": True,
                    "guidance_scale": 3.5,
                    "sample_steps": 25,
                },
            }],
        },
        "meta": {
            "name": name,
            "version": "1.0",
        },
    }
    return config


# Regex patterns for parsing ai-toolkit output
_STEP_RE = re.compile(r"step\s+(\d+)\s*/\s*(\d+)")
_LOSS_RE = re.compile(r"loss[:\s]+([0-9.e+-]+)", re.IGNORECASE)


class VideoLoRATrainWorker(BaseWorker):
    """Run Wan 2.2 video LoRA training via ai-toolkit subprocess.

    Emits ``finished_ok`` with a dict: {"high_noise": path, "low_noise": path}
    """

    def __init__(
        self,
        *,
        ai_toolkit_dir: str,
        name: str,
        model_path: str,
        dataset_dir: str,
        output_dir: str,
        hn_rank: int = 64,
        hn_alpha: int = 32,
        hn_lr: float = 1e-4,
        hn_steps: int = 1000,
        ln_rank: int = 64,
        ln_alpha: int = 32,
        ln_lr: float = 1e-4,
        ln_steps: int = 1000,
        batch_size: int = 1,
        grad_accum: int = 4,
        num_frames: int = 49,
        fps: int = 16,
        trigger_token: str = "",
        low_vram: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._toolkit_dir = ai_toolkit_dir
        self._name = name
        self._model_path = model_path
        self._dataset_dir = dataset_dir
        self._output_dir = output_dir
        self._hn = {"rank": hn_rank, "alpha": hn_alpha, "lr": hn_lr, "steps": hn_steps}
        self._ln = {"rank": ln_rank, "alpha": ln_alpha, "lr": ln_lr, "steps": ln_steps}
        self._batch_size = batch_size
        self._grad_accum = grad_accum
        self._num_frames = num_frames
        self._fps = fps
        self._trigger = trigger_token
        self._low_vram = low_vram

    def do_work(self) -> dict:
        results = {}

        # Phase 1: High-noise LoRA
        self.progress.emit(0.0, "Training high-noise LoRA...")
        hn_config = _build_config(
            name=f"{self._name}_high_noise",
            model_path=self._model_path,
            dataset_dir=self._dataset_dir,
            output_dir=str(Path(self._output_dir) / "high_noise"),
            rank=self._hn["rank"],
            alpha=self._hn["alpha"],
            lr=self._hn["lr"],
            steps=self._hn["steps"],
            batch_size=self._batch_size,
            grad_accum=self._grad_accum,
            num_frames=self._num_frames,
            fps=self._fps,
            train_high_noise=True,
            train_low_noise=False,
            sample_prompts=[f"{self._trigger} person talking"] if self._trigger else None,
            low_vram=self._low_vram,
        )
        hn_path = self._run_training(hn_config, phase="high_noise", phase_offset=0.0, phase_scale=0.5)
        if hn_path:
            results["high_noise"] = hn_path

        if self.is_aborted:
            return results

        # Phase 2: Low-noise LoRA
        self.progress.emit(0.5, "Training low-noise LoRA...")
        ln_config = _build_config(
            name=f"{self._name}_low_noise",
            model_path=self._model_path,
            dataset_dir=self._dataset_dir,
            output_dir=str(Path(self._output_dir) / "low_noise"),
            rank=self._ln["rank"],
            alpha=self._ln["alpha"],
            lr=self._ln["lr"],
            steps=self._ln["steps"],
            batch_size=self._batch_size,
            grad_accum=self._grad_accum,
            num_frames=self._num_frames,
            fps=self._fps,
            train_high_noise=False,
            train_low_noise=True,
            sample_prompts=[f"{self._trigger} person talking"] if self._trigger else None,
            low_vram=self._low_vram,
        )
        ln_path = self._run_training(ln_config, phase="low_noise", phase_offset=0.5, phase_scale=0.5)
        if ln_path:
            results["low_noise"] = ln_path

        self.progress.emit(1.0, "Training complete")
        return results

    def _run_training(self, config: dict, phase: str,
                      phase_offset: float, phase_scale: float) -> str | None:
        """Write config YAML, run ai-toolkit, return path to final LoRA."""
        output_dir = Path(config["config"]["process"][0]["training_folder"])
        output_dir.mkdir(parents=True, exist_ok=True)

        config_path = output_dir / f"{phase}_config.yaml"
        config_path.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")
        logger.info("Video LoRA training config written: %s", config_path)

        # Find python in ai-toolkit venv or use system
        toolkit = Path(self._toolkit_dir)
        venv_python = toolkit / ".venv" / "bin" / "python"
        if not venv_python.is_file():
            venv_python = toolkit / "venv" / "bin" / "python"
        if not venv_python.is_file():
            venv_python = Path(sys.executable)

        run_script = toolkit / "run.py"
        cmd = [str(venv_python), str(run_script), str(config_path)]
        logger.info("Starting training: %s", " ".join(cmd))

        total_steps = config["config"]["process"][0]["train"]["steps"]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(toolkit),
        )

        for line in proc.stdout:
            if self.is_aborted:
                proc.terminate()
                return None

            line = line.strip()
            if not line:
                continue

            # Parse step progress
            m = _STEP_RE.search(line)
            if m:
                step = int(m.group(1))
                frac = phase_offset + (step / total_steps) * phase_scale
                loss_str = ""
                lm = _LOSS_RE.search(line)
                if lm:
                    loss_str = f" loss={lm.group(1)}"
                self.progress.emit(
                    min(frac, phase_offset + phase_scale),
                    f"{phase}: step {step}/{total_steps}{loss_str}",
                )

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ai-toolkit training failed for {phase} (exit code {proc.returncode})")

        # Find the final LoRA file
        lora_files = sorted(output_dir.rglob("*.safetensors"), key=lambda p: p.stat().st_mtime)
        if lora_files:
            return str(lora_files[-1])

        logger.warning("No .safetensors output found in %s", output_dir)
        return None


# ───────────────────────────────────────────────────────────────────────────
# Image LoRA via ai-toolkit (Flux + Z-Image Turbo) — 12 GB-friendly presets.
# Thanks to Ostris for Z-Image Turbo LoRA support + the training adapter.
# NOTE: the arch strings + training-adapter reference below match recent
# ai-toolkit; if a future ai-toolkit renames them, adjust these constants.
# ───────────────────────────────────────────────────────────────────────────

AITK_ARCH = {"flux": "flux", "zimage": "z_image", "ltx": "ltx_2.3"}
ZIMAGE_TURBO_TRAINING_ADAPTER = "ostris/zimage_turbo_training_adapter"


def prepare_lowvram_dataset(dataset_dir: str, max_side: int = 512) -> str:
    """Return a copy of *dataset_dir* with images downscaled so the longest side
    is <= *max_side* (captions copied verbatim). Prevents VRAM spikes during
    latent caching on 12 GB cards. Returns the original dir if Pillow is missing.
    """
    try:
        from PIL import Image
    except Exception:
        return dataset_dir
    src = Path(dataset_dir)
    out = src.parent / f"{src.name}_{max_side}"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*"):
        f.unlink()
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    for img_path in sorted(src.iterdir()):
        if img_path.suffix.lower() in exts:
            try:
                im = Image.open(img_path).convert("RGB")
                w, h = im.size
                scale = max_side / max(w, h)
                if scale < 1.0:
                    im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                                   Image.LANCZOS)
                im.save(out / f"{img_path.stem}.png")
                cap = img_path.with_suffix(".txt")
                if cap.is_file():
                    (out / f"{img_path.stem}.txt").write_text(
                        cap.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                logger.debug("resize failed for %s", img_path, exc_info=True)
    return str(out)


def build_image_lora_config(
    *,
    name: str,
    arch: str,
    model_path: str,
    dataset_dir: str,
    output_dir: str,
    rank: int = 32,
    alpha: int = 16,
    lr: float = 1e-4,
    steps: int = 2000,
    resolutions: list[int] | None = None,
    training_adapter: str = "",
    low_vram: bool = True,
) -> dict:
    """ai-toolkit YAML config for an IMAGE LoRA (Flux / Z-Image Turbo).

    ``low_vram=True`` = the 12 GB preset (float8 quant on transformer + text
    encoder, low_vram, gradient checkpointing). ``low_vram=False`` = full bf16,
    no quantization/offload (more VRAM, faster).
    """
    resolutions = resolutions or [512, 768, 1024]
    model: dict = {"name_or_path": model_path, "arch": arch}
    if low_vram:
        model.update({
            "quantize": True, "qtype": "qfloat8",       # transformer → float8
            "quantize_te": True, "qtype_te": "qfloat8",  # text encoder → float8
            "low_vram": True,
        })
    else:
        model.update({"quantize": False, "low_vram": False})
    if training_adapter:
        # Z-Image Turbo needs Ostris's training adapter.
        model["model_kwargs"] = {"training_adapter": training_adapter}
    return {
        "job": "extension",
        "config": {
            "name": name,
            "process": [{
                "type": "sd_trainer",
                "training_folder": output_dir,
                "device": "cuda:0",
                "network": {"type": "lora", "linear": rank, "linear_alpha": alpha},
                "save": {"dtype": "bf16", "save_every": max(steps // 4, 500),
                         "max_step_saves_to_keep": 4},
                "datasets": [{
                    "folder_path": dataset_dir,
                    "caption_ext": "txt",
                    "caption_dropout_rate": 0.05,
                    "cache_latents_to_disk": True,
                    "resolution": resolutions,
                }],
                "train": {
                    "batch_size": 1,
                    "steps": steps,
                    "gradient_accumulation": 1,
                    "train_unet": True,
                    "train_text_encoder": False,
                    "gradient_checkpointing": low_vram,
                    "noise_scheduler": "flowmatch",
                    "timestep_type": "sigmoid",
                    "optimizer": "adamw8bit",
                    "lr": lr,
                    "optimizer_params": {"weight_decay": 1e-4},
                    "cache_text_embeddings": True,
                    "dtype": "bf16",
                },
                "model": model,
                "sample": {
                    "sampler": "flowmatch",
                    "sample_every": max(steps // 4, 500),
                    "width": 1024, "height": 1024,
                    "prompts": [name], "neg": "", "seed": 42, "walk_seed": True,
                    "guidance_scale": 4.0, "sample_steps": 9,
                },
            }],
        },
        "meta": {"name": name, "version": "1.0"},
    }


class ImageLoRAAIToolkitWorker(BaseWorker):
    """Train a single image LoRA (Flux / Z-Image Turbo) via ai-toolkit.

    Emits ``finished_ok`` with the path to the trained .safetensors.
    """

    def __init__(
        self,
        *,
        ai_toolkit_dir: str,
        name: str,
        family: str,            # "flux" | "zimage"
        model_path: str,
        dataset_dir: str,
        output_dir: str,
        rank: int = 32,
        lr: float = 1e-4,
        steps: int = 2000,
        resolutions: list[int] | None = None,
        low_vram: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._toolkit_dir = ai_toolkit_dir
        self._name = name
        self._family = family
        self._model_path = model_path
        self._dataset_dir = dataset_dir
        self._output_dir = output_dir
        self._rank = rank
        self._lr = lr
        self._steps = steps
        self._resolutions = resolutions
        self._low_vram = low_vram

    def do_work(self) -> str | None:
        arch = AITK_ARCH.get(self._family, "flux")
        adapter = ZIMAGE_TURBO_TRAINING_ADAPTER if self._family == "zimage" else ""
        dataset_dir = self._dataset_dir
        resolutions = self._resolutions
        if self._low_vram:
            # Pre-resize to max-side 512 + train at 512/768 to fit 12 GB.
            self.status.emit("Resizing dataset for low-VRAM training…")
            dataset_dir = prepare_lowvram_dataset(self._dataset_dir, max_side=512)
            resolutions = resolutions or [512, 768]
        else:
            resolutions = resolutions or [768, 1024]
        config = build_image_lora_config(
            name=self._name, arch=arch, model_path=self._model_path,
            dataset_dir=dataset_dir, output_dir=self._output_dir,
            rank=self._rank, alpha=max(1, self._rank // 2), lr=self._lr,
            steps=self._steps, resolutions=resolutions, training_adapter=adapter,
            low_vram=self._low_vram,
        )
        out = Path(self._output_dir)
        out.mkdir(parents=True, exist_ok=True)
        config_path = out / f"{self._name}_config.yaml"
        config_path.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")
        logger.info("Image LoRA (%s) config: %s", self._family, config_path)

        toolkit = Path(self._toolkit_dir)
        venv_python = toolkit / ".venv" / "bin" / "python"
        if not venv_python.is_file():
            venv_python = toolkit / "venv" / "bin" / "python"
        if not venv_python.is_file():
            venv_python = Path(sys.executable)
        cmd = [str(venv_python), str(toolkit / "run.py"), str(config_path)]
        self.status.emit(f"Training {self._family} LoRA via ai-toolkit…")
        logger.info("Starting image LoRA training: %s", " ".join(cmd))

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, cwd=str(toolkit))
        for line in proc.stdout:
            if self.is_aborted:
                proc.terminate()
                break
            line = line.strip()
            if not line:
                continue
            m = _STEP_RE.search(line)
            if m:
                step, total = int(m.group(1)), int(m.group(2))
                lm = _LOSS_RE.search(line)
                loss = f" loss={lm.group(1)}" if lm else ""
                self.progress.emit(step / total if total else 0.0,
                                   f"step {step}/{total}{loss}")
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ai-toolkit image training failed (exit {proc.returncode})")
        loras = sorted(out.rglob("*.safetensors"), key=lambda p: p.stat().st_mtime)
        return str(loras[-1]) if loras else None
