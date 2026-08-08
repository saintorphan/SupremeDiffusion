"""Native VACE MultiTalk pipeline — lip sync during Wan 2.2 diffusion."""

from __future__ import annotations

import gc
import logging
import os
import sys
from pathlib import Path
from typing import Callable

import torch

logger = logging.getLogger(__name__)

_APP_ROOT = Path(__file__).resolve().parent.parent.parent
_WAN2GP_DIR = Path("~/Wan2GP").expanduser()

def _ckpts_dir() -> Path:
    """Return the models directory, preferring GlobalConfig if available."""
    try:
        from supremediffusion.config.global_config import GlobalConfig
        cfg = GlobalConfig.load()
        models_root = cfg.model_paths.get("models_root", "") or str(_APP_ROOT / "models")
        return Path(models_root)
    except Exception:
        return _APP_ROOT / "models"

_MULTITALK_CKPT_NAME = "wan2.1_multitalk_14B_quanto_mbf16_int8.safetensors"
_VACE_CKPT_NAME = "wan2.1_Vace_14B_module_quanto_mbf16_int8.safetensors"


class NativeVaceMultitalkPipeline:
    """Manages VACE MultiTalk model lifecycle and inference.

    Uses Wan2GP's WanAny2V class internally for the complex VACE+MultiTalk
    generation logic, but manages lifecycle like other SupremeDiffusion pipelines.
    """

    def __init__(self) -> None:
        self._wan_any2v = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load the VACE MultiTalk pipeline."""
        if self._loaded:
            return

        logger.info("Loading VACE MultiTalk pipeline...")

        # Ensure Wan2GP is importable
        wan2gp_str = str(_WAN2GP_DIR)
        if wan2gp_str not in sys.path:
            sys.path.insert(0, wan2gp_str)

        # Need to set cwd for Wan2GP's relative path resolution
        orig_cwd = os.getcwd()
        os.chdir(wan2gp_str)

        try:
            # Suppress import errors from unneeded Wan2GP modules (scail, etc.)
            import importlib
            _orig_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else importlib.__import__

            def _safe_import(name, *args, **kwargs):
                try:
                    return _orig_import(name, *args, **kwargs)
                except (ModuleNotFoundError, ImportError) as e:
                    if any(m in name for m in ("smplfitter", "scail", "nlf", "pyannote", "steadydancer", "fantasytalking")):
                        import types
                        mod = types.ModuleType(name)
                        sys.modules[name] = mod
                        return mod
                    raise

            import builtins
            orig_bi = builtins.__import__
            builtins.__import__ = _safe_import

            try:
                from models.wan.any2video import WanAny2V
                from models.wan.configs import WAN_CONFIGS
                from models.wan.wan_handler import family_handler as WanHandler
            finally:
                builtins.__import__ = orig_bi

            import json
            from shared.utils import files_locator as fl

            # Use i2v_2_2_multitalk as base_model_type (recognized by test functions)
            # but our custom config JSON has VaceWanModel with VACE layers.
            # The config file is selected by base_model_type name, so we symlink.
            base_model_type = "i2v_2_2_multitalk"

            # Override the config to include VACE architecture
            # Copy our VACE config over the i2v_2_2_multitalk config temporarily
            import shutil
            config_dir = _WAN2GP_DIR / "models" / "wan" / "configs"
            target_config = config_dir / "i2v_2_2_multitalk.json"
            vace_config = config_dir / "vace_multitalk_14B_2_2.json"
            backup_config = config_dir / "i2v_2_2_multitalk.json.bak"

            # Backup original and replace with VACE version
            if not backup_config.exists():
                shutil.copy2(str(target_config), str(backup_config))
            shutil.copy2(str(vace_config), str(target_config))

            # Load the defaults JSON
            defaults_path = _WAN2GP_DIR / "defaults" / "vace_multitalk_14B_2_2.json"
            with open(defaults_path) as f:
                defaults = json.load(f)
            base_model_def = defaults.get("model", {})

            # Add extra fields via WanHandler
            extra = WanHandler.query_model_def(base_model_type, base_model_def)
            model_def = {**base_model_def, **extra}
            # Force VACE class flags so VACE module gets loaded
            model_def["vace_class"] = True
            model_def["parent_model_type"] = "vace_14B"

            # The config is the standard i2v-14B config
            cfg = WAN_CONFIGS["i2v-14B"]

            # Find the text encoder checkpoint — search common locations
            ckpts = _ckpts_dir()
            te_filename = None
            _te_candidates = [
                ckpts / "umt5-xxl" / "models_t5_umt5-xxl-enc-quanto_int8.safetensors",
                ckpts / "umt5-xxl" / "models_t5_umt5-xxl-enc-bf16.pth",
                _WAN2GP_DIR / "ckpts" / "umt5-xxl" / "models_t5_umt5-xxl-enc-quanto_int8.safetensors",
                _WAN2GP_DIR / "ckpts" / "umt5-xxl" / "models_t5_umt5-xxl-enc-bf16.pth",
                _WAN2GP_DIR / "ckpts" / "models_t5_umt5-xxl-enc-bf16.pth",
            ]
            for candidate in _te_candidates:
                if candidate.is_file():
                    te_filename = str(candidate)
                    break
            if not te_filename:
                te_filename = fl.locate_file(cfg.t5_checkpoint, error_if_none=False)
            if not te_filename:
                raise RuntimeError(
                    "T5 text encoder not found. Expected in "
                    f"{ckpts / 'umt5-xxl'} or ~/Wan2GP/ckpts/umt5-xxl/"
                )
            logger.info("Using T5 encoder: %s", te_filename)

            # Build model_filename list:
            # [base_high, base_low, multitalk_module, vace_module]
            # submodel_no_list: [1, 2, 0, 0]
            #   1 = high-noise transformer
            #   2 = low-noise transformer
            #   0 = shared module (loaded into both)

            # Find base Wan 2.2 i2v models
            _base_candidates_high = [
                ckpts / "wan2.2_image2video_14B_high_quanto_mbf16_int8.safetensors",
                _WAN2GP_DIR / "ckpts" / "wan2.2_image2video_14B_high_quanto_mbf16_int8.safetensors",
            ]
            _base_candidates_low = [
                ckpts / "wan2.2_image2video_14B_low_quanto_mbf16_int8.safetensors",
                _WAN2GP_DIR / "ckpts" / "wan2.2_image2video_14B_low_quanto_mbf16_int8.safetensors",
            ]

            base_high = next((str(p) for p in _base_candidates_high if p.is_file()), None)
            base_low = next((str(p) for p in _base_candidates_low if p.is_file()), None)

            if not base_high or not base_low:
                raise RuntimeError(
                    "Wan 2.2 i2v base models not found. Need both high and low noise "
                    f"int8 checkpoints in {ckpts}/ or ~/Wan2GP/ckpts/"
                )

            vace_file = str(ckpts / _VACE_CKPT_NAME)
            if not Path(vace_file).is_file():
                # Fallback to Wan2GP ckpts dir
                alt = _WAN2GP_DIR / "ckpts" / _VACE_CKPT_NAME
                if alt.is_file():
                    vace_file = str(alt)
                else:
                    raise RuntimeError(f"VACE module not found: {vace_file}")

            multitalk_file = str(ckpts / _MULTITALK_CKPT_NAME)
            if not Path(multitalk_file).is_file():
                alt = _WAN2GP_DIR / "ckpts" / _MULTITALK_CKPT_NAME
                if alt.is_file():
                    multitalk_file = str(alt)
                else:
                    raise RuntimeError(f"MultiTalk module not found: {multitalk_file}")

            # Dual transformer mode — both high and low noise models
            # Block swap handles VRAM via CPU offloading
            model_filenames = [base_high, base_low, multitalk_file, vace_file]
            submodel_no_list = [1, 2, 0, 0]

            logger.info("Loading VACE MultiTalk with:")
            logger.info("  Base HIGH: %s", base_high)
            logger.info("  Base LOW:  %s", base_low)
            logger.info("  MultiTalk: %s", multitalk_file)
            logger.info("  VACE:      %s", vace_file)

            self._wan_any2v, _pipe = WanHandler.load_model(
                model_filename=model_filenames,
                model_type="i2v",
                base_model_type=base_model_type,
                model_def=model_def,
                text_encoder_filename=te_filename,
                quantizeTransformer=True,
                dtype=torch.bfloat16,
                submodel_no_list=submodel_no_list,
            )

            # Apply mmgp memory profile — use profile 5 (most aggressive
            # CPU offloading) for 12 GB cards since VACE+MultiTalk is heavy
            from mmgp import offload
            offload.profile(_pipe, profile_no=5, verboseLevel=0)

            # Set mmgp shared state expected by generate()
            offload.shared_state["_attention"] = "sdpa"
            offload.shared_state["_radial"] = False
            offload.shared_state["_chipmunk"] = False

            # Set attributes that Wan2GP's generate() expects from the UI
            self._wan_any2v._interrupt = False
            self._wan_any2v.enable_RIFLEx = False

            # Disable skip-step cache — set to None so generate() skips it
            for attr in ('model', 'model2'):
                m = getattr(self._wan_any2v, attr, None)
                if m is not None:
                    m.cache = None

            # Restore the original config
            if backup_config.exists():
                shutil.copy2(str(backup_config), str(target_config))

            self._loaded = True
            logger.info("VACE MultiTalk pipeline ready.")
        finally:
            os.chdir(orig_cwd)

    def unload(self) -> None:
        """Free all GPU memory."""
        if self._wan_any2v is not None:
            try:
                del self._wan_any2v
            except Exception:
                pass
            self._wan_any2v = None

        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("VACE MultiTalk pipeline unloaded.")

    def run(
        self,
        source_video: str,
        audio_left: str,
        audio_right: str | None = None,
        output_path: str = "",
        prompt: str = "",
        resolution: str = "832x480",
        num_frames: int = 81,
        steps: int = 20,
        guidance_scale: float = 1.0,
        flow_shift: float = 5.0,
        seed: int = -1,
        speaker_bboxes: list[list[int]] | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> str:
        """Run VACE MultiTalk lip sync generation.

        Args:
            source_video: Path to input video (used as VACE control).
            audio_left: Path to audio for speaker 1 (or single speaker).
            audio_right: Optional path to audio for speaker 2.
            output_path: Where to write the output video.
            prompt: Text prompt (optional).
            resolution: Target resolution e.g. "832x480".
            num_frames: Number of frames to generate.
            steps: Denoising steps.
            guidance_scale: CFG scale (1.0 for VACE mode).
            flow_shift: Flow shift parameter.
            seed: Random seed (-1 for random).
            progress_callback: Optional (fraction, description) callback.

        Returns:
            Path to output video.
        """
        if not self._loaded:
            self.load()

        if progress_callback:
            progress_callback(0.05, "Preparing audio embeddings...")

        import numpy as np

        # Ensure Wan2GP importable
        wan2gp_str = str(_WAN2GP_DIR)
        if wan2gp_str not in sys.path:
            sys.path.insert(0, wan2gp_str)

        orig_cwd = os.getcwd()
        os.chdir(wan2gp_str)

        try:
            from models.wan.multitalk.multitalk import (
                get_full_audio_embeddings,
                get_window_audio_embeddings,
            )

            # Parse resolution
            w, h = resolution.split("x")
            width, height = int(w), int(h)

            # Process audio
            combination_type = "add" if audio_right else "para"
            full_audio_embs, sum_audio = get_full_audio_embeddings(
                audio_guide1=audio_left,
                audio_guide2=audio_right,
                combination_type=combination_type,
                num_frames=num_frames,
                fps=16,
                sr=16000,
                padded_frames_for_embeddings=0,
                min_audio_duration=num_frames / 16,
            )

            if progress_callback:
                progress_callback(0.10, "Running VACE MultiTalk generation...")

            # Get windowed embeddings for the transformer
            audio_proj = get_window_audio_embeddings(
                full_audio_embs,
                audio_start_idx=0,
                clip_length=num_frames,
                vae_scale=4,
                audio_window=5,
            )

            if seed < 0:
                import random
                seed = random.randint(0, 2**32 - 1)

            # Load source video frames as tensor for VACE control
            import torchvision.transforms.functional as TF
            from shared.utils.utils import get_video_info
            from PIL import Image
            import cv2

            cap = cv2.VideoCapture(source_video)
            frames = []
            while len(frames) < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                pil_img = pil_img.resize((width, height), Image.LANCZOS)
                tensor = TF.to_tensor(pil_img) * 2 - 1  # normalize to [-1, 1]
                frames.append(tensor)
            cap.release()

            if not frames:
                raise RuntimeError("Could not read frames from source video")

            # Pad if fewer frames than requested
            while len(frames) < num_frames:
                frames.append(frames[-1])

            input_video = torch.stack(frames, dim=1)  # (3, T, H, W)

            if progress_callback:
                progress_callback(0.15, f"Generating ({steps} steps)...")

            # input_frames: source video frames for VACE control (3, T, H, W)
            # input_masks: all ones = use source as structural reference
            input_masks = torch.ones(1, num_frames, height, width,
                                     device="cpu", dtype=torch.float32)

            # Callback for progress and step tracking
            _total_steps = [steps]
            def _callback(step, latent, is_init=False, override_num_inference_steps=None, **kw):
                if override_num_inference_steps:
                    _total_steps[0] = override_num_inference_steps
                if step is not None and step >= 0 and progress_callback:
                    frac = 0.15 + 0.75 * (step / max(_total_steps[0], 1))
                    progress_callback(min(frac, 0.90), f"Step {step}/{_total_steps[0]}")

            # Run generation through WanAny2V
            result = self._wan_any2v.generate(
                input_prompt=prompt or "A person talking",
                input_frames=input_video,
                input_masks=input_masks,
                width=width,
                height=height,
                frame_num=num_frames,
                shift=flow_shift,
                sample_solver="unipc",
                sampling_steps=steps,
                guide_scale=guidance_scale,
                seed=seed,
                audio_proj=audio_proj,
                speakers_bboxes=speaker_bboxes,
                VAE_tile_size=128,
                loras_slists={"phase1": [], "phase2": []},
                callback=_callback,
                audio_cfg_scale=3.0,
            )

            if progress_callback:
                progress_callback(0.90, "Saving video...")

            # generate() returns a dict {"x": video_tensor, ...}
            if isinstance(result, dict):
                video_tensor = result["x"]
            else:
                video_tensor = result

            if video_tensor is None:
                raise RuntimeError("Generation returned no video")

            from shared.utils.audio_video import save_video
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            import tempfile
            if sum_audio is not None:
                # Save video to temp, then mux audio
                tmp_vid = tempfile.NamedTemporaryFile(
                    suffix=".mp4", delete=False, prefix="vmt_"
                ).name
                save_video(video_tensor, tmp_vid, fps=16)

                # Save audio to temp wav
                import soundfile as sf
                tmp_aud = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, prefix="vmt_aud_"
                ).name
                sf.write(tmp_aud, sum_audio, 16000)

                # Mux video + audio
                import subprocess
                mux_result = subprocess.run([
                    "ffmpeg", "-y",
                    "-i", tmp_vid, "-i", tmp_aud,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", output_path,
                ], capture_output=True, text=True, timeout=60)

                if mux_result.returncode != 0:
                    # Fallback: just use the video without audio
                    logger.warning("Audio mux failed: %s", mux_result.stderr[-200:])
                    import shutil
                    shutil.move(tmp_vid, output_path)
                else:
                    Path(tmp_vid).unlink(missing_ok=True)

                Path(tmp_aud).unlink(missing_ok=True)
            else:
                save_video(video_tensor, output_path, fps=16)

            # Verify the file was created
            if not Path(output_path).is_file():
                raise RuntimeError(f"Output file not created: {output_path}")

            if progress_callback:
                progress_callback(1.0, "Done")

            return output_path
        finally:
            os.chdir(orig_cwd)
