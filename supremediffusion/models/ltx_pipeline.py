"""LTX Video 2.3 inference pipeline wrapper for Supreme Diffusion.

Wraps the vendored LTX-2 ``LTX2`` class (sourced from Wan2GP), which
pre-builds all components (transformer, VAEs, text encoder, projection,
connector, upsampler) on CPU and exposes them as flat attributes. We then
apply ``mmgp.offload.profile`` so weights stream from CPU on demand —
this is what lets the 12 GB / 6 GB GPU users run 22B models.

Mirrors the WanI2VPipeline interface: load(), unload(), generate(),
is_loaded property.
"""

from __future__ import annotations

import gc
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None

# Bootstrap the vendored package — its __init__ adds the bundled shared/
# directory to sys.path so internal "from shared.utils import …" imports work.
try:
    import supremediffusion.models.ltx2  # noqa: F401
except ImportError:
    pass

try:
    from supremediffusion.models.ltx2.ltx2 import LTX2
except ImportError:
    LTX2 = None

try:
    from supremediffusion.models.ltx2.ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
except ImportError:
    TI2VidTwoStagesPipeline = None

try:
    from supremediffusion.models.ltx2.ltx_pipelines.distilled import DistilledPipeline
except ImportError:
    DistilledPipeline = None

try:
    from supremediffusion.models.ltx2.ltx_pipelines.ic_lora import ICLoraPipeline as RetakePipeline
except ImportError:
    RetakePipeline = None

try:
    from supremediffusion.models.ltx2.ltx_core.loader import (
        LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP,
    )
except ImportError:
    LoraPathStrengthAndSDOps = None
    LTXV_LORA_COMFY_RENAMING_MAP = None

try:
    from mmgp import offload as mmgp_offload
except ImportError:
    mmgp_offload = None


# Default model_def. The vendored LTX2 class reads a few keys from this:
# - ltx2_pipeline: "distilled" or "two_stage"
# - ltx2_rope_freqs_fp32: bool
# - ltx2_spatial_upscaler_file: filename (resolved through files_locator)
# - VAE_URLs, ltx2_hdr_scene_embeddings_file: optional, we leave unset
_DEFAULT_DISTILLED_DEF: dict = {"ltx2_pipeline": "distilled"}
_DEFAULT_TWO_STAGE_DEF: dict = {"ltx2_pipeline": "two_stage"}


class LTXVideoPipeline:
    """High-level wrapper around the vendored LTX2 class for SDQT."""

    def __init__(self, global_config: Any) -> None:
        self.config = global_config
        self.ltx2: Any = None        # the vendored LTX2 instance
        self._status_cb = None       # (frac, text) sink for pre-gen progress
        self._retake_pipe: Any = None
        self._loaded: bool = False
        self._last_audio_path: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def pipe(self) -> Any:
        """Backward-compat shim: code that used to read .pipe (the
        TI2VidTwoStagesPipeline) gets the underlying ltx2.pipeline instead."""
        return self.ltx2.pipeline if self.ltx2 is not None else None

    @property
    def last_audio_path(self) -> str | None:
        return self._last_audio_path

    @staticmethod
    def _build_static_mask_inputs(
        image_path: str,
        mask_path: str,
        num_frames: int,
        height: int,
        width: int,
    ):
        """Build (input_frames, input_masks) tensors for LTX inpaint LoRAs.

        The conditioning image is replicated across ``num_frames`` to form a
        "still video"; the painted mask is replicated the same way. Tensors
        come out in [T, C, H, W] layout with values in [-1, 1] for the video
        and [0, 1] for the binary mask — what the vendored ``masking_source``
        consumer expects.
        """
        from PIL import Image as PILImage
        import numpy as np

        # Source image → [T, 3, H, W] in [-1, 1]
        img = PILImage.open(image_path).convert("RGB").resize((width, height), PILImage.LANCZOS)
        img_arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3]
        img_arr = img_arr * 2.0 - 1.0
        img_t = torch.from_numpy(img_arr).permute(2, 0, 1).contiguous()  # [3, H, W]
        input_frames = img_t.unsqueeze(0).repeat(num_frames, 1, 1, 1)    # [T, 3, H, W]

        # Mask → [T, 1, H, W] in [0, 1]
        mask = PILImage.open(mask_path).convert("L").resize((width, height), PILImage.NEAREST)
        mask_arr = np.asarray(mask, dtype=np.float32) / 255.0  # [H, W]
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0).contiguous()  # [1, H, W]
        input_masks = mask_t.unsqueeze(0).repeat(num_frames, 1, 1, 1)  # [T, 1, H, W]

        return input_frames, input_masks

    @staticmethod
    def _conform_ltx_len(n: int) -> int:
        """Largest valid LTX conditioning frame count (1 + 8k) that is <= n."""
        if n <= 1:
            return 1
        return ((n - 1) // 8) * 8 + 1

    @staticmethod
    def _extract_control_maps(frames_thwc, control_type: str, progress_cb=None):
        """Convert RGB guide frames into structural control maps.

        ``frames_thwc`` is a torch tensor [T, H, W, C] in [0, 1]. Returns the
        same shape/range. ``raw`` is a no-op (caller handles it); ``canny`` uses
        cv2 directly (no model download); ``pose``/``depth`` route through
        ``controlnet_types.run_preprocessor`` (best-effort — may download models).
        """
        import numpy as _np
        import torch as _torch

        ctype = (control_type or "raw").strip().lower()
        arr = (frames_thwc.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8")  # [T,H,W,C]

        # Disk cache — depth/pose extraction is minutes of compute but depends
        # only on (input frames, control type). Key on a content hash so
        # re-running the SAME guide+type (e.g. iterating prompt/seed) reuses the
        # maps instead of re-extracting. Stored as uint8; clear by deleting
        # ~/.supremediffusion/cache/control_maps.
        import os as _os, hashlib as _hashlib
        cache_path = None
        try:
            _h = _hashlib.sha1(ctype.encode())
            _h.update(_np.ascontiguousarray(arr).tobytes())
            _cdir = _os.path.expanduser("~/.supremediffusion/cache/control_maps")
            _os.makedirs(_cdir, exist_ok=True)
            cache_path = _os.path.join(_cdir, _h.hexdigest() + ".npy")
            if _os.path.isfile(cache_path):
                cached = _np.load(cache_path)  # uint8 [T,H,W,C]
                if progress_cb is not None:
                    try:
                        progress_cb(0.2, f"Reusing cached {ctype} maps ({len(cached)} fr)")
                    except Exception:  # noqa: BLE001
                        pass
                logger.info("Reusing cached %s control maps: %s", ctype, cache_path)
                return _torch.from_numpy(cached.astype("float32") / 255.0)
        except Exception as _exc:  # noqa: BLE001
            logger.warning("Control-map cache lookup failed (%s) — extracting fresh", _exc)
            cache_path = None

        out = []
        if ctype == "canny":
            import cv2
            for f in arr:
                gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                out.append(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
        elif ctype in ("pose", "depth"):
            from PIL import Image as _Img
            from supremediffusion.models.controlnet_types import (
                run_preprocessor, clear_preprocessor_cache,
            )
            key = "openpose" if ctype == "pose" else "depth"
            total = len(arr)
            try:
                for i, f in enumerate(arr):
                    if progress_cb is not None:
                        try:
                            # Extraction occupies the first ~20% of the bar;
                            # denoising drives the rest via the step callback.
                            progress_cb((i + 1) / max(total, 1) * 0.2,
                                        f"Extracting {ctype} — {i + 1}/{total}")
                        except Exception:  # noqa: BLE001
                            pass
                    res = run_preprocessor(key, _Img.fromarray(f))
                    out.append(_np.asarray(res.convert("RGB").resize(
                        (f.shape[1], f.shape[0]), _Img.LANCZOS)))
            finally:
                # The detector stayed cached across all frames — release it now.
                clear_preprocessor_cache()
        else:
            raise ValueError(f"unknown control type: {control_type}")
        maps_u8 = _np.stack(out, axis=0).astype("uint8")  # [T,H,W,C]
        if cache_path is not None:
            try:
                _np.save(cache_path, maps_u8)
                logger.info("Cached %s control maps: %s", ctype, cache_path)
            except Exception as _exc:  # noqa: BLE001
                logger.warning("Failed to cache control maps (%s)", _exc)
        return _torch.from_numpy(maps_u8.astype("float32") / 255.0)

    @staticmethod
    def _build_control_conditioning(
        guidance_video,
        *,
        control_type: str,
        strength: float,
        start_seconds: float,
        ramp_frames: int,
        num_frames: int,
        fps: float,
        progress_cb=None,
    ):
        """Build a ``video_conditioning`` override list from a control guide.

        Returns ``(conditioning_list, log_messages)`` where each entry is
        ``(tensor[T, H, W, C] float in [0, 1], start_frame, strength)`` per the
        vendored pipeline contract: ``start_frame`` is a pixel index, each
        tensor's T must satisfy ``(T - 1) % 8 == 0`` (T == 1 always valid), and
        ``strength`` is a per-tuple scalar. Returns ``(None, msgs)`` when no
        usable guide is present.

        Positioning: the guide lands at ``start_frame = round(start_seconds *
        fps)``; frames before it and after its span free-run. An optional ramp
        fades strength over ``ramp_frames`` single-frame tuples at each span edge
        to avoid a hard structural snap.
        """
        import torch as _torch

        msgs: list[str] = []
        if guidance_video is None or not _torch.is_tensor(guidance_video):
            return None, msgs
        gv = guidance_video
        if gv.ndim != 4:
            return None, [f"control guide ndim={gv.ndim} unexpected; skipping"]
        # Normalize to [T, C, H, W] (load_guidance_video already returns this).
        if gv.shape[1] not in (1, 3) and gv.shape[-1] in (1, 3):
            gv = gv.permute(0, 3, 1, 2).contiguous()
        gv = gv.float().clamp(0.0, 1.0)
        avail = int(gv.shape[0])
        if avail <= 0:
            return None, ["control guide has 0 frames; skipping"]

        fps = float(fps) or 1.0
        start_frame = int(round(float(start_seconds) * fps))
        if start_frame < 0:
            start_frame = 0
        if start_frame > num_frames - 1:
            msgs.append(
                f"control start {start_seconds:.2f}s (frame {start_frame}) is "
                f"beyond clip end ({num_frames} fr); clamping to last frame"
            )
            start_frame = num_frames - 1
        remaining = num_frames - start_frame
        span = min(avail, remaining)
        if avail > remaining:
            msgs.append(
                f"control guide ({avail} fr) longer than remaining output "
                f"({remaining} fr) from frame {start_frame}; trimming tail"
            )
        if span <= 0:
            return None, msgs + ["no room for control guide; skipping"]

        # [T, C, H, W] -> [T, H, W, C] in [0, 1], trimmed to span, then maps.
        frames = gv[:span].permute(0, 2, 3, 1).contiguous()
        ctype = (control_type or "raw").strip().lower()
        if ctype and ctype != "raw":
            try:
                frames = LTXVideoPipeline._extract_control_maps(frames, ctype, progress_cb=progress_cb)
                msgs.append(f"control maps extracted: {ctype}")
            except Exception as exc:
                msgs.append(f"control extraction '{ctype}' failed ({exc}); using raw frames")

        strength = max(0.0, min(1.0, float(strength)))
        ramp = max(0, int(ramp_frames))
        ramp = min(ramp, span // 2)

        def _conform(t):
            tv = LTXVideoPipeline._conform_ltx_len(int(t.shape[0]))
            return t[:tv] if tv != t.shape[0] else t

        conditioning = []
        if ramp <= 0:
            seg = _conform(frames)
            conditioning.append((seg, start_frame, strength))
        else:
            for i in range(ramp):
                s = strength * (i + 1) / (ramp + 1)
                conditioning.append((frames[i:i + 1], start_frame + i, s))
            mid = frames[ramp:span - ramp]
            if int(mid.shape[0]) >= 1:
                mid_v = _conform(mid)
                conditioning.append((mid_v, start_frame + ramp, strength))
                gap = (span - 2 * ramp) - int(mid_v.shape[0])
                if gap > 0:
                    msgs.append(
                        f"control middle trimmed {gap} fr for LTX (T-1)%8==0; "
                        "those frames free-run"
                    )
            for j in range(ramp):
                s = strength * (ramp - j) / (ramp + 1)
                idx = span - ramp + j
                conditioning.append((frames[idx:idx + 1], start_frame + idx, s))

        if not conditioning:
            return None, msgs + ["control conditioning empty; skipping"]
        return conditioning, msgs

    @staticmethod
    def _resolve_gemma_path(gemma_root: str) -> str:
        """Return the .safetensors file path for the gemma model.

        ``gemma_root`` from config may be either the directory or the file.
        ``build_gemma_text_encoder`` requires the file. Prefer int8/quanto.
        """
        if not gemma_root or os.path.isfile(gemma_root):
            return gemma_root
        if not os.path.isdir(gemma_root):
            return gemma_root
        candidates = sorted(
            f for f in os.listdir(gemma_root)
            if f.endswith(".safetensors")
        )
        if not candidates:
            return gemma_root
        for f in candidates:
            if "int8" in f.lower() or "quanto" in f.lower():
                return os.path.join(gemma_root, f)
        return os.path.join(gemma_root, candidates[0])

    def _setup_file_locator(self, model_paths: dict) -> None:
        """Register every directory we know about with the LTX files_locator.

        Vendored LTX code uses ``files_locator`` to find component files by
        name (e.g. spatial upscaler) and to locate the gemma sub-folder by its
        canonical name. We register every parent directory of a configured
        ltx_* path AND every path's parent (so a symlink named
        ``gemma-3-12b-it-qat-q4_0-unquantized`` placed next to the gemma file
        will be discovered).
        """
        from shared.utils import files_locator as fl

        dirs: list[str] = []
        seen: set[str] = set()
        def _add(d: str) -> None:
            if d and d not in seen:
                seen.add(d)
                dirs.append(d)

        for key, p in model_paths.items():
            if not isinstance(p, str) or not p or not key.startswith("ltx_"):
                continue
            d = p if os.path.isdir(p) else os.path.dirname(p)
            _add(d)
            # Also add the parent dir, so a sibling like
            # "<parent>/gemma-3-12b-it-qat-q4_0-unquantized" is reachable.
            parent = os.path.dirname(d.rstrip(os.sep))
            _add(parent)
        if dirs:
            fl.set_checkpoints_paths(dirs)

    def load(self) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required but not installed.")
        if LTX2 is None:
            raise RuntimeError(
                "Vendored LTX2 not importable. Check supremediffusion/models/ltx2/."
            )
        if mmgp_offload is None:
            raise RuntimeError("mmgp library is required for LTX low-VRAM loading.")

        if self._loaded:
            self.unload()

        model_paths = self.config.model_paths
        transformer = model_paths.get("ltx_transformer", "")
        gemma_root = self._resolve_gemma_path(model_paths.get("ltx_text_encoder", ""))
        video_vae = model_paths.get("ltx_vae", "")
        audio_vae = model_paths.get("ltx_audio_vae", "")
        vocoder = model_paths.get("ltx_vocoder", "")
        projection = model_paths.get("ltx_text_projection", "")
        connector = model_paths.get("ltx_embeddings_connector", "")
        upsampler = model_paths.get("ltx_spatial_upsampler", "")

        required = {
            "ltx_transformer": transformer,
            "ltx_text_encoder": gemma_root,
            "ltx_vae": video_vae,
            "ltx_audio_vae": audio_vae,
            "ltx_vocoder": vocoder,
            "ltx_text_projection": projection,
            "ltx_embeddings_connector": connector,
            "ltx_spatial_upsampler": upsampler,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(
                f"Missing LTX model paths in config: {missing}. "
                "Set them in Settings → LTX Video Model Paths."
            )

        self._setup_file_locator(model_paths)

        # Build the model_def. Distilled vs two-stage based on quality_tier or
        # an explicit setting; default to distilled (matches the user's int8
        # distilled checkpoint).
        model_def = dict(_DEFAULT_DISTILLED_DEF)

        # Build the per-component checkpoint_paths dict that LTX2._init_models
        # consumes. The keys must match what the vendored ltx2_handler emits.
        config_path = os.path.join(
            os.path.dirname(__import__("supremediffusion.models.ltx2", fromlist=["__file__"]).__file__),
            "configs", "ltx2_22b_config.json",
        )
        checkpoint_paths = {
            "transformer": transformer,
            "video_vae": video_vae,
            "audio_vae": audio_vae,
            "vocoder": vocoder,
            "text_embedding_projection": projection,
            "text_embeddings_connector": connector,
            "spatial_upsampler": upsampler,
            "model_config": config_path,
        }

        prev_default_device = None
        try:
            prev_default_device = torch.get_default_device()
        except Exception:
            pass
        torch.set_default_device("cpu")

        # Resolve the user's precision settings into torch dtypes. Without
        # this LTX2() falls through to its hardcoded constructor defaults
        # (bf16 transformer / fp32 VAE) and the Settings dropdowns are stubs.
        vae_pref = str(getattr(self.config, "vae_precision", "32") or "32")
        # LTX's VAE overflows in fp16 on some frames -> NaN -> white flicker in
        # the decoded video. Use bf16 for the "16" setting: it has fp32's numeric
        # range (no overflow) at half fp32's memory, and matches the bf16
        # transformer. "32" still forces true fp32 for anyone who wants it.
        VAE_dtype = torch.bfloat16 if vae_pref == "16" else torch.float32
        dtype_policy = (
            getattr(self.config, "transformer_dtype_policy", "") or ""
        ).lower()
        if dtype_policy == "fp16":
            transformer_dtype = torch.float16
        elif dtype_policy == "bf16":
            transformer_dtype = torch.bfloat16
        else:  # "auto" or empty
            transformer_dtype = torch.bfloat16
        logger.info(
            "LTX precision: transformer=%s, VAE=%s",
            str(transformer_dtype).rsplit(".", 1)[-1],
            str(VAE_dtype).rsplit(".", 1)[-1],
        )

        try:
            logger.info("Loading LTX 2.3 components (CPU pre-build)...")
            self.ltx2 = LTX2(
                model_filename=transformer,
                model_type="ltx2_22B_distilled",
                base_model_type="ltx2_22B",
                model_def=model_def,
                dtype=transformer_dtype,
                VAE_dtype=VAE_dtype,
                text_encoder_filepath=gemma_root,
                checkpoint_paths=checkpoint_paths,
            )

            # Apply mmgp memory profile so weights stream from CPU on demand.
            memory_profile = int(getattr(self.config, "memory_profile", 4) or 4)
            modules = {
                "transformer": self.ltx2.model,
                "text_encoder": self.ltx2.text_encoder,
                "text_embedding_projection": self.ltx2.text_embedding_projection,
                "text_embeddings_connector": self.ltx2.text_embeddings_connector,
                "vae": self.ltx2.video_decoder,
                "video_encoder": self.ltx2.video_encoder,
                "audio_encoder": self.ltx2.audio_encoder,
                "audio_decoder": self.ltx2.audio_decoder,
                "vocoder": self.ltx2.vocoder,
                "spatial_upsampler": self.ltx2.spatial_upsampler,
            }
            try:
                mmgp_offload.profile(modules, profile_no=memory_profile, verboseLevel=1)
                logger.info("mmgp profile %s applied to LTX components.", memory_profile)
            except Exception as exc:
                logger.warning(
                    "mmgp.profile failed (%s) — falling back to direct .to(cuda).", exc,
                )
                if torch.cuda.is_available():
                    for name, m in modules.items():
                        try:
                            m.to("cuda")
                        except Exception as e:
                            logger.warning("Failed to move %s to cuda: %s", name, e)
        finally:
            if torch.cuda.is_available():
                torch.set_default_device("cuda")
            elif prev_default_device is not None:
                torch.set_default_device(prev_default_device)

        self._loaded = True
        logger.info("LTX 2.3 pipeline loaded with mmgp offload.")

    def set_interrupt(self, value: bool = True) -> None:
        """Set the vendored LTX2 interrupt flag so the worker can abort.

        ``LTX2.generate`` (and every inner pipeline) polls ``self._interrupt``
        at each step boundary and bails out returning ``None``. The SDQT worker
        calls this from its ``abort()`` path. Safe to call when nothing is
        loaded — it's a no-op until the next ``load()``.
        """
        if self.ltx2 is not None:
            try:
                self.ltx2._interrupt = bool(value)
            except Exception as exc:
                logger.warning("Failed to set LTX interrupt flag: %s", exc)

    def unload(self) -> None:
        """Release LTX models and free GPU memory."""
        self.ltx2 = None
        self._retake_pipe = None
        self._loaded = False

        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

        logger.info("LTX pipeline unloaded.")

    def _setup_loras(self, cfg: Any, num_steps: int) -> tuple:
        """Resolve the user-selected LoRAs and load them onto the transformer.

        Returns ``(loras_slists, loras_selected)`` to thread into
        ``LTX2.generate``. ``loras_selected`` is the list of resolved full LoRA
        file paths (also used by the vendored IC-LoRA detection); ``loras_slists``
        is the per-phase multiplier schedule built from ``loras_multipliers``
        (wan2gp ``"1;0"`` phase syntax supported).

        Mirrors the wgp.py orchestration the vendored handler assumes: the
        weights are fused onto the transformer via ``offload.load_loras_into_model``
        (using the comfy→LTX key-renaming preprocessor attached in ``LTX2``), and
        the schedule dict is consumed inside ``generate`` via ``update_loras_slists``.
        Replaces the old ``_scan_loras`` which applied *every* dir LoRA at 1.0 and
        whose result was never threaded into the call.
        """
        lora_names = list(getattr(cfg, "activated_loras", []) or [])
        if not lora_names:
            return None, None

        lora_dir = (self.config.model_paths or {}).get("ltx_lora_dir", "") or ""
        if not lora_dir or not Path(lora_dir).is_dir():
            logger.warning(
                "LTX LoRA(s) selected but ltx_lora_dir is unset/missing; skipping."
            )
            return None, None

        # The dropdown stores either a stem or a filename — resolve to full paths.
        available = {f.name: f for f in Path(lora_dir).iterdir()
                     if f.suffix.lower() in (".safetensors", ".pt", ".pth")}
        resolved: list[str] = []
        for name in lora_names:
            base = os.path.basename(str(name))
            match = None
            if base in available:
                match = available[base]
            else:
                for fname, fpath in available.items():
                    if fname.rsplit(".", 1)[0] == base:
                        match = fpath
                        break
            if match is not None:
                resolved.append(str(match))
            else:
                logger.warning("LTX LoRA '%s' not found in %s; skipping.", name, lora_dir)

        if not resolved:
            return None, None

        mult_str = getattr(cfg, "loras_multipliers", "") or ""
        num_phases = int(getattr(cfg, "guidance_phases", 1) or 1)
        switch_phase = int(getattr(cfg, "model_switch_phase", 1) or 1)

        loras_slists = None
        try:
            from shared.utils.loras_mutipliers import parse_loras_multipliers
            _, loras_slists, errors = parse_loras_multipliers(
                mult_str, len(resolved), int(num_steps),
                nb_phases=max(2, num_phases), model_switch_phase=switch_phase,
            )
            if errors:
                logger.warning("LTX LoRA multiplier parse error: %s — using 1.0", errors)
                loras_slists = None
        except Exception as exc:
            logger.warning("LTX LoRA multiplier parse failed (%s) — using 1.0", exc)
            loras_slists = None

        # Fuse the weights onto the transformer through mmgp (matches wgp.py).
        if mmgp_offload is not None and self.ltx2 is not None:
            try:
                trans = self.ltx2.model
                preprocess_sd = None
                if hasattr(trans, "preprocess_loras"):
                    model_type = getattr(self.ltx2, "model_type", "")
                    preprocess_sd = lambda sd, _t=trans, _mt=model_type: _t.preprocess_loras(_mt, sd)
                mmgp_offload.load_loras_into_model(
                    trans, resolved, lora_multi=None,
                    activate_all_loras=True, preprocess_sd=preprocess_sd,
                    verboseLevel=1,
                )
                logger.info("LTX LoRAs loaded: %s", [os.path.basename(p) for p in resolved])
            except Exception as exc:
                logger.warning("Failed to load LTX LoRAs onto transformer: %s", exc)

        return loras_slists, resolved

    @staticmethod
    def list_loras(lora_dir: str) -> list[str]:
        """Return list of available LTX LoRA filenames."""
        if not lora_dir:
            return []
        lora_path = Path(lora_dir)
        if not lora_path.is_dir():
            return []
        return sorted(
            f.name for f in lora_path.iterdir()
            if f.suffix.lower() in (".safetensors", ".pt", ".pth")
        )

    def generate(
        self,
        image: Any = None,
        prompt: str = "",
        negative_prompt: str = "",
        project_config: Any = None,
        last_frame: Any = None,
        guidance_video: Any = None,
        callback: Optional[Callable] = None,
        lora_step_callback: Optional[Callable] = None,
    ) -> list:
        """Generate video frames (and audio) from the LTX pipeline.

        Returns a list of PIL Images. Audio is saved to a temp WAV file
        accessible via ``self.last_audio_path``.
        """
        if not self._loaded or self.ltx2 is None:
            self.load()

        # Clear any sticky interrupt from a prior aborted run — the vendored
        # generate() only ever *reads* this flag, so without a reset a single
        # abort would permanently bail every subsequent gen.
        if self.ltx2 is not None:
            self.ltx2._interrupt = False

        # The vendored LTX attention layer reads its dispatch mode from
        # mmgp.offload.shared_state — Wan2GP sets this in wgp.py before
        # calling generate(). Mirror that here.
        if mmgp_offload is not None:
            attn = getattr(self.config, "attention_mode", "sdpa") or "sdpa"
            mmgp_offload.shared_state["_attention"] = attn

        from PIL import Image as PILImage

        cfg = project_config
        self._last_audio_path = None

        resolution = getattr(cfg, "resolution", "1216x704")
        try:
            w_s, h_s = resolution.split("x", 1)
            w_s = w_s.strip().split()[0]
            h_s = h_s.strip().split()[0]
            width, height = int(w_s), int(h_s)
        except (ValueError, AttributeError):
            width, height = 1216, 704

        num_frames = getattr(cfg, "video_length", 121)
        num_steps = getattr(cfg, "num_inference_steps", 8)
        guidance_scale = getattr(cfg, "guidance_scale", 1.0)
        alt_guidance_scale = float(getattr(cfg, "alt_guidance_scale", 1.0) or 1.0)
        fps = getattr(cfg, "fps", 30)
        seed = getattr(cfg, "seed", -1)
        if seed < 0:
            import random
            seed = random.randint(0, 2**32 - 1)

        logger.info(
            "LTX generating: steps=%d, guidance=%.2f, frames=%d, "
            "res=%dx%d, fps=%d, seed=%d",
            num_steps, guidance_scale, num_frames, width, height, fps, seed,
        )

        # Image conditioning — pipeline expects (path, frame_idx, strength) tuples
        images: list[tuple[str, int, float]] = []
        if image is not None:
            if isinstance(image, PILImage.Image):
                tmp = tempfile.NamedTemporaryFile(suffix=".png", prefix="ltx_cond_", delete=False)
                image.save(tmp.name)
                images.append((tmp.name, 0, 1.0))
            elif isinstance(image, (str, Path)) and Path(image).is_file():
                images.append((str(image), 0, 1.0))

        if last_frame is not None:
            if isinstance(last_frame, PILImage.Image):
                tmp = tempfile.NamedTemporaryFile(suffix=".png", prefix="ltx_last_", delete=False)
                last_frame.save(tmp.name)
                images.append((tmp.name, num_frames - 1, 0.8))
            elif isinstance(last_frame, (str, Path)) and Path(last_frame).is_file():
                images.append((str(last_frame), num_frames - 1, 0.8))

        # Bridge the vendored LTX callback signature
        # `(step_idx, latents, force_progress, **kwargs)` to BaseWorker's
        # diffusers-style `_cb(_pipe, step_index, _timestep, cb_kwargs)`.
        wrapped_cb = None
        if callback is not None:
            def wrapped_cb(step_idx, latents, force_progress=False, **_kwargs):
                return callback(None, step_idx, None, {})

        # ── Inpaint mask pass-through ──────────────────────────────
        # If a static inpaint mask is configured AND we have a conditioning
        # image, build the input_frames + input_masks tensors that the LTX
        # inpaint LoRAs consume. The image is replicated across all output
        # frames as a "still video"; the mask is replicated across frames
        # as the per-frame inpaint region.
        mask_path = getattr(cfg, "ltx_inpaint_mask_path", "") or ""
        inpaint_kwargs: dict[str, Any] = {}
        if mask_path and os.path.isfile(mask_path) and images:
            try:
                source_img_path = images[0][0]
                input_frames, input_masks = self._build_static_mask_inputs(
                    source_img_path, mask_path, num_frames, height, width,
                )
                inpaint_kwargs["input_frames"] = input_frames
                inpaint_kwargs["input_masks"] = input_masks
                inpaint_kwargs["masking_strength"] = float(
                    getattr(cfg, "masking_strength", 1.0) or 1.0
                )
                # LTX2.generate forces masking_strength=0.0 (and denoising=1.0)
                # whenever "G" is absent from video_prompt_type — so "VM" alone
                # silently kills the mask. Include "G" to keep the mask branch
                # alive, and pass a non-1.0 denoising_strength so the gate
                # preserves the values instead of resetting them.
                inpaint_kwargs["video_prompt_type"] = "VMG"  # V=video, M=masked, G=guided
                inpaint_kwargs["denoising_strength"] = float(
                    getattr(cfg, "denoising_strength", 1.0) or 1.0
                )
                if inpaint_kwargs["denoising_strength"] >= 1.0:
                    inpaint_kwargs["denoising_strength"] = 0.99
                logger.info(
                    "LTX inpaint mask wired: mask=%s strength=%.2f denoise=%.2f",
                    mask_path, inpaint_kwargs["masking_strength"],
                    inpaint_kwargs["denoising_strength"],
                )
            except Exception as exc:
                logger.warning("Failed to build inpaint mask tensors: %s — running without mask", exc)
                inpaint_kwargs = {}

        # ── Union-Control guide + AV audio pass-through ────────────────
        # Strictly gated on the matching IC-LoRA being enabled so ordinary LTX
        # runs are untouched. The control guide is positioned via start_frame and
        # forwarded as an explicit video_conditioning override (see the vendored
        # ltx2.generate hook); AV audio is forwarded as input_waveform.
        control_kwargs: dict[str, Any] = {}
        active_lora_names = [
            str(n).lower() for n in (getattr(cfg, "activated_loras", []) or [])
        ]
        union_active = any("union-control" in n for n in active_lora_names)
        if union_active and guidance_video is not None and not inpaint_kwargs:
            try:
                cond, msgs = self._build_control_conditioning(
                    guidance_video,
                    control_type=getattr(cfg, "ltx_control_type", "") or "raw",
                    strength=float(getattr(cfg, "ltx_control_strength", 1.0) or 1.0),
                    start_seconds=float(getattr(cfg, "ltx_control_start_seconds", 0.0) or 0.0),
                    ramp_frames=int(getattr(cfg, "ltx_control_ramp_frames", 0) or 0),
                    num_frames=num_frames,
                    fps=fps,
                    progress_cb=getattr(self, "_status_cb", None),
                )
                for m in msgs:
                    logger.info("LTX control: %s", m)
                if cond:
                    control_kwargs["control_conditioning_override"] = cond
                    control_kwargs["video_prompt_type"] = "V"
                    logger.info(
                        "LTX union-control wired: %d span(s), type=%s",
                        len(cond), getattr(cfg, "ltx_control_type", "raw") or "raw",
                    )
            except Exception as exc:
                logger.warning("Failed to build control conditioning: %s — ignoring guide", exc)
                control_kwargs = {}

        av_active = any("talking-head" in n for n in active_lora_names)
        audio_path = getattr(cfg, "ltx_audio_path", "") or ""
        if av_active and audio_path and os.path.isfile(audio_path):
            try:
                import numpy as _np
                try:
                    import soundfile as _sf
                    wav, sr = _sf.read(audio_path, dtype="float32", always_2d=False)
                except Exception:
                    import librosa as _lr
                    wav, sr = _lr.load(audio_path, sr=None, mono=False)
                    wav = _np.asarray(wav, dtype="float32")
                    if wav.ndim == 2:
                        wav = wav.T  # (channels, samples) -> (samples, channels)
                control_kwargs["input_waveform"] = _np.ascontiguousarray(wav)
                control_kwargs["input_waveform_sample_rate"] = int(sr)
                logger.info("LTX AV audio wired: %s (sr=%s)", audio_path, sr)
            except Exception as exc:
                logger.warning("Failed to load AV audio %s: %s — running silent", audio_path, exc)

        # ── Advanced guidance + Self Refiner pass-through ──────────────
        # These controls are collected/persisted by the UI but were never
        # forwarded to LTX2.generate (which DOES consume them). cfg_star /
        # apg gate on guidance_scale > 1.0 (mirrors WanI2V). SLG maps onto
        # LTX's perturbation_* params (start/end percentages → [0,1] floats).
        adv_kwargs: dict[str, Any] = {}
        adv_kwargs["cfg_star_switch"] = int(getattr(cfg, "cfg_star_switch", 0) or 0)
        adv_kwargs["apg_switch"] = (
            int(getattr(cfg, "apg_switch", 0) or 0) if guidance_scale > 1.0 else 0
        )
        if bool(getattr(cfg, "slg_switch", 0)) and guidance_scale > 1.0:
            adv_kwargs["perturbation_switch"] = int(getattr(cfg, "slg_switch", 0) or 0)
            slg_layers = getattr(cfg, "slg_layers", None)
            if slg_layers:
                adv_kwargs["perturbation_layers"] = list(slg_layers)
            adv_kwargs["perturbation_start"] = float(getattr(cfg, "slg_start_perc", 0) or 0) / 100.0
            adv_kwargs["perturbation_end"] = float(getattr(cfg, "slg_end_perc", 100) or 100) / 100.0
        # Self Refiner — UI field names differ slightly from the LTX kwargs.
        adv_kwargs["self_refiner_setting"] = int(getattr(cfg, "self_refiner_setting", 0) or 0)
        adv_kwargs["self_refiner_f_uncertainty"] = float(
            getattr(cfg, "self_refiner_uncertainty", 0.1) or 0.1
        )
        adv_kwargs["self_refiner_certain_percentage"] = float(
            getattr(cfg, "self_refiner_certainty_skip", 0.999) or 0.999
        )

        # ── User-selected LoRAs ────────────────────────────────────────
        # Resolve activated_loras + loras_multipliers and fuse them onto the
        # transformer, then thread the per-phase schedule + selected paths into
        # generate (the vendored impl re-activates them via update_loras_slists).
        loras_slists, loras_selected = self._setup_loras(cfg, num_steps)
        if loras_selected:
            adv_kwargs["loras_slists"] = loras_slists
            adv_kwargs["loras_selected"] = loras_selected

        # Delegate to the vendored LTX2.generate which handles distilled vs
        # two-stage internally. The vendored impl assumes a few "Nones" are
        # actually filled in by the caller (Wan2GP) — we hand it sane defaults
        # so it doesn't blow up on `max(0, min(1, None))`.
        gen_kwargs = dict(
            input_prompt=prompt,
            n_prompt=negative_prompt or None,
            image_start=images[0][0] if images else None,
            image_end=last_frame if last_frame is not None else None,
            sampling_steps=num_steps,
            guide_scale=guidance_scale,
            alt_guide_scale=alt_guidance_scale,
            frame_num=num_frames,
            height=height,
            width=width,
            fps=float(fps),
            seed=seed,
            callback=wrapped_cb,
            input_video_strength=1.0,
            denoising_strength=1.0,
            masking_strength=0.0,
            audio_cfg_scale=1.0,
            video_prompt_type="",
            audio_prompt_type="",
        )
        gen_kwargs.update(adv_kwargs)      # advanced guidance + self refiner
        gen_kwargs.update(control_kwargs)  # union-control guide + AV audio
        gen_kwargs.update(inpaint_kwargs)  # mask kwargs override defaults
        result = self.ltx2.generate(**gen_kwargs)

        if result is None:
            raise RuntimeError("LTX generation returned None (likely interrupted).")

        frames = self._collect_frames_and_audio(result, fps)

        # NOTE: discard_last_frames is intentionally NOT applied here. The
        # shared core path (core/pipeline.run_mode1/2/3) owns the tail-trim so
        # it runs exactly once. Trimming here too would double the requested
        # discard (e.g. discard=8 → 16 frames removed).

        return frames

    def _collect_frames_and_audio(self, result: Any, fps: int) -> list:
        """Convert pipeline output to list of PIL images and stash audio."""
        from PIL import Image as PILImage
        import numpy as np

        # The vendored LTX2.generate returns a dict with:
        #   "x": video tensor [T, C, H, W] (signed range, -1..1)
        #   "audio": numpy array (or None)
        #   "audio_sampling_rate": int
        # The video tensor may be 5D [B, C, T, H, W] from upstream — handle both.
        video_tensor = None
        audio = None
        sample_rate = 24000
        if isinstance(result, dict):
            video_tensor = result.get("x", result.get("video", result.get("frames")))
            audio = result.get("audio")
            sample_rate = result.get("audio_sampling_rate", sample_rate)
        elif isinstance(result, tuple) and len(result) >= 1:
            video_tensor = result[0]
            audio = result[1] if len(result) > 1 else None
        elif isinstance(result, torch.Tensor):
            video_tensor = result
        else:
            raise RuntimeError(f"Unexpected LTX result type: {type(result).__name__}")

        if video_tensor is None:
            raise RuntimeError("LTX pipeline returned no video frames.")

        v = video_tensor.detach().cpu()
        # Squeeze leading batch dim if [B, C, T, H, W]
        if v.dim() == 5:
            v = v[0]
        # Convert [C, T, H, W] -> [T, H, W, C]
        if v.dim() == 4 and v.shape[0] in (1, 3):
            v = v.permute(1, 2, 3, 0)
        elif v.dim() == 4 and v.shape[1] in (1, 3):
            v = v.permute(0, 2, 3, 1)

        if v.dtype == torch.uint8:
            np_frames = v.contiguous().numpy()
        else:
            vf = v.float()
            # Pipeline may produce values in [-1, 1] (signed); remap to [0, 1].
            if vf.min() < -0.01:
                vf = (vf + 1.0) * 0.5
            np_frames = (vf.clamp(0, 1) * 255).byte().contiguous().numpy()
        frames = [PILImage.fromarray(np_frames[i]) for i in range(np_frames.shape[0])]

        if audio is not None:
            try:
                self._save_audio(audio, int(sample_rate))
            except Exception as exc:
                logger.warning("Failed to save LTX audio: %s", exc)

        return frames

    def extend_video(
        self,
        video_path: str,
        prompt: str = "",
        negative_prompt: str = "",
        project_config: Any = None,
        callback: Optional[Callable] = None,
    ) -> list:
        """Extend a video by conditioning on its last frame.

        TODO: switch to the IC-LoRA / RetakePipeline path once the new vendored
        LTX2 wrapper supports it. For now we fall back to last-frame generation
        which is what the previous wrapper did when RetakePipeline was missing.
        """
        from PIL import Image as PILImage
        from supremediffusion.utils.video import extract_frames

        with tempfile.TemporaryDirectory(prefix="ltx_ext_") as tmp:
            frame_paths = extract_frames(video_path, tmp)
            if not frame_paths:
                raise RuntimeError("No frames extracted from source video.")
            last = PILImage.open(frame_paths[-1]).convert("RGB")
            return self.generate(
                image=last,
                prompt=prompt,
                negative_prompt=negative_prompt,
                project_config=project_config,
                callback=callback,
            )

    def _save_audio(self, audio: Any, fps_or_sr: int = 24000) -> None:
        """Save audio (numpy array or torch.Tensor or dataclass) to WAV.

        ``fps_or_sr`` is the audio sample rate when ``audio`` is a raw array;
        if ``audio`` is an Audio dataclass with its own sampling_rate, that
        wins.
        """
        import numpy as np

        waveform = None
        sample_rate = int(fps_or_sr)

        if hasattr(audio, "waveform") and hasattr(audio, "sampling_rate"):
            waveform = audio.waveform
            sample_rate = audio.sampling_rate
        elif isinstance(audio, torch.Tensor):
            waveform = audio
        elif isinstance(audio, np.ndarray):
            waveform = audio

        if waveform is None:
            return

        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()

        if waveform.dtype in (np.float32, np.float64):
            waveform = np.clip(waveform, -1.0, 1.0)
            waveform = (waveform * 32767).astype(np.int16)

        if waveform.ndim > 1:
            if waveform.shape[0] <= 2:
                channels = waveform.shape[0]
                waveform = waveform.T.flatten() if channels == 2 else waveform.flatten()
            else:
                channels = waveform.shape[-1] if waveform.shape[-1] <= 2 else 1
                waveform = waveform.flatten()
        else:
            channels = 1

        import wave
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", prefix="ltx_audio_", delete=False)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(waveform.tobytes())

        self._last_audio_path = tmp.name
        logger.info("LTX audio saved: %s (%d Hz, %d ch)", tmp.name, sample_rate, channels)
