"""MimicMotion (pose-driven body animation) pipeline wrapper.

Wraps Tencent's MimicMotion (https://github.com/Tencent/MimicMotion) — an
SVD-XT-based pose-driven video generator. The upstream code lives at
``~/Projects/MimicMotion`` as a sibling clone; this module injects it into
``sys.path`` on load() and calls into the upstream ``create_pipeline()``.

Memory profile is tuned for 12 GB GPUs: VAE slicing + per-frame decode chunks,
and components live on CPU until the first __call__ moves them to GPU. The
pipeline's own ``__call__`` handles VAE/UNet device shuffling internally.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]
    logger.warning("PyTorch is not installed. MimicMotion inference unavailable.")

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]


# Upstream repo location. Cloned at ~/Projects/MimicMotion.
_UPSTREAM_REPO = Path.home() / "Projects" / "MimicMotion"


def _ensure_upstream_on_path() -> None:
    """Add the MimicMotion clone + its parent to sys.path so its modules import.

    Done lazily here (not at module import) so the wrapper file is harmless to
    import even when the upstream clone is missing.
    """
    if not _UPSTREAM_REPO.is_dir():
        raise RuntimeError(
            f"MimicMotion upstream repo not found at {_UPSTREAM_REPO}. "
            "Clone it with: git clone https://github.com/Tencent/MimicMotion.git "
            f"{_UPSTREAM_REPO}"
        )
    repo_str = str(_UPSTREAM_REPO)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


class MimicMotionPipeline:
    """High-level wrapper around upstream ``MimicMotionPipeline``.

    Lifecycle mirrors WanI2VPipeline: ``load()`` builds/loads weights,
    ``unload()`` releases GPU memory, ``is_loaded`` reports state, and
    ``generate()`` runs a single inference returning a list of PIL frames.
    """

    def __init__(self, global_config: Any) -> None:
        self.config = global_config
        self.pipe: Any = None
        self._loaded: bool = False
        self._device: Any = None

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _patch_dwpose_paths(self) -> None:
        """Repoint the upstream DWPose singleton at absolute ONNX paths.

        Upstream hardcodes ``models/DWPose/*.onnx`` relative to CWD, which
        only works when running from the MimicMotion repo root. We override
        the singleton's stored args with absolute paths under
        ``<mimicmotion_dir>/DWPose/``. Models are loaded lazily on first
        __call__, so this patch must run before any pose extraction.
        """
        model_paths = getattr(self.config, "model_paths", {}) or {}
        root = Path(model_paths.get("mimicmotion_dir") or "")
        dw_dir = root / "DWPose"
        det = dw_dir / "yolox_l.onnx"
        pose = dw_dir / "dw-ll_ucoco_384.onnx"
        if not det.exists() or not pose.exists():
            raise RuntimeError(
                f"DWPose models missing. Expected:\n  {det}\n  {pose}\n"
                "Download from https://huggingface.co/yzd-v/DWPose"
            )
        from mimicmotion.dwpose import dwpose_detector as _dwd
        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        # Replace the (model_det, model_pose, device) tuple on the singleton.
        _dwd.dwpose_detector.args = (str(det), str(pose), device)
        # If a prior session already lazy-loaded with bad paths, drop it.
        if hasattr(_dwd.dwpose_detector, "pose_estimation"):
            try:
                _dwd.dwpose_detector.release_memory()
            except Exception:  # noqa: BLE001
                pass

    def _resolve_paths(self) -> tuple[Path, Path]:
        """Return (svd_base_dir, motion_checkpoint_path) from GlobalConfig."""
        model_paths = getattr(self.config, "model_paths", {}) or {}
        root = Path(model_paths.get("mimicmotion_dir") or "")
        if not root or not root.exists():
            raise RuntimeError(
                "mimicmotion_dir is not set or does not exist. "
                "Expected the symlink at <app>/models/mimicmotion."
            )
        svd_subdir = model_paths.get("mimicmotion_svd_subdir", "svd-xt")
        motion_subdir = model_paths.get("mimicmotion_motion_subdir", "motion_module")
        svd_dir = root / svd_subdir
        motion_dir = root / motion_subdir
        if not svd_dir.exists():
            raise RuntimeError(f"SVD-XT base dir missing: {svd_dir}")
        if not motion_dir.exists():
            raise RuntimeError(f"MimicMotion module dir missing: {motion_dir}")
        # Match the checkpoint to the SVD base: the 1-1 checkpoint expects the
        # SVD-XT-1-1 base; the 1.pth checkpoint expects vanilla SVD-XT. Wrong
        # pairing causes silent quality degradation / weight-key mismatches.
        prefer_1_1 = "1-1" in svd_subdir
        ck_1_1 = motion_dir / "MimicMotion_1-1.pth"
        ck_1 = motion_dir / "MimicMotion_1.pth"
        checkpoint: Path | None = None
        if prefer_1_1 and ck_1_1.exists():
            checkpoint = ck_1_1
        elif not prefer_1_1 and ck_1.exists():
            checkpoint = ck_1
        else:
            # Fallback: take whichever exists.
            for c in (ck_1, ck_1_1):
                if c.exists():
                    checkpoint = c
                    break
        if checkpoint is None:
            raise RuntimeError(
                f"No MimicMotion checkpoint found in {motion_dir}. "
                "Download MimicMotion_1.pth or MimicMotion_1-1.pth from "
                "https://huggingface.co/tencent/MimicMotion"
            )
        return svd_dir, checkpoint

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Build the upstream MimicMotion pipeline and load weights."""
        if torch is None:
            raise RuntimeError("PyTorch required but not installed.")
        if self._loaded:
            return

        _ensure_upstream_on_path()
        from mimicmotion.utils.geglu_patch import patch_geglu_inplace
        from mimicmotion.utils.loader import create_pipeline

        patch_geglu_inplace()
        self._patch_dwpose_paths()

        svd_dir, checkpoint = self._resolve_paths()
        infer_config = SimpleNamespace(
            base_model_path=str(svd_dir),
            ckpt_path=str(checkpoint),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading MimicMotion: base=%s ckpt=%s device=%s",
                    svd_dir.name, checkpoint.name, device)

        torch.set_default_dtype(torch.float16)
        self.pipe = create_pipeline(infer_config, device)
        self._device = device

        # 12 GB VRAM friendliness — VAE slicing + small decode chunks. The
        # pipeline already moves VAE on/off GPU internally per its __call__.
        try:
            if hasattr(self.pipe.vae, "enable_slicing"):
                self.pipe.vae.enable_slicing()
            if hasattr(self.pipe.vae, "enable_tiling"):
                self.pipe.vae.enable_tiling()
        except Exception as exc:  # noqa: BLE001
            logger.warning("VAE slicing/tiling not applied: %s", exc)

        # Half-precision UNet to halve activation memory.
        try:
            self.pipe.unet.to(dtype=torch.float16)
            self.pipe.pose_net.to(dtype=torch.float16)
            self.pipe.image_encoder.to(dtype=torch.float16)
        except Exception:  # noqa: BLE001
            pass

        self._loaded = True

    def unload(self) -> None:
        """Release GPU memory and tear down the pipeline."""
        if self.pipe is not None:
            for attr in ("unet", "vae", "image_encoder", "pose_net"):
                comp = getattr(self.pipe, attr, None)
                if comp is not None:
                    try:
                        comp.to("cpu")
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        setattr(self.pipe, attr, None)
                    except Exception:  # noqa: BLE001
                        pass
                    del comp
            del self.pipe
            self.pipe = None
        self._loaded = False
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def preprocess(
        self,
        reference_image_path: str,
        driving_video_path: str,
        resolution: int = 576,
        sample_stride: int = 2,
    ) -> tuple[Any, Any]:
        """Extract DWPose sequence + crop reference image.

        Returns (pose_pixels, image_pixels) as torch tensors in [-1, 1].
        Mirrors upstream ``inference.preprocess()`` exactly so behavior stays
        identical to the reference implementation.
        """
        if torch is None or np is None:
            raise RuntimeError("torch + numpy required for preprocessing.")
        _ensure_upstream_on_path()

        from torchvision.datasets.folder import pil_loader
        from torchvision.transforms.functional import (
            pil_to_tensor, resize, center_crop,
        )
        from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose

        # SVD aspect ratio prior (portrait: w/h = 9/16 = 0.5625).
        ASPECT_RATIO = 9 / 16

        image_pixels = pil_loader(reference_image_path)
        image_pixels = pil_to_tensor(image_pixels)  # (C, H, W)
        h, w = image_pixels.shape[-2:]
        if h > w:
            w_target = resolution
            h_target = int(resolution / ASPECT_RATIO // 64) * 64
        else:
            w_target = int(resolution / ASPECT_RATIO // 64) * 64
            h_target = resolution
        h_w_ratio = float(h) / float(w)
        if h_w_ratio < h_target / w_target:
            h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
        else:
            h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target
        image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
        image_pixels = center_crop(image_pixels, [h_target, w_target])
        image_pixels = image_pixels.permute((1, 2, 0)).numpy()

        image_pose = get_image_pose(image_pixels)
        video_pose = get_video_pose(driving_video_path, image_pixels, sample_stride=sample_stride)
        pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
        image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))
        return (
            torch.from_numpy(pose_pixels.copy()) / 127.5 - 1,
            torch.from_numpy(image_pixels) / 127.5 - 1,
        )

    # ------------------------------------------------------------------
    # Pose preview (no inference, for "did DWPose find my subject?" check)
    # ------------------------------------------------------------------

    def preview_poses(
        self,
        reference_image_path: str,
        driving_video_path: str,
        resolution: int = 512,
    ) -> tuple[Any, Any]:
        """Run DWPose on the reference image + driver's first frame.

        Returns ``(ref_pose_rgb, driver_pose_rgb)`` as ``np.ndarray`` RGB images
        with the detected skeleton drawn. Use this to validate that DWPose
        finds your subject in both inputs before committing GPU time to a full
        generation.

        Cheap: only loads DWPose ONNX models (~340 MB) on CPU/GPU as configured,
        not the SVD/MimicMotion weights.
        """
        if torch is None or np is None:
            raise RuntimeError("torch + numpy required.")
        _ensure_upstream_on_path()
        self._patch_dwpose_paths()

        from torchvision.datasets.folder import pil_loader
        from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
        from mimicmotion.dwpose.preprocess import get_image_pose
        import decord

        ASPECT_RATIO = 9 / 16

        # --- Reference image: same crop logic as preprocess() ---------------
        img = pil_loader(reference_image_path)
        img_t = pil_to_tensor(img)
        h, w = img_t.shape[-2:]
        if h > w:
            w_t = resolution
            h_t = int(resolution / ASPECT_RATIO // 64) * 64
        else:
            w_t = int(resolution / ASPECT_RATIO // 64) * 64
            h_t = resolution
        h_w_ratio = float(h) / float(w)
        if h_w_ratio < h_t / w_t:
            h_r, w_r = h_t, math.ceil(h_t / h_w_ratio)
        else:
            h_r, w_r = math.ceil(w_t * h_w_ratio), w_t
        img_t = resize(img_t, [h_r, w_r], antialias=None)
        img_t = center_crop(img_t, [h_t, w_t])
        ref_np = img_t.permute((1, 2, 0)).numpy()
        ref_pose = get_image_pose(ref_np)

        # --- Driver first frame --------------------------------------------
        vr = decord.VideoReader(driving_video_path)
        first_frame = vr[0].asnumpy()  # H, W, 3 uint8
        # Resize driver frame to match ref dims (DWPose runs on the frame as-is,
        # so showing at ref dimensions makes the side-by-side comparison fair).
        from PIL import Image as _PILImage
        drv_pil = _PILImage.fromarray(first_frame)
        drv_pil = drv_pil.resize((w_t, h_t))
        drv_np = np.array(drv_pil)
        drv_pose = get_image_pose(drv_np)

        return ref_pose, drv_pose

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(
        self,
        reference_image_path: str,
        driving_video_path: str,
        *,
        resolution: int = 576,
        num_frames_per_tile: int = 16,
        tile_overlap: int = 6,
        num_inference_steps: int = 25,
        min_guidance: float = 1.0,
        max_guidance: float = 3.0,
        noise_aug_strength: float = 0.02,
        sample_stride: int = 2,
        seed: int = 42,
        fps: int = 7,
        callback: Optional[Callable[..., Any]] = None,
    ) -> Any:
        """Run a single MimicMotion inference.

        Args:
            reference_image_path: still image whose appearance is preserved.
            driving_video_path: video whose DWPose skeleton drives motion.
            resolution: short-side resolution. Width/height computed from
                the reference image aspect ratio. 576 = default upstream;
                drop to 384 if VRAM-bound.
            num_frames_per_tile: temporal tile size. SVD-XT was trained on
                14-25 frames; keep 14-25.
            tile_overlap: frames shared between adjacent tiles for smoothing.
            num_inference_steps: denoising steps. 25 default, 12-15 for draft.
            min_guidance / max_guidance: SVD's linearly-ramped CFG range.
            seed: -1 → random.
            callback: optional callback_on_step_end (step, timestep, kwargs).

        Returns:
            torch.Tensor (T, C, H, W) uint8 frames. First frame dropped
            (always identical to reference image — upstream convention).
        """
        if not self._loaded:
            self.load()
        if torch is None:
            raise RuntimeError("PyTorch required.")

        pose_pixels, image_pixels = self.preprocess(
            reference_image_path, driving_video_path,
            resolution=resolution, sample_stride=sample_stride,
        )

        from torchvision.transforms.functional import to_pil_image

        ref_pil_list = [
            to_pil_image(img.to(torch.uint8))
            for img in (image_pixels + 1.0) * 127.5
        ]

        device = self._device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device=device)
        if seed is None or seed < 0:
            generator.seed()
        else:
            generator.manual_seed(int(seed))

        frames = self.pipe(
            ref_pil_list,
            image_pose=pose_pixels,
            num_frames=pose_pixels.size(0),
            tile_size=num_frames_per_tile,
            tile_overlap=tile_overlap,
            height=pose_pixels.shape[-2],
            width=pose_pixels.shape[-1],
            fps=fps,
            noise_aug_strength=noise_aug_strength,
            num_inference_steps=num_inference_steps,
            generator=generator,
            min_guidance_scale=min_guidance,
            max_guidance_scale=max_guidance,
            decode_chunk_size=4,
            output_type="pt",
            device=device,
            callback_on_step_end=callback,
        ).frames.cpu()

        video_frames = (frames * 255.0).to(torch.uint8)
        # Upstream drops first frame (it's a near-copy of the reference).
        return video_frames[0, 1:]

    # ------------------------------------------------------------------
    # Output encoding helper
    # ------------------------------------------------------------------

    @staticmethod
    def save_mp4(frames_uint8: Any, output_path: str, fps: int = 24) -> None:
        """Encode (T, C, H, W) uint8 tensor to mp4 via torchvision.io.

        Uses libx264; for project codec preferences, the caller should
        re-encode through configured_codec_args(). This is the quick path
        used by the worker for the initial save.
        """
        if torch is None:
            raise RuntimeError("torch required.")
        import torchvision

        # torchvision.io.write_video expects (T, H, W, C) uint8 on CPU.
        frames = frames_uint8.permute(0, 2, 3, 1).contiguous()
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")
