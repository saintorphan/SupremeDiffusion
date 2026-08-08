"""Worker threads for SD image generation."""

from __future__ import annotations

import logging
from typing import Any, Optional

from supremediffusion.config.project_config import ProjectConfig

from .base import BaseWorker

logger = logging.getLogger(__name__)


def _apply_adetailer_if_enabled(
    *,
    pipeline: Any,
    config: ProjectConfig,
    output_paths: list[str],
    status_emit: Any,
    progress_emit: Any,
    worker: "BaseWorker | None" = None,
) -> list[str]:
    """If ADetailer is enabled in ``config``, run the post-step over each output.

    Returns the same list of paths (files are overwritten in place).

    ``worker`` (the calling :class:`BaseWorker`) lets the inpaint pass be
    aborted mid-run via an abort-aware step callback, and lets the per-image
    loop bail out between images — without it each ~26-step face refine ran
    uninterruptibly.
    """
    if not getattr(config, "adetailer_enabled", False):
        return output_paths
    if not output_paths:
        return output_paths

    try:
        from PIL import Image
        from supremediffusion.models.adetailer import ADetailerProcessor
    except ImportError as exc:
        logger.warning("ADetailer skipped — import failed: %s", exc)
        return output_paths

    sd = getattr(pipeline, "sd", None)
    if sd is None or not hasattr(sd, "generate_inpaint"):
        logger.warning("ADetailer skipped — SD pipeline missing generate_inpaint.")
        return output_paths

    models_dir = sd.config.model_paths.get("face_models_dir", "")
    if not models_dir:
        logger.warning("ADetailer skipped — face_models_dir not configured.")
        return output_paths

    base_steps = int(getattr(config, "img_steps", 20))
    steps_add = int(getattr(config, "adetailer_steps_add", 6))
    # Abort-aware step callback for the inpaint pass — raises InterruptedError
    # mid-denoise when the worker is aborted (see BaseWorker.make_step_callback).
    callback = None
    if worker is not None:
        callback = worker.make_step_callback(max(1, base_steps + steps_add), "ADetailer")

    proc = ADetailerProcessor(models_dir)
    try:
        n = len(output_paths)
        for i, path in enumerate(output_paths, 1):
            # Allow cancellation between images (the inpaint callback covers
            # cancellation within an image).
            if worker is not None and worker.is_aborted:
                raise InterruptedError("Aborted by user")
            status_emit(f"ADetailer: refining face in image {i}/{n}...")
            try:
                img = Image.open(path).convert("RGB")
                refined = proc.process(
                    img,
                    sd_pipeline=sd,
                    base_prompt=getattr(config, "img_prompt", "") or "",
                    base_neg_prompt=getattr(config, "img_negative_prompt", "") or "",
                    base_steps=base_steps,
                    base_cfg=float(getattr(config, "img_cfg_scale", 7.0)),
                    sampler=getattr(config, "img_sampler", "DPM++ 2M") or "DPM++ 2M",
                    scheduler=getattr(config, "img_scheduler", "Karras") or "Karras",
                    clip_skip=int(getattr(config, "img_clip_skip", 1)),
                    detector=getattr(config, "adetailer_detector", "insightface"),
                    threshold=float(getattr(config, "adetailer_threshold", 0.30)),
                    dilation_pct=float(getattr(config, "adetailer_dilation_pct", 20.0)),
                    feather_px=int(getattr(config, "adetailer_feather_px", 12)),
                    denoise=float(getattr(config, "adetailer_denoise", 0.40)),
                    steps_add=steps_add,
                    inpaint_size=int(getattr(config, "adetailer_inpaint_size", 1024)),
                    prompt_override=getattr(config, "adetailer_prompt_override", "") or "",
                    neg_prompt_override=getattr(config, "adetailer_neg_prompt_override", "") or "",
                    callback=callback,
                )
                refined.save(path)
            except InterruptedError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("ADetailer failed on %s: %s", path, exc)
            progress_emit(0.9 + 0.1 * (i / n), f"ADetailer {i}/{n}")
    finally:
        proc.release()
    return output_paths


class Txt2ImgWorker(BaseWorker):
    """Run txt2img generation on a background thread."""

    def __init__(
        self,
        pipeline: Any,
        project_name: str,
        project_config: ProjectConfig,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._project_name = project_name
        self._config = project_config

    def do_work(self) -> list[str]:
        total_steps = self._config.img_steps or 20
        callback = self.make_step_callback(total_steps)

        self.status.emit("Generating images...")
        self.progress.emit(0.0, "Starting txt2img...")

        result = self._pipeline.run_txt2img(
            project_name=self._project_name,
            config=self._config,
            callback=callback,
        )

        result = _apply_adetailer_if_enabled(
            pipeline=self._pipeline, config=self._config, output_paths=result,
            status_emit=self.status.emit, progress_emit=self.progress.emit,
            worker=self,
        )

        self.progress.emit(1.0, "Complete")
        return result


class Img2ImgWorker(BaseWorker):
    """Run img2img generation on a background thread."""

    def __init__(
        self,
        pipeline: Any,
        project_name: str,
        source_image: str,
        project_config: ProjectConfig,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._project_name = project_name
        self._source = source_image
        self._config = project_config

    def do_work(self) -> list[str]:
        total_steps = self._config.img_steps or 20
        callback = self.make_step_callback(total_steps)

        self.status.emit("Generating images...")
        self.progress.emit(0.0, "Starting img2img...")

        result = self._pipeline.run_img2img(
            project_name=self._project_name,
            source_image=self._source,
            config=self._config,
            callback=callback,
        )

        result = _apply_adetailer_if_enabled(
            pipeline=self._pipeline, config=self._config, output_paths=result,
            status_emit=self.status.emit, progress_emit=self.progress.emit,
            worker=self,
        )

        self.progress.emit(1.0, "Complete")
        return result


class InpaintWorker(BaseWorker):
    """Run inpainting on a background thread."""

    def __init__(
        self,
        pipeline: Any,
        project_name: str,
        source_image: str,
        mask: str,
        project_config: ProjectConfig,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._project_name = project_name
        self._source = source_image
        self._mask = mask
        self._config = project_config

    def do_work(self) -> list[str]:
        total_steps = self._config.img_steps or 20
        callback = self.make_step_callback(total_steps)

        self.status.emit("Inpainting...")
        self.progress.emit(0.0, "Starting inpaint...")

        result = self._pipeline.run_inpaint(
            project_name=self._project_name,
            source_image=self._source,
            mask=self._mask,
            config=self._config,
            callback=callback,
        )

        result = _apply_adetailer_if_enabled(
            pipeline=self._pipeline, config=self._config, output_paths=result,
            status_emit=self.status.emit, progress_emit=self.progress.emit,
            worker=self,
        )

        self.progress.emit(1.0, "Complete")
        return result


class BodyDoubleWorker(BaseWorker):
    """Run body double generation (pose extraction + ControlNet + IP-Adapter)."""

    def __init__(
        self,
        pipeline: Any,
        project_name: str,
        target_image: str,
        mask: str,
        source_person: str,
        project_config: ProjectConfig,
        num_generations: int = 1,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._project_name = project_name
        self._target = target_image
        self._mask = mask
        self._source_person = source_person
        self._config = project_config
        self._num_generations = num_generations

    def do_work(self) -> list[str]:
        from supremediffusion.models.controlnet_types import (
            run_preprocessor,
            preprocessor_overrides_for,
        )
        from PIL import Image
        import gc as gc_module

        # Phase 1: Run preprocessor(s) on target image (skip if CN disabled)
        cn_type = getattr(self._config, "bodydouble_controlnet_type", "") or ""
        cn2_type = getattr(self._config, "bodydouble_controlnet2_type", "") or ""

        control_images = []
        if cn_type:
            self.status.emit("Preprocessing target image...")
            self.progress.emit(0.0, f"Phase 1/2: Running {cn_type} preprocessor...")

            target_img = Image.open(self._target).convert("RGB")
            control_images = [run_preprocessor(
                cn_type, target_img,
                **preprocessor_overrides_for(cn_type, self._config),
            )]

            if cn2_type:
                self.progress.emit(0.05, f"Phase 1/2: Running {cn2_type} preprocessor...")
                control_images.append(run_preprocessor(
                    cn2_type, target_img,
                    **preprocessor_overrides_for(cn2_type, self._config),
                ))

            del target_img
            gc_module.collect()

        # Phase 2: Generate body double
        phase = "Phase 2/2" if cn_type else "Generating"
        self.progress.emit(0.1, f"{phase}: Generating body double...")
        total_steps = self._config.bodydouble_steps or 30
        callback = self.make_step_callback(total_steps)

        result = self._pipeline.run_body_double(
            project_name=self._project_name,
            target_image=self._target,
            mask=self._mask,
            source_person=self._source_person,
            control_images=control_images,
            config=self._config,
            num_images=self._num_generations,
            callback=callback,
        )

        del control_images
        gc_module.collect()

        self.progress.emit(1.0, "Complete")
        return result


def _map_repose_to_bodydouble(cfg) -> None:
    """Map repose_* config fields to bodydouble_* for the shared backend."""
    _FIELD_MAP = {
        "checkpoint": "checkpoint", "sampler": "sampler", "scheduler": "scheduler",
        "prompt": "prompt", "negative_prompt": "negative_prompt",
        "steps": "steps", "cfg_scale": "cfg_scale", "seed": "seed",
        "controlnet_strength": "controlnet_strength",
        "ip_adapter_scale": "ip_adapter_scale",
        "denoising_strength": "denoising_strength", "mask_blur": "mask_blur",
        "controlnet_type": "controlnet_type",
        "controlnet2_type": "controlnet2_type",
        "controlnet2_strength": "controlnet2_strength",
        "ip_adapter_variant": "ip_adapter_variant",
        "width": "width", "height": "height",
        "guidance_start": "guidance_start",
        "guidance_end": "guidance_end",
        "clip_skip": "clip_skip",
        "style_preset": "style_preset",
        "ip_adapter2_variant": "ip_adapter2_variant",
        "ip_adapter2_scale": "ip_adapter2_scale",
        "style_ref_path": "style_ref_path",
    }
    for src_suffix, dst_suffix in _FIELD_MAP.items():
        val = getattr(cfg, f"repose_{src_suffix}", None)
        if val is not None:
            setattr(cfg, f"bodydouble_{dst_suffix}", val)


class RePoseWorker(BaseWorker):
    """Run re-pose generation (pose from reference + ControlNet + IP-Adapter).

    Unlike BodyDoubleWorker which extracts pose from the target image,
    RePoseWorker extracts pose from a separate pose reference image and
    uses the source image as both the inpaint base and the IP-Adapter
    appearance reference (same person, new pose).
    """

    def __init__(
        self,
        pipeline: Any,
        project_name: str,
        source_image: str,
        mask: str,
        pose_reference: str,
        project_config: ProjectConfig,
        num_generations: int = 1,
        *,
        appearance_image: Optional[str] = None,
        skip_preprocess: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._project_name = project_name
        self._source = source_image
        self._mask = mask
        self._pose_reference = pose_reference
        self._config = project_config
        self._num_generations = num_generations
        self._appearance = appearance_image or source_image
        self._skip_preprocess = skip_preprocess

    def do_work(self) -> list[str]:
        from supremediffusion.models.controlnet_types import run_preprocessor
        from PIL import Image
        import gc as gc_module

        cn_type = getattr(self._config, "repose_controlnet_type", "openpose") or "openpose"
        cn2_type = getattr(self._config, "repose_controlnet2_type", "") or ""

        # Phase 1: Preprocess pose reference (or skip if pre-processed)
        if self._skip_preprocess:
            self.progress.emit(0.05, "Using pre-processed pose image...")
            control_images = [Image.open(self._pose_reference).convert("RGB")]
            if cn2_type:
                control_images.append(Image.open(self._pose_reference).convert("RGB"))
        else:
            self.status.emit("Preprocessing...")
            self.progress.emit(0.0, f"Running {cn_type} preprocessor...")

            pose_img = Image.open(self._pose_reference).convert("RGB")
            control_images = [run_preprocessor(cn_type, pose_img)]

            if cn2_type:
                self.progress.emit(0.05, f"Running {cn2_type} preprocessor...")
                control_images.append(run_preprocessor(cn2_type, pose_img))

            del pose_img
            gc_module.collect()

        if self.is_aborted:
            raise InterruptedError("Aborted by user")

        # Phase 2: Map repose config fields to bodydouble fields for the backend
        _map_repose_to_bodydouble(self._config)

        # Phase 3: Generate
        self.progress.emit(0.1, "Generating re-pose...")
        total_steps = self._config.repose_steps or 30
        callback = self.make_step_callback(total_steps)

        result = self._pipeline.run_body_double(
            project_name=self._project_name,
            target_image=self._source,
            mask=self._mask,
            source_person=self._appearance,
            control_images=control_images,
            config=self._config,
            num_images=self._num_generations,
            callback=callback,
        )

        del control_images
        gc_module.collect()

        self.progress.emit(1.0, "Complete")
        return result


class AutoSegmentWorker(BaseWorker):
    """Run BiRefNet auto-segmentation to generate a foreground mask."""

    def __init__(
        self,
        image_path: str,
        models_root: str,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._image_path = image_path
        self._models_root = models_root

    def do_work(self) -> str:
        self.status.emit("Running auto-segmentation...")
        self.progress.emit(0.0, "Loading BiRefNet...")

        from supremediffusion.models.segmentation import segment_foreground

        mask_path = segment_foreground(self._image_path, self._models_root)

        self.progress.emit(1.0, "Segmentation complete")
        return mask_path


class ControlNetWorker(BaseWorker):
    """Run ControlNet generation (preprocessing + generation)."""

    def __init__(
        self,
        pipeline: Any,
        project_name: str,
        condition_image: str,
        condition_image2: Optional[str],
        source_image: Optional[str],
        mask_image: Optional[str],
        project_config: ProjectConfig,
        num_generations: int = 1,
        *,
        skip_preprocess: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._project_name = project_name
        self._condition = condition_image
        self._condition2 = condition_image2
        self._source = source_image
        self._mask = mask_image
        self._config = project_config
        self._num_generations = num_generations
        self._skip_preprocess = skip_preprocess

    def do_work(self) -> list[str]:
        from supremediffusion.models.controlnet_types import (
            run_preprocessor,
            preprocessor_overrides_for,
        )
        from PIL import Image
        import gc as gc_module

        cn_type = self._config.controlnet_type or "canny"
        cn2_type = self._config.controlnet_type2 or ""

        # Phase 1: Run preprocessor(s) unless user marked as pre-processed
        if self._skip_preprocess:
            self.progress.emit(0.05, "Using pre-processed condition image...")
            control_images = [Image.open(self._condition).convert("RGB")]
            if cn2_type:
                cond2_path = self._condition2 or self._condition
                control_images.append(Image.open(cond2_path).convert("RGB"))
        else:
            self.status.emit("Preprocessing...")
            self.progress.emit(0.0, f"Running {cn_type} preprocessor...")

            cond_img = Image.open(self._condition).convert("RGB")
            control_images = [run_preprocessor(
                cn_type, cond_img,
                **preprocessor_overrides_for(cn_type, self._config),
            )]

            if cn2_type:
                self.progress.emit(0.05, f"Running {cn2_type} preprocessor...")
                cond2_img = cond_img
                if self._condition2:
                    cond2_img = Image.open(self._condition2).convert("RGB")
                control_images.append(run_preprocessor(
                    cn2_type, cond2_img,
                    **preprocessor_overrides_for(cn2_type, self._config),
                ))
                if self._condition2:
                    del cond2_img

            del cond_img
            gc_module.collect()

        if self.is_aborted:
            raise InterruptedError("Aborted by user")

        # Phase 2: Generate
        self.progress.emit(0.1, "Generating with ControlNet...")
        total_steps = self._config.controlnet_steps or 20
        callback = self.make_step_callback(total_steps)

        result = self._pipeline.run_controlnet(
            project_name=self._project_name,
            control_images=control_images,
            source_image=self._source,
            mask_image=self._mask,
            config=self._config,
            num_images=self._num_generations,
            callback=callback,
        )

        del control_images
        gc_module.collect()

        self.progress.emit(1.0, "Complete")
        return result


class FaceDetectWorker(BaseWorker):
    """Detect faces in an image on a background thread."""

    def __init__(
        self,
        image_path: str,
        models_dir: str,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._image_path = image_path
        self._models_dir = models_dir

    def do_work(self) -> list[dict]:
        import io
        import sys
        from supremediffusion.models.face_swap import FaceSwapPipeline

        self.status.emit("Detecting faces...")
        self.progress.emit(0.0, "Loading face analyser...")

        _orig_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            pipe = FaceSwapPipeline(self._models_dir)
        finally:
            sys.stdout = _orig_stdout
        try:
            faces = pipe.detect_faces(self._image_path)
            self.progress.emit(1.0, f"Found {len(faces)} face(s)")
            return faces
        finally:
            pipe.release()


class FaceSwapWorker(BaseWorker):
    """Run face swap on a background thread."""

    def __init__(
        self,
        source_path: str,
        target_path: str,
        project_config: ProjectConfig,
        models_dir: str,
        *,
        enhancer_strength: float = 0.5,
        source_face_map: dict[int, int] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source = source_path
        self._target = target_path
        self._config = project_config
        self._models_dir = models_dir
        self._enhancer_strength = enhancer_strength
        self._source_face_map = source_face_map

    def do_work(self) -> str:
        import io
        import sys
        from supremediffusion.models.face_swap import FaceSwapPipeline

        self.status.emit("Loading face swap models...")
        self.progress.emit(0.0, "Starting face swap...")

        _orig_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            pipe = FaceSwapPipeline(self._models_dir)
        finally:
            sys.stdout = _orig_stdout
        try:
            result = pipe.swap(
                source_path=self._source,
                target_path=self._target,
                source_face_idx=self._config.faceswap_source_face_idx,
                target_face_idx=self._config.faceswap_target_face_idx,
                swap_model=self._config.faceswap_model or "inswapper_128",
                enhancer=self._config.faceswap_enhancer or None,
                blend_ratio=self._config.faceswap_blend_ratio,
                swap_all=self._config.faceswap_swap_all,
                enhancer_strength=self._enhancer_strength,
                source_face_map=self._source_face_map,
                enhance_source=getattr(self._config, "faceswap_enhance_source", False),
            )

            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            result.save(tmp.name)
            tmp.close()

            self.progress.emit(1.0, "Face swap complete")
            return tmp.name
        finally:
            pipe.release()


class BatchFaceSwapWorker(BaseWorker):
    """Run face swap across multiple target images."""

    def __init__(
        self,
        source_path: str,
        target_paths: list[str],
        project_config: ProjectConfig,
        models_dir: str,
        output_dir: str,
        *,
        enhancer_strength: float = 0.5,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source = source_path
        self._targets = target_paths
        self._config = project_config
        self._models_dir = models_dir
        self._output_dir = output_dir
        self._enhancer_strength = enhancer_strength

    def do_work(self) -> list[str]:
        import io
        import sys
        from supremediffusion.models.face_swap import FaceSwapPipeline

        self.progress.emit(0.0, "Loading face swap models...")

        _orig_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            pipe = FaceSwapPipeline(self._models_dir)
        finally:
            sys.stdout = _orig_stdout

        try:
            results = []
            total = len(self._targets)
            for i, tgt in enumerate(self._targets):
                if self.is_aborted:
                    raise InterruptedError("Aborted by user")

                frac = i / total
                self.progress.emit(frac, f"Swapping face {i + 1}/{total}...")

                result = pipe.swap(
                    source_path=self._source,
                    target_path=tgt,
                    source_face_idx=self._config.faceswap_source_face_idx,
                    target_face_idx=self._config.faceswap_target_face_idx,
                    swap_model=self._config.faceswap_model or "inswapper_128",
                    enhancer=self._config.faceswap_enhancer or None,
                    blend_ratio=self._config.faceswap_blend_ratio,
                    swap_all=self._config.faceswap_swap_all,
                    enhancer_strength=self._enhancer_strength,
                    enhance_source=getattr(self._config, "faceswap_enhance_source", False),
                )

                from pathlib import Path
                out_name = f"faceswap_{Path(tgt).stem}.png"
                out_path = str(Path(self._output_dir) / out_name)
                result.save(out_path)
                results.append(out_path)

            self.progress.emit(1.0, f"Batch complete — {len(results)} images")
            return results
        finally:
            pipe.release()


class VideoFaceSwapWorker(BaseWorker):
    """Run face swap on every frame of a video."""

    def __init__(
        self,
        source_path: str,
        video_path: str,
        project_config: ProjectConfig,
        models_dir: str,
        output_path: str,
        *,
        enhancer_strength: float = 0.5,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source = source_path
        self._video = video_path
        self._config = project_config
        self._models_dir = models_dir
        self._output = output_path
        self._enhancer_strength = enhancer_strength

    def do_work(self) -> str:
        import io
        import json
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        import cv2
        from supremediffusion.models.face_swap import FaceSwapPipeline

        self.progress.emit(0.0, "Probing video...")

        # Probe video for fps and frame count
        probe_cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-show_format", self._video,
        ]
        probe = subprocess.run(probe_cmd, capture_output=True, text=True)
        if probe.returncode != 0:
            raise RuntimeError(
                f"ffprobe failed ({probe.returncode}): {probe.stderr.strip()}"
            )
        try:
            info = json.loads(probe.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"ffprobe returned invalid JSON: {exc}") from exc

        fps = 24.0
        total_frames = 0
        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                r = s.get("r_frame_rate", "24/1")
                num, den = r.split("/")
                fps = float(num) / float(den)
                total_frames = int(s.get("nb_frames", 0))
                break
        if total_frames <= 0:
            dur = float(info.get("format", {}).get("duration", "0"))
            total_frames = max(1, int(dur * fps))

        self.progress.emit(0.05, "Loading face swap models...")

        _orig_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            pipe = FaceSwapPipeline(self._models_dir)
        finally:
            sys.stdout = _orig_stdout

        try:
            # Optional source pre-enhance — sharpens the source face before
            # InsightFace extracts the identity embedding (matches the still
            # swap() path). swap_frame operates on a precomputed embedding, so
            # apply it here before detecting the source face.
            source_path = self._source
            if getattr(self._config, "faceswap_enhance_source", False):
                try:
                    source_path = pipe._enhance_source_face(self._source)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Source face enhance failed, using original: %s", exc)
                    source_path = self._source

            # Get source face embedding once
            source_faces = pipe.detect_faces(source_path)
            if not source_faces:
                raise ValueError("No face detected in source image")
            src_idx = self._config.faceswap_source_face_idx
            if src_idx >= len(source_faces):
                src_idx = 0
            source_embedding = source_faces[src_idx]["embedding"]

            swap_model = self._config.faceswap_model or "inswapper_128"
            enhancer_name = self._config.faceswap_enhancer or None
            blend_ratio = self._config.faceswap_blend_ratio
            swap_all = self._config.faceswap_swap_all
            tgt_face_idx = self._config.faceswap_target_face_idx

            # Open video
            cap = cv2.VideoCapture(self._video)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {self._video}")

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Write frames to temp dir, then encode with ffmpeg for codec support
            frames_dir = tempfile.mkdtemp(prefix="faceswap_frames_")
            frame_idx = 0

            self.progress.emit(0.1, "Processing frames...")

            while True:
                if self.is_aborted:
                    cap.release()
                    raise InterruptedError("Aborted by user")

                ret, frame = cap.read()
                if not ret:
                    break

                # Detect faces in this frame
                analyser = pipe._load_analyser()
                faces_raw = analyser.get(frame)
                frame_faces = []
                for fi, face in enumerate(faces_raw):
                    frame_faces.append({
                        "bbox": face.bbox.astype(int).tolist(),
                        "kps": face.kps.tolist() if face.kps is not None else None,
                        "embedding": face.normed_embedding,
                        "index": fi,
                        "_face_obj": face,
                    })

                if frame_faces:
                    frame = pipe.swap_frame(
                        frame, source_embedding, frame_faces,
                        swap_model=swap_model,
                        enhancer=enhancer_name,
                        blend_ratio=blend_ratio,
                        enhancer_strength=self._enhancer_strength,
                        swap_all=swap_all,
                        target_face_idx=tgt_face_idx,
                    )

                out_frame = str(Path(frames_dir) / f"{frame_idx:06d}.png")
                cv2.imwrite(out_frame, frame)
                frame_idx += 1

                if frame_idx % 5 == 0 or frame_idx == total_frames:
                    frac = 0.1 + 0.75 * (frame_idx / max(total_frames, 1))
                    self.progress.emit(
                        min(frac, 0.85),
                        f"Frame {frame_idx}/{total_frames}",
                    )

            cap.release()

            if frame_idx == 0:
                raise ValueError("No frames read from video")

            self.progress.emit(0.88, "Encoding video...")

            # Encode with ffmpeg, preserving original audio
            from sdqt.utils.codec import (
                configured_codec_args,
                configured_encoder_name as _enc_name,
                pix_fmt_args as _pix_fmt,
            )

            codec_args = configured_codec_args()
            encode_cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(Path(frames_dir) / "%06d.png"),
                "-i", self._video,
                "-map", "0:v",
                "-map", "1:a?",
                *codec_args,
                *_pix_fmt(_enc_name()),
                "-vf", "scale=in_range=full:out_range=full",
                "-r", str(fps),
                "-shortest",
                self._output,
            ]
            encode = subprocess.run(encode_cmd, capture_output=True, timeout=600)

            # Cleanup frames
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)

            if encode.returncode != 0:
                err = (encode.stderr or b"").decode("utf-8", "replace").strip()
                raise RuntimeError(f"ffmpeg encoding failed ({encode.returncode}): {err}")

            if not Path(self._output).is_file():
                raise RuntimeError("Video encoding failed — no output file")

            self.progress.emit(1.0, "Video face swap complete")
            return self._output
        finally:
            pipe.release()
