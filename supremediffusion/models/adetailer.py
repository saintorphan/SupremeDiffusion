"""ADetailer — post-generation face refinement via per-face inpaint.

After an image is generated (txt2img/img2img/inpaint), detect faces in the
result and run a high-detail inpaint pass on just each face region using the
same SD pipeline. Mirrors the A1111 ADetailer extension's core behavior.

Two detectors supported:
- ``insightface``: reuses the FaceSwapPipeline's buffalo_l detector. No extra
  model file. Best for AI-generated content.
- ``yolov8``: face_yolov8s.pt via ultralytics. Better on tough angles / small
  faces. Requires ``models/face/face_yolov8s.pt``.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:
    Image = None  # type: ignore[assignment]


class ADetailerProcessor:
    """Detect faces in a generated image and re-inpaint each at higher detail."""

    def __init__(self, models_dir: str) -> None:
        self._models_dir = Path(models_dir)
        self._yolo_model: Any = None  # lazy-loaded ultralytics YOLO
        self._face_pipe: Any = None  # lazy-loaded, reused FaceSwapPipeline

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(
        self,
        image: "Image.Image",
        detector: str = "insightface",
        threshold: float = 0.3,
    ) -> list[tuple[int, int, int, int]]:
        """Return list of (x1, y1, x2, y2) bboxes for detected faces."""
        if detector == "yolov8":
            return self._detect_yolov8(image, threshold)
        return self._detect_insightface(image, threshold)

    def _get_face_pipe(self) -> Any:
        """Lazily create and cache the FaceSwapPipeline.

        Reusing one instance keeps the buffalo_l analyser loaded across all
        faces/images instead of reloading the ~350MB model on every detection
        (the per-image instantiate+release was the hot path).
        """
        if self._face_pipe is None:
            from supremediffusion.models.face_swap import FaceSwapPipeline

            self._face_pipe = FaceSwapPipeline(str(self._models_dir))
        return self._face_pipe

    def _detect_insightface(
        self, image: "Image.Image", threshold: float
    ) -> list[tuple[int, int, int, int]]:
        # Save the PIL image to a temp file (FaceSwapPipeline expects a path).
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            image.save(tmp.name)
            tmp.close()
            _orig = sys.stdout
            sys.stdout = io.StringIO()
            try:
                pipe = self._get_face_pipe()
                faces = pipe.detect_faces(tmp.name)
            finally:
                sys.stdout = _orig
        finally:
            try:
                Path(tmp.name).unlink()
            except OSError:
                pass

        # Honor the panel's detection threshold via InsightFace's det_score so
        # the slider behaves consistently with the yolov8 detector.
        bboxes: list[tuple[int, int, int, int]] = []
        for f in faces:
            bbox = f.get("bbox")
            if bbox is None:
                continue
            if float(f.get("det_score", 1.0)) < float(threshold):
                continue
            x1, y1, x2, y2 = bbox
            bboxes.append((int(x1), int(y1), int(x2), int(y2)))
        return bboxes

    def _detect_yolov8(
        self, image: "Image.Image", threshold: float
    ) -> list[tuple[int, int, int, int]]:
        weights = self._models_dir / "face_yolov8s.pt"
        if not weights.exists():
            logger.warning(
                "face_yolov8s.pt missing at %s — falling back to InsightFace", weights
            )
            return self._detect_insightface(image, threshold)
        if self._yolo_model is None:
            try:
                from ultralytics import YOLO
            except ImportError:
                logger.warning(
                    "ultralytics not installed — falling back to InsightFace. "
                    "Install it for the YOLOv8 detector (pip install ultralytics)."
                )
                return self._detect_insightface(image, threshold)
            self._yolo_model = YOLO(str(weights))
        # Suppress ultralytics' verbose stdout.
        _orig = sys.stdout
        sys.stdout = io.StringIO()
        try:
            results = self._yolo_model.predict(
                source=np.array(image),
                conf=float(threshold),
                verbose=False,
            )
        finally:
            sys.stdout = _orig
        bboxes: list[tuple[int, int, int, int]] = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
            for x1, y1, x2, y2 in xyxy:
                bboxes.append((int(x1), int(y1), int(x2), int(y2)))
        return bboxes

    # ------------------------------------------------------------------
    # Mask construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_face_mask(
        image_size: tuple[int, int],
        bbox: tuple[int, int, int, int],
        dilation_pct: float = 20.0,
        feather_px: int = 12,
    ) -> "Image.Image":
        """White-filled face region on black background, optionally feathered."""
        if Image is None:
            raise RuntimeError("PIL required.")
        w, h = image_size
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        d = max(bw, bh) * (dilation_pct / 100.0)
        x1d = max(0, int(x1 - d))
        y1d = max(0, int(y1 - d))
        x2d = min(w, int(x2 + d))
        y2d = min(h, int(y2 + d))
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        # Ellipse mask is more face-shape-appropriate than a hard rectangle
        # and reduces visible seams at the edges of the inpaint region.
        draw.ellipse((x1d, y1d, x2d, y2d), fill=255)
        if feather_px > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_px))
        return mask

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def process(
        self,
        image: "Image.Image",
        *,
        sd_pipeline: Any,
        base_prompt: str,
        base_neg_prompt: str,
        base_steps: int,
        base_cfg: float,
        sampler: str,
        scheduler: str,
        clip_skip: int = 1,
        detector: str = "insightface",
        threshold: float = 0.3,
        dilation_pct: float = 20.0,
        feather_px: int = 12,
        denoise: float = 0.4,
        steps_add: int = 6,
        inpaint_size: int = 1024,
        prompt_override: str = "",
        neg_prompt_override: str = "",
        callback: Optional[Callable[..., Any]] = None,
    ) -> "Image.Image":
        """Run ADetailer on ``image``. Returns the refined PIL image.

        If no face is detected, returns ``image`` unchanged.

        ``sd_pipeline`` must be the same ``SDImagePipeline`` instance that
        produced ``image`` (or any compatible loaded pipeline). The inpaint
        pass reuses it via ``generate_inpaint(full_res=True)`` so only the
        masked face region is denoised, not the whole image.
        """
        if image is None or Image is None:
            return image

        bboxes = self.detect(image, detector=detector, threshold=threshold)
        if not bboxes:
            logger.info("ADetailer: no faces detected; passing through.")
            return image

        logger.info("ADetailer: %d face(s) detected with %s.", len(bboxes), detector)

        result = image.copy().convert("RGB")
        prompt = prompt_override.strip() or base_prompt
        neg = neg_prompt_override.strip() or base_neg_prompt
        steps = max(1, base_steps + int(steps_add))

        for i, bbox in enumerate(bboxes):
            mask = self._build_face_mask(
                result.size, bbox,
                dilation_pct=dilation_pct, feather_px=feather_px,
            )
            try:
                outputs = sd_pipeline.generate_inpaint(
                    image=result,
                    mask=mask,
                    prompt=prompt,
                    negative_prompt=neg,
                    denoising_strength=float(denoise),
                    width=int(inpaint_size),
                    height=int(inpaint_size),
                    steps=steps,
                    cfg_scale=float(base_cfg),
                    seed=-1,  # random per face — adds variation
                    sampler=sampler,
                    scheduler=scheduler,
                    mask_blur=4,
                    inpainting_fill=1,
                    full_res=True,
                    padding=32,
                    batch_size=1,
                    clip_skip=int(clip_skip),
                    callback=callback,
                )
                if outputs:
                    result = outputs[0]
            except Exception as exc:  # noqa: BLE001
                logger.warning("ADetailer face %d failed: %s", i + 1, exc)

        return result

    def release(self) -> None:
        """Free YOLO model and cached face pipeline if loaded."""
        if self._yolo_model is not None:
            del self._yolo_model
            self._yolo_model = None
        if self._face_pipe is not None:
            try:
                self._face_pipe.release()
            except Exception:  # noqa: BLE001
                pass
            self._face_pipe = None
