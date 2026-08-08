"""Worker thread for Hires Fix — upscale + optional img2img second pass.

This runs on the FINAL stitched output, NOT per-clip, to avoid progressive
degradation when clips are concatenated.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from sdqt.workers.base import BaseWorker

logger = logging.getLogger(__name__)

# Above this many pixels the upscaled target is refined in overlapping tiles
# rather than as one image, so the second pass doesn't OOM on small cards.
_TILE_PIXEL_THRESHOLD = 1_300_000  # ~1280x1024
_TILE_OVERLAP = 96


def _probe_tile_size() -> int:
    """Pick a refine tile edge (px) from total VRAM.

    <=12GB -> 768, <=16GB -> 1024, else 1280. Falls back to 1024 when torch
    or CUDA isn't available (e.g. the pipeline is running on CPU).
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 1024
        total = torch.cuda.get_device_properties(0).total_memory
        gb = total / (1024 ** 3)
    except Exception:
        return 1024
    if gb <= 12:
        return 768
    if gb <= 16:
        return 1024
    return 1280


def _feather_alpha(h: int, w: int, overlap: int) -> np.ndarray:
    """1ch [h, w] cosine feather in float32, peaked at the centre.

    Edges ramp over ``overlap`` px via a raised-cosine so overlapping tiles
    blend without visible seams; the ramp is clamped to a small epsilon so a
    single tile can still cover the extreme edge rows/cols.
    """
    def _axis(n: int) -> np.ndarray:
        a = np.ones(n, dtype=np.float32)
        ramp = max(1, min(overlap, n // 2))
        t = (np.arange(ramp, dtype=np.float32) + 0.5) / ramp
        edge = 0.5 - 0.5 * np.cos(np.pi * t)  # 0 -> 1 raised cosine
        a[:ramp] = edge
        a[n - ramp:] = edge[::-1]
        return np.clip(a, 1e-3, 1.0)

    return _axis(h)[:, None] * _axis(w)[None, :]


class HiresFixWorker(BaseWorker):
    """Upscale an image (or video frames) and optionally run an img2img second pass.

    Emits ``finished_ok(str)`` with the path to the hires output.
    """

    def __init__(
        self,
        source_path: str,
        upscale_model_path: str,
        output_dir: str,
        scale: float = 2.0,
        do_second_pass: bool = False,
        denoise_strength: float = 0.3,
        img_pipeline=None,
        project_config=None,
        prompt: str = "",
        negative_prompt: str = "",
        steps: int = 20,
        cfg_scale: float = 7.0,
        sampler: str = "euler",
        scheduler: str = "normal",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source_path = source_path
        self._upscale_model_path = upscale_model_path
        self._output_dir = output_dir
        self._scale = scale
        self._do_second_pass = do_second_pass
        self._denoise_strength = denoise_strength
        self._img_pipeline = img_pipeline
        self._project_config = project_config
        self._prompt = prompt
        self._negative_prompt = negative_prompt
        self._steps = steps
        self._cfg_scale = cfg_scale
        self._sampler = sampler
        self._scheduler = scheduler

    def do_work(self) -> str:
        from supremediffusion.postprocessing.ai_upscale import upscale_frames

        self.progress.emit(0.0, "Loading image...")
        img = Image.open(self._source_path).convert("RGB")
        frames = np.array(img)[np.newaxis, ...]  # (1, H, W, 3)

        # Step 1: Upscale
        def _progress(frac, desc=""):
            if self.is_aborted:
                raise InterruptedError("Aborted by user")
            # Scale progress to 0-0.5 for upscale phase
            phase_frac = frac * 0.5 if self._do_second_pass else frac * 0.9
            self.progress.emit(phase_frac, desc or "Upscaling...")

        self.progress.emit(0.05, "Running upscale...")
        upscaled = upscale_frames(
            frames,
            self._upscale_model_path,
            progress_cb=_progress,
        )

        if self.is_aborted:
            raise InterruptedError("Aborted by user")

        # If model's native scale doesn't match requested scale, resize
        upscaled_img = Image.fromarray(upscaled[0])
        orig_w, orig_h = img.size
        target_w = int(orig_w * self._scale)
        target_h = int(orig_h * self._scale)
        if upscaled_img.size != (target_w, target_h):
            upscaled_img = upscaled_img.resize(
                (target_w, target_h), Image.Resampling.LANCZOS
            )

        # Step 2: Optional img2img second pass
        if self._do_second_pass and self._img_pipeline is not None:
            self.progress.emit(0.5, "Running img2img second pass...")
            try:
                if target_w * target_h > _TILE_PIXEL_THRESHOLD:
                    upscaled_img = self._run_tiled_second_pass(
                        upscaled_img, target_w, target_h
                    )
                else:
                    results = self._run_img2img_second_pass(
                        upscaled_img, target_w, target_h
                    )
                    if results:
                        upscaled_img = results[0]
            except Exception as exc:
                logger.warning("Hires img2img second pass failed: %s", exc)
                self.status.emit(f"Second pass failed ({exc}), using upscaled image only")

        # Save
        self.progress.emit(0.95, "Saving result...")
        out_dir = Path(self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        src_stem = Path(self._source_path).stem
        suffix = f"_hires_{self._scale:.1f}x"
        out_path = out_dir / f"{src_stem}{suffix}.png"
        upscaled_img.save(str(out_path))

        self.progress.emit(1.0, "Done")
        return str(out_path)

    def _run_tiled_second_pass(self, image, width: int, height: int):
        """Refine a large image in overlapping tiles, blended with a feather.

        Each tile is sent through the unchanged family-dispatch
        ``_run_img2img_second_pass`` so SD / Flux / Z-Image all keep working.
        Tiles are accumulated into a float buffer weighted by a cosine feather
        so seams disappear; abort is checked between tiles.
        """
        tile = _probe_tile_size()
        overlap = min(_TILE_OVERLAP, tile // 2)
        step = max(1, tile - overlap)

        src = np.asarray(image.convert("RGB"), dtype=np.float32)

        def _starts(extent: int) -> list:
            if extent <= tile:
                return [0]
            out = list(range(0, extent - tile + 1, step))
            if out[-1] != extent - tile:
                out.append(extent - tile)
            return out

        ys = _starts(height)
        xs = _starts(width)
        total = len(ys) * len(xs)

        accum = np.zeros((height, width, 3), dtype=np.float32)
        weight = np.zeros((height, width, 1), dtype=np.float32)

        idx = 0
        for y0 in ys:
            for x0 in xs:
                if self.is_aborted:
                    raise InterruptedError("Aborted by user")
                th = min(tile, height - y0)
                tw = min(tile, width - x0)
                tile_img = Image.fromarray(
                    np.asarray(image)[y0:y0 + th, x0:x0 + tw]
                ).convert("RGB")

                results = self._run_img2img_second_pass(tile_img, tw, th)
                refined = results[0] if results else tile_img
                refined_np = np.asarray(refined.convert("RGB"), dtype=np.float32)
                # The pipeline may snap dims (e.g. to /8); match the tile back.
                if refined_np.shape[:2] != (th, tw):
                    refined = refined.resize((tw, th), Image.Resampling.LANCZOS)
                    refined_np = np.asarray(
                        refined.convert("RGB"), dtype=np.float32
                    )

                alpha = _feather_alpha(th, tw, overlap)[:, :, None]
                accum[y0:y0 + th, x0:x0 + tw] += refined_np * alpha
                weight[y0:y0 + th, x0:x0 + tw] += alpha

                idx += 1
                self.progress.emit(
                    0.5 + 0.4 * (idx / max(1, total)),
                    f"Refining tile {idx}/{total}...",
                )

        # Any cells no tile touched fall back to the upscaled source.
        uncovered = weight[:, :, 0] <= 1e-6
        blended = accum / np.clip(weight, 1e-6, None)
        if uncovered.any():
            blended[uncovered] = src[uncovered]
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    def _run_img2img_second_pass(self, image, width: int, height: int) -> list:
        """Run the img2img refinement pass, dispatching on the pipeline family.

        The wizard hands us the raw model-level pipeline (SD / Flux / Z-Image),
        each of which has a different img2img signature:
          - SD: ``generate_img2img(..., denoising_strength, cfg_scale, sampler,
            scheduler, ...)``
          - Flux: ``generate_img2img(..., strength, guidance_scale)`` (no
            sampler/scheduler)
          - Z-Image: ``run_img2img(image, prompt, strength, ...)`` (no negative
            prompt, no ``generate_img2img``)
        Calling the SD signature on the others raised TypeError/AttributeError
        and silently fell through to upscale-only, so dispatch explicitly.
        """
        pipe = self._img_pipeline
        cls_name = type(pipe).__name__
        callback = self.make_step_callback(self._steps, "Hires img2img")

        # Z-Image has no generate_img2img / negative prompt.
        if not hasattr(pipe, "generate_img2img") and hasattr(pipe, "run_img2img"):
            return pipe.run_img2img(
                image=image,
                prompt=self._prompt,
                strength=self._denoise_strength,
                width=width,
                height=height,
                steps=self._steps,
                guidance_scale=self._cfg_scale,
                seed=-1,
                callback=callback,
            )

        # Flux uses strength/guidance_scale and has no sampler/scheduler.
        if "Flux" in cls_name:
            return pipe.generate_img2img(
                image=image,
                prompt=self._prompt,
                negative_prompt=self._negative_prompt,
                strength=self._denoise_strength,
                width=width,
                height=height,
                steps=self._steps,
                guidance_scale=self._cfg_scale,
                seed=-1,
                callback=callback,
            )

        # SD 1.5 / SDXL default signature.
        return pipe.generate_img2img(
            image=image,
            prompt=self._prompt,
            negative_prompt=self._negative_prompt,
            denoising_strength=self._denoise_strength,
            width=width,
            height=height,
            steps=self._steps,
            cfg_scale=self._cfg_scale,
            seed=-1,
            sampler=self._sampler,
            scheduler=self._scheduler,
            callback=callback,
        )
