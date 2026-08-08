"""Worker for applying an effect stack to a standalone video file via ffmpeg."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from sdqt.utils.codec import configured_codec_args
from sdqt.workers.base import BaseWorker

logger = logging.getLogger(__name__)


def _build_video_filters(effects: list[dict]) -> str:
    """Build the -vf filter chain string from an effect stack.

    Reuses the per-type builders already defined in timeline.py.
    Returns an empty string when no active effects produce filters.
    Speed, compress, and ai_enhance are handled separately.
    """
    from sdqt.workers.timeline import _FILTER_BUILDERS

    parts: list[str] = []
    for fx in effects:
        if not fx.get("enabled", True):
            continue
        etype = fx.get("type", "")
        if etype in ("speed", "compress", "ai_enhance"):
            continue
        builder = _FILTER_BUILDERS.get(etype)
        if builder:
            frag = builder(fx.get("params", {}))
            if frag:
                parts.append(frag)
    return ",".join(parts)


def _get_speed_factor(effects: list[dict]) -> float:
    """Return the combined speed factor from the effect stack (default 1.0)."""
    factor = 1.0
    for fx in effects:
        if fx.get("type") == "speed" and fx.get("enabled", True):
            factor *= fx.get("params", {}).get("factor", 1.0)
    return factor


def _get_compress_params(effects: list[dict]) -> tuple[float, float] | None:
    """Return (crf, scale) from the last enabled compress effect, or None."""
    for fx in reversed(effects):
        if fx.get("type") == "compress" and fx.get("enabled", True):
            p = fx.get("params", {})
            return (p.get("crf", 23.0), p.get("scale", 1.0))
    return None


def _get_ai_enhance(effects: list[dict]) -> dict | None:
    """Return params from the last enabled ai_enhance effect, or None."""
    for fx in reversed(effects):
        if fx.get("type") == "ai_enhance" and fx.get("enabled", True):
            return fx.get("params", {})
    return None


def _build_atempo_chain(factor: float) -> list[str]:
    """Build atempo filter chain for audio speed change.

    atempo only accepts 0.5–100.0, so we chain for values outside that range.
    """
    if abs(factor - 1.0) < 0.001:
        return []
    parts: list[str] = []
    remaining = factor
    while remaining > 100.0:
        parts.append("atempo=100.0")
        remaining /= 100.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.4f}")
    return parts


class VideoEffectWorker(BaseWorker):
    """Apply an effect stack to a video file, producing a new output file.

    Always works from the original source (nondestructive).
    """

    def __init__(
        self,
        source_path: str,
        effects: list[dict],
        output_path: str | None = None,
        upscaler_dir: str = "",
        project_config=None,
        global_config=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source = source_path
        self._effects = effects
        self._output = output_path
        self._upscaler_dir = upscaler_dir
        self._project_config = project_config
        self._global_config = global_config

    def do_work(self) -> str:
        """Run effects pipeline and return the output path."""
        self.status.emit("Applying effects...")

        if not self._output:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".mp4", delete=False, prefix="vfx_",
            )
            tmp.close()
            self._output = tmp.name

        # Check for AI enhance — needs frame-by-frame processing
        ai_params = _get_ai_enhance(self._effects)
        if ai_params:
            enhanced = self._apply_ai_enhance(self._source, ai_params)
            # Apply remaining effects (color, speed, etc.) on the enhanced video
            remaining = [fx for fx in self._effects if fx.get("type") != "ai_enhance"]
            active_remaining = [fx for fx in remaining if fx.get("enabled", True)]
            if active_remaining:
                self._apply_ffmpeg_effects(enhanced, self._output, remaining)
            else:
                # AI enhance was the only effect
                import shutil
                shutil.move(enhanced, self._output)
        else:
            self._apply_ffmpeg_effects(self._source, self._output, self._effects)

        if not Path(self._output).is_file() or Path(self._output).stat().st_size == 0:
            raise RuntimeError("Effect pipeline produced no output")

        self.status.emit("Effects applied.")
        return self._output

    def _apply_ai_enhance(self, source: str, params: dict) -> str:
        """Extract frames, AI upscale, reassemble to video."""
        from supremediffusion.postprocessing.ai_upscale import (
            upscale_frames, download_default_model, scan_upscaler_models,
        )

        tile_size = int(params.get("tile_size", 512))

        # Find or download model
        upscaler_dir = self._upscaler_dir
        if not upscaler_dir:
            # Fallback to a temp dir
            upscaler_dir = str(Path.home() / ".supremediffusion" / "models" / "upscalers")

        models = scan_upscaler_models(upscaler_dir)
        if models:
            model_path = str(Path(upscaler_dir) / models[0])
        else:
            self.status.emit("Downloading RealESRGAN x4plus model...")
            model_path = download_default_model(
                upscaler_dir,
                progress_cb=lambda f, d: self.status.emit(d),
            )

        # Extract frames
        self.status.emit("Extracting frames...")
        if self.is_aborted:
            raise InterruptedError("Aborted")

        import numpy as np
        from supremediffusion.utils.video import probe_video

        info = probe_video(source)
        fps = info.get("fps", 24)
        n_frames = info.get("num_frames", 0)
        w, h = info.get("width", 0), info.get("height", 0)

        # Use ffmpeg to extract all frames as raw RGB
        frame_dir = tempfile.mkdtemp(prefix="enhance_frames_")
        cmd = [
            "ffmpeg", "-y", "-i", source,
            "-pix_fmt", "rgb24",
            f"{frame_dir}/frame_%06d.png",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Frame extraction failed: {result.stderr[-300:]}")

        # Load frames into numpy
        from PIL import Image
        frame_files = sorted(Path(frame_dir).glob("frame_*.png"))
        if not frame_files:
            raise RuntimeError("No frames extracted")

        self.status.emit(f"Loaded {len(frame_files)} frames, starting AI upscale...")
        frames = np.stack([np.array(Image.open(f).convert("RGB")) for f in frame_files])

        # Upscale
        if self.is_aborted:
            raise InterruptedError("Aborted")

        upscaled = upscale_frames(
            frames, model_path,
            tile_size=tile_size,
            progress_cb=lambda f, d: (
                self.progress.emit(f, d),
                self.status.emit(d),
            ) and None,
        )

        # Save upscaled frames
        self.status.emit("Reassembling video...")
        up_dir = tempfile.mkdtemp(prefix="enhance_up_")
        for i, frame in enumerate(upscaled):
            Image.fromarray(frame).save(f"{up_dir}/frame_{i:06d}.png")

        # Reassemble with audio from original
        out_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, prefix="enhanced_")
        out_tmp.close()

        from sdqt.utils.codec import pix_fmt_args, configured_encoder_name
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", f"{up_dir}/frame_%06d.png",
            "-i", source,
            "-map", "0:v",
            "-map", "1:a?",
            *configured_codec_args(),
            *pix_fmt_args(configured_encoder_name()),
            "-c:a", "aac", "-b:a", "192k",
            out_tmp.name,
        ]
        self._run_ffmpeg(cmd, label="reassemble enhanced video")

        # Cleanup temp frame dirs
        import shutil
        shutil.rmtree(frame_dir, ignore_errors=True)
        shutil.rmtree(up_dir, ignore_errors=True)

        return out_tmp.name

    def _apply_ffmpeg_effects(self, source: str, output: str, effects: list[dict]) -> None:
        """Apply ffmpeg-based effects (color, speed, compress, etc.)."""
        from sdqt.utils.codec import project_input_filter
        # Probe the SOURCE's real range/matrix and convert to the active
        # profile (thread-local set by BaseWorker.run()). The old
        # full_range_filter() assumed every input was full-range — on the
        # app's normal tv-range clips that double-squeezed levels, so every
        # bake washed the clip out even with zero effects.
        vf_parts: list[str] = [
            project_input_filter(source_path=source, add_setsar=False,
                                 add_matrix=True)
        ]

        visual = _build_video_filters(effects)
        if visual:
            vf_parts.append(visual)

        speed = _get_speed_factor(effects)
        if abs(speed - 1.0) > 0.001:
            pts_factor = 1.0 / speed
            vf_parts.append(f"setpts={pts_factor:.4f}*PTS")

        compress = _get_compress_params(effects)
        if compress:
            _, scale = compress
            if scale < 0.99:
                vf_parts.append(
                    f"scale=iw*{scale:.2f}:ih*{scale:.2f}:flags=lanczos"
                )

        cmd: list[str] = ["ffmpeg", "-y", "-i", source]

        if vf_parts:
            cmd += ["-vf", ",".join(vf_parts)]

        af_parts: list[str] = []
        if abs(speed - 1.0) > 0.001:
            af_parts.extend(_build_atempo_chain(speed))
        if af_parts:
            cmd += ["-af", ",".join(af_parts)]

        if compress:
            crf, _ = compress
            crf_int = int(round(crf))
            cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf_int)]
        else:
            cmd += configured_codec_args()

        from sdqt.utils.codec import pix_fmt_args, configured_encoder_name
        cmd += [
            "-c:a", "aac", "-b:a", "192k",
            *pix_fmt_args(configured_encoder_name()),
        ]
        cmd += [output]

        logger.info("VideoEffectWorker cmd: %s", " ".join(cmd))
        self._run_ffmpeg(cmd, label="video effects")
