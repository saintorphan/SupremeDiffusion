"""Quick Export worker — fixed post-processing preset for a single clip.

Applies denoise, sharpen, face restore, lanczos 1.5x, and RIFE 2x in one pass.
Designed to run unattended from the right-click "Quick Export" action.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .base import BaseWorker
from sdqt.utils.codec import (
    configured_codec_args as _codec_args,
    pix_fmt_args as _pix_fmt,
    configured_encoder_name as _enc_name,
)

logger = logging.getLogger(__name__)


class ClipPostProcessWorker(BaseWorker):
    """Run an AI post-process pass + optional RIFE over a single clip.

    Each step is individually toggleable via ``enable_*`` kwargs. Defaults
    match the legacy Quick Export preset (denoise + sharpen + face restore
    + Lanczos 1.5x + RIFE 2x) so existing callers keep working.
    """

    def __init__(
        self,
        source: str,
        output: str,
        *,
        media_offset: float = 0.0,
        duration: float = 0.0,
        tile: int = 512,
        enable_denoise: bool = True,
        enable_sharpen: bool = True,
        enable_face: bool = True,
        enable_upscale: bool = False,
        lanczos_mode: str = "lanczos1.5",  # "" to skip
        rife_mode: str = "x2",              # "" to skip
        grain: float = 0.0,                 # 0.0–1.0 — reserved, no-op for now
        effects: list[dict] | None = None,  # timeline clip effect stack to bake in
        upscaler_dir: str = "",
        face_models_dir: str = "",
        godot_mode: bool = False,
        target_fps: float = 0,
        project_config=None,
        global_config=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._output = output
        self._media_offset = float(media_offset or 0.0)
        self._duration = float(duration or 0.0)
        self._tile = tile
        self._enable_denoise = enable_denoise
        self._enable_sharpen = enable_sharpen
        self._enable_face = enable_face
        self._enable_upscale = enable_upscale
        self._lanczos_mode = lanczos_mode or ""
        self._rife_mode = rife_mode or ""
        self._grain = grain
        self._effects = list(effects) if effects else []
        self._upscaler_dir = upscaler_dir
        self._face_models_dir = face_models_dir
        self._godot_mode = godot_mode
        self._target_fps = target_fps
        # Plumbed onto BaseWorker so run() pushes the project's color profile
        # onto the thread-local — every ffmpeg call below picks it up.
        self._project_config = project_config
        self._global_config = global_config
        if godot_mode:
            # Legacy callers passing godot_mode=True want the bt709_limited /
            # yuv420p+tv look unconditionally. Override the per-project profile
            # for this run so existing Godot exports keep working as they did.
            from supremediffusion.config.color_profile import get_profile
            self._color_profile = get_profile("bt709_limited")
        self._baked_source: str | None = None  # intermediate file holding the effect-baked clip

    # ------------------------------------------------------------------

    def do_work(self) -> str:
        from supremediffusion.postprocessing.ai_upscale import (
            denoise_frames, sharpen_frames, face_restore_frames, upscale_frames,
            _resolve_model,
            DEFAULT_DENOISER, DEFAULT_SHARPENER, DEFAULT_UPSCALER,
        )
        from supremediffusion.postprocessing.spatial import spatial_upsample

        upscaler_dir = self._upscaler_dir
        if not upscaler_dir:
            upscaler_dir = str(Path.home() / ".supremediffusion" / "models" / "upscalers")

        logger.info(
            "ClipPostProcessWorker start: source=%s effects=%d (%s) "
            "media_offset=%.3f duration=%.3f",
            self._source, len(self._effects),
            [fx.get("type") for fx in self._effects],
            self._media_offset, self._duration,
        )

        # Bake the timeline clip's effect stack + trim into a temp mp4 so every
        # downstream step (decode, AI post-process, mux) sees the exact same
        # pixels the user saw in the timeline preview. The AI preset then
        # runs on those baked pixels.
        if self._effects or self._duration > 0.01 or self._media_offset > 0.01:
            self.progress.emit(0.01, "Baking timeline effects…")
            self._baked_source = self._bake_effects_and_trim(self._source)
            self._source = self._baked_source
            logger.info("Baked intermediate: %s", self._baked_source)
            # The baked file already has the trim applied, so the downstream
            # decode/mux helpers should not re-trim.
            self._media_offset = 0.0
            self._duration = 0.0

        self.progress.emit(0.02, "Probing source...")
        w, h, source_fps = self._probe_source(self._source)
        # Process and encode at the source fps so duration is preserved
        # end-to-end. Honoring the timeline's target fps is handled later:
        #   - RIFE on → output is source_fps × 2 (or × 4); RIFE wins.
        #   - RIFE off and target ≠ source → ffmpeg fps= resample at the end.
        # The previous code overrode fps = target_fps here and then encoded
        # the same frame count at the higher rate, which silently shrank
        # duration (16 fps × 4 s → re-tagged 24 fps → 2.67 s).
        fps = source_fps

        needs_frames = (
            self._enable_denoise or self._enable_sharpen
            or self._enable_face or self._enable_upscale
            or bool(self._lanczos_mode)
        )

        if needs_frames:
            self.progress.emit(0.05, "Decoding frames...")
            frames = self._decode_frames(self._source, w, h)
            if self.is_aborted:
                raise InterruptedError("Aborted")
            if frames.size == 0:
                raise RuntimeError("Could not decode any frames from source")

            if self._enable_denoise:
                self.progress.emit(0.10, "AI Denoise...")
                denoiser = _resolve_model(
                    upscaler_dir, DEFAULT_DENOISER,
                    progress_cb=lambda f, d: self.progress.emit(0.10, d),
                )
                frames = denoise_frames(
                    frames, denoiser, tile_size=self._tile,
                    progress_cb=lambda f, d: self.progress.emit(0.10 + 0.15 * f, d),
                )
                if self.is_aborted:
                    raise InterruptedError("Aborted")

            if self._enable_sharpen:
                self.progress.emit(0.25, "AI Sharpen...")
                sharpener = _resolve_model(
                    upscaler_dir, DEFAULT_SHARPENER,
                    progress_cb=lambda f, d: self.progress.emit(0.25, d),
                )
                frames = sharpen_frames(
                    frames, sharpener, tile_size=self._tile,
                    progress_cb=lambda f, d: self.progress.emit(0.25 + 0.15 * f, d),
                )
                if self.is_aborted:
                    raise InterruptedError("Aborted")

            if self._enable_face:
                self.progress.emit(0.40, "Face Restore...")
                gfpgan_path = self._resolve_gfpgan(upscaler_dir)
                frames = face_restore_frames(
                    frames, gfpgan_path, tile_size=0, half=False,
                    progress_cb=lambda f, d: self.progress.emit(0.40 + 0.15 * f, d),
                )
                if self.is_aborted:
                    raise InterruptedError("Aborted")

            if self._enable_upscale:
                self.progress.emit(0.48, "AI Upscale 4x...")
                up_model = _resolve_model(
                    upscaler_dir, DEFAULT_UPSCALER,
                    progress_cb=lambda f, d: self.progress.emit(0.48, d),
                )
                frames = upscale_frames(
                    frames, up_model, tile_size=self._tile,
                    progress_cb=lambda f, d: self.progress.emit(0.48 + 0.10 * f, d),
                )
                if self.is_aborted:
                    raise InterruptedError("Aborted")

            if self._lanczos_mode:
                self.progress.emit(0.55, f"{self._lanczos_mode} upscale...")
                frames = spatial_upsample(frames, self._lanczos_mode)
                if self.is_aborted:
                    raise InterruptedError("Aborted")

            self.progress.emit(0.60, "Encoding video...")
            vid_tmp = self._encode_frames(frames, fps)
            if self.is_aborted:
                raise InterruptedError("Aborted")

            del frames
            import gc; gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        else:
            # No frame-level AI steps. The fps resample (if any) runs at the
            # end and reads from whatever vid_tmp ends up pointing at, so we
            # only need an early re-encode when Godot color args force one.
            if self._godot_mode:
                self.progress.emit(0.05, "Re-encoding for Godot...")
                frames = self._decode_frames(self._source, w, h)
                vid_tmp = self._encode_frames(frames, fps)
                del frames
            else:
                vid_tmp = self._source

        if self._rife_mode:
            self.progress.emit(0.70, f"RIFE {self._rife_mode} temporal upscale...")
            rife_tmp = self._apply_rife(vid_tmp)
            if self.is_aborted:
                raise InterruptedError("Aborted")
        else:
            rife_tmp = vid_tmp

        # Honest fps resample (duplicate/drop) when RIFE is off and the
        # timeline's target fps differs from the source. With RIFE on the
        # output is already at source_fps × 2 (or × 4) — that's the multiplier
        # the user picked, so don't re-flatten back down to target_fps.
        if (
            not self._rife_mode
            and self._target_fps > 0
            and abs(self._target_fps - source_fps) > 0.5
        ):
            self.progress.emit(
                0.88,
                f"Resampling fps {source_fps:.1f} → {self._target_fps:.1f}…",
            )
            rife_tmp = self._convert_fps(rife_tmp, self._target_fps)

        self.progress.emit(0.92, "Muxing audio...")
        self._mux_audio(rife_tmp, self._source, self._output)

        self.progress.emit(1.0, "Post-process complete")
        return self._output

    # ------------------------------------------------------------------

    def _bake_effects_and_trim(self, src: str) -> str:
        """Bake the clip's effect stack + trim to a temp mp4.

        Matches ``VideoEffectWorker._apply_ffmpeg_effects`` — the proven-
        working "Bake Effects" pipeline from the timeline context menu.
        Minimal filter chain: ``full_range_filter()`` + the effect filters.
        Input-side ``-ss``/``-t`` for trim so we don't need trim/setpts
        filters. Output uses the app's standard ``-pix_fmt yuv420p
        -color_range pc`` convention.
        """
        from sdqt.utils.codec import project_input_filter
        from sdqt.workers.video_effects import _build_video_filters

        out = tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False, prefix="clip_qe_bake_",
        )
        out.close()

        # Probe the source's real range/matrix and convert to the active
        # profile (thread-local set by BaseWorker.run()). The old
        # full_range_filter() assumed full-range input, which washed out the
        # app's normal tv-range clips on every bake.
        vf_parts: list[str] = [
            project_input_filter(source_path=src, add_setsar=False,
                                 add_matrix=True)
        ]
        visual = _build_video_filters(self._effects) if self._effects else ""
        if visual:
            vf_parts.append(visual)
        elif self._effects:
            logger.warning(
                "Effects present but _build_video_filters returned empty. "
                "Effect types: %s",
                [fx.get("type") for fx in self._effects],
            )
            for fx in self._effects:
                if fx.get("type") == "lut3d":
                    lut_path = (fx.get("params") or {}).get("lut_file", "")
                    logger.warning(
                        "  lut3d lut_file=%r exists=%s enabled=%s",
                        lut_path, Path(lut_path).is_file() if lut_path else False,
                        fx.get("enabled", True),
                    )
        vf = ",".join(vf_parts)
        logger.info("Bake filter chain: %s", vf)

        # Input-side seek/duration — faster than a trim filter and
        # avoids the need for a setpts reset.
        cmd: list[str] = ["ffmpeg", "-y", "-v", "error"]
        if self._media_offset > 0.01:
            cmd += ["-ss", f"{self._media_offset:.4f}"]
        cmd += ["-i", src]
        if self._duration > 0.01:
            cmd += ["-t", f"{self._duration:.4f}"]
        cmd += [
            "-vf", vf,
            *_codec_args(),
            "-c:a", "aac", "-b:a", "192k",
            *_pix_fmt(_enc_name()),
            out.name,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            # Retry without audio — source may have no audio stream.
            cmd_no_a = [c for c in cmd if c not in ("-c:a", "copy")]
            cmd_no_a = cmd_no_a[:-1]  # drop output
            cmd_no_a = [c for c in cmd_no_a if c not in ("-c:a", "aac", "-b:a", "192k")]
            cmd_no_a += ["-an", out.name]
            result2 = subprocess.run(cmd_no_a, capture_output=True, text=True, timeout=600)
            if result2.returncode != 0:
                raise RuntimeError(
                    f"Effect bake failed: {result2.stderr[-400:] if result2.stderr else 'unknown'}"
                )
        return out.name

    @staticmethod
    def _probe_color_meta(path: str) -> dict:
        """Probe a video's colour metadata so we can preserve it on re-encode."""
        out: dict = {}
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries",
                 "stream=color_range,color_space,color_primaries,color_transfer",
                 "-of", "default=nw=1:nk=0", path],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.strip().splitlines():
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip()
                if v and v.lower() not in ("unknown", "n/a", "und"):
                    out[k.strip()] = v
        except Exception:
            pass
        return out

    @staticmethod
    def _probe_source(path: str) -> tuple[int, int, float]:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate",
             "-of", "csv=p=0:s=x", path],
            capture_output=True, text=True, check=True,
        )
        parts = result.stdout.strip().split("x")
        w = int(parts[0])
        h = int(parts[1])
        fps_expr = parts[2] if len(parts) > 2 else "25/1"
        if "/" in fps_expr:
            num, den = fps_expr.split("/")
            fps = float(num) / float(den) if float(den) else 25.0
        else:
            fps = float(fps_expr)
        return w, h, fps

    def _decode_frames(self, src: str, w: int, h: int) -> np.ndarray:
        cmd = ["ffmpeg", "-v", "error"]
        if self._media_offset > 0.01:
            cmd += ["-ss", f"{self._media_offset:.3f}"]
        cmd += ["-i", src]
        if self._duration > 0.01:
            cmd += ["-t", f"{self._duration:.3f}"]
        cmd += [
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-an", "pipe:1",
        ]
        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg decode failed: {result.stderr.decode(errors='replace')[-400:]}"
            )
        expected = w * h * 3
        data = result.stdout
        n = len(data) // expected
        if n == 0:
            return np.empty((0, h, w, 3), dtype=np.uint8)
        return np.frombuffer(
            data[: n * expected], dtype=np.uint8
        ).reshape(n, h, w, 3)

    def _encode_frames(self, frames: np.ndarray, fps: float) -> str:
        """Pipe (N,H,W,3) uint8 RGB frames to ffmpeg → temp mp4 (video only).

        The ``scale=in_range=full:out_range=full`` prefix is CRITICAL: without
        it, ffmpeg's implicit RGB→YUV conversion uses BT.601 tv-range and
        squeezes [0,255] into Y[16,235]. The output gets tagged ``pc`` so
        decoders read Y=16 as full-range black → RGB 16 → grey, which eats
        any contrast the LUT baked in.
        """
        from sdqt.utils.codec import pix_fmt_args

        n, h, w, _ = frames.shape
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, prefix="clip_qe_vid_")
        tmp.close()
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", f"{fps:.6f}",
            "-i", "pipe:0",
            *_codec_args(),
            *_pix_fmt(_enc_name()),
            "-an", tmp.name,
        ]
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            # Write in chunks to avoid one giant allocation
            chunk = 16
            for i in range(0, n, chunk):
                if self.is_aborted:
                    break
                proc.stdin.write(frames[i:i + chunk].tobytes())
                if (i // chunk) % 4 == 0:
                    self.progress.emit(
                        0.60 + 0.10 * (i / max(n, 1)),
                        f"Encoding {i}/{n}",
                    )
        except (BrokenPipeError, OSError):
            pass
        try:
            proc.stdin.close()
        except OSError:
            pass
        proc.stdin = None
        _, stderr = proc.communicate(timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg encode failed: {stderr.decode(errors='replace')[-400:]}"
            )
        return tmp.name

    def _convert_fps(self, input_path: str, target_fps: float) -> str:
        """Resample to *target_fps* via ffmpeg's fps= filter (duplicate/drop).

        Preserves duration — only the frame timing changes. Used when the
        timeline asks for a different rate than the source and RIFE isn't
        in play (RIFE produces a richer interpolation and runs separately).
        """
        out = tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False, prefix="clip_qe_fps_",
        )
        out.close()
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", input_path,
            "-vf", f"fps={target_fps}",
            *_codec_args(),
            *_pix_fmt(_enc_name()),
            "-an", out.name,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(
                f"fps resample failed: {result.stderr[-300:]}"
            )
        return out.name

    def _apply_rife(self, input_path: str) -> str:
        """Run RIFE temporal upscaling via PostProcessor.process(), which
        is the real (working) API — the old ``temporal_upscale`` method
        never existed."""
        try:
            import numpy as np
            from supremediffusion.postprocessing import PostProcessor

            w, h, fps = self._probe_source(input_path)
            frames = self._decode_frames(input_path, w, h)
            if frames.size == 0:
                return input_path

            mode = "rife2" if self._rife_mode == "x2" else "rife4"
            pp = PostProcessor()
            frames_out, out_fps = pp.process(
                frames, int(round(fps)),
                temporal_upsampling=mode,
                progress_cb=lambda f, d: self.progress.emit(
                    0.70 + 0.20 * f, f"RIFE {self._rife_mode}: {d}"
                ),
            )
            del pp

            # Encode the interpolated frames back at the new fps
            return self._encode_frames(frames_out, out_fps)
        except Exception:
            logger.warning("RIFE failed, returning non-interpolated video", exc_info=True)
            return input_path
        finally:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except ImportError:
                pass

    def _mux_audio(self, video_path: str, audio_src: str, output: str) -> None:
        """Copy video stream and mux trimmed audio from the original source."""
        has_audio = self._probe_has_audio(audio_src)
        cmd = ["ffmpeg", "-y", "-v", "error"]
        if has_audio and self._media_offset > 0.01:
            cmd += ["-ss", f"{self._media_offset:.3f}"]
        cmd += ["-i", video_path]
        if has_audio:
            cmd += ["-i", audio_src]
            if self._duration > 0.01:
                cmd += ["-t", f"{self._duration:.3f}"]
            cmd += [
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-shortest",
            ]
        else:
            cmd += ["-map", "0:v:0", "-c:v", "copy", "-an"]
        cmd.append(output)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            # Fallback: just copy the video file
            import shutil
            shutil.copy(video_path, output)
            logger.warning("Mux failed, wrote silent copy: %s", result.stderr[-300:])

    @staticmethod
    def _probe_has_audio(path: str) -> bool:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=10,
            )
            return "audio" in result.stdout.strip()
        except Exception:
            return False

    def _resolve_gfpgan(self, upscaler_dir: str) -> str:
        """Resolve a GFPGAN v1.4 checkpoint path."""
        from supremediffusion.postprocessing.ai_upscale import _resolve_model

        face_dir = self._face_models_dir
        if face_dir:
            candidate = Path(face_dir) / "GFPGANv1.4.pth"
            if candidate.is_file():
                return str(candidate)
        return _resolve_model(
            upscaler_dir, "GFPGANv1.4.pth",
            progress_cb=lambda f, d: self.progress.emit(0.40, d),
        )


# ---------------------------------------------------------------------------
# Back-compat alias: the legacy Quick Export preset.
# ---------------------------------------------------------------------------

class ClipQuickExportWorker(ClipPostProcessWorker):
    """Fixed-preset post-process: denoise + sharpen + face + lanczos 1.5x + RIFE 2x.

    Kept as a thin subclass so existing right-click Quick Export callers
    don't need to know about the flag-based constructor.
    """

    def __init__(
        self,
        source: str,
        output: str,
        *,
        media_offset: float = 0.0,
        duration: float = 0.0,
        tile: int = 512,
        upscaler_dir: str = "",
        face_models_dir: str = "",
        parent=None,
    ) -> None:
        super().__init__(
            source=source,
            output=output,
            media_offset=media_offset,
            duration=duration,
            tile=tile,
            enable_denoise=True,
            enable_sharpen=True,
            enable_face=True,
            enable_upscale=False,
            lanczos_mode="lanczos1.5",
            rife_mode="x2",
            upscaler_dir=upscaler_dir,
            face_models_dir=face_models_dir,
            parent=parent,
        )
