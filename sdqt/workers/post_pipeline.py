"""Unified post-processing pipeline.

After every generation (Mode 1/2/3, VideoExtender, Pose Animate, SadTalker,
Image Suite outputs), this pipeline runs the canonical post-processing chain
in the **fixed correct order**:

    color anchor → face restore → upscale → frame interp → audio mux → save

Each worker just instantiates a PostProcessPipeline, adds the steps that apply
to its output, and calls ``run()``. The pipeline ENFORCES the correct order
regardless of the order steps were added. Users can't accidentally upscale
before color-anchoring.

Quality presets (Draft / Standard / Quality / Master) live in this module too
— call :func:`build_from_quality_preset` to get a pre-configured pipeline.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from PIL import Image

from sdqt.utils.naming import cap_stem

logger = logging.getLogger(__name__)


# Canonical step order. Any step added is sorted to this position before run.
_STEP_ORDER = (
    "color_anchor",
    "face_restore",
    "upscale",
    "frame_interp",
    "audio_mux",
)


@dataclass
class _Step:
    name: str
    enabled: bool = True
    params: dict = field(default_factory=dict)


class PostProcessPipeline:
    """Ordered post-processing pipeline for generated video/image output."""

    def __init__(
        self,
        state: Any,
        project_name: str,
        *,
        global_config: Any = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        status_cb: Optional[Callable[[str], None]] = None,
        abort_cb: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.state = state
        # GlobalConfig used by the upscale + RIFE steps. Threaded in directly
        # (rather than dereferenced via ``self.state.global_config``) so it
        # works regardless of what ``state`` is — the InferenceWorker passes
        # itself while the Save-Clip worker passes ``state=None``. Falls back to
        # ``state.global_config`` for callers that only supply a state object.
        self.global_config = global_config
        if self.global_config is None and state is not None:
            self.global_config = getattr(state, "global_config", None)
        self.project_name = project_name
        self._steps: dict[str, _Step] = {}
        self._progress = progress_cb or (lambda f, d: None)
        self._status = status_cb or (lambda s: None)
        self._aborted = abort_cb or (lambda: False)

    # ------------------------------------------------------------------
    # Step builders (each adds / replaces a step)
    # ------------------------------------------------------------------

    def add_color_anchor(
        self,
        *,
        reference_image: Optional[Image.Image] = None,
        method: str = "mean-only-lab",
        strength: float = 0.7,
        persistence: float = 0.6,
    ) -> "PostProcessPipeline":
        """Color-anchor the output frames against ``reference_image``."""
        self._steps["color_anchor"] = _Step(
            name="color_anchor",
            params={
                "reference_image": reference_image,
                "method": method,
                "strength": strength,
                "persistence": persistence,
            },
        )
        return self

    def add_face_restore(
        self,
        *,
        method: str = "gfpgan",
        strength: float = 0.5,
        blend: float = 0.6,
    ) -> "PostProcessPipeline":
        """Face restoration via the existing face_swap models (GFPGAN, etc.)."""
        self._steps["face_restore"] = _Step(
            name="face_restore",
            params={"method": method, "strength": strength, "blend": blend},
        )
        return self

    def add_upscale(
        self,
        *,
        model_path: str,
        tile_size: int = 512,
        tile_overlap: int = 32,
    ) -> "PostProcessPipeline":
        """Tile-based ESRGAN upscale via UpscalerPipeline."""
        self._steps["upscale"] = _Step(
            name="upscale",
            params={
                "model_path": model_path,
                "tile_size": tile_size,
                "tile_overlap": tile_overlap,
            },
        )
        return self

    def add_frame_interp(self, *, multiplier: float = 2.0) -> "PostProcessPipeline":
        """Frame interpolation (currently ffmpeg minterpolate)."""
        self._steps["frame_interp"] = _Step(
            name="frame_interp",
            params={"multiplier": multiplier},
        )
        return self

    def add_audio_mux(self, audio_path: str) -> "PostProcessPipeline":
        """Mux an audio track onto the final video."""
        self._steps["audio_mux"] = _Step(
            name="audio_mux",
            params={"audio_path": audio_path},
        )
        return self

    def disable(self, step_name: str) -> "PostProcessPipeline":
        """Disable a step that was previously added."""
        if step_name in self._steps:
            self._steps[step_name].enabled = False
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, video_path: str, *, fps: int = 16) -> str:
        """Run the configured pipeline on ``video_path``. Returns the final path.

        Steps execute in canonical order regardless of add-order. Disabled
        steps are skipped. Each step writes a new mp4 alongside the source.
        """
        from sdqt.utils.codec import configured_codec_args

        current = Path(video_path)
        if not current.is_file():
            raise FileNotFoundError(video_path)

        # Filter + sort by canonical order
        active = [
            self._steps[name] for name in _STEP_ORDER
            if name in self._steps and self._steps[name].enabled
        ]
        if not active:
            return str(current)

        self._status(f"Post-processing: {len(active)} step(s)")
        for idx, step in enumerate(active):
            if self._aborted():
                raise InterruptedError("Aborted by user")

            stage_frac_low = idx / len(active)
            stage_frac_high = (idx + 1) / len(active)

            def _stage_progress(local_frac: float, desc: str) -> None:
                f = stage_frac_low + (stage_frac_high - stage_frac_low) * local_frac
                self._progress(f, f"[{step.name}] {desc}")

            self._status(f"Step {idx+1}/{len(active)}: {step.name}")
            current = self._run_step(step, current, fps=fps, progress_cb=_stage_progress)

        self._progress(1.0, "Complete")
        return str(current)

    # ------------------------------------------------------------------
    def _run_step(
        self,
        step: _Step,
        input_path: Path,
        *,
        fps: int,
        progress_cb: Callable[[float, str], None],
    ) -> Path:
        """Dispatch to the right backend for this step. Returns new file path."""
        if step.name == "color_anchor":
            return self._step_color_anchor(step, input_path, fps=fps, progress_cb=progress_cb)
        if step.name == "face_restore":
            return self._step_face_restore(step, input_path, fps=fps, progress_cb=progress_cb)
        if step.name == "upscale":
            return self._step_upscale(step, input_path, fps=fps, progress_cb=progress_cb)
        if step.name == "frame_interp":
            return self._step_frame_interp(step, input_path, progress_cb=progress_cb)
        if step.name == "audio_mux":
            return self._step_audio_mux(step, input_path, progress_cb=progress_cb)
        logger.warning("Unknown post-step: %s — skipping", step.name)
        return input_path

    # ----- color anchor (uses existing post_color_anchor module) ------
    def _step_color_anchor(
        self, step: _Step, in_path: Path, *, fps: int,
        progress_cb: Callable[[float, str], None],
    ) -> Path:
        from supremediffusion.core.post_color_anchor import apply_color_anchor
        ref = step.params.get("reference_image")
        if ref is None:
            progress_cb(1.0, "no reference, skipped")
            return in_path
        out_path = in_path.with_name(cap_stem(in_path.stem) + "_ca.mp4")
        frames = self._load_frames(
            in_path, progress_cb=self._subrange(progress_cb, 0.0, 0.5),
        )
        progress_cb(0.55, "color anchoring…")
        out = apply_color_anchor(
            frames, ref,
            method=step.params.get("method", "mean-only-lab"),
            strength=float(step.params.get("strength", 0.7)),
            persistence=float(step.params.get("persistence", 0.6)),
        )
        self._save_frames(out, out_path, fps=fps)
        progress_cb(1.0, "done")
        return out_path

    # ----- face restore (uses face_swap pipeline) ---------------------
    def _step_face_restore(
        self, step: _Step, in_path: Path, *, fps: int,
        progress_cb: Callable[[float, str], None],
    ) -> Path:
        # The cheap path: run face_swap's enhancer over each frame, no swap.
        # face_swap.py already has _run_enhancer which takes an aligned crop —
        # but a simpler approach is to just batch-enhance each frame's face
        # using the existing ADetailer logic. For now, **defer to ADetailer**:
        # if ADetailer-style face refinement is needed, the worker should call
        # the ADetailer step directly. Post-step face restore is a planned
        # extension (Phase 3.5). For Phase 3, fall through.
        progress_cb(1.0, "skipped (use ADetailer in image gen path)")
        return in_path

    # ----- upscale (uses UpscalerPipeline) ----------------------------
    def _step_upscale(
        self, step: _Step, in_path: Path, *, fps: int,
        progress_cb: Callable[[float, str], None],
    ) -> Path:
        from supremediffusion.models.upscaler_pipeline import get_upscaler
        model_path = step.params.get("model_path", "")
        if not model_path or not Path(model_path).is_file():
            progress_cb(1.0, f"model missing: {model_path}")
            return in_path

        # Shared cached instance — keeps weights resident across runs (batch /
        # timeline clips) instead of reloading spandrel every call. Freed on
        # GPU swap via release_upscaler() / the next pipeline load.
        up = get_upscaler(model_path, self.global_config)
        tile_size = int(step.params.get("tile_size", 512))
        tile_overlap = int(step.params.get("tile_overlap", 32))
        frames = self._load_frames(
            in_path, progress_cb=self._subrange(progress_cb, 0.0, 0.15),
        )
        out_frames = up.upscale_frames(
            frames, tile_size=tile_size, tile_overlap=tile_overlap,
            progress_cb=self._subrange(progress_cb, 0.15, 0.95),
        )
        out_path = in_path.with_name(cap_stem(in_path.stem) + f"_up{up.scale}x.mp4")
        self._save_frames(out_frames, out_path, fps=fps)
        progress_cb(1.0, "done")
        return out_path

    # ----- frame interp (uses RIFEPipeline / ffmpeg) -------------------
    def _step_frame_interp(
        self, step: _Step, in_path: Path,
        progress_cb: Callable[[float, str], None],
    ) -> Path:
        from supremediffusion.models.rife_pipeline import RIFEPipeline
        rp = RIFEPipeline(self.global_config)
        out_path = in_path.with_name(cap_stem(in_path.stem) + "_int.mp4")
        rp.interpolate_video(
            str(in_path), str(out_path),
            multiplier=float(step.params.get("multiplier", 2.0)),
            progress_cb=progress_cb,
        )
        return out_path

    # ----- audio mux --------------------------------------------------
    def _step_audio_mux(
        self, step: _Step, in_path: Path,
        progress_cb: Callable[[float, str], None],
    ) -> Path:
        from sdqt.utils.codec import configured_codec_args
        audio_path = step.params.get("audio_path", "")
        if not audio_path or not Path(audio_path).is_file():
            progress_cb(1.0, "no audio, skipped")
            return in_path
        out_path = in_path.with_name(cap_stem(in_path.stem) + "_aud.mp4")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(in_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        progress_cb(1.0, "done")
        return out_path

    # ----- I/O helpers ------------------------------------------------
    @staticmethod
    def _subrange(
        cb: Callable[[float, str], None], lo: float, hi: float,
    ) -> Callable[[float, str], None]:
        """Map an inner 0..1 progress callback onto the [lo, hi] sub-range."""
        span = hi - lo

        def _inner(frac: float, desc: str) -> None:
            cb(lo + max(0.0, min(1.0, frac)) * span, desc)

        return _inner

    @staticmethod
    def _load_frames(
        path: Path,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> List[Image.Image]:
        """Decode all frames from a video as PIL.Image.

        Uses PyAV when available (fast) — falls back to ffmpeg → PNG dump.
        Reports ``Extracting frame i/total`` via *progress_cb* so the user can
        see how far along extraction is (and gauge remaining time).
        """
        cb = progress_cb or (lambda f, d: None)
        try:
            import av  # noqa: F401
        except ImportError:
            return PostProcessPipeline._load_frames_ffmpeg(path, progress_cb=progress_cb)
        import av
        frames: List[Image.Image] = []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            total = int(stream.frames or 0)
            if total <= 0:
                # stream.frames isn't always populated — estimate from duration.
                try:
                    dur = float(container.duration or 0) / 1_000_000.0  # AV_TIME_BASE
                    rate = float(stream.average_rate or 0)
                    total = int(round(dur * rate)) if dur > 0 and rate > 0 else 0
                except Exception:
                    total = 0
            emit_every = max(1, total // 200) if total else 10
            i = 0
            for frame in container.decode(video=0):
                frames.append(frame.to_image())
                i += 1
                if total:
                    if i == total or i % emit_every == 0:
                        cb(i / total, f"Extracting frame {i}/{total}")
                elif i % emit_every == 0:
                    cb(0.0, f"Extracting frame {i}")
        cb(1.0, f"Extracted {len(frames)} frames")
        return frames

    @staticmethod
    def _load_frames_ffmpeg(
        path: Path,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> List[Image.Image]:
        cb = progress_cb or (lambda f, d: None)
        d = Path(tempfile.mkdtemp(prefix="post_frames_"))
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                 "-q:v", "2", f"{d}/f_%06d.png"],
                check=True,
            )
            png_files = sorted(d.iterdir())
            total = len(png_files)
            emit_every = max(1, total // 200) if total else 1
            frames: List[Image.Image] = []
            for i, p in enumerate(png_files, 1):
                frames.append(Image.open(p).convert("RGB"))
                if total and (i == total or i % emit_every == 0):
                    cb(i / total, f"Extracting frame {i}/{total}")
            return frames
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    @staticmethod
    def _save_frames(frames: Sequence[Image.Image], path: Path, *, fps: int) -> None:
        from sdqt.utils.codec import configured_codec_args
        d = Path(tempfile.mkdtemp(prefix="post_save_"))
        try:
            for i, f in enumerate(frames):
                f.save(d / f"f_{i:06d}.png")
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error",
                 "-framerate", str(fps), "-i", f"{d}/f_%06d.png",
                 *configured_codec_args(),
                 "-pix_fmt", "yuv420p",
                 str(path)],
                check=True,
            )
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Quality preset → pipeline builder
# ---------------------------------------------------------------------------


# UI-visible preset names
QUALITY_PRESETS = ("draft", "standard", "quality", "master")
QUALITY_LABELS = {
    "draft": "Draft (raw — no post-processing)",
    "standard": "Standard (color anchor only)",
    "quality": "Quality (+2× upscale)",
    "master": "Master (+4× upscale + RIFE 2× fps)",
}


def build_from_quality_preset(
    preset: str,
    *,
    state: Any,
    project_name: str,
    global_config: Any = None,
    source_image: Optional[Image.Image] = None,
    color_method: str = "mean-only-lab",
    color_strength: float = 0.7,
    color_persistence: float = 0.6,
    upscale_model_path: Optional[str] = None,
    rife_multiplier: float = 2.0,
    audio_path: Optional[str] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    status_cb: Optional[Callable[[str], None]] = None,
    abort_cb: Optional[Callable[[], bool]] = None,
) -> PostProcessPipeline:
    """Return a PostProcessPipeline configured for the preset."""
    p = PostProcessPipeline(
        state, project_name, global_config=global_config,
        progress_cb=progress_cb, status_cb=status_cb, abort_cb=abort_cb,
    )

    if preset == "draft":
        # No post-processing at all
        return p

    # Standard and up: always color anchor (if reference provided)
    if source_image is not None:
        p.add_color_anchor(
            reference_image=source_image,
            method=color_method,
            strength=color_strength,
            persistence=color_persistence,
        )

    if preset in ("quality", "master") and upscale_model_path:
        p.add_upscale(model_path=upscale_model_path)

    if preset == "master":
        p.add_frame_interp(multiplier=rife_multiplier)

    if audio_path:
        p.add_audio_mux(audio_path)

    return p
