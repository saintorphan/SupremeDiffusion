"""Frame interpolation pipeline (RIFE + ffmpeg fallback).

This module exposes a single :class:`RIFEPipeline` with two interpolation
backends, selected automatically per call:

1. **Real RIFE v4** — if a RIFE weights file resolves (from
   ``global_config.model_paths["rife_model"]`` or auto-discovery) *and* the
   requested ``multiplier`` is a power of two (2.0 → 2×, 4.0 → 4×), the input
   video is decoded to an ``(N, H, W, 3)`` uint8 array, run through the neural
   RIFE network (:func:`supremediffusion.postprocessing.rife.interpolate.
   temporal_interpolation`), then re-encoded. This is the high-quality path.

2. **ffmpeg ``minterpolate``** — the zero-dependency fallback used whenever no
   RIFE weights are available or the multiplier isn't a clean power of two
   (e.g. 1.5×, 3×). Motion-compensated, lower quality than RIFE 4.x but always
   available.

Interface mirrors the other pipeline wrappers so PostProcessPipeline can call
:meth:`interpolate_video` without caring which backend ran.

For 2× fps interpolation (16 → 32, 24 → 48, 30 → 60): ``multiplier = 2.0``.
"""

from __future__ import annotations

import logging
import math
import subprocess
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Weight filename patterns we auto-discover when no explicit path is configured.
_RIFE_WEIGHT_GLOBS = ("rife*.pkl", "flownet*.pkl")


class RIFEPipeline:
    """Stateless frame-interpolation wrapper.

    Picks the real RIFE v4 neural backend when weights resolve and the
    multiplier is a power of two; otherwise falls back to ffmpeg
    motion-compensated interpolation (``minterpolate``).
    """

    def __init__(self, global_config: object) -> None:
        self.config = global_config
        self._loaded: bool = True  # nothing to load eagerly; backends are lazy

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """No-op (backends load lazily). Provided for API symmetry."""
        return

    def unload(self) -> None:
        """No-op."""
        return

    # ------------------------------------------------------------------
    # Weights resolution
    # ------------------------------------------------------------------
    def _resolve_rife_weights(self) -> Optional[Path]:
        """Resolve a RIFE weights file path, or None if none found.

        Order:
          1. ``global_config.model_paths["rife_model"]`` if set and exists.
          2. Auto-discovery: glob ``rife*.pkl`` / ``flownet*.pkl`` across the
             directories of every configured ``model_paths`` value and their
             parents (so a shared ``chkpts/`` root gets scanned even if no
             path points straight at it).
        """
        model_paths = {}
        cfg = getattr(self.config, "model_paths", None)
        if isinstance(cfg, dict):
            model_paths = cfg

        # 1. Explicit configured path.
        explicit = str(model_paths.get("rife_model", "") or "").strip()
        if explicit:
            p = Path(explicit).expanduser()
            if p.is_file():
                return p
            logger.warning("Configured rife_model path does not exist: %s", explicit)

        # 2. Auto-discovery across candidate roots derived from model_paths.
        candidate_roots: list[Path] = []
        seen: set[Path] = set()

        def _add_root(path: Path) -> None:
            try:
                path = path.expanduser().resolve()
            except Exception:  # noqa: BLE001
                return
            if path in seen:
                return
            seen.add(path)
            candidate_roots.append(path)

        for value in model_paths.values():
            sval = str(value or "").strip()
            if not sval:
                continue
            vp = Path(sval).expanduser()
            # The value may be a file (a checkpoint) or a directory.
            base = vp if vp.is_dir() else vp.parent
            _add_root(base)
            # Walk up two levels so a sibling like <root>/upscalers and a
            # shared <root>/chkpts both get scanned from <root>.
            _add_root(base.parent)
            _add_root(base.parent.parent)

        for root in candidate_roots:
            if not root.is_dir():
                continue
            for pattern in _RIFE_WEIGHT_GLOBS:
                matches = sorted(root.glob(pattern))
                if matches:
                    found = matches[0]
                    logger.info("Auto-discovered RIFE weights: %s", found)
                    return found
        return None

    @staticmethod
    def _exp_for_multiplier(multiplier: float) -> Optional[int]:
        """Return the RIFE exponent for a power-of-two multiplier, else None.

        2.0 → 1 (2× frames), 4.0 → 2 (4×), 8.0 → 3, etc. Non-powers (1.5, 3)
        return None so the caller falls back to minterpolate.
        """
        try:
            m = float(multiplier)
        except (TypeError, ValueError):
            return None
        if m < 2.0:
            return None
        exp = math.log2(m)
        rounded = round(exp)
        if rounded >= 1 and abs(exp - rounded) < 1e-6:
            return int(rounded)
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def interpolate_video(
        self,
        input_path: str,
        output_path: str,
        *,
        multiplier: float = 2.0,
        target_fps: Optional[int] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> str:
        """Interpolate an input video to higher fps.

        Args:
            input_path: source mp4 / mkv / mov.
            output_path: destination mp4 (will be overwritten).
            multiplier: fps multiplier. 2.0 = double fps.
            target_fps: explicit target fps. Overrides ``multiplier`` if given.
            progress_cb: optional callback ``(fraction, desc)`` for status.

        Returns the output path on success.

        Uses the real RIFE v4 backend when weights resolve and ``multiplier``
        is a power of two (and ``target_fps`` is not forced); otherwise falls
        back to ffmpeg ``minterpolate``.
        """
        src = Path(input_path)
        if not src.is_file():
            raise FileNotFoundError(input_path)

        # Real RIFE only applies for power-of-two multipliers with no forced
        # target fps (RIFE doubles/quadruples; arbitrary fps targets go to
        # minterpolate). Requires weights to resolve.
        exp = None if target_fps is not None else self._exp_for_multiplier(multiplier)
        if exp is not None:
            weights = self._resolve_rife_weights()
            if weights is not None:
                try:
                    return self._interpolate_rife(
                        src, output_path, weights=weights, exp=exp,
                        multiplier=float(multiplier), progress_cb=progress_cb,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Real RIFE interpolation failed (%s); falling back to "
                        "ffmpeg minterpolate.", exc, exc_info=True,
                    )
            else:
                logger.info("No RIFE weights found; using ffmpeg minterpolate.")

        return self._interpolate_minterpolate(
            src, output_path, multiplier=multiplier,
            target_fps=target_fps, progress_cb=progress_cb,
        )

    # ------------------------------------------------------------------
    # Real RIFE backend
    # ------------------------------------------------------------------
    def _interpolate_rife(
        self,
        src: Path,
        output_path: str,
        *,
        weights: Path,
        exp: int,
        multiplier: float,
        progress_cb: Optional[Callable[[float, str], None]],
    ) -> str:
        """Decode → RIFE → re-encode. Returns the output path.

        Memory note: the entire clip is decoded into RAM as an
        ``(N, H, W, 3)`` uint8 array. For a 12 GB / 31 GB box this is fine for
        typical short AI clips (a few hundred frames at <= 1080p), but very
        long clips could exhaust RAM — chunked decode would be needed there.
        """
        import av  # PyAV — installed; decodes to numpy
        import numpy as np

        from sdqt.utils.codec import configured_codec_args
        from supremediffusion.postprocessing.rife.interpolate import (
            temporal_interpolation,
        )

        if progress_cb:
            progress_cb(0.0, "decoding video for RIFE")

        # --- Decode all frames into an (N, H, W, 3) uint8 array. ---
        frames: list[np.ndarray] = []
        src_fps = 30.0
        with av.open(str(src)) as container:
            vstream = container.streams.video[0]
            rate = vstream.average_rate or vstream.base_rate
            if rate:
                src_fps = float(rate)
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))

        if not frames:
            raise RuntimeError(f"no video frames decoded from {src}")

        arr = np.stack(frames, axis=0)  # (N, H, W, 3) uint8
        del frames

        # --- Choose device. ---
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except Exception:  # noqa: BLE001
            pass

        logger.info(
            "Real RIFE: %s (%d frames @ %.3f fps) exp=%d device=%s weights=%s",
            src.name, arr.shape[0], src_fps, exp, device, weights.name,
        )

        def _rife_progress(frac: float, desc: str) -> None:
            if progress_cb:
                # Reserve the back portion of the bar for encoding.
                progress_cb(0.05 + 0.80 * float(frac), desc)

        out = temporal_interpolation(
            arr, exp=exp, model_path=str(weights),
            device=device, progress_cb=_rife_progress,
        )
        del arr

        target_fps = max(1, int(round(src_fps * float(multiplier))))

        if progress_cb:
            progress_cb(0.88, f"encoding → {target_fps} fps")

        # --- Re-encode the interpolated frames with the configured codec. ---
        # Pipe raw rgb24 frames into ffmpeg; mux the source audio (if any)
        # back in via a second input with -c:a copy.
        height, width = out.shape[1], out.shape[2]
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(target_fps),
            "-i", "pipe:0",
            "-i", str(src),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            *configured_codec_args(),
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-shortest",
            str(output_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        try:
            proc.stdin.write(np.ascontiguousarray(out).tobytes())
        finally:
            if proc.stdin:
                proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)

        if progress_cb:
            progress_cb(1.0, "done")
        return str(output_path)

    # ------------------------------------------------------------------
    # ffmpeg minterpolate fallback
    # ------------------------------------------------------------------
    def _interpolate_minterpolate(
        self,
        src: Path,
        output_path: str,
        *,
        multiplier: float,
        target_fps: Optional[int],
        progress_cb: Optional[Callable[[float, str], None]],
    ) -> str:
        """Fallback: ffmpeg motion-compensated frame interpolation."""
        from sdqt.utils.codec import configured_codec_args

        # Probe source fps so we can compute the target.
        if target_fps is None:
            try:
                r = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=r_frame_rate",
                     "-of", "default=nw=1:nk=1", str(src)],
                    capture_output=True, text=True, timeout=15,
                )
                num, den = r.stdout.strip().split("/")
                src_fps = float(num) / float(den) if float(den) else 30.0
            except Exception:  # noqa: BLE001
                src_fps = 30.0
            target_fps = max(1, int(round(src_fps * float(multiplier))))

        logger.info("RIFE/minterpolate: %s → %s (fps=%s)", src.name, Path(output_path).name, target_fps)
        if progress_cb:
            progress_cb(0.0, f"frame interp → {target_fps} fps")

        # ffmpeg minterpolate filter — mci (motion-compensated interpolation)
        # is the highest-quality mode. Default fps=60, override per arg.
        vf = (
            f"minterpolate="
            f"fps={target_fps}:"
            f"mi_mode=mci:"
            f"mc_mode=aobmc:"
            f"vsbmc=1:"
            f"me_mode=bidir"
        )
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src),
            "-vf", vf,
            *configured_codec_args(),
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)

        if progress_cb:
            progress_cb(1.0, "done")
        return str(output_path)
