"""Base worker thread for background tasks."""

from __future__ import annotations

import contextlib
import gc
import logging
import traceback

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


# Sentinel context manager used when no color profile is in scope so the
# ``with`` statement in ``run()`` is uniform regardless.
_NO_OP_CTX = contextlib.nullcontext()


class BaseWorker(QThread):
    """QThread subclass with standard progress/status/error signals.

    Subclasses override ``do_work()`` which runs on the background thread.
    The main thread connects to signals for UI updates.
    """

    # Signals
    progress = Signal(float, str)       # (fraction 0-1, description)
    status = Signal(str)                # status message
    error = Signal(str)                 # error message
    finished_ok = Signal(object)        # result payload (path, dict, etc.)
    aborted = Signal()                  # emitted when worker is cancelled

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._abort = False
        # Subclasses that have access to a ProjectConfig + GlobalConfig set
        # this so every ffmpeg invocation in do_work() picks up the right
        # color profile via the thread-local in sdqt.utils.codec. Either set
        # ``self._color_profile`` directly to a ColorProfile, or set
        # ``self._project_config`` / ``self._global_config`` and let
        # ``run()`` resolve it.
        self._color_profile = None
        self._project_config = None
        self._global_config = None

    # -- Public API -------------------------------------------------------

    def abort(self) -> None:
        """Request cancellation.  do_work() should check ``self.is_aborted``."""
        self._abort = True

    @property
    def is_aborted(self) -> bool:
        return self._abort

    # -- Override point ----------------------------------------------------

    def do_work(self) -> object:
        """Override this.  Return value is emitted via ``finished_ok``."""
        raise NotImplementedError

    # -- QThread entry point -----------------------------------------------

    def run(self) -> None:
        self._abort = False
        from sdqt.utils.codec import active_profile_context, _resolve_profile
        # Resolve the color profile this worker should use. Explicit
        # ``self._color_profile`` wins; otherwise resolve from project +
        # global config; if neither was set, fall back to global default.
        profile = self._color_profile
        if profile is None and (self._project_config is not None or self._global_config is not None):
            profile = _resolve_profile(self._project_config, self._global_config)
        try:
            with active_profile_context(profile=profile) if profile is not None else _NO_OP_CTX:
                try:
                    result = self.do_work()
                    if not self._abort:
                        logger.info("%s emitting finished_ok (result type=%s)", type(self).__name__, type(result).__name__)
                        self.finished_ok.emit(result)
                        logger.info("%s finished_ok emitted successfully", type(self).__name__)
                    else:
                        self._emit_aborted()
                except InterruptedError:
                    self._emit_aborted()
                except Exception as exc:
                    tb = traceback.format_exc()
                    logger.error("Worker failed:\n%s", tb)
                    self.error.emit(str(exc))
        finally:
            self._log_vram(f"{type(self).__name__} pre-cleanup")
            self._cleanup_gpu()
            self._log_vram(f"{type(self).__name__} post-cleanup")

    def _emit_aborted(self) -> None:
        """Notify listeners that the worker was cancelled."""
        self.status.emit("Aborted.")
        self.aborted.emit()
        # Also emit error so existing UI handlers restore button state
        self.error.emit("Aborted.")

    @staticmethod
    def _cleanup_gpu() -> None:
        """Release GPU memory after worker finishes."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @staticmethod
    def _log_vram(tag: str) -> None:
        """Log VRAM allocation, reservation, and fragmentation for diagnostics."""
        try:
            import torch
            if not torch.cuda.is_available():
                return
            alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
            # Fragmentation proxy: reserved-but-not-allocated blocks.
            # Large gap between reserved and allocated => fragmented arena.
            frag = reserved - alloc
            stats = torch.cuda.memory_stats()
            active = stats.get("active.all.current", 0)
            inactive_split = stats.get("inactive_split_bytes.all.current", 0) / (1024 ** 2)
            num_alloc_retries = stats.get("num_alloc_retries", 0)
            num_ooms = stats.get("num_ooms", 0)
            logger.info(
                "[VRAM %s] alloc=%.0fMB reserved=%.0fMB frag=%.0fMB "
                "inactive_split=%.0fMB peak_alloc=%.0fMB peak_reserved=%.0fMB "
                "active_blocks=%d retries=%d ooms=%d",
                tag, alloc, reserved, frag, inactive_split,
                peak_alloc, peak_reserved, active, num_alloc_retries, num_ooms,
            )
        except Exception:
            pass

    # -- FFmpeg helper -----------------------------------------------------

    @staticmethod
    def _run_ffmpeg(cmd: list[str], label: str = "ffmpeg") -> None:
        """Run an ffmpeg command and raise on failure."""
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            err = result.stderr[-500:] if result.stderr else f"{label} failed"
            raise RuntimeError(err)

    # -- Step callback helper ---------------------------------------------

    def make_step_callback(self, total_steps: int, phase_label: str = ""):
        """Return a diffusers-compatible step callback that emits progress."""
        def _cb(_pipe, step_index, _timestep, cb_kwargs):
            if self._abort:
                raise InterruptedError("Aborted by user")
            frac = (step_index + 1) / total_steps
            label = f"Step {step_index + 1}/{total_steps}"
            if phase_label:
                label = f"{label} — {phase_label}"
            self.progress.emit(frac, label)
            return cb_kwargs
        return _cb
