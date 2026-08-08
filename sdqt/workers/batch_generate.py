"""Batch Generate controller — chains generation → loop → post-process per image."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

logger = logging.getLogger(__name__)


class BatchGenerateController(QObject):
    """Run img2vid over a list of images, optionally loop + post-process each.

    This is a main-thread orchestrator (QObject, not a QThread). It chains
    background workers together by connecting to each worker's ``finished_ok``
    signal and starting the next phase when the previous one completes.

    Signals:
        progress(int, int, str)    — (index_1based, total, status_text)
        item_done(str)              — final path of one completed item
        batch_done(int, int)        — (completed_ok, errored)
        batch_error(str)            — fatal error that aborts the whole batch
    """

    progress = Signal(int, int, str)
    item_done = Signal(str)
    batch_done = Signal(int, int)
    batch_error = Signal(str)

    def __init__(self, tab, settings: dict, parent=None) -> None:
        super().__init__(parent)
        self._tab = tab
        self._settings = settings
        self._items: list[str] = list(settings.get("images", []))
        self._output_dir = Path(settings["output_dir"])
        self._prefix = settings.get("prefix", "batch_")
        self._continue_numbering = bool(settings.get("continue_numbering", False))
        self._use_guidance = bool(settings.get("use_guidance", False))
        self._do_cc = bool(settings.get("do_color_correct", False))
        self._cc_ref = settings.get("color_correct_ref", "") or ""
        self._do_loop = bool(settings.get("do_loop", False))
        self._do_pp = bool(settings.get("do_pp", False))
        self._pp = settings.get("pp", {}) or {}

        self._i = 0
        self._completed = 0
        self._errored = 0
        self._aborted = False
        self._running = False

        self._gen_worker = None
        self._cc_worker = None
        self._loop_worker = None
        self._pp_worker = None

        # Numbering offset: if "continue from last number" is ticked, scan
        # the output dir for existing <prefix><digits>.mp4 files and start
        # from one past the highest index.
        self._num_start = 1
        if self._continue_numbering:
            self._num_start = self._scan_highest_index() + 1

        # Zero-pad to fit the highest final number the batch can produce.
        highest = self._num_start + max(len(self._items), 1) - 1
        self._pad = max(3, len(str(highest)))

    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        total = len(self._items)
        if not total:
            self._finish()
            return
        # Visible first message so the user sees the batch has kicked off
        self.progress.emit(
            0, total,
            f"Batch starting — 0/{total} (next file #{self._num_start})",
        )
        QTimer.singleShot(0, self._start_next)

    def abort(self) -> None:
        self._aborted = True
        for w in (self._gen_worker, self._cc_worker, self._loop_worker, self._pp_worker):
            if w is not None:
                try:
                    w.abort()
                except Exception:
                    pass

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Overridable hooks — subclasses customise per-item behaviour without
    # rewriting the whole state machine.
    # ------------------------------------------------------------------

    def _build_worker_kwargs(self, item) -> dict:
        """Return InferenceWorker kwargs for *item*. Default: Mode 1 batch."""
        return self._tab._build_mode1_kwargs(item, self._use_guidance)

    def _describe_item(self, item) -> str:
        """Short label for progress/status lines."""
        return Path(item).name

    def _final_filename(self, item, idx: int) -> str:
        """Filename to write at the end of the pipeline for *item*."""
        number = self._num_start + idx
        return f"{self._prefix}{number:0{self._pad}d}.mp4"

    # ------------------------------------------------------------------

    def _start_next(self) -> None:
        if self._aborted or self._i >= len(self._items):
            self._finish()
            return

        item = self._items[self._i]
        idx = self._i + 1
        total = len(self._items)
        self.progress.emit(
            idx, total,
            f"{idx}/{total} — Generating from {self._describe_item(item)}…",
        )

        try:
            kwargs = self._build_worker_kwargs(item)
        except Exception as exc:
            logger.exception("Failed to build kwargs for %s", item)
            self.progress.emit(
                idx, total,
                f"{idx}/{total} — SKIP: kwargs build failed: {exc}",
            )
            self._errored += 1
            self._i += 1
            QTimer.singleShot(0, self._start_next)
            return

        try:
            from sdqt.workers.inference import InferenceWorker
            worker = InferenceWorker(**kwargs, parent=self._tab)
        except Exception as exc:
            logger.exception("Failed to create InferenceWorker for %s", item)
            self.progress.emit(
                idx, total,
                f"{idx}/{total} — SKIP: worker init failed: {exc}",
            )
            self._errored += 1
            self._i += 1
            QTimer.singleShot(0, self._start_next)
            return

        worker.progress.connect(
            lambda f, d, idx=idx: self.progress.emit(
                idx, len(self._items), f"{idx}/{len(self._items)} — {d}"
            )
        )
        worker.finished_ok.connect(self._on_gen_done)
        worker.error.connect(self._on_step_error)
        worker.finished.connect(worker.deleteLater)
        self._gen_worker = worker
        worker.start()

    def _on_gen_done(self, result) -> None:
        self._gen_worker = None
        if self._aborted:
            self._finish()
            return

        result_path = str(result) if result else ""
        if not result_path or not Path(result_path).is_file():
            logger.warning("Batch: generation produced no file for %s",
                           self._items[self._i])
            self._errored += 1
            self._i += 1
            QTimer.singleShot(0, self._start_next)
            return

        if self._do_cc:
            self._start_color_correct(result_path)
        else:
            self._after_color_correct(result_path)

    def _start_color_correct(self, path: str) -> None:
        from sdqt.workers.color_correct import ClipColorCorrectWorker

        idx = self._i + 1
        self.progress.emit(
            idx, len(self._items),
            f"{idx}/{len(self._items)} — Color correcting…",
        )

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, prefix="batch_cc_")
        tmp.close()

        try:
            worker = ClipColorCorrectWorker(
                source=path,
                output=tmp.name,
                reference=self._cc_ref,
                parent=self._tab,
            )
        except Exception as exc:
            logger.error("Batch CC worker init failed: %s", exc)
            self._errored += 1
            self._i += 1
            QTimer.singleShot(0, self._start_next)
            return

        worker.progress.connect(
            lambda f, d, idx=idx: self.progress.emit(
                idx, len(self._items),
                f"{idx}/{len(self._items)} — CC: {d}",
            )
        )
        worker.finished_ok.connect(self._on_cc_done)
        worker.error.connect(self._on_step_error)
        worker.finished.connect(worker.deleteLater)
        self._cc_worker = worker
        worker.start()

    def _on_cc_done(self, result) -> None:
        self._cc_worker = None
        if self._aborted:
            self._finish()
            return
        self._after_color_correct(str(result))

    def _after_color_correct(self, path: str) -> None:
        if self._do_loop:
            self._start_loop(path)
        else:
            self._after_loop(path)

    def _start_loop(self, path: str) -> None:
        from sdqt.workers.loop_build import LoopBuildWorker

        idx = self._i + 1
        self.progress.emit(
            idx, len(self._items), f"{idx}/{len(self._items)} — Building loop…",
        )
        worker = LoopBuildWorker(source=path, parent=self._tab)
        worker.progress.connect(
            lambda f, d, idx=idx: self.progress.emit(
                idx, len(self._items), f"{idx}/{len(self._items)} — Loop: {d}"
            )
        )
        worker.finished_ok.connect(self._on_loop_done)
        worker.error.connect(self._on_step_error)
        worker.finished.connect(worker.deleteLater)
        self._loop_worker = worker
        worker.start()

    def _on_loop_done(self, result) -> None:
        self._loop_worker = None
        if self._aborted:
            self._finish()
            return
        self._after_loop(str(result))

    def _after_loop(self, path: str) -> None:
        if self._do_pp:
            self._start_pp(path)
        else:
            self._finalize(path)

    def _start_pp(self, path: str) -> None:
        from sdqt.workers.clip_quick_export import ClipPostProcessWorker

        idx = self._i + 1
        self.progress.emit(
            idx, len(self._items), f"{idx}/{len(self._items)} — Post-process…",
        )

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, prefix="batch_pp_")
        tmp.close()

        pp = self._pp
        lanczos_str = pp.get("pp_lanczos", "")
        lanczos_mode = (
            "lanczos1.5" if lanczos_str == "1.5"
            else "lanczos2" if lanczos_str == "2.0"
            else ""
        )

        upscaler_dir = ""
        face_models_dir = ""
        try:
            gc = self._tab.state.global_config
            upscaler_dir = gc.model_paths.get("upscaler_dir", "") or ""
            face_models_dir = gc.model_paths.get("face_models_dir", "") or ""
        except Exception:
            pass

        try:
            worker = ClipPostProcessWorker(
                source=path,
                output=tmp.name,
                enable_denoise=bool(pp.get("pp_denoise", False)),
                enable_sharpen=bool(pp.get("pp_sharpen", False)),
                enable_upscale=bool(pp.get("pp_upscale", False)),
                enable_face=bool(pp.get("pp_face", False)),
                tile=int(pp.get("pp_tile", 512)),
                lanczos_mode=lanczos_mode,
                rife_mode="",  # RIFE explicitly off in batch PP
                grain=float(pp.get("pp_grain", 0.0)),
                upscaler_dir=upscaler_dir,
                face_models_dir=face_models_dir,
                parent=self._tab,
            )
        except Exception as exc:
            logger.error("Batch PP worker init failed: %s", exc)
            self._errored += 1
            self._i += 1
            QTimer.singleShot(0, self._start_next)
            return

        worker.progress.connect(
            lambda f, d, idx=idx: self.progress.emit(
                idx, len(self._items), f"{idx}/{len(self._items)} — PP: {d}"
            )
        )
        worker.finished_ok.connect(self._on_pp_done)
        worker.error.connect(self._on_step_error)
        worker.finished.connect(worker.deleteLater)
        self._pp_worker = worker
        worker.start()

    def _on_pp_done(self, result) -> None:
        self._pp_worker = None
        if self._aborted:
            self._finish()
            return
        self._finalize(str(result))

    def _finalize(self, src_path: str) -> None:
        item = self._items[self._i]
        final_name = self._final_filename(item, self._i)
        dest = self._output_dir / final_name
        try:
            if dest.exists():
                dest.unlink()
            shutil.move(src_path, dest)
            self._completed += 1
            self.item_done.emit(str(dest))
            logger.info("Batch: wrote %s", dest)
            self.progress.emit(
                self._i + 1, len(self._items),
                f"{self._i + 1}/{len(self._items)} — wrote {dest.name}",
            )
        except Exception as exc:
            logger.error("Batch: failed to move %s → %s: %s", src_path, dest, exc)
            self._errored += 1

        self._i += 1
        QTimer.singleShot(0, self._start_next)

    def _scan_highest_index(self) -> int:
        """Return the highest numeric suffix found on files matching
        ``<prefix><digits>.mp4`` in the output directory, or 0 if none."""
        import re
        try:
            if not self._output_dir.is_dir():
                return 0
            pattern = re.compile(
                rf"^{re.escape(self._prefix)}(\d+)\.mp4$",
                re.IGNORECASE,
            )
            highest = 0
            for p in self._output_dir.iterdir():
                if not p.is_file():
                    continue
                m = pattern.match(p.name)
                if m:
                    try:
                        n = int(m.group(1))
                        if n > highest:
                            highest = n
                    except ValueError:
                        pass
            return highest
        except Exception as exc:
            logger.warning("scan_highest_index failed: %s", exc)
            return 0

    def _on_step_error(self, msg: str) -> None:
        """A per-item worker reported an error — log it, skip the item."""
        idx = self._i + 1
        total = len(self._items)
        logger.warning("Batch item #%d failed: %s", idx, msg)
        self.progress.emit(idx, total, f"{idx}/{total} — SKIP: {msg}")
        self._gen_worker = None
        self._cc_worker = None
        self._loop_worker = None
        self._pp_worker = None
        self._errored += 1
        self._i += 1
        QTimer.singleShot(0, self._start_next)

    def _finish(self) -> None:
        self._running = False
        self.batch_done.emit(self._completed, self._errored)
