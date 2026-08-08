"""Worker thread for face restoration using spandrel models (GFPGAN, RestoreFormer)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from sdqt.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class FaceRestoreWorker(BaseWorker):
    """Run face_restore_frames() on a single image in a background thread.

    Emits ``finished_ok(str)`` with the path to the restored image.
    """

    def __init__(
        self,
        source_path: str,
        model_path: str,
        output_dir: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source_path = source_path
        self._model_path = model_path
        self._output_dir = output_dir

    def do_work(self) -> str:
        from supremediffusion.postprocessing.ai_upscale import face_restore_frames

        self.progress.emit(0.0, "Loading image...")
        img = Image.open(self._source_path).convert("RGB")
        frames = np.array(img)[np.newaxis, ...]  # (1, H, W, 3)

        def _progress(frac, desc=""):
            if self.is_aborted:
                raise InterruptedError("Aborted by user")
            self.progress.emit(frac, desc or "Restoring faces...")

        self.progress.emit(0.1, "Running face restoration...")
        result = face_restore_frames(
            frames,
            self._model_path,
            progress_cb=_progress,
        )

        self.progress.emit(0.9, "Saving result...")
        out_img = Image.fromarray(result[0])
        out_dir = Path(self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        src_stem = Path(self._source_path).stem
        out_path = out_dir / f"{src_stem}_face_restored.png"
        out_img.save(str(out_path))

        self.progress.emit(1.0, "Done")
        return str(out_path)
