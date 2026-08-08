"""Worker thread for face swapping using InsightFace + inswapper/reswapper."""

from __future__ import annotations

import logging
from pathlib import Path

from sdqt.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class FaceSwapWorker(BaseWorker):
    """Run FaceSwapPipeline.swap() in a background thread.

    Emits ``finished_ok(str)`` with the path to the swapped image.
    """

    def __init__(
        self,
        source_face_path: str,
        target_path: str,
        models_dir: str,
        output_dir: str,
        swap_model: str = "inswapper_128",
        enhancer: str | None = None,
        enhancer_strength: float = 0.5,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._source_face_path = source_face_path
        self._target_path = target_path
        self._models_dir = models_dir
        self._output_dir = output_dir
        self._swap_model = swap_model
        self._enhancer = enhancer
        self._enhancer_strength = enhancer_strength

    def do_work(self) -> str:
        from supremediffusion.models.face_swap import FaceSwapPipeline

        self.progress.emit(0.1, "Loading face swap models...")
        pipeline = FaceSwapPipeline(self._models_dir)

        if self.is_aborted:
            raise InterruptedError("Aborted by user")

        self.progress.emit(0.3, "Swapping face...")
        result = pipeline.swap(
            source_path=self._source_face_path,
            target_path=self._target_path,
            swap_model=self._swap_model,
            enhancer=self._enhancer,
            enhancer_strength=self._enhancer_strength,
        )

        if self.is_aborted:
            raise InterruptedError("Aborted by user")

        self.progress.emit(0.9, "Saving result...")
        out_dir = Path(self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        target_stem = Path(self._target_path).stem
        out_path = out_dir / f"{target_stem}_faceswap.png"
        result.save(str(out_path))

        self.progress.emit(1.0, "Done")
        return str(out_path)
