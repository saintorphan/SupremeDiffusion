"""Worker threads for 3D model generation and processing."""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseWorker

logger = logging.getLogger(__name__)


class MeshGenerationWorker(BaseWorker):
    """Generate a 3D mesh from an image using TripoSR."""

    def __init__(
        self,
        pipeline: Any,
        image_path: str,
        output_path: str | None = None,
        resolution: int = 256,
        output_format: str = "obj",
        remove_bg: bool = True,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._image_path = image_path
        self._output_path = output_path
        self._resolution = resolution
        self._format = output_format
        self._remove_bg = remove_bg

    def do_work(self) -> str:
        self.status.emit("Starting 3D mesh generation...")
        self.progress.emit(0.0, "Preparing...")

        def _callback(msg: str) -> None:
            self.status.emit(msg)

        result = self._pipeline.generate_mesh(
            image=self._image_path,
            output_path=self._output_path,
            resolution=self._resolution,
            output_format=self._format,
            remove_bg=self._remove_bg,
            callback=_callback,
        )

        self.progress.emit(1.0, "Complete")
        return result
