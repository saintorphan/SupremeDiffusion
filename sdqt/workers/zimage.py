"""Worker threads for Z-Image Turbo generation."""

from __future__ import annotations

import logging
from typing import Any

from supremediffusion.config.project_config import ProjectConfig

from .base import BaseWorker

logger = logging.getLogger(__name__)


class ZImageTxt2ImgWorker(BaseWorker):
    """Run Z-Image txt2img generation on a background thread."""

    def __init__(
        self,
        pipeline: Any,
        project_name: str,
        project_config: ProjectConfig,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._project_name = project_name
        self._config = project_config

    def do_work(self) -> list[str]:
        total_steps = self._config.zimage_steps or 9
        callback = self.make_step_callback(total_steps)

        self.status.emit("Generating Z-Image images...")
        self.progress.emit(0.0, "Starting Z-Image txt2img...")

        result = self._pipeline.run_txt2img(
            project_name=self._project_name,
            config=self._config,
            callback=callback,
        )

        self.progress.emit(1.0, "Complete")
        return result


class ZImageImg2ImgWorker(BaseWorker):
    """Run Z-Image img2img generation on a background thread."""

    def __init__(
        self,
        pipeline: Any,
        project_name: str,
        source_image: str,
        project_config: ProjectConfig,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._project_name = project_name
        self._source = source_image
        self._config = project_config

    def do_work(self) -> list[str]:
        total_steps = self._config.zimage_steps or 9
        callback = self.make_step_callback(total_steps)

        self.status.emit("Generating Z-Image images...")
        self.progress.emit(0.0, "Starting Z-Image img2img...")

        result = self._pipeline.run_img2img(
            project_name=self._project_name,
            source_image=self._source,
            config=self._config,
            callback=callback,
        )

        self.progress.emit(1.0, "Complete")
        return result


class ZImageInpaintWorker(BaseWorker):
    """Run Z-Image inpainting on a background thread."""

    def __init__(
        self,
        pipeline: Any,
        project_name: str,
        source_image: str,
        mask: str,
        project_config: ProjectConfig,
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._project_name = project_name
        self._source = source_image
        self._mask = mask
        self._config = project_config

    def do_work(self) -> list[str]:
        total_steps = self._config.zimage_steps or 9
        callback = self.make_step_callback(total_steps)

        self.status.emit("Running Z-Image Inpaint...")
        self.progress.emit(0.0, "Starting Z-Image Inpaint...")

        result = self._pipeline.run_inpaint(
            project_name=self._project_name,
            source_image=self._source,
            mask=self._mask,
            config=self._config,
            callback=callback,
        )

        self.progress.emit(1.0, "Complete")
        return result
