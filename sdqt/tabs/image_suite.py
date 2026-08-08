"""Image Suite tab -- container with sub-tabs for SD image operations."""

from __future__ import annotations

import logging

from PySide6.QtWidgets import QTabWidget, QVBoxLayout

from sdqt.state import AppState
from sdqt.tabs.base import BaseTab
from sdqt.tabs.image_tabs import (
    Model3DTab,
    PngInfoTab,
    UnifiedGenTab,
)
from sdqt.tabs.image_tabs.character_tools_tab import CharacterToolsTab
from sdqt.tabs.image_tabs.lora_manager_tab import LoRAManagerTab
from sdqt.tabs.lora_training import LoRATrainingTab

logger = logging.getLogger(__name__)


class ImageSuiteTab(BaseTab):
    """Container tab with sub-tabs for SD image generation operations."""

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._sub_tabs: list[BaseTab] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._tab_widget = QTabWidget()

        self._gen = UnifiedGenTab(self.state)
        self._tab_widget.addTab(self._gen, "Generate")
        # The Refresh button in the model bar rescans the top dropdown; the
        # sub-tab checkpoint/VAE selectors are populated separately, so rescan
        # those too.
        self._gen.models_refreshed.connect(self.populate_sd_options)

        self._lora_manager = LoRAManagerTab(self.state)
        self._tab_widget.addTab(self._lora_manager, "LoRA Manager")

        self._character_tools = CharacterToolsTab(self.state)
        self._tab_widget.addTab(self._character_tools, "Character Tools")

        self._model3d = Model3DTab(self.state)
        self._tab_widget.addTab(self._model3d, "3D Modeling")

        self._png_info = PngInfoTab(self.state)
        self._tab_widget.addTab(self._png_info, "QwenAlyzer")

        self._lora = LoRATrainingTab(self.state)
        self._tab_widget.addTab(self._lora, "LoRA Training")

        self._sub_tabs = [
            self._gen, self._lora_manager, self._character_tools,
            self._model3d, self._png_info, self._lora,
        ]

        layout.addWidget(self._tab_widget)

    # ── Pass-through attributes for main_window send-to wiring ────────────────
    # Draw and Crop/Zoom now live inside UnifiedGenTab but existing send-to
    # references in main_window use self._img_edit and self._crop_zoom.

    @property
    def _img_edit(self):
        return self._gen._img_edit

    @property
    def _crop_zoom(self):
        return self._gen._crop_zoom

    @property
    def _face_swap(self):
        return self._character_tools._face_swap

    @property
    def _body_double(self):
        return self._character_tools._body_double

    @property
    def _repose(self):
        return self._character_tools._repose

    @property
    def _controlnet(self):
        return self._character_tools._controlnet

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        for tab in self._sub_tabs:
            tab.on_project_changed(project_name)

    def populate_sd_options(self) -> None:
        """Populate checkpoints, VAEs, samplers, etc. from backend scans."""
        cfg = self.state.global_config
        kwargs = {}

        try:
            from supremediffusion.models.sd_models import scan_checkpoints, scan_vaes, scan_loras
            ckpt_dir = cfg.model_paths.get("sd_checkpoint_dir", "")
            if ckpt_dir:
                ckpts = scan_checkpoints(ckpt_dir)
                kwargs["checkpoints"] = [c.name for c in ckpts] if ckpts else []
            vae_dir = cfg.model_paths.get("sd_vae_dir", "")
            if vae_dir:
                kwargs["vaes"] = scan_vaes(vae_dir)
            lora_dir = cfg.model_paths.get("sd_lora_dir", "")
            if lora_dir:
                kwargs["loras"] = scan_loras(lora_dir)
            refiner_dir = cfg.model_paths.get("sd_refiner_dir", "")
            if refiner_dir:
                refiner_ckpts = scan_checkpoints(refiner_dir)
                kwargs["refiners"] = [c.name for c in refiner_ckpts] if refiner_ckpts else []
        except Exception as exc:
            logger.warning("Failed to scan SD models: %s", exc)

        try:
            from supremediffusion.models.sd_samplers import list_samplers, list_schedulers
            kwargs["samplers"] = list_samplers()
            kwargs["schedulers"] = list_schedulers()
        except Exception as exc:
            logger.warning("Failed to list samplers: %s", exc)

        self._gen.populate_options(**kwargs)
        self._character_tools.populate_options(**kwargs)
        self._lora_manager.populate_options(**kwargs)
