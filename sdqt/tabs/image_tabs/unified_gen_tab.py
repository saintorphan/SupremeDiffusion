"""Unified image generation tab — replaces SDXLTab, FluxTab, ZImageTab."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
)

from sdqt.tabs.base import BaseTab
from sdqt.tabs.image_tabs.model_family import (
    SDXL_STRATEGY,
    STRATEGY_MAP,
    ModelFamilyStrategy,
)
from sdqt.tabs.image_tabs.img2img import Img2ImgTab
from sdqt.tabs.image_tabs.inpainter import InpainterTab
from sdqt.tabs.image_tabs.img_edit import ImgEditTab
from sdqt.tabs.image_tabs.crop_zoom import CropZoomTab
from sdqt.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class _ModelLoadWorker(BaseWorker):
    """Load a model pipeline on a background thread."""

    def __init__(self, state, strategy: ModelFamilyStrategy, model_path: str = "",
                 flux_chroma: str = "", flux_fill: str = "", parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._strategy = strategy
        self._model_path = model_path
        self._flux_chroma = flux_chroma
        self._flux_fill = flux_fill

    def do_work(self) -> str:
        s = self._strategy
        self.status.emit(f"Loading {s.display_name} pipeline...")

        if s.family == "flux":
            cfg = self._state.global_config
            if self._flux_chroma:
                cfg.model_paths["flux_chroma_dir"] = self._flux_chroma
            if self._flux_fill:
                cfg.model_paths["flux_fill_dir"] = self._flux_fill
            cfg.save()
            self._state.load_flux_pipelines(
                load_chroma=bool(self._flux_chroma),
                load_fill=bool(self._flux_fill),
            )
        elif s.family == "zimage":
            self._state.load_zimage_pipelines(model_path=self._model_path or None)
        else:
            # SD15/SDXL — checkpoint is already set in config by the caller
            self._state.load_sd_pipelines()

        return f"{s.display_name} pipeline loaded."


def _scan_gguf_files(directory: str) -> list[str]:
    d = Path(directory)
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.glob("*.gguf"))


class UnifiedGenTab(BaseTab):
    """Unified container for txt2img / img2img / inpaint — all model families.

    Layout::

        [Model: (combo with family-tagged items)] [Load] [Unload] [Refresh] [status]
        [QTabWidget: txt2img | img2img | inpaint]
    """

    #: Emitted after the Refresh button rescans model directories, so the
    #: owning ImageSuiteTab can repopulate the sub-tab checkpoint/VAE selectors.
    models_refreshed = Signal()

    def __init__(self, state, parent=None) -> None:
        super().__init__(state, parent)
        self._current_strategy: ModelFamilyStrategy = SDXL_STRATEGY
        self._load_worker: _ModelLoadWorker | None = None
        self._restoring_selection: bool = False
        self._build_ui()
        # Apply initial strategy to sub-tabs so checkpoint/VAE rows are visible
        for tab in self._sub_tabs:
            if hasattr(tab, "set_family"):
                tab.set_family(self._current_strategy)
        self._refresh_models()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Model selection bar
        model_row = QHBoxLayout()
        model_row.setSpacing(6)
        model_row.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(320)
        self._model_combo.currentIndexChanged.connect(self._on_model_selected)
        model_row.addWidget(self._model_combo, 1)

        self._load_btn = QPushButton("Load")
        self._load_btn.setStyleSheet("font-weight: bold;")
        self._load_btn.clicked.connect(self._on_load)
        model_row.addWidget(self._load_btn)

        self._unload_btn = QPushButton("Unload")
        self._unload_btn.clicked.connect(self._on_unload)
        model_row.addWidget(self._unload_btn)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setToolTip("Rescan all model directories")
        self._refresh_btn.clicked.connect(self._refresh_models)
        model_row.addWidget(self._refresh_btn)

        self._model_status = QLabel("")
        model_row.addWidget(self._model_status)
        layout.addLayout(model_row)

        # Sub-tabs
        self._tab_widget = QTabWidget()

        self._img2img = Img2ImgTab(self.state)
        self._tab_widget.addTab(self._img2img, "All2Img")

        self._inpainter = InpainterTab(self.state)
        self._tab_widget.addTab(self._inpainter, "Inpaint")

        self._img_edit = ImgEditTab(self.state)
        self._tab_widget.addTab(self._img_edit, "Draw")

        self._crop_zoom = CropZoomTab(self.state)
        self._tab_widget.addTab(self._crop_zoom, "Crop/Zoom")

        # Backward compat alias
        self._txt2img = self._img2img

        self._sub_tabs = [self._img2img, self._inpainter, self._img_edit, self._crop_zoom]
        # Wire the per-tab "Refresh LoRAs" button to rescan the current
        # family's LoRA dir (was a no-op stub before).
        for tab in self._sub_tabs:
            if hasattr(tab, "_params"):
                tab._params.set_lora_refresh_callback(
                    lambda: self._scan_loras_for(self._current_strategy)
                )
        layout.addWidget(self._tab_widget)

    # ── Model scanning ────────────────────────────────────────────────────────

    @Slot()
    def _refresh_models(self) -> None:
        """Scan all configured model directories and populate the dropdown."""
        self._model_combo.blockSignals(True)
        self._model_combo.clear()

        try:
            from supremediffusion.models.sd_models import scan_all_image_models
            models = scan_all_image_models(self.state.global_config)
        except Exception as exc:
            logger.warning("Failed to scan models: %s", exc)
            models = []

        for m in models:
            label = f"[{m.model_type.upper()}] {m.name}"
            if m.subtype:
                label += f" ({m.subtype})"
            # Store (filename, model_type, subtype) as item data
            self._model_combo.addItem(label, (m.filename, m.model_type, m.subtype))

        if self._model_combo.count() == 0:
            self._model_combo.addItem("(no models found)")

        self._model_combo.blockSignals(False)

        # Auto-select previously loaded model if possible
        self._auto_select_loaded()

        count = len(models)
        self._model_status.setText(f"{count} model(s) found")
        self.models_refreshed.emit()

    def _auto_select_loaded(self) -> None:
        """Try to select the currently loaded model in the dropdown."""
        # Check each pipeline for loaded state
        for attr, family in [("img_pipeline", "sdxl"), ("flux_pipeline", "flux"),
                             ("zimage_pipeline", "zimage")]:
            if getattr(self.state, attr, None) is not None:
                for i in range(self._model_combo.count()):
                    data = self._model_combo.itemData(i)
                    if data and data[1] == family:
                        self._model_combo.setCurrentIndex(i)
                        self._apply_strategy(STRATEGY_MAP.get(family, SDXL_STRATEGY))
                        return

    # ── Model selection ───────────────────────────────────────────────────────

    @Slot(int)
    def _on_model_selected(self, index: int) -> None:
        data = self._model_combo.itemData(index)
        if not data:
            return
        filename, model_type, subtype = data
        strategy = STRATEGY_MAP.get(model_type, SDXL_STRATEGY)
        self._apply_strategy(strategy)
        self._persist_selection(filename, model_type, subtype)

    def _persist_selection(self, filename: str, model_type: str, subtype: str) -> None:
        if self._restoring_selection:
            return
        try:
            from supremediffusion.config.project_config import ProjectConfig
            cfg = ProjectConfig.load(self.project_path)
            cfg.selected_image_model_filename = filename or ""
            cfg.selected_image_model_type = model_type or ""
            cfg.selected_image_model_subtype = subtype or ""
            cfg.save(self.project_path)
        except Exception as exc:
            logger.debug("UnifiedGenTab _persist_selection failed: %s", exc)

    def _restore_selection_from_project(self) -> bool:
        """Try to set the model combo to the saved filename for this project.

        Returns True on success. The combo's currentIndexChanged is suppressed
        while restoring; the strategy is applied explicitly afterwards.
        """
        try:
            from supremediffusion.config.project_config import ProjectConfig
            cfg = ProjectConfig.load(self.project_path)
        except Exception:
            return False
        target_filename = cfg.selected_image_model_filename or ""
        target_type = cfg.selected_image_model_type or ""
        target_subtype = cfg.selected_image_model_subtype or ""
        if not target_filename:
            return False
        for i in range(self._model_combo.count()):
            data = self._model_combo.itemData(i)
            if not data:
                continue
            fn, mt, sub = data
            if fn == target_filename and mt == target_type and (sub or "") == target_subtype:
                self._restoring_selection = True
                try:
                    self._model_combo.setCurrentIndex(i)
                finally:
                    self._restoring_selection = False
                strategy = STRATEGY_MAP.get(mt, SDXL_STRATEGY)
                self._apply_strategy(strategy)
                return True
        return False

    def _apply_strategy(self, strategy: ModelFamilyStrategy) -> None:
        if strategy is self._current_strategy:
            return
        self._current_strategy = strategy
        for tab in self._sub_tabs:
            tab.set_family(strategy)
        self._refresh_loras_for_family(strategy)

    # ── LoRA directory switching ──────────────────────────────────────────────

    def _scan_loras_for(self, strategy: ModelFamilyStrategy) -> list[str]:
        """Scan the given family's LoRA directory (recursively) for LoRA names."""
        lora_dir = self.state.global_config.model_paths.get(strategy.lora_dir_key, "")
        loras: list[str] = []
        if lora_dir:
            d = Path(lora_dir)
            if d.is_dir():
                for ext in (".safetensors", ".ckpt", ".pt"):
                    loras.extend(p.stem for p in sorted(d.rglob(f"*{ext}")))
        return loras

    def _refresh_loras_for_family(self, strategy: ModelFamilyStrategy) -> None:
        """Rescan LoRA directory for the current family and update all sub-tabs."""
        loras = self._scan_loras_for(strategy)
        for tab in self._sub_tabs:
            if hasattr(tab, "_params"):
                tab._params.populate_options(loras=loras)

    # ── Load / Unload ─────────────────────────────────────────────────────────

    @Slot()
    def _on_load(self) -> None:
        data = self._model_combo.currentData()
        if not data:
            self._model_status.setText("No model selected.")
            return

        filename, model_type, subtype = data
        strategy = STRATEGY_MAP.get(model_type, SDXL_STRATEGY)
        self._apply_strategy(strategy)

        # Prepare load arguments
        model_path = ""
        flux_chroma = ""
        flux_fill = ""

        if strategy.family == "flux":
            if subtype == "fill":
                flux_fill = filename
            else:
                flux_chroma = filename
        elif strategy.family == "zimage":
            model_path = filename
        elif strategy.family in ("sd15", "sdxl"):
            # Set the checkpoint in global config for SD loading
            cfg = self.state.global_config
            cfg.model_paths["sd_checkpoint_dir"] = str(Path(filename).parent)
            # The SD pipeline will use the selected checkpoint
            model_path = filename

        self._load_btn.setEnabled(False)
        self._model_status.setText("Loading...")

        worker = _ModelLoadWorker(
            self.state, strategy,
            model_path=model_path,
            flux_chroma=flux_chroma,
            flux_fill=flux_fill,
            parent=self,
        )
        worker.finished_ok.connect(self._on_loaded)
        worker.error.connect(self._on_load_error)
        worker.status.connect(lambda s: self._model_status.setText(s))
        worker.finished.connect(worker.deleteLater)
        self._load_worker = worker
        worker.start()

    def _on_loaded(self, msg: str) -> None:
        self._load_btn.setEnabled(True)
        self._model_status.setText(msg)

    def _on_load_error(self, msg: str) -> None:
        self._load_btn.setEnabled(True)
        self._model_status.setText(f"Error: {msg}")

    @Slot()
    def _on_unload(self) -> None:
        method = getattr(self.state, self._current_strategy.unload_method, None)
        if method:
            method()
        self._model_status.setText("Unloaded.")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_sub_tabs(self) -> list:
        """Return sub-tabs for gallery wiring."""
        return list(self._sub_tabs)

    def load_source(self, path: str) -> None:
        """Load image into img2img sub-tab."""
        self._img2img.load_source(path)
        self._tab_widget.setCurrentWidget(self._img2img)

    def load_fill_source(self, path: str) -> None:
        """Load image into inpaint sub-tab."""
        self._inpainter.load_source(path)
        self._tab_widget.setCurrentWidget(self._inpainter)

    def switch_to(self, mode: str) -> None:
        """Switch to a sub-tab by mode name."""
        if mode == "txt2img":
            self._tab_widget.setCurrentWidget(self._img2img)
            self._img2img.switch_to_txt2img()
        elif mode == "img2img":
            self._tab_widget.setCurrentWidget(self._img2img)
            self._img2img.switch_to_img2img()
        elif mode == "inpaint":
            self._tab_widget.setCurrentWidget(self._inpainter)

    # ── Project persistence ───────────────────────────────────────────────────

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        # Restore the saved model selection first so the right strategy is
        # applied before sub-tabs read sampler/scheduler/etc. from the config.
        self._restore_selection_from_project()
        for tab in self._sub_tabs:
            tab.on_project_changed(project_name)

    def populate_options(self, **kwargs) -> None:
        """Populate checkpoints, VAEs, samplers, etc. for all sub-tabs."""
        for tab in self._sub_tabs:
            if hasattr(tab, "populate_options"):
                tab.populate_options(**kwargs)
