"""All2Img sub-tab — unified txt2img + img2img with mode toggle."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from supremediffusion.config.project_config import ProjectConfig

from sdqt.tabs.base import BaseTab
from sdqt.tabs.image_tabs.model_family import SDXL_STRATEGY
from sdqt.widgets.image_drop import ImageDropWidget
from sdqt.widgets.image_gallery import ImageGalleryWidget
from sdqt.widgets.image_params import ImageParamsWidget
from sdqt.widgets.prompt_enhance import PromptEnhanceWidget
from sdqt.workers.interrogate import BLIPInterrogateWorker, WD14TaggerWorker
from sdqt.workers.prompt_enhance import PromptEnhanceWorker

logger = logging.getLogger(__name__)


class Img2ImgTab(BaseTab):
    """Unified text-to-image / image-to-image generation tab.

    A radio toggle switches between Txt2Img mode (prompt only) and
    Img2Img mode (prompt + source image with denoising strength).
    """

    def __init__(self, state, parent=None) -> None:
        super().__init__(state, parent)
        self._worker = None
        self._interrogate_worker = None
        self._strategy = SDXL_STRATEGY
        self._restoring = False
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Mode toggle row ──────────────────────────────────────────
        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(4, 2, 4, 2)
        mode_row.setSpacing(8)
        self._txt_mode = QRadioButton("Txt2Img")
        self._txt_mode.setToolTip("Generate from prompt only")
        self._img_mode = QRadioButton("Img2Img")
        self._img_mode.setToolTip("Generate from prompt + source image")
        self._img_mode.setChecked(True)
        self._txt_mode.toggled.connect(self._on_mode_changed)
        mode_row.addWidget(self._txt_mode)
        mode_row.addWidget(self._img_mode)
        mode_row.addStretch()
        outer.addLayout(mode_row)

        # ── Main content ─────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        content.setMinimumWidth(700)
        scroll.setWidget(content)

        splitter = QSplitter(Qt.Horizontal)
        content.setLayout(QVBoxLayout())
        content.layout().setContentsMargins(0, 0, 0, 0)
        content.layout().addWidget(splitter)

        # ── Left: source image (hidden in txt2img mode) ──────────────
        self._source = ImageDropWidget("Source Image")
        self._source.image_loaded.connect(self._persist_source_path)
        self._source.image_cleared.connect(lambda: self._persist_source_path(""))
        splitter.addWidget(self._source)

        # ── Right: controls ──────────────────────────────────────────
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QScrollArea.NoFrame)
        right_scroll.setMinimumWidth(360)
        right_panel = QWidget()
        # NO fixed minimum height: an explicit minimum CAPPED the panel below
        # its natural content height, so the QScrollArea (widgetResizable) sized
        # it to the viewport and never scrolled — it crammed/overlapped the
        # prompt + param rows instead. Letting the layout's real minimum drive
        # the panel makes the scrollbar engage on short windows.
        rp = QVBoxLayout(right_panel)
        rp.setContentsMargins(8, 6, 8, 6)
        rp.setSpacing(8)

        self._gallery = ImageGalleryWidget("Generated Images")
        self._gallery.setMinimumHeight(180)
        rp.addWidget(self._gallery, 1)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        self._gen_btn = QPushButton("Generate")
        self._gen_btn.setObjectName("primary")
        self._gen_btn.clicked.connect(self._on_generate)
        btn_row.addWidget(self._gen_btn)
        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setVisible(False)
        self._abort_btn.clicked.connect(self._on_abort)
        btn_row.addWidget(self._abort_btn)
        self._unload_btn = QPushButton("Unload")
        self._unload_btn.setToolTip("Free model from VRAM")
        self._unload_btn.clicked.connect(self._on_unload)
        btn_row.addWidget(self._unload_btn)
        self._save_3d_btn = QPushButton("Save 3D")
        self._save_3d_btn.setToolTip("Generate 3D mesh from selected image via TripoSR")
        self._save_3d_btn.clicked.connect(self._on_save_3d)
        btn_row.addWidget(self._save_3d_btn)
        rp.addLayout(btn_row)

        self._status = QLabel("")
        rp.addWidget(self._status)

        # Prompt
        prompt_group = QGroupBox("Prompt")
        pg_layout = QVBoxLayout(prompt_group)
        pg_layout.setContentsMargins(4, 4, 4, 4)
        self._enhance = PromptEnhanceWidget(mode="image")
        self._enhance.enhance_requested.connect(self._on_enhance)
        self._enhance_worker = None
        pg_layout.addWidget(self._enhance)
        self._prompt = QPlainTextEdit()
        self._prompt.setMaximumHeight(90)
        pg_layout.addWidget(self._prompt)
        self._neg_prompt = QPlainTextEdit()
        self._neg_prompt.setMaximumHeight(55)
        self._neg_prompt.setPlaceholderText("Negative prompt...")
        pg_layout.addWidget(self._neg_prompt)

        # Interrogate (img2img only)
        self._interr_row_widget = QWidget()
        interr_row = QHBoxLayout(self._interr_row_widget)
        interr_row.setContentsMargins(0, 0, 0, 0)
        self._clip_btn = QPushButton("CLIP")
        self._clip_btn.setToolTip("Generate a natural language caption from the source image (BLIP)")
        self._clip_btn.clicked.connect(self._on_interrogate_clip)
        interr_row.addWidget(self._clip_btn)
        self._wd14_btn = QPushButton("WD14")
        self._wd14_btn.setToolTip("Generate danbooru-style tags from the source image")
        self._wd14_btn.clicked.connect(self._on_interrogate_wd14)
        interr_row.addWidget(self._wd14_btn)
        pg_layout.addWidget(self._interr_row_widget)
        rp.addWidget(prompt_group)

        # Denoising + Resize Mode (img2img only) — split into two rows so
        # neither control cramps the other when the panel is narrow.
        self._denoise_widget = QWidget()
        denoise_col = QVBoxLayout(self._denoise_widget)
        denoise_col.setContentsMargins(0, 0, 0, 0)
        denoise_col.setSpacing(8)

        denoise_row = QHBoxLayout()
        denoise_row.setSpacing(8)
        denoise_row.addWidget(QLabel("Denoise:"))
        self._denoising = QDoubleSpinBox()
        self._denoising.setRange(0, 1)
        self._denoising.setDecimals(2)
        self._denoising.setSingleStep(0.05)
        self._denoising.setValue(0.75)
        self._denoising.setMinimumWidth(95)
        denoise_row.addWidget(self._denoising)
        denoise_row.addStretch()
        denoise_col.addLayout(denoise_row)

        resize_row = QHBoxLayout()
        resize_row.setSpacing(8)
        resize_row.addWidget(QLabel("Resize:"))
        self._resize_mode = QComboBox()
        self._resize_mode.addItems(["Just resize", "Crop and resize", "Resize and fill"])
        self._resize_mode.setMinimumWidth(160)
        self._resize_mode.setToolTip(
            "Just resize: stretch to target dimensions\n"
            "Crop and resize: center crop to aspect ratio, then resize\n"
            "Resize and fill: fit inside dimensions, pad edges"
        )
        resize_row.addWidget(self._resize_mode)
        resize_row.addStretch()
        denoise_col.addLayout(resize_row)
        rp.addWidget(self._denoise_widget)

        # ImageParamsWidget
        self._params = ImageParamsWidget()
        self._params.lora_toggled.connect(self._on_lora_toggled)
        self._params.params_changed.connect(self._persist_params)
        self._params.set_dim_source_callback(self._source_image_size)
        rp.addWidget(self._params)

        # ADetailer post-step panel (face refinement after generation)
        from sdqt.widgets.adetailer_panel import ADetailerPanel
        self._adetailer = ADetailerPanel()
        self._adetailer.changed.connect(self._persist_params)
        rp.addWidget(self._adetailer)

        # Tab-level fields (prompt, denoise, resize mode) were only saved on
        # Generate — persist them on change too, debounced for the text fields.
        self._persist_timer = QTimer(self)
        self._persist_timer.setSingleShot(True)
        self._persist_timer.setInterval(600)
        self._persist_timer.timeout.connect(self._persist_params)
        self._prompt.textChanged.connect(self._schedule_persist)
        self._neg_prompt.textChanged.connect(self._schedule_persist)
        self._denoising.valueChanged.connect(self._schedule_persist)
        self._resize_mode.currentIndexChanged.connect(self._schedule_persist)

        # Batch Post-Processing — RealESRGAN sharpen/upscale, etc.
        from sdqt.widgets.batch_postprocess import BatchPostProcessWidget
        self._postproc = BatchPostProcessWidget(self.state)
        self._postproc.progress_changed.connect(self._show_status)
        self._postproc.finished.connect(self._on_postproc_done)
        rp.addWidget(self._postproc)
        rp.addStretch()

        right_scroll.setWidget(right_panel)
        splitter.addWidget(right_scroll)

        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        splitter.setHandleWidth(3)

        self._splitter = splitter

    # ── Mode toggle ──────────────────────────────────────────────────────────

    @property
    def is_txt2img(self) -> bool:
        return self._txt_mode.isChecked()

    def _on_mode_changed(self, txt_checked: bool) -> None:
        """Show/hide source image and denoising based on mode."""
        self._source.setVisible(not txt_checked)
        self._denoise_widget.setVisible(not txt_checked)
        self._interr_row_widget.setVisible(not txt_checked)
        # Adjust splitter: in txt2img mode, right panel gets all space
        if txt_checked:
            self._splitter.setSizes([0, 1])
        else:
            self._splitter.setSizes([3, 2])

    def switch_to_txt2img(self) -> None:
        """Switch to Txt2Img mode (called externally by send-to)."""
        self._txt_mode.setChecked(True)

    def switch_to_img2img(self) -> None:
        """Switch to Img2Img mode (called externally by send-to)."""
        self._img_mode.setChecked(True)

    # ── Family switching ──────────────────────────────────────────────────────

    def set_family(self, strategy) -> None:
        self._strategy = strategy
        self._params.set_family(strategy)
        self._enhance.set_family(strategy)
        self._neg_prompt.setVisible(strategy.has_negative_prompt)

    # ── LoRA ──────────────────────────────────────────────────────────────────

    def _on_lora_toggled(self, tag: str, checked: bool) -> None:
        current = self._prompt.toPlainText()
        if checked:
            if tag not in current:
                sep = " " if current and not current.endswith(" ") else ""
                self._prompt.setPlainText(current + sep + tag)
        else:
            self._prompt.setPlainText(current.replace(tag, "").strip())

    def _persist_source_path(self, path: str) -> None:
        if self._restoring:
            return
        try:
            cfg = ProjectConfig.load(self.project_path)
            s = self._strategy
            if s.config_prefix == "img":
                cfg.img2img_source_path = path
            elif s.config_prefix == "flux":
                cfg.flux_img2img_source_path = path
            else:
                setattr(cfg, f"{s.config_prefix}_source_path", path)
            cfg.save(self.project_path)
        except Exception:
            pass

    @Slot()
    def _schedule_persist(self) -> None:
        """Debounced persist for high-frequency fields (prompt keystrokes)."""
        if self._restoring:
            return
        self._persist_timer.start()

    @Slot()
    def _persist_params(self) -> None:
        """Auto-save model/sampler/scheduler/etc. whenever the user changes them."""
        if self._restoring:
            return
        try:
            cfg = ProjectConfig.load(self.project_path)
            s = self._strategy
            self._params.collect_to_config(cfg, strategy=s)
            self._adetailer.save_to_cfg(cfg)
            s.set_config(cfg, "prompt", self._prompt.toPlainText())
            s.set_config(cfg, "negative_prompt", self._neg_prompt.toPlainText())
            s.set_config(cfg, "denoising_strength", self._denoising.value())
            cfg.img_resize_mode = self._resize_mode.currentIndex()
            cfg.save(self.project_path)
        except Exception as exc:
            logger.debug("img2img _persist_params failed: %s", exc)

    def _source_image_size(self) -> tuple[int, int] | None:
        """Return the source image's (width, height), or None if no source."""
        path = self._source.image_path
        if not path:
            return None
        from PySide6.QtGui import QImageReader
        size = QImageReader(path).size()
        if size.isValid() and size.width() > 0:
            return (size.width(), size.height())
        return None

    def load_source(self, path: str) -> None:
        self._img_mode.setChecked(True)
        self._source.load_image(path)

    # ── Generation ────────────────────────────────────────────────────────────

    @Slot()
    def _on_generate(self) -> None:
        s = self._strategy
        pipeline = s.get_pipeline(self.state)
        if pipeline is None:
            self._show_status(f"{s.display_name} pipeline not loaded. Use Load button.")
            return
        # Validate the source image before claiming the GPU lock so a missing
        # input can't leave the process-wide owner set forever.
        src = None
        if not self.is_txt2img:
            src = self._source.image_path
            if not src:
                self._show_status("No source image selected.")
                return

        if not self.acquire_gpu("Image"):
            return

        cfg = ProjectConfig.load(self.project_path)
        s.set_config(cfg, "prompt", self._prompt.toPlainText())
        s.set_config(cfg, "negative_prompt", self._neg_prompt.toPlainText())

        if self.is_txt2img:
            # Txt2Img — no source image
            self._params.collect_to_config(cfg, strategy=s)
            cfg.save(self.project_path)

            self._gen_btn.setVisible(False)
            self._abort_btn.setVisible(True)
            self._show_status("Generating...")

            WorkerCls = s.resolve_worker("txt2img")
            worker = WorkerCls(pipeline, self.project_name, cfg, parent=self)
        else:
            # Img2Img — source image validated above
            s.set_config(cfg, "denoising_strength", self._denoising.value())
            cfg.img_resize_mode = self._resize_mode.currentIndex()
            if s.config_prefix == "img":
                cfg.img2img_source_path = src
            elif s.config_prefix == "flux":
                cfg.flux_img2img_source_path = src
            else:
                setattr(cfg, f"{s.config_prefix}_source_path", src)
            self._params.collect_to_config(cfg, strategy=s)
            cfg.save(self.project_path)

            self._gen_btn.setVisible(False)
            self._abort_btn.setVisible(True)
            self._show_status("Generating...")

            WorkerCls = s.resolve_worker("img2img")
            worker = WorkerCls(pipeline, self.project_name, src, cfg, parent=self)

        worker.progress.connect(lambda f, d: self._show_status(d))
        worker.finished_ok.connect(self._on_done)
        worker.error.connect(self._on_error)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_done(self, paths: list[str]) -> None:
        self.release_gpu("Image")
        self._gen_btn.setVisible(True)
        self._abort_btn.setVisible(False)
        if paths and self._postproc.is_enabled():
            self._show_status(f"Generated {len(paths)} image(s); post-processing...")
            self._postproc.process(paths)
        else:
            self._gallery.load_images(paths)
            self._show_status(f"Generated {len(paths)} image(s).")

    def _on_postproc_done(self, paths: list[str]) -> None:
        self._gallery.load_images(paths)
        self._show_status(f"Generated + post-processed {len(paths)} image(s).")

    def _on_error(self, msg: str) -> None:
        self.release_gpu("Image")
        self._gen_btn.setVisible(True)
        self._abort_btn.setVisible(False)
        self._show_status(f"Error: {msg}")

    @Slot()
    def _on_abort(self) -> None:
        if self._worker:
            self._worker.abort()

    @Slot()
    def _on_unload(self) -> None:
        method = getattr(self.state, self._strategy.unload_method, None)
        if method:
            method()
        self._show_status(f"{self._strategy.display_name} pipelines unloaded.")

    # -- Interrogation ---------------------------------------------------

    @Slot()
    def _on_interrogate_clip(self) -> None:
        self._run_interrogate(BLIPInterrogateWorker)

    @Slot()
    def _on_interrogate_wd14(self) -> None:
        self._run_interrogate(WD14TaggerWorker)

    def _run_interrogate(self, worker_cls) -> None:
        src = self._source.image_path
        if not src:
            self._show_status("No source image to interrogate.")
            return
        self._clip_btn.setEnabled(False)
        self._wd14_btn.setEnabled(False)
        self._show_status("Interrogating...")

        worker = worker_cls(src, parent=self)
        worker.finished_ok.connect(self._on_interrogate_done)
        worker.error.connect(self._on_interrogate_error)
        worker.status.connect(lambda s: self._show_status(s))
        worker.finished.connect(worker.deleteLater)
        self._interrogate_worker = worker
        worker.start()

    def _on_interrogate_done(self, caption: str) -> None:
        self._clip_btn.setEnabled(True)
        self._wd14_btn.setEnabled(True)
        self._prompt.setPlainText(caption)
        self._show_status("Interrogation complete.")

    def _on_interrogate_error(self, msg: str) -> None:
        self._clip_btn.setEnabled(True)
        self._wd14_btn.setEnabled(True)
        self._show_status(f"Interrogation error: {msg}")

    # -- Save 3D ----------------------------------------------------------

    @Slot()
    def _on_save_3d(self) -> None:
        """Generate a 3D mesh from the selected gallery image via TripoSR."""
        path = self._gallery.selected_path
        if not path:
            self._show_status("No image selected.")
            return
        from sdqt.models.manager import check_and_prompt_download
        if not check_and_prompt_download(
            "3d_modeling", self.state.model_registry, self,
            on_progress=lambda f, d: self._show_status(d),
        ):
            return
        self._show_status("Loading TripoSR...")
        try:
            pipeline = self.state.load_triposr_pipeline()
        except Exception as exc:
            self._show_status(f"Failed: {exc}")
            return
        meshes_dir = self.project_path / "meshes"
        meshes_dir.mkdir(parents=True, exist_ok=True)
        output = str(meshes_dir / f"{Path(path).stem}.obj")
        from sdqt.workers.model3d import MeshGenerationWorker
        self._save_3d_btn.setEnabled(False)
        worker = MeshGenerationWorker(
            pipeline=pipeline, image_path=path, output_path=output,
            resolution=256, output_format="obj", remove_bg=True, parent=self,
        )
        worker.progress.connect(lambda f, d: self._show_status(d))
        worker.finished_ok.connect(lambda p: (
            self._save_3d_btn.setEnabled(True),
            self._show_status(f"Mesh saved: {Path(p).name}"),
        ))
        worker.error.connect(lambda msg: (
            self._save_3d_btn.setEnabled(True),
            self._show_status(f"3D error: {msg}"),
        ))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    # ── Project persistence ───────────────────────────────────────────────────

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        self._gallery.clear()
        self._restoring = True
        try:
            cfg = ProjectConfig.load(self.project_path)
        except Exception:
            self._restoring = False
            return
        s = self._strategy
        # Restore source path
        if s.config_prefix == "img":
            src = cfg.img2img_source_path or ""
        elif s.config_prefix == "flux":
            src = getattr(cfg, "flux_img2img_source_path", "") or ""
        else:
            src = getattr(cfg, f"{s.config_prefix}_source_path", "") or ""
        if src and Path(src).is_file():
            self._source.load_image(src)
        else:
            self._source.clear_image()
        self._prompt.setPlainText(s.get_config(cfg, "prompt") or "")
        self._neg_prompt.setPlainText(s.get_config(cfg, "negative_prompt") or "")
        self._denoising.setValue(s.get_config(cfg, "denoising_strength") or 0.75)
        self._resize_mode.setCurrentIndex(getattr(cfg, "img_resize_mode", 0) or 0)
        self._params.restore_from_config(cfg, strategy=s)
        self._adetailer.load_from_cfg(cfg)
        self._restoring = False

    # -- Prompt enhancement ------------------------------------------------

    def _on_enhance(self, which: str, _text: str, style: str) -> None:
        from sdqt.models.manager import check_and_prompt_download
        if not check_and_prompt_download(
            "prompt_enhance", self.state.model_registry, self,
            on_progress=lambda f, d: self._show_status(d),
        ):
            return
        current = self._prompt.toPlainText() if which == "pos" else self._neg_prompt.toPlainText()
        self._enhance.set_enabled_buttons(False)
        worker = PromptEnhanceWorker(current, style, which, self.state, parent=self)
        worker.finished_ok.connect(lambda r: self._on_enhance_done(which, r))
        worker.error.connect(self._on_enhance_error)
        worker.status.connect(lambda s: self._show_status(s))
        worker.finished.connect(worker.deleteLater)
        self._enhance_worker = worker
        worker.start()

    def _on_enhance_done(self, which: str, result: str) -> None:
        self._enhance.set_enabled_buttons(True)
        (self._prompt if which == "pos" else self._neg_prompt).setPlainText(result)
        self._show_status("Prompt enhanced.")

    def _on_enhance_error(self, msg: str) -> None:
        self._enhance.set_enabled_buttons(True)
        self._show_status(f"Enhance error: {msg}")

    def populate_options(self, **kwargs) -> None:
        self._params.populate_options(**kwargs)
