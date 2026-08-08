"""Style Transfer LoRA — 6-page wizard sequence."""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from supremediffusion.config.project_config import ProjectConfig

from sdqt.widgets.image_gallery import ImageGalleryWidget
from sdqt.widgets.sequence_wizard import SequenceWizard, WizardPage
from sdqt.workers.pipeline_load import PipelineLoadWorker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared style transfer state
# ---------------------------------------------------------------------------


@dataclass
class StyleTransferState:
    name: str = ""
    description: str = ""
    trigger_word: str = ""
    model_type: str = "SDXL"  # "SD 1.5" or "SDXL"
    # Image collection
    source_images: list[str] = field(default_factory=list)
    # Generation (optional)
    gen_prompt: str = ""
    gen_count: int = 4
    checkpoint: str = ""
    sampler: str = ""
    steps: int = 20
    cfg_scale: float = 7.0
    width: int = 1024
    height: int = 1024
    seed: int = -1
    # Processed
    processed_dir: str = ""
    processed_images: list[str] = field(default_factory=list)
    target_resolution: int = 1024  # 512 for SD1.5, 1024 for SDXL
    # Captions
    captions: dict = field(default_factory=dict)  # path -> caption string
    caption_method: str = "qwen"  # "qwen" or "wd14"
    # Output
    dataset_dir: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_path(app_state) -> Path:
    name = app_state.current_project or "_default"
    return app_state.project_manager.get_project_path(name)


def _project_name(app_state) -> str:
    return app_state.current_project or "_default"


def _load_project_config(app_state) -> ProjectConfig:
    return ProjectConfig.load(_project_path(app_state))


# ═══════════════════════════════════════════════════════════════════════════
# Page 1 — Style Info
# ═══════════════════════════════════════════════════════════════════════════


class StyleInfoPage(WizardPage):
    """Name, description, trigger word, model type."""

    def __init__(self, st_state: StyleTransferState, app_state, parent=None) -> None:
        super().__init__(parent)
        self._st = st_state
        self._app_state = app_state
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Name
        row = QHBoxLayout()
        row.addWidget(QLabel("Style Name:"))
        self._name = QLineEdit()
        self._name.setPlaceholderText("e.g. Ghibli Watercolor")
        self._name.textChanged.connect(self._auto_trigger)
        row.addWidget(self._name)
        layout.addLayout(row)

        # Description
        layout.addWidget(QLabel("Description (what makes this style distinctive):"))
        self._desc = QPlainTextEdit()
        self._desc.setPlaceholderText(
            "Describe the visual characteristics — color palette, brushwork, "
            "line quality, lighting style, composition tendencies, etc."
        )
        self._desc.setMaximumHeight(120)
        layout.addWidget(self._desc)

        # Trigger word
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Trigger Word:"))
        self._trigger = QLineEdit()
        self._trigger.setPlaceholderText("e.g. ghibli_watercolor")
        row2.addWidget(self._trigger)
        layout.addLayout(row2)

        # Model type
        model_group = QGroupBox("Model Type")
        model_layout = QHBoxLayout(model_group)
        self._model_group = QButtonGroup(self)
        self._rb_sd15 = QRadioButton("SD 1.5")
        self._rb_sdxl = QRadioButton("SDXL")
        self._model_group.addButton(self._rb_sd15, 0)
        self._model_group.addButton(self._rb_sdxl, 1)
        self._rb_sdxl.setChecked(True)
        self._model_group.idToggled.connect(self._on_model_type_changed)
        model_layout.addWidget(self._rb_sd15)
        model_layout.addWidget(self._rb_sdxl)
        model_layout.addStretch()
        layout.addWidget(model_group)

        layout.addStretch()

    def _auto_trigger(self, text: str) -> None:
        """Auto-fill trigger word from name."""
        trigger = text.strip().lower().replace(" ", "_")
        trigger = "".join(c for c in trigger if c.isalnum() or c == "_")
        self._trigger.setText(trigger)

    def _on_model_type_changed(self, btn_id: int, checked: bool) -> None:
        if not checked:
            return
        if btn_id == 0:
            self._st.model_type = "SD 1.5"
            self._st.target_resolution = 512
        else:
            self._st.model_type = "SDXL"
            self._st.target_resolution = 1024

    def on_enter(self) -> None:
        self._name.setText(self._st.name)
        self._desc.setPlainText(self._st.description)
        self._trigger.setText(self._st.trigger_word)
        if self._st.model_type == "SD 1.5":
            self._rb_sd15.setChecked(True)
        else:
            self._rb_sdxl.setChecked(True)

    def on_leave(self) -> None:
        self._st.name = self._name.text().strip()
        self._st.description = self._desc.toPlainText().strip()
        self._st.trigger_word = self._trigger.text().strip()
        if self._rb_sd15.isChecked():
            self._st.model_type = "SD 1.5"
            self._st.target_resolution = 512
        else:
            self._st.model_type = "SDXL"
            self._st.target_resolution = 1024

    def validate(self) -> bool:
        name = self._name.text().strip()
        trigger = self._trigger.text().strip()
        if not name:
            QMessageBox.warning(self, "Missing", "Please enter a style name.")
            return False
        if not trigger:
            QMessageBox.warning(self, "Missing", "Please enter a trigger word.")
            return False
        # Save fields without calling on_leave
        self._st.name = name
        self._st.description = self._desc.toPlainText().strip()
        self._st.trigger_word = trigger
        if self._rb_sd15.isChecked():
            self._st.model_type = "SD 1.5"
            self._st.target_resolution = 512
        else:
            self._st.model_type = "SDXL"
            self._st.target_resolution = 1024
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 2 — Image Collection
# ═══════════════════════════════════════════════════════════════════════════


class ImageCollectionPage(WizardPage):
    """Import reference images and optionally generate more."""

    def __init__(self, st_state: StyleTransferState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._st = st_state
        self._app_state = app_state
        self._wizard = wizard
        self._sd_loaded_here = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- Import section ---
        import_group = QGroupBox("Import Images")
        import_layout = QVBoxLayout(import_group)
        import_row = QHBoxLayout()
        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.setFixedWidth(100)
        self._browse_btn.clicked.connect(self._browse_images)
        import_row.addWidget(self._browse_btn)
        self._import_count = QLabel("0 imported")
        import_row.addWidget(self._import_count)
        import_row.addStretch()
        import_layout.addLayout(import_row)
        layout.addWidget(import_group)

        # --- Generate section ---
        gen_group = QGroupBox("Generate Images (optional)")
        gen_layout = QVBoxLayout(gen_group)

        gen_layout.addWidget(QLabel("Prompt:"))
        self._gen_prompt = QPlainTextEdit()
        self._gen_prompt.setMaximumHeight(80)
        self._gen_prompt.setPlaceholderText(
            "Describe the kind of images to generate in this style..."
        )
        gen_layout.addWidget(self._gen_prompt)

        gen_row = QHBoxLayout()
        gen_row.setSpacing(6)
        gen_row.addWidget(QLabel("Count:"))
        self._gen_count = QSpinBox()
        self._gen_count.setRange(1, 16)
        self._gen_count.setValue(4)
        self._gen_count.setFixedWidth(75)
        gen_row.addWidget(self._gen_count)
        self._gen_btn = QPushButton("Generate")
        self._gen_btn.setFixedWidth(100)
        self._gen_btn.clicked.connect(self._generate)
        gen_row.addWidget(self._gen_btn)
        gen_row.addStretch()
        gen_layout.addLayout(gen_row)
        layout.addWidget(gen_group)

        # --- Gallery ---
        self._gallery = ImageGalleryWidget("Collected Images")
        self._gallery.image_selected.connect(self._on_image_selected)
        layout.addWidget(self._gallery, 1)

        # Remove button
        rm_row = QHBoxLayout()
        rm_row.addStretch()
        self._remove_btn = QPushButton("Remove Selected")
        self._remove_btn.setFixedWidth(140)
        self._remove_btn.clicked.connect(self._remove_selected)
        self._remove_btn.setEnabled(False)
        rm_row.addWidget(self._remove_btn)
        layout.addLayout(rm_row)

        self._selected_image: str = ""

    def on_enter(self) -> None:
        if self._st.source_images:
            self._gallery.load_images(self._st.source_images)
            self._import_count.setText(f"{len(self._st.source_images)} total")
        if self._st.gen_prompt:
            self._gen_prompt.setPlainText(self._st.gen_prompt)

    def _browse_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Style Reference Images", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if paths:
            self._st.source_images.extend(paths)
            self._gallery.load_images(self._st.source_images)
            self._import_count.setText(f"{len(self._st.source_images)} total")

    def _on_image_selected(self, path: str) -> None:
        self._selected_image = path
        self._remove_btn.setEnabled(bool(path))

    def _remove_selected(self) -> None:
        if self._selected_image and self._selected_image in self._st.source_images:
            self._st.source_images.remove(self._selected_image)
            self._gallery.load_images(self._st.source_images)
            self._import_count.setText(f"{len(self._st.source_images)} total")
            self._selected_image = ""
            self._remove_btn.setEnabled(False)

    def _generate(self) -> None:
        prompt = self._gen_prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Missing", "Please enter a generation prompt.")
            return

        self._st.gen_prompt = prompt
        self._st.gen_count = self._gen_count.value()

        cfg = _load_project_config(self._app_state)
        cfg.img_prompt = prompt
        cfg.img_negative_prompt = ""
        cfg.img_batch_count = self._st.gen_count
        cfg.img_batch_size = 1
        cfg.img_steps = self._st.steps
        cfg.img_cfg_scale = self._st.cfg_scale
        cfg.img_width = self._st.width
        cfg.img_height = self._st.height
        cfg.img_seed = self._st.seed
        if self._st.checkpoint:
            cfg.img_checkpoint = self._st.checkpoint
        if self._st.sampler:
            cfg.img_sampler = self._st.sampler
        cfg.save(_project_path(self._app_state))

        if self._app_state.img_pipeline is None:
            self._sd_loaded_here = True
            loader = PipelineLoadWorker(self._app_state, "load_sd_pipelines", parent=self)
            self._wizard._run_worker(loader, on_done=lambda _: self._run_txt2img(cfg))
        else:
            self._run_txt2img(cfg)

    def _run_txt2img(self, cfg) -> None:
        from sdqt.workers.image import Txt2ImgWorker

        worker = Txt2ImgWorker(
            pipeline=self._app_state.img_pipeline,
            project_name=_project_name(self._app_state),
            project_config=cfg,
            parent=self,
        )
        self._wizard._run_worker(worker, on_done=self._on_images, on_error=self._on_error)

    @Slot(object)
    def _on_images(self, paths) -> None:
        if paths:
            new_paths = list(paths) if isinstance(paths, (list, tuple)) else [str(paths)]
            self._st.source_images.extend(new_paths)
            self._gallery.load_images(self._st.source_images)
            self._import_count.setText(f"{len(self._st.source_images)} total")

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self._wizard._status.setText(f"Generation error: {msg}")

    def on_leave(self) -> None:
        self._st.gen_prompt = self._gen_prompt.toPlainText().strip()
        # Free SD VRAM if we loaded it on this page
        if self._sd_loaded_here:
            self._app_state.unload_sd_pipelines()
            self._sd_loaded_here = False

    def validate(self) -> bool:
        # Save fields without calling on_leave
        self._st.gen_prompt = self._gen_prompt.toPlainText().strip()
        if len(self._st.source_images) < 5:
            QMessageBox.warning(
                self, "Not Enough",
                f"Please collect at least 5 images ({len(self._st.source_images)} so far).",
            )
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 3 — Crop / Resize
# ═══════════════════════════════════════════════════════════════════════════


class CropResizePage(WizardPage):
    """Crop and resize all collected images to a uniform resolution."""

    def __init__(self, st_state: StyleTransferState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._st = st_state
        self._app_state = app_state
        self._wizard = wizard
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Info
        self._info_label = QLabel()
        layout.addWidget(self._info_label)

        # Resolution combo
        res_row = QHBoxLayout()
        res_row.setSpacing(6)
        res_row.addWidget(QLabel("Target Resolution:"))
        self._res_combo = QComboBox()
        self._res_combo.addItems(["512x512", "768x768", "1024x1024"])
        self._res_combo.setFixedWidth(140)
        res_row.addWidget(self._res_combo)
        res_row.addStretch()
        layout.addLayout(res_row)

        # Crop mode
        crop_row = QHBoxLayout()
        crop_row.setSpacing(6)
        crop_row.addWidget(QLabel("Crop Mode:"))
        self._crop_combo = QComboBox()
        self._crop_combo.addItems(["Center Crop", "Smart Crop (face-aware)", "Resize (letterbox)"])
        self._crop_combo.setFixedWidth(200)
        crop_row.addWidget(self._crop_combo)
        crop_row.addStretch()
        layout.addLayout(crop_row)

        # Process button
        self._process_btn = QPushButton("Process All")
        self._process_btn.setFixedWidth(140)
        self._process_btn.clicked.connect(self._process_all)
        layout.addWidget(self._process_btn)

        # Gallery
        self._gallery = ImageGalleryWidget("Processed Images")
        layout.addWidget(self._gallery, 1)

        layout.addStretch()

    def on_enter(self) -> None:
        count = len(self._st.source_images)
        self._info_label.setText(
            f"<b>{count} images</b> to process. "
            f"Model type: <b>{self._st.model_type}</b>"
        )
        # Pre-select resolution based on model type
        if self._st.target_resolution <= 512:
            self._res_combo.setCurrentText("512x512")
        elif self._st.target_resolution <= 768:
            self._res_combo.setCurrentText("768x768")
        else:
            self._res_combo.setCurrentText("1024x1024")

        # Show existing processed images if any
        if self._st.processed_images:
            self._gallery.load_images(self._st.processed_images)

    def _parse_resolution(self) -> int:
        text = self._res_combo.currentText()
        return int(text.split("x")[0])

    def _process_all(self) -> None:
        from PIL import Image

        target = self._parse_resolution()
        self._st.target_resolution = target
        crop_mode = self._crop_combo.currentText()

        # Create processed directory
        tmp_dir = Path(tempfile.mkdtemp(prefix="sdqt_style_crop_"))
        self._st.processed_dir = str(tmp_dir)
        self._st.processed_images.clear()

        self._wizard.set_busy(True)
        self._wizard._status.setText("Processing images...")

        for i, src_path in enumerate(self._st.source_images):
            try:
                img = Image.open(src_path).convert("RGB")

                if "Center Crop" in crop_mode:
                    img = self._center_crop(img, target)
                elif "letterbox" in crop_mode.lower():
                    img = self._letterbox(img, target)
                else:
                    # Smart crop — use center for v1
                    img = self._center_crop(img, target)

                out_path = tmp_dir / f"style_{i + 1:03d}.png"
                img.save(str(out_path), "PNG")
                self._st.processed_images.append(str(out_path))
            except Exception as exc:
                logger.warning("Failed to process %s: %s", src_path, exc)

            self._wizard._status.setText(
                f"Processing {i + 1}/{len(self._st.source_images)}..."
            )

        self._gallery.load_images(self._st.processed_images)
        self._process_btn.setText("Re-process All")
        self._wizard.set_busy(False)
        self._wizard._status.setText(
            f"Processed {len(self._st.processed_images)} images to {target}x{target}"
        )

    @staticmethod
    def _center_crop(img, target: int):
        """Resize shortest side to target, then center crop."""
        from PIL import Image

        w, h = img.size
        scale = target / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - target) // 2
        top = (new_h - target) // 2
        return img.crop((left, top, left + target, top + target))

    @staticmethod
    def _letterbox(img, target: int):
        """Resize to fit within target maintaining aspect, pad with black."""
        from PIL import Image

        w, h = img.size
        scale = target / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new("RGB", (target, target), (0, 0, 0))
        paste_x = (target - new_w) // 2
        paste_y = (target - new_h) // 2
        canvas.paste(img, (paste_x, paste_y))
        return canvas

    def validate(self) -> bool:
        if not self._st.processed_images:
            QMessageBox.warning(self, "Missing", "Please process the images first.")
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 4 — Auto-Caption
# ═══════════════════════════════════════════════════════════════════════════


class AutoCaptionPage(WizardPage):
    """Auto-caption all processed images using Qwen VL or WD14 Tagger."""

    def __init__(self, st_state: StyleTransferState, app_state, wizard: SequenceWizard, parent=None) -> None:
        super().__init__(parent)
        self._st = st_state
        self._app_state = app_state
        self._wizard = wizard
        self._caption_idx = 0
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Method selector
        method_row = QHBoxLayout()
        method_row.setSpacing(6)
        method_row.addWidget(QLabel("Caption Method:"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(["Qwen VL", "WD14 Tagger"])
        self._method_combo.setFixedWidth(140)
        method_row.addWidget(self._method_combo)
        method_row.addStretch()
        layout.addLayout(method_row)

        # Trigger word reminder
        self._trigger_label = QLabel()
        self._trigger_label.setStyleSheet("color: #aaa; font-size: 12px;")
        layout.addWidget(self._trigger_label)

        # Caption button
        self._caption_btn = QPushButton("Caption All")
        self._caption_btn.setFixedWidth(140)
        self._caption_btn.clicked.connect(self._start_captioning)
        layout.addWidget(self._caption_btn)

        # Progress
        self._caption_progress = QLabel()
        self._caption_progress.setStyleSheet("color: #aaa; font-size: 12px;")
        layout.addWidget(self._caption_progress)

        layout.addStretch()

    def on_enter(self) -> None:
        self._trigger_label.setText(
            f"Trigger word: <b>{self._st.trigger_word}</b> "
            f"(will be prepended to all captions)"
        )
        if self._st.caption_method == "wd14":
            self._method_combo.setCurrentText("WD14 Tagger")
        else:
            self._method_combo.setCurrentText("Qwen VL")

    def _start_captioning(self) -> None:
        method_text = self._method_combo.currentText()
        self._st.caption_method = "wd14" if "WD14" in method_text else "qwen"
        self._st.captions.clear()
        self._caption_idx = 0

        if self._st.caption_method == "qwen":
            # Load Qwen VL first via PipelineLoadWorker
            loader = PipelineLoadWorker(self._app_state, "load_qwen_vl", parent=self)
            self._wizard._run_worker(loader, on_done=lambda _: self._caption_next())
        else:
            # WD14 manages its own model loading
            self._caption_next()

    def _caption_next(self) -> None:
        if self._caption_idx >= len(self._st.processed_images):
            self._caption_progress.setText(
                f"Done — {len(self._st.captions)} captions generated."
            )
            self._caption_btn.setText("Re-caption All")
            self._wizard.set_busy(False)
            return

        img_path = self._st.processed_images[self._caption_idx]
        total = len(self._st.processed_images)
        self._caption_progress.setText(
            f"Captioning {self._caption_idx + 1}/{total}..."
        )

        if self._st.caption_method == "qwen":
            from sdqt.workers.interrogate import QwenCaptionWorker

            worker = QwenCaptionWorker(
                image_path=img_path,
                state=self._app_state,
                parent=self,
            )
        else:
            from sdqt.workers.interrogate import WD14TaggerWorker

            worker = WD14TaggerWorker(
                image_path=img_path,
                parent=self,
            )

        self._wizard._run_worker(
            worker,
            on_done=self._on_caption_done,
            on_error=self._on_caption_error,
        )

    @Slot(object)
    def _on_caption_done(self, caption_text) -> None:
        caption = str(caption_text).strip()
        # Prepend trigger word
        trigger = self._st.trigger_word
        if trigger and not caption.lower().startswith(trigger.lower()):
            caption = f"{trigger}, {caption}"

        img_path = self._st.processed_images[self._caption_idx]
        self._st.captions[img_path] = caption
        self._caption_idx += 1
        self._caption_next()

    @Slot(str)
    def _on_caption_error(self, msg: str) -> None:
        img_path = self._st.processed_images[self._caption_idx]
        # Use trigger word as fallback caption
        self._st.captions[img_path] = self._st.trigger_word
        logger.warning("Caption error for %s: %s", img_path, msg)
        self._caption_idx += 1
        self._caption_next()

    def on_leave(self) -> None:
        # Free Qwen VL VRAM
        self._app_state.unload_qwen_vl()

    def validate(self) -> bool:
        if not self._st.captions:
            QMessageBox.warning(self, "Missing", "Please caption the images first.")
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 5 — Caption Review
# ═══════════════════════════════════════════════════════════════════════════


class CaptionReviewPage(WizardPage):
    """Review and edit captions for each processed image."""

    def __init__(self, st_state: StyleTransferState, app_state, parent=None) -> None:
        super().__init__(parent)
        self._st = st_state
        self._app_state = app_state
        self._caption_edits: dict[str, QPlainTextEdit] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Apply trigger word button
        btn_row = QHBoxLayout()
        self._apply_trigger_btn = QPushButton("Apply Trigger Word to All")
        self._apply_trigger_btn.setFixedWidth(200)
        self._apply_trigger_btn.clicked.connect(self._apply_trigger_word)
        btn_row.addWidget(self._apply_trigger_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Scrollable grid of thumbnails + caption editors
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll_content = QWidget()
        self._grid = QGridLayout(self._scroll_content)
        self._grid.setSpacing(8)
        self._scroll.setWidget(self._scroll_content)
        layout.addWidget(self._scroll, 1)

    def on_enter(self) -> None:
        # Rebuild the grid each time we enter (captions may have changed)
        self._caption_edits.clear()

        # Clear existing grid items
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        for i, img_path in enumerate(self._st.processed_images):
            row = i
            # Thumbnail
            thumb = QLabel()
            thumb.setFixedSize(128, 128)
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setStyleSheet("background: #2a2a2a; border: 1px solid #444;")
            if Path(img_path).is_file():
                pm = QPixmap(img_path).scaled(
                    128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
                thumb.setPixmap(pm)
            self._grid.addWidget(thumb, row, 0)

            # Caption editor
            edit = QPlainTextEdit()
            edit.setMaximumHeight(80)
            caption = self._st.captions.get(img_path, "")
            edit.setPlainText(caption)
            self._caption_edits[img_path] = edit
            self._grid.addWidget(edit, row, 1)

    def _apply_trigger_word(self) -> None:
        trigger = self._st.trigger_word
        if not trigger:
            return
        for img_path, edit in self._caption_edits.items():
            text = edit.toPlainText().strip()
            if not text.lower().startswith(trigger.lower()):
                edit.setPlainText(f"{trigger}, {text}")

    def on_leave(self) -> None:
        self._save_captions()

    def _save_captions(self) -> None:
        for img_path, edit in self._caption_edits.items():
            self._st.captions[img_path] = edit.toPlainText().strip()

    def validate(self) -> bool:
        self._save_captions()
        # Check all images have non-empty captions
        for img_path in self._st.processed_images:
            caption = self._st.captions.get(img_path, "").strip()
            if not caption:
                QMessageBox.warning(
                    self, "Empty Caption",
                    "All images must have non-empty captions.",
                )
                return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Page 6 — Send to LoRA
# ═══════════════════════════════════════════════════════════════════════════


class SendToLoRAPage(WizardPage):
    """Prepare dataset and send to LoRA training."""

    def __init__(self, st_state: StyleTransferState, app_state, wizard, parent=None) -> None:
        super().__init__(parent)
        self._st = st_state
        self._app_state = app_state
        self._wizard = wizard
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(
            "<b>Prepare & Send to LoRA Training</b><br>"
            "This will create a training dataset with images and caption files."
        ))

        # Model type
        model_row = QHBoxLayout()
        model_row.setSpacing(6)
        model_row.addWidget(QLabel("LoRA Model Type:"))
        self._model_type = QComboBox()
        self._model_type.addItems(["SD 1.5", "SDXL"])
        self._model_type.setFixedWidth(100)
        model_row.addWidget(self._model_type)

        model_row.addWidget(QLabel("Checkpoint:"))
        self._ckpt_combo = QComboBox()
        self._ckpt_combo.setMinimumWidth(250)
        model_row.addWidget(self._ckpt_combo, 1)
        model_row.addStretch()
        layout.addLayout(model_row)

        # Summary
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        self._summary.setTextFormat(Qt.RichText)
        self._summary.setStyleSheet("color: #ccc; font-size: 12px;")
        layout.addWidget(self._summary)

        # Send button
        self._send_btn = QPushButton("Prepare && Send to LoRA")
        self._send_btn.setFixedWidth(220)
        self._send_btn.setStyleSheet(
            "QPushButton { background: #0078d4; color: white; font-weight: bold; }"
            "QPushButton:hover { background: #1a8ae8; }"
        )
        self._send_btn.clicked.connect(self._prepare_and_send)
        layout.addWidget(self._send_btn)

        layout.addStretch()

    def on_enter(self) -> None:
        # Pre-fill model type from page 1
        self._model_type.setCurrentText(self._st.model_type)

        # Populate checkpoint combo if empty
        if self._ckpt_combo.count() == 0:
            self._ckpt_infos: dict[str, str] = {}  # name -> model_type
            ckpt_dir = self._app_state.global_config.model_paths.get("sd_checkpoint_dir", "")
            if ckpt_dir:
                try:
                    from supremediffusion.models.sd_models import scan_checkpoints

                    for info in scan_checkpoints(ckpt_dir):
                        self._ckpt_combo.addItem(info.name, info.filename)
                        self._ckpt_infos[info.name] = info.model_type
                except Exception:
                    pass

            # Auto-set model type when checkpoint changes
            self._ckpt_combo.currentTextChanged.connect(self._on_ckpt_changed)

            # Default to the checkpoint used for generation
            if self._st.checkpoint:
                idx = self._ckpt_combo.findText(self._st.checkpoint)
                if idx >= 0:
                    self._ckpt_combo.setCurrentIndex(idx)
                    mt = self._ckpt_infos.get(self._st.checkpoint, "")
                    if mt == "sdxl":
                        self._model_type.setCurrentText("SDXL")
                    elif mt == "sd15":
                        self._model_type.setCurrentText("SD 1.5")

        # Summary
        self._summary.setText(
            f"<b>Style:</b> {self._st.name}<br>"
            f"<b>Trigger:</b> {self._st.trigger_word}<br>"
            f"<b>Images:</b> {len(self._st.processed_images)}<br>"
            f"<b>Captions:</b> {len(self._st.captions)}"
        )

    def _on_ckpt_changed(self, name: str) -> None:
        mt = getattr(self, "_ckpt_infos", {}).get(name, "")
        if mt == "sdxl":
            self._model_type.setCurrentText("SDXL")
        elif mt == "sd15":
            self._model_type.setCurrentText("SD 1.5")

    def _prepare_and_send(self) -> None:
        safe_name = "".join(
            c if c.isalnum() or c in " _-" else "_" for c in self._st.name
        ).strip() or "style"

        dataset_dir = _project_path(self._app_state) / "lora_datasets" / f"{safe_name}_style"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Copy images and write caption .txt files
        for i, img_path in enumerate(self._st.processed_images):
            src = Path(img_path)
            if not src.is_file():
                continue

            dst_img = dataset_dir / f"{safe_name}_{i + 1:03d}{src.suffix}"
            shutil.copy2(str(src), str(dst_img))

            # Write caption file
            caption = self._st.captions.get(img_path, self._st.trigger_word)
            caption_file = dataset_dir / f"{dst_img.stem}.txt"
            caption_file.write_text(caption, encoding="utf-8")

        self._st.dataset_dir = str(dataset_dir)

        model_type = self._model_type.currentText()
        checkpoint = self._ckpt_combo.currentText()

        self._wizard.send_to_lora.emit(str(dataset_dir), model_type, checkpoint)
        self._wizard.accept()


# ═══════════════════════════════════════════════════════════════════════════
# StyleTransferWizard — ties all 6 pages together
# ═══════════════════════════════════════════════════════════════════════════


class StyleTransferWizard(SequenceWizard):
    """6-page wizard: Info -> Collect -> Crop -> Caption -> Review -> LoRA."""

    send_to_lora = Signal(str, str, str)  # dataset_dir, model_type, checkpoint

    def __init__(self, state, *, img_params: dict | None = None, parent=None) -> None:
        self._st = StyleTransferState()

        # Persistent header: checkpoint selector
        header = QWidget()
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.addWidget(QLabel("Image Model:"))
        self._ckpt_combo = QComboBox()
        self._ckpt_combo.setMinimumWidth(300)
        h_layout.addWidget(self._ckpt_combo, 1)
        h_layout.addStretch()

        super().__init__(
            "Style Transfer LoRA",
            state,
            header_widget=header,
            parent=parent,
        )
        self.setMinimumSize(850, 700)

        # Populate checkpoint combo
        self._populate_checkpoints()
        self._ckpt_combo.currentTextChanged.connect(self._on_checkpoint_changed)

        # Apply image params from Sequences tab
        if img_params:
            self._st.checkpoint = img_params.get("checkpoint", "")
            self._st.sampler = img_params.get("sampler", "")
            self._st.steps = img_params.get("steps", 20)
            self._st.cfg_scale = img_params.get("cfg_scale", 7.0)
            self._st.width = img_params.get("width", 1024)
            self._st.height = img_params.get("height", 1024)
            self._st.seed = img_params.get("seed", -1)
        else:
            cfg = ProjectConfig.load(
                state.project_manager.get_project_path(state.current_project or "_default")
            )
            self._st.checkpoint = cfg.img_checkpoint
            self._st.sampler = cfg.img_sampler
            self._st.steps = cfg.img_steps
            self._st.cfg_scale = cfg.img_cfg_scale
            self._st.width = cfg.img_width
            self._st.height = cfg.img_height
            self._st.seed = cfg.img_seed

        if self._st.checkpoint:
            idx = self._ckpt_combo.findText(self._st.checkpoint)
            if idx >= 0:
                self._ckpt_combo.setCurrentIndex(idx)

        # Build pages
        self._pages = [
            StyleInfoPage(self._st, state, self),
            ImageCollectionPage(self._st, state, self, self),
            CropResizePage(self._st, state, self, self),
            AutoCaptionPage(self._st, state, self, self),
            CaptionReviewPage(self._st, state, self),
            SendToLoRAPage(self._st, state, self, self),
        ]
        self._finish_setup()

    def _populate_checkpoints(self) -> None:
        ckpt_dir = self._state.global_config.model_paths.get("sd_checkpoint_dir", "")
        if not ckpt_dir:
            return
        try:
            from supremediffusion.models.sd_models import scan_checkpoints

            infos = scan_checkpoints(ckpt_dir)
            for info in infos:
                self._ckpt_combo.addItem(info.name)
        except Exception as exc:
            logger.warning("Could not scan checkpoints: %s", exc)

    @Slot(str)
    def _on_checkpoint_changed(self, name: str) -> None:
        self._st.checkpoint = name

    def _on_finish(self) -> None:
        self.accept()
