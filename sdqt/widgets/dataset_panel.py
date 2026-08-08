"""Dataset browser panel for LoRA training — image list + caption editor."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_THUMB_SIZE = 80


class DatasetPanel(QWidget):
    """Left-side panel: dataset folder, image list, caption editor, tools.

    Signals:
        dataset_changed(str): emitted when dataset folder changes (path)
        caption_batch_requested(str): 'blip' or 'wd14' — request batch captioning
    """

    dataset_changed = Signal(str)
    caption_batch_requested = Signal(str)  # 'blip', 'wd14', or 'qwen'
    prepare_requested = Signal(str, str)   # (source_folder, dest_folder)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._dataset_path: str = ""
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self._save_caption)
        self._current_image_path: str = ""
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Folder picker
        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Dataset:"))
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Select image folder...")
        self._folder_edit.setReadOnly(True)
        folder_row.addWidget(self._folder_edit, 1)
        self._browse_btn = QPushButton("Browse")
        self._browse_btn.clicked.connect(self._on_browse)
        folder_row.addWidget(self._browse_btn)
        self._prepare_btn = QPushButton("Prepare Dataset")
        self._prepare_btn.setToolTip(
            "Import a folder of raw images: copies, renames sequentially, "
            "and auto-captions with Qwen VL"
        )
        self._prepare_btn.clicked.connect(self._on_prepare)
        folder_row.addWidget(self._prepare_btn)
        layout.addLayout(folder_row)

        # Image count
        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(self._count_label)

        # Image list + preview splitter
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Image list (thumbnails)
        self._image_list = QListWidget()
        self._image_list.setIconSize(QPixmap(_THUMB_SIZE, _THUMB_SIZE).size())
        self._image_list.setSpacing(2)
        self._image_list.currentItemChanged.connect(self._on_image_selected)
        splitter.addWidget(self._image_list)

        # Preview + caption editor
        editor_widget = QWidget()
        editor_lay = QVBoxLayout(editor_widget)
        editor_lay.setContentsMargins(0, 0, 0, 0)
        editor_lay.setSpacing(2)

        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setFixedHeight(200)
        self._preview.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        editor_lay.addWidget(self._preview)

        editor_lay.addWidget(QLabel("Caption:"))
        self._caption_edit = QPlainTextEdit()
        self._caption_edit.setPlaceholderText("(no caption — type to create)")
        self._caption_edit.setMaximumHeight(100)
        self._caption_edit.textChanged.connect(self._on_caption_changed)
        editor_lay.addWidget(self._caption_edit)

        splitter.addWidget(editor_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        # Caption tools
        tools_group = QGroupBox("Caption Tools")
        tools_lay = QVBoxLayout(tools_group)
        tools_lay.setContentsMargins(4, 4, 4, 4)
        tools_lay.setSpacing(2)

        btn_row1 = QHBoxLayout()
        self._blip_btn = QPushButton("Caption All (BLIP)")
        self._blip_btn.setToolTip("Auto-caption all uncaptioned images with BLIP")
        self._blip_btn.clicked.connect(lambda: self.caption_batch_requested.emit("blip"))
        btn_row1.addWidget(self._blip_btn)
        self._wd14_btn = QPushButton("Caption All (WD14)")
        self._wd14_btn.setToolTip("Auto-tag all uncaptioned images with WD14")
        self._wd14_btn.clicked.connect(lambda: self.caption_batch_requested.emit("wd14"))
        btn_row1.addWidget(self._wd14_btn)
        self._qwen_btn = QPushButton("Caption All (Qwen)")
        self._qwen_btn.setToolTip("Auto-caption all uncaptioned images with Qwen 2.5 VL 7B vision model")
        self._qwen_btn.clicked.connect(lambda: self.caption_batch_requested.emit("qwen"))
        btn_row1.addWidget(self._qwen_btn)
        tools_lay.addLayout(btn_row1)

        # Prepend / Append
        prepend_row = QHBoxLayout()
        prepend_row.addWidget(QLabel("Prepend:"))
        self._prepend_edit = QLineEdit()
        self._prepend_edit.setPlaceholderText("e.g. sks person, ")
        prepend_row.addWidget(self._prepend_edit, 1)
        self._prepend_btn = QPushButton("Apply")
        self._prepend_btn.clicked.connect(self._on_prepend)
        prepend_row.addWidget(self._prepend_btn)
        tools_lay.addLayout(prepend_row)

        append_row = QHBoxLayout()
        append_row.addWidget(QLabel("Append:"))
        self._append_edit = QLineEdit()
        self._append_edit.setPlaceholderText("e.g. , high quality")
        append_row.addWidget(self._append_edit, 1)
        self._append_btn = QPushButton("Apply")
        self._append_btn.clicked.connect(self._on_append)
        append_row.addWidget(self._append_btn)
        tools_lay.addLayout(append_row)

        # Find & Replace
        fr_row = QHBoxLayout()
        fr_row.addWidget(QLabel("Find:"))
        self._find_edit = QLineEdit()
        fr_row.addWidget(self._find_edit, 1)
        fr_row.addWidget(QLabel("Replace:"))
        self._replace_edit = QLineEdit()
        fr_row.addWidget(self._replace_edit, 1)
        self._fr_btn = QPushButton("Replace All")
        self._fr_btn.clicked.connect(self._on_find_replace)
        fr_row.addWidget(self._fr_btn)
        tools_lay.addLayout(fr_row)

        layout.addWidget(tools_group)

        # Dataset config
        config_group = QGroupBox("Dataset Config")
        cfg_lay = QVBoxLayout(config_group)
        cfg_lay.setContentsMargins(4, 4, 4, 4)

        cfg_row1 = QHBoxLayout()
        cfg_row1.addWidget(QLabel("Repeats:"))
        self.repeats = QSpinBox()
        self.repeats.setRange(1, 100)
        self.repeats.setValue(10)
        self.repeats.setFixedWidth(60)
        self.repeats.setToolTip("How many times to repeat each image per epoch")
        cfg_row1.addWidget(self.repeats)
        cfg_row1.addWidget(QLabel("Resolution:"))
        self.resolution = QComboBox()
        self.resolution.addItems(["512x512", "768x768", "1024x1024"])
        self.resolution.setCurrentIndex(0)
        self.resolution.setFixedWidth(100)
        cfg_row1.addWidget(self.resolution)
        cfg_row1.addStretch()
        cfg_lay.addLayout(cfg_row1)

        cfg_row2 = QHBoxLayout()
        self.enable_buckets = QCheckBox("Aspect Ratio Buckets")
        self.enable_buckets.setChecked(True)
        cfg_row2.addWidget(self.enable_buckets)
        self.flip_aug = QCheckBox("Flip Augmentation")
        cfg_row2.addWidget(self.flip_aug)
        cfg_row2.addWidget(QLabel("Keep Tokens:"))
        self.keep_tokens = QSpinBox()
        self.keep_tokens.setRange(0, 10)
        self.keep_tokens.setValue(0)
        self.keep_tokens.setFixedWidth(50)
        self.keep_tokens.setToolTip("Keep first N tokens from caption shuffle")
        cfg_row2.addWidget(self.keep_tokens)
        cfg_row2.addStretch()
        cfg_lay.addLayout(cfg_row2)

        layout.addWidget(config_group)

        # Progress bar (for batch captioning)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setFixedHeight(16)
        layout.addWidget(self._progress)

    # -- Properties --------------------------------------------------------

    @property
    def dataset_path(self) -> str:
        return self._dataset_path

    @property
    def image_count(self) -> int:
        return self._image_list.count()

    def get_image_paths(self) -> list[str]:
        """Return all image paths in the dataset."""
        paths = []
        for i in range(self._image_list.count()):
            item = self._image_list.item(i)
            p = item.data(Qt.ItemDataRole.UserRole)
            if p:
                paths.append(p)
        return paths

    def get_uncaptioned_paths(self) -> list[str]:
        """Return image paths that don't have a .txt caption file."""
        result = []
        for p in self.get_image_paths():
            txt = Path(p).with_suffix(".txt")
            if not txt.is_file():
                result.append(p)
        return result

    def set_dataset_path(self, path: str) -> None:
        """Set dataset folder programmatically (e.g. from 3D pipeline)."""
        if path and Path(path).is_dir():
            self._dataset_path = path
            self._folder_edit.setText(path)
            self._refresh_images()
            self.dataset_changed.emit(path)

    # -- Folder browsing ---------------------------------------------------

    @Slot()
    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if path:
            self.set_dataset_path(path)

    @Slot()
    def _on_prepare(self) -> None:
        """Import raw images: copy, rename sequentially, then auto-caption."""
        from PySide6.QtWidgets import QInputDialog, QMessageBox
        import shutil

        source = QFileDialog.getExistingDirectory(
            self, "Select Source Folder (raw images)"
        )
        if not source:
            return

        source_path = Path(source)
        images = sorted(
            f for f in source_path.iterdir()
            if f.suffix.lower() in _IMG_EXTS
        )
        if not images:
            QMessageBox.information(self, "No Images", "No images found in the selected folder.")
            return

        # Ask for a concept/subject name
        name, ok = QInputDialog.getText(
            self, "Dataset Name",
            f"Enter a name for this dataset ({len(images)} images).\n"
            "This becomes the output folder name:",
        )
        if not ok or not name.strip():
            return
        name = name.strip().replace(" ", "_").lower()

        # Create destination folder next to the source
        dest = source_path.parent / f"{name}_dataset"
        if dest.exists():
            reply = QMessageBox.question(
                self, "Folder Exists",
                f"{dest.name} already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            shutil.rmtree(dest)
        dest.mkdir(parents=True)

        # Copy and rename images sequentially
        for i, img in enumerate(images, 1):
            ext = img.suffix.lower()
            new_name = f"{name}_{i:04d}{ext}"
            shutil.copy2(img, dest / new_name)

        # Point the dataset panel to the new folder
        self.set_dataset_path(str(dest))

        # Trigger Qwen VL captioning on all images
        self.prepare_requested.emit(source, str(dest))

    def _refresh_images(self) -> None:
        self._image_list.clear()
        if not self._dataset_path:
            self._count_label.setText("")
            return
        folder = Path(self._dataset_path)
        if not folder.is_dir():
            self._count_label.setText("Folder not found")
            return
        images = sorted(
            f for f in folder.iterdir()
            if f.suffix.lower() in _IMG_EXTS
        )
        for img_path in images:
            item = QListWidgetItem(img_path.name)
            item.setData(Qt.ItemDataRole.UserRole, str(img_path))
            # Check if captioned
            has_cap = img_path.with_suffix(".txt").is_file()
            item.setForeground(
                Qt.GlobalColor.white if has_cap else Qt.GlobalColor.darkYellow
            )
            try:
                pm = QPixmap(str(img_path))
                if not pm.isNull():
                    item.setIcon(QIcon(pm.scaled(
                        _THUMB_SIZE, _THUMB_SIZE,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )))
            except Exception:
                pass
            self._image_list.addItem(item)
        captioned = sum(1 for f in images if f.with_suffix(".txt").is_file())
        self._count_label.setText(
            f"{len(images)} images ({captioned} captioned, "
            f"{len(images) - captioned} uncaptioned)"
        )

    # -- Image selection + caption editing ---------------------------------

    @Slot()
    def _on_image_selected(self, current, previous) -> None:
        if current is None:
            self._preview.clear()
            self._caption_edit.blockSignals(True)
            self._caption_edit.clear()
            self._caption_edit.blockSignals(False)
            self._current_image_path = ""
            return

        path = current.data(Qt.ItemDataRole.UserRole)
        self._current_image_path = path or ""

        # Preview
        if path:
            pm = QPixmap(path)
            if not pm.isNull():
                self._preview.setPixmap(pm.scaled(
                    self._preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))

        # Load caption
        self._caption_edit.blockSignals(True)
        txt_path = Path(path).with_suffix(".txt") if path else None
        if txt_path and txt_path.is_file():
            self._caption_edit.setPlainText(txt_path.read_text(encoding="utf-8"))
        else:
            self._caption_edit.clear()
        self._caption_edit.blockSignals(False)

    def _on_caption_changed(self) -> None:
        """Debounced save — wait 500ms after last edit."""
        self._save_timer.start()

    def _save_caption(self) -> None:
        """Write caption text to .txt file alongside the image."""
        if not self._current_image_path:
            return
        txt_path = Path(self._current_image_path).with_suffix(".txt")
        text = self._caption_edit.toPlainText()
        txt_path.write_text(text, encoding="utf-8")
        # Update item color to indicate captioned
        item = self._image_list.currentItem()
        if item:
            item.setForeground(Qt.GlobalColor.white if text.strip() else Qt.GlobalColor.darkYellow)

    # -- Batch caption tools -----------------------------------------------

    @Slot()
    def _on_prepend(self) -> None:
        prefix = self._prepend_edit.text()
        if not prefix:
            return
        self._batch_modify_captions(lambda cap: prefix + cap)

    @Slot()
    def _on_append(self) -> None:
        suffix = self._append_edit.text()
        if not suffix:
            return
        self._batch_modify_captions(lambda cap: cap + suffix)

    @Slot()
    def _on_find_replace(self) -> None:
        find = self._find_edit.text()
        replace = self._replace_edit.text()
        if not find:
            return
        self._batch_modify_captions(lambda cap: cap.replace(find, replace))

    def _batch_modify_captions(self, transform) -> None:
        """Apply a transform function to all .txt caption files in the dataset."""
        if not self._dataset_path:
            return
        folder = Path(self._dataset_path)
        modified = 0
        for txt in folder.glob("*.txt"):
            content = txt.read_text(encoding="utf-8")
            new_content = transform(content)
            if new_content != content:
                txt.write_text(new_content, encoding="utf-8")
                modified += 1
        logger.info("Modified %d caption files", modified)
        # Reload current caption if visible
        if self._current_image_path:
            txt_path = Path(self._current_image_path).with_suffix(".txt")
            if txt_path.is_file():
                self._caption_edit.blockSignals(True)
                self._caption_edit.setPlainText(txt_path.read_text(encoding="utf-8"))
                self._caption_edit.blockSignals(False)

    def set_batch_progress(self, current: int, total: int) -> None:
        """Update the progress bar during batch captioning."""
        self._progress.setVisible(total > 0)
        self._progress.setRange(0, total)
        self._progress.setValue(current)
        if current >= total:
            self._progress.setVisible(False)
            self._refresh_images()
