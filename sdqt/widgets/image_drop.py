"""Compact image source widget with drag/drop, thumbnail, folder, and clear."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from sdqt.widgets.send_targets import DISABLED_TARGETS

from PySide6.QtCore import Qt, Signal, Slot, QUrl, QMimeData
from PySide6.QtGui import QDrag, QDragEnterEvent, QDropEvent, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from sdqt.utils.file_dialog import get_open_filename

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

_ICON_BTN_STYLE = (
    "QPushButton { border: none; padding: 2px 4px; font-size: 13px; color: #aaa; }"
    "QPushButton:hover { color: #fff; background: #444; border-radius: 3px; }"
)


class ImageDropWidget(QWidget):
    """Compact image source with thumbnail preview, drag/drop, folder icon, and clear.

    Replaces the [QLineEdit][Browse] pattern with a visual drop target.

    Signals:
        image_loaded(str): emitted when an image is loaded (path)
        image_cleared(): emitted when the image is cleared
    """

    image_loaded = Signal(str)
    image_cleared = Signal()
    send_requested = Signal(str)
    final_frame_requested = Signal(str)
    guide_video_requested = Signal(str)

    def __init__(
        self,
        label: str = "Source Image",
        thumb_height: int = 120,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._current_path: str | None = None
        self._thumb_height = thumb_height
        self._drag_start_pos = None
        self._send_targets: list[tuple[str, str]] = []
        self._final_frame_targets: list[tuple[str, str]] = []
        self._guide_video_targets: list[tuple[str, str]] = []
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # -- Title bar: [label] [spacer] [folder] [clear] -----------------
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(2)
        self._title_label = QLabel(f"<b>{label}</b>")
        title_row.addWidget(self._title_label)
        title_row.addStretch()

        self._folder_btn = QPushButton("\U0001F4C2")
        self._folder_btn.setFixedSize(24, 20)
        self._folder_btn.setToolTip("Open containing folder")
        self._folder_btn.setStyleSheet(_ICON_BTN_STYLE)
        self._folder_btn.clicked.connect(self._open_folder)
        self._folder_btn.setVisible(False)
        title_row.addWidget(self._folder_btn)

        self._clear_btn = QPushButton("\u2715")
        self._clear_btn.setFixedSize(24, 20)
        self._clear_btn.setToolTip("Clear image")
        self._clear_btn.setStyleSheet(_ICON_BTN_STYLE)
        self._clear_btn.clicked.connect(self.clear_image)
        self._clear_btn.setVisible(False)
        title_row.addWidget(self._clear_btn)

        layout.addLayout(title_row)

        # -- Thumbnail / drop zone -----------------------------------------
        self._thumb = QLabel()
        self._thumb.setAlignment(Qt.AlignCenter)
        self._thumb.setMinimumHeight(thumb_height)
        self._thumb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._thumb.setStyleSheet(
            "QLabel { background: #1a1a1a; border: 2px dashed #444; border-radius: 6px; }"
        )
        self._thumb.setText("Drop image here")
        layout.addWidget(self._thumb, 1)

        # Browse overlay (visible when empty)
        self._browse_overlay = QPushButton("\u2191", self._thumb)  # ↑
        self._browse_overlay.setToolTip("Browse for image file")
        self._browse_overlay.setFixedSize(28, 28)
        self._browse_overlay.setStyleSheet(
            "QPushButton { background: rgba(80,80,80,180); color: #ddd; border: 1px solid #666; "
            "border-radius: 6px; font-size: 16px; font-weight: bold; }"
            "QPushButton:hover { background: rgba(100,140,200,220); color: #fff; }"
        )
        self._browse_overlay.clicked.connect(self._browse_image)
        self._browse_overlay.setCursor(Qt.PointingHandCursor)

    # -- Public API --------------------------------------------------------

    def load_image(self, path: str | None) -> None:
        """Load an image (or clear if None/empty)."""
        if path and Path(path).is_file():
            self._current_path = path
            self._source_pixmap = QPixmap(path)
            if not self._source_pixmap.isNull():
                self._update_thumb()
                self._thumb.setStyleSheet(
                    "QLabel { background: #1a1a1a; border: 1px solid #555; border-radius: 4px; }"
                )
            self._folder_btn.setVisible(True)
            self._clear_btn.setVisible(True)
            self._browse_overlay.setVisible(False)
            self.image_loaded.emit(path)
        else:
            self.clear_image()

    def _update_thumb(self) -> None:
        """Scale the source pixmap to fit the current thumb label size."""
        if hasattr(self, "_source_pixmap") and self._source_pixmap and not self._source_pixmap.isNull():
            size = self._thumb.size()
            scaled = self._source_pixmap.scaled(
                size.width() - 4, size.height() - 4,
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self._thumb.setPixmap(scaled)

    def clear_image(self) -> None:
        """Clear the loaded image."""
        self._current_path = None
        self._source_pixmap = None
        self._thumb.clear()
        self._thumb.setText("Drop image here")
        self._thumb.setStyleSheet(
            "QLabel { background: #1a1a1a; border: 2px dashed #444; border-radius: 6px; }"
        )
        self._folder_btn.setVisible(False)
        self._clear_btn.setVisible(False)
        self._browse_overlay.setVisible(True)
        self.image_cleared.emit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Position browse overlay in upper-right of thumb
        tr = self._thumb.geometry()
        self._browse_overlay.move(tr.right() - self._browse_overlay.width() - 6, tr.top() + 6)
        # Refit thumbnail to new size
        self._update_thumb()

    @property
    def image_path(self) -> str | None:
        return self._current_path

    # -- Drag/drop in ------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in _IMAGE_EXTS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in _IMAGE_EXTS:
                self.load_image(path)
                event.acceptProposedAction()
                return

    # -- Drag out ----------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._current_path:
            self._drag_start_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            self._drag_start_pos is not None
            and self._current_path
            and (event.position().toPoint() - self._drag_start_pos).manhattanLength() > 20
        ):
            drag = QDrag(self)
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(self._current_path)])
            drag.setMimeData(mime)
            drag.exec(Qt.CopyAction)
            self._drag_start_pos = None
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_start_pos = None
        super().mouseReleaseEvent(event)

    # -- Context menu targets -----------------------------------------------

    def set_send_targets(self, targets: list[tuple[str, str]]) -> None:
        self._send_targets = targets

    def set_final_frame_targets(self, targets: list[tuple[str, str]]) -> None:
        self._final_frame_targets = targets

    def set_guide_video_targets(self, targets: list[tuple[str, str]]) -> None:
        self._guide_video_targets = targets

    def contextMenuEvent(self, event) -> None:
        if not self._current_path:
            return

        from sdqt.widgets.send_targets import build_target_menu

        menu = QMenu(self)
        has_items = False

        if self._send_targets:
            build_target_menu(menu, "Send to", self._send_targets,
                              self.send_requested.emit)
            has_items = True

        if self._final_frame_targets:
            build_target_menu(menu, "Final Frame", self._final_frame_targets,
                              self.final_frame_requested.emit)
            has_items = True

        if self._guide_video_targets:
            build_target_menu(menu, "Guide Video", self._guide_video_targets,
                              self.guide_video_requested.emit)
            has_items = True

        if has_items:
            menu.addSeparator()

        # Always-available actions
        save_as = menu.addAction("Save Image As...")
        save_as.triggered.connect(self._save_as)

        open_folder = menu.addAction("Open containing folder")
        open_folder.triggered.connect(self._open_folder)

        menu.exec(event.globalPos())

    # -- Save As -----------------------------------------------------------

    @Slot()
    def _save_as(self) -> None:
        if not self._current_path or not Path(self._current_path).is_file():
            return
        from sdqt.utils.file_dialog import get_save_filename
        default_name = Path(self._current_path).name
        dest, _ = get_save_filename(
            self, "Save Image As", default_name, "Images (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if dest:
            import shutil
            shutil.copy2(self._current_path, dest)

    # -- Browse ------------------------------------------------------------

    @Slot()
    def _browse_image(self) -> None:
        path, _ = get_open_filename(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff)",
        )
        if path:
            self.load_image(path)

    # -- Folder ------------------------------------------------------------

    @Slot()
    def _open_folder(self) -> None:
        if self._current_path and Path(self._current_path).is_file():
            folder = str(Path(self._current_path).parent)
            try:
                if os.name == "nt":
                    subprocess.Popen(["explorer", "/select,", self._current_path])
                elif os.name == "posix":
                    subprocess.Popen(["xdg-open", folder])
                else:
                    subprocess.Popen(["open", folder])
            except Exception:
                logger.warning("Failed to open folder", exc_info=True)
