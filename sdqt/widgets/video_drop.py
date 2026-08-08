"""Compact video source widget with drag/drop, thumbnail preview, folder, and clear.

Shows a thumbnail extracted from the first frame of the video, matching
the look and feel of ImageDropWidget.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot, QUrl, QMimeData
from PySide6.QtGui import QDrag, QDragEnterEvent, QDropEvent, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from sdqt.utils.file_dialog import get_open_filename

logger = logging.getLogger(__name__)

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

_ICON_BTN_STYLE = (
    "QPushButton { border: none; padding: 2px 4px; font-size: 13px; color: #aaa; }"
    "QPushButton:hover { color: #fff; background: #444; border-radius: 3px; }"
)


class VideoDropWidget(QWidget):
    """Video drop target that shows a thumbnail from the video's first frame.

    Signals:
        file_loaded(str): emitted when a video is loaded (path)
        file_cleared(): emitted when cleared
    """

    file_loaded = Signal(str)
    file_cleared = Signal()

    def __init__(
        self,
        label: str = "Video",
        thumb_height: int = 80,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._current_path: str | None = None
        self._thumb_height = thumb_height
        self._drag_start_pos = None
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
        self._clear_btn.setToolTip("Clear video")
        self._clear_btn.setStyleSheet(_ICON_BTN_STYLE)
        self._clear_btn.clicked.connect(self.clear_file)
        self._clear_btn.setVisible(False)
        title_row.addWidget(self._clear_btn)

        layout.addLayout(title_row)

        # -- Thumbnail / drop zone -----------------------------------------
        self._thumb = QLabel()
        self._thumb.setAlignment(Qt.AlignCenter)
        self._thumb.setFixedHeight(thumb_height)
        self._thumb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._thumb.setStyleSheet(
            "QLabel { background: #1a1a1a; border: 2px dashed #444; border-radius: 6px; }"
        )
        self._thumb.setText("Drop video here")
        layout.addWidget(self._thumb)

        # Browse overlay (visible when empty)
        self._browse_overlay = QPushButton("\u2191", self._thumb)  # ↑
        self._browse_overlay.setToolTip("Browse for video file")
        self._browse_overlay.setFixedSize(28, 28)
        self._browse_overlay.setStyleSheet(
            "QPushButton { background: rgba(80,80,80,180); color: #ddd; border: 1px solid #666; "
            "border-radius: 6px; font-size: 16px; font-weight: bold; }"
            "QPushButton:hover { background: rgba(100,140,200,220); color: #fff; }"
        )
        self._browse_overlay.clicked.connect(self._browse_file)
        self._browse_overlay.setCursor(Qt.PointingHandCursor)

    # -- Public API --------------------------------------------------------

    def load_file(self, path: str | None) -> None:
        """Load a video file and show its first frame as a thumbnail."""
        if path and Path(path).is_file():
            self._current_path = path
            pixmap = self._extract_thumbnail(path)
            if pixmap and not pixmap.isNull():
                self._thumb.setPixmap(
                    pixmap.scaledToHeight(self._thumb_height - 4, Qt.SmoothTransformation)
                )
            else:
                self._thumb.setText(Path(path).name)
            self._thumb.setStyleSheet(
                "QLabel { background: #1a1a1a; border: 1px solid #555; border-radius: 4px; }"
            )
            self._folder_btn.setVisible(True)
            self._clear_btn.setVisible(True)
            self._browse_overlay.setVisible(False)
            self.file_loaded.emit(path)
        else:
            self.clear_file()

    def clear_file(self) -> None:
        """Clear the loaded video."""
        self._current_path = None
        self._thumb.clear()
        self._thumb.setText("Drop video here")
        self._thumb.setStyleSheet(
            "QLabel { background: #1a1a1a; border: 2px dashed #444; border-radius: 6px; }"
        )
        self._folder_btn.setVisible(False)
        self._clear_btn.setVisible(False)
        self._browse_overlay.setVisible(True)
        self.file_cleared.emit()

    @property
    def file_path(self) -> str | None:
        return self._current_path

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        tr = self._thumb.geometry()
        self._browse_overlay.move(tr.right() - self._browse_overlay.width() - 6, tr.top() + 6)

    # -- Thumbnail extraction ----------------------------------------------

    @staticmethod
    def _extract_thumbnail(video_path: str) -> QPixmap | None:
        """Extract the first frame from a video as a QPixmap."""
        import tempfile

        try:
            from supremediffusion.utils.video import extract_single_frame
            img = extract_single_frame(str(video_path), 0)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            img.save(tmp.name, "PNG")
            pix = QPixmap(tmp.name)
            Path(tmp.name).unlink(missing_ok=True)
            return pix if not pix.isNull() else None
        except Exception:
            logger.debug("Failed to extract video thumbnail", exc_info=True)
        return None

    # -- Drag/drop in ------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in _VIDEO_EXTS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in _VIDEO_EXTS:
                self.load_file(path)
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

    # -- Browse ------------------------------------------------------------

    @Slot()
    def _browse_file(self) -> None:
        ext_list = " ".join(f"*{e}" for e in sorted(_VIDEO_EXTS))
        path, _ = get_open_filename(
            self, "Select Video", "", f"Videos ({ext_list})",
        )
        if path:
            self.load_file(path)

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
