"""Compact file input widget with drag/drop, folder icon, and clear button."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot, QUrl, QMimeData
from PySide6.QtGui import QDrag, QDragEnterEvent, QDropEvent, QMouseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from sdqt.utils.file_dialog import get_open_filename

logger = logging.getLogger(__name__)

_ICON_BTN_STYLE = (
    "QPushButton { border: none; padding: 2px 4px; font-size: 13px; color: #aaa; }"
    "QPushButton:hover { color: #fff; background: #444; border-radius: 3px; }"
)


class FileDropWidget(QWidget):
    """Compact file path display with drag/drop, folder icon, and clear.

    Replaces [QLineEdit][Browse] for file paths like guidance videos.

    Signals:
        file_loaded(str): emitted when a file is loaded
        file_cleared(): emitted when cleared
    """

    file_loaded = Signal(str)
    file_cleared = Signal()

    def __init__(
        self,
        label: str = "File",
        extensions: set[str] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._current_path: str | None = None
        self._extensions = extensions or {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        self._drag_start_pos = None
        self.setAcceptDrops(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._label = QLabel(f"<b>{label}:</b>")
        layout.addWidget(self._label)

        self._path_label = QLabel()
        self._path_label.setStyleSheet(
            "QLabel { color: #888; padding: 2px 6px; background: #1a1a1a; "
            "border: 1px dashed #444; border-radius: 3px; }"
        )
        self._path_label.setText("Drop file here...")
        self._path_label.setMinimumWidth(80)
        layout.addWidget(self._path_label, 1)

        self._browse_btn = QPushButton("\u2191")  # ↑
        self._browse_btn.setFixedSize(24, 20)
        self._browse_btn.setToolTip("Browse for file")
        self._browse_btn.setStyleSheet(_ICON_BTN_STYLE)
        self._browse_btn.clicked.connect(self._browse_file)
        self._browse_btn.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self._browse_btn)

        self._folder_btn = QPushButton("\U0001F4C2")
        self._folder_btn.setFixedSize(24, 20)
        self._folder_btn.setToolTip("Open containing folder")
        self._folder_btn.setStyleSheet(_ICON_BTN_STYLE)
        self._folder_btn.clicked.connect(self._open_folder)
        self._folder_btn.setVisible(False)
        layout.addWidget(self._folder_btn)

        self._clear_btn = QPushButton("\u2715")
        self._clear_btn.setFixedSize(24, 20)
        self._clear_btn.setToolTip("Clear")
        self._clear_btn.setStyleSheet(_ICON_BTN_STYLE)
        self._clear_btn.clicked.connect(self.clear_file)
        self._clear_btn.setVisible(False)
        layout.addWidget(self._clear_btn)

    # -- Public API --------------------------------------------------------

    def load_file(self, path: str | None) -> None:
        if path and Path(path).is_file():
            self._current_path = path
            name = Path(path).name
            self._path_label.setText(name)
            self._path_label.setToolTip(path)
            self._path_label.setStyleSheet(
                "QLabel { color: #ccc; padding: 2px 6px; background: #1a1a1a; "
                "border: 1px solid #555; border-radius: 3px; }"
            )
            self._browse_btn.setVisible(False)
            self._folder_btn.setVisible(True)
            self._clear_btn.setVisible(True)
            self.file_loaded.emit(path)
        else:
            self.clear_file()

    def clear_file(self) -> None:
        self._current_path = None
        self._path_label.setText("Drop file here...")
        self._path_label.setToolTip("")
        self._path_label.setStyleSheet(
            "QLabel { color: #888; padding: 2px 6px; background: #1a1a1a; "
            "border: 1px dashed #444; border-radius: 3px; }"
        )
        self._browse_btn.setVisible(True)
        self._folder_btn.setVisible(False)
        self._clear_btn.setVisible(False)
        self.file_cleared.emit()

    @property
    def file_path(self) -> str | None:
        return self._current_path

    # -- Drag/drop in ------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in self._extensions:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in self._extensions:
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
        ext_list = " ".join(f"*{e}" for e in sorted(self._extensions))
        path, _ = get_open_filename(
            self, "Select File", "", f"Files ({ext_list})",
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
