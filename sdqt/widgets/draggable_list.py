"""QListWidget subclass that supports dragging items out as file URLs."""

from __future__ import annotations

from PySide6.QtCore import QMimeData, QUrl, Qt
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import QListWidget


class DraggableFileList(QListWidget):
    """List widget where items can be dragged out as file URLs.

    Each item's toolTip() must contain the absolute file path.
    Supports dragging to:
    - Other widgets in the app (FileDropWidget, AudioPlayerWidget)
    - Desktop / file manager for file copy
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)

    def startDrag(self, supportedActions) -> None:
        item = self.currentItem()
        if item is None:
            return

        path = item.toolTip()
        if not path:
            return

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(path)])

        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)
