"""File dialog helpers that use Qt's built-in dialog with image preview."""

from __future__ import annotations

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QListView,
    QTreeView,
    QVBoxLayout,
    QWidget,
)


def _configure_dialog(dlg: QFileDialog) -> None:
    """Set up a QFileDialog with image preview panel."""
    dlg.setOption(QFileDialog.DontUseNativeDialog, True)
    dlg.setViewMode(QFileDialog.List)

    # Set icon size on internal views
    for view in dlg.findChildren(QListView):
        view.setIconSize(QSize(64, 64))
        view.setGridSize(QSize(100, 80))
        view.setViewMode(QListView.IconMode)
        view.setResizeMode(QListView.Adjust)
        view.setWrapping(True)
    for view in dlg.findChildren(QTreeView):
        view.setIconSize(QSize(64, 64))

    # Add image preview panel on the right side
    preview = QLabel()
    preview.setAlignment(Qt.AlignCenter)
    preview.setFixedWidth(250)
    preview.setMinimumHeight(250)
    preview.setStyleSheet(
        "QLabel { background: #1a1a1a; border: 1px solid #333;"
        " border-radius: 4px; padding: 4px; }"
    )
    preview.setText("Preview")

    # Inject into the dialog's layout
    layout = dlg.layout()
    if layout is not None:
        # QFileDialog uses a QGridLayout; add preview to the right
        from PySide6.QtWidgets import QGridLayout
        if isinstance(layout, QGridLayout):
            layout.addWidget(preview, 0, layout.columnCount(), layout.rowCount(), 1)
        else:
            layout.addWidget(preview)

    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}

    def _on_selection_changed(path: str) -> None:
        from pathlib import Path
        p = Path(path)
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            pm = QPixmap(str(p))
            if not pm.isNull():
                preview.setPixmap(pm.scaled(
                    preview.width() - 8, preview.height() - 8,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                ))
                preview.setText("")
                return
        preview.clear()
        preview.setText("Preview")

    dlg.currentChanged.connect(_on_selection_changed)
    dlg.resize(1100, 600)


def get_open_filename(
    parent: QWidget | None,
    title: str = "Open File",
    directory: str = "",
    filter: str = "",
) -> tuple[str, str]:
    """Like QFileDialog.getOpenFileName but with large icon view."""
    dlg = QFileDialog(parent, title, directory, filter)
    dlg.setFileMode(QFileDialog.ExistingFile)
    dlg.setAcceptMode(QFileDialog.AcceptOpen)
    _configure_dialog(dlg)
    if dlg.exec():
        files = dlg.selectedFiles()
        return (files[0] if files else "", dlg.selectedNameFilter())
    return ("", "")


def get_open_filenames(
    parent: QWidget | None,
    title: str = "Open Files",
    directory: str = "",
    filter: str = "",
) -> tuple[list[str], str]:
    """Like QFileDialog.getOpenFileNames but with large icon view."""
    dlg = QFileDialog(parent, title, directory, filter)
    dlg.setFileMode(QFileDialog.ExistingFiles)
    dlg.setAcceptMode(QFileDialog.AcceptOpen)
    _configure_dialog(dlg)
    if dlg.exec():
        return (dlg.selectedFiles(), dlg.selectedNameFilter())
    return ([], "")


def get_save_filename(
    parent: QWidget | None,
    title: str = "Save File",
    directory: str = "",
    filter: str = "",
) -> tuple[str, str]:
    """Like QFileDialog.getSaveFileName but with large icon view."""
    dlg = QFileDialog(parent, title, directory, filter)
    dlg.setFileMode(QFileDialog.AnyFile)
    dlg.setAcceptMode(QFileDialog.AcceptSave)
    _configure_dialog(dlg)
    if dlg.exec():
        files = dlg.selectedFiles()
        return (files[0] if files else "", dlg.selectedNameFilter())
    return ("", "")


def get_existing_directory(
    parent: QWidget | None,
    title: str = "Select Directory",
    directory: str = "",
) -> str:
    """Like QFileDialog.getExistingDirectory but with large icon view."""
    dlg = QFileDialog(parent, title, directory)
    dlg.setFileMode(QFileDialog.Directory)
    dlg.setAcceptMode(QFileDialog.AcceptOpen)
    dlg.setOption(QFileDialog.ShowDirsOnly, True)
    _configure_dialog(dlg)
    if dlg.exec():
        files = dlg.selectedFiles()
        return files[0] if files else ""
    return ""
