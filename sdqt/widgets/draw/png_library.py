"""PNG Library -- sidebar for browsing, managing, and dragging PNG assets."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

_THUMB_SIZE = 64

_ICON_BTN_STYLE = (
    "QPushButton { border: 1px solid #555; padding: 2px 6px; font-size: 12px;"
    " color: #ccc; background: #333; border-radius: 3px; }"
    "QPushButton:hover { color: #fff; background: #555; }"
)


class PNGLibrary(QWidget):
    """Sidebar for browsing PNG assets with full CRUD operations."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._root_dir: str = ""
        self._subfolders: list[str] = []
        self._current_dir: Path | None = None  # currently displayed directory
        self.setMinimumWidth(180)
        self.setMaximumWidth(260)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # -- Header -----------------------------------------------------------
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(2)
        header.addWidget(QLabel("<b>PNG Library</b>"))
        header.addStretch()

        self._btn_new_folder = QPushButton("+\U0001f4c1")
        self._btn_new_folder.setToolTip("New subfolder")
        self._btn_new_folder.setStyleSheet(_ICON_BTN_STYLE)
        self._btn_new_folder.setFixedSize(36, 22)
        self._btn_new_folder.clicked.connect(self._on_new_folder)
        header.addWidget(self._btn_new_folder)

        self._btn_import = QPushButton("+\U0001f5bc")
        self._btn_import.setToolTip("Import PNG(s) into current folder")
        self._btn_import.setStyleSheet(_ICON_BTN_STYLE)
        self._btn_import.setFixedSize(36, 22)
        self._btn_import.clicked.connect(self._on_import)
        header.addWidget(self._btn_import)

        self._btn_refresh = QPushButton("\u21bb")
        self._btn_refresh.setToolTip("Refresh")
        self._btn_refresh.setStyleSheet(_ICON_BTN_STYLE)
        self._btn_refresh.setFixedSize(24, 22)
        self._btn_refresh.clicked.connect(self.refresh)
        header.addWidget(self._btn_refresh)

        layout.addLayout(header)

        # -- Folder selector --------------------------------------------------
        self._folder_combo = QComboBox()
        self._folder_combo.setMaxVisibleItems(15)
        self._folder_combo.currentIndexChanged.connect(self._on_folder_changed)
        layout.addWidget(self._folder_combo)

        # -- Thumbnail grid ---------------------------------------------------
        self._grid = QListWidget()
        self._grid.setViewMode(QListWidget.ViewMode.IconMode)
        self._grid.setIconSize(QSize(_THUMB_SIZE, _THUMB_SIZE))
        self._grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._grid.setSpacing(4)
        self._grid.setWrapping(True)
        self._grid.setDragEnabled(True)
        self._grid.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._grid.itemDoubleClicked.connect(self._on_double_click)
        self._grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._grid.customContextMenuRequested.connect(self._on_grid_context_menu)
        layout.addWidget(self._grid, 1)

        # -- Empty state label ------------------------------------------------
        self._empty_label = QLabel("No PNG library configured.\nSet path in Settings.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #888; font-style: italic; padding: 8px;")
        layout.addWidget(self._empty_label)

    # -- Public API -----------------------------------------------------------

    @property
    def root_dir(self) -> str:
        return self._root_dir

    def set_root_dir(self, path: str) -> None:
        self._root_dir = path
        self.refresh()

    def save_layer_image(self, image: QImage, suggested_name: str = "layer") -> None:
        """Save a QImage to the library. Prompts for name and subfolder."""
        if not self._root_dir:
            QMessageBox.warning(self, "PNG Library", "No library directory configured.\nSet path in Settings.")
            return
        root = Path(self._root_dir)
        root.mkdir(parents=True, exist_ok=True)

        name, ok = QInputDialog.getText(
            self, "Save to PNG Library", "File name (without .png):",
            text=suggested_name,
        )
        if not ok or not name.strip():
            return
        name = name.strip()

        # Use currently selected subfolder as save target
        target_dir = self._current_dir if self._current_dir else root
        target_dir.mkdir(parents=True, exist_ok=True)

        out_path = target_dir / f"{name}.png"
        # Avoid overwrite without asking
        if out_path.exists():
            ret = QMessageBox.question(
                self, "Overwrite?",
                f"{out_path.name} already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ret != QMessageBox.StandardButton.Yes:
                return

        image.save(str(out_path), "PNG")
        logger.info("Saved to PNG library: %s", out_path)
        self.refresh()

    def refresh(self) -> None:
        self._folder_combo.blockSignals(True)
        self._folder_combo.clear()

        root = Path(self._root_dir)
        if not self._root_dir:
            self._empty_label.setVisible(True)
            self._grid.setVisible(False)
            self._folder_combo.blockSignals(False)
            return
        root.mkdir(parents=True, exist_ok=True)

        self._empty_label.setVisible(False)
        self._grid.setVisible(True)

        # Collect subfolders
        self._subfolders = ["(all)"]
        for d in sorted(root.rglob("*")):
            if d.is_dir():
                rel = str(d.relative_to(root))
                self._subfolders.append(rel)
        self._folder_combo.addItems(self._subfolders)
        self._folder_combo.blockSignals(False)

        self._current_dir = root
        self._load_thumbnails(root)

    # -- Folder navigation ----------------------------------------------------

    def _on_folder_changed(self, index: int) -> None:
        root = Path(self._root_dir)
        if not root.is_dir():
            return
        if index <= 0:
            self._current_dir = root
            self._load_thumbnails(root)
        else:
            subfolder = self._subfolders[index]
            self._current_dir = root / subfolder
            self._load_thumbnails(self._current_dir)

    def _load_thumbnails(self, directory: Path) -> None:
        self._grid.clear()
        if not directory.is_dir():
            return
        files = sorted(f for f in directory.rglob("*.png") if f.is_file())
        for f in files[:500]:
            pixmap = QPixmap(str(f))
            if pixmap.isNull():
                continue
            thumb = pixmap.scaled(
                _THUMB_SIZE, _THUMB_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            item = QListWidgetItem()
            item.setIcon(QIcon(thumb))
            item.setText(f.stem)
            item.setData(Qt.ItemDataRole.UserRole, str(f))
            item.setToolTip(str(f))
            self._grid.addItem(item)

    # -- Double-click → add to canvas ----------------------------------------

    def _on_double_click(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.add_to_canvas_requested(path)

    def add_to_canvas_requested(self, path: str) -> None:
        """Hook for parent to connect: adds the PNG as a new layer."""
        pass  # overridden by connection in img_edit.py

    @property
    def selected_path(self) -> str | None:
        item = self._grid.currentItem()
        if item:
            return item.data(Qt.ItemDataRole.UserRole)
        return None

    # -- CRUD: Context menu on grid -------------------------------------------

    def _on_grid_context_menu(self, pos) -> None:
        item = self._grid.itemAt(pos)
        menu = QMenu(self)

        if item is not None:
            path = item.data(Qt.ItemDataRole.UserRole)
            act_add = menu.addAction("Add to Canvas")
            act_add.triggered.connect(lambda: self.add_to_canvas_requested(path))
            menu.addSeparator()
            act_rename = menu.addAction("Rename...")
            act_rename.triggered.connect(lambda: self._rename_file(path))
            act_move = menu.addAction("Move to Folder...")
            act_move.triggered.connect(lambda: self._move_file(path))
            act_del = menu.addAction("Delete")
            act_del.triggered.connect(lambda: self._delete_file(path))
        else:
            act_import = menu.addAction("Import PNG(s)...")
            act_import.triggered.connect(self._on_import)
            act_folder = menu.addAction("New Subfolder...")
            act_folder.triggered.connect(self._on_new_folder)

        menu.exec(self._grid.viewport().mapToGlobal(pos))

    def _rename_file(self, path: str) -> None:
        p = Path(path)
        if not p.is_file():
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename", "New name (without .png):", text=p.stem,
        )
        if not ok or not new_name.strip():
            return
        new_path = p.parent / f"{new_name.strip()}.png"
        if new_path.exists() and new_path != p:
            QMessageBox.warning(self, "Rename", f"{new_path.name} already exists.")
            return
        p.rename(new_path)
        logger.info("Renamed: %s → %s", p.name, new_path.name)
        self.refresh()

    def _move_file(self, path: str) -> None:
        p = Path(path)
        if not p.is_file():
            return
        root = Path(self._root_dir)
        if not root.is_dir():
            return
        # Build list of folders for selection
        folders = ["."]
        for d in sorted(root.rglob("*")):
            if d.is_dir():
                folders.append(str(d.relative_to(root)))
        choice, ok = QInputDialog.getItem(
            self, "Move to Folder", "Select destination folder:", folders, 0, False,
        )
        if not ok:
            return
        dest_dir = root if choice == "." else root / choice
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / p.name
        if dest.exists():
            QMessageBox.warning(self, "Move", f"{p.name} already exists in {choice}.")
            return
        shutil.move(str(p), str(dest))
        logger.info("Moved: %s → %s", p, dest)
        self.refresh()

    def _delete_file(self, path: str) -> None:
        p = Path(path)
        if not p.is_file():
            return
        ret = QMessageBox.question(
            self, "Delete",
            f"Delete {p.name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ret == QMessageBox.StandardButton.Yes:
            p.unlink()
            logger.info("Deleted: %s", p)
            self.refresh()

    # -- CRUD: New folder / Import --------------------------------------------

    def _on_new_folder(self) -> None:
        root = Path(self._root_dir)
        if not root.is_dir():
            QMessageBox.warning(self, "PNG Library", "No library directory configured.")
            return
        parent = self._current_dir if self._current_dir else root
        name, ok = QInputDialog.getText(self, "New Subfolder", "Folder name:")
        if not ok or not name.strip():
            return
        new_dir = parent / name.strip()
        new_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created folder: %s", new_dir)
        self.refresh()

    def _on_import(self) -> None:
        root = Path(self._root_dir)
        if not root.is_dir():
            QMessageBox.warning(self, "PNG Library", "No library directory configured.")
            return
        target = self._current_dir if self._current_dir else root
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import PNGs", "", "PNG Images (*.png)",
        )
        if not paths:
            return
        for src in paths:
            src_p = Path(src)
            dest = target / src_p.name
            if dest.exists():
                # Auto-suffix to avoid overwrite
                i = 1
                while dest.exists():
                    dest = target / f"{src_p.stem}_{i}.png"
                    i += 1
            shutil.copy2(str(src_p), str(dest))
            logger.info("Imported: %s → %s", src_p.name, dest)
        self.refresh()
