"""Shared library import dialogs for Face Library and Character Library."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
_THUMB = 80


class FaceLibraryDialog(QDialog):
    """Thumbnail grid dialog for selecting a face from the Face Library."""

    def __init__(self, face_dir: Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Face Library")
        self.setMinimumSize(400, 340)
        self.selected_path: str | None = None

        layout = QVBoxLayout(self)
        self._grid = QListWidget()
        self._grid.setViewMode(QListWidget.ViewMode.IconMode)
        self._grid.setIconSize(QSize(_THUMB, _THUMB))
        self._grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._grid.setSpacing(6)
        self._grid.setWrapping(True)
        self._grid.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._grid.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._grid, 1)

        pngs: list[Path] = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            pngs.extend(face_dir.glob(ext))
        for p in sorted(pngs):
            pixmap = QPixmap(str(p))
            if pixmap.isNull():
                continue
            thumb = pixmap.scaled(
                _THUMB, _THUMB,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            item = QListWidgetItem()
            item.setIcon(QIcon(thumb))
            item.setText(p.stem)
            item.setData(Qt.ItemDataRole.UserRole, str(p))
            item.setToolTip(str(p))
            self._grid.addItem(item)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        item = self._grid.currentItem()
        if item:
            self.selected_path = item.data(Qt.ItemDataRole.UserRole)
        self.accept()

    def _on_double_click(self, item: QListWidgetItem) -> None:
        self.selected_path = item.data(Qt.ItemDataRole.UserRole)
        self.accept()


class CharacterLibraryDialog(QDialog):
    """Two-panel dialog: left lists characters, right shows all images
    for the selected character.  User picks a specific image to import."""

    _THUMB_SM = 64
    _THUMB_LG = 100

    def __init__(
        self,
        characters_dir: Path,
        parent=None,
        *,
        prefer_face: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Character Library")
        self.setMinimumSize(640, 420)
        self.selected_path: str | None = None
        self._prefer_face = prefer_face

        outer = QVBoxLayout(self)

        panels = QHBoxLayout()
        panels.setSpacing(8)

        # -- Left: character list --
        left = QVBoxLayout()
        left.addWidget(QLabel("Characters"))
        self._char_list = QListWidget()
        self._char_list.setIconSize(QSize(self._THUMB_SM, self._THUMB_SM))
        self._char_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._char_list.currentItemChanged.connect(self._on_char_selected)
        left.addWidget(self._char_list, 1)
        panels.addLayout(left, 1)

        # -- Right: image grid for selected character --
        right = QVBoxLayout()
        self._images_label = QLabel("Select a character")
        right.addWidget(self._images_label)
        self._image_grid = QListWidget()
        self._image_grid.setViewMode(QListWidget.ViewMode.IconMode)
        self._image_grid.setIconSize(QSize(self._THUMB_LG, self._THUMB_LG))
        self._image_grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._image_grid.setSpacing(6)
        self._image_grid.setWrapping(True)
        self._image_grid.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._image_grid.itemDoubleClicked.connect(self._on_image_double_click)
        right.addWidget(self._image_grid, 1)
        panels.addLayout(right, 2)

        outer.addLayout(panels, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        # Populate characters
        self._char_data: dict[str, Path] = {}  # name → dir
        if characters_dir.is_dir():
            for d in sorted(characters_dir.iterdir()):
                if not d.is_dir():
                    continue
                # Accept any dir that has images, character.json, or a poses/ subdir
                has_content = (
                    (d / "character.json").is_file()
                    or any(f.suffix.lower() in _IMG_EXTS for f in d.iterdir() if f.is_file())
                    or (d / "poses").is_dir()
                )
                if not has_content:
                    continue

                # Character name from metadata
                name = d.name
                meta = d / "character.json"
                if meta.is_file():
                    try:
                        m = json.loads(meta.read_text())
                        name = m.get("name", d.name)
                    except Exception:
                        pass

                # Thumbnail: prefer face_ref or base, fall back to first image
                thumb_src = None
                for candidate in (d / "face_ref.png", d / "base.png"):
                    if candidate.is_file():
                        thumb_src = candidate
                        break
                if not thumb_src:
                    for f in sorted(d.iterdir()):
                        if f.is_file() and f.suffix.lower() in _IMG_EXTS:
                            thumb_src = f
                            break

                item = QListWidgetItem()
                item.setText(name)
                item.setData(Qt.ItemDataRole.UserRole, str(d))
                if thumb_src:
                    pm = QPixmap(str(thumb_src))
                    if not pm.isNull():
                        item.setIcon(QIcon(
                            pm.scaled(self._THUMB_SM, self._THUMB_SM,
                                      Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
                        ))
                self._char_list.addItem(item)
                self._char_data[name] = d

    def _on_char_selected(self, current: QListWidgetItem | None, _prev=None) -> None:
        self._image_grid.clear()
        if not current:
            self._images_label.setText("Select a character")
            return
        char_dir = Path(current.data(Qt.ItemDataRole.UserRole))
        name = current.text()
        self._images_label.setText(f"Images for {name}")

        # Collect all images from character dir + poses/
        images: list[Path] = []
        for f in sorted(char_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in _IMG_EXTS:
                images.append(f)
        poses_dir = char_dir / "poses"
        if poses_dir.is_dir():
            for f in sorted(poses_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in _IMG_EXTS:
                    images.append(f)

        for img in images:
            pm = QPixmap(str(img))
            if pm.isNull():
                continue
            thumb = pm.scaled(
                self._THUMB_LG, self._THUMB_LG,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            item = QListWidgetItem()
            item.setIcon(QIcon(thumb))
            item.setText(img.name)
            item.setData(Qt.ItemDataRole.UserRole, str(img))
            item.setToolTip(str(img))
            self._image_grid.addItem(item)

        # Auto-select preferred image
        if self._prefer_face:
            preferred = ["face_ref.png", "base.png"]
        else:
            preferred = ["base.png", "face_ref.png"]
        for pname in preferred:
            for i in range(self._image_grid.count()):
                if self._image_grid.item(i).text() == pname:
                    self._image_grid.setCurrentRow(i)
                    return
        if self._image_grid.count() > 0:
            self._image_grid.setCurrentRow(0)

    def _on_accept(self) -> None:
        item = self._image_grid.currentItem()
        if item:
            self.selected_path = item.data(Qt.ItemDataRole.UserRole)
        self.accept()

    def _on_image_double_click(self, item: QListWidgetItem) -> None:
        self.selected_path = item.data(Qt.ItemDataRole.UserRole)
        self.accept()
