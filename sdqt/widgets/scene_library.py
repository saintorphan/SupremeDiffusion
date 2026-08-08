"""Scene library widget — thumbnail browser for scenes and scene instances.

Provides two panels:
  1. Scene list: all scenes in the project's ``scenes/`` dir
  2. Instance list: all saved instances within the selected scene

Supports create, rename, delete, and thumbnail display.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sdqt.models.scene3d import Scene3D, SceneInstance, load_instance, save_instance

_THUMB = 96


class SceneLibraryWidget(QWidget):
    """Two-panel scene/instance browser."""

    scene_selected = Signal(str)       # scene directory path
    instance_selected = Signal(str)    # instance JSON path
    scene_created = Signal(str)        # new scene dir path
    render_output_selected = Signal(str)  # path to rendered image

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scenes_root: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Vertical splitter for all three panels — drag handles to resize
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(True)
        splitter.setHandleWidth(5)
        splitter.setStyleSheet(
            "QSplitter::handle { background: #3a3a3a; }"
            "QSplitter::handle:hover { background: #888; }"
        )

        # -- Scene panel ---------------------------------------------------
        scene_pane = QWidget()
        scene_pl = QVBoxLayout(scene_pane)
        scene_pl.setContentsMargins(0, 0, 0, 0)
        scene_pl.setSpacing(2)
        scene_header = QHBoxLayout()
        scene_header.setSpacing(4)
        scene_header.addWidget(QLabel("Scenes"))
        scene_header.addStretch()
        self._btn_new_scene = QPushButton("+")
        self._btn_new_scene.setFixedSize(24, 24)
        self._btn_new_scene.setToolTip("New scene")
        self._btn_new_scene.clicked.connect(self._on_new_scene)
        scene_header.addWidget(self._btn_new_scene)
        scene_pl.addLayout(scene_header)

        self._scene_list = QListWidget()
        self._scene_list.setViewMode(QListWidget.ViewMode.IconMode)
        self._scene_list.setIconSize(QSize(_THUMB, _THUMB))
        self._scene_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._scene_list.setSpacing(4)
        self._scene_list.setWrapping(True)
        self._scene_list.currentItemChanged.connect(self._on_scene_clicked)
        self._scene_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._scene_list.customContextMenuRequested.connect(self._scene_context_menu)
        scene_pl.addWidget(self._scene_list, 1)
        splitter.addWidget(scene_pane)

        # -- Instance panel ------------------------------------------------
        inst_pane = QWidget()
        inst_pl = QVBoxLayout(inst_pane)
        inst_pl.setContentsMargins(0, 0, 0, 0)
        inst_pl.setSpacing(2)
        inst_header = QHBoxLayout()
        inst_header.setSpacing(4)
        inst_header.addWidget(QLabel("Instances"))
        inst_header.addStretch()
        self._btn_new_inst = QPushButton("+")
        self._btn_new_inst.setFixedSize(24, 24)
        self._btn_new_inst.setToolTip("Save current as new instance")
        inst_header.addWidget(self._btn_new_inst)
        inst_pl.addLayout(inst_header)

        self._inst_list = QListWidget()
        self._inst_list.setViewMode(QListWidget.ViewMode.IconMode)
        self._inst_list.setIconSize(QSize(_THUMB, _THUMB))
        self._inst_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._inst_list.setSpacing(4)
        self._inst_list.setWrapping(True)
        self._inst_list.currentItemChanged.connect(self._on_instance_clicked)
        self._inst_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._inst_list.customContextMenuRequested.connect(self._inst_context_menu)
        inst_pl.addWidget(self._inst_list, 1)
        splitter.addWidget(inst_pane)

        # -- Outputs panel -------------------------------------------------
        output_pane = QWidget()
        output_pl = QVBoxLayout(output_pane)
        output_pl.setContentsMargins(0, 0, 0, 0)
        output_pl.setSpacing(2)
        output_pl.addWidget(QLabel("Outputs"))
        self._output_list = QListWidget()
        self._output_list.setViewMode(QListWidget.ViewMode.IconMode)
        self._output_list.setIconSize(QSize(_THUMB, _THUMB))
        self._output_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._output_list.setSpacing(4)
        self._output_list.setWrapping(True)
        self._output_list.itemDoubleClicked.connect(self._on_output_double_clicked)
        output_pl.addWidget(self._output_list, 1)
        splitter.addWidget(output_pane)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)
        layout.addWidget(splitter, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_scenes_root(self, root: Path) -> None:
        """Point to a project's ``scenes/`` directory and refresh."""
        self._scenes_root = root
        root.mkdir(parents=True, exist_ok=True)
        self.refresh_scenes()

    def refresh_scenes(self) -> None:
        self._scene_list.clear()
        if not self._scenes_root or not self._scenes_root.is_dir():
            return
        for d in sorted(self._scenes_root.iterdir()):
            if not d.is_dir():
                continue
            scene_json = d / "scene.json"
            if not scene_json.is_file():
                continue
            name = d.name
            try:
                meta = json.loads(scene_json.read_text())
                name = meta.get("name", d.name)
            except Exception:
                pass

            item = QListWidgetItem()
            item.setText(name)
            item.setData(Qt.ItemDataRole.UserRole, str(d))
            item.setToolTip(str(d))

            # Thumbnail: look for default instance thumb or scene thumb
            thumb_path = self._find_scene_thumb(d)
            if thumb_path:
                pm = QPixmap(str(thumb_path))
                if not pm.isNull():
                    item.setIcon(QIcon(pm.scaled(
                        _THUMB, _THUMB,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )))
            self._scene_list.addItem(item)

    def refresh_instances(self, scene_dir: Path) -> None:
        self._inst_list.clear()
        inst_dir = scene_dir / "instances"
        if not inst_dir.is_dir():
            return
        for f in sorted(inst_dir.glob("*.json")):
            try:
                inst = load_instance(f)
            except Exception:
                continue
            item = QListWidgetItem()
            item.setText(inst.name)
            item.setData(Qt.ItemDataRole.UserRole, str(f))
            item.setToolTip(str(f))

            thumb = scene_dir / "thumbnails" / f"{f.stem}.png"
            if thumb.is_file():
                pm = QPixmap(str(thumb))
                if not pm.isNull():
                    item.setIcon(QIcon(pm.scaled(
                        _THUMB, _THUMB,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )))
            self._inst_list.addItem(item)

    def refresh_outputs(self, scene_dir: Path) -> None:
        self._output_list.clear()
        out_dir = scene_dir / "outputs"
        if not out_dir.is_dir():
            return
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        for f in sorted(out_dir.iterdir(), reverse=True):
            if f.suffix.lower() not in exts:
                continue
            item = QListWidgetItem()
            item.setText(f.stem)
            item.setData(Qt.ItemDataRole.UserRole, str(f))
            pm = QPixmap(str(f))
            if not pm.isNull():
                item.setIcon(QIcon(pm.scaled(
                    _THUMB, _THUMB,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )))
            self._output_list.addItem(item)

    def current_scene_dir(self) -> Path | None:
        item = self._scene_list.currentItem()
        if item:
            return Path(item.data(Qt.ItemDataRole.UserRole))
        return None

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_scene_clicked(self, current: QListWidgetItem | None, _prev) -> None:
        if not current:
            return
        scene_dir = Path(current.data(Qt.ItemDataRole.UserRole))
        self.refresh_instances(scene_dir)
        self.refresh_outputs(scene_dir)
        self.scene_selected.emit(str(scene_dir))

    def _on_instance_clicked(self, current: QListWidgetItem | None, _prev) -> None:
        if not current:
            return
        self.instance_selected.emit(current.data(Qt.ItemDataRole.UserRole))

    def _on_output_double_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.render_output_selected.emit(path)

    def _on_new_scene(self) -> None:
        if not self._scenes_root:
            return
        name, ok = QInputDialog.getText(self, "New Scene", "Scene name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)
        scene_dir = self._scenes_root / safe_name
        if scene_dir.exists():
            QMessageBox.warning(self, "Exists", f"Scene '{safe_name}' already exists.")
            return

        scene_dir.mkdir(parents=True)
        (scene_dir / "instances").mkdir()
        (scene_dir / "thumbnails").mkdir()
        (scene_dir / "outputs").mkdir()

        scene = Scene3D(name=name)
        scene.save_meta(scene_dir / "scene.json")

        # Create default instance
        default_inst = SceneInstance(name="default")
        save_instance(default_inst, scene_dir / "instances" / "default.json")

        self.refresh_scenes()
        self.scene_created.emit(str(scene_dir))

    # -- Context menus -----------------------------------------------------

    def _scene_context_menu(self, pos) -> None:
        item = self._scene_list.itemAt(pos)
        if not item:
            return
        scene_dir = Path(item.data(Qt.ItemDataRole.UserRole))
        menu = QMenu(self)
        rename_action = menu.addAction("Rename")
        delete_action = menu.addAction("Delete")
        action = menu.exec(self._scene_list.mapToGlobal(pos))
        if action == rename_action:
            self._rename_scene(scene_dir)
        elif action == delete_action:
            self._delete_scene(scene_dir)

    def _inst_context_menu(self, pos) -> None:
        item = self._inst_list.itemAt(pos)
        if not item:
            return
        inst_path = Path(item.data(Qt.ItemDataRole.UserRole))
        menu = QMenu(self)
        rename_action = menu.addAction("Rename")
        delete_action = menu.addAction("Delete")
        action = menu.exec(self._inst_list.mapToGlobal(pos))
        if action == rename_action:
            self._rename_instance(inst_path)
        elif action == delete_action:
            self._delete_instance(inst_path)

    def _rename_scene(self, scene_dir: Path) -> None:
        old_name = scene_dir.name
        new_name, ok = QInputDialog.getText(self, "Rename Scene", "New name:", text=old_name)
        if not ok or not new_name.strip() or new_name.strip() == old_name:
            return
        safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in new_name.strip())
        new_dir = scene_dir.parent / safe
        if new_dir.exists():
            QMessageBox.warning(self, "Exists", f"'{safe}' already exists.")
            return
        scene_dir.rename(new_dir)
        # Update scene.json name
        meta_path = new_dir / "scene.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
                meta["name"] = new_name.strip()
                meta_path.write_text(json.dumps(meta, indent=2))
            except Exception:
                pass
        self.refresh_scenes()

    def _delete_scene(self, scene_dir: Path) -> None:
        reply = QMessageBox.question(
            self, "Delete Scene",
            f"Delete scene '{scene_dir.name}' and all its instances/outputs?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            shutil.rmtree(scene_dir)
            self.refresh_scenes()

    def _rename_instance(self, inst_path: Path) -> None:
        old_name = inst_path.stem
        new_name, ok = QInputDialog.getText(self, "Rename Instance", "New name:", text=old_name)
        if not ok or not new_name.strip() or new_name.strip() == old_name:
            return
        try:
            inst = load_instance(inst_path)
            inst.name = new_name.strip()
            safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in new_name.strip())
            new_path = inst_path.parent / f"{safe}.json"
            save_instance(inst, new_path)
            if new_path != inst_path:
                inst_path.unlink()
        except Exception:
            pass
        scene_dir = inst_path.parent.parent
        self.refresh_instances(scene_dir)

    def _delete_instance(self, inst_path: Path) -> None:
        reply = QMessageBox.question(
            self, "Delete Instance",
            f"Delete instance '{inst_path.stem}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            inst_path.unlink(missing_ok=True)
            # Also remove thumbnail
            thumb = inst_path.parent.parent / "thumbnails" / f"{inst_path.stem}.png"
            thumb.unlink(missing_ok=True)
            scene_dir = inst_path.parent.parent
            self.refresh_instances(scene_dir)

    # -- Helpers -----------------------------------------------------------

    @staticmethod
    def _find_scene_thumb(scene_dir: Path) -> Path | None:
        thumb_dir = scene_dir / "thumbnails"
        if thumb_dir.is_dir():
            default = thumb_dir / "default.png"
            if default.is_file():
                return default
            for f in thumb_dir.iterdir():
                if f.suffix.lower() == ".png":
                    return f
        return None
