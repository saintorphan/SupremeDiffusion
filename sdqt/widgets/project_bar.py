"""Project selector bar -- prominent header banner widget."""

from __future__ import annotations

import logging
import subprocess
import sys

from PySide6.QtCore import Signal, Slot, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from supremediffusion.config.project_config import ProjectConfig

from sdqt.state import AppState

logger = logging.getLogger(__name__)

_VIDEO_RESOLUTION_PRESETS = [
    "832x480 (16:9)",
    "480x832 (9:16)",
    "512x512 (1:1)",
    "720x720 (1:1)",
    "832x624 (4:3)",
    "624x832 (3:4)",
    "960x544 (16:9)",
    "544x960 (9:16)",
    "1024x1024 (1:1)",
    "1280x720 (16:9)",
    "720x1280 (9:16)",
    "1280x544 (21:9)",
    "544x1280 (9:21)",
]

_IMAGE_RESOLUTION_PRESETS = [
    "512x512 (1:1)",
    "512x768 (2:3)",
    "768x512 (3:2)",
    "768x768 (1:1)",
    "1024x1024 (1:1)",
    "1024x768 (4:3)",
    "768x1024 (3:4)",
    "1024x1536 (2:3)",
    "1536x1024 (3:2)",
    "2048x2048 (1:1)",
]

_VIDEO_MODEL_PRESETS = [
    "(use Settings default)",
    "Wan 2.2 I2V 14B",
    "Wan 2.2 I2V 14B Lightning v2",
    "Wan 2.2 SVI 2 Pro (LoRAs)",
    "Wan 2.2 SVI 2 Pro Enhanced Lightning v2",
]

_SZ = 40
_ICON_BTN_STYLE = (
    f"QPushButton {{ border: 1px solid #444; padding: 0px;"
    f" background: #2a2a2a; border-radius: 4px;"
    f" min-width: {_SZ}px; max-width: {_SZ}px;"
    f" min-height: {_SZ}px; max-height: {_SZ}px; }} "
    "QPushButton:hover { background: #505050; border-color: #888; } "
    "QPushButton:pressed { background: #606060; }"
)
_ICON_PX = 28


def _make_project_icon(key: str, size: int = _ICON_PX) -> QIcon:
    """Paint a project action icon."""
    pm = QPixmap(size, size)
    pm.fill(QColor(0, 0, 0, 0))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    c = QColor(210, 210, 210)

    if key == "new":
        # Plus sign in a document outline
        p.setPen(QPen(c, 1.5))
        p.drawRect(4, 2, 14, 18)
        p.drawLine(7, 2, 7, 6)
        p.drawLine(7, 6, 18, 6)
        p.setPen(QPen(QColor(100, 220, 100), 2.5))
        p.drawLine(11, 10, 11, 18)
        p.drawLine(7, 14, 15, 14)
    elif key == "copy":
        # Two overlapping documents
        p.setPen(QPen(c, 1.5))
        p.drawRect(2, 4, 12, 16)
        p.setPen(QPen(QColor(100, 180, 255), 1.5))
        p.drawRect(8, 2, 12, 16)
    elif key == "rename":
        # Pencil
        p.setPen(QPen(c, 2.0))
        p.drawLine(4, 20, 18, 6)
        p.drawLine(18, 6, 20, 4)
        p.drawLine(20, 4, 18, 2)
        p.drawLine(18, 2, 16, 4)
        p.setPen(QPen(QColor(255, 200, 80), 1.5))
        p.drawLine(4, 20, 8, 18)
    elif key == "delete":
        # Trash can
        p.setPen(QPen(QColor(255, 100, 100), 2.0))
        p.drawLine(7, 7, 17, 7)
        p.drawRect(8, 7, 8, 13)
        p.drawLine(12, 4, 12, 7)
        p.drawLine(9, 4, 15, 4)
        p.setPen(QPen(QColor(255, 100, 100), 1.5))
        p.drawLine(10, 10, 10, 17)
        p.drawLine(14, 10, 14, 17)
    elif key == "folder":
        # Open folder
        p.setPen(QPen(c, 1.5))
        p.drawLine(2, 8, 2, 20)
        p.drawLine(2, 20, 20, 20)
        p.drawLine(20, 20, 22, 10)
        p.drawLine(22, 10, 6, 10)
        p.drawLine(2, 8, 8, 8)
        p.drawLine(8, 8, 10, 6)
        p.drawLine(10, 6, 16, 6)
        p.drawLine(16, 6, 16, 10)
        p.setPen(QPen(QColor(255, 200, 80), 1.5))
        p.drawLine(6, 10, 4, 20)
    elif key == "settings":
        # Gear
        p.setPen(QPen(c, 1.5))
        p.drawEllipse(7, 7, 10, 10)
        import math
        cx, cy, r = 12, 12, 10
        for i in range(8):
            angle = math.radians(i * 45)
            x1 = cx + (r - 3) * math.cos(angle)
            y1 = cy + (r - 3) * math.sin(angle)
            x2 = cx + r * math.cos(angle)
            y2 = cy + r * math.sin(angle)
            p.drawLine(int(x1), int(y1), int(x2), int(y2))

    p.end()
    return QIcon(pm)

_COMBO_STYLE = (
    "QComboBox { border: 2px solid #0078d4; padding: 4px 10px; font-size: 14px;"
    " font-weight: bold; color: #fff; background: #2a2a2a; border-radius: 4px;"
    " min-width: 220px; max-width: 320px; min-height: 30px; }"
    "QComboBox:hover { border-color: #3399ff; background: #333; }"
    "QComboBox::drop-down { border: none; width: 20px; }"
    "QComboBox::down-arrow { image: none; border-left: 4px solid transparent;"
    " border-right: 4px solid transparent; border-top: 6px solid #0078d4;"
    " margin-right: 6px; }"
    "QComboBox QAbstractItemView { background: #2a2a2a; color: #fff;"
    " selection-background-color: #0078d4; border: 1px solid #555; }"
)


class ProjectSettingsDialog(QDialog):
    """Project settings dialog — resolution, default models, and per-project options."""

    def __init__(
        self,
        resolution: str = "512x512",
        image_resolution: str = "1024x1024",
        video_model: str = "",
        image_model: str = "",
        image_steps: int = 20,
        image_cfg: float = 7.0,
        image_sampler: str = "",
        checkpoints: list[str] | None = None,
        samplers: list[str] | None = None,
        color_profile: str = "",
        global_default_color_profile: str = "",
        name: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Project Settings" if not name else "New Project")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Project name (only shown for new projects)
        if name is not None and name == "":
            name_row = QHBoxLayout()
            name_row.addWidget(QLabel("Project name:"))
            self._name_edit = QLineEdit()
            self._name_edit.setPlaceholderText("Enter project name")
            name_row.addWidget(self._name_edit, 1)
            layout.addLayout(name_row)
            self._show_name = True
        else:
            self._name_edit = None
            self._show_name = False

        # Video Resolution
        vres_row = QHBoxLayout()
        vres_row.setSpacing(6)
        vres_row.addWidget(QLabel("Video Resolution:"))
        self._resolution = QComboBox()
        self._resolution.setMinimumWidth(160)
        self._resolution.addItems(_VIDEO_RESOLUTION_PRESETS)
        self._select_combo(self._resolution, resolution)
        vres_row.addWidget(self._resolution)
        vres_row.addStretch()
        layout.addLayout(vres_row)

        # Image Resolution
        ires_row = QHBoxLayout()
        ires_row.setSpacing(6)
        ires_row.addWidget(QLabel("Image Resolution:"))
        self._image_resolution = QComboBox()
        self._image_resolution.setMinimumWidth(160)
        self._image_resolution.addItems(_IMAGE_RESOLUTION_PRESETS)
        self._select_combo(self._image_resolution, image_resolution)
        ires_row.addWidget(self._image_resolution)
        ires_row.addStretch()
        layout.addLayout(ires_row)

        # Default Video Model
        vm_row = QHBoxLayout()
        vm_row.setSpacing(6)
        vm_row.addWidget(QLabel("Video Model:"))
        self._video_model = QComboBox()
        self._video_model.setMinimumWidth(260)
        self._video_model.addItems(_VIDEO_MODEL_PRESETS)
        if video_model:
            idx = self._video_model.findText(video_model)
            if idx >= 0:
                self._video_model.setCurrentIndex(idx)
        vm_row.addWidget(self._video_model)
        vm_row.addStretch()
        layout.addLayout(vm_row)

        # Default Image Model (checkpoint)
        im_row = QHBoxLayout()
        im_row.setSpacing(6)
        im_row.addWidget(QLabel("Image Model:"))
        self._image_model = QComboBox()
        self._image_model.setMinimumWidth(260)
        self._image_model.addItem("(use Settings default)")
        if checkpoints:
            self._image_model.addItems(checkpoints)
        if image_model:
            idx = self._image_model.findText(image_model)
            if idx >= 0:
                self._image_model.setCurrentIndex(idx)
        im_row.addWidget(self._image_model)
        im_row.addStretch()
        layout.addLayout(im_row)

        # Default Image Generation Parameters
        from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox
        param_row = QHBoxLayout()
        param_row.setSpacing(6)

        param_row.addWidget(QLabel("Steps:"))
        self._image_steps = QSpinBox()
        self._image_steps.setRange(1, 150)
        self._image_steps.setValue(image_steps)
        self._image_steps.setMinimumWidth(90)
        param_row.addWidget(self._image_steps)

        param_row.addWidget(QLabel("CFG:"))
        self._image_cfg = QDoubleSpinBox()
        self._image_cfg.setRange(1.0, 30.0)
        self._image_cfg.setDecimals(1)
        self._image_cfg.setSingleStep(0.5)
        self._image_cfg.setValue(image_cfg)
        self._image_cfg.setMinimumWidth(95)
        param_row.addWidget(self._image_cfg)

        param_row.addWidget(QLabel("Sampler:"))
        self._image_sampler = QComboBox()
        self._image_sampler.setMinimumWidth(160)
        if samplers:
            self._image_sampler.addItems(samplers)
        else:
            # Fallback common samplers
            self._image_sampler.addItems([
                "Euler a", "Euler", "DPM++ 2M", "DPM++ 2M Karras",
                "DPM++ SDE Karras", "DDIM", "UniPC",
            ])
        if image_sampler:
            idx = self._image_sampler.findText(image_sampler)
            if idx >= 0:
                self._image_sampler.setCurrentIndex(idx)
        param_row.addWidget(self._image_sampler)

        param_row.addStretch()
        layout.addLayout(param_row)

        # Color Profile (per-project; empty = inherit global default)
        cp_row = QHBoxLayout()
        cp_row.setSpacing(6)
        cp_row.addWidget(QLabel("Color Profile:"))
        self._color_profile = QComboBox()
        self._color_profile.setMinimumWidth(280)
        from supremediffusion.config.color_profile import (
            list_profiles as _list_profiles,
            DEFAULT_PROFILE_KEY as _DEFAULT_KEY,
        )
        # First entry inherits global default; remaining entries pin a profile.
        self._color_profile.addItem(
            "(inherit Settings default)", "",
        )
        for p in _list_profiles():
            self._color_profile.addItem(p.label, p.key)
        # Select either the explicitly-set profile, or the inherit-default
        # entry. Saving the dialog with "(inherit …)" leaves color_profile=""
        # on the project so future changes to the global default propagate.
        target_key = (color_profile or "").strip()
        cp_idx = self._color_profile.findData(target_key) if target_key else 0
        if cp_idx < 0:
            cp_idx = self._color_profile.findData(_DEFAULT_KEY)
        if cp_idx >= 0:
            self._color_profile.setCurrentIndex(cp_idx)
        cp_row.addWidget(self._color_profile)
        # Tooltip surfaces what the global default is right now.
        if global_default_color_profile:
            self._color_profile.setToolTip(
                f"Global default: {global_default_color_profile}. "
                "Per-project overrides apply to every render in this project."
            )
        cp_row.addStretch()
        layout.addLayout(cp_row)

        layout.addStretch()

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _select_combo(combo: QComboBox, value: str) -> None:
        """Select a combo item that starts with the given value string."""
        if not value:
            return
        for i in range(combo.count()):
            if combo.itemText(i).startswith(value):
                combo.setCurrentIndex(i)
                return

    def get_resolution(self) -> str:
        return self._resolution.currentText().split(" ")[0]

    def get_image_resolution(self) -> str:
        return self._image_resolution.currentText().split(" ")[0]

    def get_video_model(self) -> str:
        text = self._video_model.currentText()
        return "" if text.startswith("(") else text

    def get_image_model(self) -> str:
        text = self._image_model.currentText()
        return "" if text.startswith("(") else text

    def get_image_steps(self) -> int:
        return self._image_steps.value()

    def get_image_cfg(self) -> float:
        return self._image_cfg.value()

    def get_image_sampler(self) -> str:
        return self._image_sampler.currentText()

    def get_color_profile(self) -> str:
        """Return the selected profile slug — empty string for inherit-global."""
        data = self._color_profile.currentData()
        return data or ""

    def get_name(self) -> str:
        if self._name_edit:
            return self._name_edit.text().strip()
        return ""


class ProjectBar(QWidget):
    """Prominent project selector banner with dropdown + action buttons.

    Signals:
        project_changed(str): emitted when active project changes
    """

    project_changed = Signal(str)

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Project label
        lbl = QLabel("Project:")
        lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #ccc;")
        layout.addWidget(lbl)

        # Project combo (prominent)
        self._combo = QComboBox()
        self._combo.setStyleSheet(_COMBO_STYLE)
        self._combo.currentTextChanged.connect(self._on_selection)
        layout.addWidget(self._combo)

        # Action buttons (icon-based) — grouped in a tight cluster, set off
        # from the project combo with a little leading space.
        layout.addSpacing(8)
        btn_group = QHBoxLayout()
        btn_group.setContentsMargins(0, 0, 0, 0)
        btn_group.setSpacing(4)
        _btns = [
            ("new", "New project", self._new_project),
            ("copy", "Copy project", self._copy_project),
            ("rename", "Rename project", self._rename_project),
            ("delete", "Delete project", self._delete_project),
            ("folder", "Open folder", self._open_folder),
            ("settings", "Project settings", self._project_settings),
        ]
        for key, tip, handler in _btns:
            btn = QPushButton()
            btn.setIcon(_make_project_icon(key))
            btn.setIconSize(QPixmap(_ICON_PX, _ICON_PX).size())
            btn.setFixedSize(_SZ, _SZ)
            btn.setToolTip(tip)
            btn.setStyleSheet(_ICON_BTN_STYLE)
            btn.clicked.connect(handler)
            btn_group.addWidget(btn)
        layout.addLayout(btn_group)

        self._refresh()

    # -- Public API --------------------------------------------------------

    def emit_initial_project(self) -> None:
        """Emit the current project so all tabs get on_project_changed at startup."""
        if self._combo.currentText():
            self.project_changed.emit(self._combo.currentText())

    @property
    def current_project(self) -> str:
        return self._combo.currentText()

    def _scan_checkpoints(self) -> list[str]:
        """Scan the SD checkpoint directory for available models (recursive)."""
        try:
            from supremediffusion.models.sd_models import scan_checkpoints
            ckpt_dir = self._state.global_config.model_paths.get("sd_checkpoint_dir", "")
            if ckpt_dir:
                infos = scan_checkpoints(ckpt_dir, recursive=True)
                return [info.name for info in infos]
        except Exception:
            pass
        return []

    def _scan_samplers(self) -> list[str]:
        """Get available sampler names from the SD backend."""
        try:
            from supremediffusion.models.sd_samplers import list_samplers
            return list_samplers()
        except Exception:
            return []

    def _refresh(self) -> None:
        self._combo.blockSignals(True)
        prev = self._combo.currentText()
        self._combo.clear()
        projects = self._state.project_manager.list_projects()
        self._combo.addItems(projects)
        if prev in projects:
            self._combo.setCurrentText(prev)
        elif projects:
            self._combo.setCurrentIndex(0)
        self._combo.blockSignals(False)

    # -- Slots -------------------------------------------------------------

    @Slot(str)
    def _on_selection(self, name: str) -> None:
        if name:
            self.project_changed.emit(name)

    @Slot()
    def _new_project(self) -> None:
        checkpoints = self._scan_checkpoints()
        samplers = self._scan_samplers()
        global_default_cp = getattr(
            self._state.global_config, "default_color_profile", "",
        ) or ""
        dlg = ProjectSettingsDialog(
            resolution="832x480", image_resolution="1024x1024",
            image_steps=20, image_cfg=7.0,
            checkpoints=checkpoints, samplers=samplers,
            color_profile="",  # new project starts as inherit-global
            global_default_color_profile=global_default_cp,
            name="", parent=self,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        name = dlg.get_name()
        if not name:
            QMessageBox.warning(self, "Error", "Project name cannot be empty.")
            return
        try:
            self._state.project_manager.create_project(name)
            path = self._state.project_manager.get_project_path(name)
            cfg = ProjectConfig.load(path)
            cfg.resolution = dlg.get_resolution()
            cfg.image_resolution = dlg.get_image_resolution()
            cfg.default_video_model = dlg.get_video_model()
            cfg.default_image_model = dlg.get_image_model()
            cfg.default_image_steps = dlg.get_image_steps()
            cfg.default_image_cfg = dlg.get_image_cfg()
            cfg.default_image_sampler = dlg.get_image_sampler()
            cfg.color_profile = dlg.get_color_profile()
            cfg.save(path)
            self._refresh()
            self._combo.setCurrentText(name)
        except FileExistsError:
            QMessageBox.warning(self, "Error", f"Project '{name}' already exists.")

    @Slot()
    def _copy_project(self) -> None:
        source = self._combo.currentText()
        if not source:
            return
        name, ok = QInputDialog.getText(
            self, "Copy Project", "Name for new project:", text=f"{source}_copy",
        )
        if ok and name.strip():
            name = name.strip()
            try:
                self._state.project_manager.copy_project(source, name)
                self._refresh()
                self._combo.setCurrentText(name)
            except FileExistsError:
                QMessageBox.warning(self, "Error", f"Project '{name}' already exists.")
            except Exception as exc:
                QMessageBox.warning(self, "Error", str(exc))

    @Slot()
    def _rename_project(self) -> None:
        old = self._combo.currentText()
        if old == "_default":
            QMessageBox.warning(self, "Error", "Cannot rename the default project.")
            return
        name, ok = QInputDialog.getText(self, "Rename Project", "New name:", text=old)
        if ok and name.strip() and name.strip() != old:
            try:
                self._state.project_manager.rename_project(old, name.strip())
                self._refresh()
                self._combo.setCurrentText(name.strip())
            except Exception as exc:
                QMessageBox.warning(self, "Error", str(exc))

    @Slot()
    def _open_folder(self) -> None:
        name = self._combo.currentText()
        if not name:
            return
        path = self._state.project_manager.get_project_path(name)
        if not path.is_dir():
            return
        if sys.platform == "linux":
            subprocess.Popen(["xdg-open", str(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer", str(path)])

    @Slot()
    def _project_settings(self) -> None:
        name = self._combo.currentText()
        if not name:
            return
        path = self._state.project_manager.get_project_path(name)
        cfg = ProjectConfig.load(path)
        checkpoints = self._scan_checkpoints()
        samplers = self._scan_samplers()

        global_default_cp = getattr(
            self._state.global_config, "default_color_profile", "",
        ) or ""
        dlg = ProjectSettingsDialog(
            resolution=getattr(cfg, "resolution", "832x480") or "832x480",
            image_resolution=getattr(cfg, "image_resolution", "1024x1024") or "1024x1024",
            video_model=getattr(cfg, "default_video_model", "") or "",
            image_model=getattr(cfg, "default_image_model", "") or "",
            image_steps=getattr(cfg, "default_image_steps", 20) or 20,
            image_cfg=getattr(cfg, "default_image_cfg", 7.0) or 7.0,
            image_sampler=getattr(cfg, "default_image_sampler", "") or "",
            checkpoints=checkpoints,
            samplers=samplers,
            color_profile=getattr(cfg, "color_profile", "") or "",
            global_default_color_profile=global_default_cp,
            name=None,
            parent=self,
        )
        if dlg.exec() != QDialog.Accepted:
            return

        cfg.resolution = dlg.get_resolution()
        cfg.image_resolution = dlg.get_image_resolution()
        cfg.default_video_model = dlg.get_video_model()
        cfg.default_image_model = dlg.get_image_model()
        cfg.default_image_steps = dlg.get_image_steps()
        cfg.default_image_cfg = dlg.get_image_cfg()
        cfg.default_image_sampler = dlg.get_image_sampler()
        cfg.color_profile = dlg.get_color_profile()
        cfg.save(path)
        self.project_changed.emit(name)

    @Slot()
    def _delete_project(self) -> None:
        name = self._combo.currentText()
        if name == "_default":
            QMessageBox.warning(self, "Error", "Cannot delete the default project.")
            return
        reply = QMessageBox.question(
            self, "Delete Project",
            f"Delete project '{name}' and all its files?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            try:
                self._state.project_manager.delete_project(name)
                self._refresh()
            except Exception as exc:
                QMessageBox.warning(self, "Error", str(exc))
