"""FK bone posing widget — tree view with per-bone XYZ rotation sliders.

Bones are grouped by body region (head, spine, arms, legs, hands, face).
Selecting a bone shows its rotation sliders.  Changes emit ``pose_changed``
with the full pose dict for the active figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sdqt.models.scene3d import Figure, Morph, Skeleton
from sdqt.models.ik_solver import (
    IKChain,
    find_ik_chain,
    get_available_ik_chains,
    solve_fabrik,
)

# Body region display order and labels
_REGION_ORDER = [
    ("hip", "Hip / Pelvis"),
    ("spine", "Spine / Torso"),
    ("head", "Head"),
    ("face", "Face"),
    ("lArm", "Left Arm"),
    ("rArm", "Right Arm"),
    ("lHand", "Left Hand"),
    ("rHand", "Right Hand"),
    ("lLeg", "Left Leg"),
    ("rLeg", "Right Leg"),
    ("other", "Other"),
]


class _AxisSlider(QWidget):
    """Single axis rotation slider: label + slider + spinbox."""

    value_changed = Signal(float)

    def __init__(self, label: str, min_val: float = -180, max_val: float = 180,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        lbl = QLabel(label)
        lbl.setFixedWidth(16)
        lbl.setStyleSheet("font-weight: bold; color: " + {
            "X": "#e06060", "Y": "#60c060", "Z": "#6080e0"
        }.get(label, "#aaa"))
        layout.addWidget(lbl)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(int(min_val * 10), int(max_val * 10))
        self._slider.setValue(0)
        self._slider.setTickInterval(450)  # 45 degrees
        layout.addWidget(self._slider, 1)

        self._spin = QDoubleSpinBox()
        self._spin.setRange(min_val, max_val)
        self._spin.setDecimals(1)
        self._spin.setSingleStep(1.0)
        self._spin.setFixedWidth(70)
        layout.addWidget(self._spin)

        self._updating = False
        self._slider.valueChanged.connect(self._on_slider)
        self._spin.valueChanged.connect(self._on_spin)

    def _on_slider(self, val: int) -> None:
        if self._updating:
            return
        self._updating = True
        v = val / 10.0
        self._spin.setValue(v)
        self._updating = False
        self.value_changed.emit(v)

    def _on_spin(self, val: float) -> None:
        if self._updating:
            return
        self._updating = True
        self._slider.setValue(int(val * 10))
        self._updating = False
        self.value_changed.emit(val)

    def set_value(self, val: float) -> None:
        self._updating = True
        self._slider.setValue(int(val * 10))
        self._spin.setValue(val)
        self._updating = False

    def set_range(self, min_val: float, max_val: float) -> None:
        self._slider.setRange(int(min_val * 10), int(max_val * 10))
        self._spin.setRange(min_val, max_val)

    def value(self) -> float:
        return self._spin.value()


class BoneEditorWidget(QWidget):
    """Bone tree + rotation sliders for FK posing."""

    pose_changed = Signal(str, dict)  # (figure_id, {bone_name: (rx, ry, rz)})
    morph_changed = Signal(str)  # figure_id — emitted when any morph value changes
    figure_switch_requested = Signal(str)  # figure_id — emitted when user picks a different figure

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figures: list[Figure] = []
        self._figure: Figure | None = None
        self._current_bone: str = ""
        # Per-figure pose state: figure_id → {bone_name: (rx, ry, rz)}
        self._poses: dict[str, dict[str, tuple[float, float, float]]] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Figure selector (for multi-figure scenes)
        fig_row = QHBoxLayout()
        fig_row.setSpacing(4)
        fig_row.addWidget(QLabel("Figure:"))
        self._fig_combo = QComboBox()
        self._fig_combo.setMinimumWidth(120)
        self._fig_combo.currentIndexChanged.connect(self._on_figure_selected)
        fig_row.addWidget(self._fig_combo, 1)
        layout.addLayout(fig_row)

        # IK/FK toggle
        ik_row = QHBoxLayout()
        ik_row.setSpacing(4)
        ik_row.addWidget(QLabel("Mode:"))
        self._ik_toggle = QComboBox()
        self._ik_toggle.addItems(["FK (rotate bones)", "IK (drag targets)"])
        self._ik_toggle.setFixedWidth(160)
        ik_row.addWidget(self._ik_toggle)
        ik_row.addStretch()
        layout.addLayout(ik_row)

        self._ik_chains: dict[str, IKChain] = {}

        # Bone tree
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setIndentation(14)
        self._tree.currentItemChanged.connect(self._on_bone_selected)
        layout.addWidget(self._tree, 1)

        # Rotation sliders
        slider_group = QGroupBox("Rotation")
        slider_layout = QVBoxLayout(slider_group)
        slider_layout.setContentsMargins(4, 8, 4, 4)
        slider_layout.setSpacing(2)

        self._bone_label = QLabel("(no bone selected)")
        self._bone_label.setStyleSheet("font-weight: bold; color: #ccc;")
        slider_layout.addWidget(self._bone_label)

        self._slider_x = _AxisSlider("X")
        self._slider_y = _AxisSlider("Y")
        self._slider_z = _AxisSlider("Z")
        for s in (self._slider_x, self._slider_y, self._slider_z):
            s.value_changed.connect(self._on_rotation_changed)
            slider_layout.addWidget(s)

        layout.addWidget(slider_group)

        # Morph dials
        morph_group = QGroupBox("Morphs")
        morph_layout = QVBoxLayout(morph_group)
        morph_layout.setContentsMargins(4, 8, 4, 4)
        morph_layout.setSpacing(2)

        self._morph_filter = QComboBox()
        self._morph_filter.addItem("All Groups")
        self._morph_filter.currentIndexChanged.connect(self._filter_morphs)
        morph_layout.addWidget(self._morph_filter)

        self._morph_tree = QTreeWidget()
        self._morph_tree.setHeaderLabels(["Morph", "Value"])
        self._morph_tree.setIndentation(14)
        self._morph_tree.setColumnWidth(0, 150)
        morph_layout.addWidget(self._morph_tree, 1)

        morph_slider_row = QHBoxLayout()
        morph_slider_row.setSpacing(4)
        self._morph_slider = QSlider(Qt.Orientation.Horizontal)
        self._morph_slider.setRange(0, 1000)
        self._morph_slider.setValue(0)
        morph_slider_row.addWidget(self._morph_slider, 1)
        self._morph_spin = QDoubleSpinBox()
        self._morph_spin.setRange(0.0, 1.0)
        self._morph_spin.setDecimals(3)
        self._morph_spin.setSingleStep(0.01)
        self._morph_spin.setFixedWidth(70)
        morph_slider_row.addWidget(self._morph_spin)
        morph_layout.addLayout(morph_slider_row)

        self._morph_tree.currentItemChanged.connect(self._on_morph_selected)
        self._morph_slider.valueChanged.connect(self._on_morph_slider)
        self._morph_spin.valueChanged.connect(self._on_morph_spin)
        self._updating_morph = False
        self._current_morph: Morph | None = None

        layout.addWidget(morph_group)

        # Pose library
        pose_group = QGroupBox("Pose Library")
        pose_layout = QVBoxLayout(pose_group)
        pose_layout.setContentsMargins(4, 8, 4, 4)
        pose_layout.setSpacing(2)

        self._pose_list = QListWidget()
        self._pose_list.setMaximumHeight(80)
        self._pose_list.itemDoubleClicked.connect(self._on_pose_load)
        pose_layout.addWidget(self._pose_list)

        pose_btn_row = QHBoxLayout()
        pose_btn_row.setSpacing(4)
        btn_save_pose = QPushButton("Save")
        btn_save_pose.setFixedWidth(50)
        btn_save_pose.clicked.connect(self._on_pose_save)
        pose_btn_row.addWidget(btn_save_pose)
        btn_load_pose = QPushButton("Load")
        btn_load_pose.setFixedWidth(50)
        btn_load_pose.clicked.connect(self._on_pose_load)
        pose_btn_row.addWidget(btn_load_pose)
        btn_del_pose = QPushButton("Delete")
        btn_del_pose.setFixedWidth(50)
        btn_del_pose.clicked.connect(self._on_pose_delete)
        pose_btn_row.addWidget(btn_del_pose)
        pose_btn_row.addStretch()
        pose_layout.addLayout(pose_btn_row)

        self._poses_dir: Path | None = None
        layout.addWidget(pose_group)

    @property
    def _pose(self) -> dict[str, tuple[float, float, float]]:
        """Current figure's pose dict."""
        if self._figure:
            return self._poses.setdefault(self._figure.id, {})
        return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_figures(self, figures: list[Figure]) -> None:
        """Populate the figure combo from a scene's figure list."""
        self._figures = list(figures)
        self._fig_combo.blockSignals(True)
        self._fig_combo.clear()
        for fig in figures:
            self._fig_combo.addItem(fig.name, fig.id)
        self._fig_combo.blockSignals(False)
        if figures:
            self._set_figure(figures[0])

    def set_pose(self, pose: dict[str, tuple[float, float, float]],
                 figure_id: str | None = None) -> None:
        """Load a saved pose into the editor."""
        fid = figure_id or (self._figure.id if self._figure else "")
        if fid:
            self._poses[fid] = dict(pose)
        if self._current_bone and self._current_bone in self._pose:
            self._load_bone_sliders(self._current_bone)

    def get_pose(self, figure_id: str | None = None) -> dict[str, tuple[float, float, float]]:
        fid = figure_id or (self._figure.id if self._figure else "")
        return dict(self._poses.get(fid, {}))

    def reset_pose(self) -> None:
        """Reset all bones to rest position."""
        if self._figure:
            self._poses.pop(self._figure.id, None)
        if self._current_bone:
            self._load_bone_sliders(self._current_bone)
        if self._figure:
            self.pose_changed.emit(self._figure.id, {})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_figure_selected(self, index: int) -> None:
        fig_id = self._fig_combo.itemData(index)
        if not fig_id:
            return
        # Find figure in our stored list
        for fig in self._figures:
            if fig.id == fig_id:
                self._set_figure(fig)
                self.figure_switch_requested.emit(fig_id)
                return

    @property
    def ik_mode(self) -> bool:
        return self._ik_toggle.currentIndex() == 1

    def _set_figure(self, fig: Figure) -> None:
        self._figure = fig
        self._current_bone = ""
        self._build_tree(fig.skeleton)
        self._build_morph_tree()
        # Discover IK chains
        self._ik_chains = get_available_ik_chains(fig.skeleton)

    def solve_ik(self, bone_name: str, target_x: float, target_y: float,
                 target_z: float) -> None:
        """Solve IK for the chain ending at bone_name toward target."""
        if not self._figure:
            return
        target = np.array([target_x, target_y, target_z], dtype=np.float32)

        # Find chain — use predefined or build one dynamically
        chain = self._ik_chains.get(bone_name)
        if not chain:
            chain = find_ik_chain(self._figure.skeleton, bone_name)
        if not chain or len(chain.bone_indices) < 2:
            return

        pose = self._pose  # current figure's pose dict
        new_pose = solve_fabrik(self._figure, chain, target, pose)

        # Update our pose state with solved rotations
        for name, rot in new_pose.items():
            if name in [self._figure.skeleton.bones[i].name for i in chain.bone_indices]:
                pose[name] = rot

        # Update sliders if selected bone is in chain
        if self._current_bone and self._current_bone in pose:
            self._load_bone_sliders(self._current_bone)

        self.pose_changed.emit(self._figure.id, dict(pose))

    def _build_tree(self, skel: Skeleton) -> None:
        """Build the bone tree grouped by body region."""
        self._tree.clear()
        if not skel.bones:
            return

        # Group bones by region
        region_bones: dict[str, list[str]] = {}
        for bone in skel.bones:
            region = bone.region or "other"
            region_bones.setdefault(region, []).append(bone.name)

        # Build tree items in display order
        for region_key, region_label in _REGION_ORDER:
            names = region_bones.get(region_key, [])
            if not names:
                continue
            parent = QTreeWidgetItem(self._tree, [region_label])
            parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            parent.setExpanded(region_key in ("hip", "spine"))
            for name in names:
                item = QTreeWidgetItem(parent, [name])
                item.setData(0, Qt.ItemDataRole.UserRole, name)

    def _on_bone_selected(self, current: QTreeWidgetItem | None, _prev) -> None:
        if current is None:
            return
        bone_name = current.data(0, Qt.ItemDataRole.UserRole)
        if bone_name is None:
            return  # region header
        self._current_bone = bone_name
        self._load_bone_sliders(bone_name)

    def _load_bone_sliders(self, bone_name: str) -> None:
        self._bone_label.setText(bone_name)
        rot = self._pose.get(bone_name, (0.0, 0.0, 0.0))

        # Apply rotation limits if available
        if self._figure:
            bone = self._figure.skeleton.bone_by_name(bone_name)
            if bone:
                if bone.rotation_min is not None and bone.rotation_max is not None:
                    self._slider_x.set_range(float(bone.rotation_min[0]), float(bone.rotation_max[0]))
                    self._slider_y.set_range(float(bone.rotation_min[1]), float(bone.rotation_max[1]))
                    self._slider_z.set_range(float(bone.rotation_min[2]), float(bone.rotation_max[2]))
                else:
                    for s in (self._slider_x, self._slider_y, self._slider_z):
                        s.set_range(-180, 180)

        self._slider_x.set_value(rot[0])
        self._slider_y.set_value(rot[1])
        self._slider_z.set_value(rot[2])

    def _on_rotation_changed(self, _val: float) -> None:
        if not self._current_bone or not self._figure:
            return
        pose = self._pose  # property — returns current figure's dict
        rot = (self._slider_x.value(), self._slider_y.value(), self._slider_z.value())
        if rot == (0.0, 0.0, 0.0):
            pose.pop(self._current_bone, None)
        else:
            pose[self._current_bone] = rot
        self.pose_changed.emit(self._figure.id, dict(pose))

    # -- Morph dials --------------------------------------------------------

    def _build_morph_tree(self) -> None:
        """Rebuild morph tree for the active figure."""
        self._morph_tree.clear()
        self._morph_filter.blockSignals(True)
        self._morph_filter.clear()
        self._morph_filter.addItem("All Groups")
        self._morph_filter.blockSignals(False)
        self._current_morph = None

        if not self._figure or not self._figure.morphs:
            return

        groups: dict[str, list[Morph]] = {}
        for m in self._figure.morphs:
            g = m.group or "Other"
            groups.setdefault(g, []).append(m)

        for g in sorted(groups):
            self._morph_filter.addItem(g)

        self._populate_morph_items(groups)

    def _populate_morph_items(self, groups: dict[str, list[Morph]] | None = None,
                              filter_group: str = "") -> None:
        self._morph_tree.clear()
        if not self._figure:
            return
        if groups is None:
            groups = {}
            for m in self._figure.morphs:
                g = m.group or "Other"
                groups.setdefault(g, []).append(m)

        for g in sorted(groups):
            if filter_group and g != filter_group:
                continue
            parent = QTreeWidgetItem(self._morph_tree, [g, ""])
            parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            parent.setExpanded(True)
            for m in groups[g]:
                item = QTreeWidgetItem(parent, [m.name, f"{m.value:.3f}"])
                item.setData(0, Qt.ItemDataRole.UserRole, m.name)

    def _filter_morphs(self, index: int) -> None:
        g = self._morph_filter.currentText()
        self._populate_morph_items(filter_group="" if g == "All Groups" else g)

    def _on_morph_selected(self, current: QTreeWidgetItem | None, _prev) -> None:
        if current is None:
            return
        name = current.data(0, Qt.ItemDataRole.UserRole)
        if name is None or not self._figure:
            return
        morph = self._figure.morph_by_name(name)
        if not morph:
            return
        self._current_morph = morph
        self._updating_morph = True
        self._morph_slider.setRange(int(morph.min_value * 1000), int(morph.max_value * 1000))
        self._morph_spin.setRange(morph.min_value, morph.max_value)
        self._morph_slider.setValue(int(morph.value * 1000))
        self._morph_spin.setValue(morph.value)
        self._updating_morph = False

    def _on_morph_slider(self, val: int) -> None:
        if self._updating_morph or not self._current_morph:
            return
        self._updating_morph = True
        v = val / 1000.0
        self._morph_spin.setValue(v)
        self._current_morph.value = v
        self._updating_morph = False
        self._update_morph_display()
        if self._figure:
            self.morph_changed.emit(self._figure.id)

    def _on_morph_spin(self, val: float) -> None:
        if self._updating_morph or not self._current_morph:
            return
        self._updating_morph = True
        self._morph_slider.setValue(int(val * 1000))
        self._current_morph.value = val
        self._updating_morph = False
        self._update_morph_display()
        if self._figure:
            self.morph_changed.emit(self._figure.id)

    def _update_morph_display(self) -> None:
        """Update the value column for the currently selected morph item."""
        item = self._morph_tree.currentItem()
        if item and self._current_morph:
            item.setText(1, f"{self._current_morph.value:.3f}")

    def select_bone(self, bone_name: str) -> None:
        """Programmatically select a bone (e.g. from viewport click)."""
        for i in range(self._tree.topLevelItemCount()):
            region_item = self._tree.topLevelItem(i)
            if not region_item:
                continue
            for j in range(region_item.childCount()):
                child = region_item.child(j)
                if child and child.data(0, Qt.ItemDataRole.UserRole) == bone_name:
                    region_item.setExpanded(True)
                    self._tree.setCurrentItem(child)
                    return

    # -- Pose library ------------------------------------------------------

    def set_poses_dir(self, poses_dir: Path) -> None:
        """Set the directory for pose presets and refresh the list."""
        self._poses_dir = poses_dir
        self._poses_dir.mkdir(parents=True, exist_ok=True)
        self._refresh_pose_list()

    def _refresh_pose_list(self) -> None:
        self._pose_list.clear()
        if not self._poses_dir or not self._poses_dir.is_dir():
            return
        for f in sorted(self._poses_dir.glob("*.json")):
            self._pose_list.addItem(f.stem)

    def _on_pose_save(self) -> None:
        if not self._poses_dir or not self._figure:
            return
        name, ok = QInputDialog.getText(self, "Save Pose", "Pose name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        pose = self.get_pose()
        data = {
            "name": name,
            "figure_name": self._figure.name,
            "pose": {bone: list(rot) for bone, rot in pose.items()},
        }
        path = self._poses_dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2))
        self._refresh_pose_list()

    def _on_pose_load(self, item: QListWidgetItem | None = None) -> None:
        if not self._poses_dir or not self._figure:
            return
        if item is None:
            item = self._pose_list.currentItem()
        if not item:
            return
        path = self._poses_dir / f"{item.text()}.json"
        if not path.is_file():
            return
        data = json.loads(path.read_text())
        pose = {bone: tuple(rot) for bone, rot in data.get("pose", {}).items()}
        self.set_pose(pose)
        self.pose_changed.emit(self._figure.id, pose)

    def _on_pose_delete(self) -> None:
        if not self._poses_dir:
            return
        item = self._pose_list.currentItem()
        if not item:
            return
        path = self._poses_dir / f"{item.text()}.json"
        if path.is_file():
            path.unlink()
        self._refresh_pose_list()
