"""Daz2Supreme tab — import Daz3D figures, pose, build scenes, render stills.

Layout (3-panel):
  Left:   Bone editor + Scene library (scenes/instances)
  Center: PBR 3D viewport
  Right:  Properties (render resolution, camera, lights, figure list)
          + Output gallery with send-to
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sdqt.tabs.base import BaseTab
from sdqt.state import AppState
from sdqt.widgets.bone_editor import BoneEditorWidget
from sdqt.widgets.gl_viewport_pbr import PBRViewportWidget
from sdqt.widgets.scene_library import SceneLibraryWidget
from sdqt.widgets.image_gallery import ImageGalleryWidget
from sdqt.widgets.send_targets import IMAGE_TARGETS
from sdqt.models.scene3d import (
    Light,
    Scene3D,
    SceneInstance,
    load_instance,
    save_instance,
)
from sdqt.models.daz_parser import DazParser

logger = logging.getLogger(__name__)

# ── Resolution presets (SDXL + Wan 2.2 native) ──────────────────────────────

_RENDER_PRESETS: list[tuple[str, int, int]] = [
    # SDXL
    ("1024\u00d71024 (1:1)",   1024, 1024),
    ("1152\u00d7896 (9:7)",    1152, 896),
    ("896\u00d71152 (7:9)",    896,  1152),
    ("1216\u00d7832 (3:2)",    1216, 832),
    ("832\u00d71216 (2:3)",    832,  1216),
    ("1344\u00d7768 (16:9)",   1344, 768),
    ("768\u00d71344 (9:16)",   768,  1344),
    ("1536\u00d7640 (21:9)",   1536, 640),
    ("640\u00d71536 (9:21)",   640,  1536),
    # Wan 2.2
    ("832\u00d7480 (16:9)",    832,  480),
    ("480\u00d7832 (9:16)",    480,  832),
    ("512\u00d7512 (1:1)",     512,  512),
    ("720\u00d7720 (1:1)",     720,  720),
    ("960\u00d7544 (16:9)",    960,  544),
    ("544\u00d7960 (9:16)",    544,  960),
    # Legacy SD 1.5
    ("768\u00d7768 (1:1)",     768,  768),
]


class Daz2SupremeTab(BaseTab):
    """Main Daz2Supreme tab — 3D import, pose, render."""

    send_image_requested = Signal(str, str)  # (target_key, image_path)

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._scene: Scene3D | None = None
        self._current_scene_dir: Path | None = None
        self._current_instance_path: Path | None = None
        self._parser = DazParser()

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 0, 2, 2)
        root.setSpacing(4)

        # ── Toolbar ──────────────────────────────────────────────────────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        self._btn_import = QPushButton("Import .duf")
        self._btn_import.setFixedWidth(100)
        toolbar.addWidget(self._btn_import)

        self._btn_save = QPushButton("Save Instance")
        self._btn_save.setFixedWidth(100)
        toolbar.addWidget(self._btn_save)

        self._btn_reset_pose = QPushButton("Reset Pose")
        self._btn_reset_pose.setFixedWidth(90)
        toolbar.addWidget(self._btn_reset_pose)

        toolbar.addStretch()

        toolbar.addWidget(QLabel("Resolution:"))
        self._res_combo = QComboBox()
        self._res_combo.setFixedWidth(160)
        for label, _w, _h in _RENDER_PRESETS:
            self._res_combo.addItem(label)
        toolbar.addWidget(self._res_combo)

        self._btn_render = QPushButton("Render")
        self._btn_render.setFixedWidth(80)
        self._btn_render.setStyleSheet(
            "QPushButton { background: #0078d4; color: white; font-weight: bold;"
            " border-radius: 4px; padding: 4px 12px; }"
            "QPushButton:hover { background: #1a8ae8; }"
        )
        toolbar.addWidget(self._btn_render)

        root.addLayout(toolbar)

        # ── Main 3-panel splitter ────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # LEFT: Bone editor + scene library
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        self._bone_editor = BoneEditorWidget()
        left_layout.addWidget(self._bone_editor, 1)

        self._scene_library = SceneLibraryWidget()
        left_layout.addWidget(self._scene_library, 1)

        splitter.addWidget(left)

        # CENTER: 3D viewport
        self._viewport = PBRViewportWidget()
        splitter.addWidget(self._viewport)

        # RIGHT: Properties + output gallery
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Display options
        display_group = QGroupBox("Display")
        dg_layout = QVBoxLayout(display_group)
        dg_layout.setContentsMargins(4, 8, 4, 4)
        self._chk_bones = QCheckBox("Show bones")
        self._chk_bones.setChecked(True)
        self._chk_grid = QCheckBox("Show grid")
        self._chk_grid.setChecked(True)
        dg_layout.addWidget(self._chk_bones)
        dg_layout.addWidget(self._chk_grid)

        hdri_row = QHBoxLayout()
        hdri_row.setSpacing(4)
        self._btn_hdri = QPushButton("HDRI...")
        self._btn_hdri.setFixedWidth(60)
        self._btn_hdri.clicked.connect(self._on_load_hdri)
        hdri_row.addWidget(self._btn_hdri)
        self._btn_clear_hdri = QPushButton("Clear")
        self._btn_clear_hdri.setFixedWidth(50)
        self._btn_clear_hdri.clicked.connect(lambda: self._viewport.set_hdri(""))
        hdri_row.addWidget(self._btn_clear_hdri)
        self._hdri_label = QLabel("(none)")
        self._hdri_label.setStyleSheet("color: #888; font-size: 11px;")
        hdri_row.addWidget(self._hdri_label, 1)
        dg_layout.addLayout(hdri_row)

        right_layout.addWidget(display_group)

        # Scene objects list
        objects_group = QGroupBox("Scene Objects")
        og_layout = QVBoxLayout(objects_group)
        og_layout.setContentsMargins(4, 8, 4, 4)
        og_layout.setSpacing(2)

        self._object_list = QListWidget()
        self._object_list.setMaximumHeight(120)
        self._object_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._object_list.customContextMenuRequested.connect(self._object_context_menu)
        og_layout.addWidget(self._object_list)

        obj_btn_row = QHBoxLayout()
        obj_btn_row.setSpacing(4)
        self._btn_toggle_vis = QPushButton("Toggle Visible")
        self._btn_toggle_vis.setFixedWidth(100)
        self._btn_toggle_vis.clicked.connect(self._toggle_object_visibility)
        obj_btn_row.addWidget(self._btn_toggle_vis)
        self._btn_remove_obj = QPushButton("Remove")
        self._btn_remove_obj.setFixedWidth(70)
        self._btn_remove_obj.clicked.connect(self._remove_selected_object)
        obj_btn_row.addWidget(self._btn_remove_obj)
        obj_btn_row.addStretch()
        og_layout.addLayout(obj_btn_row)

        right_layout.addWidget(objects_group)

        # Light editor
        light_group = QGroupBox("Lighting")
        lg_layout = QVBoxLayout(light_group)
        lg_layout.setContentsMargins(4, 8, 4, 4)
        lg_layout.setSpacing(4)

        self._light_widgets: list[dict] = []
        for li in range(2):
            lbl = QLabel(f"Light {li + 1}")
            lbl.setStyleSheet("font-weight: bold; color: #ccc;")
            lg_layout.addWidget(lbl)

            row1 = QHBoxLayout()
            row1.setSpacing(4)
            row1.addWidget(QLabel("Dir X:"))
            dx = QDoubleSpinBox()
            dx.setRange(-1.0, 1.0); dx.setDecimals(2); dx.setSingleStep(0.1)
            dx.setFixedWidth(60)
            dx.setValue(-0.3 if li == 0 else 0.5)
            row1.addWidget(dx)
            row1.addWidget(QLabel("Y:"))
            dy = QDoubleSpinBox()
            dy.setRange(-1.0, 1.0); dy.setDecimals(2); dy.setSingleStep(0.1)
            dy.setFixedWidth(60)
            dy.setValue(-0.8 if li == 0 else -0.6)
            row1.addWidget(dy)
            row1.addWidget(QLabel("Z:"))
            dz = QDoubleSpinBox()
            dz.setRange(-1.0, 1.0); dz.setDecimals(2); dz.setSingleStep(0.1)
            dz.setFixedWidth(60)
            dz.setValue(-0.5 if li == 0 else 0.3)
            row1.addWidget(dz)
            row1.addStretch()
            lg_layout.addLayout(row1)

            row2 = QHBoxLayout()
            row2.setSpacing(4)
            row2.addWidget(QLabel("Intensity:"))
            intensity = QDoubleSpinBox()
            intensity.setRange(0.0, 5.0); intensity.setDecimals(2); intensity.setSingleStep(0.1)
            intensity.setFixedWidth(60)
            intensity.setValue(1.0 if li == 0 else 0.4)
            row2.addWidget(intensity)
            color_btn = QPushButton("")
            color_btn.setFixedSize(24, 24)
            init_color = (255, 255, 255) if li == 0 else (200, 220, 255)
            color_btn.setStyleSheet(
                f"background: rgb({init_color[0]},{init_color[1]},{init_color[2]}); border: 1px solid #555;"
            )
            color_btn.setProperty("color", init_color)
            row2.addWidget(color_btn)
            row2.addStretch()
            lg_layout.addLayout(row2)

            lw = {"dx": dx, "dy": dy, "dz": dz, "intensity": intensity, "color_btn": color_btn}
            self._light_widgets.append(lw)

            # Connect changes
            for spin in (dx, dy, dz, intensity):
                spin.valueChanged.connect(self._on_light_changed)
            color_btn.clicked.connect(lambda checked=False, idx=li: self._pick_light_color(idx))

        right_layout.addWidget(light_group)

        # Daz content directory hint
        content_group = QGroupBox("Daz Content")
        cg_layout = QHBoxLayout(content_group)
        cg_layout.setContentsMargins(4, 8, 4, 4)
        self._btn_set_content = QPushButton("Set Content Dir")
        self._btn_set_content.setFixedWidth(120)
        cg_layout.addWidget(self._btn_set_content)
        self._content_label = QLabel("(not set)")
        self._content_label.setStyleSheet("color: #888;")
        cg_layout.addWidget(self._content_label, 1)
        right_layout.addWidget(content_group)

        # Status
        self._status = QLabel("")
        self._status.setStyleSheet("color: #888; font-size: 12px;")
        right_layout.addWidget(self._status)

        # Output gallery
        right_layout.addWidget(QLabel("Outputs"))
        self._gallery = ImageGalleryWidget()
        self._gallery.set_send_targets(IMAGE_TARGETS)
        right_layout.addWidget(self._gallery, 1)

        splitter.addWidget(right)

        # Splitter proportions: left=250, center=stretch, right=280
        splitter.setSizes([250, 600, 280])
        root.addWidget(splitter, 1)

    def _connect_signals(self) -> None:
        self._btn_import.clicked.connect(self._on_import)
        self._btn_save.clicked.connect(self._on_save_instance)
        self._btn_reset_pose.clicked.connect(self._on_reset_pose)
        self._btn_render.clicked.connect(self._on_render)
        self._btn_set_content.clicked.connect(self._on_set_content_dir)

        self._chk_bones.toggled.connect(self._viewport.set_show_bones)
        self._chk_grid.toggled.connect(self._viewport.set_show_grid)

        self._bone_editor.pose_changed.connect(self._on_pose_changed)
        self._bone_editor.morph_changed.connect(self._on_morph_changed)
        self._bone_editor.figure_switch_requested.connect(self._on_figure_switched)

        self._viewport.bone_selected.connect(self._on_bone_picked)
        self._viewport.ik_drag.connect(self._on_ik_drag)

        self._scene_library.scene_selected.connect(self._on_scene_selected)
        self._scene_library.instance_selected.connect(self._on_instance_selected)
        self._scene_library.scene_created.connect(self._on_scene_created)
        self._scene_library.render_output_selected.connect(
            lambda p: self.send_image_requested.emit("img2img", p)
        )

        self._gallery.send_requested.connect(
            lambda key: self._send_gallery_image(key)
        )

    # ------------------------------------------------------------------
    # BaseTab overrides
    # ------------------------------------------------------------------

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        scenes_root = self.project_path / "scenes"
        scenes_root.mkdir(parents=True, exist_ok=True)
        self._scene_library.set_scenes_root(scenes_root)
        self._bone_editor.set_poses_dir(self.project_path / "poses")
        self._scene = None
        self._current_scene_dir = None

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def _on_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Daz File",
            str(Path.home()),
            "Daz Files (*.duf *.dsf);;All Files (*)",
        )
        if not path:
            return

        self._show_status("Parsing Daz file...")
        try:
            result = self._parser.parse_scene(path)
        except Exception as exc:
            self._show_status(f"Import failed: {exc}")
            logger.error("Daz import failed: %s", exc, exc_info=True)
            return

        figures = result["figures"]
        props = result["props"]
        cameras = result["cameras"]
        lights = result["lights"]

        if not figures and not props:
            self._show_status("No geometry found in file")
            return

        # Create or extend scene
        name = figures[0].name if figures else props[0].name if props else "Scene"
        if not self._scene:
            self._scene = Scene3D(name=name)
        self._scene.figures.extend(figures)
        self._scene.props.extend(props)

        # Apply imported camera
        if cameras:
            self._viewport.set_camera_from(cameras[0])

        # Apply imported lights
        if lights:
            self._viewport.set_lights(lights)
            self._update_light_ui(lights)

        self._bone_editor.set_figures(self._scene.figures)
        self._viewport.set_scene(self._scene)
        self._refresh_object_list()

        # Cache to disk
        if self._current_scene_dir:
            self._scene.save_figure_data(self._current_scene_dir)
            self._scene.save_meta(self._current_scene_dir / "scene.json")

        parts = []
        if figures:
            total_verts = sum(len(f.mesh.vertices) for f in figures)
            total_bones = sum(len(f.skeleton.bones) for f in figures)
            parts.append(f"{len(figures)} figure(s) ({total_verts} verts, {total_bones} bones)")
        if props:
            total_pverts = sum(len(p.mesh.vertices) for p in props)
            parts.append(f"{len(props)} prop(s) ({total_pverts} verts)")
        if cameras:
            parts.append(f"{len(cameras)} camera(s)")
        if lights:
            parts.append(f"{len(lights)} light(s)")
        self._show_status(f"Imported: {', '.join(parts)}")

    # ------------------------------------------------------------------
    # Content directory
    # ------------------------------------------------------------------

    def _on_load_hdri(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load HDRI Environment",
            str(Path.home()),
            "HDR Images (*.hdr *.exr *.png *.jpg *.jpeg);;All Files (*)",
        )
        if path:
            self._viewport.set_hdri(path)
            self._hdri_label.setText(Path(path).name)
            if self._scene:
                self._scene.environment["hdri"] = path

    def _on_set_content_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Daz Content Directory")
        if d:
            self._parser.content_dirs = [Path(d)]
            self._content_label.setText(d)

    # ------------------------------------------------------------------
    # Posing
    # ------------------------------------------------------------------

    def _on_pose_changed(self, figure_id: str, pose: dict) -> None:
        self._viewport.update_pose(figure_id, pose)

    def _on_morph_changed(self, figure_id: str) -> None:
        self._viewport.update_morphs(figure_id)

    # ------------------------------------------------------------------
    # Lighting
    # ------------------------------------------------------------------

    def _build_lights_from_ui(self) -> list[Light]:
        """Build Light list from the light editor widgets."""
        import numpy as np
        lights = []
        for lw in self._light_widgets:
            c = lw["color_btn"].property("color") or (255, 255, 255)
            lights.append(Light(
                kind="directional",
                color=(c[0] / 255.0, c[1] / 255.0, c[2] / 255.0),
                intensity=lw["intensity"].value(),
                direction=np.array([lw["dx"].value(), lw["dy"].value(), lw["dz"].value()],
                                   dtype=np.float32),
            ))
        return lights

    def _on_light_changed(self, _val=None) -> None:
        self._viewport.set_lights(self._build_lights_from_ui())

    def _pick_light_color(self, light_idx: int) -> None:
        from PySide6.QtGui import QColor
        lw = self._light_widgets[light_idx]
        c = lw["color_btn"].property("color") or (255, 255, 255)
        color = QColorDialog.getColor(QColor(*c), self, f"Light {light_idx + 1} Color")
        if color.isValid():
            rgb = (color.red(), color.green(), color.blue())
            lw["color_btn"].setProperty("color", rgb)
            lw["color_btn"].setStyleSheet(
                f"background: rgb({rgb[0]},{rgb[1]},{rgb[2]}); border: 1px solid #555;"
            )
            self._on_light_changed()

    def _on_figure_switched(self, figure_id: str) -> None:
        """Bone editor switched active figure — update viewport highlight."""
        pass  # Viewport already shows all figures; bone editor handles tree rebuild

    def _on_bone_picked(self, figure_id: str, bone_name: str) -> None:
        """User clicked a bone in the 3D viewport — select it in the editor."""
        # Switch to the correct figure in the combo
        for i in range(self._bone_editor._fig_combo.count()):
            if self._bone_editor._fig_combo.itemData(i) == figure_id:
                self._bone_editor._fig_combo.setCurrentIndex(i)
                break
        self._bone_editor.select_bone(bone_name)

    def _on_ik_drag(self, figure_id: str, bone_name: str,
                    tx: float, ty: float, tz: float) -> None:
        """Viewport IK drag — solve and update pose."""
        if not self._bone_editor.ik_mode:
            return  # IK disabled in bone editor
        # Switch to correct figure
        for i in range(self._bone_editor._fig_combo.count()):
            if self._bone_editor._fig_combo.itemData(i) == figure_id:
                if self._bone_editor._fig_combo.currentIndex() != i:
                    self._bone_editor._fig_combo.setCurrentIndex(i)
                break
        self._bone_editor.solve_ik(bone_name, tx, ty, tz)

    def _on_reset_pose(self) -> None:
        self._bone_editor.reset_pose()

    # ------------------------------------------------------------------
    # Scene / Instance management
    # ------------------------------------------------------------------

    def _on_scene_selected(self, scene_dir_str: str) -> None:
        scene_dir = Path(scene_dir_str)
        self._current_scene_dir = scene_dir
        meta_path = scene_dir / "scene.json"
        if meta_path.is_file():
            try:
                self._scene = Scene3D.load_meta(meta_path)
                self._viewport.set_scene(self._scene)
                self._bone_editor.set_figures(self._scene.figures)
                self._refresh_object_list()
                self._show_status(f"Loaded scene: {self._scene.name}")
            except Exception as exc:
                self._show_status(f"Failed to load scene: {exc}")

        # Refresh output gallery
        self._refresh_outputs()

    def _on_scene_created(self, scene_dir_str: str) -> None:
        self._on_scene_selected(scene_dir_str)

    def _on_instance_selected(self, inst_path_str: str) -> None:
        inst_path = Path(inst_path_str)
        self._current_instance_path = inst_path
        try:
            inst = load_instance(inst_path)
            self._viewport.set_camera_from(inst.camera)
            # Restore poses for all figures
            for fig_id, poses in inst.figure_poses.items():
                self._bone_editor.set_pose(poses, figure_id=fig_id)
                self._viewport.update_pose(fig_id, poses)
            # Restore morph values
            for fig_id, morphs in inst.figure_morphs.items():
                if self._scene:
                    fig = self._scene.figure_by_id(fig_id)
                    if fig:
                        for mname, mval in morphs.items():
                            m = fig.morph_by_name(mname)
                            if m:
                                m.value = mval
            # Set resolution
            for i, (_label, w, h) in enumerate(_RENDER_PRESETS):
                if w == inst.render_width and h == inst.render_height:
                    self._res_combo.setCurrentIndex(i)
                    break
            self._show_status(f"Loaded instance: {inst.name}")
        except Exception as exc:
            self._show_status(f"Failed to load instance: {exc}")

    def _on_save_instance(self) -> None:
        if not self._current_scene_dir:
            self._show_status("No scene selected — create or select a scene first")
            return

        inst_dir = self._current_scene_dir / "instances"
        inst_dir.mkdir(exist_ok=True)
        thumb_dir = self._current_scene_dir / "thumbnails"
        thumb_dir.mkdir(exist_ok=True)

        # Build instance from current state
        cam = self._viewport.get_camera_state()
        idx = self._res_combo.currentIndex()
        _, rw, rh = _RENDER_PRESETS[idx] if idx >= 0 else ("", 1024, 1024)

        inst = SceneInstance(
            name="default",
            camera=cam if cam else SceneInstance().camera,
            render_width=rw,
            render_height=rh,
        )

        # Collect per-figure poses from bone editor
        if self._scene:
            for fig in self._scene.figures:
                pose = self._bone_editor.get_pose(figure_id=fig.id)
                if pose:
                    inst.figure_poses[fig.id] = pose
                # Collect morph values
                morph_vals = {m.name: m.value for m in fig.morphs if m.value != 0.0}
                if morph_vals:
                    inst.figure_morphs[fig.id] = morph_vals

        # Determine save path
        if self._current_instance_path:
            save_path = self._current_instance_path
            # Preserve name from existing
            try:
                old = load_instance(save_path)
                inst.name = old.name
                inst.id = old.id
            except Exception:
                pass
        else:
            save_path = inst_dir / "default.json"

        save_instance(inst, save_path)

        # Save thumbnail (small viewport capture)
        thumb_img = self._viewport.render_to_image(256, 256)
        if thumb_img:
            thumb_path = thumb_dir / f"{save_path.stem}.png"
            thumb_img.save(str(thumb_path))
            inst.thumbnail = f"thumbnails/{save_path.stem}.png"
            save_instance(inst, save_path)

        # Also save scene meta + figure data
        if self._scene:
            self._scene.save_meta(self._current_scene_dir / "scene.json")
            self._scene.save_figure_data(self._current_scene_dir)

        self._scene_library.refresh_instances(self._current_scene_dir)
        self._show_status(f"Saved instance: {inst.name}")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _on_render(self) -> None:
        if not self._current_scene_dir:
            self._show_status("No scene selected — create or select a scene first")
            return

        idx = self._res_combo.currentIndex()
        if idx < 0:
            return
        _, width, height = _RENDER_PRESETS[idx]

        self._btn_render.setEnabled(False)
        self._show_status(f"Rendering {width}x{height}...")

        # GL render must happen on main thread but is fast
        img = self._viewport.render_to_image(width, height)
        if img is None:
            self._show_status("Render failed — OpenGL not available")
            self._btn_render.setEnabled(True)
            return

        # Save to scene outputs
        out_dir = self._current_scene_dir / "outputs"
        out_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        out_path = out_dir / f"render_{timestamp}.png"
        img.save(str(out_path))

        self._refresh_outputs()
        self._btn_render.setEnabled(True)
        self._show_status(f"Rendered: {out_path.name} ({width}x{height})")

    def render_all_instances(self) -> None:
        """Batch render all instances in the current scene."""
        if not self._current_scene_dir:
            return
        inst_dir = self._current_scene_dir / "instances"
        if not inst_dir.is_dir():
            return
        instances = sorted(inst_dir.glob("*.json"))
        if not instances:
            self._show_status("No instances to render")
            return

        out_dir = self._current_scene_dir / "outputs"
        out_dir.mkdir(exist_ok=True)
        idx = self._res_combo.currentIndex()
        _, width, height = _RENDER_PRESETS[idx] if idx >= 0 else ("", 1024, 1024)

        self._btn_render.setEnabled(False)
        rendered = 0
        for inst_path in instances:
            try:
                inst = load_instance(inst_path)
                self._viewport.set_camera_from(inst.camera)
                for fig_id, poses in inst.figure_poses.items():
                    self._viewport.update_pose(fig_id, poses)
                img = self._viewport.render_to_image(width, height)
                if img:
                    out_path = out_dir / f"{inst_path.stem}_{int(time.time())}.png"
                    img.save(str(out_path))
                    rendered += 1
            except Exception as exc:
                logger.error("Batch render failed for %s: %s", inst_path.name, exc)

        self._refresh_outputs()
        self._btn_render.setEnabled(True)
        self._show_status(f"Batch rendered {rendered}/{len(instances)} instances")

    def _refresh_outputs(self) -> None:
        if not self._current_scene_dir:
            return
        out_dir = self._current_scene_dir / "outputs"
        if not out_dir.is_dir():
            self._gallery.load_images([])
            return
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        paths = sorted(
            [str(f) for f in out_dir.iterdir() if f.suffix.lower() in exts],
            reverse=True,
        )
        self._gallery.load_images(paths)

    # ------------------------------------------------------------------
    # Send-to
    # ------------------------------------------------------------------

    def _send_gallery_image(self, key: str) -> None:
        path = self._gallery_selected_path()
        if path:
            self.send_image_requested.emit(key, path)

    def _gallery_selected_path(self) -> str | None:
        """Get the currently selected image path from the output gallery."""
        items = self._gallery._list.selectedItems()
        if items:
            return items[0].data(Qt.ItemDataRole.UserRole)
        return None

    # ------------------------------------------------------------------
    # Scene object list
    # ------------------------------------------------------------------

    def _refresh_object_list(self) -> None:
        """Rebuild the scene objects list widget."""
        self._object_list.clear()
        if not self._scene:
            return
        for fig in self._scene.figures:
            item = QListWidgetItem(f"🦴 {fig.name}")
            item.setData(Qt.ItemDataRole.UserRole, ("figure", fig.id))
            self._object_list.addItem(item)
        for prop in self._scene.props:
            vis = "👁" if prop.visible else "🚫"
            item = QListWidgetItem(f"{vis} {prop.name}")
            item.setData(Qt.ItemDataRole.UserRole, ("prop", prop.id))
            self._object_list.addItem(item)

    def _toggle_object_visibility(self) -> None:
        item = self._object_list.currentItem()
        if not item or not self._scene:
            return
        kind, obj_id = item.data(Qt.ItemDataRole.UserRole)
        if kind == "prop":
            prop = self._scene.prop_by_id(obj_id)
            if prop:
                prop.visible = not prop.visible
                self._refresh_object_list()
                self._viewport.set_scene(self._scene)

    def _remove_selected_object(self) -> None:
        item = self._object_list.currentItem()
        if not item or not self._scene:
            return
        kind, obj_id = item.data(Qt.ItemDataRole.UserRole)
        if kind == "figure":
            self._scene.figures = [f for f in self._scene.figures if f.id != obj_id]
            self._bone_editor.set_figures(self._scene.figures)
        elif kind == "prop":
            self._scene.props = [p for p in self._scene.props if p.id != obj_id]
        self._refresh_object_list()
        self._viewport.set_scene(self._scene)

    def _object_context_menu(self, pos) -> None:
        item = self._object_list.itemAt(pos)
        if not item:
            return
        kind, obj_id = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        if kind == "prop":
            prop = self._scene.prop_by_id(obj_id) if self._scene else None
            if prop:
                vis_text = "Hide" if prop.visible else "Show"
                menu.addAction(vis_text, self._toggle_object_visibility)
        menu.addAction("Remove", self._remove_selected_object)
        menu.exec(self._object_list.mapToGlobal(pos))

    # ------------------------------------------------------------------
    # Light UI sync
    # ------------------------------------------------------------------

    def _update_light_ui(self, lights: list[Light]) -> None:
        """Update light editor widgets from imported light data."""
        for i, lt in enumerate(lights[:2]):
            if i >= len(self._light_widgets):
                break
            lw = self._light_widgets[i]
            lw["dx"].blockSignals(True)
            lw["dy"].blockSignals(True)
            lw["dz"].blockSignals(True)
            lw["intensity"].blockSignals(True)
            lw["dx"].setValue(float(lt.direction[0]))
            lw["dy"].setValue(float(lt.direction[1]))
            lw["dz"].setValue(float(lt.direction[2]))
            lw["intensity"].setValue(lt.intensity)
            rgb = (int(lt.color[0] * 255), int(lt.color[1] * 255), int(lt.color[2] * 255))
            lw["color_btn"].setProperty("color", rgb)
            lw["color_btn"].setStyleSheet(
                f"background: rgb({rgb[0]},{rgb[1]},{rgb[2]}); border: 1px solid #555;"
            )
            lw["dx"].blockSignals(False)
            lw["dy"].blockSignals(False)
            lw["dz"].blockSignals(False)
            lw["intensity"].blockSignals(False)
