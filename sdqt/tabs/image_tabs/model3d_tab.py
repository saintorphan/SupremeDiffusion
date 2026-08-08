"""3D Modeling tab -- composite 3D meshes over backgrounds, send to image/video tabs."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sdqt.tabs.base import BaseTab
from sdqt.utils.codec import (
    configured_codec_args as _codec_args,
    configured_encoder_name as _enc_name,
    pix_fmt_args as _pix_fmt,
)
from sdqt.widgets.send_targets import DISABLED_TARGETS

logger = logging.getLogger(__name__)

# Check for 3D dependencies at import time (no heavy imports)
import importlib.util as _ilu
_HAS_OPENGL = _ilu.find_spec("OpenGL") is not None
_HAS_TRIMESH = _ilu.find_spec("trimesh") is not None
_3D_AVAILABLE = _HAS_OPENGL and _HAS_TRIMESH
del _ilu

_MESH_EXTS = {".obj", ".glb", ".gltf", ".ply", ".stl"}


class Model3DTab(BaseTab):
    """3D Modeling tab -- generate meshes, compose over backgrounds, send to any tab.

    Workflow:
        1. Load a background image (sent from other tabs or dropped)
        2. Generate a mesh from an image via TripoSR, or load from inventory
        3. Position/rotate/scale meshes in the OpenGL viewport
        4. Capture the viewport composite and send to Img2Img, Inpaint, Img2Vid, etc.
    """

    send_composite = Signal(str)  # "target|path" (same format as Draw tab)
    lora_dataset_ready = Signal(str)  # path to prepared dataset folder

    def __init__(self, state, parent=None) -> None:
        super().__init__(state, parent)
        self._mesh_worker = None
        self._tex_worker = None
        self._bg_image_path: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        if not _3D_AVAILABLE:
            msg = QLabel(
                "3D Modeling requires PyOpenGL and trimesh.\n\n"
                "Install with:\n  pip install PyOpenGL trimesh\n\n"
                "Then restart the application."
            )
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            msg.setStyleSheet("font-size: 14px; color: #888; padding: 40px;")
            layout.addWidget(msg)
            return

        from sdqt.widgets.gl_viewport import GLViewportWidget
        from sdqt.workers.model3d import MeshGenerationWorker

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left panel ────────────────────────────────────────────────────
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        lv.setSpacing(4)

        # Background plate
        bg_group = QGroupBox("Background Plate")
        bg_lay = QVBoxLayout(bg_group)
        self._bg_preview = QLabel()
        self._bg_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._bg_preview.setFixedHeight(120)
        self._bg_preview.setStyleSheet("background: #1a1a1a; border: 1px dashed #444;")
        self._bg_preview.setText("No background image")
        bg_lay.addWidget(self._bg_preview)
        self._clear_bg_btn = QPushButton("Clear Background")
        self._clear_bg_btn.clicked.connect(self._on_clear_bg)
        self._clear_bg_btn.setVisible(False)
        bg_lay.addWidget(self._clear_bg_btn)
        lv.addWidget(bg_group)

        # Generate mesh from image
        gen_group = QGroupBox("Generate Mesh")
        gen_lay = QVBoxLayout(gen_group)
        gen_lay.setContentsMargins(4, 4, 4, 4)
        gen_row1 = QHBoxLayout()
        gen_row1.addWidget(QLabel("Resolution:"))
        self._mesh_res = QSpinBox()
        self._mesh_res.setRange(64, 512)
        self._mesh_res.setSingleStep(64)
        self._mesh_res.setValue(256)
        self._mesh_res.setFixedWidth(75)
        self._mesh_res.setToolTip("Mesh resolution — higher = more detail, slower")
        gen_row1.addWidget(self._mesh_res)
        gen_row1.addWidget(QLabel("Format:"))
        self._mesh_fmt = QComboBox()
        self._mesh_fmt.addItems(["obj", "glb", "ply", "stl"])
        self._mesh_fmt.setFixedWidth(70)
        gen_row1.addWidget(self._mesh_fmt)
        self._remove_bg_cb = QCheckBox("Remove BG")
        self._remove_bg_cb.setChecked(True)
        self._remove_bg_cb.setToolTip("Remove background with BiRefNet before mesh generation")
        gen_row1.addWidget(self._remove_bg_cb)
        gen_row1.addStretch()
        gen_lay.addLayout(gen_row1)

        gen_row2 = QHBoxLayout()
        self._gen_from_bg_btn = QPushButton("Generate from Background")
        self._gen_from_bg_btn.setToolTip("Generate a 3D mesh from the loaded background image")
        self._gen_from_bg_btn.clicked.connect(self._on_generate_from_bg)
        gen_row2.addWidget(self._gen_from_bg_btn)
        self._gen_abort_btn = QPushButton("Abort")
        self._gen_abort_btn.setVisible(False)
        self._gen_abort_btn.clicked.connect(self._on_abort_generate)
        gen_row2.addWidget(self._gen_abort_btn)
        gen_lay.addLayout(gen_row2)
        lv.addWidget(gen_group)

        # Texture Refinement
        tex_group = QGroupBox("Texture Refinement")
        tex_lay = QVBoxLayout(tex_group)
        tex_lay.setContentsMargins(4, 4, 4, 4)
        tex_lay.setSpacing(4)

        tex_lay.addWidget(QLabel("Prompt:"))
        self._tex_prompt = QPlainTextEdit()
        self._tex_prompt.setPlaceholderText("Describe the surface (e.g. 'weathered bronze statue, fine patina')")
        self._tex_prompt.setMaximumHeight(50)
        tex_lay.addWidget(self._tex_prompt)

        tex_lay.addWidget(QLabel("Negative:"))
        self._tex_neg_prompt = QLineEdit()
        self._tex_neg_prompt.setPlaceholderText("low quality, blurry, flat color")
        self._tex_neg_prompt.setText("low quality, blurry, flat color, smooth plastic")
        tex_lay.addWidget(self._tex_neg_prompt)

        tex_r1 = QHBoxLayout()
        tex_r1.addWidget(QLabel("Denoise:"))
        self._tex_denoise = QDoubleSpinBox()
        self._tex_denoise.setRange(0.1, 0.8)
        self._tex_denoise.setSingleStep(0.05)
        self._tex_denoise.setValue(0.35)
        self._tex_denoise.setFixedWidth(70)
        self._tex_denoise.setToolTip("Lower = preserve more structure, higher = more creative detail")
        tex_r1.addWidget(self._tex_denoise)
        tex_r1.addWidget(QLabel("Tex Size:"))
        self._tex_size_combo = QComboBox()
        self._tex_size_combo.addItems(["512", "1024", "2048"])
        self._tex_size_combo.setCurrentText("1024")
        self._tex_size_combo.setFixedWidth(70)
        tex_r1.addWidget(self._tex_size_combo)
        tex_r1.addWidget(QLabel("Views:"))
        self._tex_views = QSpinBox()
        self._tex_views.setRange(4, 16)
        self._tex_views.setValue(8)
        self._tex_views.setFixedWidth(50)
        self._tex_views.setToolTip("Number of camera angles (4=cardinal, 8=+diagonals, 16=two elevations)")
        tex_r1.addWidget(self._tex_views)
        tex_r1.addStretch()
        tex_lay.addLayout(tex_r1)

        tex_r2 = QHBoxLayout()
        self._tex_refine_btn = QPushButton("Refine Texture")
        self._tex_refine_btn.setToolTip(
            "Render multi-angle views, enhance with SD img2img, "
            "project back onto UV texture map"
        )
        self._tex_refine_btn.clicked.connect(self._on_refine_texture)
        tex_r2.addWidget(self._tex_refine_btn)
        self._tex_abort_btn = QPushButton("Abort")
        self._tex_abort_btn.setVisible(False)
        self._tex_abort_btn.clicked.connect(self._on_abort_refine)
        tex_r2.addWidget(self._tex_abort_btn)
        tex_r2.addStretch()
        tex_lay.addLayout(tex_r2)

        self._tex_progress = QProgressBar()
        self._tex_progress.setRange(0, 100)
        self._tex_progress.setFixedHeight(16)
        self._tex_progress.setVisible(False)
        tex_lay.addWidget(self._tex_progress)

        lv.addWidget(tex_group)

        # 3D Inventory
        inv_group = QGroupBox("3D Item Inventory")
        inv_lay = QVBoxLayout(inv_group)
        self._inventory = QListWidget()
        self._inventory.setMaximumHeight(140)
        self._inventory.setToolTip("Saved 3D models — double-click to add to scene")
        self._inventory.itemDoubleClicked.connect(self._on_inventory_add)
        inv_lay.addWidget(self._inventory)
        inv_btns = QHBoxLayout()
        self._import_mesh_btn = QPushButton("Import")
        self._import_mesh_btn.setToolTip("Import a mesh file from anywhere on disk")
        self._import_mesh_btn.clicked.connect(self._on_import_mesh)
        inv_btns.addWidget(self._import_mesh_btn)
        self._refresh_inv_btn = QPushButton("Refresh")
        self._refresh_inv_btn.clicked.connect(self._refresh_inventory)
        inv_btns.addWidget(self._refresh_inv_btn)
        self._delete_item_btn = QPushButton("Delete")
        self._delete_item_btn.clicked.connect(self._on_delete_item)
        inv_btns.addWidget(self._delete_item_btn)
        inv_lay.addLayout(inv_btns)
        lv.addWidget(inv_group)

        # Scene
        scene_group = QGroupBox("Scene")
        sg_lay = QVBoxLayout(scene_group)
        self._scene_list = QListWidget()
        self._scene_list.setMaximumHeight(100)
        self._scene_list.currentRowChanged.connect(self._on_scene_selection)
        sg_lay.addWidget(self._scene_list)
        scene_btns = QHBoxLayout()
        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._on_remove_from_scene)
        scene_btns.addWidget(self._remove_btn)
        self._clear_scene_btn = QPushButton("Clear Scene")
        self._clear_scene_btn.clicked.connect(self._on_clear_scene)
        scene_btns.addWidget(self._clear_scene_btn)
        sg_lay.addLayout(scene_btns)
        lv.addWidget(scene_group)

        # Per-mesh transforms
        xform_group = QGroupBox("Transform")
        xf_lay = QVBoxLayout(xform_group)
        xf_lay.setContentsMargins(4, 4, 4, 4)
        xf_lay.setSpacing(2)

        def _spin(lo, hi, val, step, decimals=1, width=70):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setValue(val)
            s.setSingleStep(step)
            s.setDecimals(decimals)
            s.setFixedWidth(width)
            return s

        pos_row = QHBoxLayout()
        pos_row.addWidget(QLabel("Pos:"))
        self._pos_x = _spin(-10, 10, 0, 0.1)
        self._pos_y = _spin(-10, 10, 0, 0.1)
        self._pos_z = _spin(-10, 10, 0, 0.1)
        for label, spin in [("X", self._pos_x), ("Y", self._pos_y), ("Z", self._pos_z)]:
            pos_row.addWidget(QLabel(label))
            pos_row.addWidget(spin)
            spin.valueChanged.connect(self._on_transform_changed)
        pos_row.addStretch()
        xf_lay.addLayout(pos_row)

        rot_row = QHBoxLayout()
        rot_row.addWidget(QLabel("Rot:"))
        self._rot_x = _spin(-180, 180, 0, 5, 0)
        self._rot_y = _spin(-180, 180, 0, 5, 0)
        self._rot_z = _spin(-180, 180, 0, 5, 0)
        for label, spin in [("X", self._rot_x), ("Y", self._rot_y), ("Z", self._rot_z)]:
            rot_row.addWidget(QLabel(label))
            rot_row.addWidget(spin)
            spin.valueChanged.connect(self._on_transform_changed)
        rot_row.addStretch()
        xf_lay.addLayout(rot_row)

        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Scale:"))
        self._scale = _spin(0.1, 10, 1.0, 0.1)
        self._scale.valueChanged.connect(self._on_transform_changed)
        scale_row.addWidget(self._scale)
        scale_row.addStretch()
        xf_lay.addLayout(scale_row)

        lv.addWidget(xform_group)

        # Lighting controls
        light_group = QGroupBox("Lighting")
        lt_lay = QVBoxLayout(light_group)
        lt_lay.setContentsMargins(4, 4, 4, 4)
        lt_lay.setSpacing(2)

        az_row = QHBoxLayout()
        az_row.addWidget(QLabel("Azimuth:"))
        self._light_az = QSlider(Qt.Orientation.Horizontal)
        self._light_az.setRange(0, 360)
        self._light_az.setValue(45)
        self._light_az.valueChanged.connect(self._on_light_changed)
        az_row.addWidget(self._light_az)
        self._light_az_label = QLabel("45°")
        self._light_az_label.setFixedWidth(30)
        az_row.addWidget(self._light_az_label)
        lt_lay.addLayout(az_row)

        el_row = QHBoxLayout()
        el_row.addWidget(QLabel("Elevation:"))
        self._light_el = QSlider(Qt.Orientation.Horizontal)
        self._light_el.setRange(-90, 90)
        self._light_el.setValue(45)
        self._light_el.valueChanged.connect(self._on_light_changed)
        el_row.addWidget(self._light_el)
        self._light_el_label = QLabel("45°")
        self._light_el_label.setFixedWidth(30)
        el_row.addWidget(self._light_el_label)
        lt_lay.addLayout(el_row)

        int_row = QHBoxLayout()
        int_row.addWidget(QLabel("Intensity:"))
        self._light_int = QSlider(Qt.Orientation.Horizontal)
        self._light_int.setRange(0, 200)
        self._light_int.setValue(80)
        self._light_int.valueChanged.connect(self._on_light_changed)
        int_row.addWidget(self._light_int)
        int_row.addWidget(QLabel("Ambient:"))
        self._light_amb = QSlider(Qt.Orientation.Horizontal)
        self._light_amb.setRange(0, 100)
        self._light_amb.setValue(30)
        self._light_amb.valueChanged.connect(self._on_light_changed)
        int_row.addWidget(self._light_amb)
        lt_lay.addLayout(int_row)

        lv.addWidget(light_group)

        # Camera presets
        cam_group = QGroupBox("Camera")
        cam_lay = QVBoxLayout(cam_group)
        cam_lay.setContentsMargins(4, 4, 4, 4)
        cam_row = QHBoxLayout()
        self._cam_preset = QComboBox()
        self._cam_preset.addItems([
            "Front", "Right", "Back", "Left",
            "Top", "3/4 Right", "3/4 Left",
        ])
        self._cam_preset.setFixedWidth(100)
        cam_row.addWidget(self._cam_preset)
        self._cam_go_btn = QPushButton("Go")
        self._cam_go_btn.setFixedWidth(36)
        self._cam_go_btn.clicked.connect(self._on_camera_preset)
        cam_row.addWidget(self._cam_go_btn)
        self._cam_save_btn = QPushButton("Save")
        self._cam_save_btn.setToolTip("Save current camera as a named position")
        self._cam_save_btn.setFixedWidth(42)
        self._cam_save_btn.clicked.connect(self._on_save_camera)
        cam_row.addWidget(self._cam_save_btn)
        cam_row.addStretch()
        cam_lay.addLayout(cam_row)
        lv.addWidget(cam_group)

        # Interaction mode
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Orbit", "Translate", "Rotate", "Scale"])
        self._mode_combo.setFixedWidth(100)
        self._mode_combo.setToolTip(
            "Orbit: rotate/pan/zoom camera\n"
            "Translate: drag gizmo arrows to move selected mesh\n"
            "Rotate: drag gizmo rings to rotate selected mesh\n"
            "Scale: drag gizmo handles to scale selected mesh"
        )
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode_combo)
        mode_row.addStretch()
        lv.addLayout(mode_row)

        # Viewport options
        view_row = QHBoxLayout()
        self._wireframe_cb = QCheckBox("Wireframe")
        view_row.addWidget(self._wireframe_cb)
        self._grid_cb = QCheckBox("Grid")
        self._grid_cb.setChecked(True)
        view_row.addWidget(self._grid_cb)
        self._snap_views_btn = QPushButton("4 Views")
        self._snap_views_btn.setToolTip("Render front/right/back/left and save to project")
        self._snap_views_btn.clicked.connect(self._on_render_views)
        view_row.addWidget(self._snap_views_btn)
        self._lora_dataset_btn = QPushButton("LoRA Dataset")
        self._lora_dataset_btn.setToolTip(
            "Render 16 angles, create training dataset, auto-caption with Qwen VL"
        )
        self._lora_dataset_btn.clicked.connect(self._on_render_lora_dataset)
        view_row.addWidget(self._lora_dataset_btn)
        lv.addLayout(view_row)

        lv.addStretch()
        splitter.addWidget(left)

        # ── Right panel: viewport + send buttons ──────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        self._viewport = GLViewportWidget()
        rv.addWidget(self._viewport, 1)

        self._wireframe_cb.toggled.connect(lambda c: self._viewport.set_wireframe(c))
        self._grid_cb.toggled.connect(lambda c: self._viewport.set_grid(c))

        self._status = QLabel("")
        rv.addWidget(self._status)

        # Send-to dropdown
        btn_row = QHBoxLayout()
        self._send_btn = QPushButton("Send Composite \u25bc")
        self._send_btn.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 14px;"
            " color: #fff; background: #0078d4; border-radius: 3px;"
            " padding: 6px 16px; } "
            "QPushButton:hover { background: #005a9e; }")
        self._send_btn.setToolTip("Capture viewport composited over background and send to another tab")
        self._send_btn.clicked.connect(self._show_send_menu)
        btn_row.addWidget(self._send_btn)
        btn_row.addStretch()
        rv.addLayout(btn_row)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        layout.addWidget(splitter, 1)

        # ── Animation Timeline (bottom) ──────────────────────────────────
        from sdqt.widgets.animation_timeline import AnimationTimelineWidget
        self._timeline = AnimationTimelineWidget()
        self._timeline.frame_changed.connect(self._on_animation_frame)
        self._timeline.keyframe_requested.connect(self._on_keyframe_request)
        self._timeline.render_requested.connect(self._on_render_sequence)
        layout.addWidget(self._timeline)

        # Render state
        self._render_timer = None
        self._render_frame_idx = 0
        self._render_frames_dir = None

    # ── Public API ────────────────────────────────────────────────────────

    def load_source(self, path: str) -> None:
        """Accept an image as the background plate."""
        if not path or not Path(path).is_file():
            return
        self._bg_image_path = path
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self._bg_preview.setPixmap(
                pixmap.scaled(self._bg_preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation))
            self._bg_preview.setStyleSheet("background: #1a1a1a;")
            self._clear_bg_btn.setVisible(True)
        if _3D_AVAILABLE:
            self._viewport.set_background_image(path)
        self._show_status(f"Background: {Path(path).name}")

    def load_mesh(self, path: str) -> None:
        """Load a mesh file directly into the viewport."""
        self._add_mesh_to_scene(path)

    # ── Background ────────────────────────────────────────────────────────

    @Slot()
    def _on_clear_bg(self) -> None:
        self._bg_image_path = None
        self._bg_preview.clear()
        self._bg_preview.setText("No background image")
        self._bg_preview.setStyleSheet("background: #1a1a1a; border: 1px dashed #444;")
        self._clear_bg_btn.setVisible(False)
        if _3D_AVAILABLE:
            self._viewport.set_background_image(None)

    # ── Mesh Generation ──────────────────────────────────────────────────

    @Slot()
    def _on_generate_from_bg(self) -> None:
        """Generate a 3D mesh from the background image via TripoSR."""
        if not self._bg_image_path:
            self._show_status("No background image loaded.")
            return

        from sdqt.models.manager import check_and_prompt_download
        if not check_and_prompt_download(
            "3d_modeling", self.state.model_registry, self,
            on_progress=lambda f, d: self._show_status(d),
        ):
            return

        self._show_status("Loading TripoSR pipeline...")
        try:
            pipeline = self.state.load_triposr_pipeline()
        except Exception as exc:
            self._show_status(f"Failed to load TripoSR: {exc}")
            return

        meshes_dir = self.project_path / "meshes"
        meshes_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(self._bg_image_path).stem
        fmt = self._mesh_fmt.currentText()
        output = str(meshes_dir / f"{stem}.{fmt}")

        from sdqt.workers.model3d import MeshGenerationWorker
        self._gen_from_bg_btn.setVisible(False)
        self._gen_abort_btn.setVisible(True)

        worker = MeshGenerationWorker(
            pipeline=pipeline,
            image_path=self._bg_image_path,
            output_path=output,
            resolution=self._mesh_res.value(),
            output_format=fmt,
            remove_bg=self._remove_bg_cb.isChecked(),
            parent=self,
        )
        worker.progress.connect(lambda f, d: self._show_status(d))
        worker.finished_ok.connect(self._on_mesh_generated)
        worker.error.connect(self._on_mesh_error)
        worker.finished.connect(worker.deleteLater)
        self._mesh_worker = worker
        worker.start()

    def _on_mesh_generated(self, path: str) -> None:
        self._gen_from_bg_btn.setVisible(True)
        self._gen_abort_btn.setVisible(False)
        self._show_status(f"Mesh saved: {Path(path).name}")
        self._refresh_inventory()
        # Auto-add to scene
        self._add_mesh_to_scene(path)

    def _on_mesh_error(self, msg: str) -> None:
        self._gen_from_bg_btn.setVisible(True)
        self._gen_abort_btn.setVisible(False)
        self._show_status(f"Mesh generation failed: {msg}")

    @Slot()
    def _on_abort_generate(self) -> None:
        if self._mesh_worker:
            self._mesh_worker.abort()

    # ── Inventory ─────────────────────────────────────────────────────────

    def _refresh_inventory(self) -> None:
        self._inventory.clear()
        meshes_dir = self.project_path / "meshes"
        if not meshes_dir.is_dir():
            return
        for f in sorted(meshes_dir.iterdir()):
            if f.suffix.lower() in _MESH_EXTS:
                item = QListWidgetItem(f.name)
                item.setData(Qt.ItemDataRole.UserRole, str(f))
                self._inventory.addItem(item)

    @Slot()
    def _on_import_mesh(self) -> None:
        """Import a mesh file from anywhere on disk into project/meshes/."""
        from PySide6.QtWidgets import QFileDialog
        import shutil
        path, _ = QFileDialog.getOpenFileName(
            self, "Import 3D Mesh", "",
            "3D Meshes (*.obj *.glb *.gltf *.ply *.stl)")
        if not path:
            return
        meshes_dir = self.project_path / "meshes"
        meshes_dir.mkdir(parents=True, exist_ok=True)
        dest = meshes_dir / Path(path).name
        if dest.exists():
            stem, suffix, idx = Path(path).stem, Path(path).suffix, 1
            while dest.exists():
                dest = meshes_dir / f"{stem}_{idx}{suffix}"
                idx += 1
        shutil.copy2(path, dest)
        self._refresh_inventory()
        self._show_status(f"Imported: {dest.name}")

    @Slot(QListWidgetItem)
    def _on_inventory_add(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self._add_mesh_to_scene(path)

    @Slot()
    def _on_delete_item(self) -> None:
        item = self._inventory.currentItem()
        if not item:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and Path(path).is_file():
            Path(path).unlink()
            self._refresh_inventory()
            self._show_status(f"Deleted: {Path(path).name}")

    # ── Scene ─────────────────────────────────────────────────────────────

    def _add_mesh_to_scene(self, path: str) -> None:
        if not _3D_AVAILABLE:
            return
        ok = self._viewport.load_mesh(path)
        if ok:
            name = Path(path).name
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, path)
            self._scene_list.addItem(item)
            self._scene_list.setCurrentRow(self._scene_list.count() - 1)
            self._show_status(f"Added: {name}")
        else:
            self._show_status("Failed to load mesh.")

    @Slot()
    def _on_remove_from_scene(self) -> None:
        row = self._scene_list.currentRow()
        if row < 0:
            return
        self._scene_list.takeItem(row)
        # Remove from viewport
        if _3D_AVAILABLE and hasattr(self._viewport, "_canvas"):
            canvas = self._viewport._canvas
            if canvas and 0 <= row < len(canvas._meshes):
                canvas._meshes.pop(row)
                canvas.update()
        self._show_status("Removed from scene.")

    @Slot()
    def _on_clear_scene(self) -> None:
        self._scene_list.clear()
        if _3D_AVAILABLE:
            self._viewport.clear()
        self._show_status("Scene cleared.")

    # ── Per-mesh transforms ───────────────────────────────────────────────

    @Slot()
    def _on_transform_changed(self) -> None:
        """Apply transform spinner values to the selected mesh."""
        if not _3D_AVAILABLE:
            return
        row = self._scene_list.currentRow()
        canvas = getattr(self._viewport, "_canvas", None)
        if not canvas or row < 0 or row >= len(canvas._meshes):
            return
        mesh = canvas._meshes[row]
        mesh["position"] = [self._pos_x.value(), self._pos_y.value(), self._pos_z.value()]
        mesh["rotation"] = [self._rot_x.value(), self._rot_y.value(), self._rot_z.value()]
        mesh["user_scale"] = self._scale.value()
        # Apply uniform scale to the mesh vertices would be destructive;
        # instead we'll use the scale in the render transform.
        # The viewport _draw_mesh already reads position/rotation.
        # We need to add scale support to the viewport's draw code.
        canvas.update()

    # ── Send composite ────────────────────────────────────────────────────

    @Slot()
    def _show_send_menu(self) -> None:
        menu = QMenu(self)
        targets = [
            ("img2img", "Img2Img"),
            ("inpaint", "Inpaint"),
            ("imgedit", "Draw"),
            ("faceswap", "Face Swap"),
            ("img2vid", "Img2Vid"),
            ("bodydouble_src", "Body Double (source)"),
            ("repose_src", "RePose (source)"),
            ("cropzoom", "Crop/Zoom"),
        ]
        for key, label in targets:
            action = menu.addAction(label)
            if key in DISABLED_TARGETS:
                action.setEnabled(False)
            else:
                action.triggered.connect(lambda checked, k=key: self._do_send(k))
        menu.exec(self._send_btn.mapToGlobal(self._send_btn.rect().bottomLeft()))

    def _do_send(self, target: str) -> None:
        """Capture viewport + BG composite and emit send signal."""
        if not _3D_AVAILABLE:
            self._show_status("3D viewport not available.")
            return
        canvas = getattr(self._viewport, "_canvas", None)
        if canvas is None:
            self._show_status("OpenGL viewport not available.")
            return

        try:
            qimage = canvas.grabFramebuffer()

            if self._bg_image_path and Path(self._bg_image_path).is_file():
                bg = QImage(self._bg_image_path)
                bg = bg.scaled(qimage.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                               Qt.TransformationMode.SmoothTransformation)
                painter = QPainter(bg)
                painter.drawImage(0, 0, qimage)
                painter.end()
                qimage = bg

            tmp = tempfile.NamedTemporaryFile(suffix=".png", prefix="3d_composite_", delete=False)
            qimage.save(tmp.name, "PNG")
            tmp.close()

            self.send_composite.emit(f"{target}|{tmp.name}")
            self._show_status(f"Composite sent to {target}.")
        except Exception as exc:
            self._show_status(f"Capture error: {exc}")

    # ── Project change ────────────────────────────────────────────────────

    # ── Interaction mode ─────────────────────────────────────────────────

    def _on_mode_changed(self, text: str) -> None:
        mode_map = {"Orbit": "orbit", "Translate": "translate", "Rotate": "rotate", "Scale": "scale"}
        mode = mode_map.get(text, "orbit")
        if _3D_AVAILABLE:
            self._viewport.set_interaction_mode(mode)

    @Slot(int)
    def _on_scene_selection(self, row: int) -> None:
        """When a mesh is selected in the scene list, sync to viewport + load transforms."""
        if _3D_AVAILABLE:
            self._viewport.set_selected_mesh(row)

        if not _3D_AVAILABLE or row < 0:
            return
        canvas = getattr(self._viewport, "_canvas", None)
        if not canvas or row >= len(canvas._meshes):
            return
        mesh = canvas._meshes[row]
        pos = mesh.get("position", [0, 0, 0])
        rot = mesh.get("rotation", [0, 0, 0])
        for spin in (self._pos_x, self._pos_y, self._pos_z,
                     self._rot_x, self._rot_y, self._rot_z, self._scale):
            spin.blockSignals(True)
        self._pos_x.setValue(pos[0])
        self._pos_y.setValue(pos[1])
        self._pos_z.setValue(pos[2])
        self._rot_x.setValue(rot[0])
        self._rot_y.setValue(rot[1])
        self._rot_z.setValue(rot[2])
        self._scale.setValue(mesh.get("user_scale", 1.0))
        for spin in (self._pos_x, self._pos_y, self._pos_z,
                     self._rot_x, self._rot_y, self._rot_z, self._scale):
            spin.blockSignals(False)

    # ── Camera presets ────────────────────────────────────────────────────

    _CAMERA_PRESETS = {
        "Front":     {"rot_x": -20, "rot_y": 0,   "pan_x": 0, "pan_y": 0, "zoom": 3.0},
        "Right":     {"rot_x": -20, "rot_y": 90,  "pan_x": 0, "pan_y": 0, "zoom": 3.0},
        "Back":      {"rot_x": -20, "rot_y": 180, "pan_x": 0, "pan_y": 0, "zoom": 3.0},
        "Left":      {"rot_x": -20, "rot_y": 270, "pan_x": 0, "pan_y": 0, "zoom": 3.0},
        "Top":       {"rot_x": -89, "rot_y": 0,   "pan_x": 0, "pan_y": 0, "zoom": 3.0},
        "3/4 Right": {"rot_x": -25, "rot_y": 45,  "pan_x": 0, "pan_y": 0, "zoom": 3.0},
        "3/4 Left":  {"rot_x": -25, "rot_y": -45, "pan_x": 0, "pan_y": 0, "zoom": 3.0},
    }

    @Slot()
    def _on_camera_preset(self) -> None:
        name = self._cam_preset.currentText()
        state = self._CAMERA_PRESETS.get(name)
        if state and _3D_AVAILABLE and self._viewport._canvas:
            self._viewport._canvas.set_camera_state(state)

    @Slot()
    def _on_save_camera(self) -> None:
        """Save current camera position as a named preset in the dropdown."""
        if not _3D_AVAILABLE or not self._viewport._canvas:
            return
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save Camera", "Camera name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        state = self._viewport._canvas.get_camera_state()
        self._CAMERA_PRESETS[name] = state
        # Add to combo if not already there
        if self._cam_preset.findText(name) < 0:
            self._cam_preset.addItem(name)
        self._cam_preset.setCurrentText(name)
        self._show_status(f"Camera saved: {name}")

    # ── Scene save/load ───────────────────────────────────────────────────

    def _save_scene(self) -> None:
        """Auto-save scene state (meshes, camera, keyframes) to project JSON."""
        import json
        if not _3D_AVAILABLE:
            return
        scene_file = self.project_path / "meshes" / "scene.json"
        scene_file.parent.mkdir(parents=True, exist_ok=True)

        canvas = getattr(self._viewport, "_canvas", None)
        data: dict = {
            "background": self._bg_image_path or "",
            "camera": canvas.get_camera_state() if canvas else {},
            "meshes": [],
            "animation": {
                "fps": self._timeline.animation.fps,
                "duration": self._timeline.animation.duration_seconds,
                "tracks": {},
            },
        }

        # Save mesh references + transforms
        for i in range(self._scene_list.count()):
            item = self._scene_list.item(i)
            path = item.data(Qt.ItemDataRole.UserRole) or ""
            mesh_data = {}
            if canvas and i < len(canvas._meshes):
                m = canvas._meshes[i]
                mesh_data = {
                    "path": path,
                    "position": m.get("position", [0, 0, 0]),
                    "rotation": m.get("rotation", [0, 0, 0]),
                    "scale": m.get("user_scale", 1.0),
                }
            else:
                mesh_data = {"path": path}
            data["meshes"].append(mesh_data)

        # Save animation keyframes
        for target, track in self._timeline.animation.tracks.items():
            kfs = []
            for kf in track.keyframes:
                kfs.append({
                    "frame": kf.frame,
                    "rot_x": kf.rot_x, "rot_y": kf.rot_y,
                    "pan_x": kf.pan_x, "pan_y": kf.pan_y,
                    "zoom": kf.zoom,
                    "pos_x": kf.pos_x, "pos_y": kf.pos_y, "pos_z": kf.pos_z,
                    "rot_mx": kf.rot_mx, "rot_my": kf.rot_my, "rot_mz": kf.rot_mz,
                    "scale": kf.scale,
                })
            data["animation"]["tracks"][target] = kfs

        scene_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_scene(self) -> None:
        """Restore scene state from project JSON."""
        import json
        if not _3D_AVAILABLE:
            return
        scene_file = self.project_path / "meshes" / "scene.json"
        if not scene_file.is_file():
            return

        try:
            data = json.loads(scene_file.read_text(encoding="utf-8"))
        except Exception:
            return

        # Background
        bg = data.get("background", "")
        if bg and Path(bg).is_file():
            self.load_source(bg)

        # Camera
        cam = data.get("camera", {})
        if cam and self._viewport._canvas:
            self._viewport._canvas.set_camera_state(cam)

        # Meshes
        self._scene_list.clear()
        if self._viewport._canvas:
            self._viewport._canvas.clear_meshes()
        for mesh_data in data.get("meshes", []):
            path = mesh_data.get("path", "")
            if path and Path(path).is_file():
                self._add_mesh_to_scene(path)
                # Apply saved transforms
                canvas = self._viewport._canvas
                if canvas and canvas._meshes:
                    m = canvas._meshes[-1]
                    m["position"] = mesh_data.get("position", [0, 0, 0])
                    m["rotation"] = mesh_data.get("rotation", [0, 0, 0])
                    m["user_scale"] = mesh_data.get("scale", 1.0)

        # Animation
        from sdqt.widgets.animation_timeline import Keyframe
        anim_data = data.get("animation", {})
        anim = self._timeline.animation
        anim.clear()
        anim.fps = anim_data.get("fps", 24)
        anim.duration_seconds = anim_data.get("duration", 5.0)
        self._timeline._fps_spin.setValue(anim.fps)
        self._timeline._dur_spin.setValue(int(anim.duration_seconds))

        for target, kf_list in anim_data.get("tracks", {}).items():
            track = anim.get_or_create_track(target)
            for kf_data in kf_list:
                track.add_keyframe(Keyframe(**kf_data))

        self._timeline._bar.update()
        if self._viewport._canvas:
            self._viewport._canvas.update()

    # ── Animation ─────────────────────────────────────────────────────────

    def _on_animation_frame(self, frame: int) -> None:
        """Apply animation state at the given frame to viewport."""
        if not _3D_AVAILABLE:
            return
        anim = self._timeline.animation
        states = anim.evaluate(frame)

        # Apply camera track
        cam_kf = states.get("camera")
        if cam_kf and self._viewport._canvas:
            self._viewport._canvas.set_camera_state({
                "rot_x": cam_kf.rot_x,
                "rot_y": cam_kf.rot_y,
                "pan_x": cam_kf.pan_x,
                "pan_y": cam_kf.pan_y,
                "zoom": cam_kf.zoom,
            })

        # Apply mesh tracks
        canvas = getattr(self._viewport, "_canvas", None)
        if canvas:
            for target, kf in states.items():
                if not target.startswith("mesh_"):
                    continue
                idx = int(target.split("_")[1])
                if 0 <= idx < len(canvas._meshes):
                    m = canvas._meshes[idx]
                    m["position"] = [kf.pos_x, kf.pos_y, kf.pos_z]
                    m["rotation"] = [kf.rot_mx, kf.rot_my, kf.rot_mz]
                    m["user_scale"] = kf.scale
            canvas.update()

    def _on_keyframe_request(self, frame: int) -> None:
        """Add a keyframe at the given frame, or apply preset if frame == -1."""
        if not _3D_AVAILABLE:
            return

        if frame == -1:
            # Apply preset
            self._apply_animation_preset()
            return

        from sdqt.widgets.animation_timeline import Keyframe
        anim = self._timeline.animation
        canvas = getattr(self._viewport, "_canvas", None)

        # Camera keyframe
        if canvas:
            cam = canvas.get_camera_state()
            track = anim.get_or_create_track("camera")
            track.add_keyframe(Keyframe(
                frame=frame,
                rot_x=cam["rot_x"], rot_y=cam["rot_y"],
                pan_x=cam["pan_x"], pan_y=cam["pan_y"],
                zoom=cam["zoom"],
            ))

        # Mesh keyframes (for all loaded meshes)
        if canvas:
            for idx, m in enumerate(canvas._meshes):
                target = f"mesh_{idx}"
                track = anim.get_or_create_track(target)
                pos = m.get("position", [0, 0, 0])
                rot = m.get("rotation", [0, 0, 0])
                scale = m.get("user_scale", 1.0)
                track.add_keyframe(Keyframe(
                    frame=frame,
                    pos_x=pos[0], pos_y=pos[1], pos_z=pos[2],
                    rot_mx=rot[0], rot_my=rot[1], rot_mz=rot[2],
                    scale=scale,
                ))

        self._timeline._bar.update()
        self._show_status(f"Keyframe added at frame {frame}")

    def _apply_animation_preset(self) -> None:
        """Apply the selected animation preset from the timeline dropdown."""
        from sdqt.widgets.animation_timeline import (
            preset_turntable, preset_dolly_in, preset_object_spin,
            preset_reveal, preset_orbit_rise,
        )
        preset = self._timeline.preset_combo.currentText()
        anim = self._timeline.animation
        canvas = getattr(self._viewport, "_canvas", None)
        cam = canvas.get_camera_state() if canvas else {}

        if preset == "Turntable":
            preset_turntable(anim, cam)
        elif preset == "Dolly In":
            preset_dolly_in(anim, cam)
        elif preset == "Object Spin":
            preset_object_spin(anim, mesh_idx=max(0, self._scene_list.currentRow()))
        elif preset == "Reveal":
            preset_reveal(anim, cam)
        elif preset == "Orbit + Rise":
            preset_orbit_rise(anim, cam)
        else:
            return

        self._timeline._bar.update()
        self._show_status(f"Preset applied: {preset}")

    # ── Frame sequence rendering ──────────────────────────────────────────

    @Slot()
    def _on_render_sequence(self) -> None:
        """Render the animation to a frame sequence, then assemble to video."""
        if not _3D_AVAILABLE:
            self._show_status("3D viewport not available.")
            return
        anim = self._timeline.animation
        if not anim.has_keyframes():
            self._show_status("No keyframes. Add keyframes or apply a preset first.")
            return

        import tempfile
        self._render_frames_dir = Path(tempfile.mkdtemp(prefix="3d_render_"))
        self._render_frame_idx = 0
        self._render_total = anim.duration_frames

        # Start rendering frames on the main thread (GL requires it)
        from PySide6.QtCore import QTimer
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._render_next_frame)
        self._render_timer.start(0)  # as fast as possible
        self._show_status("Rendering frames...")

    def _render_next_frame(self) -> None:
        """Render one frame and save to disk. Called by QTimer on main thread."""
        anim = self._timeline.animation
        idx = self._render_frame_idx

        if idx >= self._render_total:
            # All frames rendered — assemble video
            self._render_timer.stop()
            self._render_timer = None
            self._assemble_video()
            return

        # Evaluate animation at this frame
        self._on_animation_frame(idx)

        # Grab framebuffer
        canvas = self._viewport._canvas
        if canvas is None:
            self._render_timer.stop()
            self._show_status("Render failed — viewport not available.")
            return

        qimage = canvas.grabFramebuffer()

        # Composite over background
        from sdqt.workers.render3d import composite_frame
        qimage = composite_frame(qimage, self._bg_image_path)

        # Save frame
        frame_path = self._render_frames_dir / f"frame_{idx:05d}.png"
        qimage.save(str(frame_path), "PNG")

        self._render_frame_idx += 1
        self._show_status(f"Rendering frame {idx + 1}/{self._render_total}")

    def _assemble_video(self) -> None:
        """Assemble rendered frames into an MP4 video."""
        import subprocess

        out_dir = self.project_path / "clips" / "3d_renders"
        out_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = str(out_dir / f"3d_render_{ts}.mp4")

        fps = self._timeline.animation.fps
        self._show_status("Assembling video with ffmpeg...")

        try:
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(self._render_frames_dir / "frame_%05d.png"),
                *_codec_args(),
                *_pix_fmt(_enc_name()),
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2:in_range=full:out_range=full",
                output,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                self._show_status(f"ffmpeg error: {result.stderr[:200]}")
                return

            self._show_status(f"Rendered: {Path(output).name}")
            self._last_render_path = output

            # Show send menu for the rendered video
            self._show_render_send_menu(output)

        except Exception as exc:
            self._show_status(f"Assembly error: {exc}")

    def _show_render_send_menu(self, video_path: str) -> None:
        """After rendering, offer to send the video to Wan or Timeline."""
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, "Render Complete",
            f"Animation rendered to:\n{Path(video_path).name}\n\n"
            "Send to Img2Vid (Mode 3) for AI refinement?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.send_composite.emit(f"img2vid|{video_path}")

    # ── Lighting ───────────────────────────────────────────────────────────

    def _on_light_changed(self) -> None:
        az = self._light_az.value()
        el = self._light_el.value()
        intensity = self._light_int.value() / 100.0
        ambient = self._light_amb.value() / 100.0
        self._light_az_label.setText(f"{az}°")
        self._light_el_label.setText(f"{el}°")
        if _3D_AVAILABLE:
            self._viewport.set_light(az, el, intensity=intensity, ambient=ambient)

    # ── Texture Refinement ─────────────────────────────────────────────────

    @Slot()
    def _on_refine_texture(self) -> None:
        """Run iterative texture refinement on the selected mesh."""
        if not _3D_AVAILABLE or not self._viewport._canvas:
            self._show_status("Viewport not available.")
            return
        if self._viewport.mesh_count == 0:
            self._show_status("No mesh loaded. Generate or import a mesh first.")
            return

        mesh_idx = max(0, self._viewport._canvas._selected_mesh_idx)
        mesh_path = self._viewport.get_mesh_source_path(mesh_idx)
        if not mesh_path or not Path(mesh_path).is_file():
            self._show_status("Mesh source file not found — import or re-generate.")
            return

        prompt = self._tex_prompt.toPlainText().strip()
        if not prompt:
            self._show_status("Enter a texture prompt describing the desired surface.")
            return

        neg_prompt = self._tex_neg_prompt.text().strip()
        denoise = self._tex_denoise.value()
        tex_size = int(self._tex_size_combo.currentText())
        n_views = self._tex_views.value()

        # Build angle list
        from sdqt.workers.texture_refine import DEFAULT_ANGLES
        angles = DEFAULT_ANGLES[:n_views]

        render_func = self._viewport.render_single_view

        from sdqt.workers.texture_refine import TextureRefineWorker
        worker = TextureRefineWorker(
            mesh_path=mesh_path,
            render_func=render_func,
            app_state=self.state,
            prompt=prompt,
            negative_prompt=neg_prompt,
            denoise_strength=denoise,
            tex_size=tex_size,
            render_size=512,
            angles=angles,
            parent=self,
        )
        worker.progress.connect(self._on_tex_progress)
        worker.status.connect(lambda s: self._show_status(s))
        worker.finished_ok.connect(self._on_tex_done)
        worker.error.connect(self._on_tex_error)
        worker.finished.connect(worker.deleteLater)

        self._tex_worker = worker
        self._tex_refine_btn.setVisible(False)
        self._tex_abort_btn.setVisible(True)
        self._tex_progress.setVisible(True)
        self._tex_progress.setValue(0)
        worker.start()

    @Slot()
    def _on_abort_refine(self) -> None:
        if self._tex_worker and self._tex_worker.isRunning():
            self._tex_worker.abort()

    @Slot(float, str)
    def _on_tex_progress(self, frac: float, desc: str) -> None:
        self._tex_progress.setValue(int(frac * 100))

    @Slot(object)
    def _on_tex_done(self, tex_path) -> None:
        self._tex_refine_btn.setVisible(True)
        self._tex_abort_btn.setVisible(False)
        self._tex_progress.setVisible(False)
        self._tex_worker = None

        # Reload the texture in the viewport
        mesh_idx = max(0, self._viewport._canvas._selected_mesh_idx)
        self._viewport.reload_texture(mesh_idx)

        self._show_status(f"Texture refined: {Path(str(tex_path)).name}")

    @Slot(str)
    def _on_tex_error(self, msg: str) -> None:
        self._tex_refine_btn.setVisible(True)
        self._tex_abort_btn.setVisible(False)
        self._tex_progress.setVisible(False)
        self._tex_worker = None
        self._show_status(f"Texture refinement error: {msg}")

    # ── Multi-angle render ────────────────────────────────────────────────

    @Slot()
    def _on_render_views(self) -> None:
        """Render front/right/back/left views and save to project/meshes/."""
        if not _3D_AVAILABLE or not self._viewport._canvas:
            self._show_status("Viewport not available.")
            return
        views_dir = self.project_path / "meshes" / "views"
        views_dir.mkdir(parents=True, exist_ok=True)

        angle_names = [
            ((-20, 0), "front"),
            ((-20, 90), "right"),
            ((-20, 180), "back"),
            ((-20, 270), "left"),
        ]
        angles = [a for a, _ in angle_names]
        images = self._viewport.render_views(angles)

        for img, ((_, _), name) in zip(images, angle_names):
            path = views_dir / f"view_{name}.png"
            img.save(str(path), "PNG")

        self._show_status(f"Rendered {len(images)} views to {views_dir.name}/")

    @Slot()
    def _on_render_lora_dataset(self) -> None:
        """Render 16 angles with BG compositing, create a LoRA training dataset."""
        if not _3D_AVAILABLE or not self._viewport._canvas:
            self._show_status("Viewport not available.")
            return

        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self, "LoRA Dataset Name",
            "Enter a name for this 3D model dataset:",
        )
        if not ok or not name.strip():
            return
        name = name.strip().replace(" ", "_").lower()

        dataset_dir = self.project_path / "lora_datasets" / f"{name}_dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # 16 angles: 2 elevation tiers × 8 azimuths
        angles = []
        angle_labels = []
        for el in (-10, -30):
            for i, az in enumerate(range(0, 360, 45)):
                angles.append((el, az))
                tier = "high" if el == -10 else "low"
                angle_labels.append(f"{name}_{tier}_{i + 1:02d}")

        self._show_status(f"Rendering {len(angles)} views...")
        images = self._viewport.render_views(angles)

        from sdqt.workers.render3d import composite_frame as _cf
        for img, label in zip(images, angle_labels):
            # Convert QImage for compositing
            from PySide6.QtGui import QImage
            if isinstance(img, QImage):
                qimg = img
            else:
                # PIL Image → save and reload as QImage
                tmp_p = dataset_dir / f"_tmp_{label}.png"
                img.save(str(tmp_p), "PNG")
                qimg = QImage(str(tmp_p))
                tmp_p.unlink(missing_ok=True)

            composited = _cf(qimg, self._bg_image_path)
            out_path = dataset_dir / f"{label}.png"
            composited.save(str(out_path), "PNG")

        self._show_status(f"Rendered {len(images)} views. Sending to LoRA training...")
        self.lora_dataset_ready.emit(str(dataset_dir))

    # ── Project change ────────────────────────────────────────────────────

    def on_project_changed(self, project_name: str) -> None:
        # Save current project scene before switching
        if _3D_AVAILABLE and hasattr(self, "_timeline"):
            try:
                self._save_scene()
            except Exception:
                pass
        super().on_project_changed(project_name)
        if _3D_AVAILABLE:
            self._refresh_inventory()
            self._scene_list.clear()
            if self._viewport._canvas:
                self._viewport._canvas.clear_meshes()
            self._timeline.reset()
            self._load_scene()
