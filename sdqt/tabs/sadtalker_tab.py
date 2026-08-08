"""SadTalker tab — audio-driven talking head, full control surface."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from supremediffusion.config.project_config import ProjectConfig

from sdqt.state import AppState
from sdqt.widgets.collapsible_section import CollapsibleSection
from sdqt.widgets.file_drop import FileDropWidget
from sdqt.widgets.image_drop import ImageDropWidget
from sdqt.widgets.video_player import VideoPlayerWidget
from sdqt.workers.sadtalker import SadTalkerWorker

from .base import BaseTab

logger = logging.getLogger(__name__)

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


class SadTalkerTab(BaseTab):
    """Full SadTalker UI — every knob from the upstream Gradio app + extras."""

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._worker: SadTalkerWorker | None = None
        self._build_ui()
        self._wire()
        self._load_persisted()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # Wrap the whole control surface in a scroll area so the lower
        # collapsible sections + action row remain reachable on short windows.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        content.setMinimumHeight(640)
        scroll.setWidget(content)

        root = QVBoxLayout(content)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(8)

        # Top: source image (left) | output preview (right)
        top = QSplitter(Qt.Horizontal)
        self._src_image = ImageDropWidget(label="Source Image", thumb_height=240)
        self._preview = VideoPlayerWidget("Generated Preview")
        top.addWidget(self._src_image)
        top.addWidget(self._preview)
        top.setStretchFactor(0, 1)
        top.setStretchFactor(1, 1)
        root.addWidget(top)

        # Audio drop (always visible)
        audio_row = QHBoxLayout()
        audio_row.setSpacing(6)
        self._audio = FileDropWidget("Driving Audio", extensions=_AUDIO_EXTS)
        audio_row.addWidget(self._audio)
        root.addLayout(audio_row)

        # ── Basic controls (always visible) ─────────────────────────────────
        basic = QGroupBox("Basic")
        bl = QVBoxLayout(basic)
        bl.setContentsMargins(8, 6, 8, 6)
        bl.setSpacing(8)

        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row1.addWidget(QLabel("Preprocess:"))
        self._preprocess = QComboBox()
        self._preprocess.addItems(["crop", "resize", "full", "extcrop", "extfull"])
        self._preprocess.setCurrentText("extfull")
        self._preprocess.setMinimumWidth(120)
        self._preprocess.setToolTip(
            "How to handle the source image. extfull = best for landscape with "
            "centered subject. crop / resize = face-only output."
        )
        row1.addWidget(self._preprocess)

        row1.addWidget(QLabel("Resolution:"))
        self._size = QComboBox()
        self._size.addItems(["256", "512"])
        self._size.setCurrentText("256")
        self._size.setMinimumWidth(120)
        self._size.setToolTip("Face model resolution. 512 = sharper, slower.")
        row1.addWidget(self._size)

        row1.addWidget(QLabel("Pose style:"))
        self._pose_style = QSpinBox()
        self._pose_style.setRange(0, 46)
        self._pose_style.setValue(0)
        self._pose_style.setMinimumWidth(90)
        self._pose_style.setToolTip(
            "Head-motion template index (0-46). Discrete — try several to find one you like."
        )
        row1.addWidget(self._pose_style)

        row1.addWidget(QLabel("Expression scale:"))
        self._exp_scale = QDoubleSpinBox()
        self._exp_scale.setRange(0.0, 1.5)
        self._exp_scale.setSingleStep(0.05)
        self._exp_scale.setValue(1.0)
        self._exp_scale.setMinimumWidth(95)
        self._exp_scale.setToolTip("0 = neutral, ~1.0 = natural, >1.2 = exaggerated.")
        row1.addWidget(self._exp_scale)
        row1.addStretch()
        bl.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        self._still_mode = QCheckBox("Still Mode")
        self._still_mode.setToolTip("Reduce head motion (works best with preprocess=full).")
        row2.addWidget(self._still_mode)

        self._use_blink = QCheckBox("Use blink")
        self._use_blink.setChecked(True)
        self._use_blink.setToolTip("Generate eye blinks. Uncheck for stylized characters.")
        row2.addWidget(self._use_blink)

        row2.addWidget(QLabel("Batch size:"))
        self._batch_size = QSpinBox()
        self._batch_size.setRange(1, 10)
        self._batch_size.setValue(2)
        self._batch_size.setMinimumWidth(90)
        row2.addWidget(self._batch_size)
        row2.addStretch()
        bl.addLayout(row2)

        root.addWidget(basic)

        # ── Enhancers ───────────────────────────────────────────────────────
        enh = CollapsibleSection("Face / Background Enhancers", collapsed=True)
        enh_layout = QVBoxLayout()
        enh_layout.setSpacing(4)
        row_enh = QHBoxLayout()
        row_enh.setSpacing(6)
        row_enh.addWidget(QLabel("Face enhancer:"))
        self._enhancer_method = QComboBox()
        self._enhancer_method.addItems(["None", "gfpgan", "RestoreFormer"])
        self._enhancer_method.setMinimumWidth(160)
        self._enhancer_method.setToolTip(
            "gfpgan = sharp realistic faces; RestoreFormer = better on stylized."
        )
        row_enh.addWidget(self._enhancer_method)
        self._background_enhancer = QCheckBox("Background (RealESRGAN)")
        self._background_enhancer.setToolTip("Upscales the rest of the frame. Slow.")
        row_enh.addWidget(self._background_enhancer)
        row_enh.addStretch()
        enh_layout.addLayout(row_enh)
        enh.add_layout(enh_layout)
        root.addWidget(enh)

        # ── Reference Video ─────────────────────────────────────────────────
        ref = CollapsibleSection("Reference Video (borrow pose / blink)", collapsed=True)
        ref_layout = QVBoxLayout()
        ref_layout.setSpacing(4)
        self._ref_video = FileDropWidget("Reference video", extensions=_VIDEO_EXTS)
        ref_layout.addWidget(self._ref_video)
        ref_info_row = QHBoxLayout()
        ref_info_row.setSpacing(6)
        ref_info_row.addWidget(QLabel("Borrow:"))
        self._ref_info = QComboBox()
        self._ref_info.addItems(["pose", "blink", "pose+blink", "all"])
        self._ref_info.setCurrentText("pose+blink")
        self._ref_info.setMinimumWidth(160)
        self._ref_info.setToolTip(
            "'all' = use the reference's audio too (overrides driving_audio)."
        )
        ref_info_row.addWidget(self._ref_info)
        self._use_ref_video = QCheckBox("Use reference video")
        ref_info_row.addWidget(self._use_ref_video)
        ref_info_row.addStretch()
        ref_layout.addLayout(ref_info_row)
        ref.add_layout(ref_layout)
        root.addWidget(ref)

        # ── Idle Mode ───────────────────────────────────────────────────────
        idle = CollapsibleSection("Idle Mode (silent talking head)", collapsed=True)
        idle_layout = QVBoxLayout()
        idle_layout.setSpacing(4)
        idle_row = QHBoxLayout()
        idle_row.setSpacing(6)
        self._use_idle_mode = QCheckBox("Use idle mode (no audio)")
        idle_row.addWidget(self._use_idle_mode)
        idle_row.addWidget(QLabel("Length (sec):"))
        self._length_of_audio = QSpinBox()
        self._length_of_audio.setRange(0, 120)
        self._length_of_audio.setValue(5)
        self._length_of_audio.setMinimumWidth(90)
        idle_row.addWidget(self._length_of_audio)
        idle_row.addStretch()
        idle_layout.addLayout(idle_row)
        idle.add_layout(idle_layout)
        root.addWidget(idle)

        # ── Free-View Head Pose ─────────────────────────────────────────────
        fv = CollapsibleSection("Free-View Head Pose (degrees, comma-separated)", collapsed=True)
        fv_layout = QVBoxLayout()
        fv_layout.setSpacing(4)
        for label, attr in (("Yaw (left/right):", "_input_yaw_str"),
                            ("Pitch (up/down):", "_input_pitch_str"),
                            ("Roll (tilt):", "_input_roll_str")):
            r = QHBoxLayout()
            r.setSpacing(6)
            r.addWidget(QLabel(label))
            edit = QLineEdit()
            edit.setPlaceholderText("e.g. 0, 30, -30  (interpolated across clip)")
            setattr(self, attr, edit)
            r.addWidget(edit, 1)
            fv_layout.addLayout(r)
        fv.add_layout(fv_layout)
        root.addWidget(fv)

        # ── Debug ───────────────────────────────────────────────────────────
        dbg = CollapsibleSection("Debug", collapsed=True)
        dbg_layout = QVBoxLayout()
        self._face3dvis = QCheckBox("Render 3D face/landmarks debug (3dface.mp4)")
        self._face3dvis.setToolTip("Requires SadTalker's BFM_Fitting weights.")
        dbg_layout.addWidget(self._face3dvis)
        dbg.add_layout(dbg_layout)
        root.addWidget(dbg)

        # Action row
        action = QHBoxLayout()
        action.setSpacing(8)
        self._generate_btn = QPushButton("Generate")
        self._generate_btn.setObjectName("primary")
        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setEnabled(False)
        action.addWidget(self._generate_btn)
        action.addWidget(self._abort_btn)
        action.addStretch()
        self._status = QLabel("")
        self._status.setStyleSheet("color: #aaa;")
        action.addWidget(self._status, 1)
        root.addLayout(action)

    # ------------------------------------------------------------------
    def _wire(self) -> None:
        self._generate_btn.clicked.connect(self._on_generate)
        self._abort_btn.clicked.connect(self._on_abort)

        # Persist on any change
        widgets = [
            self._src_image, self._audio, self._preprocess, self._size,
            self._pose_style, self._exp_scale, self._still_mode, self._use_blink,
            self._batch_size, self._enhancer_method, self._background_enhancer,
            self._ref_video, self._ref_info, self._use_ref_video,
            self._use_idle_mode, self._length_of_audio,
            self._input_yaw_str, self._input_pitch_str, self._input_roll_str,
            self._face3dvis,
        ]
        for w in widgets:
            for sig in ("image_loaded", "file_loaded", "file_cleared",
                        "currentTextChanged", "valueChanged",
                        "toggled", "textChanged"):
                s = getattr(w, sig, None)
                if s is not None and hasattr(s, "connect"):
                    s.connect(self._persist)

        # Auto-flag use_ref_video on upload
        def _on_ref(p):
            self._use_ref_video.setChecked(bool(p))
        self._ref_video.file_loaded.connect(_on_ref)
        self._ref_video.file_cleared.connect(lambda: self._use_ref_video.setChecked(False))

    # ------------------------------------------------------------------
    # Public hooks (called from MainWindow on send-to)
    # ------------------------------------------------------------------
    def load_source(self, path: str | None) -> None:
        if path:
            self._src_image.load_image(path)

    def load_audio(self, path: str | None) -> None:
        if path:
            self._audio.load_file(path)

    def load_ref_video(self, path: str | None) -> None:
        if path:
            self._ref_video.load_file(path)

    # ------------------------------------------------------------------
    def _project_cfg(self) -> ProjectConfig:
        return ProjectConfig.load(self.project_path)

    def _persist(self, *_args) -> None:
        if self._restoring:
            return
        cfg = self._project_cfg()
        cfg.sadtalker_source_path = self._src_image.image_path() or ""
        cfg.sadtalker_audio_path = self._audio.file_path or ""
        cfg.sadtalker_preprocess = self._preprocess.currentText()
        cfg.sadtalker_size = int(self._size.currentText())
        cfg.sadtalker_pose_style = self._pose_style.value()
        cfg.sadtalker_exp_scale = self._exp_scale.value()
        cfg.sadtalker_still_mode = self._still_mode.isChecked()
        cfg.sadtalker_use_blink = self._use_blink.isChecked()
        cfg.sadtalker_batch_size = self._batch_size.value()
        cfg.sadtalker_enhancer_method = self._enhancer_method.currentText()
        cfg.sadtalker_background_enhancer = self._background_enhancer.isChecked()
        cfg.sadtalker_ref_video = self._ref_video.file_path or ""
        cfg.sadtalker_ref_info = self._ref_info.currentText()
        cfg.sadtalker_use_ref_video = self._use_ref_video.isChecked()
        cfg.sadtalker_use_idle_mode = self._use_idle_mode.isChecked()
        cfg.sadtalker_length_of_audio = self._length_of_audio.value()
        cfg.sadtalker_input_yaw_str = self._input_yaw_str.text()
        cfg.sadtalker_input_pitch_str = self._input_pitch_str.text()
        cfg.sadtalker_input_roll_str = self._input_roll_str.text()
        cfg.sadtalker_face3dvis = self._face3dvis.isChecked()
        cfg.save(self.project_path)

    def _load_persisted(self) -> None:
        self._restoring = True
        try:
            cfg = self._project_cfg()
            src = getattr(cfg, "sadtalker_source_path", "") or ""
            aud = getattr(cfg, "sadtalker_audio_path", "") or ""
            if src and Path(src).is_file():
                self._src_image.load_image(src)
            if aud and Path(aud).is_file():
                self._audio.load_file(aud)
            pp = getattr(cfg, "sadtalker_preprocess", "extfull") or "extfull"
            i = self._preprocess.findText(pp)
            if i >= 0:
                self._preprocess.setCurrentIndex(i)
            sz = str(getattr(cfg, "sadtalker_size", 256))
            i = self._size.findText(sz)
            if i >= 0:
                self._size.setCurrentIndex(i)
            self._pose_style.setValue(int(getattr(cfg, "sadtalker_pose_style", 0)))
            self._exp_scale.setValue(float(getattr(cfg, "sadtalker_exp_scale", 1.0)))
            self._still_mode.setChecked(bool(getattr(cfg, "sadtalker_still_mode", False)))
            self._use_blink.setChecked(bool(getattr(cfg, "sadtalker_use_blink", True)))
            self._batch_size.setValue(int(getattr(cfg, "sadtalker_batch_size", 2)))
            em = getattr(cfg, "sadtalker_enhancer_method", "None") or "None"
            i = self._enhancer_method.findText(em)
            if i >= 0:
                self._enhancer_method.setCurrentIndex(i)
            self._background_enhancer.setChecked(bool(getattr(cfg, "sadtalker_background_enhancer", False)))
            rv = getattr(cfg, "sadtalker_ref_video", "") or ""
            if rv and Path(rv).is_file():
                self._ref_video.load_file(rv)
            ri = getattr(cfg, "sadtalker_ref_info", "pose+blink") or "pose+blink"
            i = self._ref_info.findText(ri)
            if i >= 0:
                self._ref_info.setCurrentIndex(i)
            self._use_ref_video.setChecked(bool(getattr(cfg, "sadtalker_use_ref_video", False)))
            self._use_idle_mode.setChecked(bool(getattr(cfg, "sadtalker_use_idle_mode", False)))
            self._length_of_audio.setValue(int(getattr(cfg, "sadtalker_length_of_audio", 5)))
            self._input_yaw_str.setText(getattr(cfg, "sadtalker_input_yaw_str", "") or "")
            self._input_pitch_str.setText(getattr(cfg, "sadtalker_input_pitch_str", "") or "")
            self._input_roll_str.setText(getattr(cfg, "sadtalker_input_roll_str", "") or "")
            self._face3dvis.setChecked(bool(getattr(cfg, "sadtalker_face3dvis", False)))
            last_result = getattr(cfg, "sadtalker_result_path", "") or ""
            if last_result and Path(last_result).is_file():
                self._preview.load_video(last_result)
        finally:
            self._restoring = False

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        self._load_persisted()

    # ------------------------------------------------------------------
    def _on_generate(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._show_status("Already generating.")
            return
        src = self._src_image.image_path()
        if not src or not Path(src).is_file():
            self._show_status("Load a source image first.")
            return
        # Audio is optional if idle mode or ref_info='all'
        aud = self._audio.file_path
        if not (self._use_idle_mode.isChecked()
                or (self._use_ref_video.isChecked() and self._ref_info.currentText() == "all")):
            if not aud or not Path(aud).is_file():
                self._show_status("Load a driving audio file (or enable Idle Mode or ref_info='all').")
                return

        self._show_status("Starting SadTalker...")
        self._generate_btn.setEnabled(False)
        self._abort_btn.setEnabled(True)

        em = self._enhancer_method.currentText()
        self._worker = SadTalkerWorker(
            self.state, self.project_name,
            source_image=src,
            driven_audio=aud if aud else None,
            preprocess=self._preprocess.currentText(),
            still_mode=self._still_mode.isChecked(),
            enhancer_method=None if em == "None" else em,
            background_enhancer=self._background_enhancer.isChecked(),
            batch_size=self._batch_size.value(),
            size=int(self._size.currentText()),
            pose_style=self._pose_style.value(),
            exp_scale=self._exp_scale.value(),
            use_ref_video=self._use_ref_video.isChecked(),
            ref_video=self._ref_video.file_path or None,
            ref_info=self._ref_info.currentText() if self._use_ref_video.isChecked() else None,
            use_idle_mode=self._use_idle_mode.isChecked(),
            length_of_audio=self._length_of_audio.value(),
            use_blink=self._use_blink.isChecked(),
            input_yaw_str=self._input_yaw_str.text(),
            input_pitch_str=self._input_pitch_str.text(),
            input_roll_str=self._input_roll_str.text(),
            face3dvis=self._face3dvis.isChecked(),
        )
        self._worker.progress.connect(lambda _f, d: self._show_status(d))
        self._worker.status.connect(self._show_status)
        self._worker.error.connect(self._on_error)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.start()

    def _on_abort(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.abort()
            self._show_status("Abort requested...")

    def _on_done(self, result: object) -> None:
        path = str(result) if result else ""
        self._show_status(f"Done: {Path(path).name if path else 'no output'}")
        self._generate_btn.setEnabled(True)
        self._abort_btn.setEnabled(False)
        if path and Path(path).is_file():
            self._preview.load_video(path)
            cfg = self._project_cfg()
            cfg.sadtalker_result_path = path
            cfg.save(self.project_path)

    def _on_error(self, msg: str) -> None:
        self._show_status(f"Error: {msg}")
        self._generate_btn.setEnabled(True)
        self._abort_btn.setEnabled(False)
