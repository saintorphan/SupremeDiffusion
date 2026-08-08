"""Pose Animate tab — pose-driven body animation via MimicMotion."""

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
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from supremediffusion.config.project_config import ProjectConfig

from sdqt.state import AppState
from sdqt.widgets.file_drop import FileDropWidget
from sdqt.widgets.image_drop import ImageDropWidget
from sdqt.widgets.video_player import VideoPlayerWidget
from sdqt.workers.pose_animate import PoseAnimateWorker

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

from .base import BaseTab

logger = logging.getLogger(__name__)


_RESOLUTIONS = ["384", "448", "480", "512", "576"]


class PoseAnimateTab(BaseTab):
    """Tab UI for MimicMotion — pose-driven body animation on a still image."""

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._worker: PoseAnimateWorker | None = None
        self._build_ui()
        self._wire()
        self._load_persisted()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 0, 2, 2)
        root.setSpacing(2)

        # Top splitter: reference image (left) | driving video (right) | output (full row below)
        top_splitter = QSplitter(Qt.Horizontal)
        self._ref_drop = ImageDropWidget(label="Reference Image", thumb_height=240)
        self._driver_player = VideoPlayerWidget("Driving Video (pose source)")
        top_splitter.addWidget(self._ref_drop)
        top_splitter.addWidget(self._driver_player)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 1)
        root.addWidget(top_splitter)

        # Settings row
        settings_box = QGroupBox("Generation Settings")
        settings_layout = QVBoxLayout(settings_box)
        settings_layout.setContentsMargins(6, 6, 6, 6)
        settings_layout.setSpacing(4)

        row1 = QHBoxLayout()
        row1.setSpacing(6)
        row1.addWidget(QLabel("Resolution:"))
        self._resolution = QComboBox()
        self._resolution.addItems(_RESOLUTIONS)
        self._resolution.setCurrentText("512")
        self._resolution.setFixedWidth(100)
        self._resolution.setToolTip(
            "Short-side resolution. 384 = fastest, 576 = upstream default. "
            "12 GB VRAM comfortable up to 512; 576 may OOM."
        )
        row1.addWidget(self._resolution)

        row1.addWidget(QLabel("Steps:"))
        self._steps = QSpinBox()
        self._steps.setRange(4, 50)
        self._steps.setValue(25)
        self._steps.setFixedWidth(75)
        self._steps.setToolTip("Denoising steps. 25 default. 12–15 for fast drafts.")
        row1.addWidget(self._steps)

        row1.addWidget(QLabel("FPS:"))
        self._fps = QSpinBox()
        self._fps.setRange(8, 60)
        self._fps.setValue(24)
        self._fps.setFixedWidth(75)
        row1.addWidget(self._fps)

        row1.addWidget(QLabel("Seed:"))
        self._seed = QSpinBox()
        self._seed.setRange(-1, 2_147_483_647)
        self._seed.setValue(-1)
        self._seed.setFixedWidth(100)
        self._seed.setToolTip("-1 = random.")
        row1.addWidget(self._seed)
        row1.addStretch()
        settings_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(6)
        row2.addWidget(QLabel("Min guidance:"))
        self._min_g = QDoubleSpinBox()
        self._min_g.setRange(0.5, 5.0)
        self._min_g.setSingleStep(0.1)
        self._min_g.setValue(1.0)
        self._min_g.setFixedWidth(80)
        row2.addWidget(self._min_g)

        row2.addWidget(QLabel("Max guidance:"))
        self._max_g = QDoubleSpinBox()
        self._max_g.setRange(0.5, 10.0)
        self._max_g.setSingleStep(0.1)
        self._max_g.setValue(3.0)
        self._max_g.setFixedWidth(80)
        self._max_g.setToolTip(
            "SVD linearly ramps CFG from min (first frame) to max (last frame). "
            "Default 1.0→3.0 matches upstream."
        )
        row2.addWidget(self._max_g)

        row2.addWidget(QLabel("Noise aug:"))
        self._noise_aug = QDoubleSpinBox()
        self._noise_aug.setRange(0.0, 0.5)
        self._noise_aug.setSingleStep(0.01)
        self._noise_aug.setValue(0.02)
        self._noise_aug.setFixedWidth(80)
        self._noise_aug.setToolTip(
            "Higher = more motion deviation from reference. 0.02 is upstream default."
        )
        row2.addWidget(self._noise_aug)
        row2.addStretch()
        settings_layout.addLayout(row2)

        root.addWidget(settings_box)

        # Optional audio for lip sync — runs LatentSync after MimicMotion completes
        lipsync_box = QGroupBox("Optional: Lip Sync (LatentSync post-step)")
        ls_layout = QVBoxLayout(lipsync_box)
        ls_layout.setContentsMargins(6, 6, 6, 6)
        ls_layout.setSpacing(4)
        self._audio_drop = FileDropWidget("Audio (drop .wav / .mp3)", extensions=_AUDIO_EXTS)
        ls_layout.addWidget(self._audio_drop)
        ls_info = QLabel(
            "If an audio file is set, the MimicMotion output runs through LatentSync "
            "to re-sync the mouth to the audio. Pipelines load sequentially (MimicMotion "
            "unloads before LatentSync), so the full chain fits in 12 GB."
        )
        ls_info.setWordWrap(True)
        ls_info.setStyleSheet("color: #888; font-size: 11px;")
        ls_layout.addWidget(ls_info)
        root.addWidget(lipsync_box)

        # Action row
        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        self._generate_btn = QPushButton("Generate")
        self._generate_btn.setStyleSheet(
            "QPushButton { background:#2a6; color:white; font-weight:bold; padding:6px 16px; }"
        )
        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setEnabled(False)
        self._preview_pose_btn = QPushButton("Preview Pose")
        self._preview_pose_btn.setToolTip(
            "Run DWPose on the reference image + driver's first frame and show "
            "the detected skeletons side-by-side. Use this to verify pose "
            "detection before committing to a full generation."
        )
        action_row.addWidget(self._generate_btn)
        action_row.addWidget(self._abort_btn)
        action_row.addWidget(self._preview_pose_btn)
        action_row.addStretch()
        self._status = QLabel("")
        action_row.addWidget(self._status, 1)
        root.addLayout(action_row)

        # Output preview
        self._preview = VideoPlayerWidget("Generated Preview")
        root.addWidget(self._preview)

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def _wire(self) -> None:
        self._generate_btn.clicked.connect(self._on_generate)
        self._abort_btn.clicked.connect(self._on_abort)
        self._preview_pose_btn.clicked.connect(self._on_preview_pose)

        for w in (
            self._resolution, self._steps, self._fps, self._seed,
            self._min_g, self._max_g, self._noise_aug,
        ):
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(self._persist)  # type: ignore[attr-defined]
            elif hasattr(w, "currentTextChanged"):
                w.currentTextChanged.connect(self._persist)  # type: ignore[attr-defined]

        self._ref_drop.image_loaded.connect(self._persist)
        self._driver_player.video_loaded.connect(self._persist)
        self._audio_drop.file_loaded.connect(self._persist)
        self._audio_drop.file_cleared.connect(self._persist)

    # ------------------------------------------------------------------
    # Public hooks called from MainWindow
    # ------------------------------------------------------------------

    def load_source(self, path: str | None) -> None:
        """Receive a video from another tab as the driving video."""
        if path:
            self._driver_player.load_video(path)

    def load_reference_image(self, path: str | None) -> None:
        """Receive an image from another tab as the reference."""
        if path:
            self._ref_drop.load_image(path)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _project_cfg(self) -> ProjectConfig:
        return ProjectConfig.load(self.project_path)

    def _persist(self, *_args) -> None:
        if self._restoring:
            return
        cfg = self._project_cfg()
        cfg.mimicmotion_reference_path = self._ref_drop.image_path() or ""
        cfg.mimicmotion_driving_path = self._driver_player.video_path() or ""
        cfg.mimicmotion_audio_path = self._audio_drop.file_path or ""
        cfg.mimicmotion_resolution = self._resolution.currentText()
        cfg.mimicmotion_steps = self._steps.value()
        cfg.mimicmotion_fps = self._fps.value()
        cfg.mimicmotion_seed = self._seed.value()
        cfg.mimicmotion_min_guidance = self._min_g.value()
        cfg.mimicmotion_max_guidance = self._max_g.value()
        cfg.mimicmotion_noise_aug = self._noise_aug.value()
        cfg.save(self.project_path)

    def _load_persisted(self) -> None:
        self._restoring = True
        try:
            cfg = self._project_cfg()
            ref = getattr(cfg, "mimicmotion_reference_path", "") or ""
            drv = getattr(cfg, "mimicmotion_driving_path", "") or ""
            aud = getattr(cfg, "mimicmotion_audio_path", "") or ""
            if ref and Path(ref).is_file():
                self._ref_drop.load_image(ref)
            if drv and Path(drv).is_file():
                self._driver_player.load_video(drv)
            if aud and Path(aud).is_file():
                self._audio_drop.load_file(aud)
            res = getattr(cfg, "mimicmotion_resolution", "512") or "512"
            # Strip "WxH" form if present (legacy), keep just short side.
            short = res.split("x")[0]
            if short in _RESOLUTIONS:
                self._resolution.setCurrentText(short)
            self._steps.setValue(int(getattr(cfg, "mimicmotion_steps", 25)))
            self._fps.setValue(int(getattr(cfg, "mimicmotion_fps", 24)))
            self._seed.setValue(int(getattr(cfg, "mimicmotion_seed", -1)))
            self._min_g.setValue(float(getattr(cfg, "mimicmotion_min_guidance", 1.0)))
            self._max_g.setValue(float(getattr(cfg, "mimicmotion_max_guidance", 3.0)))
            self._noise_aug.setValue(float(getattr(cfg, "mimicmotion_noise_aug", 0.02)))
            last_result = getattr(cfg, "mimicmotion_result_path", "") or ""
            if last_result and Path(last_result).is_file():
                self._preview.load_video(last_result)
        finally:
            self._restoring = False

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        self._load_persisted()

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def _on_generate(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._show_status("Already generating.")
            return

        ref = self._ref_drop.image_path()
        drv = self._driver_player.video_path()
        if not ref or not Path(ref).is_file():
            self._show_status("Load a reference image first.")
            return
        if not drv or not Path(drv).is_file():
            self._show_status("Load a driving video first.")
            return

        self._show_status("Starting MimicMotion...")
        self._generate_btn.setEnabled(False)
        self._abort_btn.setEnabled(True)

        try:
            res = int(self._resolution.currentText())
        except ValueError:
            res = 512

        audio_path = self._audio_drop.file_path
        if audio_path and not Path(audio_path).is_file():
            audio_path = None

        self._worker = PoseAnimateWorker(
            self.state,
            self.project_name,
            reference_image_path=ref,
            driving_video_path=drv,
            audio_path=audio_path,
            resolution=res,
            num_inference_steps=self._steps.value(),
            min_guidance=self._min_g.value(),
            max_guidance=self._max_g.value(),
            noise_aug_strength=self._noise_aug.value(),
            seed=self._seed.value(),
            fps=self._fps.value(),
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
            cfg.mimicmotion_result_path = path
            cfg.save(self.project_path)

    def _on_error(self, msg: str) -> None:
        self._show_status(f"Error: {msg}")
        self._generate_btn.setEnabled(True)
        self._abort_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Pose preview — sanity-check DWPose detection before paying for a run
    # ------------------------------------------------------------------

    def _on_preview_pose(self) -> None:
        from PySide6.QtGui import QImage, QPixmap
        from PySide6.QtWidgets import QDialog, QHBoxLayout as _HB, QLabel as _Lb

        ref = self._ref_drop.image_path()
        drv = self._driver_player.video_path()
        if not ref or not Path(ref).is_file():
            self._show_status("Load a reference image first.")
            return
        if not drv or not Path(drv).is_file():
            self._show_status("Load a driving video first.")
            return

        self._show_status("Running DWPose preview...")
        self._preview_pose_btn.setEnabled(False)
        try:
            try:
                res = int(self._resolution.currentText())
            except ValueError:
                res = 512
            from supremediffusion.models.mimicmotion_pipeline import MimicMotionPipeline
            pipe = self.state.mimicmotion_pipeline
            if pipe is None:
                pipe = MimicMotionPipeline(self.state.global_config)
            ref_pose, drv_pose = pipe.preview_poses(ref, drv, resolution=res)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pose preview failed")
            self._show_status(f"Pose preview error: {exc}")
            self._preview_pose_btn.setEnabled(True)
            return
        finally:
            self._preview_pose_btn.setEnabled(True)

        def _np_to_pixmap(arr) -> QPixmap:
            h, w = arr.shape[:2]
            buf = bytes(arr.astype("uint8").tobytes())
            img = QImage(buf, w, h, w * 3, QImage.Format_RGB888).copy()
            return QPixmap.fromImage(img)

        dlg = QDialog(self)
        dlg.setWindowTitle("DWPose Preview — Reference (left) / Driver (right)")
        dlg.resize(900, 600)
        h = _HB(dlg)
        h.setContentsMargins(4, 4, 4, 4)
        h.setSpacing(4)
        for arr, title in ((ref_pose, "Reference"), (drv_pose, "Driver frame 0")):
            col = QVBoxLayout()
            col.addWidget(_Lb(f"<b>{title}</b>"))
            img_label = _Lb()
            img_label.setPixmap(_np_to_pixmap(arr).scaled(420, 560,
                                  Qt.KeepAspectRatio, Qt.SmoothTransformation))
            col.addWidget(img_label)
            h.addLayout(col)
        self._show_status("Pose preview done.")
        dlg.exec()
