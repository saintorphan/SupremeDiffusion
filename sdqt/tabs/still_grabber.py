"""StillGrabber tab — extract evenly-spaced still frames from a video."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from sdqt.state import AppState
from sdqt.tabs.base import BaseTab
from sdqt.utils.naming import cap_stem
from sdqt.widgets.video_player import VideoPlayerWidget
from sdqt.workers.base import BaseWorker

logger = logging.getLogger(__name__)


class _StillGrabWorker(BaseWorker):
    """Extract evenly-spaced frames from a video."""

    def __init__(self, video_path: str, count: int, output_dir: str,
                 target_w: int = 0, target_h: int = 0, parent=None):
        super().__init__(parent)
        self._path = video_path
        self._count = count
        self._out_dir = output_dir
        self._target_w = target_w
        self._target_h = target_h

    def do_work(self) -> list[str]:
        import json

        Path(self._out_dir).mkdir(parents=True, exist_ok=True)

        # Probe duration
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "format=duration",
             "-of", "json", self._path],
            capture_output=True, text=True, timeout=10,
        )
        dur = float(json.loads(probe.stdout)["format"]["duration"])
        if dur <= 0:
            raise RuntimeError("Could not determine video duration")

        count = max(1, self._count)
        # Spread timestamps evenly (exclude very start and very end)
        if count == 1:
            timestamps = [dur / 2.0]
        else:
            step = dur / (count + 1)
            timestamps = [step * (i + 1) for i in range(count)]

        stem = cap_stem(Path(self._path).stem)
        paths: list[str] = []

        from supremediffusion.utils.video import extract_single_frame, probe_video
        info = probe_video(self._path)
        fps = info.get("fps", 24)

        for i, ts in enumerate(timestamps):
            if self.is_aborted:
                break
            out = str(Path(self._out_dir) / f"{stem}_still_{i + 1:03d}.png")
            self.progress.emit(
                (i + 1) / count,
                f"Extracting frame {i + 1}/{count}...",
            )
            frame_num = max(0, int(ts * fps))
            try:
                img = extract_single_frame(self._path, frame_num)
                # Resize if target dimensions specified (img is PIL Image)
                if self._target_w > 0 and self._target_h > 0:
                    from PIL.Image import Resampling
                    img = img.resize(
                        (self._target_w, self._target_h),
                        resample=Resampling.LANCZOS,
                    )
                img.save(out, "PNG")
                paths.append(out)
            except Exception:
                pass

        self.progress.emit(1.0, f"Done — {len(paths)} frames extracted")
        return paths


class StillGrabberTab(BaseTab):
    """Extract evenly-spaced still frames from a video."""

    send_image_requested = Signal(str, str)  # (target_key, image_path)

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._worker: _StillGrabWorker | None = None
        self._output_dir: str = ""
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 2)

        # Source video
        self._player = VideoPlayerWidget("Source Video")
        layout.addWidget(self._player)

        # ── Output Settings group (frames + resolution) ──
        settings_group = QGroupBox("Output Settings")
        settings_lay = QVBoxLayout(settings_group)
        settings_lay.setContentsMargins(8, 6, 8, 6)
        settings_lay.setSpacing(8)

        settings_row = QHBoxLayout()
        settings_row.setSpacing(8)

        settings_row.addWidget(QLabel("Frames:"))
        self._count = QSpinBox()
        self._count.setRange(1, 999)
        self._count.setValue(10)
        self._count.setMinimumWidth(90)
        self._count.setToolTip("Number of evenly-spaced frames to extract")
        settings_row.addWidget(self._count)

        settings_row.addWidget(QLabel("Output:"))
        self._output_size = QComboBox()
        self._output_size.setMinimumWidth(200)
        self._output_size.setToolTip("Output resolution for extracted frames")
        self._output_size_presets = [
            ("Native", 0, 0),
            # Wan 2.2
            ("832×480 (Wan 16:9)", 832, 480),
            ("480×832 (Wan 9:16)", 480, 832),
            ("720×720 (Wan 1:1)", 720, 720),
            ("960×544 (Wan 16:9+)", 960, 544),
            ("544×960 (Wan 9:16+)", 544, 960),
            # SDXL
            ("1024×1024 (SDXL 1:1)", 1024, 1024),
            ("1344×768 (SDXL 16:9)", 1344, 768),
            ("768×1344 (SDXL 9:16)", 768, 1344),
            ("1152×896 (SDXL 9:7)", 1152, 896),
            ("896×1152 (SDXL 7:9)", 896, 1152),
            ("1216×832 (SDXL 3:2)", 1216, 832),
            ("832×1216 (SDXL 2:3)", 832, 1216),
            # SD 1.5
            ("768×768 (SD 1:1)", 768, 768),
            ("512×512 (SD 1:1)", 512, 512),
        ]
        for label, _w, _h in self._output_size_presets:
            self._output_size.addItem(label)
        settings_row.addWidget(self._output_size)

        self._res_label = QLabel("")
        self._res_label.setStyleSheet("color: #bbb; font-size: 13px;")
        settings_row.addWidget(self._res_label)

        settings_row.addStretch()
        settings_lay.addLayout(settings_row)
        layout.addWidget(settings_group)

        # ── Action row (primary Grab on its own row) ──
        action_row = QHBoxLayout()
        action_row.setSpacing(8)

        self._grab_btn = QPushButton("Grab Stills")
        self._grab_btn.setObjectName("primary")
        self._grab_btn.setMinimumWidth(150)
        self._grab_btn.clicked.connect(self._on_grab)
        action_row.addWidget(self._grab_btn)

        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setVisible(False)
        self._abort_btn.clicked.connect(self._on_abort)
        action_row.addWidget(self._abort_btn)

        action_row.addStretch()

        self._open_folder_btn = QPushButton("Open Folder")
        self._open_folder_btn.setEnabled(False)
        self._open_folder_btn.clicked.connect(self._on_open_folder)
        action_row.addWidget(self._open_folder_btn)

        layout.addLayout(action_row)

        # Batch Post-Processing (QGroupBox already imported at module scope)
        from PySide6.QtWidgets import QCheckBox, QGridLayout
        pp_row = QHBoxLayout()
        pp_row.setSpacing(6)

        self._pp_check = QCheckBox("Batch Post-Process")
        self._pp_check.setToolTip("Apply AI enhancement to each extracted frame")
        self._pp_check.setChecked(True)
        self._pp_check.toggled.connect(lambda on: self._pp_group.setVisible(on))
        pp_row.addWidget(self._pp_check)
        pp_row.addStretch()
        layout.addLayout(pp_row)

        self._pp_group = QGroupBox()
        self._pp_group.setVisible(True)
        pp_grid = QGridLayout(self._pp_group)
        pp_grid.setContentsMargins(10, 8, 10, 8)
        pp_grid.setSpacing(8)

        def _combo(row, label, items, tooltip):
            pp_grid.addWidget(QLabel(label), row, 0)
            c = QComboBox()
            c.addItems(items)
            c.setMinimumWidth(160)
            c.setToolTip(tooltip)
            pp_grid.addWidget(c, row, 1)
            return c

        self._pp_denoise = _combo(0, "Denoise:", ["Disabled", "SCUNet"],
                                   "Remove grain/noise")
        self._pp_sharpen = _combo(1, "Sharpen:", ["Disabled", "RealESRGAN 2x"],
                                   "Neural detail recovery")
        self._pp_sharpen.setCurrentIndex(1)  # RealESRGAN 2x on by default
        self._pp_upscale = _combo(2, "Upscale:", ["Disabled", "RealESRGAN 4x"],
                                   "4x neural super-resolution")
        self._pp_upscale.setCurrentIndex(1)  # RealESRGAN 4x on by default
        self._pp_face = _combo(3, "Face Restore:", ["Disabled", "GFPGAN v1.4"],
                                "Fix face artifacts")

        from PySide6.QtWidgets import QSpinBox as _Spin
        pp_grid.addWidget(QLabel("Tile Size:"), 4, 0)
        self._pp_tile = _Spin()
        self._pp_tile.setRange(0, 1024)
        self._pp_tile.setValue(512)
        self._pp_tile.setSingleStep(128)
        self._pp_tile.setSuffix("px")
        self._pp_tile.setMinimumWidth(90)
        pp_grid.addWidget(self._pp_tile, 4, 1)

        layout.addWidget(self._pp_group)

        # Status
        self._status = QLabel("")
        self._status.setStyleSheet("color: #aaa; font-style: italic; font-size: 13px;")
        layout.addWidget(self._status)

        # Results gallery
        self._results = QListWidget()
        self._results.setViewMode(QListWidget.IconMode)
        self._results.setIconSize(self._results.iconSize().expandedTo(
            __import__("PySide6.QtCore", fromlist=["QSize"]).QSize(160, 160)))
        from PySide6.QtCore import QSize
        self._results.setIconSize(QSize(160, 160))
        self._results.setResizeMode(QListWidget.Adjust)
        self._results.setSpacing(4)
        self._results.setWrapping(True)
        self._results.setContextMenuPolicy(Qt.CustomContextMenu)
        self._results.customContextMenuRequested.connect(self._on_result_context_menu)
        layout.addWidget(self._results, 1)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_source(self, path: str) -> None:
        """Load a video (called from send-to)."""
        if path and Path(path).is_file():
            self._player.load_video(path)
            self._update_res_label(path)
            self._show_status(f"Loaded: {Path(path).name}")

    def _update_res_label(self, path: str) -> None:
        """Show source video resolution."""
        try:
            from supremediffusion.utils.video import probe_video
            info = probe_video(path)
            w, h = info.get("width", 0), info.get("height", 0)
            if w and h:
                self._res_label.setText(f"Source: {w}×{h}")
            else:
                self._res_label.setText("")
        except Exception:
            self._res_label.setText("")

    # ── Actions ───────────────────────────────────────────────────────────────

    @Slot()
    def _on_grab(self) -> None:
        path = self._player.video_path
        if not path or not Path(path).is_file():
            self._show_status("No video loaded.")
            return

        self._output_dir = str(self.project_path / "outputs" / "stills")
        count = self._count.value()

        self._grab_btn.setVisible(False)
        self._abort_btn.setVisible(True)
        self._results.clear()
        self._open_folder_btn.setEnabled(False)

        # Get target output size
        idx = self._output_size.currentIndex()
        _, target_w, target_h = self._output_size_presets[idx] if idx >= 0 else ("", 0, 0)

        worker = _StillGrabWorker(path, count, self._output_dir,
                                  target_w=target_w, target_h=target_h, parent=self)
        worker.progress.connect(lambda f, d: self._show_status(d))
        worker.finished_ok.connect(self._on_done)
        worker.error.connect(self._on_error)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    @Slot()
    def _on_abort(self) -> None:
        if self._worker:
            self._worker.abort()

    def _on_done(self, paths: list[str]) -> None:
        if self._pp_check.isChecked() and paths and self._has_pp_enabled():
            self._pending_pp = list(paths)
            self._pp_results: list[str] = []
            self._show_status("Post-processing extracted frames...")
            self._run_next_pp()
            return

        self._show_results(paths)

    def _has_pp_enabled(self) -> bool:
        return (self._pp_denoise.currentIndex() > 0 or
                self._pp_sharpen.currentIndex() > 0 or
                self._pp_upscale.currentIndex() > 0 or
                self._pp_face.currentIndex() > 0)

    def _run_next_pp(self) -> None:
        if not self._pending_pp:
            self._show_results(self._pp_results)
            return

        path = self._pending_pp.pop(0)
        total = len(self._pp_results) + len(self._pending_pp) + 1
        current = len(self._pp_results) + 1
        self._show_status(f"Post-processing {current}/{total}: {Path(path).name}")

        upscaler_dir = self.state.global_config.model_paths.get("upscaler_dir", "")
        face_models_dir = self.state.global_config.model_paths.get("face_models_dir", "")

        from sdqt.workers.image_postprocess import ImagePostProcessWorker
        worker = ImagePostProcessWorker(
            source_path=path,
            ai_denoise=self._pp_denoise.currentIndex() > 0,
            ai_sharpen=self._pp_sharpen.currentIndex() > 0,
            ai_enhance=self._pp_upscale.currentIndex() > 0,
            ai_face_restore=self._pp_face.currentIndex() > 0,
            ai_tile=self._pp_tile.value(),
            upscaler_dir=upscaler_dir,
            face_models_dir=face_models_dir,
            parent=self,
        )
        worker.progress.connect(lambda f, d: self._show_status(d))
        worker.finished_ok.connect(self._on_pp_done)
        worker.error.connect(lambda msg: (
            self._pp_results.append(path),  # keep original on error
            logger.warning("PP failed for %s: %s", path, msg),
            self._run_next_pp(),
        ))
        worker.finished.connect(worker.deleteLater)
        self._pp_worker = worker
        worker.start()

    def _on_pp_done(self, result_path: str) -> None:
        self._pp_results.append(result_path)
        self._run_next_pp()

    def _show_results(self, paths: list[str]) -> None:
        self._grab_btn.setVisible(True)
        self._abort_btn.setVisible(False)

        self._results.clear()
        from PySide6.QtCore import QSize
        for p in paths:
            pix = QPixmap(p)
            if pix.isNull():
                continue
            thumb = pix.scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item = QListWidgetItem()
            item.setIcon(QIcon(thumb))
            item.setText(Path(p).name)
            item.setData(Qt.UserRole, p)
            item.setToolTip(p)
            self._results.addItem(item)

        self._open_folder_btn.setEnabled(bool(paths))
        self._show_status(f"{len(paths)} frame(s) ready in {self._output_dir}")

    def _on_error(self, msg: str) -> None:
        self._grab_btn.setVisible(True)
        self._abort_btn.setVisible(False)
        self._show_status(f"Error: {msg}")

    @Slot()
    def _on_open_folder(self) -> None:
        if self._output_dir and Path(self._output_dir).is_dir():
            try:
                if os.name == "nt":
                    subprocess.Popen(["explorer", self._output_dir])
                elif os.name == "posix":
                    subprocess.Popen(["xdg-open", self._output_dir])
                else:
                    subprocess.Popen(["open", self._output_dir])
            except Exception:
                logger.warning("Failed to open folder", exc_info=True)

    # ── Context menu / send-to ──────────────────────────────────────────────

    def _selected_result_path(self) -> str | None:
        item = self._results.currentItem()
        if item:
            return item.data(Qt.UserRole)
        return None

    def _on_result_context_menu(self, pos) -> None:
        path = self._selected_result_path()
        if not path:
            return

        from sdqt.widgets.send_targets import IMAGE_TARGETS_SIMPLE, build_target_menu

        menu = QMenu(self)
        build_target_menu(menu, "Send to", IMAGE_TARGETS_SIMPLE,
                          lambda key: self.send_image_requested.emit(key, path))
        menu.addSeparator()
        menu.addAction("Open in File Manager", lambda: self._open_file(path))
        menu.exec(self._results.mapToGlobal(pos))

    def _open_file(self, path: str) -> None:
        import subprocess as _sp
        try:
            if os.name == "nt":
                _sp.Popen(["explorer", "/select,", path])
            else:
                _sp.Popen(["xdg-open", str(Path(path).parent)])
        except Exception:
            pass

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        self._results.clear()
        self._open_folder_btn.setEnabled(False)
