"""Trim & Crop tabs -- separate video trimming and cropping utilities."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot, QPointF, QRectF, QSizeF
from PySide6.QtGui import QBrush, QColor, QCursor, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from supremediffusion.config.project_config import ProjectConfig

from sdqt.state import AppState
from sdqt.utils.naming import cap_stem
from sdqt.widgets.range_slider import RangeSliderWidget
from sdqt.widgets.video_player import VideoPlayerWidget
from sdqt.workers.trim_crop import CropWorker, TrimWorker

from .base import BaseTab

logger = logging.getLogger(__name__)

_CROP_PRESETS = [
    ("512x512", 512, 512),
    ("640x480", 640, 480),
    ("720x480", 720, 480),
    ("768x512", 768, 512),
    ("832x480", 832, 480),
    ("480x720", 480, 720),
    ("512x768", 512, 768),
    ("480x832", 480, 832),
    ("1024x576", 1024, 576),
    ("576x1024", 576, 1024),
    ("1280x720", 1280, 720),
    ("720x1280", 720, 1280),
]


# ══════════════════════════════════════════════════════════════════════
#  TrimTab
# ══════════════════════════════════════════════════════════════════════

class TrimTab(BaseTab):
    """Video trimming with range slider and time display."""

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._worker: TrimWorker | None = None
        self._source_info: dict = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 2)
        layout.setSpacing(4)

        # ── Video players side by side ────────────────────────────────
        video_splitter = QSplitter(Qt.Horizontal)
        self._source_player = VideoPlayerWidget("Source Video")
        self._source_player.video_loaded.connect(self._on_source_dropped)
        self._source_player.video_cleared.connect(self._on_source_cleared)
        video_splitter.addWidget(self._source_player)
        self._result_player = VideoPlayerWidget("Result")
        video_splitter.addWidget(self._result_player)
        video_splitter.setStretchFactor(0, 1)
        video_splitter.setStretchFactor(1, 1)
        # ── Range slider ──────────────────────────────────────────────
        trim_group = QGroupBox("Trim Range")
        trim_gl = QVBoxLayout(trim_group)
        trim_gl.setContentsMargins(4, 6, 4, 6)
        trim_gl.setSpacing(6)

        self._range_slider = RangeSliderWidget()
        self._range_slider.range_changed.connect(self._on_range_changed)
        self._range_slider.begin_dragging.connect(self._on_begin_drag)
        self._range_slider.end_dragging.connect(self._on_end_drag)
        trim_gl.addWidget(self._range_slider)

        range_row = QHBoxLayout()
        range_row.setSpacing(8)
        _thumb_ss = ("background: #1a1a1a; border: 1px solid #444; "
                     "color: #999; font-size: 14px; font-weight: bold;")
        self._begin_thumb = QLabel("IN")
        self._begin_thumb.setFixedSize(100, 60)
        self._begin_thumb.setAlignment(Qt.AlignCenter)
        self._begin_thumb.setStyleSheet(_thumb_ss)
        self._begin_thumb.setScaledContents(True)
        range_row.addWidget(self._begin_thumb)
        self._range_begin_label = QLabel("Begin: 0.0s")
        range_row.addWidget(self._range_begin_label)
        range_row.addStretch()
        self._range_duration_label = QLabel("Duration: 0.0s")
        range_row.addWidget(self._range_duration_label)
        range_row.addStretch()
        self._range_end_label = QLabel("End: 0.0s")
        range_row.addWidget(self._range_end_label)
        self._end_thumb = QLabel("OUT")
        self._end_thumb.setFixedSize(100, 60)
        self._end_thumb.setAlignment(Qt.AlignCenter)
        self._end_thumb.setStyleSheet(_thumb_ss)
        self._end_thumb.setScaledContents(True)
        range_row.addWidget(self._end_thumb)
        trim_gl.addLayout(range_row)

        # Pack video + trim range into a top pane
        top_pane = QWidget()
        top_pane_layout = QVBoxLayout(top_pane)
        top_pane_layout.setContentsMargins(0, 0, 0, 0)
        top_pane_layout.setSpacing(2)
        top_pane_layout.addWidget(video_splitter, 1)
        top_pane_layout.addWidget(trim_group)

        # ── Bottom controls ───────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls = QWidget()
        ctrl_layout = QVBoxLayout(controls)
        ctrl_layout.setContentsMargins(4, 4, 4, 4)

        self._info_label = QLabel("No source loaded.")
        ctrl_layout.addWidget(self._info_label)

        btn_row = QHBoxLayout()
        self._trim_btn = QPushButton("Trim Video")
        self._trim_btn.setObjectName("primary")
        self._trim_btn.setMinimumWidth(150)
        self._trim_btn.clicked.connect(self._on_trim)
        btn_row.addWidget(self._trim_btn)
        btn_row.addStretch()
        ctrl_layout.addLayout(btn_row)

        self._status = QLabel("")
        ctrl_layout.addWidget(self._status)

        # Save to project + frame capture
        actions_row = QHBoxLayout()

        save_proj_group = QGroupBox("Save to project")
        sp_layout = QHBoxLayout(save_proj_group)
        self._save_name = QLineEdit()
        self._save_name.setPlaceholderText("Name...")
        sp_layout.addWidget(self._save_name, 1)
        self._save_to_project_btn = QPushButton("Save")
        self._save_to_project_btn.clicked.connect(self._on_save_to_project)
        sp_layout.addWidget(self._save_to_project_btn)
        actions_row.addWidget(save_proj_group)

        capture_group = QGroupBox("Capture frame at playhead")
        capture_layout = QHBoxLayout(capture_group)
        dl_btn = QPushButton("Download Frame")
        dl_btn.clicked.connect(self._on_download_frame)
        capture_layout.addWidget(dl_btn)
        save_btn = QPushButton("Save Frame to Project")
        save_btn.clicked.connect(self._on_save_frame)
        capture_layout.addWidget(save_btn)
        actions_row.addWidget(capture_group)

        ctrl_layout.addLayout(actions_row)
        ctrl_layout.addStretch()
        scroll.setWidget(controls)

        # Vertical splitter: video+trim on top, controls on bottom
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(5)
        main_splitter.setStyleSheet(
            "QSplitter::handle { background: #3a3a3a; }"
            "QSplitter::handle:hover { background: #888; }"
        )
        main_splitter.addWidget(top_pane)
        main_splitter.addWidget(scroll)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)
        layout.addWidget(main_splitter, 1)

    # -- Range slider events -----------------------------------------------

    def _on_range_changed(self, begin: float, end: float) -> None:
        duration = self._source_info.get("duration", 0.0)
        b_sec = begin * duration
        e_sec = end * duration
        self._range_begin_label.setText(f"Begin: {b_sec:.2f}s")
        self._range_end_label.setText(f"End: {e_sec:.2f}s")
        self._range_duration_label.setText(f"Duration: {e_sec - b_sec:.2f}s")

    def _on_begin_drag(self, val: float) -> None:
        duration = self._source_info.get("duration", 0.0)
        sec = val * duration
        self._source_player._player.setPosition(int(sec * 1000))
        self._extract_thumb(sec, self._begin_thumb)

    def _on_end_drag(self, val: float) -> None:
        duration = self._source_info.get("duration", 0.0)
        sec = val * duration
        self._source_player._player.setPosition(int(sec * 1000))
        self._extract_thumb(sec, self._end_thumb)

    def _extract_thumb(self, sec: float, label: QLabel) -> None:
        path = self._source_player.video_path
        if not path:
            return
        try:
            from supremediffusion.utils.video import extract_single_frame
            fps = self._source_info.get("fps", 16)
            frame_num = max(0, int(sec * fps))
            img = extract_single_frame(path, frame_num)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(tmp.name)
            tmp.close()
            pixmap = QPixmap(tmp.name)
            if not pixmap.isNull():
                label.setPixmap(pixmap)
            Path(tmp.name).unlink(missing_ok=True)
        except Exception:
            pass

    def _get_range_seconds(self) -> tuple[float, float]:
        duration = self._source_info.get("duration", 0.0)
        return (
            self._range_slider.begin() * duration,
            self._range_slider.end() * duration,
        )

    # -- Source loading ----------------------------------------------------

    def _on_source_dropped(self, path: str) -> None:
        self._probe_source(path)
        try:
            cfg = ProjectConfig.load(self.project_path)
            cfg.tc_source_path = path
            cfg.save(self.project_path)
        except Exception:
            pass

    def _on_source_cleared(self) -> None:
        self._source_info = {}
        try:
            cfg = ProjectConfig.load(self.project_path)
            cfg.tc_source_path = ""
            cfg.save(self.project_path)
        except Exception:
            pass

    def load_source(self, path: str) -> None:
        self._result_player.clear_video()
        self._source_player.load_video(path)
        self._probe_source(path)

    def _probe_source(self, path: str) -> None:
        try:
            from supremediffusion.utils.video import probe_video
            info = probe_video(path)
            info["_path"] = path
            self._source_info = info
            self._info_label.setText(
                f"Duration: {info['duration']:.1f}s  |  "
                f"FPS: {info['fps']:.1f}  |  "
                f"Resolution: {info['width']}\u00d7{info['height']}"
            )
            self._range_slider.set_range(0.0, 1.0)
            self._on_range_changed(0.0, 1.0)
            self._extract_thumb(0.0, self._begin_thumb)
            self._extract_thumb(info["duration"], self._end_thumb)
        except Exception as exc:
            self._info_label.setText(f"Error: {exc}")
            self._source_info = {}

    @Slot()
    def _on_trim(self) -> None:
        src = self._source_player.video_path
        if not src:
            self._show_status("No source video.")
            return

        start_sec, end_sec = self._get_range_seconds()
        if end_sec <= start_sec:
            self._show_status("End must be after Begin.")
            return

        trims_dir = self.project_path / "clips" / "trims"
        trims_dir.mkdir(parents=True, exist_ok=True)

        name = cap_stem(Path(src).stem)
        output = str(trims_dir / f"{name}_trimmed.mp4")

        self._trim_btn.setEnabled(False)
        self._show_status("Trimming...")

        worker = TrimWorker(
            src, output, start_sec, end_sec, parent=self,
        )
        worker.progress.connect(lambda f, d: self._show_status(d))
        worker.finished_ok.connect(self._on_trim_done)
        worker.error.connect(self._on_error)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_trim_done(self, result: str) -> None:
        self._trim_btn.setEnabled(True)
        self._result_player.load_video(result, auto_play=True)
        self._show_status(f"Trimmed: {Path(result).name}")

        try:
            cfg = ProjectConfig.load(self.project_path)
            cfg.trim_video_path = result
            cfg.save(self.project_path)
        except Exception:
            pass

    @Slot()
    def _on_save_to_project(self) -> None:
        player = self._result_player if self._result_player.video_path else None
        vid = player.video_path if player else self._source_player.video_path
        if not vid or not Path(vid).is_file():
            self._show_status("No video to save.")
            return
        name = self._save_name.text().strip()
        if not name:
            name = cap_stem(Path(vid).stem)
        trims_dir = self.project_path / "clips" / "trims"
        trims_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(vid).suffix
        dest = trims_dir / f"{name}{ext}"
        idx = 1
        while dest.exists():
            dest = trims_dir / f"{name}_{idx}{ext}"
            idx += 1
        shutil.copy2(vid, dest)
        self._show_status(f"Saved: {dest.name}")

    def _on_error(self, msg: str) -> None:
        self._trim_btn.setEnabled(True)
        self._show_status(f"Error: {msg}")

    @Slot()
    def _on_download_frame(self) -> None:
        player = self._result_player if self._result_player.video_path else self._source_player
        path = player.capture_frame()
        if path:
            self.download_frame(path)

    @Slot()
    def _on_save_frame(self) -> None:
        player = self._result_player if self._result_player.video_path else self._source_player
        path = player.capture_frame()
        if path:
            saved = self.save_frame_to_project(path)
            if saved:
                self._show_status("Saved frame to project.")

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        self._source_player.clear_video()
        self._result_player.clear_video()

        try:
            cfg = ProjectConfig.load(self.project_path)
        except Exception:
            return
        if cfg.tc_source_path and Path(cfg.tc_source_path).is_file():
            self.load_source(cfg.tc_source_path)
        if cfg.trim_video_path and Path(cfg.trim_video_path).is_file():
            self._result_player.load_video(cfg.trim_video_path)


# ══════════════════════════════════════════════════════════════════════
#  CropPreviewScene -- draggable crop rectangle on a frame
# ══════════════════════════════════════════════════════════════════════

class _CropHandle(QGraphicsEllipseItem):
    """Small circular handle for resizing / moving."""

    _SIZE = 10

    def __init__(self, parent_rect: "_CropRectItem") -> None:
        super().__init__(-self._SIZE / 2, -self._SIZE / 2, self._SIZE, self._SIZE)
        self._rect_item = parent_rect
        self.setBrush(QBrush(QColor(255, 255, 255, 200)))
        self.setPen(QPen(QColor(0, 0, 0), 1))
        self.setFlag(self.GraphicsItemFlag.ItemIsMovable, False)
        self.setFlag(self.GraphicsItemFlag.ItemSendsGeometryChanges, False)
        self.setCursor(QCursor(Qt.SizeAllCursor))
        self.setZValue(10)


class _CropRectItem(QGraphicsRectItem):
    """Draggable rectangle that stays within scene bounds."""

    def __init__(self, scene_w: float, scene_h: float, crop_w: int, crop_h: int) -> None:
        super().__init__(0, 0, crop_w, crop_h)
        self._scene_w = scene_w
        self._scene_h = scene_h
        self.setPen(QPen(QColor(0, 200, 255), 2, Qt.DashLine))
        self.setBrush(QBrush(QColor(0, 200, 255, 30)))
        self.setFlag(self.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(self.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setCursor(QCursor(Qt.SizeAllCursor))
        self.setZValue(5)
        # Center initially
        x = max(0, (scene_w - crop_w) / 2)
        y = max(0, (scene_h - crop_h) / 2)
        self.setPos(x, y)

    def itemChange(self, change, value):
        if change == self.GraphicsItemChange.ItemPositionChange and self.scene():
            r = self.rect()
            x = max(0, min(value.x(), self._scene_w - r.width()))
            y = max(0, min(value.y(), self._scene_h - r.height()))
            return QPointF(x, y)
        return super().itemChange(change, value)

    def update_bounds(self, scene_w: float, scene_h: float) -> None:
        self._scene_w = scene_w
        self._scene_h = scene_h
        # Clamp position
        r = self.rect()
        x = max(0, min(self.x(), scene_w - r.width()))
        y = max(0, min(self.y(), scene_h - r.height()))
        self.setPos(x, y)

    def set_crop_size(self, w: int, h: int) -> None:
        """Update crop rectangle size while keeping it centered and in-bounds."""
        old_center = QPointF(
            self.x() + self.rect().width() / 2,
            self.y() + self.rect().height() / 2,
        )
        cw = min(w, self._scene_w)
        ch = min(h, self._scene_h)
        self.setRect(0, 0, cw, ch)
        # Re-center around old center
        nx = max(0, min(old_center.x() - cw / 2, self._scene_w - cw))
        ny = max(0, min(old_center.y() - ch / 2, self._scene_h - ch))
        self.setPos(nx, ny)

    def crop_rect_in_scene(self) -> tuple[int, int, int, int]:
        """Return (x, y, w, h) in scene (image) coordinates."""
        return (
            int(round(self.x())),
            int(round(self.y())),
            int(round(self.rect().width())),
            int(round(self.rect().height())),
        )


_CROP_DROP_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm",
                   ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}


class CropPreviewView(QGraphicsView):
    """QGraphicsView showing a video frame with a draggable crop rectangle."""

    crop_changed = Signal(int, int, int, int)  # x, y, w, h
    file_dropped = Signal(str)  # emitted when a video/image is dropped

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")
        self.setRenderHints(self.renderHints())
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumHeight(200)
        self.setAcceptDrops(True)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._crop_rect: _CropRectItem | None = None
        self._frame_w: int = 0
        self._frame_h: int = 0
        self._scale: float = 1.0

    # -- Drop handling --

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in _CROP_DROP_EXTS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in _CROP_DROP_EXTS:
                self.file_dropped.emit(path)
                event.acceptProposedAction()
                return

    def set_frame(self, pixmap: QPixmap) -> None:
        """Display a frame and fit the view."""
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._frame_w = pixmap.width()
        self._frame_h = pixmap.height()
        self._scene.setSceneRect(0, 0, self._frame_w, self._frame_h)
        self._crop_rect = None
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def set_crop_size(self, w: int, h: int) -> None:
        """Set or update the crop rectangle on the preview."""
        if self._frame_w == 0 or self._frame_h == 0:
            return
        if self._crop_rect is None:
            self._crop_rect = _CropRectItem(self._frame_w, self._frame_h, w, h)
            self._scene.addItem(self._crop_rect)
        else:
            self._crop_rect.update_bounds(self._frame_w, self._frame_h)
            self._crop_rect.set_crop_size(w, h)

    def get_crop_rect(self) -> tuple[int, int, int, int] | None:
        """Return (x, y, w, h) in original frame coordinates, or None."""
        if self._crop_rect is None:
            return None
        return self._crop_rect.crop_rect_in_scene()

    def clear_frame(self) -> None:
        self._scene.clear()
        self._pixmap_item = None
        self._crop_rect = None
        self._frame_w = 0
        self._frame_h = 0

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._frame_w > 0:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)


# ══════════════════════════════════════════════════════════════════════
#  CropTab
# ══════════════════════════════════════════════════════════════════════

class CropTab(BaseTab):
    """Video and image cropping with visual rectangle preview and preset sizes."""

    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}
    _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._worker: CropWorker | None = None
        self._source_info: dict = {}
        self._source_path: str = ""
        self._is_image: bool = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 2)
        layout.setSpacing(1)

        # ── Top: crop preview + result side by side ───────────────────
        top_splitter = QSplitter(Qt.Horizontal)

        # Left: crop preview (the source IS the crop region)
        left_w = QWidget()
        left_layout = QVBoxLayout(left_w)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self._source_label = QLabel("Drop a video or image here, or use Send-To")
        self._source_label.setStyleSheet("font-size: 13px; color: #bbb; padding: 2px;")
        left_layout.addWidget(self._source_label)

        self._crop_preview = CropPreviewView()
        self._crop_preview.setMinimumHeight(300)
        self._crop_preview.file_dropped.connect(self.load_source)
        left_layout.addWidget(self._crop_preview, 1)

        top_splitter.addWidget(left_w)

        # Right: result player (video or image depending on source)
        self._result_player = VideoPlayerWidget("Cropped Result")
        top_splitter.addWidget(self._result_player)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 1)
        # ── Bottom controls ───────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls = QWidget()
        ctrl_layout = QVBoxLayout(controls)
        ctrl_layout.setContentsMargins(8, 8, 8, 8)
        ctrl_layout.setSpacing(8)

        self._info_label = QLabel("No source loaded.")
        ctrl_layout.addWidget(self._info_label)

        # Preset + crop button row
        crop_row = QHBoxLayout()
        crop_row.setSpacing(8)
        crop_row.addWidget(QLabel("Crop Size:"))
        self._crop_preset = QComboBox()
        self._crop_preset.setMinimumWidth(160)
        self._crop_preset.setMaxVisibleItems(14)
        for label, w, h in _CROP_PRESETS:
            self._crop_preset.addItem(label, (w, h))
        self._crop_preset.currentIndexChanged.connect(self._on_preset_changed)
        crop_row.addWidget(self._crop_preset)

        crop_row.addStretch()

        self._crop_btn = QPushButton("Crop")
        self._crop_btn.setObjectName("primary")
        self._crop_btn.setMinimumWidth(150)
        self._crop_btn.clicked.connect(self._on_crop)
        crop_row.addWidget(self._crop_btn)
        ctrl_layout.addLayout(crop_row)
        ctrl_layout.addSpacing(12)

        self._status = QLabel("")
        ctrl_layout.addWidget(self._status)

        # Save to project
        save_group = QGroupBox("Save to project")
        save_row = QHBoxLayout(save_group)
        save_row.setContentsMargins(8, 6, 8, 6)
        save_row.setSpacing(8)
        self._save_name = QLineEdit()
        self._save_name.setPlaceholderText("Save name...")
        save_row.addWidget(self._save_name, 1)
        self._save_to_project_btn = QPushButton("Save to Project")
        self._save_to_project_btn.clicked.connect(self._on_save_to_project)
        save_row.addWidget(self._save_to_project_btn)
        ctrl_layout.addWidget(save_group)

        ctrl_layout.addStretch()
        scroll.setWidget(controls)

        # Vertical splitter: preview on top, controls on bottom
        crop_splitter = QSplitter(Qt.Orientation.Vertical)
        crop_splitter.setChildrenCollapsible(False)
        crop_splitter.setHandleWidth(5)
        crop_splitter.setStyleSheet(
            "QSplitter::handle { background: #3a3a3a; }"
            "QSplitter::handle:hover { background: #888; }"
        )
        crop_splitter.addWidget(top_splitter)
        crop_splitter.addWidget(scroll)
        crop_splitter.setStretchFactor(0, 3)
        crop_splitter.setStretchFactor(1, 1)
        layout.addWidget(crop_splitter, 1)

        # Apply initial preset
        self._on_preset_changed(0)

    # -- Source loading --------------------------------------------------------

    def _detect_source_type(self, path: str) -> bool:
        """Returns True if image, False if video."""
        return Path(path).suffix.lower() in self._IMAGE_EXTS

    def load_source(self, path: str) -> None:
        """Load a video or image as the crop source."""
        if not path or not Path(path).is_file():
            return
        self._result_player.clear_video()
        self._source_path = path
        self._is_image = self._detect_source_type(path)

        if self._is_image:
            self._load_image_source(path)
        else:
            self._load_video_source(path)

    def _load_image_source(self, path: str) -> None:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._show_status("Failed to load image.")
            return
        self._source_info = {
            "width": pixmap.width(),
            "height": pixmap.height(),
            "_path": path,
        }
        self._source_label.setText(f"Image: {Path(path).name}")
        self._info_label.setText(
            f"Resolution: {pixmap.width()}\u00d7{pixmap.height()}"
        )
        self._crop_preview.set_frame(pixmap)
        self._apply_current_preset()

    def _load_video_source(self, path: str) -> None:
        try:
            from supremediffusion.utils.video import probe_video, extract_single_frame
            info = probe_video(path)
            info["_path"] = path
            self._source_info = info
            self._source_label.setText(f"Video: {Path(path).name}")
            self._info_label.setText(
                f"Duration: {info['duration']:.1f}s  |  "
                f"FPS: {info['fps']:.1f}  |  "
                f"Resolution: {info['width']}\u00d7{info['height']}"
            )
            # Extract first frame for crop preview
            img = extract_single_frame(path, 0)
            if img:
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                img.save(tmp.name)
                tmp.close()
                pixmap = QPixmap(tmp.name)
                Path(tmp.name).unlink(missing_ok=True)
                if not pixmap.isNull():
                    self._crop_preview.set_frame(pixmap)
                    self._apply_current_preset()
        except Exception as exc:
            self._info_label.setText(f"Error: {exc}")
            self._source_info = {}

    def _apply_current_preset(self) -> None:
        idx = self._crop_preset.currentIndex()
        if idx >= 0:
            data = self._crop_preset.itemData(idx)
            if data:
                w, h = data
                self._crop_preview.set_crop_size(w, h)

    def _on_preset_changed(self, index: int) -> None:
        if index < 0:
            return
        data = self._crop_preset.itemData(index)
        if data:
            w, h = data
            self._crop_preview.set_crop_size(w, h)

    @Slot()
    def _on_crop(self) -> None:
        if not self._source_path or not Path(self._source_path).is_file():
            self._show_status("No source loaded.")
            return

        rect = self._crop_preview.get_crop_rect()
        if rect is None:
            self._show_status("No crop region set.")
            return

        x, y, w, h = rect
        if w < 1 or h < 1:
            self._show_status("Invalid crop dimensions.")
            return

        trims_dir = self.project_path / "clips" / "trims"
        trims_dir.mkdir(parents=True, exist_ok=True)
        name = cap_stem(Path(self._source_path).stem)

        if self._is_image:
            self._crop_image(x, y, w, h, name, trims_dir)
        else:
            self._crop_video(x, y, w, h, name, trims_dir)

    def _crop_image(self, x, y, w, h, name, trims_dir) -> None:
        """Crop an image using PIL."""
        try:
            from PIL import Image as PILImage
            img = PILImage.open(self._source_path)
            cropped = img.crop((x, y, x + w, y + h))
            output = str(trims_dir / f"{name}_cropped.png")
            cropped.save(output)
            self._show_status(f"Cropped: {Path(output).name}")
            # Show result in crop preview (no video player for images)
            pixmap = QPixmap(output)
            if not pixmap.isNull():
                self._result_player.clear_video()
                # For images, just update the status — result is the saved file
                self._show_status(f"Saved cropped image: {Path(output).name}")
        except Exception as exc:
            self._show_status(f"Crop failed: {exc}")

    def _crop_video(self, x, y, w, h, name, trims_dir) -> None:
        """Crop a video using ffmpeg."""
        output = str(trims_dir / f"{name}_cropped.mp4")
        self._crop_btn.setEnabled(False)
        self._show_status(f"Cropping to {w}x{h} at ({x}, {y})...")

        worker = CropWorker(
            self._source_path, output, x, y, w, h, parent=self,
        )
        worker.progress.connect(lambda f, d: self._show_status(d))
        worker.finished_ok.connect(self._on_crop_done)
        worker.error.connect(self._on_error)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_crop_done(self, result: str) -> None:
        self._crop_btn.setEnabled(True)
        self._result_player.load_video(result, auto_play=True)
        self._show_status(f"Cropped: {Path(result).name}")
        try:
            cfg = ProjectConfig.load(self.project_path)
            cfg.crop_video_path = result
            cfg.save(self.project_path)
        except Exception:
            pass

    @Slot()
    def _on_save_to_project(self) -> None:
        vid = self._result_player.video_path or self._source_path
        if not vid or not Path(vid).is_file():
            self._show_status("Nothing to save.")
            return
        name = self._save_name.text().strip()
        if not name:
            name = cap_stem(Path(vid).stem)
        trims_dir = self.project_path / "clips" / "trims"
        trims_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(vid).suffix
        dest = trims_dir / f"{name}{ext}"
        idx = 1
        while dest.exists():
            dest = trims_dir / f"{name}_{idx}{ext}"
            idx += 1
        shutil.copy2(vid, dest)
        self._show_status(f"Saved: {dest.name}")

    def _on_error(self, msg: str) -> None:
        self._crop_btn.setEnabled(True)
        self._show_status(f"Error: {msg}")

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        self._result_player.clear_video()
        self._crop_preview.clear_frame()
        self._source_path = ""
        self._is_image = False
        self._source_label.setText("Drop a video or image here, or use Send-To")
        self._info_label.setText("No source loaded.")

        try:
            cfg = ProjectConfig.load(self.project_path)
        except Exception:
            return
        if cfg.tc_source_path and Path(cfg.tc_source_path).is_file():
            self.load_source(cfg.tc_source_path)
        if cfg.crop_video_path and Path(cfg.crop_video_path).is_file():
            self._result_player.load_video(cfg.crop_video_path)


# Backwards compat alias -- old code imported TrimCropTab
TrimCropTab = TrimTab
