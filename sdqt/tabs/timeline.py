"""Timeline tab — multitrack clip library, timeline, preview, transport, and export."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot, QPoint, QTimer
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from supremediffusion.config.project_config import ProjectConfig

from sdqt.state import AppState
from sdqt.utils.naming import cap_stem
from sdqt.widgets.clip_library import ClipLibraryWidget
from sdqt.widgets.multitrack_widget import MultiTrackWidget
from sdqt.widgets.timeline_track import (
    TimelineClip,
    TimelineMarker,
    TimelineTrack,
    TrackType,
    ToolMode,
)
from sdqt.preview import HAS_OPENGL, HAS_PYAV
from sdqt.preview.frame_cache import FrameCache
from sdqt.widgets.gl_preview import TimelineGLPreview

from sdqt.workers.base import BaseWorker
from sdqt.workers.timeline import TimelineExportWorker, _probe_has_audio

from .base import BaseTab
from sdqt.utils.codec import (
    pix_fmt_args,
    configured_codec_args as _codec_args,
    configured_encoder_name as _enc_name,
    codec_labels,
    get_codec_args,
)

# Mixin imports — extracted modules
from sdqt.tabs.timeline_clip_ops import TimelineClipOps
from sdqt.tabs.timeline_context_menus import TimelineContextMenus
from sdqt.tabs.timeline_export import ExportDialog, TimelineExportMixin
from sdqt.tabs.timeline_match_grade import TimelineMatchGradeMixin
from sdqt.tabs.timeline_dialogs import SpeedDialog as _SpeedDialog, PostProcessDialog as _PostProcessDialog

logger = logging.getLogger(__name__)



# ExportDialog → sdqt.tabs.timeline_export
# _SpeedDialog → sdqt.tabs.timeline_dialogs.SpeedDialog
# _PostProcessDialog → sdqt.tabs.timeline_dialogs.PostProcessDialog


class _UndoStack:
    """Simple undo/redo stack for timeline state snapshots."""

    def __init__(self, max_size: int = 50) -> None:
        self._undo: list[dict] = []
        self._redo: list[dict] = []
        self._max = max_size

    def push(self, state: dict) -> None:
        self._undo.append(state)
        if len(self._undo) > self._max:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self) -> dict | None:
        return self._undo.pop() if self._undo else None

    def push_redo(self, state: dict) -> None:
        self._redo.append(state)
        if len(self._redo) > self._max:
            self._redo.pop(0)

    def redo(self) -> dict | None:
        return self._redo.pop() if self._redo else None

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()



# _MatchGradeWorker → sdqt.tabs.timeline_match_grade


class TimelineTab(
    TimelineClipOps,
    TimelineContextMenus,
    TimelineExportMixin,
    TimelineMatchGradeMixin,
    BaseTab,
):
    """Video editing timeline with multitrack support."""

    # Emitted when a clip should be sent to another video tab
    send_clip_to = Signal(str, str)    # (target_key, clip_path)
    # Emitted when an extracted frame should be sent to an image tab
    send_frame_to = Signal(str, str)   # (target_key, frame_png_path)
    # Emitted when an extracted frame should be used as a final frame
    send_final_frame_to = Signal(str, str)  # (target_key, frame_png_path)
    # Emitted when an extracted frame should become a guide video
    send_guide_video_to = Signal(str, str)  # (target_key, frame_png_path)
    # Emitted when an extracted frame should be used as color correct reference
    send_cc_ref_to = Signal(str)  # frame_png_path
    # Emitted when a clip should be sent to Color Correct as source (with replace support)
    send_cc_source_from_timeline = Signal(str)  # clip_path

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._export_worker: TimelineExportWorker | None = None
        self._clip_targets: list[tuple[str, str]] = []
        self._frame_targets: list[tuple[str, str]] = []
        self._final_frame_targets: list[tuple[str, str]] = []
        self._guide_video_targets: list[tuple[str, str]] = []
        self._undo_stack = _UndoStack()
        self._pre_change_state: dict | None = None
        self._restoring_undo = False
        self._has_missing_clips = False
        self._markers: list[TimelineMarker] = []
        self._frame_cache = FrameCache() if HAS_PYAV else None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 2)
        layout.setSpacing(1)

        # -- Top: (Library + Tools) | Preview --
        top_splitter = QSplitter(Qt.Horizontal)

        # Left column: library on top, tool panel on bottom
        left_panel = QSplitter(Qt.Vertical)
        left_panel.setMinimumWidth(220)
        left_panel.setMaximumWidth(400)

        # Version selector
        ver_layout = QVBoxLayout()
        ver_layout.setContentsMargins(4, 2, 4, 2)
        ver_layout.setSpacing(2)

        ver_top = QHBoxLayout()
        ver_top.setSpacing(4)
        ver_lbl = QLabel("Version:")
        ver_top.addWidget(ver_lbl)
        self._version_combo = QComboBox()
        self._version_combo.setMinimumWidth(80)
        self._version_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ver_top.addWidget(self._version_combo)
        ver_layout.addLayout(ver_top)

        # 2x2 grid so "Rename"/"Delete" are not truncated in the narrow panel.
        from PySide6.QtWidgets import QGridLayout
        ver_btns = QGridLayout()
        ver_btns.setSpacing(6)
        self._ver_save_btn = QPushButton("Save")
        self._ver_save_btn.setMinimumHeight(28)
        self._ver_save_btn.setToolTip("Save current canvas to this version")
        ver_btns.addWidget(self._ver_save_btn, 0, 0)
        self._ver_new_btn = QPushButton("New")
        self._ver_new_btn.setMinimumHeight(28)
        self._ver_new_btn.setToolTip("New version")
        ver_btns.addWidget(self._ver_new_btn, 0, 1)
        self._ver_rename_btn = QPushButton("Rename")
        self._ver_rename_btn.setMinimumHeight(28)
        self._ver_rename_btn.setToolTip("Rename version")
        ver_btns.addWidget(self._ver_rename_btn, 1, 0)
        self._ver_del_btn = QPushButton("Delete")
        self._ver_del_btn.setMinimumHeight(28)
        self._ver_del_btn.setToolTip("Delete version")
        ver_btns.addWidget(self._ver_del_btn, 1, 1)
        ver_layout.addLayout(ver_btns)

        ver_container = QWidget()
        ver_container.setLayout(ver_layout)
        left_panel.addWidget(ver_container)

        self._library = ClipLibraryWidget()
        left_panel.addWidget(self._library)

        self._tool_panel = self._build_tool_panel()
        # Wrap the tools/zoom/clip-actions/tracks column in a scroll area so it
        # SCROLLS when the splitter is short instead of compressing children
        # into each other (the cramped left panel can't show ~450px at once).
        _tool_scroll = QScrollArea()
        _tool_scroll.setWidgetResizable(True)
        _tool_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        _tool_scroll.setWidget(self._tool_panel)
        _tool_scroll.setMinimumHeight(200)
        left_panel.addWidget(_tool_scroll)

        # Library list takes the extra vertical space; version box stays compact,
        # tool column keeps its own (scrollable) size.
        left_panel.setStretchFactor(0, 0)
        left_panel.setStretchFactor(1, 1)
        left_panel.setStretchFactor(2, 0)
        top_splitter.addWidget(left_panel)

        # Preview area: GL preview (real-time decode + effects)
        if self._frame_cache and HAS_OPENGL and HAS_PYAV:
            self._gl_preview = TimelineGLPreview(self._frame_cache)
            self._gl_preview.position_changed.connect(self._on_gl_preview_position)
            top_splitter.addWidget(self._gl_preview)
        else:
            self._gl_preview = None
            from PySide6.QtWidgets import QLabel as _QL
            fallback = _QL("GL Preview unavailable.\nInstall PyAV and PyOpenGL.")
            fallback.setAlignment(Qt.AlignCenter)
            fallback.setStyleSheet("color: #888; background: black;")
            top_splitter.addWidget(fallback)

        top_splitter.setStretchFactor(0, 0)
        top_splitter.setStretchFactor(1, 1)
        top_splitter.setSizes([240, 600])

        # -- Multitrack timeline --
        self._multitrack = MultiTrackWidget()
        self._multitrack.persist_clip_fn = self._persist_clip
        self._multitrack.setMinimumHeight(200)
        if hasattr(self, "_deferred_zoom"):
            self._multitrack.set_zoom(float(self._deferred_zoom))

        # -- Vertical splitter: preview on top, tracks on bottom --
        self._vert_splitter = QSplitter(Qt.Vertical)
        self._vert_splitter.addWidget(top_splitter)
        self._vert_splitter.addWidget(self._multitrack)
        self._vert_splitter.setStretchFactor(0, 2)
        self._vert_splitter.setStretchFactor(1, 3)
        self._vert_splitter.setChildrenCollapsible(False)
        self._vert_splitter.setHandleWidth(4)
        layout.addWidget(self._vert_splitter, 1)

        # -- Transport controls --
        transport = QHBoxLayout()
        transport.setSpacing(8)
        transport.setContentsMargins(6, 6, 6, 6)

        btn_style = (
            "QPushButton { min-width: 32px; min-height: 24px; }"
        )

        self._btn_start = QPushButton("|<")
        self._btn_start.setToolTip("Go to start")
        self._btn_start.setStyleSheet(btn_style)
        self._btn_start.clicked.connect(self._go_start)
        transport.addWidget(self._btn_start)

        self._btn_prev = QPushButton("<<")
        self._btn_prev.setToolTip("Previous clip")
        self._btn_prev.setStyleSheet(btn_style)
        self._btn_prev.clicked.connect(self._go_prev_clip)
        transport.addWidget(self._btn_prev)

        self._btn_stop = QPushButton("\u25A0")
        self._btn_stop.setToolTip("Stop")
        self._btn_stop.setStyleSheet(btn_style)
        self._btn_stop.clicked.connect(self._stop)
        transport.addWidget(self._btn_stop)

        self._btn_play = QPushButton("\u25B6")
        self._btn_play.setToolTip("Play / Pause")
        self._btn_play.setStyleSheet(btn_style)
        self._btn_play.clicked.connect(self._play_pause)
        transport.addWidget(self._btn_play)

        self._btn_next = QPushButton(">>")
        self._btn_next.setToolTip("Next clip")
        self._btn_next.setStyleSheet(btn_style)
        self._btn_next.clicked.connect(self._go_next_clip)
        transport.addWidget(self._btn_next)

        self._btn_end = QPushButton(">|")
        self._btn_end.setToolTip("Go to end")
        self._btn_end.setStyleSheet(btn_style)
        self._btn_end.clicked.connect(self._go_end)
        transport.addWidget(self._btn_end)

        # FPS selector
        from PySide6.QtWidgets import QSpinBox as _QSB  # avoid polluting module namespace
        transport.addWidget(QLabel("FPS:"))
        self._fps_combo = QComboBox()
        self._fps_combo.setEditable(False)
        self._fps_combo.setMinimumWidth(90)
        for fps in (16, 24, 25, 30, 48, 60):
            self._fps_combo.addItem(str(fps), float(fps))
        self._fps_combo.setCurrentText("24")
        self._fps_combo.currentIndexChanged.connect(self._on_fps_changed)
        transport.addWidget(self._fps_combo)

        # Timecode display
        self._time_label = QLabel("00:00.0 / 00:00.0")
        self._time_label.setStyleSheet(
            "font-size: 13px; font-family: monospace; color: #ccc; padding: 0 6px;"
        )
        self._time_label.setMinimumWidth(150)
        transport.addWidget(self._time_label)

        # Duration summary
        self._duration_label = QLabel("")
        self._duration_label.setStyleSheet("font-size: 13px; color: #999;")
        transport.addWidget(self._duration_label)

        transport.addStretch()

        # Progress bar (for export)
        self._progress = QProgressBar()
        self._progress.setFixedHeight(18)
        self._progress.setFixedWidth(200)
        self._progress.setVisible(False)
        transport.addWidget(self._progress)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("font-size: 13px; color: #aaa;")
        transport.addWidget(self._status_label)

        transport.addStretch()

        # Volume control
        self._mute_btn = QPushButton("\U0001F50A")  # 🔊
        self._mute_btn.setFixedSize(24, 24)
        self._mute_btn.setToolTip("Mute / Unmute")
        self._mute_btn.setStyleSheet(
            "QPushButton { border: none; font-size: 14px; }"
            "QPushButton:hover { background: #3a3a3a; border-radius: 3px; }"
        )
        self._mute_btn.clicked.connect(self._toggle_mute)
        transport.addWidget(self._mute_btn)

        self._volume_slider = QSlider(Qt.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(50)
        self._volume_slider.setMinimumWidth(90)
        self._volume_slider.setFixedHeight(20)
        self._volume_slider.setToolTip("Volume")
        self._volume_slider.valueChanged.connect(self._on_volume_changed)
        transport.addWidget(self._volume_slider)

        self._btn_export = QPushButton("Export")
        self._btn_export.setObjectName("primary")
        self._btn_export.setToolTip("Export timeline as video")
        self._btn_export.setMinimumWidth(120)
        self._btn_export.clicked.connect(self._on_export)
        transport.addWidget(self._btn_export)

        layout.addLayout(transport)

        # -- Connect signals --
        self._library.add_to_timeline_requested.connect(self._add_library_clip_to_timeline)
        self._library.clip_context_menu_requested.connect(self._on_library_context_menu)
        self._library.clip_removed.connect(self._on_library_clip_removed)
        self._multitrack.clips_changed.connect(self._on_clips_changed)
        self._multitrack.clip_selected.connect(lambda *_: self._update_combine_button())
        self._multitrack.clips_changed.connect(self._update_combine_button)
        self._multitrack.playhead_moved.connect(self._on_playhead_moved)
        self._multitrack.clip_context_menu_requested.connect(self._on_track_context_menu)
        self._multitrack.multi_clip_context_menu_requested.connect(self._on_multi_clip_context_menu)
        self._multitrack.track_header_context_menu.connect(self._on_track_header_context_menu)
        self._multitrack._canvas.space_pressed.connect(self._play_pause)
        self._multitrack._canvas.clip_deleted.connect(self._check_remove_from_library)
        self._multitrack._canvas.razor_split.connect(self._on_razor_split)
        self._multitrack._canvas.marker_added.connect(self._on_marker_added)
        self._multitrack.zoom_changed.connect(self._on_zoom_changed)

        # Version controls
        self._version_combo.currentTextChanged.connect(self._on_version_switched)
        self._ver_save_btn.clicked.connect(self._save_version)
        self._ver_new_btn.clicked.connect(self._new_version)
        self._ver_rename_btn.clicked.connect(self._rename_version)
        self._ver_del_btn.clicked.connect(self._delete_version)

    # -- Tool panel --------------------------------------------------------

    @staticmethod
    def _make_timeline_tool_icon(key: str, size: int = 28) -> QIcon:
        """Paint a timeline tool icon."""
        pm = QPixmap(size, size)
        pm.fill(QColor(0, 0, 0, 0))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QColor(210, 210, 210)
        pen = QPen(c, 2.0)
        p.setPen(pen)

        if key == "select":
            # Arrow cursor
            p.setPen(QPen(c, 2.5))
            p.drawLine(6, 4, 6, 22)
            p.drawLine(6, 4, 14, 12)
            p.drawLine(6, 22, 14, 16)
            p.drawLine(14, 16, 14, 22)
            p.drawLine(14, 12, 20, 16)
            p.drawLine(14, 16, 20, 16)
        elif key == "scrub":
            # Playhead line with horizontal arrows
            p.setPen(QPen(c, 2.5))
            p.drawLine(14, 4, 14, 24)
            p.drawLine(14, 4, 10, 8)
            p.drawLine(14, 4, 18, 8)
            # Horizontal arrows (scrub direction)
            p.setPen(QPen(QColor(180, 180, 180), 1.5))
            p.drawLine(3, 14, 9, 14)
            p.drawLine(3, 14, 6, 11)
            p.drawLine(3, 14, 6, 17)
            p.drawLine(19, 14, 25, 14)
            p.drawLine(25, 14, 22, 11)
            p.drawLine(25, 14, 22, 17)
        elif key == "razor":
            # Scissors / blade
            p.setPen(QPen(c, 2.5))
            p.drawLine(14, 3, 14, 25)
            p.setPen(QPen(QColor(255, 100, 100), 2.0))
            p.drawLine(8, 8, 20, 20)
            p.drawLine(20, 8, 8, 20)
        elif key == "slip":
            # Clip box with content arrows inside
            p.setPen(QPen(c, 1.5))
            p.drawRect(4, 8, 20, 12)
            p.setPen(QPen(QColor(255, 200, 80), 2.0))
            p.drawLine(8, 14, 20, 14)
            p.drawLine(8, 14, 11, 11)
            p.drawLine(8, 14, 11, 17)
            p.drawLine(20, 14, 17, 11)
            p.drawLine(20, 14, 17, 17)
        elif key == "slide":
            # Clip box with external arrows
            p.setPen(QPen(c, 1.5))
            p.drawRect(8, 8, 12, 12)
            p.setPen(QPen(QColor(100, 200, 255), 2.0))
            p.drawLine(2, 14, 7, 14)
            p.drawLine(2, 14, 5, 11)
            p.drawLine(2, 14, 5, 17)
            p.drawLine(21, 14, 26, 14)
            p.drawLine(26, 14, 23, 11)
            p.drawLine(26, 14, 23, 17)

        p.end()
        return QIcon(pm)

    def _build_tool_panel(self) -> QWidget:
        from PySide6.QtWidgets import QGridLayout

        panel = QWidget()
        panel.setMinimumHeight(220)
        vlay = QVBoxLayout(panel)
        vlay.setContentsMargins(8, 6, 8, 6)
        vlay.setSpacing(8)

        hdr_row = QHBoxLayout()
        hdr_row.setSpacing(6)
        hdr = QLabel("<b>Tools</b>")
        hdr.setStyleSheet("font-size: 13px;")
        hdr_row.addWidget(hdr)
        hdr_row.addStretch()
        self._btn_shortcut_help = QPushButton("?")
        self._btn_shortcut_help.setFixedSize(22, 22)
        self._btn_shortcut_help.setStyleSheet(
            "QPushButton { font-size: 13px; font-weight: bold; border: 1px solid #555;"
            " border-radius: 11px; } QPushButton:hover { background: #3a3a3a; }"
        )
        self._btn_shortcut_help.setToolTip("Keyboard shortcuts")
        self._btn_shortcut_help.clicked.connect(self._show_shortcut_help)
        hdr_row.addWidget(self._btn_shortcut_help)
        vlay.addLayout(hdr_row)

        toggle_style = (
            "QPushButton { min-height: 34px; text-align: left; padding: 2px 8px;"
            "  border: 1px solid #555; border-radius: 4px; } "
            "QPushButton:checked { background: #4a90d9; border-color: #6ab0ff; } "
            "QPushButton:hover { background: #3a3a3a; } "
            "QPushButton:checked:hover { background: #5aa0e9; }"
        )
        _ICON = 24

        tool_defs = [
            ("select", "Select", "Select — click and drag clips"),
            ("scrub", "Scrub", "Scrub — click to position playhead"),
            ("razor", "Razor", "Razor — click to split clips"),
            ("slip", "Slip", "Slip (Y) — shift content within clip bounds"),
            ("slide", "Slide", "Slide (U) — move clip, adjust neighbors"),
        ]

        tool_grid = QGridLayout()
        tool_grid.setSpacing(4)

        btns = []
        for i, (key, label, tip) in enumerate(tool_defs):
            btn = QPushButton(f" {label}")
            btn.setIcon(self._make_timeline_tool_icon(key, _ICON))
            btn.setIconSize(QPixmap(_ICON, _ICON).size())
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.setStyleSheet(toggle_style)
            # Hard minimum (Python, not just CSS min-height) so a scroll area
            # cannot compress these to overlapping — single full-width column
            # reads cleanly in the narrow (~220px) left panel.
            btn.setMinimumHeight(34)
            tool_grid.addWidget(btn, i, 0)
            btns.append(btn)

        self._btn_select, self._btn_scrub, self._btn_razor, self._btn_slip, self._btn_slide = btns
        self._btn_select.setChecked(True)

        vlay.addLayout(tool_grid)

        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)
        self._tool_group.addButton(self._btn_select, ToolMode.SELECT)
        self._tool_group.addButton(self._btn_scrub, ToolMode.SCRUB)
        self._tool_group.addButton(self._btn_razor, ToolMode.RAZOR)
        self._tool_group.addButton(self._btn_slip, ToolMode.SLIP)
        self._tool_group.addButton(self._btn_slide, ToolMode.SLIDE)
        self._tool_group.idToggled.connect(self._on_tool_changed)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #444;")
        vlay.addWidget(sep)

        # Zoom controls
        zoom_lbl = QLabel("<b>Zoom</b>")
        zoom_lbl.setStyleSheet("font-size: 13px;")
        vlay.addWidget(zoom_lbl)

        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(8)

        saved_zoom = self.state.global_config.tl_clip_zoom or 80
        self._zoom_slider = QSlider(Qt.Horizontal)
        self._zoom_slider.setRange(10, 500)
        self._zoom_slider.setValue(saved_zoom)
        self._zoom_slider.setFixedWidth(100)
        self._zoom_slider.setFixedHeight(16)
        self._zoom_slider.setToolTip("Zoom level (Ctrl+Scroll)")
        self._zoom_slider.valueChanged.connect(self._on_zoom_slider)
        zoom_row.addWidget(self._zoom_slider)
        # NOTE: _multitrack doesn't exist yet — zoom is applied in _build_ui
        # after _multitrack is created via self._deferred_zoom = saved_zoom
        self._deferred_zoom = saved_zoom

        action_style = (
            "QPushButton { min-height: 26px; padding: 2px 6px;"
            "  border: 1px solid #555; border-radius: 3px; }"
            "QPushButton:hover { background: #3a3a3a; }"
            "QPushButton:pressed { background: #4a4a4a; }"
        )

        self._btn_zoom_fit = QPushButton("Fit")
        self._btn_zoom_fit.setToolTip("Zoom to fit (Ctrl+0)")
        self._btn_zoom_fit.setStyleSheet(action_style)
        self._btn_zoom_fit.setMinimumWidth(50)
        self._btn_zoom_fit.clicked.connect(lambda: self._multitrack.zoom_to_fit())
        zoom_row.addWidget(self._btn_zoom_fit)

        self._zoom_label = QLabel("100%")
        self._zoom_label.setStyleSheet("font-size: 13px; color: #aaa;")
        self._zoom_label.setMinimumWidth(50)
        zoom_row.addWidget(self._zoom_label)

        vlay.addLayout(zoom_row)

        # Separator
        sep1b = QFrame()
        sep1b.setFrameShape(QFrame.HLine)
        sep1b.setStyleSheet("color: #444;")
        vlay.addWidget(sep1b)

        # Ripple mode toggle (not part of tool group — it's a modifier)
        ripple_row = QHBoxLayout()
        ripple_row.setSpacing(4)

        self._btn_ripple = QCheckBox("Ripple")
        self._btn_ripple.setToolTip("Ripple edit mode — delete/trim shifts subsequent clips")
        self._btn_ripple.toggled.connect(self._on_ripple_toggled)
        ripple_row.addWidget(self._btn_ripple)

        self._btn_ripple_all = QCheckBox("All Tracks")
        self._btn_ripple_all.setToolTip("Apply ripple to all tracks")
        self._btn_ripple_all.setEnabled(False)
        self._btn_ripple_all.toggled.connect(self._on_ripple_all_toggled)
        ripple_row.addWidget(self._btn_ripple_all)

        ripple_row.addStretch()
        vlay.addLayout(ripple_row)

        # Snap toggle
        snap_row = QHBoxLayout()
        snap_row.setSpacing(4)
        self._snap_enabled = True
        self._btn_snap = QCheckBox("Snap")
        self._btn_snap.setChecked(True)
        self._btn_snap.setToolTip("Snap clips to edges, playhead, and markers")
        self._btn_snap.toggled.connect(self._on_snap_toggled)
        snap_row.addWidget(self._btn_snap)
        snap_row.addStretch()
        vlay.addLayout(snap_row)

        # Separator
        sep_actions = QFrame()
        sep_actions.setFrameShape(QFrame.HLine)
        sep_actions.setStyleSheet("color: #444;")
        vlay.addWidget(sep_actions)

        # Clip actions header
        clip_hdr = QLabel("<b>Clip Actions</b>")
        clip_hdr.setStyleSheet("font-size: 13px;")
        vlay.addWidget(clip_hdr)

        _CSZ = 38
        clip_btn_style = (
            f"QPushButton {{ min-width: {_CSZ}px; max-width: {_CSZ}px;"
            f" min-height: {_CSZ}px; max-height: {_CSZ}px;"
            "  border: 1px solid #555; border-radius: 4px; padding: 0px; } "
            "QPushButton:hover { background: #3a3a3a; } "
            "QPushButton:pressed { background: #4a4a4a; } "
            "QPushButton:disabled { border-color: #333; }"
        )
        _CIC = 28

        # Row 1: Split, Speed, Combine, Duplicate, Delete
        clip_row1 = QHBoxLayout()
        clip_row1.setSpacing(6)

        clip_actions_1 = [
            ("split", "Split clip at playhead", self._on_split),
            ("speed", "Change speed / reverse", self._on_speed),
            ("combine", "Combine adjacent clips", self._on_combine),
            ("duplicate", "Duplicate selected clip", self._on_duplicate),
            ("delete", "Delete selected clip (ripple)", self._on_ripple_delete),
        ]
        self._clip_action_btns = {}
        for key, tip, handler in clip_actions_1:
            btn = QPushButton()
            btn.setIcon(self._make_clip_action_icon(key, _CIC))
            btn.setIconSize(QPixmap(_CIC, _CIC).size())
            btn.setToolTip(tip)
            btn.setStyleSheet(clip_btn_style)
            btn.clicked.connect(handler)
            clip_row1.addWidget(btn)
            self._clip_action_btns[key] = btn

        clip_row1.addStretch()
        vlay.addLayout(clip_row1)

        # Row 2: Freeze Frame, Crossfade, Color Match, Disable, Snap
        clip_row2 = QHBoxLayout()
        clip_row2.setSpacing(6)

        clip_actions_2 = [
            ("freeze", "Freeze frame — hold current frame for N seconds", self._on_freeze_frame),
            ("crossfade", "Add crossfade with adjacent clip", self._on_crossfade),
            ("color_match", "Color-match clip to reference", self._on_color_match),
            ("disable", "Toggle clip enabled/disabled", self._on_disable_clip),
            ("snap_playhead", "Snap clip start to playhead", self._on_snap_to_playhead),
        ]
        for key, tip, handler in clip_actions_2:
            btn = QPushButton()
            btn.setIcon(self._make_clip_action_icon(key, _CIC))
            btn.setIconSize(QPixmap(_CIC, _CIC).size())
            btn.setToolTip(tip)
            btn.setStyleSheet(clip_btn_style)
            btn.clicked.connect(handler)
            clip_row2.addWidget(btn)
            self._clip_action_btns[key] = btn

        clip_row2.addStretch()
        vlay.addLayout(clip_row2)

        # Aliases for existing code that references these by name
        self._btn_split = self._clip_action_btns["split"]
        self._btn_speed = self._clip_action_btns["speed"]
        self._btn_combine = self._clip_action_btns["combine"]
        self._btn_combine.setEnabled(False)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("color: #444;")
        vlay.addWidget(sep2)

        # Track management: + and trash
        track_hdr = QLabel("<b>Tracks</b>")
        track_hdr.setStyleSheet("font-size: 13px;")
        vlay.addWidget(track_hdr)

        track_row = QHBoxLayout()
        track_row.setSpacing(6)

        self._btn_add_track = QPushButton()
        self._btn_add_track.setIcon(self._make_clip_action_icon("add_track", _CIC))
        self._btn_add_track.setIconSize(QPixmap(_CIC, _CIC).size())
        self._btn_add_track.setToolTip("Add track")
        self._btn_add_track.setStyleSheet(clip_btn_style)
        self._btn_add_track.clicked.connect(self._on_add_track_menu)
        track_row.addWidget(self._btn_add_track)

        self._btn_del_track = QPushButton()
        self._btn_del_track.setIcon(self._make_clip_action_icon("del_track", _CIC))
        self._btn_del_track.setIconSize(QPixmap(_CIC, _CIC).size())
        self._btn_del_track.setToolTip("Delete selected track")
        self._btn_del_track.setStyleSheet(clip_btn_style)
        self._btn_del_track.clicked.connect(self._on_delete_track)
        track_row.addWidget(self._btn_del_track)

        track_row.addStretch()
        vlay.addLayout(track_row)

        vlay.addStretch()
        return panel

    @staticmethod
    def _make_clip_action_icon(key: str, size: int = 28) -> QIcon:
        """Paint a clip action tool icon."""
        pm = QPixmap(size, size)
        pm.fill(QColor(0, 0, 0, 0))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QColor(210, 210, 210)

        if key == "split":
            # Vertical cut line through a rectangle
            p.setPen(QPen(QColor(160, 160, 160), 1.5))
            p.drawRect(3, 8, 22, 12)
            p.setPen(QPen(QColor(255, 100, 100), 2.5))
            p.drawLine(14, 4, 14, 24)
        elif key == "speed":
            # Clock with fast-forward arrows
            p.setPen(QPen(c, 1.5))
            p.drawEllipse(4, 4, 16, 16)
            p.drawLine(12, 6, 12, 12)
            p.drawLine(12, 12, 16, 14)
            p.setPen(QPen(QColor(255, 200, 80), 2.0))
            p.drawLine(20, 10, 26, 14)
            p.drawLine(20, 18, 26, 14)
        elif key == "combine":
            # Two rectangles merging into one
            p.setPen(QPen(c, 1.5))
            p.drawRect(3, 10, 8, 8)
            p.drawRect(17, 10, 8, 8)
            p.setPen(QPen(QColor(100, 220, 100), 2.0))
            p.drawLine(12, 14, 16, 14)
            p.drawLine(14, 12, 14, 16)
        elif key == "duplicate":
            # Two stacked rectangles
            p.setPen(QPen(c, 1.5))
            p.drawRect(6, 4, 14, 10)
            p.setPen(QPen(QColor(100, 180, 255), 1.5))
            p.drawRect(8, 14, 14, 10)
        elif key == "delete":
            # Trash can
            p.setPen(QPen(QColor(255, 100, 100), 2.0))
            p.drawLine(8, 8, 20, 8)
            p.drawRect(9, 8, 10, 14)
            p.drawLine(14, 5, 14, 8)
            p.drawLine(11, 5, 17, 5)
            p.drawLine(12, 11, 12, 19)
            p.drawLine(16, 11, 16, 19)
        elif key == "freeze":
            # Snowflake / pause symbol on a frame
            p.setPen(QPen(c, 1.5))
            p.drawRect(4, 6, 20, 16)
            p.setPen(QPen(QColor(100, 200, 255), 2.5))
            p.drawLine(11, 10, 11, 18)
            p.drawLine(17, 10, 17, 18)
        elif key == "crossfade":
            # Two overlapping triangles (dissolve)
            p.setPen(QPen(QColor(255, 180, 80), 2.0))
            p.drawLine(4, 20, 16, 8)
            p.setPen(QPen(QColor(100, 180, 255), 2.0))
            p.drawLine(12, 8, 24, 20)
            p.setPen(QPen(QColor(160, 160, 160), 1.0))
            p.drawLine(4, 20, 24, 20)
        elif key == "color_match":
            # Two color squares with arrow between
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(200, 100, 100))
            p.drawRect(3, 8, 8, 12)
            p.setBrush(QColor(100, 200, 100))
            p.drawRect(17, 8, 8, 12)
            p.setPen(QPen(c, 2.0))
            p.drawLine(12, 14, 16, 14)
            p.drawLine(14, 12, 16, 14)
            p.drawLine(14, 16, 16, 14)
        elif key == "disable":
            # Eye with slash
            p.setPen(QPen(c, 1.5))
            p.drawEllipse(6, 9, 16, 10)
            p.drawEllipse(11, 12, 6, 5)
            p.setPen(QPen(QColor(255, 80, 80), 2.0))
            p.drawLine(6, 22, 22, 6)
        elif key == "snap_playhead":
            # Clip snapping to a vertical line
            p.setPen(QPen(QColor(255, 100, 100), 2.0))
            p.drawLine(14, 4, 14, 24)
            p.setPen(QPen(c, 1.5))
            p.drawRect(16, 10, 10, 8)
            p.setPen(QPen(QColor(255, 200, 80), 2.0))
            p.drawLine(10, 14, 15, 14)
            p.drawLine(13, 12, 15, 14)
            p.drawLine(13, 16, 15, 14)
        elif key == "add_track":
            # Plus in a circle
            p.setPen(QPen(QColor(100, 220, 100), 2.0))
            p.drawEllipse(4, 4, 20, 20)
            p.drawLine(14, 8, 14, 20)
            p.drawLine(8, 14, 20, 14)
        elif key == "del_track":
            # Minus in a circle
            p.setPen(QPen(QColor(255, 100, 100), 2.0))
            p.drawEllipse(4, 4, 20, 20)
            p.drawLine(8, 14, 20, 14)

        p.end()
        return QIcon(pm)

    @Slot(int, bool)
    def _on_tool_changed(self, tool_id: int, checked: bool) -> None:
        if checked:
            self._multitrack.set_tool_mode(ToolMode(tool_id))

    def _show_shortcut_help(self) -> None:
        """Show a popup with all keyboard shortcuts."""
        shortcuts = (
            "<b>Playback</b><br>"
            "Space — Play / Pause<br>"
            "← / → — Step one frame<br>"
            "Home — Go to start<br>"
            "End — Go to end<br><br>"
            "<b>Editing</b><br>"
            "S — Split at playhead<br>"
            "D — Duplicate selected clip<br>"
            "Delete — Delete selected<br>"
            "Shift+Delete — Ripple delete<br>"
            "Alt+← / → — Nudge clip one frame<br>"
            "M — Add marker at playhead<br>"
            "Ctrl+Z — Undo<br>"
            "Ctrl+Shift+Z — Redo<br><br>"
            "<b>Tools</b><br>"
            "V — Select tool<br>"
            "B — Razor tool<br>"
            "Y — Slip tool<br>"
            "U — Slide tool<br><br>"
            "<b>Zoom</b><br>"
            "Ctrl+Scroll — Zoom in/out<br>"
            "Ctrl+0 — Zoom to fit"
        )
        QMessageBox.information(self, "Timeline Shortcuts", shortcuts)

    # -- Add track menu ----------------------------------------------------

    def _on_add_track_menu(self) -> None:
        menu = QMenu(self)
        menu.addAction("Video Track", lambda: self._multitrack.add_track(TrackType.VIDEO))
        menu.addAction("Audio Track", lambda: self._multitrack.add_track(TrackType.AUDIO))
        menu.exec(self._btn_add_track.mapToGlobal(
            self._btn_add_track.rect().bottomLeft()
        ))

    # -- Combine button ----------------------------------------------------

    def _on_combine(self) -> None:
        selection = self._multitrack.multi_selection
        if len(selection) >= 2 and self._can_combine(selection):
            self._combine_clips(selection[:])

    def _update_combine_button(self) -> None:
        """Enable Combine only when 2+ adjacent clips are selected."""
        selection = self._multitrack.multi_selection
        self._btn_combine.setEnabled(
            len(selection) >= 2 and self._can_combine(selection)
        )

    # -- Zoom handlers -----------------------------------------------------

    @Slot(int)
    def _on_zoom_slider(self, value: int) -> None:
        self._multitrack.set_zoom(float(value))
        # Persist globally
        self.state.global_config.tl_clip_zoom = value
        self.state.global_config.save()

    @Slot(float)
    def _on_fps_changed(self, idx: int) -> None:
        fps = self._fps_combo.currentData()
        if fps is None:
            return
        logger.info("Timeline FPS changed to %.0f", fps)
        if self._gl_preview:
            self._gl_preview.set_fps(fps)
        # Persist to project config
        try:
            cfg = ProjectConfig.load(self.project_path)
            cfg.tl_fps = fps
            cfg.save(self.project_path)
        except Exception:
            pass
        self._status_label.setText(f"Timeline FPS: {int(fps)}")

    @property
    def timeline_fps(self) -> float:
        return self._fps_combo.currentData() or 24.0

    def _on_zoom_changed(self, pps: float) -> None:
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setValue(int(pps))
        self._zoom_slider.blockSignals(False)
        pct = int(pps / 80.0 * 100)
        self._zoom_label.setText(f"{pct}%")
        # Persist globally
        self.state.global_config.tl_clip_zoom = int(pps)
        self.state.global_config.save()

    # -- Ripple handlers ---------------------------------------------------

    @Slot(bool)
    def _on_ripple_toggled(self, checked: bool) -> None:
        self._multitrack.ripple_mode = checked
        self._btn_ripple_all.setEnabled(checked)

    @Slot(bool)
    def _on_ripple_all_toggled(self, checked: bool) -> None:
        self._multitrack.ripple_all_tracks = checked

    @Slot(bool)
    def _on_snap_toggled(self, checked: bool) -> None:
        self._snap_enabled = checked
        self._multitrack._canvas.snap_enabled = checked

    # -- Timecode helpers --------------------------------------------------

    @staticmethod
    def _fmt_timecode(seconds: float) -> str:
        """Format seconds as MM:SS.F (tenths)."""
        s = max(0.0, seconds)
        m = int(s) // 60
        sec = s - m * 60
        return f"{m:02d}:{sec:04.1f}"

    def _update_timecode(self, current: float | None = None) -> None:
        """Update the timecode label with current / total."""
        if current is None:
            current = self._multitrack._canvas.playhead
        total = max(t.total_duration() for t in self._multitrack.tracks) if self._multitrack.tracks else 0.0
        self._time_label.setText(
            f"{self._fmt_timecode(current)} / {self._fmt_timecode(total)}"
        )

    def _update_duration_summary(self) -> None:
        """Update the duration summary label: clip count + total duration."""
        tracks = self._multitrack.tracks
        clip_count = sum(len(t.clips) for t in tracks)
        total = max((t.total_duration() for t in tracks), default=0.0)
        if clip_count:
            self._duration_label.setText(
                f"{clip_count} clip{'s' if clip_count != 1 else ''} | {self._fmt_timecode(total)}"
            )
        else:
            self._duration_label.setText("")

    # -- Razor handler -----------------------------------------------------

    @Slot(str, int, float)
    def _on_razor_split(self, track_id: str, clip_idx: int, offset_sec: float) -> None:
        track = self._multitrack.get_track(track_id)
        if track and 0 <= clip_idx < len(track.clips):
            self._do_split(track, clip_idx, offset_sec)

    # -- Marker handlers ---------------------------------------------------

    @Slot(float)
    def _on_marker_added(self, time: float) -> None:
        marker = TimelineMarker(time=time, name=f"M{len(self._markers) + 1}")
        self._markers.append(marker)
        self._multitrack._canvas.set_markers(self._markers)
        self._save_state()

    def _add_marker_at_playhead(self) -> None:
        time = self._multitrack.playhead
        self._on_marker_added(time)
        self._status_label.setText(f"Marker added at {time:.2f}s")

    # -- Tab activation / keyboard -----------------------------------------

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._gl_preview:
            self._gl_preview.restore_gl_resources()
            self._sync_gl_data()
            self._gl_preview.seek(self._multitrack.playhead)

    def hideEvent(self, event) -> None:
        """Free GPU memory when tabbing away from timeline."""
        super().hideEvent(event)
        # Stop playback to avoid stale state
        self._multitrack.set_playing(False)
        if self._gl_preview:
            self._gl_preview.release_gl_resources()
        # Persist timeline state on tab-away
        self._save_state()
        # If no export is running, release any lingering GPU memory
        # (e.g. RIFE model from a recent export)
        if self._export_worker is None:
            self.state._force_gc()

    def keyPressEvent(self, event) -> None:
        mods = event.modifiers()
        ctrl = bool(mods & Qt.ControlModifier)
        shift = bool(mods & Qt.ShiftModifier)

        if event.key() == Qt.Key_Space:
            self._play_pause()
            return
        # Undo: Ctrl+Z
        if event.key() == Qt.Key_Z and ctrl and not shift:
            if not self._undo_normalize():
                if not self._undo_match_all_to():
                    if not self._undo_match_similar():
                        if not self._undo_match_grade():
                            self._undo()
            return
        # Redo: Ctrl+Y or Ctrl+Shift+Z
        if (event.key() == Qt.Key_Y and ctrl) or (event.key() == Qt.Key_Z and ctrl and shift):
            self._redo()
            return
        # Zoom: Ctrl+= / Ctrl+- / Ctrl+0
        if ctrl:
            if event.key() in (Qt.Key_Equal, Qt.Key_Plus):
                self._multitrack.zoom_in()
                return
            if event.key() == Qt.Key_Minus:
                self._multitrack.zoom_out()
                return
            if event.key() == Qt.Key_0:
                self._multitrack.zoom_to_fit()
                return
            # Duplicate clip: Ctrl+D
            if event.key() == Qt.Key_D:
                self._duplicate_selected_clip()
                return
        # Split at playhead: S
        if event.key() == Qt.Key_S and not ctrl:
            self._split_selected_at_playhead()
            return
        # Nudge selected clip: Alt+Left / Alt+Right
        alt = bool(mods & Qt.AltModifier)
        if alt and event.key() in (Qt.Key_Left, Qt.Key_Right):
            self._nudge_selected_clip(event.key() == Qt.Key_Right)
            return
        # Marker: M key
        if event.key() == Qt.Key_M:
            self._add_marker_at_playhead()
            return
        # Tool shortcuts: Y=slip, U=slide
        if event.key() == Qt.Key_Y and not mods:
            self._btn_slip.setChecked(True)
            return
        if event.key() == Qt.Key_U and not mods:
            self._btn_slide.setChecked(True)
            return
        super().keyPressEvent(event)

    # -- Undo / Redo -------------------------------------------------------

    def _snapshot_state(self) -> dict:
        return {
            "tracks": [t.to_dict() for t in self._multitrack.tracks],
            "library": list(self._library.clip_paths()),
            "markers": [
                {"time": m.time, "name": m.name, "color": m.color}
                for m in self._markers
            ],
        }

    def _apply_snapshot(self, state: dict) -> None:
        self._restoring_undo = True
        try:
            # Restore tracks
            tracks = []
            for td in state["tracks"]:
                tracks.append(TimelineTrack.from_dict(td))
            self._multitrack.set_tracks_from_data(tracks)

            # Restore markers from snapshot
            self._markers = []
            for md in state.get("markers", []):
                self._markers.append(TimelineMarker(
                    time=md.get("time", 0.0),
                    name=md.get("name", ""),
                    color=md.get("color", "#ffcc00"),
                ))
            self._multitrack._canvas.set_markers(self._markers)

            # Sync library to match snapshot
            current_lib = set(self._library.clip_paths())
            target_lib = set(state.get("library", []))
            for p in current_lib - target_lib:
                self._library.remove_clip(p)
            for p in target_lib - current_lib:
                if Path(p).is_file():
                    dur = self._probe_duration(p)
                    self._library.add_clip(p, dur)

            if self._gl_preview:
                self._sync_gl_data()
            self._save_state()
        finally:
            self._restoring_undo = False

    def _undo(self) -> None:
        state = self._undo_stack.undo()
        if state is None:
            self._status_label.setText("Nothing to undo")
            return
        self._undo_stack.push_redo(self._snapshot_state())
        self._apply_snapshot(state)
        self._status_label.setText("Undo")

    def _redo(self) -> None:
        state = self._undo_stack.redo()
        if state is None:
            self._status_label.setText("Nothing to redo")
            return
        # Push current state onto undo without clearing redo
        self._undo_stack._undo.append(self._snapshot_state())
        self._apply_snapshot(state)
        self._status_label.setText("Redo")

    # -- Library <-> Timeline sync -----------------------------------------

    @Slot(str)
    def _on_library_clip_removed(self, path: str) -> None:
        """When a clip is removed from library, remove from timeline (unless split)."""
        if self._restoring or self._restoring_undo:
            return
        # Count how many timeline clips reference this path
        count = sum(1 for t in self._multitrack.tracks for c in t.clips if c.path == path)
        if count <= 1:
            # Remove all instances from timeline
            for t in self._multitrack.tracks:
                t.clips = [c for c in t.clips if c.path != path]
            self._multitrack._sync_canvas()
            self._multitrack.clips_changed.emit()

    @Slot(str)
    def _check_remove_from_library(self, path: str) -> None:
        """After a clip is deleted from timeline, remove from library if no longer referenced."""
        if self._restoring or self._restoring_undo:
            return
        # Check if any track still has a clip with this path
        for t in self._multitrack.tracks:
            for c in t.clips:
                if c.path == path:
                    return  # still referenced
        self._library.remove_clip(path)

    # -- Selected clip helpers ---------------------------------------------

    def _get_selected(self) -> tuple[TimelineTrack | None, int]:
        """Return (track, clip_idx) for the currently selected clip."""
        canvas = self._multitrack._canvas
        track_id = canvas._selected_track_id
        idx = canvas._selected_clip_idx
        if track_id and idx >= 0:
            track = self._multitrack.get_track(track_id)
            if track and 0 <= idx < len(track.clips):
                return track, idx
        return None, -1
    def _on_delete_track(self) -> None:
        """Delete the selected track (if empty), or the last empty track."""
        tracks = self._multitrack.tracks
        if len(tracks) <= 1:
            self._status_label.setText("Cannot delete the only track")
            return

        # Prefer the header-selected track if it's empty
        sel_id = self._multitrack.selected_track_id
        if sel_id:
            sel_track = self._multitrack.get_track(sel_id)
            if sel_track and not sel_track.clips:
                self._multitrack.remove_track(sel_id)
                self._status_label.setText(f"Deleted track: {sel_track.name}")
                return

        # Fallback: find any empty track to delete
        for t in reversed(tracks):
            if not t.clips:
                self._multitrack.remove_track(t.id)
                self._status_label.setText(f"Deleted track: {t.name}")
                return

        self._status_label.setText("All tracks have clips — remove clips first")
    def _export_trimmed_clip(self, clip_path: str, media_offset: float, clip_duration: float,
                             clip: TimelineClip | None = None) -> str:
        """Export the trimmed portion of a clip to a temp file, with effects baked in.

        Pass *clip* when the caller knows the exact instance — path lookup is
        ambiguous after Split/Duplicate and can bake a sibling's effects."""
        if clip is None:
            clip = self._find_clip(clip_path, media_offset, clip_duration)
        effects = getattr(clip, "effects", None) if clip else None

        # Check if we need to do anything (trim or effects)
        needs_trim = media_offset > 0.01 or abs(clip_duration - self._probe_duration(clip_path)) > 0.1
        has_effects = bool(effects)

        if not needs_trim and not has_effects:
            return clip_path

        import tempfile
        out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, prefix="tl_send_")
        out.close()

        # Build -vf with full range + any clip effects
        from sdqt.workers.timeline import _build_effects_filter
        vf_parts = ["scale=in_range=full:out_range=full"]
        if has_effects:
            effects_vf = _build_effects_filter(effects)
            if effects_vf:
                vf_parts.append(effects_vf.lstrip(","))

        cmd = [
            "ffmpeg", "-y", "-ss", str(media_offset), "-i", clip_path,
            "-t", str(clip_duration),
            *_codec_args(),
            "-c:a", "aac", "-b:a", "192k",
            "-vf", ",".join(vf_parts),
            *pix_fmt_args(_enc_name()), out.name,
        ]
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.warning("Trimmed export failed: %s", result.stderr[-300:])
            return clip_path
        return out.name
    def _on_effects_changed(self) -> None:
        # Sync effects from the panel's clip reference to the actual track clip
        # in case track objects were replaced by a reload
        if hasattr(self, "_effects_panel") and self._effects_panel is not None:
            panel_clip = self._effects_panel._clip
            # Find the matching clip on the track by path + name
            for t in self._multitrack.tracks:
                for c in t.clips:
                    if c.path == panel_clip.path and c.name == panel_clip.name and c is not panel_clip:
                        c.effects = panel_clip.effects
                        break
        self._multitrack.update()
        self._save_state(force=True)

    def _on_effects_preview(self) -> None:
        """Live effect parameter update — re-render GL preview without rebuilding."""
        # Sync panel clip effects to track clip (in case of stale reference)
        if hasattr(self, "_effects_panel") and self._effects_panel is not None:
            panel_clip = self._effects_panel._clip
            for t in self._multitrack.tracks:
                for c in t.clips:
                    if c.path == panel_clip.path and c.name == panel_clip.name and c is not panel_clip:
                        c.effects = panel_clip.effects
                        break
        if self._gl_preview:
            self._gl_preview.seek(self._multitrack.playhead)

    def _save_clip_to_library(self, track_id: str, clip_idx: int) -> None:
        """Save a timeline clip to the library, exporting if trimmed/altered."""
        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        clip = track.clips[clip_idx]
        logger.info("Save to library: %s (offset=%.2f dur=%.2f)",
                     clip.name, getattr(clip, "media_offset", 0.0), clip.duration)

        # Check if clip is already in library as-is
        if clip.path in self._library.clip_paths():
            mo = getattr(clip, "media_offset", 0.0) or 0.0
            md = getattr(clip, "media_duration", clip.duration)
            # Already in library AND not trimmed → nothing to do
            if mo < 0.01 and abs(clip.duration - md) < 0.01:
                self._status_label.setText(f"Already in library: {clip.name}")
                return

        # Check if export is needed (trimmed, offset, or different duration)
        mo = getattr(clip, "media_offset", 0.0) or 0.0
        md = getattr(clip, "media_duration", clip.duration)
        needs_export = mo > 0.01 or abs(clip.duration - md) > 0.01

        if needs_export:
            self._export_clip_to_library(clip)
        else:
            # Direct add — file is the full clip
            dur = self._probe_duration(clip.path)
            thumb = self._generate_thumbnail(clip.path)
            self._library.add_clip(clip.path, dur, thumb)
            self._save_state()
            self._status_label.setText(f"Added to library: {clip.name}")

    def _export_clip_to_library(self, clip) -> None:
        """Export the visible portion of a trimmed clip and add to library."""
        import subprocess

        out_dir = self.project_path / "clips" / "library"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / f"{cap_stem(clip.name)}.mp4")

        mo = getattr(clip, "media_offset", 0.0) or 0.0
        self._status_label.setText(f"Exporting {clip.name}...")

        from sdqt.workers.base import BaseWorker

        class _ExportWorker(BaseWorker):
            def __init__(self, src, dst, offset, duration, parent=None):
                super().__init__(parent)
                self._src = src
                self._dst = dst
                self._offset = offset
                self._duration = duration

            def run(self):
                try:
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", self._src,
                        "-ss", str(self._offset),
                        "-t", str(self._duration),
                        "-avoid_negative_ts", "make_zero",
                        self._dst,
                    ]
                    subprocess.run(cmd, capture_output=True, timeout=120)
                    if Path(self._dst).is_file() and Path(self._dst).stat().st_size > 0:
                        self.finished_ok.emit(self._dst)
                    else:
                        self.error.emit("Export produced empty file")
                except Exception as e:
                    self.error.emit(str(e))

        worker = _ExportWorker(clip.path, out_path, mo, clip.duration, parent=self)

        def _on_done(result_path):
            dur = self._probe_duration(result_path)
            thumb = self._generate_thumbnail(result_path)
            self._library.add_clip(result_path, dur, thumb)
            self._save_state()
            self._status_label.setText(f"Saved to library: {Path(result_path).stem}")

        worker.finished_ok.connect(_on_done)
        worker.error.connect(lambda msg: self._status_label.setText(f"Export failed: {msg}"))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _move_clip_to_track(self, src_track_id: str, clip_idx: int, dst_track_id: str) -> None:
        src = self._multitrack.get_track(src_track_id)
        dst = self._multitrack.get_track(dst_track_id)
        if not src or not dst or clip_idx < 0 or clip_idx >= len(src.clips):
            return
        clip = src.clips.pop(clip_idx)
        # Place at end of destination track
        clip.start_time = dst.total_duration()
        dst.clips.append(clip)
        self._multitrack._canvas.clear_selection()
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._status_label.setText(f"Moved to {dst.name}")

    def _reverse_library_clip(self, clip_path: str) -> None:
        self._status_label.setText("Reversing...")
        try:
            import subprocess
            stem = cap_stem(Path(clip_path).stem)
            suffix = Path(clip_path).suffix
            tmp = tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False, prefix=f"{stem}_rev_"
            )
            tmp.close()
            subprocess.run(
                ["ffmpeg", "-y", "-i", clip_path,
                 "-vf", "scale=in_range=full:out_range=full,reverse,setpts=PTS-STARTPTS",
                 "-af", "areverse,asetpts=PTS-STARTPTS",
                 *_codec_args(),
                 "-c:a", "aac", "-b:a", "192k",
                 *pix_fmt_args(),
                 tmp.name],
                capture_output=True, timeout=120,
            )
            if not Path(tmp.name).is_file() or Path(tmp.name).stat().st_size == 0:
                self._status_label.setText("Reverse failed")
                return
            persisted = self._persist_clip(tmp.name)
            dur = self._probe_duration(persisted)
            self._library.add_clip(persisted, dur)
            self._status_label.setText("Reversed clip added to library")
        except Exception:
            logger.warning("Library reverse failed", exc_info=True)
            self._status_label.setText("Reverse failed")

    def _speed_library_clip(self, clip_path: str) -> None:
        dlg = _SpeedDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        speed = dlg.speed_value()
        if abs(speed - 1.0) < 0.01:
            return
        self._status_label.setText(f"Applying {speed:.2f}x speed...")
        try:
            import subprocess
            stem = cap_stem(Path(clip_path).stem)
            suffix = Path(clip_path).suffix
            tmp = tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False, prefix=f"{stem}_spd_"
            )
            tmp.close()
            pts_factor = 1.0 / speed
            atempo_filters: list[str] = []
            remaining = speed
            while remaining < 0.5:
                atempo_filters.append("atempo=0.5")
                remaining /= 0.5
            while remaining > 100.0:
                atempo_filters.append("atempo=100.0")
                remaining /= 100.0
            atempo_filters.append(f"atempo={remaining:.6f}")
            vf = f"scale=in_range=full:out_range=full,setpts={pts_factor:.6f}*PTS"
            af = ",".join(atempo_filters)
            subprocess.run(
                ["ffmpeg", "-y", "-i", clip_path,
                 "-vf", vf, "-af", af,
                 *_codec_args(),
                 "-c:a", "aac", "-b:a", "192k",
                 *pix_fmt_args(),
                 tmp.name],
                capture_output=True, timeout=120,
            )
            if not Path(tmp.name).is_file() or Path(tmp.name).stat().st_size == 0:
                self._status_label.setText("Speed change failed")
                return
            persisted = self._persist_clip(tmp.name)
            dur = self._probe_duration(persisted)
            self._library.add_clip(persisted, dur)
            self._status_label.setText(f"Speed {speed:.2f}x clip added to library")
        except Exception:
            logger.warning("Library speed change failed", exc_info=True)
            self._status_label.setText("Speed change failed")

    # -- Public API --------------------------------------------------------

    def _persist_clip(self, path: str) -> str:
        """Copy a clip into the project ``clips/`` dir if it lives outside."""
        return self.persist_file(path, "clips")

    def add_clip(self, path: str) -> None:
        """Add a clip to the library (called from MainWindow send-to)."""
        if not path or not Path(path).is_file():
            return
        path = self._persist_clip(path)
        dur = self._probe_duration(path)
        thumb = self._generate_thumbnail(path)
        self._library.add_clip(path, dur, thumb)
        self._save_state()

    def set_send_targets(
        self,
        clip_targets: list[tuple[str, str]] | None = None,
        frame_targets: list[tuple[str, str]] | None = None,
        final_frame_targets: list[tuple[str, str]] | None = None,
        guide_video_targets: list[tuple[str, str]] | None = None,
    ) -> None:
        if clip_targets is not None:
            self._clip_targets = clip_targets
        if frame_targets is not None:
            self._frame_targets = frame_targets
        if final_frame_targets is not None:
            self._final_frame_targets = final_frame_targets
        if guide_video_targets is not None:
            self._guide_video_targets = guide_video_targets

    # -- Frame extraction --------------------------------------------------

    def _find_clip(self, clip_path: str, media_offset: float | None = None,
                   duration: float | None = None) -> TimelineClip | None:
        """Find a TimelineClip by path — ambiguous after Split/Duplicate, which
        create multiple clips sharing one source file. When media_offset (and
        optionally duration) is given, prefer the instance whose trim window
        matches so callers get the segment that was actually clicked."""
        candidates = [
            c for t in self._multitrack.tracks for c in t.clips
            if c.path == clip_path
        ]
        if not candidates:
            return None
        if media_offset is not None:
            for c in candidates:
                if abs((c.media_offset or 0.0) - media_offset) > 0.02:
                    continue
                if duration is not None and abs(c.duration - duration) > 0.02:
                    continue
                return c
        return candidates[0]

    def _extract_frame(
        self, clip_path: str, which: str,
        media_offset: float = 0.0, clip_duration: float = 0.0,
        clip: TimelineClip | None = None,
    ) -> str | None:
        try:
            from supremediffusion.utils.video import extract_single_frame, probe_video

            info = probe_video(clip_path)
            num_frames = info.get("num_frames", 1)
            fps = info.get("fps", 16)

            # Use explicit params if provided, otherwise look up clip.
            # Pass *clip* when the exact instance is known — path lookup is
            # ambiguous after Split/Duplicate: the wrong sibling's effects
            # and start_time would be used.
            if media_offset < 0.01 and clip_duration < 0.01:
                if clip is None:
                    clip = self._find_clip(clip_path)
                media_offset = clip.media_offset if clip else 0.0
                visible_dur = clip.duration if clip else (num_frames / fps)
            else:
                if clip is None:
                    clip = self._find_clip(clip_path, media_offset, clip_duration)
                visible_dur = clip_duration if clip_duration > 0 else (num_frames / fps)

            if which == "first":
                # First visible frame after trim-in
                frame_num = int(media_offset * fps)
            elif which == "last":
                # Last visible frame before trim-out
                frame_num = int((media_offset + visible_dur) * fps) - 1
            else:
                # "current" — playhead position relative to clip
                ph = self._multitrack.playhead
                clip_start = clip.start_time if clip else 0.0
                timeline_offset = max(0.0, ph - clip_start)
                frame_num = int((media_offset + timeline_offset) * fps)

            frame_num = max(0, min(frame_num, num_frames - 1))

            img = extract_single_frame(clip_path, frame_num)
            if img is None:
                return None

            # Send the frame exactly as previewed — do NOT neutralize here.
            # Color-neutralization is reserved for the automated chained-
            # generation feed-forward (to break warm drift); applying it to a
            # manual "send frame to" pushed warm frames cool/blue so the sent
            # image no longer matched the preview.

            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(tmp.name)
            tmp.close()

            # Apply clip effects (color grading, etc.) to the extracted frame
            if clip and getattr(clip, "effects", None):
                tmp_name = self._apply_effects_to_frame(tmp.name, clip.effects)
                if tmp_name:
                    return tmp_name

            return tmp.name
        except Exception:
            logger.warning("Frame extraction failed for %s", clip_path, exc_info=True)
            return None

    def _apply_effects_to_frame(self, frame_path: str,
                                effects: list[dict]) -> str | None:
        """Run ffmpeg to bake clip effects into an extracted frame PNG."""
        try:
            import subprocess
            from sdqt.workers.timeline import _build_effects_filter

            vf = _build_effects_filter(effects)
            if not vf:
                return None  # no active effects

            # vf starts with a comma — strip it
            vf = vf.lstrip(",")

            out = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            out.close()
            cmd = [
                "ffmpeg", "-y", "-i", frame_path,
                "-vf", vf,
                "-frames:v", "1",
                out.name,
            ]
            result = subprocess.run(
                cmd, capture_output=True, timeout=10,
            )
            if result.returncode == 0:
                import os
                os.unlink(frame_path)
                return out.name
            else:
                logger.warning("Effects apply failed: %s", result.stderr[:200])
                import os
                os.unlink(out.name)
                return None
        except Exception:
            logger.warning("Could not apply effects to frame", exc_info=True)
            return None

    def _capture_gl_or_extract(self, clip_path: str, which: str,
                                media_offset: float = 0.0,
                                clip_duration: float = 0.0,
                                clip: TimelineClip | None = None) -> str | None:
        """Try GL cached frame for 'current', raw extraction for first/last."""
        if which == "current" and self._gl_preview:
            frame = self._gl_preview.capture_frame()
            if frame:
                return frame
        return self._extract_frame(clip_path, which, media_offset, clip_duration, clip=clip)

    def _extract_and_send_frame(
        self, clip_path: str, target_key: str, which: str,
        media_offset: float = 0.0, clip_duration: float = 0.0,
        clip: TimelineClip | None = None,
    ) -> None:
        frame = self._capture_gl_or_extract(clip_path, which, media_offset, clip_duration, clip=clip)
        if frame:
            self.send_frame_to.emit(target_key, frame)
        else:
            self._status_label.setText("Frame extraction failed")

    def _extract_and_send_final_frame(self, clip_path: str, target_key: str,
                                      clip: TimelineClip | None = None) -> None:
        mo = (clip.media_offset or 0.0) if clip is not None else 0.0
        dur = clip.duration if clip is not None else 0.0
        frame = self._capture_gl_or_extract(clip_path, "last", mo, dur, clip=clip)
        if frame:
            self.send_final_frame_to.emit(target_key, frame)
        else:
            self._status_label.setText("Frame extraction failed")

    def _set_as_neutralizer_reference(
        self, clip_path: str, which: str,
        media_offset: float = 0.0, clip_duration: float = 0.0,
        clip: TimelineClip | None = None,
    ) -> None:
        """Extract the chosen frame raw (no neutralization applied) and save it
        as the global neutralizer reference image."""
        try:
            from supremediffusion.utils.video import extract_single_frame, probe_video
            from sdqt.utils.color_neutralize import save_as_neutralizer_reference

            info = probe_video(clip_path)
            num_frames = info.get("num_frames", 1)
            fps = info.get("fps", 16) or 16
            if clip is None:
                clip = self._find_clip(clip_path, media_offset, clip_duration)
            mo = media_offset if media_offset >= 0.01 else (clip.media_offset if clip else 0.0)
            visible_dur = (
                clip_duration if clip_duration >= 0.01 else
                (clip.duration if clip else (num_frames / fps))
            )
            if which == "first":
                frame_num = int(mo * fps)
            elif which == "last":
                frame_num = int((mo + visible_dur) * fps) - 1
            else:  # current
                ph = self._multitrack.playhead
                clip_start = clip.start_time if clip else 0.0
                timeline_offset = max(0.0, ph - clip_start)
                frame_num = int((mo + timeline_offset) * fps)
            frame_num = max(0, min(frame_num, num_frames - 1))

            img = extract_single_frame(clip_path, frame_num)
            if img is None:
                self._status_label.setText("Neutralizer reference: extraction failed")
                return

            ref_path = save_as_neutralizer_reference(img, self.state.global_config)
            self._status_label.setText(
                f"Neutralizer reference set ({which} frame) → {ref_path}"
            )
        except Exception as exc:
            logger.warning(
                "Set neutralizer reference failed for %s", clip_path, exc_info=True,
            )
            self._status_label.setText(f"Neutralizer reference failed: {exc}")

    def _extract_and_send_guide_video(self, clip_path: str, target_key: str,
                                      clip: TimelineClip | None = None) -> None:
        """Send guide video with effects baked in."""
        if not clip_path:
            return
        if clip is None:
            clip = self._find_clip(clip_path)
        if clip:
            mo = getattr(clip, "media_offset", 0.0) or 0.0
            exported = self._export_trimmed_clip(clip_path, mo, clip.duration, clip=clip)
            self.send_guide_video_to.emit(target_key, exported)
        else:
            self.send_guide_video_to.emit(target_key, clip_path)

    def _extract_and_send_cc_ref(
        self, clip_path: str, which: str,
        media_offset: float = 0.0, clip_duration: float = 0.0,
    ) -> None:
        frame = self._extract_frame(clip_path, which, media_offset, clip_duration)
        if frame:
            self.send_cc_ref_to.emit(frame)

    # -- FPS auto-detect ---------------------------------------------------

    def _check_clip_fps(self, path: str) -> None:
        """Detect clip FPS and offer to match if it differs from project FPS."""
        try:
            from supremediffusion.utils.video import probe_video
            info = probe_video(path)
            clip_fps = info.get("fps", 0)
            if clip_fps <= 0:
                return
        except Exception:
            return

        project_fps = self.timeline_fps
        # Only prompt if there's a meaningful difference
        if abs(clip_fps - project_fps) < 1.0:
            return

        # Check if timeline is empty (first clip) — auto-switch silently
        total_clips = sum(len(t.clips) for t in self._multitrack.tracks)
        if total_clips <= 1:
            self._set_fps(clip_fps)
            self._status_label.setText(f"FPS set to {clip_fps:.0f} (from clip)")
            return

        # Timeline has clips — ask user
        from PySide6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setWindowTitle("FPS Mismatch")
        msg.setText(
            f"This clip is {clip_fps:.0f} fps but the timeline is set to {project_fps:.0f} fps."
        )
        msg.setInformativeText(f"Switch timeline to {clip_fps:.0f} fps?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        if msg.exec() == QMessageBox.Yes:
            self._set_fps(clip_fps)

    def _set_fps(self, fps: float) -> None:
        """Set the timeline FPS, updating combo, GL preview, and config."""
        # Find closest combo entry
        best_idx = 0
        best_diff = 999.0
        for i in range(self._fps_combo.count()):
            val = self._fps_combo.itemData(i)
            if val is not None and abs(val - fps) < best_diff:
                best_diff = abs(val - fps)
                best_idx = i
        # If no close match, add it
        if best_diff > 0.5:
            self._fps_combo.blockSignals(True)
            self._fps_combo.addItem(str(int(fps)), float(fps))
            self._fps_combo.setCurrentIndex(self._fps_combo.count() - 1)
            self._fps_combo.blockSignals(False)
        else:
            self._fps_combo.blockSignals(True)
            self._fps_combo.setCurrentIndex(best_idx)
            self._fps_combo.blockSignals(False)
        # Update GL preview and canvas frame step
        if self._gl_preview:
            self._gl_preview.set_fps(fps)
        self._multitrack._canvas.set_project_fps(fps)
        # Persist
        try:
            cfg = ProjectConfig.load(self.project_path)
            cfg.tl_fps = fps
            cfg.save(self.project_path)
        except Exception:
            pass

    # -- Library -> Timeline -----------------------------------------------

    @Slot(str)
    def _add_library_clip_to_timeline(self, path: str) -> None:
        dur = self._library.get_duration(path)
        # Add to first video track by default (or first audio track for audio files)
        audio_exts = {".mp3", ".wav", ".flac", ".ogg", ".aac"}
        is_audio = Path(path).suffix.lower() in audio_exts

        target_track = None
        for t in self._multitrack.tracks:
            if is_audio and t.track_type == TrackType.AUDIO:
                target_track = t
                break
            elif not is_audio and t.track_type == TrackType.VIDEO:
                target_track = t
                break

        if not target_track:
            # Create appropriate track
            tt = TrackType.AUDIO if is_audio else TrackType.VIDEO
            target_track = self._multitrack.add_track(tt)

        # Generate waveform
        waveform = None
        try:
            from sdqt.utils.waveform import extract_waveform
            waveform = extract_waveform(path)
        except Exception:
            pass

        self._multitrack.add_clip_to_track(
            target_track.id, path, dur,
            has_audio=True, waveform=waveform,
        )

        # Auto-detect FPS and prompt if mismatch
        if not is_audio:
            self._check_clip_fps(path)

    # -- GL Preview helpers ------------------------------------------------

    def _sync_gl_data(self) -> None:
        """Push current track data to the GL preview."""
        if self._gl_preview:
            self._gl_preview.set_timeline_data(self._multitrack.tracks)

    @Slot(float)
    def _on_gl_preview_position(self, seconds: float) -> None:
        """GL playback position changed — sync the multitrack playhead."""
        self._multitrack.set_playhead(seconds)
        self._update_timecode(seconds)

    # -- Track events ------------------------------------------------------

    @Slot()
    def _on_clips_changed(self) -> None:
        self._update_duration_summary()
        self._update_timecode()
        # Release stale frame cache decoders for clips no longer on any track
        if self._frame_cache:
            active_paths = set(self._multitrack.all_clip_paths())
            self._frame_cache.cleanup_stale_decoders(active_paths)
        if self._gl_preview:
            self._sync_gl_data()
        if not self._restoring and not self._restoring_undo:
            # User made an intentional change — safe to save again
            self._has_missing_clips = False
            # Push previous state for undo
            if self._pre_change_state is not None:
                self._undo_stack.push(self._pre_change_state)
            self._pre_change_state = self._snapshot_state()
            self._save_state()

    @Slot(float)
    def _on_playhead_moved(self, seconds: float) -> None:
        self._update_timecode(seconds)
        if self._gl_preview:
            self._gl_preview.seek(seconds)

    # -- Transport ---------------------------------------------------------

    @Slot()
    def _play_pause(self) -> None:
        if not self._multitrack.all_clip_paths():
            self._status_label.setText("No clips on timeline")
            return
        if not self._gl_preview:
            self._status_label.setText("GL preview unavailable")
            return
        if self._gl_preview.is_playing():
            self._gl_preview.pause()
            self._multitrack.set_playing(False)
            self._btn_play.setText("\u25B6")
        else:
            self._sync_gl_data()
            self._gl_preview.play()
            self._multitrack.set_playing(True)
            self._btn_play.setText("\u23F8")

    @Slot()
    def _stop(self) -> None:
        self._multitrack.set_playing(False)
        self._btn_play.setText("\u25B6")
        if self._gl_preview:
            self._gl_preview.stop()
        self._multitrack.set_playhead(0)
        self._update_timecode(0.0)
    def _resync_audio(self) -> None:
        """Flush the audio mixer buffer and resync to the current playhead."""
        if self._gl_preview and hasattr(self._gl_preview, "_audio_mixer"):
            mixer = self._gl_preview._audio_mixer
            if mixer:
                mixer._flush_and_resync()
                self._status_label.setText("Audio resynced")
            else:
                self._status_label.setText("No audio mixer active")
        else:
            self._status_label.setText("Resync requires GL preview mode")
    def _seek_to_clip_edge(self, clip, which: str) -> None:
        """Seek playhead to the first or last frame of a clip."""
        if not clip:
            return
        if which == "first":
            pos = clip.start_time
        else:
            pos = clip.start_time + clip.duration - 0.05
        self._multitrack.set_playhead(pos)
        self._update_timecode(pos)
        if self._gl_preview:
            self._gl_preview.seek(pos)

    @staticmethod
    def _probe_clip_tech_info(clip_path: str) -> str:
        """Probe video file and return formatted technical info HTML."""
        import subprocess, json, os
        lines = []
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", "-show_streams", clip_path],
                capture_output=True, text=True, timeout=10,
            )
            data = json.loads(r.stdout)
            vs = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
            aus = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
            fmt = data.get("format", {})

            if vs:
                w, h = vs.get("width", "?"), vs.get("height", "?")
                codec = vs.get("codec_name", "?")
                pix = vs.get("pix_fmt", "?")
                cr = vs.get("color_range", "unknown")
                cs = vs.get("color_space", "unknown")
                ct = vs.get("color_transfer", "unknown")
                rfr = vs.get("r_frame_rate", "?")
                if "/" in str(rfr):
                    n, d = rfr.split("/")
                    fps = f"{float(n)/float(d):.1f}" if float(d) else rfr
                else:
                    fps = rfr
                nb = vs.get("nb_frames", vs.get("nb_read_frames", "?"))
                lines.append(f"<b>Resolution:</b> {w}x{h}")
                lines.append(f"<b>Codec:</b> {codec}")
                lines.append(f"<b>Pixel Format:</b> {pix}")
                lines.append(f"<b>Color Range:</b> {cr}")
                lines.append(f"<b>Color Space:</b> {cs}")
                lines.append(f"<b>Color Transfer:</b> {ct}")
                lines.append(f"<b>FPS:</b> {fps}")
                lines.append(f"<b>Frames:</b> {nb}")

            dur = fmt.get("duration", "?")
            if dur != "?":
                dur = f"{float(dur):.2f}s"
            lines.append(f"<b>Duration:</b> {dur}")

            size = fmt.get("size", "?")
            if size != "?":
                mb = int(size) / (1024 * 1024)
                lines.append(f"<b>File Size:</b> {mb:.1f} MB")

            if aus:
                ac = aus.get("codec_name", "?")
                ar = aus.get("sample_rate", "?")
                ach = aus.get("channels", "?")
                lines.append(f"<b>Audio:</b> {ac}, {ar} Hz, {ach}ch")
            else:
                lines.append("<b>Audio:</b> none")

        except Exception as exc:
            lines.append(f"<i>Probe failed: {exc}</i>")
        return "<br>".join(lines)

    @staticmethod
    def _probe_clip_tooltip(clip_path: str) -> str:
        """Short single-line tooltip for hover."""
        import subprocess
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,r_frame_rate,codec_name,pix_fmt,color_range",
                 "-show_entries", "format=duration",
                 "-of", "csv=p=0", clip_path],
                capture_output=True, text=True, timeout=5,
            )
            parts = r.stdout.strip().split("\n")
            if len(parts) >= 2:
                vp = parts[0].split(",")
                dur = parts[1].strip()
                if len(vp) >= 6:
                    codec, w, h, pix, cr, rfr = vp[0], vp[1], vp[2], vp[3], vp[4], vp[5]
                    if "/" in rfr:
                        n, d = rfr.split("/")
                        fps = f"{float(n)/float(d):.0f}"
                    else:
                        fps = rfr
                    dur_s = f"{float(dur):.1f}s" if dur else "?"
                    return f"{w}x{h} | {codec} {pix} {cr} | {fps}fps | {dur_s}"
        except Exception:
            pass
        return Path(clip_path).name

    def _show_clip_info(self, clip_path: str) -> None:
        """Show technical info + generation metadata embedded in the video file."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTextEdit, QVBoxLayout, QPushButton, QHBoxLayout
        from supremediffusion.utils.video import read_video_metadata

        meta = read_video_metadata(clip_path)

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Clip Info — {Path(clip_path).name}")
        dlg.setMinimumWidth(550)
        layout = QVBoxLayout(dlg)

        info = QTextEdit()
        info.setReadOnly(True)

        lines = []

        # Technical info first
        lines.append("<h3>Technical</h3>")
        lines.append(self._probe_clip_tech_info(clip_path))

        # Generation metadata
        if meta:
            lines.append("<br><h3>Generation</h3>")
            if meta.get("prompt"):
                lines.append(f"<b>Prompt:</b><br>{meta['prompt']}<br>")
            if meta.get("negative_prompt"):
                lines.append(f"<b>Negative:</b><br>{meta['negative_prompt']}<br>")
            for key in ("model_type", "resolution", "fps", "video_length",
                        "steps", "guidance_scale", "guidance2_scale",
                        "flow_shift", "solver", "seed", "denoising_strength", "mode"):
                if key in meta:
                    label = key.replace("_", " ").title()
                    lines.append(f"<b>{label}:</b> {meta[key]}")

        info.setHtml("<br>".join(lines))

        layout.addWidget(info)

        # Send Prompt to buttons
        if meta and meta.get("prompt"):
            btn_row = QHBoxLayout()
            btn_gen = QPushButton("Send Prompt to Img2Vid")
            btn_gen.clicked.connect(lambda _checked=False: (
                self._send_prompt_to_gen(meta),
                dlg.accept(),
            ))
            btn_row.addWidget(btn_gen)
            btn_ve = QPushButton("Send Prompt to Video Extender")
            btn_ve.clicked.connect(lambda _checked=False: (
                self._send_prompt_to_ve(meta),
                dlg.accept(),
            ))
            btn_row.addWidget(btn_ve)
            btn_row.addStretch()
            layout.addLayout(btn_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        dlg.exec()

    def _send_prompt_to_gen(self, meta: dict) -> None:
        """Load prompt + settings into the Generate tab."""
        main = self.window()
        if not hasattr(main, "_gen_tab"):
            return
        tab = main._gen_tab
        prompt = meta.get("prompt", "")

        # Detect lightning-style prompt: "(at N seconds: ...)"
        if "(at " in prompt and "seconds:" in prompt:
            # Switch to lightning model if not already
            if not tab._is_lightning():
                # Find and select lightning model
                for i in range(tab._model_type.count()):
                    if "lightning" in (tab._model_type.itemData(i) or ""):
                        tab._model_type.setCurrentIndex(i)
                        break
            tab._set_lightning_prompt(prompt)
        else:
            tab._prompt.setPlainText(prompt)

        tab._neg_prompt.setText(meta.get("negative_prompt", ""))
        if meta.get("steps"):
            tab._steps.setValue(int(meta["steps"]))
        if meta.get("guidance_scale"):
            tab._guidance_scale.setValue(float(meta["guidance_scale"]))
        if meta.get("seed"):
            tab._seed.setValue(int(meta["seed"]))
        if meta.get("flow_shift"):
            tab._flow_shift.setValue(float(meta["flow_shift"]))
        main._tab_widget.setCurrentWidget(main._generate_suite)
        main._generate_tabs.setCurrentWidget(tab)
        self._status_label.setText("Prompt loaded into Img2Vid")

    def _send_prompt_to_ve(self, meta: dict) -> None:
        """Load prompt into the Video Extender tab."""
        main = self.window()
        if hasattr(main, "_ve_tab"):
            tab = main._ve_tab
            if hasattr(tab, "_params") and hasattr(tab._params, "_prompt"):
                tab._params._prompt.setPlainText(meta.get("prompt", ""))
            main._tab_widget.setCurrentWidget(main._generate_suite)
            main._generate_tabs.setCurrentWidget(tab)
            self._status_label.setText("Prompt loaded into Video Extender")

    def _go_start(self) -> None:
        self._multitrack.set_playhead(0)
        if self._gl_preview:
            self._gl_preview.seek(0)

    @Slot()
    def _go_end(self) -> None:
        dur = self._multitrack.total_duration()
        self._multitrack.set_playhead(dur)
        if self._gl_preview:
            self._gl_preview.seek(dur)

    @Slot()
    def _go_prev_clip(self) -> None:
        # Find the previous clip boundary across all tracks
        ph = self._multitrack.playhead
        boundaries = set()
        boundaries.add(0.0)
        for t in self._multitrack.tracks:
            for c in t.clips:
                boundaries.add(c.start_time)
                boundaries.add(c.start_time + c.duration)

        sorted_b = sorted(boundaries)
        prev_t = 0.0
        for b in sorted_b:
            if b >= ph - 0.05:
                break
            prev_t = b
        self._multitrack.set_playhead(prev_t)
        self._multitrack.playhead_moved.emit(prev_t)

    @Slot()
    def _go_next_clip(self) -> None:
        ph = self._multitrack.playhead
        boundaries = set()
        for t in self._multitrack.tracks:
            for c in t.clips:
                boundaries.add(c.start_time)
                boundaries.add(c.start_time + c.duration)

        for b in sorted(boundaries):
            if b > ph + 0.05:
                self._multitrack.set_playhead(b)
                self._multitrack.playhead_moved.emit(b)
                return

    # -- Volume ------------------------------------------------------------

    @Slot(int)
    def _on_volume_changed(self, value: int) -> None:
        vol = value / 100.0
        if self._gl_preview:
            self._gl_preview.set_volume(vol)
        if value == 0:
            self._mute_btn.setText("\U0001F507")  # 🔇
        elif value < 50:
            self._mute_btn.setText("\U0001F509")  # 🔉
        else:
            self._mute_btn.setText("\U0001F50A")  # 🔊

    @Slot()
    def _toggle_mute(self) -> None:
        if self._volume_slider.value() > 0:
            self._pre_mute_volume = self._volume_slider.value()
            self._volume_slider.setValue(0)
        else:
            self._volume_slider.setValue(getattr(self, "_pre_mute_volume", 50))
    def _save_state(self, force: bool = False) -> None:
        if not force and getattr(self, "_has_missing_clips", False):
            logger.debug("Skipping timeline save — missing clips detected on restore")
            return
        self._has_missing_clips = False  # clear once we do save
        try:
            cfg = ProjectConfig.load(self.project_path)
            cfg.tl_library_clips = self._library.clip_paths()
            tracks_data = [t.to_dict() for t in self._multitrack.tracks]
            # Debug: log clip 0 effects
            if tracks_data and tracks_data[0].get("clips"):
                c0 = tracks_data[0]["clips"][0]
                logger.info(
                    "SAVE_STATE clip0: effects=%s bake=%s path=%s",
                    c0.get("effects", "NONE"),
                    "YES" if c0.get("bake_data") else "no",
                    c0.get("path", "?")[-40:],
                )
            cfg.tl_tracks = tracks_data
            # Keep legacy field in sync for backward compat
            cfg.tl_timeline_clips = self._multitrack.all_clip_paths()
            # Save markers
            cfg.tl_markers = [
                {"time": m.time, "name": m.name, "color": m.color}
                for m in self._markers
            ]
            # Save zoom range
            zs, ze = self._multitrack._zoom_bar.visible_range()
            cfg.tl_zoom_start = zs
            cfg.tl_zoom_end = ze
            cfg.save(self.project_path)
        except Exception:
            logger.warning("Failed to save timeline state", exc_info=True)

    # -- Timeline versions ---------------------------------------------------

    def _populate_versions(self, cfg=None) -> None:
        """Refresh the version dropdown from saved project config."""
        if cfg is None:
            cfg = ProjectConfig.load(self.project_path)
        self._version_combo.blockSignals(True)
        self._version_combo.clear()
        versions = getattr(cfg, "tl_versions", {}) or {}
        active = getattr(cfg, "tl_active_version", "") or ""
        if not versions:
            self._version_combo.addItem("Version 1")
            active = ""
        else:
            for name in versions:
                self._version_combo.addItem(name)
            if active and active in versions:
                self._version_combo.setCurrentText(active)
            else:
                self._version_combo.setCurrentIndex(0)
        self._version_combo.blockSignals(False)

    def _version_snapshot(self) -> dict:
        """Capture the current canvas state for a version."""
        return {
            "tracks": [t.to_dict() for t in self._multitrack.tracks],
            "markers": [
                {"time": m.time, "name": m.name, "color": m.color}
                for m in self._markers
            ],
        }

    def _apply_version_snapshot(self, data: dict) -> None:
        """Restore canvas state from a version snapshot."""
        # Close effects panel to avoid stale clip references
        if hasattr(self, "_effects_panel") and self._effects_panel is not None:
            self._effects_panel.close()
            self._effects_panel = None
        self._restoring = True
        try:
            tracks = []
            for td in data.get("tracks", []):
                track = TimelineTrack.from_dict(td)
                track.clips = [c for c in track.clips if Path(c.path).is_file()]
                tracks.append(track)
            if tracks:
                self._multitrack.set_tracks_from_data(tracks)
                self._regenerate_waveforms(tracks)
            else:
                self._multitrack.reset_to_default()

            self._markers = []
            for md in data.get("markers", []):
                self._markers.append(TimelineMarker(
                    time=md.get("time", 0.0),
                    name=md.get("name", ""),
                    color=md.get("color", "#ffcc00"),
                ))
            self._multitrack._canvas.set_markers(self._markers)

            if self._gl_preview:
                self._sync_gl_data()
            self._save_state()
        finally:
            self._restoring = False
            self._undo_stack.clear()
            self._pre_change_state = self._snapshot_state()

    @Slot()
    def _save_version(self) -> None:
        """Save the current canvas state to the active version."""
        name = self._version_combo.currentText().strip()
        if not name:
            return
        # Force-flush live state to disk first
        self._save_state(force=True)
        cfg = ProjectConfig.load(self.project_path)
        versions = getattr(cfg, "tl_versions", {}) or {}
        versions[name] = self._version_snapshot()
        cfg.tl_versions = versions
        cfg.tl_active_version = name
        cfg.save(self.project_path)
        self._status_label.setText(f"Version '{name}' saved")

    @Slot()
    def _new_version(self) -> None:
        """Create a new version from the current canvas."""
        existing = [self._version_combo.itemText(i)
                    for i in range(self._version_combo.count())]
        # Auto-generate next name
        n = len(existing) + 1
        default_name = f"Version {n}"
        while default_name in existing:
            n += 1
            default_name = f"Version {n}"

        name, ok = QInputDialog.getText(
            self, "New Version", "Version name:", text=default_name,
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in existing:
            QMessageBox.warning(self, "Duplicate", f"Version '{name}' already exists.")
            return

        cfg = ProjectConfig.load(self.project_path)
        versions = getattr(cfg, "tl_versions", {}) or {}
        versions[name] = self._version_snapshot()
        cfg.tl_versions = versions
        cfg.tl_active_version = name
        cfg.save(self.project_path)

        self._version_combo.blockSignals(True)
        self._version_combo.addItem(name)
        self._version_combo.setCurrentText(name)
        self._version_combo.blockSignals(False)
        self._status_label.setText(f"Version '{name}' created")

    @Slot()
    def _rename_version(self) -> None:
        """Rename the currently selected version."""
        old_name = self._version_combo.currentText().strip()
        if not old_name:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename Version", "New name:", text=old_name,
        )
        if not ok or not new_name.strip() or new_name.strip() == old_name:
            return
        new_name = new_name.strip()

        cfg = ProjectConfig.load(self.project_path)
        versions = getattr(cfg, "tl_versions", {}) or {}
        if old_name in versions:
            versions[new_name] = versions.pop(old_name)
        cfg.tl_versions = versions
        if cfg.tl_active_version == old_name:
            cfg.tl_active_version = new_name
        cfg.save(self.project_path)

        idx = self._version_combo.currentIndex()
        self._version_combo.blockSignals(True)
        self._version_combo.setItemText(idx, new_name)
        self._version_combo.blockSignals(False)
        self._status_label.setText(f"Renamed '{old_name}' → '{new_name}'")

    @Slot()
    def _delete_version(self) -> None:
        """Delete the currently selected version (with confirmation)."""
        name = self._version_combo.currentText().strip()
        if not name:
            return
        if self._version_combo.count() <= 1:
            QMessageBox.warning(self, "Cannot Delete", "Cannot delete the last version.")
            return
        reply = QMessageBox.question(
            self, "Delete Version",
            f"Delete version '{name}'? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        cfg = ProjectConfig.load(self.project_path)
        versions = getattr(cfg, "tl_versions", {}) or {}
        versions.pop(name, None)
        cfg.tl_versions = versions

        self._version_combo.blockSignals(True)
        idx = self._version_combo.currentIndex()
        self._version_combo.removeItem(idx)
        self._version_combo.blockSignals(False)

        # Switch to the now-current version
        new_name = self._version_combo.currentText()
        cfg.tl_active_version = new_name
        cfg.save(self.project_path)

        if new_name in versions:
            self._apply_version_snapshot(versions[new_name])
        self._status_label.setText(f"Version '{name}' deleted")

    @Slot(str)
    def _on_version_switched(self, name: str) -> None:
        """Switch to a different version."""
        if not name or self._restoring:
            return
        cfg = ProjectConfig.load(self.project_path)
        versions = getattr(cfg, "tl_versions", {}) or {}
        if name not in versions:
            return
        cfg.tl_active_version = name
        cfg.save(self.project_path)
        self._apply_version_snapshot(versions[name])
        self._status_label.setText(f"Switched to version '{name}'")

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        # Close effects panel to avoid stale clip references
        if hasattr(self, "_effects_panel") and self._effects_panel is not None:
            self._effects_panel.close()
            self._effects_panel = None
        self._restoring = True
        self._has_missing_clips = False
        try:
            self._library.clear()
            self._multitrack.clear_all()

            cfg = ProjectConfig.load(self.project_path)

            # Restore library
            for path in cfg.tl_library_clips:
                if Path(path).is_file():
                    dur = self._probe_duration(path)
                    self._library.add_clip(path, dur)

            # If an active version exists, load its tracks instead of tl_tracks
            versions = getattr(cfg, "tl_versions", {}) or {}
            active_ver = getattr(cfg, "tl_active_version", "") or ""
            if active_ver and active_ver in versions:
                ver_data = versions[active_ver]
                source_tracks = ver_data.get("tracks", [])
                source_markers = ver_data.get("markers", [])
            else:
                source_tracks = cfg.tl_tracks
                source_markers = getattr(cfg, "tl_markers", []) or []

            # Restore tracks (new format)
            missing_clips: list[str] = []
            if source_tracks:
                tracks = []
                for td in source_tracks:
                    track = TimelineTrack.from_dict(td)
                    # Filter missing clips but record what was lost
                    valid = []
                    for c in track.clips:
                        if Path(c.path).is_file():
                            valid.append(c)
                        else:
                            missing_clips.append(c.path)
                    track.clips = valid
                    tracks.append(track)
                if tracks:
                    self._multitrack.set_tracks_from_data(tracks)
                    # Regenerate waveforms (not persisted, cheap to recompute)
                    self._regenerate_waveforms(tracks)
                else:
                    # All saved tracks were empty/invalid — reset to default
                    self._multitrack.reset_to_default()
            elif cfg.tl_timeline_clips:
                # Backward compat: migrate single-track clip list
                self._multitrack.reset_to_default()
                video_track = self._multitrack.get_first_video_track()
                if video_track:
                    start = 0.0
                    for path in cfg.tl_timeline_clips:
                        if Path(path).is_file():
                            dur = self._library.get_duration(path)
                            if dur <= 0:
                                dur = self._probe_duration(path)
                            self._multitrack.add_clip_to_track(
                                video_track.id, path, dur, start_time=start,
                            )
                            start += dur
                        else:
                            missing_clips.append(path)
            else:
                # Fresh project with no saved tracks — reset to single default track
                self._multitrack.reset_to_default()

            if missing_clips:
                logger.warning(
                    "Timeline restore: %d clip(s) missing — "
                    "skipping save to preserve project.json data: %s",
                    len(missing_clips), missing_clips,
                )
                # Block auto-save until the user makes an intentional change
                # so the missing-clip metadata is preserved in project.json.
                self._has_missing_clips = True

            # Restore markers
            self._markers = []
            for md in source_markers:
                self._markers.append(TimelineMarker(
                    time=md.get("time", 0.0),
                    name=md.get("name", ""),
                    color=md.get("color", "#ffcc00"),
                ))
            self._multitrack._canvas.set_markers(self._markers)

            # Restore FPS
            saved_fps = getattr(cfg, "tl_fps", 24.0) or 24.0
            self._set_fps(saved_fps)

            # Restore zoom range
            zs = getattr(cfg, "tl_zoom_start", 0.0) or 0.0
            ze = getattr(cfg, "tl_zoom_end", 1.0) or 1.0
            if 0.0 <= zs < ze <= 1.0:
                self._multitrack._zoom_bar.set_range(zs, ze)
                self._multitrack._on_zoom_bar_changed(zs, ze)

            has_clips = any(t.clips for t in self._multitrack.tracks)
            if has_clips and self._gl_preview:
                self._sync_gl_data()
            # Populate version dropdown
            self._populate_versions(cfg)
        except Exception:
            logger.debug("Failed to restore timeline state", exc_info=True)
        finally:
            self._restoring = False
            self._undo_stack.clear()
            self._pre_change_state = self._snapshot_state()

    # -- Helpers -----------------------------------------------------------

    def _regenerate_waveforms(self, tracks: list[TimelineTrack]) -> None:
        """Regenerate waveforms for all clips that have audio (runs in background)."""
        import threading

        def _worker():
            try:
                from sdqt.utils.waveform import extract_waveform
            except ImportError:
                return
            for track in tracks:
                for clip in track.clips:
                    if clip.waveform:
                        continue
                    if not clip.has_audio or not Path(clip.path).is_file():
                        continue
                    try:
                        clip.waveform = extract_waveform(clip.path)
                    except Exception:
                        pass
            # Schedule canvas repaint on main thread
            QTimer.singleShot(0, self._multitrack._canvas.update)

        threading.Thread(target=_worker, daemon=True).start()

    @staticmethod
    def _probe_duration(path: str) -> float:
        # Still images get a default duration
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
        if Path(path).suffix.lower() in image_exts:
            return 5.0
        try:
            from supremediffusion.utils.video import probe_video
            info = probe_video(path)
            return info.get("duration", info.get("num_frames", 81) / info.get("fps", 16))
        except Exception:
            pass
        # Fallback for audio-only files (probe_video raises on no video stream)
        try:
            import json, subprocess
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_format", path],
                capture_output=True, text=True, timeout=5,
            )
            data = json.loads(result.stdout)
            dur = float(data.get("format", {}).get("duration", 0))
            if dur > 0:
                return dur
        except Exception:
            pass
        return 5.0

    @staticmethod
    def _generate_thumbnail(path: str) -> QPixmap | None:
        # Skip for audio files
        audio_exts = {".mp3", ".wav", ".flac", ".ogg", ".aac"}
        if Path(path).suffix.lower() in audio_exts:
            return None
        # Still images — load directly as QPixmap
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
        if Path(path).suffix.lower() in image_exts:
            pix = QPixmap(path)
            return pix if not pix.isNull() else None
        try:
            from supremediffusion.utils.video import extract_single_frame
            from PySide6.QtGui import QImage
            img = extract_single_frame(path, 0)
            if img is None:
                return None
            if img.mode != "RGB":
                img = img.convert("RGB")
            data = img.tobytes("raw", "RGB")
            qimg = QImage(data, img.width, img.height, 3 * img.width, QImage.Format_RGB888)
            return QPixmap.fromImage(qimg)
        except Exception:
            return None
