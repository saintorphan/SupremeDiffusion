"""Multi-track canvas — custom-painted area with ruler, clips, waveforms, playhead."""

from __future__ import annotations

import logging
import math
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QMimeData, QPoint, QPointF, QRectF, QSize
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDrag,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QFont,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygon,
    QWheelEvent,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from sdqt.widgets.timeline_track import (
    MIME_TIMELINE_CLIP,
    TimelineClip,
    TimelineMarker,
    TimelineTrack,
    TrackType,
    ToolMode,
    _CLIP_COLORS,
    _CLIP_MARGIN,
    _CLIP_RADIUS,
    _PLAYHEAD_HIT,
)

logger = logging.getLogger(__name__)

_RULER_HEIGHT = 20
_WAVEFORM_COLOR_AUDIO = QColor("#1abc9c")
_WAVEFORM_COLOR_VIDEO = QColor("#5577aa")


# Cache duration probes so a multi-file drop of N files pays the ffprobe
# cost once per unique path even if the OS hands us the same paths twice.
_DURATION_CACHE: dict[str, float] = {}


def _probe_duration_cached(path: str) -> float:
    """Return a clip's duration in seconds, falling back to 5.0 on failure.

    Used by ``dropEvent`` to chain multiple dropped clips end-to-end.
    """
    cached = _DURATION_CACHE.get(path)
    if cached is not None:
        return cached
    try:
        from supremediffusion.utils.video import probe_video
        info = probe_video(path)
        dur = info.get("duration")
        if not dur:
            frames = info.get("num_frames")
            fps = info.get("fps")
            if frames and fps:
                dur = float(frames) / float(fps)
        if not dur:
            # ffprobe format.duration fallback (covers audio-only files)
            import subprocess
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=10,
            )
            dur = float(r.stdout.strip() or 0.0)
        dur = float(dur) if dur else 5.0
    except Exception:
        logger.warning("duration probe failed for %s", path, exc_info=True)
        dur = 5.0
    _DURATION_CACHE[path] = dur
    return dur
_EDGE_HIT_PX = 6  # pixels from clip edge to trigger trim cursor
_SNAP_PX = 12  # snap distance in pixels for edge snapping
_FADE_HANDLE_PX = 8  # radius of fade drag handles
_PPS_MIN = 10.0
_PPS_MAX = 500.0
_PPS_DEFAULT = 80.0

# -- Cached fonts and colors (avoid per-frame allocation) --
# Legibility (2026-05 UI pass): raised to readable sizes for the 90px tracks.
_FONT_RULER = QFont("sans-serif", 9)
_FONT_CLIP = QFont("sans-serif", 11, QFont.Bold)
_FONT_SMALL = QFont("sans-serif", 9)
_FONT_DUR = QFont("sans-serif", 9)
_FONT_AUDIO_LABEL = QFont("sans-serif", 9)
_FONT_BADGE = QFont("sans-serif", 8)
_FONT_MUTED = QFont("sans-serif", 11, QFont.Bold)

_COLOR_BG = QColor("#1a1a1a")
_COLOR_RULER_BG = QColor("#222")
_COLOR_RULER_TICK = QColor("#666")
_COLOR_RULER_SEP = QColor("#444")
_COLOR_TRACK_VIDEO_BG = QColor("#1e1e1e")
_COLOR_TRACK_AUDIO_BG = QColor("#1b1e1e")
_COLOR_TRACK_SEP = QColor("#333")
_COLOR_HATCH = QColor(255, 255, 255, 20)
_COLOR_PLAYHEAD = QColor("#ff3333")
_COLOR_SNAP = QColor("#2ecc71")
_COLOR_WHITE = QColor("#fff")
_COLOR_WHITE_120 = QColor(255, 255, 255, 120)
_COLOR_WHITE_160 = QColor(255, 255, 255, 160)
_COLOR_WHITE_60 = QColor(255, 255, 255, 60)
_COLOR_DIM = QColor(0, 0, 0, 150)
_COLOR_DIM_120 = QColor(0, 0, 0, 120)
_COLOR_DIM_100 = QColor(0, 0, 0, 100)
_COLOR_DIM_160 = QColor(0, 0, 0, 160)
_COLOR_MUTED_TEXT = QColor("#e55")
_COLOR_GROUP = QColor("#e67e22")
_COLOR_ROTATION_BADGE = QColor("#f0c040")
_COLOR_GHOST_BORDER = QColor(255, 255, 255, 120)
_COLOR_RAZOR = QColor("#e74c3c")
_COLOR_DROP = QColor("#3498db")
_COLOR_RUBBERBAND = QColor(100, 180, 255)
_COLOR_RUBBERBAND_FILL = QColor(100, 180, 255, 30)
_COLOR_TRANSITION = QColor(128, 0, 200, 60)
_COLOR_TRANSITION_LINE = QColor(200, 100, 255, 120)
_COLOR_AUDIO_LABEL = QColor("#ccc")
_COLOR_GHOST_DEFAULT = QColor("#4a90d9")

_PEN_PLAYHEAD = QPen(QColor("#ff3333"), 2)
_PEN_SNAP = QPen(QColor("#2ecc71"), 2)
_PEN_SELECT = QPen(QColor("#fff"), 2)


class MultiTrackCanvasWidget(QWidget):
    """Custom-painted multi-track timeline area.

    Signals:
        playhead_moved(float): playhead position in seconds
        clip_selected(str, int): (track_id, clip_index) selected
        clip_context_menu_requested(str, int, QPoint): (track_id, clip_idx, global_pos)
        clip_dropped(str, str, float): (track_id, clip_path, start_time)
        clip_moved(str, int, str, float): (src_track_id, src_clip_idx, dst_track_id, start_time)
        clips_changed(): emitted when any clip is repositioned
        razor_split(str, int, float): (track_id, clip_idx, offset_seconds)
        zoom_changed(float): emitted when pps changes
        marker_added(float): emitted when user double-clicks ruler to add marker
    """

    playhead_moved = Signal(float)
    clip_selected = Signal(str, int)  # (track_id, clip_index)
    clip_context_menu_requested = Signal(str, int, QPoint)
    multi_clip_context_menu_requested = Signal(QPoint)  # right-click with multi-selection
    clip_dropped = Signal(str, str, float)  # (track_id, path, start_time)
    clip_moved = Signal(str, int, str, float)  # (src_track, src_idx, dst_track, start_time)
    clips_changed = Signal()
    space_pressed = Signal()
    clip_deleted = Signal(str)  # clip path when Delete key removes a clip
    razor_split = Signal(str, int, float)  # (track_id, clip_idx, offset_seconds)
    zoom_changed = Signal(float)  # pps value
    marker_added = Signal(float)  # time in seconds

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._tracks: list[TimelineTrack] = []
        self._track_by_id: dict[str, TimelineTrack] = {}  # O(1) lookup
        self._track_y_cache: dict[str, int] = {}  # O(1) y-offset lookup
        self._playhead: float = 0.0
        self._pps: float = _PPS_DEFAULT  # pixels per second (zoom level)
        self._selected_track_id: str = ""
        self._selected_clip_idx: int = -1
        # Multi-selection: list of (track_id, clip_idx) tuples
        self._multi_selection: list[tuple[str, int]] = []

        # Drag state (playhead/scrub handled directly, rest delegated to tool)
        self._dragging_playhead: bool = False

        # Snap indicator (seconds, or -1 if not snapping)
        self._snap_line: float = -1.0
        # Snap candidate cache (invalidated on track/clip changes)
        self._snap_cache: list[tuple[float, int]] | None = None
        self._snap_cache_exclude: tuple[str, int] = ("", -1)

        # Ghost preview during drag
        self._ghost: dict | None = None  # {rect: QRectF, track_y: int, name: str, color: QColor}

        # Rubber-band selection rectangle (set by SelectTool)
        self._rubber_band_rect: tuple[int, int, int, int] | None = None

        # Razor tool preview line
        self._razor_preview_x: int = -1

        # Drop indicator
        self._drop_track_id: str = ""
        self._drop_x: int = -1

        # Markers
        self._markers: list[TimelineMarker] = []

        # Current tool instance (lazy-init on first use)
        self._tool_mode: ToolMode = ToolMode.SELECT
        self._current_tool: object | None = None  # TimelineTool instance
        self._init_tool()

        # Snap toggle
        self.snap_enabled: bool = True

        # Waveform pixmap cache: (clip_path, waveform_len, width, height, pps_int) -> QPixmap
        self._waveform_cache: dict[tuple, QPixmap] = {}

        # Invalidate snap cache when clips change
        self.clips_changed.connect(self._invalidate_snap_cache)

    def _invalidate_snap_cache(self) -> None:
        self._snap_cache = None

    def _init_tool(self) -> None:
        """Initialize the current tool from _tool_mode."""
        from sdqt.widgets.timeline_tools import (
            SelectTool, ScrubTool, RazorTool, SlipTool, SlideTool,
        )
        tool_map = {
            ToolMode.SELECT: SelectTool,
            ToolMode.SCRUB: ScrubTool,
            ToolMode.RAZOR: RazorTool,
            ToolMode.SLIP: SlipTool,
            ToolMode.SLIDE: SlideTool,
        }
        cls = tool_map.get(self._tool_mode, SelectTool)
        self._current_tool = cls(self)

    # -- Public API --------------------------------------------------------

    @property
    def playhead(self) -> float:
        return self._playhead

    @property
    def tracks(self) -> list[TimelineTrack]:
        return self._tracks

    @property
    def pps(self) -> float:
        return self._pps

    def set_pps(self, value: float) -> None:
        value = max(_PPS_MIN, min(_PPS_MAX, value))
        if abs(value - self._pps) < 0.01:
            return
        self._pps = value
        self._waveform_cache.clear()
        self._update_size()
        self.update()
        self.zoom_changed.emit(value)

    def set_tracks(self, tracks: list[TimelineTrack]) -> None:
        self._tracks = tracks
        self._track_by_id = {t.id: t for t in tracks}
        self._snap_cache = None  # invalidate snap cache
        self._rebuild_y_offsets()
        self._update_size()
        self.update()

    def _rebuild_y_offsets(self) -> None:
        """Cache y-offsets for each track (invalidated on set_tracks)."""
        self._track_y_cache: dict[str, int] = {}
        y = _RULER_HEIGHT
        for t in self._tracks:
            self._track_y_cache[t.id] = y
            y += t.height

    def set_playhead(self, seconds: float) -> None:
        total = self._total_duration()
        self._playhead = max(0.0, min(seconds, total)) if total > 0 else 0.0
        self._snap_cache = None  # playhead is a snap target
        self.update()

    def set_tool_mode(self, mode: ToolMode) -> None:
        self._tool_mode = mode
        self._init_tool()
        cursor = self._current_tool.cursor() if self._current_tool else Qt.ArrowCursor
        self.setCursor(cursor)

    def set_markers(self, markers: list[TimelineMarker]) -> None:
        self._markers = markers
        self._snap_cache = None  # markers are snap targets
        self.update()

    def select_clip(self, track_id: str, clip_idx: int) -> None:
        self._selected_track_id = track_id
        self._selected_clip_idx = clip_idx
        self.update()

    def clear_selection(self) -> None:
        self._selected_track_id = ""
        self._selected_clip_idx = -1
        self._multi_selection.clear()
        self.update()

    def zoom_to_fit(self, visible_width: int) -> None:
        total = self._total_duration()
        if total > 0.1:
            # Add 5% padding so the last clip doesn't touch the edge
            self.set_pps(max(_PPS_MIN, (visible_width * 0.95) / total))
        else:
            self.set_pps(_PPS_DEFAULT)

    # -- Internal helpers --------------------------------------------------

    def _total_duration(self) -> float:
        if not self._tracks:
            return 0.0
        return max((t.total_duration() for t in self._tracks), default=0.0)

    def _update_size(self) -> None:
        total_h = _RULER_HEIGHT + sum(t.height for t in self._tracks)
        # Canvas must be at least as wide as the parent viewport so the
        # ruler and track backgrounds span the full visible area.
        parent_w = self.parentWidget().width() if self.parentWidget() else 800
        content_w = int(self._total_duration() * self._pps) + 100
        total_w = max(content_w, parent_w, 600)
        self.setMinimumSize(total_w, max(total_h, 100))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_size()

    def _track_y_offset(self, track_id: str) -> int:
        """Get the y-offset for a track (below ruler). O(1) via cache."""
        if hasattr(self, "_track_y_cache") and track_id in self._track_y_cache:
            return self._track_y_cache[track_id]
        # Fallback: compute and cache
        y = _RULER_HEIGHT
        for t in self._tracks:
            if t.id == track_id:
                return y
            y += t.height
        return y

    def _track_at_y(self, y: int) -> TimelineTrack | None:
        """Find which track the y coordinate falls in."""
        cy = _RULER_HEIGHT
        for t in self._tracks:
            if cy <= y < cy + t.height:
                return t
            cy += t.height
        return None

    def _clip_at_pos(self, x: int, track: TimelineTrack) -> int:
        """Return clip index in track at pixel x, or -1."""
        sec = x / self._pps
        for i, c in enumerate(track.clips):
            if c.start_time <= sec < c.start_time + c.duration:
                return i
        return -1

    def _pos_to_seconds(self, x: int) -> float:
        return max(0.0, x / self._pps)

    def _is_on_playhead(self, x: int, y: int) -> bool:
        ph_x = int(self._playhead * self._pps)
        return abs(x - ph_x) <= _PLAYHEAD_HIT and y <= _RULER_HEIGHT

    def _edge_at_pos(self, x: int, y: int) -> tuple[TimelineTrack | None, int, str]:
        """Check if (x, y) is on a clip edge. Returns (track, clip_idx, "left"|"right"|"")."""
        track = self._track_at_y(y)
        if not track:
            return None, -1, ""
        for i, c in enumerate(track.clips):
            left_px = int(c.start_time * self._pps)
            right_px = int((c.start_time + c.duration) * self._pps)
            if abs(x - left_px) <= _EDGE_HIT_PX:
                return track, i, "left"
            if abs(x - right_px) <= _EDGE_HIT_PX:
                return track, i, "right"
        return None, -1, ""

    def _fade_handle_at_pos(self, x: int, y: int) -> tuple[TimelineTrack | None, int, str]:
        """Check if (x, y) is on a fade handle. Returns (track, clip_idx, "fade_in"|"fade_out"|"")."""
        track = self._track_at_y(y)
        if not track or track.track_type != TrackType.VIDEO:
            return None, -1, ""
        track_y = self._track_y_offset(track.id)
        for i, c in enumerate(track.clips):
            cx = int(c.start_time * self._pps)
            cw = int(c.duration * self._pps)
            m = _CLIP_MARGIN
            # Fade in handle: at top of clip, offset by fade_in pixels from left
            if c.fade_in > 0:
                handle_x = cx + m + int(c.fade_in * self._pps)
                handle_y = track_y + m
                if abs(x - handle_x) <= _FADE_HANDLE_PX and abs(y - handle_y) <= _FADE_HANDLE_PX:
                    return track, i, "fade_in"
            # Fade out handle: at top of clip, offset by fade_out pixels from right
            if c.fade_out > 0:
                handle_x = cx + cw - m - int(c.fade_out * self._pps)
                handle_y = track_y + m
                if abs(x - handle_x) <= _FADE_HANDLE_PX and abs(y - handle_y) <= _FADE_HANDLE_PX:
                    return track, i, "fade_out"
        return None, -1, ""

    # -- Snapping (priority-aware) -----------------------------------------

    def _collect_snap_candidates(
        self, exclude_track_id: str, exclude_clip_idx: int,
    ) -> list[tuple[float, int]]:
        """Collect snap candidates as (time, threshold_px) tuples, sorted by priority.

        Results are cached and reused when called with the same exclude params.
        Returns empty list when snapping is disabled.
        """
        if not self.snap_enabled:
            return []
        cache_key = (exclude_track_id, exclude_clip_idx)
        if self._snap_cache is not None and self._snap_cache_exclude == cache_key:
            return self._snap_cache

        candidates: list[tuple[float, int]] = []
        # Origin
        candidates.append((0.0, 10))
        # Playhead
        candidates.append((self._playhead, 10))
        # Markers
        for m in self._markers:
            candidates.append((m.time, 14))
        # Clip edges
        for t in self._tracks:
            is_same_track = t.id == exclude_track_id
            for ci, c in enumerate(t.clips):
                if t.id == exclude_track_id and ci == exclude_clip_idx:
                    continue
                threshold = 8 if is_same_track else 12
                candidates.append((c.start_time, threshold))
                candidates.append((c.start_time + c.duration, threshold))

        self._snap_cache = candidates
        self._snap_cache_exclude = cache_key
        return candidates

    def _collect_snap_edges(self, exclude_track_id: str, exclude_clip_idx: int) -> list[float]:
        """Backward-compat: collect flat list of snap edge times."""
        return [t for t, _ in self._collect_snap_candidates(exclude_track_id, exclude_clip_idx)]

    def _snap_edge(
        self, seconds: float, snap_candidates: list, is_trim: bool = False,
    ) -> tuple[float, bool]:
        """Snap a time value to the nearest candidate if within threshold.

        snap_candidates can be list[float] (legacy) or list[tuple[float, int]] (priority).
        Returns (snapped_value, did_snap).
        """
        best = seconds
        best_dist = float("inf")
        snapped = False
        for item in snap_candidates:
            if isinstance(item, tuple):
                edge_time, threshold_px = item
            else:
                edge_time = item
                threshold_px = _SNAP_PX
            if is_trim:
                threshold_px = max(4, threshold_px // 2)
            threshold_sec = threshold_px / self._pps
            dist = abs(seconds - edge_time)
            if dist < threshold_sec and dist < best_dist:
                best = edge_time
                best_dist = dist
                snapped = True
        return best, snapped

    def _snap_to_playhead(self, seconds: float) -> float:
        """Snap a time value to playhead or clip edges (used during trim)."""
        candidates = self._collect_snap_candidates("", -1)
        result, _ = self._snap_edge(seconds, candidates, is_trim=True)
        return result

    # -- Paint -------------------------------------------------------------

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        vis_rect = event.rect()

        # Background
        p.fillRect(0, 0, w, h, _COLOR_BG)

        # Time ruler
        self._paint_ruler(p, w)

        # Track lanes (with culling)
        y = _RULER_HEIGHT
        for track in self._tracks:
            self._paint_track(p, track, y, w, vis_rect)
            y += track.height

        # Playhead (full height)
        ph_x = int(self._playhead * self._pps)
        if 0 <= ph_x <= w:
            p.setPen(_PEN_PLAYHEAD)
            p.drawLine(ph_x, 0, ph_x, h)
            # Triangle at top
            p.setBrush(_COLOR_PLAYHEAD)
            p.setPen(Qt.NoPen)
            p.drawPolygon(QPolygon([
                QPoint(ph_x - 7, 0), QPoint(ph_x + 7, 0), QPoint(ph_x, 11),
            ]))

        # Snap indicator (green line when clip edge snaps)
        if self._snap_line >= 0:
            sx = int(self._snap_line * self._pps)
            if 0 <= sx <= w:
                p.setPen(_PEN_SNAP)
                p.drawLine(sx, _RULER_HEIGHT, sx, h)

        # Ghost preview during drag
        if self._ghost:
            g = self._ghost
            ghost_color = QColor(g.get("color", _COLOR_GHOST_DEFAULT))
            ghost_color.setAlpha(100)
            p.setBrush(QBrush(ghost_color))
            p.setPen(QPen(_COLOR_GHOST_BORDER, 1, Qt.DashLine))
            rect = g.get("rect")
            track_y = g.get("track_y", _RULER_HEIGHT)
            if rect:
                draw_rect = QRectF(rect.x(), track_y, rect.width(), rect.height())
                p.drawRoundedRect(draw_rect, _CLIP_RADIUS, _CLIP_RADIUS)
                p.setPen(_COLOR_WHITE_160)
                p.setFont(_FONT_CLIP)
                text_rect = QRectF(draw_rect.x() + 4, draw_rect.y() + 2, draw_rect.width() - 8, draw_rect.height() - 4)
                p.drawText(text_rect, Qt.AlignLeft | Qt.AlignTop, g.get("name", ""))

        # Razor preview line
        if self._razor_preview_x >= 0:
            p.setPen(QPen(_COLOR_RAZOR, 1, Qt.DashDotLine))
            p.drawLine(self._razor_preview_x, _RULER_HEIGHT, self._razor_preview_x, h)

        # Drop indicator
        if self._drop_x >= 0 and self._drop_track_id:
            dy = self._track_y_offset(self._drop_track_id)
            track = next((t for t in self._tracks if t.id == self._drop_track_id), None)
            if track:
                p.setPen(QPen(_COLOR_DROP, 2, Qt.DashLine))
                p.drawLine(self._drop_x, dy, self._drop_x, dy + track.height)

        # Rubber-band selection rectangle
        if self._rubber_band_rect is not None:
            rx, ry, rw, rh = self._rubber_band_rect
            p.setPen(QPen(_COLOR_RUBBERBAND, 1, Qt.DashLine))
            p.setBrush(QBrush(_COLOR_RUBBERBAND_FILL))
            p.drawRect(rx, ry, rw, rh)

        p.end()

    def _paint_ruler(self, p: QPainter, w: int) -> None:
        p.fillRect(0, 0, w, _RULER_HEIGHT, _COLOR_RULER_BG)
        p.setPen(QPen(_COLOR_RULER_TICK, 1))
        p.setFont(_FONT_RULER)

        total = max(self._total_duration(), w / self._pps)

        # Adaptive tick spacing based on zoom
        if self._pps >= 40:
            major_step = 1
            minor_count = 4
        elif self._pps >= 15:
            major_step = 5
            minor_count = 5
        else:
            major_step = 10
            minor_count = 2

        sec = 0
        while sec <= total:
            x = int(sec * self._pps)
            if x > w:
                break
            # Major tick
            p.setPen(QPen(_COLOR_RULER_TICK, 1))
            p.drawLine(x, _RULER_HEIGHT - 8, x, _RULER_HEIGHT)
            p.drawText(x + 2, _RULER_HEIGHT - 10, f"{sec}s")
            # Minor ticks
            for q in range(1, minor_count):
                qx = int((sec + q * major_step / minor_count) * self._pps)
                if qx <= w:
                    p.drawLine(qx, _RULER_HEIGHT - 4, qx, _RULER_HEIGHT)
            sec += major_step

        # Markers on ruler
        for marker in self._markers:
            mx = int(marker.time * self._pps)
            if 0 <= mx <= w:
                mc = QColor(marker.color)
                p.setBrush(mc)
                p.setPen(Qt.NoPen)
                # Diamond shape
                p.drawPolygon(QPolygon([
                    QPoint(mx, 2), QPoint(mx + 5, 8),
                    QPoint(mx, 14), QPoint(mx - 5, 8),
                ]))
                # Name text if space permits
                if marker.name:
                    p.setPen(mc)
                    p.setFont(_FONT_SMALL)
                    p.drawText(mx + 6, 10, marker.name)
                    p.setFont(_FONT_RULER)

        # Separator line
        p.setPen(QPen(_COLOR_RULER_SEP, 1))
        p.drawLine(0, _RULER_HEIGHT - 1, w, _RULER_HEIGHT - 1)

    def _paint_track(
        self, p: QPainter, track: TimelineTrack, y: int, w: int, vis_rect: QRectF,
    ) -> None:
        th = track.height
        vis_left = vis_rect.x()
        vis_right = vis_rect.x() + vis_rect.width()

        # Track background
        bg = _COLOR_TRACK_VIDEO_BG if track.track_type == TrackType.VIDEO else _COLOR_TRACK_AUDIO_BG
        p.fillRect(0, y, w, th, bg)

        # Locked track overlay (diagonal hatch)
        if track.locked:
            p.setPen(QPen(_COLOR_HATCH, 1))
            # Draw 45° stripes
            step = 12
            for offset in range(0, w + th, step):
                p.drawLine(offset, y, offset - th, y + th)

        # Track separator
        p.setPen(QPen(_COLOR_TRACK_SEP, 1))
        p.drawLine(0, y + th - 1, w, y + th - 1)

        # Clips (with viewport culling)
        p.setFont(_FONT_CLIP)

        for i, clip in enumerate(track.clips):
            cx = int(clip.start_time * self._pps)
            cw = int(clip.duration * self._pps)
            clip_right = cx + cw

            # Viewport culling: skip offscreen clips
            if clip_right < vis_left - 10 or cx > vis_right + 10:
                continue

            is_selected = (
                (track.id == self._selected_track_id and i == self._selected_clip_idx)
                or (track.id, i) in self._multi_selection
            )

            if track.track_type == TrackType.VIDEO:
                self._paint_video_clip(p, clip, cx, y, cw, th, is_selected, vis_left, vis_right)
            else:
                self._paint_audio_clip(p, clip, cx, y, cw, th, is_selected, vis_left, vis_right)

            # Trim handles (subtle edge indicators)
            self._paint_trim_handles(p, cx, y, cw, th)

    def _paint_video_clip(
        self, p: QPainter, clip: TimelineClip,
        x: int, y: int, w: int, h: int, selected: bool,
        vis_left: float = 0, vis_right: float = 99999,
    ) -> None:
        m = _CLIP_MARGIN
        # Upper 60% for video, lower 40% for waveform sublane — at the 90px
        # track height the waveform sublane now gets a legible ~33px.
        video_h = int((h - 2 * m) * 0.6)
        wave_h = h - 2 * m - video_h

        rect = QRectF(x + m, y + m, w - 2 * m, h - 2 * m)

        # Clip body
        base = QColor(clip.color)
        if selected:
            base = base.lighter(130)
        p.setBrush(QBrush(base))
        p.setPen(QPen(base.darker(150), 1))
        p.drawRoundedRect(rect, _CLIP_RADIUS, _CLIP_RADIUS)

        # Thumbnail at left edge (if available and clip wide enough)
        text_x_offset = 4
        if clip.thumbnail and not clip.thumbnail.isNull() and w > 60:
            thumb_h = h - 2 * m - 4
            thumb_w = min(int(thumb_h * clip.thumbnail.width() / max(clip.thumbnail.height(), 1)), w // 3)
            thumb_rect = QRectF(x + m + 2, y + m + 2, thumb_w, thumb_h)
            p.drawPixmap(thumb_rect.toRect(), clip.thumbnail)
            text_x_offset = thumb_w + 6
            # Second thumbnail at right edge for wide clips
            if w > 200:
                thumb_rect2 = QRectF(x + w - m - thumb_w - 2, y + m + 2, thumb_w, thumb_h)
                p.drawPixmap(thumb_rect2.toRect(), clip.thumbnail)

        # Label
        p.setPen(_COLOR_WHITE)
        text_rect = QRectF(x + m + text_x_offset, y + m + 2, w - 2 * m - text_x_offset - 4, video_h - 4)
        p.drawText(text_rect, Qt.AlignLeft | Qt.AlignTop, clip.name)

        p.setFont(_FONT_DUR)
        p.drawText(text_rect, Qt.AlignLeft | Qt.AlignBottom, f"{clip.duration:.1f}s")
        p.setFont(_FONT_CLIP)

        # Fade-in triangle
        fade_in = getattr(clip, "fade_in", 0.0) or 0.0
        fade_out = getattr(clip, "fade_out", 0.0) or 0.0
        if fade_in > 0 and w > 20:
            fi_px = int(fade_in * self._pps)
            fi_px = min(fi_px, w - 2 * m)
            path = QPainterPath()
            path.moveTo(x + m, y + m + h - 2 * m)
            path.lineTo(x + m, y + m)
            path.lineTo(x + m + fi_px, y + m)
            path.closeSubpath()
            p.fillPath(path, _COLOR_DIM_100)
            p.setBrush(_COLOR_WHITE)
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(x + m + fi_px, y + m), 4, 4)

        if fade_out > 0 and w > 20:
            fo_px = int(fade_out * self._pps)
            fo_px = min(fo_px, w - 2 * m)
            right_edge = x + w - m
            path = QPainterPath()
            path.moveTo(right_edge, y + m + h - 2 * m)
            path.lineTo(right_edge, y + m)
            path.lineTo(right_edge - fo_px, y + m)
            path.closeSubpath()
            p.fillPath(path, _COLOR_DIM_100)
            p.setBrush(_COLOR_WHITE)
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(right_edge - fo_px, y + m), 4, 4)

        # Transition indicator (crossfade)
        transition = getattr(clip, "transition_in", None)
        if transition and isinstance(transition, dict):
            t_dur = transition.get("duration", 0.5)
            t_px = int(t_dur * self._pps)
            p.fillRect(QRectF(x + m, y + m, min(t_px, w - 2 * m), h - 2 * m), _COLOR_TRANSITION)
            p.setPen(QPen(_COLOR_TRANSITION_LINE, 1))
            p.drawLine(x + m, y + m, x + m + t_px, y + h - m)
            p.drawLine(x + m, y + h - m, x + m + t_px, y + m)

        # Waveform sublane (if available)
        if clip.waveform and clip.has_audio:
            wave_y = y + m + video_h
            self._paint_waveform(
                p, clip.waveform, x + m, wave_y, w - 2 * m, wave_h,
                _WAVEFORM_COLOR_VIDEO, vis_left, vis_right,
            )

        # Selection border
        if selected:
            p.setBrush(Qt.NoBrush)
            p.setPen(_PEN_SELECT)
            p.drawRoundedRect(rect, _CLIP_RADIUS, _CLIP_RADIUS)

        # Group indicator
        group_id = getattr(clip, "group_id", "")
        if group_id:
            p.setPen(QPen(_COLOR_GROUP, 2, Qt.DotLine))
            p.setBrush(Qt.NoBrush)
            p.drawLine(x + m, y + h - m - 1, x + w - m, y + h - m - 1)

        # Rotation badge (bottom-right corner)
        rot = getattr(clip, "rotation", 0) % 360
        if rot and w > 40:
            p.setFont(_FONT_BADGE)
            label = f"{rot}\u00b0"
            p.setPen(Qt.NoPen)
            p.setBrush(_COLOR_DIM_160)
            bx = x + w - m - 34
            by = y + h - m - 16
            p.drawRoundedRect(QRectF(bx, by, 32, 14), 3, 3)
            p.setPen(_COLOR_ROTATION_BADGE)
            p.drawText(QRectF(bx, by, 32, 14), Qt.AlignCenter, label)
            p.setFont(_FONT_CLIP)

        # Muted overlay
        if getattr(clip, "muted", False):
            p.fillRect(QRectF(x + m, y + m, w - 2 * m, h - 2 * m), _COLOR_DIM_120)
            p.setPen(_COLOR_MUTED_TEXT)
            p.setFont(_FONT_MUTED)
            p.drawText(QRectF(x + m, y + m, w - 2 * m, h - 2 * m),
                       Qt.AlignCenter, "\U0001F507")
            p.setFont(_FONT_CLIP)

    def _paint_audio_clip(
        self, p: QPainter, clip: TimelineClip,
        x: int, y: int, w: int, h: int, selected: bool,
        vis_left: float = 0, vis_right: float = 99999,
    ) -> None:
        m = _CLIP_MARGIN
        rect = QRectF(x + m, y + m, w - 2 * m, h - 2 * m)

        # Background
        base = QColor(clip.color) if clip.color != QColor("#4a90d9") else QColor("#1abc9c")
        if selected:
            base = base.lighter(130)
        darker = base.darker(200)
        p.setBrush(QBrush(darker))
        p.setPen(QPen(base.darker(150), 1))
        p.drawRoundedRect(rect, _CLIP_RADIUS, _CLIP_RADIUS)

        # Label
        p.setPen(_COLOR_AUDIO_LABEL)
        p.setFont(_FONT_AUDIO_LABEL)
        p.drawText(QRectF(x + m + 4, y + m + 1, w - 2 * m - 8, 14),
                   Qt.AlignLeft | Qt.AlignTop, clip.name)

        # Waveform (full height minus label)
        if clip.waveform:
            self._paint_waveform(
                p, clip.waveform, x + m, y + m + 14,
                w - 2 * m, h - 2 * m - 16, _WAVEFORM_COLOR_AUDIO,
                vis_left, vis_right,
            )
        p.setFont(_FONT_CLIP)

        # Audio fade indicators
        fade_in = getattr(clip, "fade_in", 0.0) or 0.0
        fade_out = getattr(clip, "fade_out", 0.0) or 0.0
        if fade_in > 0:
            fi_px = min(int(fade_in * self._pps), w - 2 * m)
            p.setPen(QPen(QColor("#1abc9c"), 2))
            p.drawLine(x + m, y + h // 2, x + m + fi_px, y + m + 2)
        if fade_out > 0:
            fo_px = min(int(fade_out * self._pps), w - 2 * m)
            p.setPen(QPen(QColor("#1abc9c"), 2))
            p.drawLine(x + w - m - fo_px, y + m + 2, x + w - m, y + h // 2)

        # Selection border
        if selected:
            p.setBrush(Qt.NoBrush)
            p.setPen(_PEN_SELECT)
            p.drawRoundedRect(rect, _CLIP_RADIUS, _CLIP_RADIUS)

        # Muted overlay
        if getattr(clip, "muted", False):
            p.fillRect(QRectF(x + m, y + m, w - 2 * m, h - 2 * m), _COLOR_DIM_120)
            p.setPen(_COLOR_MUTED_TEXT)
            p.setFont(_FONT_MUTED)
            p.drawText(QRectF(x + m, y + m, w - 2 * m, h - 2 * m),
                       Qt.AlignCenter, "\U0001F507")
            p.setFont(_FONT_CLIP)

    def _paint_waveform(
        self, p: QPainter, waveform: list[float],
        x: int, y: int, w: int, h: int, color: QColor,
        vis_left: float = 0, vis_right: float = 99999,
    ) -> None:
        if not waveform or w <= 0 or h <= 0:
            return

        # Use cached QPixmap keyed by content identity + dimensions
        pps_key = int(self._pps * 10)  # round to avoid excessive invalidation
        cache_key = (id(waveform), len(waveform), w, h, pps_key, color.name())
        pm = self._waveform_cache.get(cache_key)
        if pm is None:
            pm = QPixmap(w, h)
            pm.fill(QColor(0, 0, 0, 0))
            wp = QPainter(pm)
            wp.setPen(QPen(color, 1))
            n = len(waveform)
            center_y = h // 2
            half_h = h / 2.0
            for px in range(w):
                idx = min(int(px * n / w), n - 1)
                amp = waveform[idx] * half_h
                wp.drawLine(px, int(center_y - amp), px, int(center_y + amp))
            wp.end()
            self._waveform_cache[cache_key] = pm
            # Limit cache size
            if len(self._waveform_cache) > 200:
                # Remove oldest entries
                keys = list(self._waveform_cache.keys())
                for k in keys[:100]:
                    del self._waveform_cache[k]

        # Draw only visible portion
        start_px = max(0, int(vis_left - x))
        end_px = min(w, int(vis_right - x) + 1)
        if start_px < end_px:
            p.drawPixmap(x + start_px, y, pm, start_px, 0, end_px - start_px, h)

    def _paint_trim_handles(
        self, p: QPainter, x: int, y: int, w: int, h: int,
    ) -> None:
        """Paint subtle bracket indicators at clip edges."""
        m = _CLIP_MARGIN
        handle_w = 3
        # Left handle [
        p.fillRect(x + m, y + m, handle_w, h - 2 * m, _COLOR_WHITE_60)
        # Right handle ]
        p.fillRect(x + w - m - handle_w, y + m, handle_w, h - 2 * m, _COLOR_WHITE_60)

    # -- Mouse events (delegate to tool) -----------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        x = int(event.position().x())
        y = int(event.position().y())

        if event.button() == Qt.RightButton:
            track = self._track_at_y(y)
            if track:
                idx = self._clip_at_pos(x, track)
                if idx >= 0:
                    entry = (track.id, idx)
                    if len(self._multi_selection) > 1 and entry in self._multi_selection:
                        self.multi_clip_context_menu_requested.emit(
                            self.mapToGlobal(event.position().toPoint()),
                        )
                    else:
                        self._multi_selection.clear()
                        self._selected_track_id = track.id
                        self._selected_clip_idx = idx
                        self.clip_selected.emit(track.id, idx)
                        self.update()
                        self.clip_context_menu_requested.emit(
                            track.id, idx,
                            self.mapToGlobal(event.position().toPoint()),
                        )
            return

        if event.button() != Qt.LeftButton:
            return

        # Playhead grab or ruler click-to-jump (universal, not tool-specific)
        if self._is_on_playhead(x, y):
            self._dragging_playhead = True
            return
        if y < _RULER_HEIGHT:
            # Double-click on ruler → add marker
            # (handled by mouseDoubleClickEvent)
            self._playhead = self._pos_to_seconds(x)
            self._dragging_playhead = True
            self.playhead_moved.emit(self._playhead)
            self.update()
            return

        # Delegate to current tool
        if self._current_tool:
            self._current_tool.on_press(x, y, event)
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        x = int(event.position().x())
        y = int(event.position().y())
        if event.button() == Qt.LeftButton and y < _RULER_HEIGHT:
            # Double-click on ruler: add marker (if not on playhead)
            if not self._is_on_playhead(x, y):
                time = self._pos_to_seconds(x)
                self.marker_added.emit(time)
                return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        x = int(event.position().x())
        y = int(event.position().y())

        if self._dragging_playhead:
            raw = self._pos_to_seconds(x)
            total = self._total_duration()
            raw = max(0.0, min(raw, total))
            # Snap playhead to clip edges during scrub
            snap_threshold = 8 / self._pps  # 8 pixels
            for t in self._tracks:
                for c in t.clips:
                    for edge in (c.start_time, c.start_time + c.duration):
                        if abs(raw - edge) < snap_threshold:
                            raw = edge
                            break
            self._playhead = raw
            self.playhead_moved.emit(self._playhead)
            self.update()
            return

        # Delegate to current tool
        if self._current_tool:
            self._current_tool.on_move(x, y, event)
            # Update cursor from tool hover
            cursor = self._current_tool.on_hover(x, y)
            if cursor is not None:
                self.setCursor(cursor)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging_playhead = False

        # Delegate to current tool
        if self._current_tool:
            self._current_tool.on_release(
                int(event.position().x()), int(event.position().y()), event,
            )
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Ctrl+Scroll to zoom, preserving cursor-time position."""
        if event.modifiers() & Qt.ControlModifier:
            scroll_area = self.parent()
            scroll_x = 0
            if scroll_area and hasattr(scroll_area, "horizontalScrollBar"):
                scroll_x = scroll_area.horizontalScrollBar().value()

            old_pps = self._pps
            cursor_x = event.position().x()
            time_at_cursor = (scroll_x + cursor_x) / old_pps

            steps = event.angleDelta().y() / 120.0
            factor = 1.15 ** steps
            new_pps = max(_PPS_MIN, min(_PPS_MAX, old_pps * factor))
            self.set_pps(new_pps)

            # Adjust scroll to keep cursor-time position stable
            new_scroll_x = int(time_at_cursor * new_pps - cursor_x)
            if scroll_area and hasattr(scroll_area, "horizontalScrollBar"):
                scroll_area.horizontalScrollBar().setValue(max(0, new_scroll_x))

            event.accept()
            return
        super().wheelEvent(event)

    def _frame_step(self) -> float:
        """Return one frame duration based on project FPS (default 25)."""
        return 1.0 / getattr(self, "_project_fps", 25)

    def set_project_fps(self, fps: float) -> None:
        self._project_fps = max(1.0, fps)

    def keyPressEvent(self, event) -> None:
        total = self._total_duration()
        mods = event.modifiers()
        ctrl = bool(mods & Qt.ControlModifier)
        shift = bool(mods & Qt.ShiftModifier)

        if event.key() == Qt.Key_Space:
            self.space_pressed.emit()
        elif event.key() == Qt.Key_Left:
            step = self._frame_step()
            self._playhead = max(0, self._playhead - step)
            self.playhead_moved.emit(self._playhead)
            self.update()
        elif event.key() == Qt.Key_Right:
            step = self._frame_step()
            self._playhead = min(total, self._playhead + step)
            self.playhead_moved.emit(self._playhead)
            self.update()
        elif event.key() == Qt.Key_Escape:
            if self._current_tool and hasattr(self._current_tool, "cancel"):
                self._current_tool.cancel()
                self.update()
        elif event.key() == Qt.Key_A and ctrl:
            # Select all clips across all tracks
            self._multi_selection.clear()
            for t in self._tracks:
                for i in range(len(t.clips)):
                    self._multi_selection.append((t.id, i))
            if self._multi_selection:
                first = self._multi_selection[0]
                self._selected_track_id = first[0]
                self._selected_clip_idx = first[1]
            self.update()
        elif event.key() == Qt.Key_Delete:
            self._delete_selected(ripple=shift)
        else:
            super().keyPressEvent(event)

    def _delete_selected(self, ripple: bool = False) -> None:
        """Delete selected clips. If multi-selection, delete all. Shift+Delete = ripple."""
        targets = list(self._multi_selection) if self._multi_selection else []
        if not targets and self._selected_track_id and self._selected_clip_idx >= 0:
            targets = [(self._selected_track_id, self._selected_clip_idx)]
        if not targets:
            return

        # Group by track and sort indices descending so pops don't shift later indices
        by_track: dict[str, list[int]] = {}
        for track_id, idx in targets:
            by_track.setdefault(track_id, []).append(idx)

        for track_id, indices in by_track.items():
            track = self._track_by_id.get(track_id)
            if not track or track.locked:
                continue
            for idx in sorted(indices, reverse=True):
                if 0 <= idx < len(track.clips):
                    clip = track.clips[idx]
                    gap_start = clip.start_time
                    gap_dur = clip.duration
                    self.clip_deleted.emit(clip.path)
                    track.clips.pop(idx)
                    # Ripple: shift subsequent clips left to fill the gap
                    if ripple:
                        track.ripple_shift(gap_start, -gap_dur)
                        # Multi-track ripple: apply same shift to all other tracks
                        if getattr(self, "_ripple_all_tracks", False):
                            for other in self._tracks:
                                if other.id != track_id and not other.locked:
                                    other.ripple_shift(gap_start, -gap_dur)

        self.clear_selection()
        self.clips_changed.emit()  # triggers stale decoder cleanup in TimelineTab
        self._update_size()
        self.update()

    # -- Drag & Drop -------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasFormat(MIME_TIMELINE_CLIP):
            event.acceptProposedAction()
        elif event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        x = int(event.position().x())
        y = int(event.position().y())
        track = self._track_at_y(y)
        if track and not track.locked:
            self._drop_track_id = track.id
            self._drop_x = x
        else:
            self._drop_track_id = ""
            self._drop_x = -1
        self.update()
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        self._drop_track_id = ""
        self._drop_x = -1

        x = int(event.position().x())
        y = int(event.position().y())
        track = self._track_at_y(y)
        if not track or track.locked:
            event.ignore()
            self.update()
            return

        start_time = self._pos_to_seconds(x)

        if event.mimeData().hasFormat(MIME_TIMELINE_CLIP):
            data = bytes(event.mimeData().data(MIME_TIMELINE_CLIP)).decode()

            # Internal move: "track_id:clip_idx"
            if ":" in data:
                parts = data.split(":", 1)
                try:
                    src_track_id = parts[0]
                    src_idx = int(parts[1])
                    self.clip_moved.emit(src_track_id, src_idx, track.id, start_time)
                    event.acceptProposedAction()
                    self.update()
                    return
                except (ValueError, IndexError):
                    pass

            # External drop from library (data is a path)
            if Path(data).is_file():
                self.clip_dropped.emit(track.id, data, start_time)
                event.acceptProposedAction()
                self.update()
                return

        if event.mimeData().hasUrls():
            exts = {".mp4", ".mov", ".avi", ".mkv", ".webm",
                    ".mp3", ".wav", ".flac", ".ogg", ".aac"}
            # Collect drop-order file list first so we can chain offsets
            paths: list[str] = []
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    if Path(path).suffix.lower() in exts:
                        paths.append(path)

            # Place each clip end-to-end starting at the drop position.
            # Probe each clip's duration so we know where the next one
            # starts. Single clips still land exactly at start_time, so
            # legacy single-drop behaviour is preserved.
            cursor = start_time
            for p in paths:
                self.clip_dropped.emit(track.id, p, cursor)
                cursor += _probe_duration_cached(p)

            event.acceptProposedAction()
            self.update()
            return

        event.ignore()
        self.update()

    def dragLeaveEvent(self, event) -> None:
        self._drop_track_id = ""
        self._drop_x = -1
        self.update()

    # -- Size hint ---------------------------------------------------------

    def sizeHint(self) -> QSize:
        total_h = _RULER_HEIGHT + sum(t.height for t in self._tracks)
        return QSize(600, max(total_h, 100))
