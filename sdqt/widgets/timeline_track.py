"""Timeline track data model and legacy single-track widget."""

from __future__ import annotations

import enum
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot, QMimeData, QPoint, QRectF, QSize
from PySide6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QDrag,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QFont,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QPolygon,
    QWheelEvent,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

logger = logging.getLogger(__name__)

MIME_TIMELINE_CLIP = "application/x-timeline-clip"
_CLIP_COLORS = [
    QColor("#4a90d9"), QColor("#d94a4a"), QColor("#4ad97a"),
    QColor("#d9c74a"), QColor("#9b59b6"), QColor("#e67e22"),
    QColor("#1abc9c"), QColor("#e74c3c"),
]
_PIXELS_PER_SECOND = 80
_TRACK_HEIGHT = 90
_CLIP_MARGIN = 2
_CLIP_RADIUS = 4
_PLAYHEAD_HIT = 8  # pixels either side of playhead for grab detection


class ToolMode(enum.IntEnum):
    SELECT = 0
    SCRUB = 1
    RAZOR = 2
    SLIP = 3
    SLIDE = 4


class TrackType(enum.IntEnum):
    VIDEO = 0
    AUDIO = 1


@dataclass
class TimelineMarker:
    """A named marker on the timeline ruler."""
    time: float
    name: str = ""
    color: str = "#ffcc00"  # amber default


@dataclass
class TimelineClip:
    """A single clip on the timeline."""
    path: str
    duration: float  # visible duration on timeline (seconds)
    start_time: float = 0.0  # position on track in seconds
    name: str = ""
    color: QColor = field(default_factory=lambda: QColor("#4a90d9"))
    thumbnail: QPixmap | None = None
    volume: float = 1.0  # per-clip volume 0.0–1.0
    has_audio: bool = True  # clip contains audio stream
    waveform: list[float] | None = None  # cached amplitude peaks
    media_duration: float = 0.0  # full source media length (0 = same as duration)
    media_offset: float = 0.0  # offset into source media for trim-in
    fade_in: float = 0.0  # fade-in duration in seconds
    fade_out: float = 0.0  # fade-out duration in seconds
    group_id: str = ""  # clips with the same group_id move/select together
    transition_in: dict | None = None  # e.g. {"type": "crossfade", "duration": 0.5}
    effects: list = field(default_factory=list)  # ordered effect stack: [{"type": ..., "enabled": ..., "params": {...}}, ...]
    rotation: int = 0  # 0, 90, 180, 270 — metadata-only, no re-encode
    muted: bool = False  # per-clip audio mute
    bake_data: dict | None = None  # stashed original state for unbake
    # Source video color range — True if encoded as limited (TV) range
    # YUV (Y in [16,235]); False if full (PC) range (Y in [0,255]).
    # Set by probe_color_range() when clip is added. Used by the GL preview
    # and effects ffmpeg path to apply TV→PC range expansion so captured
    # frames don't look blue-washed. Default True since most H.264/HEVC video
    # is TV-range; explicit False prevents double-expansion on PC sources.
    is_tv_range: bool = True

    def __post_init__(self):
        if not self.name:
            self.name = Path(self.path).stem
        if self.media_duration <= 0:
            self.media_duration = self.duration

    @property
    def media_end(self) -> float:
        """End offset in source media."""
        return self.media_offset + self.duration

    @property
    def trim_in_available(self) -> float:
        """How much the clip can be extended at the start (seconds)."""
        return self.media_offset

    @property
    def trim_out_available(self) -> float:
        """How much the clip can be extended at the end (seconds)."""
        return self.media_duration - self.media_end


@dataclass
class TimelineTrack:
    """A single track (video or audio) containing positioned clips."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = "Video 1"
    track_type: TrackType = TrackType.VIDEO
    clips: list[TimelineClip] = field(default_factory=list)
    volume: float = 1.0
    muted: bool = False
    visible: bool = True  # video tracks only
    height: int = 90  # pixel height
    locked: bool = False  # locked tracks reject all edits

    def total_duration(self) -> float:
        """End time of the last clip on this track."""
        if not self.clips:
            return 0.0
        return max(c.start_time + c.duration for c in self.clips)

    def sorted_clips(self) -> list[TimelineClip]:
        """Return clips sorted by start_time."""
        return sorted(self.clips, key=lambda c: c.start_time)

    def find_gaps(self) -> list[tuple[float, float]]:
        """Return list of (gap_start, gap_end) gaps on this track."""
        gaps = []
        sc = self.sorted_clips()
        for i in range(len(sc) - 1):
            end_i = sc[i].start_time + sc[i].duration
            start_next = sc[i + 1].start_time
            if start_next > end_i + 0.001:
                gaps.append((end_i, start_next))
        return gaps

    def clip_before(self, clip_idx: int) -> TimelineClip | None:
        """Return the clip immediately before this one (by start_time)."""
        if clip_idx < 0 or clip_idx >= len(self.clips):
            return None
        target = self.clips[clip_idx]
        sc = self.sorted_clips()
        for i, c in enumerate(sc):
            if c is target and i > 0:
                return sc[i - 1]
        return None

    def clip_after(self, clip_idx: int) -> TimelineClip | None:
        """Return the clip immediately after this one (by start_time)."""
        if clip_idx < 0 or clip_idx >= len(self.clips):
            return None
        target = self.clips[clip_idx]
        sc = self.sorted_clips()
        for i, c in enumerate(sc):
            if c is target and i < len(sc) - 1:
                return sc[i + 1]
        return None

    def ripple_shift(self, from_time: float, delta: float) -> None:
        """Shift all clips starting >= from_time by delta seconds."""
        for c in self.clips:
            if c.start_time >= from_time - 0.001:
                c.start_time = max(0.0, c.start_time + delta)

    def close_all_gaps(self) -> None:
        """Butt all clips together with no gaps."""
        sc = self.sorted_clips()
        if not sc:
            return
        cursor = sc[0].start_time
        for c in sc:
            c.start_time = cursor
            cursor += c.duration

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        clips_data = []
        for c in self.clips:
            cd: dict = {
                "path": c.path,
                "start_time": c.start_time,
                "duration": c.duration,
                "name": c.name,
                "volume": c.volume,
                "media_duration": c.media_duration,
                "media_offset": c.media_offset,
            }
            if c.fade_in > 0:
                cd["fade_in"] = c.fade_in
            if c.fade_out > 0:
                cd["fade_out"] = c.fade_out
            if c.group_id:
                cd["group_id"] = c.group_id
            if c.transition_in:
                cd["transition_in"] = c.transition_in
            if c.effects:
                cd["effects"] = c.effects
            if c.rotation:
                cd["rotation"] = c.rotation
            if c.muted:
                cd["muted"] = c.muted
            if not c.has_audio:
                cd["has_audio"] = False
            if c.bake_data:
                cd["bake_data"] = c.bake_data
            if not c.is_tv_range:
                # Only serialize when False — saves space; default is True.
                cd["is_tv_range"] = False
            clips_data.append(cd)
        return {
            "id": self.id,
            "name": self.name,
            "track_type": int(self.track_type),
            "volume": self.volume,
            "muted": self.muted,
            "visible": self.visible,
            "locked": self.locked,
            "clips": clips_data,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TimelineTrack:
        """Deserialize from persistence."""
        from sdqt.models.effects import migrate_legacy_effects

        clips = []
        for cd in data.get("clips", []):
            dur = cd["duration"]
            clips.append(TimelineClip(
                path=cd["path"],
                duration=dur,
                start_time=cd.get("start_time", 0.0),
                name=cd.get("name", ""),
                volume=cd.get("volume", 1.0),
                media_duration=cd.get("media_duration", dur),
                media_offset=cd.get("media_offset", 0.0),
                fade_in=cd.get("fade_in", 0.0),
                fade_out=cd.get("fade_out", 0.0),
                group_id=cd.get("group_id", ""),
                transition_in=cd.get("transition_in", None),
                effects=migrate_legacy_effects(cd.get("effects", [])),
                rotation=cd.get("rotation", 0),
                muted=cd.get("muted", False),
                has_audio=cd.get("has_audio", True),
                bake_data=cd.get("bake_data", None),
                is_tv_range=cd.get("is_tv_range", True),
            ))
        return cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            name=data.get("name", "Track"),
            track_type=TrackType(data.get("track_type", 0)),
            clips=clips,
            volume=data.get("volume", 1.0),
            muted=data.get("muted", False),
            visible=data.get("visible", True),
            locked=data.get("locked", False),
        )


class TimelineTrackWidget(QWidget):
    """Custom-painted horizontal timeline track.

    Signals:
        playhead_moved(float): playhead position in seconds
        clips_changed(): emitted when clips are added, removed, or reordered
        clip_selected(int): index of selected clip (-1 for none)
        clip_context_menu_requested(int, QPoint): right-click on clip
        splice_requested(float): playhead seconds when splice is requested
    """

    playhead_moved = Signal(float)
    clips_changed = Signal()
    clip_selected = Signal(int)
    clip_context_menu_requested = Signal(int, QPoint)  # (clip_index, global_pos)
    splice_requested = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumHeight(_TRACK_HEIGHT)
        self.setFixedHeight(_TRACK_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._clips: list[TimelineClip] = []
        self._playhead: float = 0.0  # seconds
        self._selected: int = -1
        self._drag_idx: int = -1
        self._drag_start_pos: QPoint | None = None
        self._color_idx = 0
        self._scroll_offset: int = 0  # horizontal pixel offset
        self._tool: ToolMode = ToolMode.SELECT
        self._dragging_playhead: bool = False
        self._scrubbing: bool = False

    # -- Public API --------------------------------------------------------

    @property
    def clips(self) -> list[TimelineClip]:
        return self._clips

    @property
    def total_duration(self) -> float:
        return sum(c.duration for c in self._clips)

    @property
    def playhead(self) -> float:
        return self._playhead

    @property
    def tool_mode(self) -> ToolMode:
        return self._tool

    def set_tool_mode(self, mode: ToolMode) -> None:
        self._tool = mode
        if mode == ToolMode.SCRUB:
            self.setCursor(Qt.IBeamCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def set_playhead(self, seconds: float) -> None:
        self._playhead = max(0.0, min(seconds, self.total_duration))
        self.update()

    def add_clip(self, path: str, duration: float, name: str = "",
                 thumbnail: QPixmap | None = None, index: int = -1) -> None:
        color = _CLIP_COLORS[self._color_idx % len(_CLIP_COLORS)]
        self._color_idx += 1
        # Probe TV/PC color range so the GL preview + effects ffmpeg can apply
        # the correct range expansion. Mislabeled or unprobed defaults to TV.
        is_tv = True
        try:
            from supremediffusion.utils.video import probe_color_range
            is_tv = (probe_color_range(path) == "tv")
        except Exception:  # noqa: BLE001
            pass
        clip = TimelineClip(path=path, duration=duration, name=name,
                            color=color, thumbnail=thumbnail,
                            is_tv_range=is_tv)
        if index < 0 or index >= len(self._clips):
            self._clips.append(clip)
        else:
            self._clips.insert(index, clip)
        self._update_minimum_width()
        self.clips_changed.emit()
        self.update()

    def remove_clip(self, index: int) -> None:
        if 0 <= index < len(self._clips):
            self._clips.pop(index)
            if self._selected >= len(self._clips):
                self._selected = len(self._clips) - 1
            self._update_minimum_width()
            self.clips_changed.emit()
            self.update()

    def clear_clips(self) -> None:
        self._clips.clear()
        self._selected = -1
        self._playhead = 0.0
        self._update_minimum_width()
        self.clips_changed.emit()
        self.update()

    def move_clip(self, from_idx: int, to_idx: int) -> None:
        if from_idx == to_idx:
            return
        if 0 <= from_idx < len(self._clips) and 0 <= to_idx <= len(self._clips):
            clip = self._clips.pop(from_idx)
            if to_idx > from_idx:
                to_idx -= 1
            self._clips.insert(to_idx, clip)
            self._selected = to_idx
            self.clips_changed.emit()
            self.update()

    def clip_paths(self) -> list[str]:
        return [c.path for c in self._clips]

    # -- Internal helpers --------------------------------------------------

    def _update_minimum_width(self) -> None:
        total = self.total_duration
        min_w = max(int(total * _PIXELS_PER_SECOND) + 40, 200)
        self.setMinimumWidth(min_w)

    def _clip_rect(self, index: int) -> QRectF:
        """Get the rectangle for clip at index."""
        x = 0.0
        for i, c in enumerate(self._clips):
            w = c.duration * _PIXELS_PER_SECOND
            if i == index:
                return QRectF(x + _CLIP_MARGIN, _CLIP_MARGIN,
                              w - 2 * _CLIP_MARGIN,
                              _TRACK_HEIGHT - 2 * _CLIP_MARGIN)
            x += w
        return QRectF()

    def _pos_to_seconds(self, x: int) -> float:
        return max(0.0, x / _PIXELS_PER_SECOND)

    def _clip_at_pos(self, x: int) -> int:
        """Return clip index at pixel x, or -1."""
        px = 0.0
        for i, c in enumerate(self._clips):
            w = c.duration * _PIXELS_PER_SECOND
            if px <= x < px + w:
                return i
            px += w
        return -1

    def _insert_index_at_pos(self, x: int) -> int:
        """Return insertion index for a drop at pixel x."""
        px = 0.0
        for i, c in enumerate(self._clips):
            w = c.duration * _PIXELS_PER_SECOND
            if x < px + w / 2:
                return i
            px += w
        return len(self._clips)

    def _is_on_playhead(self, x: int, y: int) -> bool:
        """Check if (x, y) is on the playhead triangle / top area."""
        ph_x = int(self._playhead * _PIXELS_PER_SECOND)
        return abs(x - ph_x) <= _PLAYHEAD_HIT and y <= 14

    # -- Paint -------------------------------------------------------------

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        p.fillRect(0, 0, w, h, QColor("#1e1e1e"))

        # Time markers
        p.setPen(QPen(QColor("#444"), 1))
        marker_font = QFont("sans-serif", 9)
        p.setFont(marker_font)
        total = max(self.total_duration, w / _PIXELS_PER_SECOND)
        # Skip labels when seconds are tightly packed so they don't overlap at
        # low zoom (each label needs ~28px of horizontal room).
        label_step = max(1, int(round(28 / _PIXELS_PER_SECOND)))
        sec = 0
        while sec <= total:
            x = int(sec * _PIXELS_PER_SECOND)
            if x > w:
                break
            p.drawLine(x, 0, x, h)
            if sec % label_step == 0:
                p.drawText(x + 2, 12, f"{sec}s")
            sec += 1

        # Clips
        clip_font = QFont("sans-serif", 11)
        clip_font.setBold(True)
        p.setFont(clip_font)
        x_pos = 0.0
        for i, clip in enumerate(self._clips):
            cw = clip.duration * _PIXELS_PER_SECOND
            rect = QRectF(x_pos + _CLIP_MARGIN, _CLIP_MARGIN,
                          cw - 2 * _CLIP_MARGIN, h - 2 * _CLIP_MARGIN)

            # Clip body
            base_color = QColor(clip.color)
            if i == self._selected:
                base_color = base_color.lighter(130)
            p.setBrush(QBrush(base_color))
            p.setPen(QPen(base_color.darker(150), 1))
            p.drawRoundedRect(rect, _CLIP_RADIUS, _CLIP_RADIUS)

            # Thumbnail
            if clip.thumbnail and not clip.thumbnail.isNull():
                th_rect = QRectF(rect.x() + 2, rect.y() + 2,
                                 min(rect.width() - 4, 60),
                                 rect.height() - 4)
                p.drawPixmap(th_rect.toRect(), clip.thumbnail)
                text_x = th_rect.right() + 4
            else:
                text_x = rect.x() + 4

            # Label
            p.setPen(QColor("#fff"))
            text_rect = QRectF(text_x, rect.y() + 4,
                               rect.right() - text_x - 4, rect.height() - 8)
            label = clip.name
            dur_str = f"{clip.duration:.1f}s"
            p.drawText(text_rect, Qt.AlignLeft | Qt.AlignTop, label)
            p.setFont(QFont("sans-serif", 9))
            p.drawText(text_rect, Qt.AlignLeft | Qt.AlignBottom, dur_str)
            p.setFont(clip_font)

            x_pos += cw

        # Selection highlight border
        if 0 <= self._selected < len(self._clips):
            sel_rect = self._clip_rect(self._selected)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor("#fff"), 2))
            p.drawRoundedRect(sel_rect, _CLIP_RADIUS, _CLIP_RADIUS)

        # Playhead
        ph_x = int(self._playhead * _PIXELS_PER_SECOND)
        if 0 <= ph_x <= w:
            p.setPen(QPen(QColor("#ff3333"), 2))
            p.drawLine(ph_x, 0, ph_x, h)
            # Triangle head (larger for easier grabbing)
            p.setBrush(QColor("#ff3333"))
            p.setPen(Qt.NoPen)
            p.drawPolygon(QPolygon([
                QPoint(ph_x - 7, 0), QPoint(ph_x + 7, 0), QPoint(ph_x, 9),
            ]))

        p.end()

    # -- Mouse events ------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        x = int(event.position().x())
        y = int(event.position().y())

        # Right-click: always context menu on clips
        if event.button() == Qt.RightButton:
            idx = self._clip_at_pos(x)
            if idx >= 0:
                self._selected = idx
                self.clip_selected.emit(idx)
                self.update()
                self.clip_context_menu_requested.emit(
                    idx, self.mapToGlobal(event.position().toPoint())
                )
            return

        if event.button() != Qt.LeftButton:
            return

        # Playhead grab — always available regardless of tool
        if self._is_on_playhead(x, y):
            self._dragging_playhead = True
            return

        if self._tool == ToolMode.SCRUB:
            # Scrub: move playhead to click position, start scrubbing
            self._scrubbing = True
            self._playhead = self._pos_to_seconds(x)
            self.playhead_moved.emit(self._playhead)
            self.update()
        else:
            # Select: click on clip to select/drag, click empty to deselect
            idx = self._clip_at_pos(x)
            if idx >= 0:
                self._selected = idx
                self._drag_idx = idx
                self._drag_start_pos = event.position().toPoint()
                self.clip_selected.emit(idx)
            else:
                self._selected = -1
                self.clip_selected.emit(-1)
            self.update()

    _tooltip_cache: dict = {}

    def event(self, event) -> bool:
        if event.type() == event.Type.ToolTip:
            x = int(event.pos().x())
            idx = self._clip_at_pos(x)
            clip = self._clips[idx] if 0 <= idx < len(self._clips) else None
            if clip:
                tip = self._tooltip_cache.get(clip.path)
                if tip is None:
                    tip = self._build_clip_tooltip(clip)
                    self._tooltip_cache[clip.path] = tip
                from PySide6.QtWidgets import QToolTip
                QToolTip.showText(event.globalPos(), tip, self)
                return True
            else:
                from PySide6.QtWidgets import QToolTip
                QToolTip.hideText()
                return True
        return super().event(event)

    @staticmethod
    def _build_clip_tooltip(clip) -> str:
        import subprocess
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,r_frame_rate,codec_name,pix_fmt,color_range",
                 "-show_entries", "format=duration",
                 "-of", "csv=p=0", clip.path],
                capture_output=True, text=True, timeout=3,
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
                    return f"{clip.name}\n{w}x{h} | {codec} {pix} {cr}\n{fps} fps | {dur_s}"
        except Exception:
            pass
        return clip.name

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        x = int(event.position().x())

        # Playhead drag
        if self._dragging_playhead:
            self._playhead = self._pos_to_seconds(x)
            self._playhead = max(0.0, min(self._playhead, self.total_duration))
            self.playhead_moved.emit(self._playhead)
            self.update()
            return

        # Scrub drag
        if self._scrubbing:
            self._playhead = self._pos_to_seconds(x)
            self._playhead = max(0.0, min(self._playhead, self.total_duration))
            self.playhead_moved.emit(self._playhead)
            self.update()
            return

        # Select tool: clip drag for rearrange
        if (
            self._tool == ToolMode.SELECT
            and self._drag_idx >= 0
            and self._drag_start_pos is not None
            and (event.position().toPoint() - self._drag_start_pos).manhattanLength() > 10
        ):
            drag = QDrag(self)
            mime = QMimeData()
            mime.setData(MIME_TIMELINE_CLIP, str(self._drag_idx).encode())
            drag.setMimeData(mime)
            drag.exec(Qt.MoveAction)
            self._drag_idx = -1
            self._drag_start_pos = None

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging_playhead = False
        self._scrubbing = False
        self._drag_idx = -1
        self._drag_start_pos = None

    # -- Keyboard ----------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Left:
            self._playhead = max(0, self._playhead - 1.0 / 16)
            self.playhead_moved.emit(self._playhead)
            self.update()
        elif event.key() == Qt.Key_Right:
            self._playhead = min(self.total_duration, self._playhead + 1.0 / 16)
            self.playhead_moved.emit(self._playhead)
            self.update()
        elif event.key() == Qt.Key_Delete and 0 <= self._selected < len(self._clips):
            self.remove_clip(self._selected)
        else:
            super().keyPressEvent(event)

    # -- Drag & Drop -------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasFormat(MIME_TIMELINE_CLIP):
            event.acceptProposedAction()
        elif event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        x = int(event.position().x())
        insert_idx = self._insert_index_at_pos(x)

        if event.mimeData().hasFormat(MIME_TIMELINE_CLIP):
            data = bytes(event.mimeData().data(MIME_TIMELINE_CLIP)).decode()
            # Internal rearrange (data is source index)
            try:
                src_idx = int(data)
                if 0 <= src_idx < len(self._clips):
                    self.move_clip(src_idx, insert_idx)
                    event.acceptProposedAction()
                    return
            except ValueError:
                pass
            # External drop from library (data is path)
            if Path(data).is_file():
                self._drop_clip_path(data, insert_idx)
                event.acceptProposedAction()
                return

        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    if Path(path).suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                        self._drop_clip_path(path, insert_idx)
            event.acceptProposedAction()
            return

        event.ignore()

    def _drop_clip_path(self, path: str, index: int) -> None:
        """Add a clip from a dropped path. Duration probed on-demand."""
        try:
            from supremediffusion.utils.video import probe_video
            info = probe_video(path)
            dur = info.get("duration", info.get("num_frames", 81) / info.get("fps", 16))
        except Exception:
            dur = 5.0
        self.add_clip(path, dur, index=index)

    # -- Size hint ---------------------------------------------------------

    def sizeHint(self) -> QSize:
        return QSize(600, _TRACK_HEIGHT)
