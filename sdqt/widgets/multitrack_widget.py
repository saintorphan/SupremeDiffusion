"""Multi-track timeline widget — composes track headers + scrollable canvas."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from sdqt.widgets.multitrack_canvas import MultiTrackCanvasWidget
from sdqt.widgets.zoom_scrollbar import ZoomScrollBar
from sdqt.widgets.timeline_track import (
    TimelineClip,
    TimelineTrack,
    TrackType,
    ToolMode,
    _CLIP_COLORS,
)
from sdqt.widgets.track_header import TrackHeaderWidget, _HEADER_WIDTH

logger = logging.getLogger(__name__)


class MultiTrackWidget(QWidget):
    """Container widget composing track headers and scrollable canvas.

    Signals:
        playhead_moved(float)
        clip_selected(str, int): (track_id, clip_index)
        clip_context_menu_requested(str, int, QPoint)
        clips_changed()
        track_header_context_menu(str, QPoint): (track_id, global_pos)
    """

    playhead_moved = Signal(float)
    clip_selected = Signal(str, int)
    clip_context_menu_requested = Signal(str, int, QPoint)
    multi_clip_context_menu_requested = Signal(QPoint)
    clips_changed = Signal()
    track_header_context_menu = Signal(str, QPoint)
    zoom_changed = Signal(float)  # pps value

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._tracks: list[TimelineTrack] = []
        self._headers: dict[str, TrackHeaderWidget] = {}
        self._color_idx = 0
        self._video_count = 0
        self._audio_count = 0
        self._selected_track_id: str = ""
        self._ripple_mode: bool = False
        self._ripple_all_tracks: bool = False
        self._auto_scroll_enabled: bool = True
        self._is_playing: bool = False
        self._syncing_zoom: bool = False
        # Optional callback to persist clip files into the project directory.
        # Signature: (path: str) -> str  (returns persisted path)
        self.persist_clip_fn: callable | None = None

        self._build_ui()
        # Start with one default video track
        self.add_track(TrackType.VIDEO)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left: header column (fixed width)
        self._header_container = QWidget()
        self._header_container.setFixedWidth(_HEADER_WIDTH)
        self._header_layout = QVBoxLayout(self._header_container)
        self._header_layout.setContentsMargins(0, 0, 0, 0)
        self._header_layout.setSpacing(0)

        # Ruler spacer (matches canvas ruler)
        ruler_spacer = QWidget()
        ruler_spacer.setFixedHeight(20)
        ruler_spacer.setStyleSheet("background: #222;")
        self._header_layout.addWidget(ruler_spacer)

        # Track headers go here (added dynamically)
        self._headers_inner = QVBoxLayout()
        self._headers_inner.setContentsMargins(0, 0, 0, 0)
        self._headers_inner.setSpacing(0)
        self._header_layout.addLayout(self._headers_inner)

        self._header_container.setAcceptDrops(True)
        self._header_container.dragEnterEvent = self._header_drag_enter
        self._header_container.dragMoveEvent = self._header_drag_move
        self._header_container.dropEvent = self._header_drop

        self._header_layout.addStretch()

        # Header scroll area
        self._header_scroll = QScrollArea()
        self._header_scroll.setWidget(self._header_container)
        self._header_scroll.setWidgetResizable(True)
        self._header_scroll.setFixedWidth(_HEADER_WIDTH + 2)
        self._header_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._header_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._header_scroll.setFrameShape(QScrollArea.NoFrame)
        main_layout.addWidget(self._header_scroll)

        # Right: scrollable canvas
        self._canvas = MultiTrackCanvasWidget()
        self._tracks_scroll = QScrollArea()
        self._tracks_scroll.setWidget(self._canvas)
        self._tracks_scroll.setWidgetResizable(False)
        self._tracks_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._tracks_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._tracks_scroll.setFrameShape(QScrollArea.NoFrame)
        # Tall enough to show 2–3 tracks (90px each) + ruler by default, with a
        # vertical scrollbar as affordance when more tracks are present.
        self._tracks_scroll.setMinimumHeight(220)
        main_layout.addWidget(self._tracks_scroll, 1)

        outer.addLayout(main_layout, 1)

        # Zoom scrollbar (replaces default horizontal scrollbar)
        self._zoom_bar = ZoomScrollBar()
        self._zoom_bar.zoom_changed.connect(self._on_zoom_bar_changed)
        outer.addWidget(self._zoom_bar)

        # Sync vertical scroll
        self._tracks_scroll.verticalScrollBar().valueChanged.connect(
            self._header_scroll.verticalScrollBar().setValue
        )
        # Sync zoom bar when user scrolls with scrollbar/wheel
        self._tracks_scroll.horizontalScrollBar().valueChanged.connect(
            lambda _: self._sync_zoom_bar()
        )

        # Canvas signals
        self._canvas.playhead_moved.connect(self._on_playhead_moved)
        self._canvas.clip_selected.connect(self.clip_selected.emit)
        self._canvas.clip_context_menu_requested.connect(
            self.clip_context_menu_requested.emit
        )
        self._canvas.multi_clip_context_menu_requested.connect(
            self.multi_clip_context_menu_requested.emit
        )
        self._canvas.clip_dropped.connect(self._on_clip_dropped)
        self._canvas.clip_moved.connect(self._on_clip_moved)
        self._canvas.clips_changed.connect(self._sync_zoom_bar)
        self._canvas.clips_changed.connect(self.clips_changed.emit)
        self._canvas.zoom_changed.connect(self._on_canvas_zoom_changed)
        self._canvas.zoom_changed.connect(self.zoom_changed.emit)

        self.setMinimumHeight(220)

    # -- Public API --------------------------------------------------------

    @property
    def tracks(self) -> list[TimelineTrack]:
        return self._tracks

    @property
    def selected_track_id(self) -> str:
        return self._selected_track_id

    @property
    def playhead(self) -> float:
        return self._canvas.playhead

    def set_playhead(self, seconds: float) -> None:
        self._canvas.set_playhead(seconds)
        # Auto-scroll to keep playhead visible (don't re-emit playhead_moved
        # to avoid feedback loops with GL preview seeking)
        self._auto_scroll_to(seconds)

    def _auto_scroll_to(self, seconds: float) -> None:
        """Scroll the canvas to keep the given time position visible."""
        if not self._auto_scroll_enabled:
            return
        ph_px = int(seconds * self._canvas.pps)
        scroll = self._tracks_scroll.horizontalScrollBar()
        viewport_w = self._tracks_scroll.viewport().width()
        visible_left = scroll.value()
        visible_right = visible_left + viewport_w
        if ph_px < visible_left or ph_px > visible_right - 50:
            scroll.setValue(max(0, ph_px - viewport_w // 4))

    def set_tool_mode(self, mode: ToolMode) -> None:
        self._canvas.set_tool_mode(mode)

    # -- Zoom API ----------------------------------------------------------

    def set_zoom(self, pps: float) -> None:
        self._canvas.set_pps(pps)

    def zoom_in(self) -> None:
        self._canvas.set_pps(self._canvas.pps * 1.25)

    def zoom_out(self) -> None:
        self._canvas.set_pps(self._canvas.pps / 1.25)

    def zoom_to_fit(self) -> None:
        viewport_w = self._tracks_scroll.viewport().width()
        self._canvas.zoom_to_fit(viewport_w)

    # -- Ripple mode -------------------------------------------------------

    @property
    def ripple_mode(self) -> bool:
        return self._ripple_mode

    @ripple_mode.setter
    def ripple_mode(self, value: bool) -> None:
        self._ripple_mode = value

    @property
    def ripple_all_tracks(self) -> bool:
        return self._ripple_all_tracks

    @ripple_all_tracks.setter
    def ripple_all_tracks(self, value: bool) -> None:
        self._ripple_all_tracks = value
        self._canvas._ripple_all_tracks = value

    # -- Playback state (for auto-scroll) ----------------------------------

    def set_playing(self, playing: bool) -> None:
        self._is_playing = playing

    @property
    def multi_selection(self) -> list[tuple[str, int]]:
        return self._canvas._multi_selection

    def close_gaps(self, selection: list[tuple[str, int]]) -> None:
        """Butt all selected clips together with no gaps, preserving order.

        Also trims solid-black frames from the head/tail of each clip
        (common artifact from AI video generation).
        """
        if not selection:
            return
        # Group by track
        by_track: dict[str, list[int]] = {}
        for track_id, clip_idx in selection:
            by_track.setdefault(track_id, []).append(clip_idx)

        for track_id, indices in by_track.items():
            track = self._get_track(track_id)
            if not track:
                continue
            valid = [i for i in indices if 0 <= i < len(track.clips)]
            if not valid:
                continue
            valid.sort(key=lambda i: track.clips[i].start_time)

            # Trim black frames from each clip before butting
            for i in valid:
                self._trim_black_frames(track.clips[i])

            # Butt clips together
            cursor = track.clips[valid[0]].start_time
            for i in valid:
                track.clips[i].start_time = cursor
                cursor += track.clips[i].duration

        self._sync_canvas()
        self.clips_changed.emit()

    @staticmethod
    def _trim_black_frames(clip) -> None:
        """Trim solid-black frames from the start and end of a clip (non-destructive).

        Adjusts media_offset and duration so the black frames are hidden.
        """
        import subprocess
        try:
            # Use ffmpeg blackdetect to find black segments
            cmd = [
                "ffmpeg", "-i", clip.path,
                "-vf", f"trim={clip.media_offset}:{clip.media_offset + clip.duration},"
                       "setpts=PTS-STARTPTS,"
                       "blackdetect=d=0.03:pix_th=0.10",
                "-an", "-f", "null", "-",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            stderr = result.stderr

            # Parse blackdetect output lines like:
            # [blackdetect] black_start:0 black_end:0.125 black_duration:0.125
            import re
            blacks = []
            for m in re.finditer(
                r"black_start:([\d.]+)\s+black_end:([\d.]+)\s+black_duration:([\d.]+)",
                stderr,
            ):
                blacks.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))

            if not blacks:
                return

            visible_dur = clip.duration
            trim_head = 0.0
            trim_tail = 0.0

            # Check for black at the very start (within first 0.5s)
            for bs, be, bd in blacks:
                if bs < 0.02:  # starts at beginning
                    trim_head = be
                    break

            # Check for black at the very end
            for bs, be, bd in reversed(blacks):
                if be > visible_dur - 0.02:  # ends at clip end
                    trim_tail = visible_dur - bs
                    break

            if trim_head < 0.01 and trim_tail < 0.01:
                return

            # Apply non-destructive trim
            min_dur = 0.1
            new_offset = clip.media_offset + trim_head
            new_dur = clip.duration - trim_head - trim_tail

            if new_dur < min_dur:
                return  # don't trim if it would make clip too short

            clip.media_offset = new_offset
            clip.duration = new_dur

        except Exception:
            pass  # non-fatal — skip trimming on error

    def total_duration(self) -> float:
        if not self._tracks:
            return 0.0
        return max((t.total_duration() for t in self._tracks), default=0.0)

    def all_clip_paths(self) -> list[str]:
        """Return all clip paths across all tracks (for backward compat)."""
        paths = []
        for t in self._tracks:
            for c in t.clips:
                if c.path not in paths:
                    paths.append(c.path)
        return paths

    def add_track(self, track_type: TrackType, name: str = "") -> TimelineTrack:
        if track_type == TrackType.VIDEO:
            self._video_count += 1
            if not name:
                name = f"Video {self._video_count}"
        else:
            self._audio_count += 1
            if not name:
                name = f"Audio {self._audio_count}"

        track = TimelineTrack(name=name, track_type=track_type)
        self._tracks.append(track)
        self._add_header(track)
        self._sync_canvas()
        self.clips_changed.emit()
        return track

    def remove_track(self, track_id: str) -> None:
        self._tracks = [t for t in self._tracks if t.id != track_id]
        header = self._headers.pop(track_id, None)
        if header:
            self._headers_inner.removeWidget(header)
            header.deleteLater()
        self._sync_canvas()
        self.clips_changed.emit()

    def add_clip_to_track(
        self,
        track_id: str,
        path: str,
        duration: float,
        start_time: float = -1.0,
        name: str = "",
        volume: float = 1.0,
        has_audio: bool = True,
        waveform: list[float] | None = None,
    ) -> None:
        track = self._get_track(track_id)
        if not track:
            return

        # Auto-position at end if not specified
        if start_time < 0:
            start_time = track.total_duration()

        color = _CLIP_COLORS[self._color_idx % len(_CLIP_COLORS)]
        self._color_idx += 1

        clip = TimelineClip(
            path=path,
            duration=duration,
            start_time=start_time,
            name=name or Path(path).stem if path else "",
            color=color,
            volume=volume,
            has_audio=has_audio,
            waveform=waveform,
        )
        track.clips.append(clip)
        self._sync_canvas()
        self.clips_changed.emit()

    def remove_clip_from_track(self, track_id: str, clip_idx: int) -> None:
        track = self._get_track(track_id)
        if track and 0 <= clip_idx < len(track.clips):
            removed = track.clips.pop(clip_idx)
            # Ripple mode: shift subsequent clips left
            if self._ripple_mode:
                track.ripple_shift(removed.start_time, -removed.duration)
                if self._ripple_all_tracks:
                    for t in self._tracks:
                        if t.id != track_id:
                            t.ripple_shift(removed.start_time, -removed.duration)
            self._canvas.clear_selection()
            self._sync_canvas()
            self.clips_changed.emit()

    def clear_all(self) -> None:
        for t in self._tracks:
            t.clips.clear()
        self._canvas.set_playhead(0)
        self._sync_canvas()
        self.clips_changed.emit()

    def reset_to_default(self) -> None:
        """Remove all tracks and headers, then create a single default video track."""
        for h in self._headers.values():
            self._headers_inner.removeWidget(h)
            h.deleteLater()
        self._headers.clear()
        self._tracks.clear()
        self._video_count = 0
        self._audio_count = 0
        self._canvas.set_playhead(0)
        self._sync_canvas()
        self.add_track(TrackType.VIDEO)

    def set_tracks_from_data(self, tracks: list[TimelineTrack]) -> None:
        """Replace all tracks (used during restore)."""
        # Clear headers
        for h in self._headers.values():
            self._headers_inner.removeWidget(h)
            h.deleteLater()
        self._headers.clear()

        self._tracks = tracks
        self._video_count = sum(1 for t in tracks if t.track_type == TrackType.VIDEO)
        self._audio_count = sum(1 for t in tracks if t.track_type == TrackType.AUDIO)

        for t in tracks:
            self._add_header(t)
        self._sync_canvas()

    def get_track(self, track_id: str) -> TimelineTrack | None:
        return self._get_track(track_id)

    def get_first_video_track(self) -> TimelineTrack | None:
        for t in self._tracks:
            if t.track_type == TrackType.VIDEO:
                return t
        return None

    def move_track(self, from_idx: int, to_idx: int) -> None:
        if from_idx == to_idx:
            return
        if 0 <= from_idx < len(self._tracks) and 0 <= to_idx < len(self._tracks):
            track = self._tracks.pop(from_idx)
            self._tracks.insert(to_idx, track)
            self._rebuild_headers()
            self._sync_canvas()
            self.clips_changed.emit()

    # -- Internal ----------------------------------------------------------

    def _get_track(self, track_id: str) -> TimelineTrack | None:
        for t in self._tracks:
            if t.id == track_id:
                return t
        return None

    def _add_header(self, track: TimelineTrack) -> None:
        header = TrackHeaderWidget(
            track.id, track.name, track.track_type, track.height,
        )
        header.volume_changed.connect(self._on_header_volume)
        header.mute_toggled.connect(self._on_header_mute)
        header.visible_toggled.connect(self._on_header_visible)
        header.lock_toggled.connect(self._on_header_lock)
        header.context_menu_requested.connect(self.track_header_context_menu.emit)
        header.clicked.connect(self._on_header_clicked)
        # Sync initial state from track data
        header.set_locked(track.locked)
        header.set_volume(track.volume)
        header.set_muted(track.muted)
        if hasattr(header, "set_visible") and hasattr(track, "visible"):
            header.set_visible(track.visible)
        self._headers[track.id] = header
        self._headers_inner.addWidget(header)

    def _rebuild_headers(self) -> None:
        # Remove all from layout (don't delete)
        for h in self._headers.values():
            self._headers_inner.removeWidget(h)
        # Re-add in track order
        for t in self._tracks:
            if t.id in self._headers:
                self._headers_inner.addWidget(self._headers[t.id])

    def _sync_canvas(self) -> None:
        self._canvas.set_tracks(self._tracks)

    def _on_header_volume(self, track_id: str, volume: float) -> None:
        track = self._get_track(track_id)
        if track:
            track.volume = volume
            self.clips_changed.emit()

    def _on_header_mute(self, track_id: str, muted: bool) -> None:
        track = self._get_track(track_id)
        if track:
            track.muted = muted
            self.clips_changed.emit()

    def _on_header_visible(self, track_id: str, visible: bool) -> None:
        track = self._get_track(track_id)
        if track:
            track.visible = visible
            self.clips_changed.emit()

    def _on_header_lock(self, track_id: str, locked: bool) -> None:
        track = self._get_track(track_id)
        if track:
            track.locked = locked
            self._sync_canvas()

    def _on_playhead_moved(self, seconds: float) -> None:
        """Forward playhead signal and auto-scroll to keep playhead visible."""
        self.playhead_moved.emit(seconds)
        if not self._auto_scroll_enabled:
            return
        ph_px = int(seconds * self._canvas.pps)
        scroll = self._tracks_scroll.horizontalScrollBar()
        viewport_w = self._tracks_scroll.viewport().width()
        visible_left = scroll.value()
        visible_right = visible_left + viewport_w
        if ph_px < visible_left or ph_px > visible_right - 50:
            scroll.setValue(max(0, ph_px - viewport_w // 4))

    def _on_header_clicked(self, track_id: str) -> None:
        self._selected_track_id = track_id
        for tid, header in self._headers.items():
            header.set_selected(tid == track_id)

    # -- Zoom scrollbar sync -----------------------------------------------

    def _sync_zoom_bar(self) -> None:
        """Update the zoom scrollbar to reflect current scroll + zoom state."""
        if self._syncing_zoom:
            return
        self._syncing_zoom = True
        try:
            total_dur = self.total_duration()
            if total_dur <= 0:
                self._zoom_bar.set_range(0.0, 1.0)
                return
            content_w = total_dur * self._canvas.pps + 100
            viewport_w = self._tracks_scroll.viewport().width()
            if content_w <= 0:
                self._zoom_bar.set_range(0.0, 1.0)
                return
            scroll_x = self._tracks_scroll.horizontalScrollBar().value()
            start = scroll_x / content_w
            end = (scroll_x + viewport_w) / content_w
            self._zoom_bar.set_range(
                max(0.0, min(start, 1.0)),
                max(0.0, min(end, 1.0)),
            )
        finally:
            self._syncing_zoom = False

    def _on_canvas_zoom_changed(self, pps: float) -> None:
        """Canvas zoom changed (wheel event) — update the zoom bar."""
        self._sync_zoom_bar()

    def _on_zoom_bar_changed(self, start: float, end: float) -> None:
        """User dragged the zoom scrollbar — update canvas zoom and scroll."""
        if self._syncing_zoom:
            return
        self._syncing_zoom = True
        try:
            total_dur = self.total_duration()
            if total_dur <= 0:
                return

            visible_frac = max(end - start, 0.001)
            viewport_w = self._tracks_scroll.viewport().width()

            # Derive pps from the visible fraction
            visible_dur = total_dur * visible_frac
            if visible_dur > 0:
                new_pps = viewport_w / visible_dur
            else:
                new_pps = self._canvas.pps

            # Clamp to canvas limits
            from sdqt.widgets.multitrack_canvas import _PPS_MIN, _PPS_MAX
            new_pps = max(_PPS_MIN, min(_PPS_MAX, new_pps))

            self._canvas.set_pps(new_pps)

            # Set scroll position
            content_w = total_dur * new_pps + 100
            scroll_x = int(start * content_w)
            self._tracks_scroll.horizontalScrollBar().setValue(max(0, scroll_x))
        finally:
            self._syncing_zoom = False

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync_zoom_bar()

    def _header_drag_enter(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasFormat("application/x-track-header"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def _header_drag_move(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasFormat("application/x-track-header"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def _header_drop(self, event: QDropEvent) -> None:
        if not event.mimeData().hasFormat("application/x-track-header"):
            event.ignore()
            return
        src_id = bytes(event.mimeData().data("application/x-track-header")).decode("utf-8")
        # Find source index
        src_idx = -1
        for i, t in enumerate(self._tracks):
            if t.id == src_id:
                src_idx = i
                break
        if src_idx < 0:
            event.ignore()
            return
        # Calculate target index from drop y position
        # Account for the ruler spacer (20px) at top
        drop_y = int(event.position().y()) - 20
        dst_idx = 0
        cumulative = 0
        for i, t in enumerate(self._tracks):
            mid = cumulative + t.height // 2
            if drop_y > mid:
                dst_idx = i + 1
            cumulative += t.height
        dst_idx = min(dst_idx, len(self._tracks) - 1)
        if src_idx != dst_idx:
            self.move_track(src_idx, dst_idx)
        event.acceptProposedAction()

    def _on_clip_dropped(self, track_id: str, path: str, start_time: float) -> None:
        """Handle a clip dropped onto the canvas."""
        if self.persist_clip_fn:
            path = self.persist_clip_fn(path)
        try:
            from supremediffusion.utils.video import probe_video
            info = probe_video(path)
            dur = info.get("duration", info.get("num_frames", 81) / info.get("fps", 16))
        except Exception:
            # Fallback for audio-only files (probe_video raises on no video stream)
            dur = 5.0
            try:
                import json, subprocess as sp
                result = sp.run(
                    ["ffprobe", "-v", "quiet", "-print_format", "json",
                     "-show_format", path],
                    capture_output=True, text=True, timeout=5,
                )
                data = json.loads(result.stdout)
                d = float(data.get("format", {}).get("duration", 0))
                if d > 0:
                    dur = d
            except Exception:
                pass

        has_audio = True
        audio_exts = {".mp3", ".wav", ".flac", ".ogg", ".aac"}
        is_audio_file = Path(path).suffix.lower() in audio_exts

        # Generate waveform lazily
        waveform = None
        try:
            from sdqt.utils.waveform import extract_waveform
            waveform = extract_waveform(path)
        except Exception:
            pass

        self.add_clip_to_track(
            track_id, path, dur, start_time=start_time,
            has_audio=has_audio, waveform=waveform,
        )

    def _on_clip_moved(
        self, src_track_id: str, src_idx: int, dst_track_id: str, start_time: float,
    ) -> None:
        """Move a clip between tracks or reposition within a track."""
        src_track = self._get_track(src_track_id)
        if not src_track or src_idx < 0 or src_idx >= len(src_track.clips):
            return

        clip = src_track.clips.pop(src_idx)
        clip.start_time = start_time

        dst_track = self._get_track(dst_track_id)
        if dst_track:
            dst_track.clips.append(clip)

        self._canvas.clear_selection()
        self._sync_canvas()
        self.clips_changed.emit()
