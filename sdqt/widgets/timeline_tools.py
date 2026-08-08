"""Timeline tool classes for the multi-track NLE canvas.

Implements the Tool System Refactor with a base class and concrete tools:
SelectTool (with ghost drag, roll edit, fade handles), ScrubTool, RazorTool,
SlipTool, and SlideTool.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QMouseEvent

if TYPE_CHECKING:
    from sdqt.widgets.multitrack_canvas import MultiTrackCanvasWidget

from sdqt.widgets.timeline_track import TimelineClip, TimelineTrack, TrackType, ToolMode

logger = logging.getLogger(__name__)

_MIN_CLIP_DURATION = 0.1
_EDGE_HIT_PX = 6
_FADE_HANDLE_PX = 8
_DRAG_THRESHOLD = 6


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class TimelineTool(ABC):
    """Abstract base for all timeline interaction tools."""

    def __init__(self, canvas: MultiTrackCanvasWidget) -> None:
        self.canvas = canvas

    @abstractmethod
    def on_press(self, x: int, y: int, event: QMouseEvent) -> bool:
        """Handle mouse press. Return True if the event was consumed."""
        ...

    @abstractmethod
    def on_move(self, x: int, y: int, event: QMouseEvent) -> None:
        """Handle mouse move (button held or tracking)."""
        ...

    @abstractmethod
    def on_release(self, x: int, y: int, event: QMouseEvent) -> None:
        """Handle mouse release."""
        ...

    def on_hover(self, x: int, y: int) -> Qt.CursorShape | None:
        """Return a cursor override for hover, or None to keep the tool default."""
        return None

    def cursor(self) -> Qt.CursorShape:
        """Default cursor for this tool."""
        return Qt.ArrowCursor

    # -- Helpers available to all tools ------------------------------------

    def _pps(self) -> float:
        return getattr(self.canvas, "_pps", 80.0)

    def _pos_to_seconds(self, x: int) -> float:
        return max(0.0, x / self._pps())

    def _find_track(self, track_id: str) -> TimelineTrack | None:
        return self.canvas._track_by_id.get(track_id)

    def _track_locked(self, track: TimelineTrack) -> bool:
        return getattr(track, "locked", False)

    def _sorted_clip_indices(self, track: TimelineTrack) -> list[int]:
        """Return clip indices sorted by start_time."""
        return sorted(range(len(track.clips)), key=lambda i: track.clips[i].start_time)

    def _neighbor_clips(
        self, track: TimelineTrack, clip_idx: int
    ) -> tuple[TimelineClip | None, int, TimelineClip | None, int]:
        """Return (clip_before, idx_before, clip_after, idx_after) sorted by start_time."""
        if clip_idx < 0 or clip_idx >= len(track.clips):
            return None, -1, None, -1
        clip = track.clips[clip_idx]
        sorted_idxs = self._sorted_clip_indices(track)
        pos = sorted_idxs.index(clip_idx)

        before_clip: TimelineClip | None = None
        before_idx = -1
        after_clip: TimelineClip | None = None
        after_idx = -1

        if pos > 0:
            before_idx = sorted_idxs[pos - 1]
            before_clip = track.clips[before_idx]
        if pos < len(sorted_idxs) - 1:
            after_idx = sorted_idxs[pos + 1]
            after_clip = track.clips[after_idx]

        return before_clip, before_idx, after_clip, after_idx


# ---------------------------------------------------------------------------
# SelectTool
# ---------------------------------------------------------------------------


class SelectTool(TimelineTool):
    """Selection, drag, trim, roll-edit, and fade-handle interactions."""

    def __init__(self, canvas: MultiTrackCanvasWidget) -> None:
        super().__init__(canvas)
        # Drag state
        self._dragging_clip: bool = False
        self._drag_clip_track: str = ""
        self._drag_clip_idx: int = -1
        self._drag_start_pos: tuple[int, int] | None = None
        self._drag_grab_offset: float = 0.0
        self._drag_orig_start: float = 0.0
        self._drag_orig_track_id: str = ""

        # Trim state
        self._trimming: bool = False
        self._trim_edge: str = ""  # "left" | "right"
        self._trim_track_id: str = ""
        self._trim_clip_idx: int = -1
        self._trim_start_x: int = 0
        self._trim_orig_start_time: float = 0.0
        self._trim_orig_duration: float = 0.0
        self._trim_orig_media_offset: float = 0.0

        # Roll edit (Ctrl + trim)
        self._roll_trim: bool = False
        self._roll_neighbor_idx: int = -1
        self._roll_neighbor_orig_start: float = 0.0
        self._roll_neighbor_orig_duration: float = 0.0
        self._roll_neighbor_orig_media_offset: float = 0.0

        # Fade handle drag
        self._fading: bool = False
        self._fade_side: str = ""  # "in" | "out"
        self._fade_track_id: str = ""
        self._fade_clip_idx: int = -1
        self._fade_start_x: int = 0
        self._fade_orig_value: float = 0.0

        # Rubber-band (marquee) selection
        self._rubber_band: bool = False
        self._rubber_start: tuple[int, int] = (0, 0)
        self._rubber_end: tuple[int, int] = (0, 0)

    def cursor(self) -> Qt.CursorShape:
        return Qt.ArrowCursor

    # -- Press -------------------------------------------------------------

    def on_press(self, x: int, y: int, event: QMouseEvent) -> bool:
        c = self.canvas

        # Check fade handles first (highest priority, smallest target)
        fade_info = self._fade_handle_at(x, y)
        if fade_info is not None:
            track, idx, side = fade_info
            clip = track.clips[idx]
            self._fading = True
            self._fade_side = side
            self._fade_track_id = track.id
            self._fade_clip_idx = idx
            self._fade_start_x = x
            self._fade_orig_value = getattr(clip, "fade_in", 0.0) if side == "in" else getattr(clip, "fade_out", 0.0)
            c._selected_track_id = track.id
            c._selected_clip_idx = idx
            c.clip_selected.emit(track.id, idx)
            c.update()
            return True

        # Edge trim detection
        edge_track, edge_idx, edge_side = c._edge_at_pos(x, y)
        if edge_track and edge_idx >= 0 and edge_side:
            if self._track_locked(edge_track):
                return False
            clip = edge_track.clips[edge_idx]
            self._trimming = True
            self._trim_edge = edge_side
            self._trim_track_id = edge_track.id
            self._trim_clip_idx = edge_idx
            self._trim_start_x = x
            self._trim_orig_start_time = clip.start_time
            self._trim_orig_duration = clip.duration
            self._trim_orig_media_offset = clip.media_offset

            # Roll edit: Ctrl held
            ctrl = bool(event.modifiers() & Qt.ControlModifier)
            self._roll_trim = ctrl
            if ctrl:
                self._setup_roll_neighbor(edge_track, edge_idx, edge_side)

            c._selected_track_id = edge_track.id
            c._selected_clip_idx = edge_idx
            c.clip_selected.emit(edge_track.id, edge_idx)
            c.update()
            return True

        # Clip body click
        track = c._track_at_y(y)
        ctrl = bool(event.modifiers() & Qt.ControlModifier)
        shift = bool(event.modifiers() & Qt.ShiftModifier)
        if track:
            idx = c._clip_at_pos(x, track)
            if idx >= 0:
                clip = track.clips[idx]
                entry = (track.id, idx)
                if shift and c._selected_track_id == track.id and c._selected_clip_idx >= 0:
                    # Shift-click: range select on same track
                    anchor = c._selected_clip_idx
                    lo = min(anchor, idx)
                    hi = max(anchor, idx)
                    c._multi_selection.clear()
                    for j in range(lo, hi + 1):
                        c._multi_selection.append((track.id, j))
                    c._selected_clip_idx = idx
                    c.clip_selected.emit(track.id, idx)
                    c.update()
                elif ctrl:
                    # Toggle multi-selection
                    if entry in c._multi_selection:
                        c._multi_selection.remove(entry)
                    else:
                        if (
                            not c._multi_selection
                            and c._selected_track_id
                            and c._selected_clip_idx >= 0
                        ):
                            prev = (c._selected_track_id, c._selected_clip_idx)
                            if prev not in c._multi_selection:
                                c._multi_selection.append(prev)
                        c._multi_selection.append(entry)
                    c._selected_track_id = track.id
                    c._selected_clip_idx = idx
                    c.update()
                else:
                    # Single select + prepare drag
                    c._multi_selection.clear()
                    c._selected_track_id = track.id
                    c._selected_clip_idx = idx
                    self._drag_clip_track = track.id
                    self._drag_clip_idx = idx
                    self._drag_start_pos = (x, y)
                    self._drag_grab_offset = self._pos_to_seconds(x) - clip.start_time
                    self._drag_orig_start = clip.start_time
                    self._drag_orig_track_id = track.id
                    c.clip_selected.emit(track.id, idx)
                c.update()
                return True
            else:
                # Empty space on a track — start rubber-band
                c.clear_selection()
                c.clip_selected.emit("", -1)
                self._rubber_band = True
                self._rubber_start = (x, y)
                self._rubber_end = (x, y)
                c.update()
                return True
        else:
            # Empty space below tracks — start rubber-band
            c.clear_selection()
            c.clip_selected.emit("", -1)
            self._rubber_band = True
            self._rubber_start = (x, y)
            self._rubber_end = (x, y)
            c.update()
            return True

    # -- Move --------------------------------------------------------------

    def on_move(self, x: int, y: int, event: QMouseEvent) -> None:
        c = self.canvas

        # Rubber-band marquee drag
        if self._rubber_band:
            self._rubber_end = (x, y)
            c._rubber_band_rect = self._get_rubber_rect()
            c.update()
            return

        # Active fade drag
        if self._fading:
            self._do_fade_drag(x)
            return

        # Active trim drag
        if self._trimming:
            self._do_trim_drag(x)
            return

        # Active clip drag (ghost)
        if self._dragging_clip:
            self._do_ghost_drag(x, y)
            return

        # Start drag after threshold
        if (
            self._drag_clip_idx >= 0
            and self._drag_start_pos is not None
        ):
            dx = abs(x - self._drag_start_pos[0])
            dy = abs(y - self._drag_start_pos[1])
            if dx + dy > _DRAG_THRESHOLD:
                self._dragging_clip = True
                c.setCursor(Qt.ClosedHandCursor)
                # Snapshot for ghost — do NOT mutate clip yet
                return

    def on_hover(self, x: int, y: int) -> Qt.CursorShape | None:
        c = self.canvas
        # Fade handle check
        if self._fade_handle_at(x, y) is not None:
            return Qt.PointingHandCursor
        # Edge check
        _, _, edge_side = c._edge_at_pos(x, y)
        if edge_side:
            return Qt.SplitHCursor
        return None

    # -- Release -----------------------------------------------------------

    def on_release(self, x: int, y: int, event: QMouseEvent) -> None:
        c = self.canvas

        # Finish rubber-band: select all clips inside the rectangle
        if self._rubber_band:
            self._rubber_end = (x, y)
            rect = self._get_rubber_rect()
            self._rubber_band = False
            c._rubber_band_rect = None
            self._select_clips_in_rect(rect)
            c.update()
            return

        if self._fading:
            self._fading = False
            self._fade_side = ""
            c.clips_changed.emit()

        if self._trimming:
            self._trimming = False
            self._trim_edge = ""
            self._trim_track_id = ""
            self._trim_clip_idx = -1
            self._roll_trim = False
            self._roll_neighbor_idx = -1
            c._snap_line = -1.0
            c.clips_changed.emit()
            c._update_size()

        if self._dragging_clip:
            # Commit ghost position to clip
            self._commit_ghost(x, y)
            self._dragging_clip = False
            c._ghost = None
            c._snap_line = -1.0
            c.unsetCursor()
            c.clips_changed.emit()
            c._update_size()

        self._drag_clip_idx = -1
        self._drag_clip_track = ""
        self._drag_start_pos = None

    # -- Escape cancel -----------------------------------------------------

    def cancel(self) -> None:
        """Cancel in-progress drag or trim, restoring original state."""
        c = self.canvas
        if self._dragging_clip:
            # Restore original position
            track = self._find_track(self._drag_orig_track_id)
            if track and self._drag_clip_idx >= 0 and self._drag_clip_idx < len(track.clips):
                track.clips[self._drag_clip_idx].start_time = self._drag_orig_start
            # If clip was moved to another track, move it back
            if self._drag_clip_track != self._drag_orig_track_id:
                cur_track = self._find_track(self._drag_clip_track)
                orig_track = self._find_track(self._drag_orig_track_id)
                if cur_track and orig_track and self._drag_clip_idx < len(cur_track.clips):
                    clip = cur_track.clips.pop(self._drag_clip_idx)
                    clip.start_time = self._drag_orig_start
                    orig_track.clips.append(clip)
            self._dragging_clip = False
            c._ghost = None
            c._snap_line = -1.0
            c.unsetCursor()

        if self._trimming:
            track = self._find_track(self._trim_track_id)
            if track and 0 <= self._trim_clip_idx < len(track.clips):
                clip = track.clips[self._trim_clip_idx]
                clip.start_time = self._trim_orig_start_time
                clip.duration = self._trim_orig_duration
                clip.media_offset = self._trim_orig_media_offset
            # Roll neighbor restore
            if self._roll_trim and self._roll_neighbor_idx >= 0 and track:
                if self._roll_neighbor_idx < len(track.clips):
                    nb = track.clips[self._roll_neighbor_idx]
                    nb.start_time = self._roll_neighbor_orig_start
                    nb.duration = self._roll_neighbor_orig_duration
                    nb.media_offset = self._roll_neighbor_orig_media_offset
            self._trimming = False
            self._trim_edge = ""
            self._roll_trim = False
            c._snap_line = -1.0

        self._drag_clip_idx = -1
        self._drag_clip_track = ""
        self._drag_start_pos = None
        self._fading = False
        c.update()

    # -- Rubber-band (marquee) selection -----------------------------------

    def _get_rubber_rect(self) -> tuple[int, int, int, int]:
        """Return normalized (x, y, w, h) from rubber-band start/end."""
        x1, y1 = self._rubber_start
        x2, y2 = self._rubber_end
        left = min(x1, x2)
        top = min(y1, y2)
        return (left, top, abs(x2 - x1), abs(y2 - y1))

    def _select_clips_in_rect(self, rect: tuple[int, int, int, int]) -> None:
        """Find all clips overlapping the rubber-band rectangle and multi-select them."""
        c = self.canvas
        rx, ry, rw, rh = rect
        if rw < 4 and rh < 4:
            return  # too small, was just a click

        c._multi_selection.clear()
        from sdqt.widgets.multitrack_canvas import _RULER_HEIGHT

        for track in c._tracks:
            ty = c._track_y_offset(track.id)
            track_bottom = ty + track.height
            # Check vertical overlap
            if ry + rh < ty or ry > track_bottom:
                continue
            for i, clip in enumerate(track.clips):
                clip_x = int(clip.start_time * c._pps)
                clip_w = int(clip.duration * c._pps)
                # Check horizontal overlap
                if clip_x + clip_w < rx or clip_x > rx + rw:
                    continue
                c._multi_selection.append((track.id, i))

        if c._multi_selection:
            # Set the first selected as the primary selection
            first = c._multi_selection[0]
            c._selected_track_id = first[0]
            c._selected_clip_idx = first[1]
            c.clip_selected.emit(first[0], first[1])

    # -- Ghost drag --------------------------------------------------------

    def _do_ghost_drag(self, x: int, y: int) -> None:
        """Compute ghost position without mutating the clip until release."""
        c = self.canvas
        track = self._find_track(self._drag_clip_track)
        if not track or self._drag_clip_idx < 0 or self._drag_clip_idx >= len(track.clips):
            self._dragging_clip = False
            c._ghost = None
            return

        clip = track.clips[self._drag_clip_idx]
        pps = self._pps()
        raw_start = self._pos_to_seconds(x) - self._drag_grab_offset

        # Snap edges
        snap_edges = c._collect_snap_edges(self._drag_clip_track, self._drag_clip_idx)
        snapped_left, left_hit = c._snap_edge(raw_start, snap_edges)
        raw_right = raw_start + clip.duration
        snapped_right, right_hit = c._snap_edge(raw_right, snap_edges)

        left_delta = abs(snapped_left - raw_start) if left_hit else float("inf")
        right_delta = abs(snapped_right - raw_right) if right_hit else float("inf")

        c._snap_line = -1.0
        if left_hit and left_delta <= right_delta:
            new_start = snapped_left
            c._snap_line = snapped_left
        elif right_hit and right_delta < left_delta:
            new_start = snapped_right - clip.duration
            c._snap_line = snapped_right
        else:
            new_start = raw_start

        new_start = max(0.0, new_start)

        # Determine destination track
        dst_track = c._track_at_y(y)
        if dst_track is None:
            dst_track = track
        dst_track_id = dst_track.id

        # Compute ghost rect (visual position)
        ghost_x = new_start * pps
        ghost_w = clip.duration * pps
        track_y = c._track_y_offset(dst_track_id)
        track_h = dst_track.height

        c._ghost = {
            "rect": QRectF(ghost_x, 0, ghost_w, track_h),
            "track_y": track_y,
            "name": clip.name,
            "color": QColor(clip.color).lighter(120),
            "new_start": new_start,
            "dst_track_id": dst_track_id,
        }

        c.update()

    def _clips_overlap(self, track, clip, new_start: float, exclude_idx: int = -1) -> bool:
        """Check if placing *clip* at *new_start* would overlap any other clip on *track*."""
        new_end = new_start + clip.duration
        for i, other in enumerate(track.clips):
            if i == exclude_idx:
                continue
            other_end = other.start_time + other.duration
            if new_start < other_end and new_end > other.start_time:
                return True
        return False

    def _commit_ghost(self, x: int, y: int) -> None:
        """Apply the ghost position to the actual clip on release."""
        c = self.canvas
        ghost = getattr(c, "_ghost", None)
        if ghost is None:
            return

        new_start = ghost.get("new_start", 0.0)
        dst_track_id = ghost.get("dst_track_id", self._drag_clip_track)

        track = self._find_track(self._drag_clip_track)
        if not track or self._drag_clip_idx < 0 or self._drag_clip_idx >= len(track.clips):
            return

        clip = track.clips[self._drag_clip_idx]

        # Cross-track move
        if dst_track_id != self._drag_clip_track:
            dst_track = self._find_track(dst_track_id)
            if not dst_track:
                return
            # Block cross-type moves (video ↔ audio)
            if dst_track.track_type != track.track_type:
                clip.start_time = self._drag_orig_start  # revert
                return
            # Block if would overlap on destination track
            if self._clips_overlap(dst_track, clip, new_start):
                clip.start_time = self._drag_orig_start
                return
            clip.start_time = new_start
            track.clips.pop(self._drag_clip_idx)
            dst_track.clips.append(clip)
            new_idx = len(dst_track.clips) - 1
            self._drag_clip_track = dst_track_id
            self._drag_clip_idx = new_idx
            c._selected_track_id = dst_track_id
            c._selected_clip_idx = new_idx
        else:
            # Same-track move: block if would overlap
            if self._clips_overlap(track, clip, new_start, exclude_idx=self._drag_clip_idx):
                clip.start_time = self._drag_orig_start  # revert
                return
            clip.start_time = new_start

    # -- Trim drag ---------------------------------------------------------

    def _do_trim_drag(self, x: int) -> None:
        """Apply non-destructive trim as user drags a clip edge."""
        c = self.canvas
        track = self._find_track(self._trim_track_id)
        if not track or self._trim_clip_idx < 0 or self._trim_clip_idx >= len(track.clips):
            self._trimming = False
            return

        clip = track.clips[self._trim_clip_idx]
        pps = self._pps()
        delta_px = x - self._trim_start_x
        delta_sec = delta_px / pps

        # Collect snap edges for trim snapping
        snap_edges = c._collect_snap_edges(self._trim_track_id, self._trim_clip_idx)

        if self._trim_edge == "left":
            self._trim_left(clip, delta_sec, track, snap_edges)
        elif self._trim_edge == "right":
            self._trim_right(clip, delta_sec, track, snap_edges)

        c.update()

    def _trim_left(
        self,
        clip: TimelineClip,
        delta_sec: float,
        track: TimelineTrack,
        snap_edges: list[float],
    ) -> None:
        c = self.canvas

        new_offset = self._trim_orig_media_offset + delta_sec
        new_offset = max(0.0, new_offset)
        max_offset = self._trim_orig_media_offset + self._trim_orig_duration - _MIN_CLIP_DURATION
        new_offset = min(new_offset, max_offset)

        offset_change = new_offset - self._trim_orig_media_offset
        new_start = self._trim_orig_start_time + offset_change
        new_dur = self._trim_orig_duration - offset_change

        # Snap left edge
        snapped, did_snap = c._snap_edge(new_start, snap_edges)
        if did_snap:
            snap_delta = snapped - new_start
            new_start = snapped
            new_dur -= snap_delta
            new_offset += snap_delta
            c._snap_line = snapped
        else:
            c._snap_line = -1.0

        # Final clamps
        new_start = max(0.0, new_start)
        new_dur = max(_MIN_CLIP_DURATION, new_dur)
        new_offset = max(0.0, min(new_offset, clip.media_duration - _MIN_CLIP_DURATION))

        # Prevent left-trim from overlapping the previous clip
        _, before_idx, _, _ = self._neighbor_clips(track, self._trim_clip_idx)
        if before_idx >= 0 and before_idx < len(track.clips):
            prev_end = track.clips[before_idx].start_time + track.clips[before_idx].duration
            if new_start < prev_end:
                delta = prev_end - new_start
                new_start = prev_end
                new_dur -= delta
                new_offset += delta

        clip.start_time = new_start
        clip.duration = max(_MIN_CLIP_DURATION, new_dur)
        clip.media_offset = new_offset

        # Roll edit: adjust adjacent clip
        if self._roll_trim and self._roll_neighbor_idx >= 0:
            self._roll_left_neighbor(track, new_start)

    def _trim_right(
        self,
        clip: TimelineClip,
        delta_sec: float,
        track: TimelineTrack,
        snap_edges: list[float],
    ) -> None:
        c = self.canvas

        new_dur = self._trim_orig_duration + delta_sec
        new_dur = max(_MIN_CLIP_DURATION, new_dur)
        max_dur = clip.media_duration - clip.media_offset
        new_dur = min(new_dur, max_dur)

        # Snap right edge
        right_edge = clip.start_time + new_dur
        snapped, did_snap = c._snap_edge(right_edge, snap_edges)
        if did_snap:
            new_dur = snapped - clip.start_time
            new_dur = max(_MIN_CLIP_DURATION, min(new_dur, max_dur))
            c._snap_line = snapped
        else:
            c._snap_line = -1.0

        # Prevent right-trim from overlapping the next clip
        _, _, after_clip, after_idx = self._neighbor_clips(track, self._trim_clip_idx)
        if after_clip is not None:
            max_right = after_clip.start_time
            if clip.start_time + new_dur > max_right:
                new_dur = max_right - clip.start_time
                new_dur = max(_MIN_CLIP_DURATION, new_dur)

        clip.duration = new_dur

        # Roll edit: adjust adjacent clip
        if self._roll_trim and self._roll_neighbor_idx >= 0:
            self._roll_right_neighbor(track, clip.start_time + new_dur)

    # -- Roll edit helpers -------------------------------------------------

    def _setup_roll_neighbor(
        self, track: TimelineTrack, clip_idx: int, edge_side: str
    ) -> None:
        """Find the adjacent clip for roll editing and store its original state."""
        clip_before, idx_before, clip_after, idx_after = self._neighbor_clips(track, clip_idx)

        if edge_side == "left" and clip_before is not None:
            self._roll_neighbor_idx = idx_before
            self._roll_neighbor_orig_start = clip_before.start_time
            self._roll_neighbor_orig_duration = clip_before.duration
            self._roll_neighbor_orig_media_offset = clip_before.media_offset
        elif edge_side == "right" and clip_after is not None:
            self._roll_neighbor_idx = idx_after
            self._roll_neighbor_orig_start = clip_after.start_time
            self._roll_neighbor_orig_duration = clip_after.duration
            self._roll_neighbor_orig_media_offset = clip_after.media_offset
        else:
            self._roll_neighbor_idx = -1

    def _roll_left_neighbor(self, track: TimelineTrack, new_left_edge: float) -> None:
        """Extend the preceding clip's duration to meet the new left edge of the trimmed clip."""
        if self._roll_neighbor_idx < 0 or self._roll_neighbor_idx >= len(track.clips):
            return
        nb = track.clips[self._roll_neighbor_idx]
        # The preceding clip's right edge should meet our left edge
        new_nb_dur = new_left_edge - nb.start_time
        max_nb_dur = nb.media_duration - nb.media_offset
        new_nb_dur = max(_MIN_CLIP_DURATION, min(new_nb_dur, max_nb_dur))
        nb.duration = new_nb_dur

    def _roll_right_neighbor(self, track: TimelineTrack, new_right_edge: float) -> None:
        """Adjust the following clip's start_time, media_offset, and duration to meet the new right edge."""
        if self._roll_neighbor_idx < 0 or self._roll_neighbor_idx >= len(track.clips):
            return
        nb = track.clips[self._roll_neighbor_idx]
        orig_end = self._roll_neighbor_orig_start + self._roll_neighbor_orig_duration
        delta = new_right_edge - self._roll_neighbor_orig_start
        new_nb_start = new_right_edge
        new_nb_dur = orig_end - new_right_edge
        new_nb_offset = self._roll_neighbor_orig_media_offset + delta

        # Clamps
        new_nb_dur = max(_MIN_CLIP_DURATION, new_nb_dur)
        new_nb_offset = max(0.0, new_nb_offset)
        new_nb_start = max(0.0, new_nb_start)

        nb.start_time = new_nb_start
        nb.duration = new_nb_dur
        nb.media_offset = new_nb_offset

    # -- Fade handle interaction -------------------------------------------

    def _fade_handle_at(
        self, x: int, y: int
    ) -> tuple[TimelineTrack, int, str] | None:
        """Check if (x, y) is within range of a fade endpoint handle.

        Returns (track, clip_idx, "in"|"out") or None.
        """
        c = self.canvas
        track = c._track_at_y(y)
        if not track:
            return None
        # Fades only apply to video tracks
        if track.track_type != TrackType.VIDEO:
            return None

        pps = self._pps()
        for i, clip in enumerate(track.clips):
            # Only check clips that have fade attributes
            fade_in = getattr(clip, "fade_in", 0.0)
            fade_out = getattr(clip, "fade_out", 0.0)

            clip_left_px = int(clip.start_time * pps)
            clip_right_px = int((clip.start_time + clip.duration) * pps)

            # Fade-in handle: at clip_left + fade_in_px
            if fade_in > 0:
                handle_x = clip_left_px + int(fade_in * pps)
                if abs(x - handle_x) <= _FADE_HANDLE_PX:
                    return track, i, "in"
            else:
                # Allow grabbing from the left edge to create a fade
                if abs(x - clip_left_px) <= _FADE_HANDLE_PX:
                    track_y = c._track_y_offset(track.id)
                    track_h = track.height
                    # Only trigger fade if in the top portion of the clip
                    if y < track_y + track_h // 3:
                        return track, i, "in"

            # Fade-out handle: at clip_right - fade_out_px
            if fade_out > 0:
                handle_x = clip_right_px - int(fade_out * pps)
                if abs(x - handle_x) <= _FADE_HANDLE_PX:
                    return track, i, "out"
            else:
                if abs(x - clip_right_px) <= _FADE_HANDLE_PX:
                    track_y = c._track_y_offset(track.id)
                    track_h = track.height
                    if y < track_y + track_h // 3:
                        return track, i, "out"

        return None

    def _do_fade_drag(self, x: int) -> None:
        """Adjust fade_in or fade_out based on handle drag."""
        c = self.canvas
        track = self._find_track(self._fade_track_id)
        if not track or self._fade_clip_idx < 0 or self._fade_clip_idx >= len(track.clips):
            self._fading = False
            return

        clip = track.clips[self._fade_clip_idx]
        pps = self._pps()
        max_fade = clip.duration / 2.0

        if self._fade_side == "in":
            clip_left_px = int(clip.start_time * pps)
            fade_px = x - clip_left_px
            fade_sec = max(0.0, min(fade_px / pps, max_fade))
            # Set fade_in attribute (may not exist yet on TimelineClip)
            clip.fade_in = fade_sec  # type: ignore[attr-defined]
        elif self._fade_side == "out":
            clip_right_px = int((clip.start_time + clip.duration) * pps)
            fade_px = clip_right_px - x
            fade_sec = max(0.0, min(fade_px / pps, max_fade))
            clip.fade_out = fade_sec  # type: ignore[attr-defined]

        c.update()


# ---------------------------------------------------------------------------
# ScrubTool
# ---------------------------------------------------------------------------


class ScrubTool(TimelineTool):
    """Click/drag to scrub the playhead."""

    def __init__(self, canvas: MultiTrackCanvasWidget) -> None:
        super().__init__(canvas)
        self._scrubbing: bool = False

    def cursor(self) -> Qt.CursorShape:
        return Qt.IBeamCursor

    def on_press(self, x: int, y: int, event: QMouseEvent) -> bool:
        self._scrubbing = True
        sec = self._pos_to_seconds(x)
        total = self.canvas._total_duration()
        sec = max(0.0, min(sec, total)) if total > 0 else 0.0
        self.canvas._playhead = sec
        self.canvas.playhead_moved.emit(sec)
        self.canvas.update()
        return True

    def on_move(self, x: int, y: int, event: QMouseEvent) -> None:
        if not self._scrubbing:
            return
        sec = self._pos_to_seconds(x)
        total = self.canvas._total_duration()
        sec = max(0.0, min(sec, total)) if total > 0 else 0.0
        self.canvas._playhead = sec
        self.canvas.playhead_moved.emit(sec)
        self.canvas.update()

    def on_release(self, x: int, y: int, event: QMouseEvent) -> None:
        self._scrubbing = False


# ---------------------------------------------------------------------------
# RazorTool
# ---------------------------------------------------------------------------


class RazorTool(TimelineTool):
    """Click to split a clip at the cursor position."""

    def __init__(self, canvas: MultiTrackCanvasWidget) -> None:
        super().__init__(canvas)

    def cursor(self) -> Qt.CursorShape:
        return Qt.CrossCursor

    def on_press(self, x: int, y: int, event: QMouseEvent) -> bool:
        c = self.canvas
        track = c._track_at_y(y)
        if not track:
            return False
        if self._track_locked(track):
            return False

        idx = c._clip_at_pos(x, track)
        if idx < 0:
            return False

        clip = track.clips[idx]
        split_time = self._pos_to_seconds(x)
        offset_in_clip = split_time - clip.start_time

        # Guard: don't split too close to either edge
        if offset_in_clip <= _MIN_CLIP_DURATION or (clip.duration - offset_in_clip) <= _MIN_CLIP_DURATION:
            return False

        # Emit razor_split signal if available
        if hasattr(c, "razor_split"):
            c.razor_split.emit(track.id, idx, offset_in_clip)

        # Clear razor preview
        c._razor_preview_x = -1
        c.update()
        return True

    def on_hover(self, x: int, y: int) -> Qt.CursorShape | None:
        c = self.canvas
        c._razor_preview_x = x
        c.update()
        return Qt.CrossCursor

    def on_move(self, x: int, y: int, event: QMouseEvent) -> None:
        c = self.canvas
        c._razor_preview_x = x
        c.update()

    def on_release(self, x: int, y: int, event: QMouseEvent) -> None:
        c = self.canvas
        c._razor_preview_x = -1
        c.update()


# ---------------------------------------------------------------------------
# SlipTool
# ---------------------------------------------------------------------------


class SlipTool(TimelineTool):
    """Shift the media content within a clip without moving the clip on the timeline.

    Changes media_offset while keeping start_time and duration fixed.
    """

    def __init__(self, canvas: MultiTrackCanvasWidget) -> None:
        super().__init__(canvas)
        self._slipping: bool = False
        self._slip_track_id: str = ""
        self._slip_clip_idx: int = -1
        self._slip_start_x: int = 0
        self._slip_orig_offset: float = 0.0

    def cursor(self) -> Qt.CursorShape:
        return Qt.SizeHorCursor

    def on_press(self, x: int, y: int, event: QMouseEvent) -> bool:
        c = self.canvas
        track = c._track_at_y(y)
        if not track:
            return False
        if self._track_locked(track):
            return False

        idx = c._clip_at_pos(x, track)
        if idx < 0:
            return False

        clip = track.clips[idx]
        self._slipping = True
        self._slip_track_id = track.id
        self._slip_clip_idx = idx
        self._slip_start_x = x
        self._slip_orig_offset = clip.media_offset

        c._selected_track_id = track.id
        c._selected_clip_idx = idx
        c.clip_selected.emit(track.id, idx)
        c.update()
        return True

    def on_move(self, x: int, y: int, event: QMouseEvent) -> None:
        if not self._slipping:
            return

        c = self.canvas
        track = self._find_track(self._slip_track_id)
        if not track or self._slip_clip_idx < 0 or self._slip_clip_idx >= len(track.clips):
            self._slipping = False
            return

        clip = track.clips[self._slip_clip_idx]
        pps = self._pps()
        delta_sec = (x - self._slip_start_x) / pps

        new_offset = self._slip_orig_offset + delta_sec
        # Clamp: can't go before media start, can't go past media end minus visible duration
        max_offset = clip.media_duration - clip.duration
        new_offset = max(0.0, min(new_offset, max_offset))

        clip.media_offset = new_offset
        c.update()

    def on_release(self, x: int, y: int, event: QMouseEvent) -> None:
        if self._slipping:
            self._slipping = False
            self.canvas.clips_changed.emit()


# ---------------------------------------------------------------------------
# SlideTool
# ---------------------------------------------------------------------------


class SlideTool(TimelineTool):
    """Move a clip on the timeline while adjusting neighbors to fill the gap.

    The clip_before is extended (duration grows) and clip_after is shifted
    (start_time and media_offset adjust) to keep the overall sequence length
    constant.
    """

    def __init__(self, canvas: MultiTrackCanvasWidget) -> None:
        super().__init__(canvas)
        self._sliding: bool = False
        self._slide_track_id: str = ""
        self._slide_clip_idx: int = -1
        self._slide_start_x: int = 0
        self._slide_orig_start: float = 0.0

        # Neighbors
        self._before_idx: int = -1
        self._before_orig_dur: float = 0.0
        self._after_idx: int = -1
        self._after_orig_start: float = 0.0
        self._after_orig_dur: float = 0.0
        self._after_orig_media_offset: float = 0.0

    def cursor(self) -> Qt.CursorShape:
        return Qt.SizeHorCursor

    def on_press(self, x: int, y: int, event: QMouseEvent) -> bool:
        c = self.canvas
        track = c._track_at_y(y)
        if not track:
            return False
        if self._track_locked(track):
            return False

        idx = c._clip_at_pos(x, track)
        if idx < 0:
            return False

        clip = track.clips[idx]
        self._sliding = True
        self._slide_track_id = track.id
        self._slide_clip_idx = idx
        self._slide_start_x = x
        self._slide_orig_start = clip.start_time

        # Find neighbors
        clip_before, idx_before, clip_after, idx_after = self._neighbor_clips(track, idx)

        if clip_before is not None:
            self._before_idx = idx_before
            self._before_orig_dur = clip_before.duration
        else:
            self._before_idx = -1

        if clip_after is not None:
            self._after_idx = idx_after
            self._after_orig_start = clip_after.start_time
            self._after_orig_dur = clip_after.duration
            self._after_orig_media_offset = clip_after.media_offset
        else:
            self._after_idx = -1

        c._selected_track_id = track.id
        c._selected_clip_idx = idx
        c.clip_selected.emit(track.id, idx)
        c.update()
        return True

    def on_move(self, x: int, y: int, event: QMouseEvent) -> None:
        if not self._sliding:
            return

        c = self.canvas
        track = self._find_track(self._slide_track_id)
        if not track or self._slide_clip_idx < 0 or self._slide_clip_idx >= len(track.clips):
            self._sliding = False
            return

        clip = track.clips[self._slide_clip_idx]
        pps = self._pps()
        delta_sec = (x - self._slide_start_x) / pps

        # Compute limits based on neighbors
        min_delta = -float("inf")
        max_delta = float("inf")

        if self._before_idx >= 0 and self._before_idx < len(track.clips):
            nb_before = track.clips[self._before_idx]
            # Don't let the before clip shrink below minimum
            min_delta = max(min_delta, -(self._before_orig_dur - _MIN_CLIP_DURATION))

        if self._after_idx >= 0 and self._after_idx < len(track.clips):
            nb_after = track.clips[self._after_idx]
            # Don't let the after clip shrink below minimum
            max_delta = min(max_delta, self._after_orig_dur - _MIN_CLIP_DURATION)

        # Clamp: clip itself can't go before t=0
        if self._slide_orig_start + delta_sec < 0:
            delta_sec = -self._slide_orig_start

        delta_sec = max(min_delta, min(delta_sec, max_delta))

        # Apply
        clip.start_time = self._slide_orig_start + delta_sec

        # Adjust before neighbor: extend/shrink its duration
        if self._before_idx >= 0 and self._before_idx < len(track.clips):
            nb_before = track.clips[self._before_idx]
            nb_before.duration = max(_MIN_CLIP_DURATION, self._before_orig_dur + delta_sec)

        # Adjust after neighbor: shift start_time + media_offset, shrink duration
        if self._after_idx >= 0 and self._after_idx < len(track.clips):
            nb_after = track.clips[self._after_idx]
            nb_after.start_time = self._after_orig_start + delta_sec
            nb_after.media_offset = max(0.0, self._after_orig_media_offset + delta_sec)
            nb_after.duration = max(_MIN_CLIP_DURATION, self._after_orig_dur - delta_sec)

        c.update()

    def on_release(self, x: int, y: int, event: QMouseEvent) -> None:
        if self._sliding:
            self._sliding = False
            self._before_idx = -1
            self._after_idx = -1
            self.canvas.clips_changed.emit()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_TOOL_MAP: dict[ToolMode, type[TimelineTool]] = {
    ToolMode.SELECT: SelectTool,
    ToolMode.SCRUB: ScrubTool,
    ToolMode.RAZOR: RazorTool,
    ToolMode.SLIP: SlipTool,
    ToolMode.SLIDE: SlideTool,
}


def create_tool(mode: ToolMode, canvas: MultiTrackCanvasWidget) -> TimelineTool:
    """Instantiate the tool class for *mode*."""
    cls = _TOOL_MAP.get(mode, SelectTool)
    return cls(canvas)
