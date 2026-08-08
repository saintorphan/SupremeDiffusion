"""Command-based undo/redo system for the timeline NLE.

STATUS: Complete but not yet wired into the UI. The current TimelineTab uses
_UndoStack with full-state snapshots, which works but is O(n_clips) per edit.
This module provides O(1) per-command undo/redo as a future improvement.
To integrate: replace _UndoStack usage in timeline.py with CommandHistory
and emit the appropriate commands from each edit operation.

Replaces the old snapshot-based undo that serialized ALL tracks on every edit.
Each command stores only the minimal state required to execute and reverse
a single operation, making undo O(1) instead of O(n_clips).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from PySide6.QtGui import QColor

from sdqt.widgets.timeline_track import TimelineClip, TimelineMarker, TimelineTrack

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _find_track(tracks: list[TimelineTrack], track_id: str) -> TimelineTrack | None:
    """Return the track with *track_id*, or ``None``."""
    for t in tracks:
        if t.id == track_id:
            return t
    return None


def _clip_to_dict(clip: TimelineClip) -> dict[str, Any]:
    """Serialize a TimelineClip to a plain dict (enough to reconstruct it)."""
    return {
        "path": clip.path,
        "duration": clip.duration,
        "start_time": clip.start_time,
        "name": clip.name,
        "color": clip.color.name() if isinstance(clip.color, QColor) else str(clip.color),
        "volume": clip.volume,
        "has_audio": clip.has_audio,
        "media_duration": clip.media_duration,
        "media_offset": clip.media_offset,
        "fade_in": clip.fade_in,
        "fade_out": clip.fade_out,
        "group_id": clip.group_id,
        "transition_in": clip.transition_in,
    }


def _clip_from_dict(data: dict[str, Any]) -> TimelineClip:
    """Reconstruct a TimelineClip from a dict produced by :func:`_clip_to_dict`.

    Non-serialisable fields (thumbnail, waveform) are left at defaults; the
    canvas will regenerate them on the next repaint.
    """
    return TimelineClip(
        path=data["path"],
        duration=data["duration"],
        start_time=data.get("start_time", 0.0),
        name=data.get("name", ""),
        color=QColor(data["color"]) if "color" in data else QColor("#4a90d9"),
        volume=data.get("volume", 1.0),
        has_audio=data.get("has_audio", True),
        media_duration=data.get("media_duration", data["duration"]),
        media_offset=data.get("media_offset", 0.0),
        fade_in=data.get("fade_in", 0.0),
        fade_out=data.get("fade_out", 0.0),
        group_id=data.get("group_id", ""),
        transition_in=data.get("transition_in"),
    )


def _marker_to_dict(marker: TimelineMarker) -> dict[str, Any]:
    """Serialize a TimelineMarker."""
    return {"time": marker.time, "name": marker.name, "color": marker.color}


def _marker_from_dict(data: dict[str, Any]) -> TimelineMarker:
    """Reconstruct a TimelineMarker."""
    return TimelineMarker(
        time=data["time"],
        name=data.get("name", ""),
        color=data.get("color", "#ffcc00"),
    )


# ---------------------------------------------------------------------------
#  Base command
# ---------------------------------------------------------------------------

class TimelineCommand(ABC):
    """Abstract base for every undoable timeline operation."""

    @abstractmethod
    def execute(self) -> None:
        """Apply the edit (first time or redo)."""

    @abstractmethod
    def undo(self) -> None:
        """Reverse the edit."""

    def merge_id(self) -> str | None:
        """Return a non-``None`` string to enable merging with the previous
        command on the undo stack when the IDs match.  Default is ``None``
        (no merging)."""
        return None

    # Subclasses that support merging must override _merge().
    def _merge(self, other: TimelineCommand) -> None:  # noqa: ARG002
        """Absorb *other*'s "new" values into this command.

        Called by :class:`CommandStack` when ``other.merge_id()`` equals
        ``self.merge_id()`` and both are not ``None``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares merge_id() but does not implement _merge()"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__}>"


# ---------------------------------------------------------------------------
#  Concrete commands
# ---------------------------------------------------------------------------

class MoveClipCommand(TimelineCommand):
    """Move a clip to a new start time and/or a different track."""

    def __init__(
        self,
        tracks: list[TimelineTrack],
        track_id: str,
        clip_idx: int,
        old_start: float,
        new_start: float,
        old_track_id: str | None = None,
        new_track_id: str | None = None,
    ) -> None:
        self._tracks = tracks
        self._track_id = track_id
        self._clip_idx = clip_idx
        self._old_start = old_start
        self._new_start = new_start
        self._old_track_id = old_track_id or track_id
        self._new_track_id = new_track_id or track_id

    # -- helpers --
    def _move(self, from_tid: str, to_tid: str, start: float) -> None:
        src = _find_track(self._tracks, from_tid)
        if src is None or self._clip_idx >= len(src.clips):
            logger.warning("MoveClipCommand: source track/clip not found")
            return
        clip = src.clips[self._clip_idx]
        if from_tid != to_tid:
            src.clips.pop(self._clip_idx)
            dst = _find_track(self._tracks, to_tid)
            if dst is None:
                logger.warning("MoveClipCommand: destination track not found")
                src.clips.insert(self._clip_idx, clip)
                return
            clip.start_time = start
            dst.clips.append(clip)
            dst.clips.sort(key=lambda c: c.start_time)
        else:
            clip.start_time = start

    def execute(self) -> None:
        self._move(self._old_track_id, self._new_track_id, self._new_start)

    def undo(self) -> None:
        # Reverse: find clip in new track and move back
        if self._old_track_id != self._new_track_id:
            dst = _find_track(self._tracks, self._new_track_id)
            if dst is None:
                return
            # Find clip that was moved (match by old start, name, path)
            for i, c in enumerate(dst.clips):
                if abs(c.start_time - self._new_start) < 1e-6:
                    clip = dst.clips.pop(i)
                    clip.start_time = self._old_start
                    src = _find_track(self._tracks, self._old_track_id)
                    if src is not None:
                        src.clips.insert(self._clip_idx, clip)
                    break
        else:
            track = _find_track(self._tracks, self._track_id)
            if track and self._clip_idx < len(track.clips):
                track.clips[self._clip_idx].start_time = self._old_start

    def __repr__(self) -> str:
        return (
            f"<MoveClipCommand track={self._track_id} idx={self._clip_idx} "
            f"{self._old_start:.2f}->{self._new_start:.2f}>"
        )


class TrimClipCommand(TimelineCommand):
    """Adjust a clip's in/out point (start, duration, media_offset)."""

    def __init__(
        self,
        tracks: list[TimelineTrack],
        track_id: str,
        clip_idx: int,
        old_start: float,
        new_start: float,
        old_dur: float,
        new_dur: float,
        old_offset: float,
        new_offset: float,
    ) -> None:
        self._tracks = tracks
        self._track_id = track_id
        self._clip_idx = clip_idx
        self._old_start = old_start
        self._new_start = new_start
        self._old_dur = old_dur
        self._new_dur = new_dur
        self._old_offset = old_offset
        self._new_offset = new_offset

    def _apply(self, start: float, dur: float, offset: float) -> None:
        track = _find_track(self._tracks, self._track_id)
        if track is None or self._clip_idx >= len(track.clips):
            return
        clip = track.clips[self._clip_idx]
        clip.start_time = start
        clip.duration = dur
        clip.media_offset = offset

    def execute(self) -> None:
        self._apply(self._new_start, self._new_dur, self._new_offset)

    def undo(self) -> None:
        self._apply(self._old_start, self._old_dur, self._old_offset)

    def merge_id(self) -> str | None:
        return f"trim:{self._track_id}:{self._clip_idx}"

    def _merge(self, other: TrimClipCommand) -> None:  # type: ignore[override]
        self._new_start = other._new_start
        self._new_dur = other._new_dur
        self._new_offset = other._new_offset

    def __repr__(self) -> str:
        return (
            f"<TrimClipCommand track={self._track_id} idx={self._clip_idx} "
            f"dur {self._old_dur:.2f}->{self._new_dur:.2f}>"
        )


class SplitClipCommand(TimelineCommand):
    """Split a clip at *offset_sec* into two clips (A and B)."""

    def __init__(
        self,
        tracks: list[TimelineTrack],
        track_id: str,
        clip_idx: int,
        offset_sec: float,
        clip_a_data: dict[str, Any],
        clip_b_data: dict[str, Any],
    ) -> None:
        self._tracks = tracks
        self._track_id = track_id
        self._clip_idx = clip_idx
        self._offset_sec = offset_sec
        self._clip_a_data = clip_a_data
        self._clip_b_data = clip_b_data
        # Snapshot the original clip for undo (set lazily on first execute).
        self._original_data: dict[str, Any] | None = None

    def execute(self) -> None:
        track = _find_track(self._tracks, self._track_id)
        if track is None or self._clip_idx >= len(track.clips):
            return
        # Save original clip data on first execute so undo can restore it.
        if self._original_data is None:
            self._original_data = _clip_to_dict(track.clips[self._clip_idx])
        track.clips.pop(self._clip_idx)
        clip_a = _clip_from_dict(self._clip_a_data)
        clip_b = _clip_from_dict(self._clip_b_data)
        track.clips.insert(self._clip_idx, clip_b)
        track.clips.insert(self._clip_idx, clip_a)

    def undo(self) -> None:
        track = _find_track(self._tracks, self._track_id)
        if track is None:
            return
        # Remove the two split clips (A at clip_idx, B at clip_idx+1).
        if self._clip_idx + 1 < len(track.clips):
            track.clips.pop(self._clip_idx + 1)
        if self._clip_idx < len(track.clips):
            track.clips.pop(self._clip_idx)
        # Restore original.
        if self._original_data is not None:
            track.clips.insert(self._clip_idx, _clip_from_dict(self._original_data))

    def __repr__(self) -> str:
        return (
            f"<SplitClipCommand track={self._track_id} idx={self._clip_idx} "
            f"@{self._offset_sec:.2f}s>"
        )


class DeleteClipCommand(TimelineCommand):
    """Remove a clip from a track."""

    def __init__(
        self,
        tracks: list[TimelineTrack],
        track_id: str,
        clip_idx: int,
        clip_data: dict[str, Any],
    ) -> None:
        self._tracks = tracks
        self._track_id = track_id
        self._clip_idx = clip_idx
        self._clip_data = clip_data

    def execute(self) -> None:
        track = _find_track(self._tracks, self._track_id)
        if track is None or self._clip_idx >= len(track.clips):
            return
        track.clips.pop(self._clip_idx)

    def undo(self) -> None:
        track = _find_track(self._tracks, self._track_id)
        if track is None:
            return
        clip = _clip_from_dict(self._clip_data)
        idx = min(self._clip_idx, len(track.clips))
        track.clips.insert(idx, clip)

    def __repr__(self) -> str:
        return f"<DeleteClipCommand track={self._track_id} idx={self._clip_idx}>"


class AddClipCommand(TimelineCommand):
    """Add a clip to a track."""

    def __init__(
        self,
        tracks: list[TimelineTrack],
        track_id: str,
        clip_data: dict[str, Any],
    ) -> None:
        self._tracks = tracks
        self._track_id = track_id
        self._clip_data = clip_data
        self._inserted_idx: int | None = None

    def execute(self) -> None:
        track = _find_track(self._tracks, self._track_id)
        if track is None:
            return
        clip = _clip_from_dict(self._clip_data)
        track.clips.append(clip)
        track.clips.sort(key=lambda c: c.start_time)
        self._inserted_idx = track.clips.index(clip)

    def undo(self) -> None:
        track = _find_track(self._tracks, self._track_id)
        if track is None:
            return
        if self._inserted_idx is not None and self._inserted_idx < len(track.clips):
            track.clips.pop(self._inserted_idx)
        else:
            # Fallback: remove by matching start_time + path.
            st = self._clip_data.get("start_time", 0.0)
            path = self._clip_data.get("path", "")
            for i, c in enumerate(track.clips):
                if abs(c.start_time - st) < 1e-6 and c.path == path:
                    track.clips.pop(i)
                    break

    def __repr__(self) -> str:
        return f"<AddClipCommand track={self._track_id}>"


class FadeChangeCommand(TimelineCommand):
    """Change a clip's fade-in and/or fade-out durations."""

    def __init__(
        self,
        tracks: list[TimelineTrack],
        track_id: str,
        clip_idx: int,
        old_fade_in: float,
        new_fade_in: float,
        old_fade_out: float,
        new_fade_out: float,
    ) -> None:
        self._tracks = tracks
        self._track_id = track_id
        self._clip_idx = clip_idx
        self._old_fade_in = old_fade_in
        self._new_fade_in = new_fade_in
        self._old_fade_out = old_fade_out
        self._new_fade_out = new_fade_out

    def _apply(self, fade_in: float, fade_out: float) -> None:
        track = _find_track(self._tracks, self._track_id)
        if track is None or self._clip_idx >= len(track.clips):
            return
        track.clips[self._clip_idx].fade_in = fade_in
        track.clips[self._clip_idx].fade_out = fade_out

    def execute(self) -> None:
        self._apply(self._new_fade_in, self._new_fade_out)

    def undo(self) -> None:
        self._apply(self._old_fade_in, self._old_fade_out)

    def merge_id(self) -> str | None:
        return f"fade:{self._track_id}:{self._clip_idx}"

    def _merge(self, other: FadeChangeCommand) -> None:  # type: ignore[override]
        self._new_fade_in = other._new_fade_in
        self._new_fade_out = other._new_fade_out

    def __repr__(self) -> str:
        return (
            f"<FadeChangeCommand track={self._track_id} idx={self._clip_idx} "
            f"in={self._new_fade_in:.2f} out={self._new_fade_out:.2f}>"
        )


class RippleDeleteCommand(TimelineCommand):
    """Delete a clip and shift all subsequent clips left to close the gap."""

    def __init__(
        self,
        tracks: list[TimelineTrack],
        track_id: str,
        clip_idx: int,
        clip_data: dict[str, Any],
        shifts: list[tuple[int, float]],
    ) -> None:
        """
        Parameters
        ----------
        shifts:
            List of ``(clip_index_after_removal, shift_amount)`` pairs
            describing how subsequent clips were shifted.  The indices
            refer to positions *after* the deleted clip has been removed.
        """
        self._tracks = tracks
        self._track_id = track_id
        self._clip_idx = clip_idx
        self._clip_data = clip_data
        self._shifts = shifts

    def execute(self) -> None:
        track = _find_track(self._tracks, self._track_id)
        if track is None or self._clip_idx >= len(track.clips):
            return
        track.clips.pop(self._clip_idx)
        # Apply shifts to subsequent clips.
        for idx, amount in self._shifts:
            if idx < len(track.clips):
                track.clips[idx].start_time -= amount

    def undo(self) -> None:
        track = _find_track(self._tracks, self._track_id)
        if track is None:
            return
        # Un-shift subsequent clips first (using reversed order for safety).
        for idx, amount in reversed(self._shifts):
            if idx < len(track.clips):
                track.clips[idx].start_time += amount
        # Re-insert deleted clip.
        clip = _clip_from_dict(self._clip_data)
        idx = min(self._clip_idx, len(track.clips))
        track.clips.insert(idx, clip)

    def __repr__(self) -> str:
        return (
            f"<RippleDeleteCommand track={self._track_id} idx={self._clip_idx} "
            f"shifts={len(self._shifts)}>"
        )


class AddRemoveMarkerCommand(TimelineCommand):
    """Add or remove a timeline marker."""

    def __init__(
        self,
        markers_list: list[TimelineMarker],
        marker_data: dict[str, Any],
        is_add: bool,
    ) -> None:
        self._markers = markers_list
        self._data = marker_data
        self._is_add = is_add

    def _add(self) -> None:
        self._markers.append(_marker_from_dict(self._data))

    def _remove(self) -> None:
        t = self._data["time"]
        for i, m in enumerate(self._markers):
            if abs(m.time - t) < 1e-6:
                self._markers.pop(i)
                return

    def execute(self) -> None:
        if self._is_add:
            self._add()
        else:
            self._remove()

    def undo(self) -> None:
        if self._is_add:
            self._remove()
        else:
            self._add()

    def __repr__(self) -> str:
        verb = "Add" if self._is_add else "Remove"
        return f"<{verb}MarkerCommand t={self._data.get('time', '?')}>"


class GroupCommand(TimelineCommand):
    """Group or ungroup a set of clips by setting/clearing their ``group_id``."""

    def __init__(
        self,
        tracks: list[TimelineTrack],
        clip_refs: list[tuple[str, int]],
        new_group_id: str,
        old_group_ids: list[str],
    ) -> None:
        """
        Parameters
        ----------
        clip_refs:
            List of ``(track_id, clip_idx)`` identifying each clip.
        new_group_id:
            The group ID to assign (empty string to ungroup).
        old_group_ids:
            The previous ``group_id`` for each clip, in the same order as
            *clip_refs*.
        """
        self._tracks = tracks
        self._clip_refs = clip_refs
        self._new_group_id = new_group_id
        self._old_group_ids = old_group_ids

    def _apply(self, group_ids: list[str] | str) -> None:
        for i, (tid, cidx) in enumerate(self._clip_refs):
            track = _find_track(self._tracks, tid)
            if track is None or cidx >= len(track.clips):
                continue
            gid = group_ids if isinstance(group_ids, str) else group_ids[i]
            track.clips[cidx].group_id = gid

    def execute(self) -> None:
        self._apply(self._new_group_id)

    def undo(self) -> None:
        self._apply(self._old_group_ids)

    def __repr__(self) -> str:
        action = "Group" if self._new_group_id else "Ungroup"
        return f"<GroupCommand {action} clips={len(self._clip_refs)}>"


# ---------------------------------------------------------------------------
#  Command stack (undo/redo manager)
# ---------------------------------------------------------------------------

class CommandStack:
    """Manages the undo/redo history as a stack of :class:`TimelineCommand`
    objects.

    Supports optional command merging: if the most recent command on the undo
    stack has the same non-``None`` ``merge_id()`` as a newly pushed command,
    the two are merged instead of creating a new entry.  This collapses
    continuous drag operations (e.g. trimming) into a single undoable step.
    """

    def __init__(self, max_size: int = 200) -> None:
        self._undo: list[TimelineCommand] = []
        self._redo: list[TimelineCommand] = []
        self._max = max_size

    # -- state queries --

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_text(self) -> str:
        """Human-readable label for the top undo command."""
        if self._undo:
            return repr(self._undo[-1])
        return ""

    @property
    def redo_text(self) -> str:
        """Human-readable label for the top redo command."""
        if self._redo:
            return repr(self._redo[-1])
        return ""

    # -- core operations --

    def push(self, cmd: TimelineCommand) -> None:
        """Push *cmd*, executing it first if it hasn't been applied yet.

        If merging is possible (same ``merge_id``), the top command is updated
        in-place instead of adding a new entry.
        """
        mid = cmd.merge_id()
        if (
            mid is not None
            and self._undo
            and self._undo[-1].merge_id() == mid
        ):
            self._undo[-1]._merge(cmd)
        else:
            self._undo.append(cmd)
            if len(self._undo) > self._max:
                self._undo.pop(0)
        # Any new action invalidates the redo history.
        self._redo.clear()

    def undo(self) -> bool:
        """Undo the most recent command.  Returns ``True`` on success."""
        if not self._undo:
            return False
        cmd = self._undo.pop()
        cmd.undo()
        self._redo.append(cmd)
        return True

    def redo(self) -> bool:
        """Redo the most recently undone command.  Returns ``True`` on success."""
        if not self._redo:
            return False
        cmd = self._redo.pop()
        cmd.execute()
        self._undo.append(cmd)
        return True

    def clear(self) -> None:
        """Discard all history."""
        self._undo.clear()
        self._redo.clear()

    def __len__(self) -> int:
        return len(self._undo)

    def __repr__(self) -> str:
        return f"<CommandStack undo={len(self._undo)} redo={len(self._redo)}>"
