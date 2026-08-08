"""Timeline clip operations mixin — split, reverse, speed, duplicate, freeze, crossfade, etc."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QDialog, QInputDialog

from sdqt.utils.naming import cap_stem

if TYPE_CHECKING:
    from sdqt.widgets.timeline_track import TimelineClip, TimelineTrack

logger = logging.getLogger(__name__)


class TimelineClipOps:
    """Mixin providing clip operation methods for TimelineTab."""

    # -- Split ---------------------------------------------------------------

    def _on_split(self) -> None:
        """Split the selected clip at the playhead position (non-destructive)."""
        track, idx = self._get_selected()
        if not track or idx < 0:
            self._status_label.setText("No clip selected")
            return
        clip = track.clips[idx]
        ph = self._multitrack.playhead
        if not (clip.start_time + 0.05 < ph < clip.start_time + clip.duration - 0.05):
            self._status_label.setText("Playhead not within selected clip")
            return
        offset = ph - clip.start_time
        self._do_split(track, idx, offset)

    def _do_split(self, track, clip_idx: int, offset_sec: float) -> None:
        """Non-destructive split: creates two clips referencing the same source file."""
        import copy
        from sdqt.widgets.timeline_track import TimelineClip, _CLIP_COLORS

        clip = track.clips[clip_idx]
        dur1 = offset_sec
        dur2 = clip.duration - offset_sec

        if dur1 < 0.1 or dur2 < 0.1:
            self._status_label.setText("Split point too close to edge")
            return

        name = clip.name
        start = clip.start_time
        mo = clip.media_offset
        md = clip.media_duration
        color = clip.color

        effects_a = copy.deepcopy(clip.effects)
        effects_b = copy.deepcopy(clip.effects)

        clip_a = TimelineClip(
            path=clip.path, duration=dur1, start_time=start,
            name=f"{name}_A", color=color,
            volume=clip.volume, has_audio=clip.has_audio,
            waveform=clip.waveform,
            media_duration=md, media_offset=mo,
            fade_in=clip.fade_in, effects=effects_a,
            rotation=clip.rotation,
        )

        clip_b = TimelineClip(
            path=clip.path, duration=dur2, start_time=start + dur1,
            name=f"{name}_B",
            color=_CLIP_COLORS[(hash(name) + 1) % len(_CLIP_COLORS)],
            volume=clip.volume, has_audio=clip.has_audio,
            waveform=clip.waveform,
            media_duration=md, media_offset=mo + dur1,
            fade_out=clip.fade_out, effects=effects_b,
            rotation=clip.rotation,
        )

        track.clips.pop(clip_idx)
        track.clips.insert(clip_idx, clip_a)
        track.clips.insert(clip_idx + 1, clip_b)

        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._status_label.setText("Split complete")

    # -- Clip process helper ------------------------------------------------

    def _run_clip_process(
        self, cmd: list[str], output: str,
        track, idx: int, clip, new_name: str,
        status_msg: str, done_msg: str,
        preserve_duration: bool = False,
    ) -> None:
        """Run an ffmpeg clip operation on a background thread.

        *preserve_duration*: for ops that don't change length (reverse),
        keep the clip's timeline duration (clamped to the new media) instead
        of adopting the re-probed one — encoders round to whole frames /
        pad audio, and writing that growth back overlaps snapped neighbors.
        """
        from sdqt.workers.clip_process import ClipProcessWorker

        self._status_label.setText(status_msg)
        worker = ClipProcessWorker(cmd, output, parent=self)

        def _on_done(result):
            out_path, dur = result
            out_path = self._persist_clip(out_path)
            clip.path = out_path
            clip.duration = min(clip.duration, dur) if preserve_duration else dur
            clip.media_duration = dur
            clip.media_offset = 0.0
            clip.name = new_name
            try:
                from sdqt.utils.waveform import extract_waveform
                clip.waveform = extract_waveform(out_path)
            except Exception:
                clip.waveform = None
            self._library.add_clip(out_path, dur)
            self._multitrack._sync_canvas()
            self._multitrack.clips_changed.emit()
            self._status_label.setText(done_msg)

        def _on_error(msg):
            self._status_label.setText(f"Error: {msg}")

        worker.finished_ok.connect(_on_done)
        worker.error.connect(_on_error)
        worker.status.connect(lambda s: self._status_label.setText(s))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    # -- Reverse -------------------------------------------------------------

    def _on_reverse(self) -> None:
        track, idx = self._get_selected()
        if not track or idx < 0:
            self._status_label.setText("No clip selected")
            return
        clip = track.clips[idx]
        self._do_reverse(track, idx, clip)

    def _do_reverse(self, track, idx: int, clip) -> None:
        from sdqt.utils.codec import pix_fmt_args, configured_codec_args as _codec_args

        src = clip.path
        mo = getattr(clip, "media_offset", 0.0) or 0.0
        trim_end = mo + clip.duration
        stem = cap_stem(Path(src).stem)
        suffix = Path(src).suffix
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix=f"{stem}_rev_")
        tmp.close()

        cmd = [
            "ffmpeg", "-y", "-i", src,
            "-vf", f"scale=in_range=full:out_range=full,trim={mo}:{trim_end},setpts=PTS-STARTPTS,reverse,setpts=PTS-STARTPTS",
            "-af", f"atrim={mo}:{trim_end},asetpts=PTS-STARTPTS,areverse,asetpts=PTS-STARTPTS",
            *_codec_args(),
            "-c:a", "aac", "-b:a", "192k",
            *pix_fmt_args(),
            tmp.name,
        ]

        name = clip.name
        new_name = f"{name}_rev" if not name.endswith("_rev") else name[:-4]
        self._run_clip_process(
            cmd, tmp.name, track, idx, clip, new_name, "Reversing...", "Reverse complete",
            preserve_duration=True,
        )

    # -- Speed ---------------------------------------------------------------

    def _on_speed(self) -> None:
        from sdqt.tabs.timeline_dialogs import SpeedDialog
        track, idx = self._get_selected()
        if not track or idx < 0:
            self._status_label.setText("No clip selected")
            return

        clip = track.clips[idx]
        dlg = SpeedDialog(clip_duration=clip.duration, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        clip = track.clips[idx]

        if dlg.reverse():
            self._do_reverse(track, idx, clip)
            if idx < len(track.clips):
                clip = track.clips[idx]
            else:
                return

        speed = dlg.speed_value()
        if abs(speed - 1.0) < 0.01:
            return
        self._apply_speed(track, idx, clip, speed)

    def _apply_speed(self, track, idx: int, clip, speed: float) -> None:
        from sdqt.utils.codec import pix_fmt_args, configured_codec_args as _codec_args

        src = clip.path
        mo = getattr(clip, "media_offset", 0.0) or 0.0
        trim_end = mo + clip.duration
        stem = cap_stem(Path(src).stem)
        suffix = Path(src).suffix
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix=f"{stem}_spd_")
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

        vf = f"scale=in_range=full:out_range=full,trim={mo}:{trim_end},setpts=PTS-STARTPTS,setpts={pts_factor:.6f}*PTS"
        af = f"atrim={mo}:{trim_end},asetpts=PTS-STARTPTS," + ",".join(atempo_filters)

        cmd = [
            "ffmpeg", "-y", "-i", src,
            "-vf", vf, "-af", af,
            *_codec_args(),
            "-c:a", "aac", "-b:a", "192k",
            *pix_fmt_args(),
            tmp.name,
        ]

        self._run_clip_process(
            cmd, tmp.name, track, idx, clip,
            f"{clip.name}_{speed:.1f}x",
            f"Applying {speed:.2f}x speed...",
            f"Speed {speed:.2f}x applied",
        )

    # -- Duplicate -----------------------------------------------------------

    def _on_duplicate(self) -> None:
        """Duplicate the selected clip, placing the copy right after it."""
        track, idx = self._get_selected()
        if not track or idx < 0:
            self._status_label.setText("No clip selected")
            return
        clip = track.clips[idx]
        import copy
        dup = copy.deepcopy(clip)
        dup.start_time = clip.start_time + clip.duration
        dup.name = f"{clip.name} (copy)"
        track.clips.insert(idx + 1, dup)
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._status_label.setText(f"Duplicated: {clip.name}")

    # -- Ripple delete -------------------------------------------------------

    def _on_ripple_delete(self) -> None:
        """Delete the selected clip and close the gap."""
        track, idx = self._get_selected()
        if not track or idx < 0:
            self._status_label.setText("No clip selected")
            return
        clip = track.clips[idx]
        gap = clip.duration
        track.clips.pop(idx)
        for c in track.clips[idx:]:
            c.start_time = max(0, c.start_time - gap)
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._status_label.setText("Clip deleted (ripple)")

    # -- Freeze frame --------------------------------------------------------

    def _on_freeze_frame(self) -> None:
        """Create a freeze frame clip from the current playhead position."""
        from sdqt.widgets.timeline_track import TimelineClip
        from sdqt.utils.codec import (
            configured_codec_args as _codec_args,
            configured_encoder_name as _enc_name,
            pix_fmt_args as _pix_fmt,
        )

        track, idx = self._get_selected()
        if not track or idx < 0:
            self._status_label.setText("No clip selected")
            return
        clip = track.clips[idx]
        playhead = self._multitrack.playhead
        offset = max(0, playhead - clip.start_time) + clip.media_offset
        duration, ok = QInputDialog.getDouble(
            self, "Freeze Frame", "Freeze duration (seconds):",
            value=2.0, min=0.5, max=30.0, decimals=1)
        if not ok:
            return
        try:
            import subprocess
            frame_tmp = tempfile.NamedTemporaryFile(suffix=".png", prefix="freeze_frame_", delete=False)
            frame_tmp.close()
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(offset), "-i", clip.path,
                 "-vframes", "1", "-q:v", "2", frame_tmp.name],
                capture_output=True, timeout=10,
            )
            vid_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", prefix="freeze_", delete=False)
            vid_tmp.close()
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", frame_tmp.name,
                "-t", str(duration),
                *_codec_args(),
                *_pix_fmt(_enc_name()),
                "-r", "25",
                "-an",
                vid_tmp.name,
            ]
            subprocess.run(cmd, capture_output=True, timeout=30)
            Path(frame_tmp.name).unlink(missing_ok=True)
            if Path(vid_tmp.name).is_file() and Path(vid_tmp.name).stat().st_size > 0:
                freeze_clip = TimelineClip(
                    path=vid_tmp.name, name=f"{clip.name}_freeze",
                    start_time=clip.start_time + clip.duration,
                    duration=duration, media_duration=duration,
                )
                track.clips.insert(idx + 1, freeze_clip)
                self._multitrack._sync_canvas()
                self._multitrack.clips_changed.emit()
                self._status_label.setText(f"Freeze frame: {duration}s")
            else:
                self._status_label.setText("Freeze frame failed")
        except Exception as exc:
            self._status_label.setText(f"Freeze error: {exc}")

    # -- Crossfade -----------------------------------------------------------

    def _on_crossfade(self) -> None:
        """Set crossfade duration with the next clip."""
        track, idx = self._get_selected()
        if not track or idx < 0 or idx >= len(track.clips) - 1:
            self._status_label.setText("Select a clip that has a next clip")
            return
        duration, ok = QInputDialog.getDouble(
            self, "Crossfade", "Crossfade duration (seconds):",
            value=0.5, min=0.1, max=5.0, decimals=1)
        if not ok:
            return
        clip = track.clips[idx]
        # Find the next clip by timeline position, not index
        sorted_clips = sorted(enumerate(track.clips), key=lambda x: x[1].start_time)
        next_clip = None
        for i, (ci, c) in enumerate(sorted_clips):
            if ci == idx and i + 1 < len(sorted_clips):
                next_clip = sorted_clips[i + 1][1]
                break
        if next_clip is None:
            return
        clip.fade_out = duration
        next_clip.fade_in = duration
        next_clip.transition_in = {"type": "crossfade", "duration": duration}
        next_clip.start_time = clip.start_time + clip.duration - duration
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._status_label.setText(f"Crossfade: {duration}s")

    # -- Color match ---------------------------------------------------------

    def _on_color_match(self) -> None:
        """Send selected clip to Color Correct with the previous clip's last frame as reference."""
        track, idx = self._get_selected()
        if not track or idx < 0:
            self._status_label.setText("No clip selected")
            return
        clip = track.clips[idx]
        self.send_clip_to.emit("cc", clip.path)
        self._status_label.setText(f"Sent to Color Correct: {clip.name}")

    # -- Disable/Enable clip -------------------------------------------------

    def _on_disable_clip(self) -> None:
        """Toggle the selected clip's muted state."""
        track, idx = self._get_selected()
        if not track or idx < 0:
            self._status_label.setText("No clip selected")
            return
        clip = track.clips[idx]
        clip.muted = not getattr(clip, "muted", False)
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        state = "muted" if clip.muted else "unmuted"
        self._status_label.setText(f"Clip {state}: {clip.name}")

    # -- Snap to playhead ----------------------------------------------------

    def _on_snap_to_playhead(self) -> None:
        """Snap the selected clip's start to the current playhead position."""
        track, idx = self._get_selected()
        if not track or idx < 0:
            self._status_label.setText("No clip selected")
            return
        clip = track.clips[idx]
        clip.start_time = self._multitrack.playhead
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._status_label.setText(f"Snapped to playhead: {clip.name}")

    # -- Context menu edit actions (track-aware) -----------------------------

    def _split_clip_on_track(self, track_id: str, clip_idx: int) -> None:
        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        clip = track.clips[clip_idx]
        ph = self._multitrack.playhead
        offset = ph - clip.start_time
        if 0.05 < offset < clip.duration - 0.05:
            self._do_split(track, clip_idx, offset)
        else:
            self._status_label.setText("Playhead not within this clip")

    def _reverse_clip_on_track(self, track_id: str, clip_idx: int) -> None:
        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        self._do_reverse(track, clip_idx, track.clips[clip_idx])

    def _speed_clip_on_track(self, track_id: str, clip_idx: int) -> None:
        from sdqt.tabs.timeline_dialogs import SpeedDialog
        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        dlg = SpeedDialog(clip_duration=track.clips[clip_idx].duration, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        speed = dlg.speed_value()
        if abs(speed - 1.0) < 0.01:
            return
        self._apply_speed(track, clip_idx, track.clips[clip_idx], speed)

    def _rotate_clip_on_track(self, track_id: str, clip_idx: int, direction: str) -> None:
        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        clip = track.clips[clip_idx]
        delta = 90 if direction == "cw" else -90
        clip.rotation = (clip.rotation + delta) % 360
        logger.info("Rotate clip %s %s → %d°", clip.name, direction, clip.rotation)
        self._multitrack.update()
        self._multitrack.clips_changed.emit()
        self._status_label.setText(f"Rotation: {clip.rotation}°")

    def _resize_clip_dialog(self, track_id: str, clip_idx: int) -> None:
        """Open Resize dialog and re-encode clip to chosen dimensions."""
        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        clip = track.clips[clip_idx]

        import subprocess
        import json as _json

        cur_w, cur_h = 0, 0
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_streams", clip.path],
                capture_output=True, text=True, timeout=5,
            )
            data = _json.loads(r.stdout)
            vs = next(s for s in data["streams"] if s["codec_type"] == "video")
            cur_w = int(vs["width"])
            cur_h = int(vs["height"])
        except Exception as exc:
            logger.warning("Could not probe clip resolution: %s", exc)

        from sdqt.tabs.timeline_dialogs import ResizeDialog
        dlg = ResizeDialog(current_w=cur_w, current_h=cur_h, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        tw = (dlg.target_width() // 2) * 2
        th = (dlg.target_height() // 2) * 2
        mode = dlg.scale_mode()

        if tw == cur_w and th == cur_h:
            self._status_label.setText("Already at target size")
            return

        self._status_label.setText(f"Resizing to {tw}x{th} ({mode})...")

        output = self._resize_output_path(clip, tw, th)

        from sdqt.utils.codec import (
            configured_codec_args, configured_encoder_name, pix_fmt_args,
            probe_color_range, _resolve_profile,
        )
        codec = configured_codec_args()
        in_range = probe_color_range(clip.path)
        out_range = _resolve_profile().output_range()

        if mode == "fit":
            vf = (
                f"scale={tw}:{th}:force_original_aspect_ratio=decrease:"
                f"in_range={in_range}:out_range={out_range},"
                f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
            )
        elif mode == "fill":
            vf = (
                f"scale={tw}:{th}:force_original_aspect_ratio=increase:"
                f"in_range={in_range}:out_range={out_range},"
                f"crop={tw}:{th},setsar=1"
            )
        else:
            vf = f"scale={tw}:{th}:in_range={in_range}:out_range={out_range},setsar=1"

        # Trim to the clip's media window (like _do_reverse) — the output
        # contains exactly this clip's frames, so media_offset=0 afterward is
        # correct. Re-encoding the full source and zeroing the offset made
        # split/trimmed clips play the head of the source instead.
        mo = getattr(clip, "media_offset", 0.0) or 0.0
        te = mo + clip.duration
        cmd = [
            "ffmpeg", "-y", "-i", clip.path,
            "-vf", f"trim={mo}:{te},setpts=PTS-STARTPTS,{vf}",
            "-af", f"atrim={mo}:{te},asetpts=PTS-STARTPTS",
            *codec,
            "-c:a", "aac", "-b:a", "192k",
            *pix_fmt_args(configured_encoder_name()),
            output,
        ]

        from sdqt.workers.base import BaseWorker

        class _ResizeWorker(BaseWorker):
            def __init__(self, cmd, parent=None):
                super().__init__(parent)
                self._cmd = cmd
            def do_work(self):
                result = subprocess.run(self._cmd, capture_output=True, text=True, timeout=600)
                if result.returncode != 0:
                    err = result.stderr[-300:] if result.stderr else "ffmpeg failed"
                    raise RuntimeError(err)
                return output

        worker = _ResizeWorker(cmd, parent=self)
        worker.progress.connect(lambda f, d: self._status_label.setText(d))
        worker.finished_ok.connect(
            lambda out_path: self._on_resize_done(clip, out_path, tw, th)
        )
        worker.error.connect(lambda msg: self._status_label.setText(f"Resize failed: {msg[:80]}"))
        worker.finished.connect(worker.deleteLater)
        self._resize_worker = worker
        worker.start()

    def _resize_clip_to_neighbor(self, track_id: str, clip_idx: int,
                                 direction: str = "left") -> None:
        """Re-encode clip to match the resolution/fps of its neighbor.

        *direction* is ``"left"`` or ``"right"`` — the neighbor by timeline
        position on the same track.
        """
        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        clip = track.clips[clip_idx]
        # Find the neighbor by timeline position, not index
        sorted_clips = sorted(enumerate(track.clips), key=lambda x: x[1].start_time)
        neighbor = None
        for i, (ci, c) in enumerate(sorted_clips):
            if ci == clip_idx:
                if direction == "left" and i > 0:
                    neighbor = sorted_clips[i - 1][1]
                elif direction == "right" and i + 1 < len(sorted_clips):
                    neighbor = sorted_clips[i + 1][1]
                break
        if neighbor is None:
            self._status_label.setText(
                f"No clip to the {direction} of this one on its track."
            )
            return

        import subprocess
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_streams", neighbor.path],
                capture_output=True, text=True, timeout=5,
            )
            data = __import__("json").loads(r.stdout)
            vs = next(s for s in data["streams"] if s["codec_type"] == "video")
            tw = int(vs["width"])
            th = int(vs["height"])
            rfr = vs.get("r_frame_rate", "25/1")
            n, d = rfr.split("/")
            tfps = round(int(n) / max(1, int(d)), 2)
        except Exception as exc:
            self._status_label.setText(f"Could not probe neighbor clip: {exc}")
            return

        self._status_label.setText(f"Resizing to {tw}x{th} @ {tfps}fps...")

        output = self._resize_output_path(clip, tw, th)

        from sdqt.utils.codec import (
            configured_codec_args, configured_encoder_name, pix_fmt_args,
            probe_color_range, _resolve_profile,
        )
        codec = configured_codec_args()
        in_range = probe_color_range(clip.path)
        out_range = _resolve_profile().output_range()

        # Trim to the clip's media window (like _do_reverse) — the output
        # contains exactly this clip's frames, so media_offset=0 afterward is
        # correct. Re-encoding the full source and zeroing the offset made
        # split/trimmed clips play the head of the source instead.
        mo = getattr(clip, "media_offset", 0.0) or 0.0
        te = mo + clip.duration
        cmd = [
            "ffmpeg", "-y", "-i", clip.path,
            "-vf", (
                f"trim={mo}:{te},setpts=PTS-STARTPTS,"
                f"scale={tw}:{th}:in_range={in_range}:out_range={out_range},"
                f"fps={tfps},setsar=1"
            ),
            "-af", f"atrim={mo}:{te},asetpts=PTS-STARTPTS",
            *codec,
            "-c:a", "aac", "-b:a", "192k",
            *pix_fmt_args(configured_encoder_name()),
            output,
        ]

        from sdqt.workers.base import BaseWorker

        class _ResizeWorker(BaseWorker):
            def __init__(self, cmd, parent=None):
                super().__init__(parent)
                self._cmd = cmd
            def do_work(self):
                result = subprocess.run(self._cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    err = result.stderr[-300:] if result.stderr else "ffmpeg failed"
                    raise RuntimeError(err)
                return output

        worker = _ResizeWorker(cmd, parent=self)
        worker.progress.connect(lambda f, d: self._status_label.setText(d))
        worker.finished_ok.connect(
            lambda out_path: self._on_resize_done(clip, out_path, tw, th)
        )
        worker.error.connect(lambda msg: self._status_label.setText(f"Resize failed: {msg[:80]}"))
        worker.finished.connect(worker.deleteLater)
        self._resize_worker = worker
        worker.start()

    def _resize_output_path(self, clip, tw: int, th: int) -> str:
        """Unique output path for a resize — never reuse an existing filename.

        Split/duplicate siblings share a source stem; overwriting an existing
        ``{stem}_{w}x{h}.mp4`` would silently swap the media of whichever clip
        already references it.
        """
        out_dir = self.project_path / "clips" / "resized"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = cap_stem(Path(clip.path).stem)
        dest = out_dir / f"{stem}_{tw}x{th}.mp4"
        n = 1
        while dest.exists():
            dest = out_dir / f"{stem}_{tw}x{th}_{n}.mp4"
            n += 1
        return str(dest)

    def _on_resize_done(self, clip, out_path: str, tw: int, th: int) -> None:
        # Re-locate the clip by object identity — a (track, index) pair goes
        # stale while the encode runs (split/delete/move would retarget the
        # index and swap another clip's media).
        on_timeline = any(
            c is clip for t in self._multitrack.tracks for c in t.clips
        )
        if not on_timeline:
            self._status_label.setText(
                "Resize finished, but the clip is no longer on the timeline."
            )
            return
        orig_duration = clip.duration
        out_path = self._persist_clip(out_path)
        clip.path = out_path
        clip.media_offset = 0.0
        clip.media_duration = self._probe_duration(out_path)
        clip.duration = min(orig_duration, clip.media_duration)
        clip.name = Path(out_path).stem
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._save_state()
        self._status_label.setText(f"Resized to {tw}x{th}: {clip.name}")

    def _effects_clip_on_track(self, track_id: str, clip_idx: int) -> None:
        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        clip = track.clips[clip_idx]
        if hasattr(self, "_effects_panel") and self._effects_panel is not None:
            self._effects_panel.close()
            self._effects_panel = None
        from sdqt.widgets.effects_manager import EffectsManagerPanel
        panel = EffectsManagerPanel(clip, parent=self)
        panel.effects_changed.connect(lambda: self._on_effects_changed())
        panel.preview_changed.connect(self._on_effects_preview)
        panel.show()
        self._effects_panel = panel

    def _postprocess_clip_on_track(self, track_id: str, clip_idx: int) -> None:
        """Open the Post Processing dialog for a clip."""
        from sdqt.tabs.timeline_dialogs import PostProcessDialog

        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        clip = track.clips[clip_idx]

        dlg = PostProcessDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            return

        temporal = dlg.temporal()
        spatial = dlg.spatial()
        grain_intensity = dlg.grain_intensity()
        grain_saturation = dlg.grain_saturation()
        ai_enhance = dlg.ai_enhance()
        ai_denoise = dlg.ai_denoise()
        ai_sharpen = dlg.ai_sharpen()
        ai_face_restore = dlg.ai_face_restore()
        ai_tile_size = dlg.ai_tile_size()

        has_any_ai = ai_enhance or ai_denoise or ai_sharpen or ai_face_restore
        if not temporal and not spatial and grain_intensity <= 0 and not has_any_ai:
            self._status_label.setText("No post-processing selected")
            return

        self._status_label.setText("Post processing...")

        from sdqt.workers.base import BaseWorker

        class _PPWorker(BaseWorker):
            def __init__(self, clip_path, mo, dur, temporal, spatial,
                         grain_i, grain_s, ai_enhance=False, ai_denoise=False,
                         ai_sharpen=False, ai_face_restore=False,
                         ai_tile=512, upscaler_dir="", face_models_dir="",
                         parent=None):
                super().__init__(parent)
                self._clip_path = clip_path
                self._mo = mo
                self._dur = dur
                self._temporal = temporal
                self._spatial = spatial
                self._grain_i = grain_i
                self._grain_s = grain_s
                self._ai_enhance = ai_enhance
                self._ai_denoise = ai_denoise
                self._ai_sharpen = ai_sharpen
                self._ai_face_restore = ai_face_restore
                self._ai_tile = ai_tile
                self._upscaler_dir = upscaler_dir
                self._face_models_dir = face_models_dir

            def do_work(self):
                import tempfile, shutil
                import numpy as np
                from PIL import Image
                from supremediffusion.utils.video import extract_frames, probe_video
                from supremediffusion.postprocessing import PostProcessor

                info = probe_video(self._clip_path)
                fps = int(round(info.get("fps", 16)))

                start_frame = int(round(self._mo * fps)) if self._mo else 0
                end_frame = int(round((self._mo + self._dur) * fps)) if self._dur else None

                self.progress.emit(0.0, "Extracting frames...")
                tmp_dir = tempfile.mkdtemp(prefix="sdqt_pp_")
                try:
                    frame_paths = extract_frames(
                        self._clip_path, tmp_dir,
                        start_frame=start_frame, end_frame=end_frame,
                    )
                    if not frame_paths:
                        raise RuntimeError("No frames extracted")
                    frames = np.array([np.array(Image.open(p)) for p in frame_paths])
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

                # AI processing steps (before traditional post-processing)
                from supremediffusion.postprocessing.ai_upscale import (
                    upscale_frames, denoise_frames, sharpen_frames,
                    face_restore_frames, _resolve_model,
                    DEFAULT_UPSCALER, DEFAULT_DENOISER, DEFAULT_SHARPENER,
                )

                upscaler_dir = self._upscaler_dir
                if not upscaler_dir:
                    upscaler_dir = str(Path.home() / ".supremediffusion" / "models" / "upscalers")

                ai_steps = sum([self._ai_denoise, self._ai_sharpen,
                                self._ai_enhance, self._ai_face_restore])
                ai_done = 0
                ai_frac = 0.6 if ai_steps > 0 else 0.0  # Reserve 60% for AI

                def _ai_progress(f, d, step=0):
                    base = (step / ai_steps) * ai_frac if ai_steps else 0
                    span = ai_frac / ai_steps if ai_steps else 0
                    self.progress.emit(base + f * span, d)

                # 1) Denoise first (clean up before other processing)
                if self._ai_denoise:
                    self.progress.emit(ai_done / max(ai_steps, 1) * ai_frac, "AI Denoise: loading...")
                    model_path = _resolve_model(
                        upscaler_dir, DEFAULT_DENOISER,
                        progress_cb=lambda f, d: self.progress.emit(0, d),
                    )
                    step = ai_done
                    frames = denoise_frames(
                        frames, model_path, tile_size=self._ai_tile,
                        progress_cb=lambda f, d, s=step: _ai_progress(f, d, s),
                    )
                    ai_done += 1

                # 2) Sharpen (after denoise, before upscale)
                if self._ai_sharpen:
                    self.progress.emit(ai_done / max(ai_steps, 1) * ai_frac, "AI Sharpen: loading...")
                    model_path = _resolve_model(
                        upscaler_dir, DEFAULT_SHARPENER,
                        progress_cb=lambda f, d: self.progress.emit(0, d),
                    )
                    step = ai_done
                    frames = sharpen_frames(
                        frames, model_path, tile_size=self._ai_tile,
                        progress_cb=lambda f, d, s=step: _ai_progress(f, d, s),
                    )
                    ai_done += 1

                # 3) Upscale (changes resolution — do after denoise/sharpen)
                if self._ai_enhance:
                    self.progress.emit(ai_done / max(ai_steps, 1) * ai_frac, "AI Upscale: loading...")
                    model_path = _resolve_model(
                        upscaler_dir, DEFAULT_UPSCALER,
                        progress_cb=lambda f, d: self.progress.emit(0, d),
                    )
                    step = ai_done
                    frames = upscale_frames(
                        frames, model_path, tile_size=self._ai_tile,
                        progress_cb=lambda f, d, s=step: _ai_progress(f, d, s),
                    )
                    ai_done += 1

                # 4) Face restore (last — operates on final resolution)
                if self._ai_face_restore:
                    self.progress.emit(ai_done / max(ai_steps, 1) * ai_frac, "Face Restore: loading...")
                    face_dir = self._face_models_dir
                    gfpgan_path = ""
                    if face_dir:
                        gfpgan = Path(face_dir) / "GFPGANv1.4.pth"
                        if gfpgan.is_file():
                            gfpgan_path = str(gfpgan)
                    if not gfpgan_path:
                        # Try upscaler dir as fallback
                        gfpgan_path = _resolve_model(
                            upscaler_dir, "GFPGANv1.4.pth",
                            progress_cb=lambda f, d: self.progress.emit(0, d),
                        )
                    step = ai_done
                    frames = face_restore_frames(
                        frames, gfpgan_path, tile_size=0,
                        progress_cb=lambda f, d, s=step: _ai_progress(f, d, s),
                    )
                    ai_done += 1

                pp = PostProcessor()
                pp_start = ai_frac
                pp_range = 0.8 - pp_start
                frames, out_fps = pp.process(
                    frames, fps,
                    temporal_upsampling=self._temporal,
                    spatial_upsampling=self._spatial,
                    film_grain_intensity=self._grain_i,
                    film_grain_saturation=self._grain_s,
                    progress_cb=lambda f, d: self.progress.emit(pp_start + f * pp_range, d),
                )

                self.progress.emit(0.85, "Encoding...")
                from supremediffusion.utils.video import encode_video
                out_dir = Path(self._clip_path).parent
                stem = Path(self._clip_path).stem
                suffix = ""
                if self._ai_denoise:
                    suffix += "_dn"
                if self._ai_sharpen:
                    suffix += "_sharp"
                if self._ai_enhance:
                    suffix += "_4x"
                if self._ai_face_restore:
                    suffix += "_face"
                if self._temporal:
                    suffix += f"_{self._temporal}"
                if self._spatial:
                    suffix += f"_{self._spatial}"
                if self._grain_i > 0:
                    suffix += "_grain"
                output = str(out_dir / f"{cap_stem(stem)}{suffix}.mp4")
                encode_video(frames, output, fps=out_fps)
                self.progress.emit(1.0, "Done")
                return output, float(len(frames)) / out_fps

        mo = getattr(clip, "media_offset", 0.0) or 0.0
        upscaler_dir = ""
        face_models_dir = ""
        if hasattr(self, "state") and self.state:
            upscaler_dir = self.state.global_config.model_paths.get("upscaler_dir", "")
            face_models_dir = self.state.global_config.model_paths.get("face_models_dir", "")
        worker = _PPWorker(
            clip.path, mo, clip.duration,
            temporal, spatial, grain_intensity, grain_saturation,
            ai_enhance=ai_enhance, ai_denoise=ai_denoise,
            ai_sharpen=ai_sharpen, ai_face_restore=ai_face_restore,
            ai_tile=ai_tile_size,
            upscaler_dir=upscaler_dir, face_models_dir=face_models_dir,
            parent=self,
        )
        worker.progress.connect(lambda f, d: self._status_label.setText(d))
        worker.finished_ok.connect(
            lambda result: self._on_postprocess_done(track_id, clip_idx, result)
        )
        worker.error.connect(lambda msg: self._status_label.setText(f"Post-proc error: {msg[:80]}"))
        worker.finished.connect(worker.deleteLater)
        self._postproc_worker = worker
        worker.start()

    def _on_postprocess_done(self, track_id: str, clip_idx: int, result) -> None:
        from PySide6.QtWidgets import QMessageBox
        import shutil

        out_path, dur = result
        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        clip = track.clips[clip_idx]
        original_path = clip.path

        box = QMessageBox(self)
        box.setWindowTitle("Save Post-Processed Clip")
        box.setText(f"Post-processing complete.\n\n{Path(out_path).name}")
        box.setInformativeText("Save as a copy or overwrite the original?")
        copy_btn = box.addButton("Save as Copy", QMessageBox.AcceptRole)
        overwrite_btn = box.addButton("Overwrite Original", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()

        clicked = box.clickedButton()
        if clicked == overwrite_btn:
            shutil.move(out_path, original_path)
            clip.duration = dur
            clip.media_duration = dur
            clip.media_offset = 0.0
            clip.name = Path(original_path).stem
        elif clicked == copy_btn:
            out_path = self._persist_clip(out_path)
            clip.path = out_path
            clip.duration = dur
            clip.media_duration = dur
            clip.media_offset = 0.0
            clip.name = Path(out_path).stem
            self._library.add_clip(out_path, dur)
        else:
            # Cancelled
            self._status_label.setText("Post-processing cancelled.")
            return

        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._save_state()
        self._status_label.setText(f"Post-processed: {clip.name}")

    def _lipsync_clip_on_track(self, track_id: str, clip_idx: int, engine: str = "musetalk") -> None:
        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        clip = track.clips[clip_idx]
        logger.info("Lip Sync (%s) dialog opened for clip %s (track=%s idx=%d)", engine, clip.name, track_id, clip_idx)

        # Auto-bake effects so lip sync sees what the GL preview shows
        source_path = clip.path
        active_effects = [fx for fx in clip.effects if fx.get("enabled", True)]
        if active_effects:
            source_path = self._bake_for_lipsync(clip)
            if source_path is None:
                self._status_label.setText("Failed to bake effects for lip sync")
                return
            logger.info("Auto-baked effects for lip sync: %s", source_path)

        if engine == "ltx_a2vid":
            from sdqt.widgets.lipsync_ltx_dialog import LipSyncLTXDialog
            dlg = LipSyncLTXDialog(source_path, self.state, parent=self)
        elif engine == "vace_multitalk":
            from sdqt.widgets.lipsync_vace_multitalk_dialog import LipSyncVaceMultitalkDialog
            dlg = LipSyncVaceMultitalkDialog(source_path, self.state, parent=self)
        elif engine == "latentsync":
            from sdqt.widgets.lipsync_latentsync_dialog import LipSyncLatentSyncDialog
            dlg = LipSyncLatentSyncDialog(source_path, self.state, parent=self)
        elif engine == "video_retalking":
            from sdqt.widgets.lipsync_video_retalking_dialog import LipSyncVideoRetalkingDialog
            dlg = LipSyncVideoRetalkingDialog(source_path, self.state, parent=self)
        else:
            from sdqt.widgets.musetalk_dialog import MuseTalkClipDialog
            # MuseTalkClipDialog takes the clip object, update path temporarily
            orig_path = clip.path
            clip.path = source_path
            dlg = MuseTalkClipDialog(clip, self.state, self.project_path, parent=self)
            clip.path = orig_path
        if dlg.exec() == QDialog.Accepted and dlg.result_path and Path(dlg.result_path).is_file():
            result_path = self._persist_clip(dlg.result_path)
            clip.path = result_path
            clip.media_offset = 0.0
            clip.media_duration = clip.duration
            clip.name = Path(result_path).stem
            self._multitrack._sync_canvas()
            self._multitrack.clips_changed.emit()
            dur = self._probe_duration(result_path)
            thumb = self._generate_thumbnail(result_path)
            self._library.add_clip(result_path, dur, thumb)
            self._save_state()
            self._status_label.setText(f"Lip sync applied: {clip.name}")

    def _bake_for_lipsync(self, clip) -> str | None:
        """Render effects into a temp file for lip sync input."""
        import subprocess
        import tempfile
        from sdqt.workers.timeline import _build_effects_filter
        from sdqt.utils.codec import (
            configured_codec_args, pix_fmt_args, project_input_filter,
        )

        vf = _build_effects_filter(clip.effects)
        if not vf:
            return clip.path
        # Probe-aware range/matrix conversion — full_range_filter() assumed
        # full-range input and washed out tv-range clips.
        in_filt = project_input_filter(source_path=clip.path, add_setsar=False,
                                       add_matrix=True)
        vf = f"{in_filt},{vf.lstrip(',')}"

        tmp = tempfile.NamedTemporaryFile(
            suffix=".mp4", prefix="lipsync_baked_", delete=False,
        )
        tmp.close()

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(clip.media_offset), "-i", clip.path,
            "-t", str(clip.duration),
            "-vf", vf,
            *configured_codec_args(),
            *pix_fmt_args(),
            "-c:a", "aac", "-b:a", "192k",
            tmp.name,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode == 0 and Path(tmp.name).is_file():
                return tmp.name
            logger.warning("Bake for lipsync failed: %s",
                           result.stderr.decode(errors="replace")[:200])
        except Exception as exc:
            logger.warning("Bake for lipsync error: %s", exc)
        return None

    def _faceswap_clip_on_track(self, track_id: str, clip_idx: int) -> None:
        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        clip = track.clips[clip_idx]
        logger.info("Face Swap dialog opened for clip %s (track=%s idx=%d)",
                     clip.name, track_id, clip_idx)

        from sdqt.widgets.faceswap_clip_dialog import FaceSwapClipDialog
        dlg = FaceSwapClipDialog(clip, self.state, self.project_path, parent=self)
        if dlg.exec() == QDialog.Accepted and dlg.result_path and Path(dlg.result_path).is_file():
            result_path = self._persist_clip(dlg.result_path)
            clip.path = result_path
            clip.media_offset = 0.0
            clip.media_duration = clip.duration
            clip.name = Path(result_path).stem
            self._multitrack._sync_canvas()
            self._multitrack.clips_changed.emit()
            dur = self._probe_duration(result_path)
            thumb = self._generate_thumbnail(result_path)
            self._library.add_clip(result_path, dur, thumb)
            self._save_state()
            self._status_label.setText(f"Face swap applied: {clip.name}")

    def _sequence_character_replacement(self, track_id: str, clip_idx: int) -> None:
        """Launch the Character Replacement wizard with the clip's video pre-loaded."""
        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        clip = track.clips[clip_idx]
        logger.info("Character Replacement wizard for clip %s (track=%s idx=%d)",
                     clip.name, track_id, clip_idx)

        # Export the trimmed clip range (respects media_offset + duration)
        mo = getattr(clip, "media_offset", 0.0) or 0.0
        video_path = self._export_trimmed_clip(clip.path, mo, clip.duration, clip=clip)

        from sdqt.sequences.character_replacement import CharacterReplacementWizard
        wizard = CharacterReplacementWizard(self.state, video_path=video_path, parent=self)
        wizard.exec()

        # If the wizard produced an output, update the clip
        if wizard._rs.output_path and Path(wizard._rs.output_path).is_file():
            result_path = self._persist_clip(wizard._rs.output_path)
            clip.path = result_path
            clip.media_offset = 0.0
            clip.media_duration = clip.duration
            clip.name = Path(result_path).stem
            self._multitrack._sync_canvas()
            self._multitrack.clips_changed.emit()
            dur = self._probe_duration(result_path)
            thumb = self._generate_thumbnail(result_path)
            self._library.add_clip(result_path, dur, thumb)
            self._save_state()
            self._status_label.setText(f"Character replacement applied: {clip.name}")

    # -- Keyboard-driven clip actions ----------------------------------------

    def _nudge_selected_clip(self, forward: bool) -> None:
        """Nudge the selected clip by one frame in the given direction."""
        track_id = self._multitrack._canvas._selected_track_id
        clip_idx = self._multitrack._canvas._selected_clip_idx
        if not track_id or clip_idx < 0:
            return
        track = self._multitrack.get_track(track_id)
        if not track or track.locked or clip_idx >= len(track.clips):
            return
        clip = track.clips[clip_idx]
        fps = getattr(self._multitrack._canvas, "_project_fps", 25)
        step = 1.0 / fps
        new_start = clip.start_time + (step if forward else -step)
        new_start = max(0.0, new_start)
        clip.start_time = new_start
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()

    def _duplicate_selected_clip(self) -> None:
        """Duplicate the selected clip, placing the copy right after it."""
        track_id = self._multitrack._canvas._selected_track_id
        clip_idx = self._multitrack._canvas._selected_clip_idx
        if not track_id or clip_idx < 0:
            return
        track = self._multitrack.get_track(track_id)
        if not track or clip_idx >= len(track.clips):
            return
        import copy
        clip = track.clips[clip_idx]
        dup = copy.deepcopy(clip)
        dup.start_time = clip.start_time + clip.duration
        dup.name = f"{clip.name} (copy)"
        track.clips.insert(clip_idx + 1, dup)
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._status_label.setText(f"Duplicated: {clip.name}")

    def _split_selected_at_playhead(self) -> None:
        """Split the selected clip (or clip under playhead) at the playhead position."""
        track_id = self._multitrack._canvas._selected_track_id
        clip_idx = self._multitrack._canvas._selected_clip_idx
        if track_id and clip_idx >= 0:
            self._split_clip_on_track(track_id, clip_idx)
            return
        # If no selection, try the clip under the playhead
        ph = self._multitrack.playhead
        for track in self._multitrack.tracks:
            for i, c in enumerate(track.clips):
                if c.start_time < ph < c.start_time + c.duration:
                    self._split_clip_on_track(track.id, i)
                    return

    def _toggle_clip_mute(self, track_id: str, clip_idx: int) -> None:
        """Toggle mute on a single clip."""
        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        clip = track.clips[clip_idx]
        clip.muted = not getattr(clip, "muted", False)
        state = "muted" if clip.muted else "unmuted"
        self._status_label.setText(f"{clip.name}: {state}")
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()

    def _ripple_delete_clip(self, track_id: str, clip_idx: int) -> None:
        """Remove a clip and shift all subsequent clips left to close the gap."""
        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        clip = track.clips[clip_idx]
        gap_start = clip.start_time
        gap_dur = clip.duration
        track.clips.pop(clip_idx)
        track.ripple_shift(gap_start, -gap_dur)
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._status_label.setText(f"Ripple deleted: {clip.name}")

    def _separate_audio(self, track_id: str, clip_idx: int, audio_track_id: str) -> None:
        """Separate audio from a video clip onto an audio track."""
        from sdqt.widgets.timeline_track import TimelineClip, TrackType

        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        clip = track.clips[clip_idx]

        if audio_track_id:
            audio_track = self._multitrack.get_track(audio_track_id)
        else:
            audio_track = self._multitrack.add_track(TrackType.AUDIO, name="Audio")

        if not audio_track:
            return

        try:
            from sdqt.utils.waveform import extract_waveform
            waveform = extract_waveform(clip.path)
        except Exception:
            waveform = None

        audio_clip = TimelineClip(
            path=clip.path,
            duration=clip.duration,
            start_time=clip.start_time,
            name=f"{clip.name} (audio)",
            has_audio=True,
            waveform=waveform,
            media_offset=clip.media_offset,
            media_duration=clip.media_duration,
            volume=clip.volume,
        )
        audio_track.clips.append(audio_clip)

        clip.has_audio = False
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._status_label.setText(f"Audio separated: {clip.name}")
