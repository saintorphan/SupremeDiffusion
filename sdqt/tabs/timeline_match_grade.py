"""Timeline match grade mixin — color grading analysis and application."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QVBoxLayout, QHBoxLayout, QWidget,
)

from sdqt.utils.naming import cap_stem
from sdqt.workers.base import BaseWorker

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


# ── Region selector widget ──────────────────────────────────────────────────

class _RegionSelector(QLabel):
    """QLabel that lets the user draw a rectangle on an image."""

    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self._base_pixmap = pixmap
        self.setPixmap(pixmap)
        self.setFixedSize(pixmap.size())
        self._origin: QPoint | None = None
        self._rect: QRect | None = None
        self._drawing = False

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._origin = ev.position().toPoint()
            self._rect = QRect(self._origin, self._origin)
            self._drawing = True

    def mouseMoveEvent(self, ev):
        if self._drawing and self._origin is not None:
            self._rect = QRect(self._origin, ev.position().toPoint()).normalized()
            self._update_overlay()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._drawing = False

    def _update_overlay(self):
        pix = self._base_pixmap.copy()
        p = QPainter(pix)
        p.setPen(QPen(QColor(0, 255, 0, 200), 2))
        p.setBrush(QColor(0, 255, 0, 40))
        if self._rect and self._rect.width() > 4 and self._rect.height() > 4:
            p.drawRect(self._rect)
        p.end()
        self.setPixmap(pix)

    def normalized_rect(self) -> tuple[float, float, float, float] | None:
        """Return (x, y, w, h) as 0–1 fractions, or None if no selection."""
        if not self._rect or self._rect.width() < 5 or self._rect.height() < 5:
            return None
        pw = self._base_pixmap.width()
        ph = self._base_pixmap.height()
        r = self._rect
        return (
            max(0, r.x()) / pw,
            max(0, r.y()) / ph,
            min(r.width(), pw - r.x()) / pw,
            min(r.height(), ph - r.y()) / ph,
        )


class MatchGradeDialog(QDialog):
    """Dialog to select a reference region for Match Grade."""

    def __init__(self, ref_frame: "np.ndarray", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Match Grade — Select Reference Region")
        self.setMinimumSize(700, 700)
        import numpy as np

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        hint = QLabel("Draw a rectangle over the area to color-match (e.g. skin).\n"
                       "Leave empty to use the full frame.")
        hint.setStyleSheet("color: #aaa; font-style: italic; font-size: 13px; padding-bottom: 4px;")
        layout.addWidget(hint)

        # Scale frame for display (max 640px wide)
        h, w = ref_frame.shape[:2]
        scale = min(640 / w, 480 / h, 1.0)
        dw, dh = int(w * scale), int(h * scale)
        from PIL import Image
        pil = Image.fromarray(ref_frame).resize((dw, dh), Image.LANCZOS)
        qimg = QImage(pil.tobytes(), dw, dh, dw * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self._selector = _RegionSelector(pixmap)
        self._scale = scale
        layout.addWidget(self._selector, 0, Qt.AlignCenter)

        # Options row
        from PySide6.QtWidgets import QCheckBox, QRadioButton, QGroupBox
        opts = QHBoxLayout()
        opts.setSpacing(8)

        scope_box = QGroupBox("Scope")
        scope_lay = QVBoxLayout(scope_box)
        scope_lay.setContentsMargins(8, 6, 8, 6)
        scope_lay.setSpacing(8)
        self._all_rb = QRadioButton("All clips after")
        self._all_rb.setChecked(True)
        self._next_rb = QRadioButton("Next clip only")
        self._chain_rb = QRadioButton("Chain (progressive)")
        self._chain_rb.setToolTip(
            "Grade each clip from the one before it — "
            "the correction flows through the sequence"
        )
        scope_lay.addWidget(self._all_rb)
        scope_lay.addWidget(self._next_rb)
        scope_lay.addWidget(self._chain_rb)
        opts.addWidget(scope_box)

        options_box = QGroupBox("Options")
        options_lay = QVBoxLayout(options_box)
        options_lay.setContentsMargins(8, 6, 8, 6)
        options_lay.setSpacing(8)

        self._color_only = QCheckBox("Color only (skip brightness)")
        self._color_only.setToolTip("Only correct color cast — don't touch brightness, contrast, levels, or shadows/highlights")
        options_lay.addWidget(self._color_only)

        self._bake_permanent = QCheckBox("Bake permanently")
        self._bake_permanent.setChecked(True)
        self._bake_permanent.setToolTip(
            "Render the match-grade LUT into each clip's source file so the "
            "look survives export, combine, and send-to. Reversible via "
            "right-click → Unbake Effects on each clip."
        )
        options_lay.addWidget(self._bake_permanent)
        options_lay.addStretch()
        opts.addWidget(options_box)

        opts.addStretch()
        layout.addLayout(opts)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Ok).setText("Match Grade")
        layout.addWidget(buttons)

    def crop_rect(self) -> tuple[float, float, float, float] | None:
        """Normalized crop rect (0–1), or None for full frame."""
        return self._selector.normalized_rect()

    def next_only(self) -> bool:
        return self._next_rb.isChecked()

    def chain(self) -> bool:
        return self._chain_rb.isChecked()

    def color_only(self) -> bool:
        return self._color_only.isChecked()

    def bake_permanent(self) -> bool:
        return self._bake_permanent.isChecked()


# ── Worker ──────────────────────────────────────────────────────────────────

class _MatchAllToWorker(BaseWorker):
    """Background worker for Match All To.

    Uses color-matcher MKL optimal transport to generate a 3D LUT per clip
    that preserves the full nonlinear colour mapping.
    """

    def __init__(self, ref_clip, target_clips, lut_dir: str, tag: str = "_match_all_to",
                 ref_frame=None, parent=None):
        super().__init__(parent)
        self._ref_clip = ref_clip
        self._targets = target_clips
        self._lut_dir = lut_dir
        self._tag = tag
        # Optional precomputed reference frame (uint8 RGB). When given, it IS
        # the reference — used by "Match All to Playhead Frame", where the
        # user picks the exact frame under the playhead as the source look.
        self._ref_frame = ref_frame

    def do_work(self):
        from sdqt.utils.grade_compute import (
            sample_clip_grade_frames,
            generate_lut3d,
        )

        if self._ref_frame is not None:
            ref_frame = self._ref_frame
        else:
            # Robust whole-clip sample (not one frame) → the transform reflects
            # the reference clip's actual grade, not an arbitrary frame.
            # keep_match=True: the reference's own match LUT is part of the
            # look being matched to.
            ref_frame = sample_clip_grade_frames(self._ref_clip, keep_match=True)
        if ref_frame is None:
            raise RuntimeError(
                f"Failed to extract reference frame from {self._ref_clip.name}"
            )

        total = len(self._targets)
        if total == 0:
            return 0

        matched = 0

        for clip_num, (track_id, clip_idx, clip) in enumerate(self._targets):
            if self.is_aborted:
                break

            self.progress.emit(
                clip_num / total,
                f"Matching {clip.name} ({clip_num + 1}/{total})...",
            )

            target_frame = sample_clip_grade_frames(clip)
            if target_frame is None:
                continue

            lut_path = str(
                Path(self._lut_dir)
                / f"{clip.name}_match_all_{clip_num}.cube"
            )

            try:
                generate_lut3d(ref_frame, target_frame, lut_path)
            except Exception as exc:
                logger.warning(
                    "%s: LUT generation failed (%s), skipping", clip.name, exc,
                )
                continue

            TimelineMatchGradeMixin._apply_lut3d(
                clip, lut_path, tag=self._tag,
            )
            matched += 1

        return matched


class _MatchGradeWorker(BaseWorker):
    """Background worker for Match Grade — generates a 3D LUT per clip."""

    def __init__(self, ref_idx, clips, lut_dir: str, crop_rect=None,
                 target_indices=None, chain=False, direction="right",
                 parent=None):
        super().__init__(parent)
        self._ref_idx = ref_idx
        self._clips = clips
        self._lut_dir = lut_dir
        self._crop = crop_rect
        self._target_indices = target_indices or []
        self._chain = chain
        self._direction = direction

    def do_work(self):
        from sdqt.tabs.timeline_match_grade import TimelineMatchGradeMixin
        from sdqt.utils.grade_compute import generate_lut3d, sample_clip_grade_frames

        anchor_clip = self._clips[self._ref_idx]
        # Robust whole-clip sample of the reference's grade (crop-aware) — not
        # a single edge frame, which made the match depend on one arbitrary
        # frame. "Match Grade" means match the overall grade.
        # keep_match=True: match to the reference AS SEEN (including its own
        # match LUT), not its raw file color — otherwise correcting down a row
        # matches each clip to the ungraded original look.
        anchor_cropped = sample_clip_grade_frames(
            anchor_clip, crop=self._crop, keep_match=True,
        )
        if anchor_cropped is None:
            raise RuntimeError(f"Failed to extract reference frame from {anchor_clip.name}")

        total = len(self._target_indices)
        if total == 0:
            return 0

        matched = 0
        # In chain mode, the reference advances to each graded clip
        current_ref = anchor_cropped

        for step, i in enumerate(self._target_indices):
            if self.is_aborted:
                break
            target_clip = self._clips[i]
            clip_num = step + 1
            self.progress.emit(
                clip_num / total,
                f"Match Grade: clip {clip_num}/{total} -- {target_clip.name}",
            )

            target_cropped = sample_clip_grade_frames(target_clip, crop=self._crop)
            if target_cropped is None:
                continue

            lut_path = str(
                Path(self._lut_dir)
                / f"{target_clip.name}_grade_{step}.cube"
            )

            try:
                generate_lut3d(current_ref, target_cropped, lut_path)
            except Exception as exc:
                logger.warning(
                    "%s: LUT generation failed (%s), skipping",
                    target_clip.name, exc,
                )
                continue

            TimelineMatchGradeMixin._apply_lut3d(
                target_clip, lut_path, tag="_match_grade",
            )
            matched += 1

            # Chain mode: the next reference is THIS clip's newly-graded look —
            # sample it WITH the match LUT kept so the correction flows down the
            # sequence (progressive grade).
            if self._chain:
                chained = sample_clip_grade_frames(
                    target_clip, crop=self._crop, keep_match=True,
                )
                if chained is not None:
                    current_ref = chained

        return matched


class _NormalizeTrackWorker(BaseWorker):
    """Concatenate all clips on a track, run ffmpeg normalize, split back."""

    def __init__(self, clips, output_dir: str, smoothing: int = 30, parent=None):
        super().__init__(parent)
        self._clips = clips
        self._output_dir = output_dir
        self._smoothing = smoothing

    def do_work(self):
        import subprocess
        import tempfile
        from sdqt.utils.codec import (
            configured_codec_args, pix_fmt_args, probe_color_range,
            matrix_convert_filter, _resolve_profile,
        )
        from supremediffusion.utils.video import probe_video

        clips = self._clips
        if not clips:
            return []

        tmp_dir = Path(tempfile.mkdtemp(prefix="normalize_"))
        out_dir = Path(self._output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Probe each clip's real duration + size. Clips on one track are often
        # DIFFERENT codecs (HEVC + H.264) — the concat *demuxer* corrupts
        # mixed-codec input ("Invalid NAL unit size" → black/broken frames), so
        # we decode each clip independently with the concat *filter* instead.
        # Split points follow each file's actual duration (not the timeline
        # clip.duration, which may be a trim and would drift every later cut).
        durs: list[float] = []
        sizes: list[tuple[int, int]] = []
        for clip in clips:
            try:
                info = probe_video(clip.path)
                durs.append(float(info.get("duration", 0.0)) or 0.0)
                sizes.append((int(info.get("width", 0)), int(info.get("height", 0))))
            except Exception:
                durs.append(float(getattr(clip, "duration", 0.0)) or 0.0)
                sizes.append((0, 0))

        # Common working resolution = first valid clip's size. Clips on a track
        # are usually identical, so this only rescales the rare mismatch.
        tw, th = next(((w, h) for (w, h) in sizes if w and h), (0, 0))
        if not tw or not th:
            raise RuntimeError("Could not determine clip resolution for normalize")

        n = len(clips)

        # 1. Normalize pass — concat FILTER (decodes each clip independently, so
        #    mixed codecs are fine), scale/pad each to the common size, then
        #    smooth-normalize across the WHOLE sequence and re-encode. Video
        #    only; each clip's original audio is re-attached in the split step.
        self.progress.emit(0.10, "Normalizing colors...")
        normalized_video = str(tmp_dir / "normalized.mp4")

        inputs: list[str] = []
        for clip in clips:
            inputs += ["-i", clip.path]

        # Per-input range/matrix conversion into the active profile BEFORE
        # concat — each clip's real range/matrix is probed individually, so
        # mixed tv/full/bt601 clips all land in one convention. The old
        # post-concat full_range_filter() assumed every input was full-range,
        # which washed out normal tv-range clips.
        profile = _resolve_profile()
        out_r = profile.output_range()
        scale_parts = []
        for idx, clip in enumerate(clips):
            in_r = probe_color_range(clip.path)
            rng = f":in_range={in_r}:out_range={out_r}"
            # True matrix conversion (colorspace filter — scale's matrix
            # options are no-ops for YUV→YUV rescales).
            conv = matrix_convert_filter(clip.path, profile)
            conv = f"{conv}," if conv else ""
            scale_parts.append(
                f"[{idx}:v]{conv}"
                f"scale={tw}:{th}:force_original_aspect_ratio=decrease{rng},"
                f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{idx}]"
            )
        concat_in = "".join(f"[v{idx}]" for idx in range(n))
        filtergraph = (
            ";".join(scale_parts)
            + f";{concat_in}concat=n={n}:v=1:a=0[cat]"
            + f";[cat]normalize=smoothing={self._smoothing}:independence=0.3[out]"
        )
        cmd_norm = [
            "ffmpeg", "-y", "-v", "error",
            *inputs,
            "-filter_complex", filtergraph,
            "-map", "[out]",
            *configured_codec_args(),
            *pix_fmt_args(),
            "-an",
            normalized_video,
        ]
        proc = subprocess.run(cmd_norm, capture_output=True, timeout=1200)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Normalize failed: {proc.stderr.decode(errors='replace')[-300:]}"
            )

        if self.is_aborted:
            return []

        # 2. Split back into per-clip segments — RE-ENCODE each (a stream copy
        #    cuts on keyframes only, so a mid-GOP cut shows black/frozen frames
        #    until the next keyframe: the "black flashes" bug). Re-encoding
        #    starts every segment on a fresh keyframe = frame-accurate. Each
        #    clip's ORIGINAL audio is re-attached from its source file.
        self.progress.emit(0.80, "Splitting normalized video...")
        results = []
        offset = 0.0
        for i, clip in enumerate(clips):
            if self.is_aborted:
                break

            stem = Path(clip.path).stem
            while stem.endswith("_norm"):
                stem = stem[:-5]
            out_path = str(out_dir / f"{cap_stem(stem)}_norm.mp4")

            seg_dur = durs[i]
            cmd_split = [
                "ffmpeg", "-y", "-v", "error",
                "-ss", f"{offset:.4f}", "-i", normalized_video,
                "-i", clip.path,
            ]
            if seg_dur > 0:
                cmd_split += ["-t", f"{seg_dur:.4f}"]
            cmd_split += [
                "-map", "0:v:0", "-map", "1:a?",
                *configured_codec_args(),
                *pix_fmt_args(),
                "-c:a", "aac", "-b:a", "192k",
                out_path,
            ]
            proc = subprocess.run(cmd_split, capture_output=True, timeout=300)
            if proc.returncode != 0 or not Path(out_path).is_file():
                logger.warning(
                    "Split failed for clip %s: %s",
                    clip.name, proc.stderr.decode(errors="replace")[-200:],
                )
                results.append(None)
            else:
                results.append(out_path)

            offset += seg_dur
            self.progress.emit(
                0.80 + 0.20 * (i + 1) / n,
                f"Split {i + 1}/{n}",
            )

        return results


class TimelineMatchGradeMixin:
    """Mixin providing match grade methods for TimelineTab."""

    def _match_grade_from(self, track_id: str, ref_idx: int,
                          direction: str = "right") -> None:
        """Color-match clips in the given direction from the reference.

        direction: "right" → clips after, "left" → clips before.
        """
        import copy

        track = self._multitrack.get_track(track_id)
        if not track or ref_idx < 0 or ref_idx >= len(track.clips):
            return

        clips = track.clips
        ref_clip = clips[ref_idx]
        ref_start = ref_clip.start_time
        going_right = direction == "right"

        # Direction is defined by start_time ordering, NOT strict
        # non-overlap: bake/resize completions re-probe durations (frame
        # rounding, audio padding) and can grow a clip a few ms over its
        # snapped neighbor — requiring "ends before ref starts" then
        # disqualifies a visually adjacent clip ("no clips before it").
        if going_right:
            # Clips starting after the reference, sorted left-to-right
            target_indices = sorted(
                [i for i, c in enumerate(clips)
                 if i != ref_idx and c.start_time > ref_start + 0.01],
                key=lambda i: clips[i].start_time,
            )
        else:
            # Clips starting before the reference, sorted right-to-left
            # (closest to reference first)
            target_indices = sorted(
                [i for i, c in enumerate(clips)
                 if i != ref_idx and c.start_time < ref_start - 0.01],
                key=lambda i: clips[i].start_time,
                reverse=True,
            )

        arrow = "→" if going_right else "←"
        logger.info("Match Grade %s: ref clip %d/%d (%s) on track %s, %d targets",
                     arrow, ref_idx, len(clips), ref_clip.name, track.name,
                     len(target_indices))
        if not target_indices:
            side = "after" if going_right else "before"
            self._status_label.setText(f"No clips {side} this one to match")
            return

        # NOTE: the reference's own match LUT is deliberately KEPT — the
        # match target is the clip's on-screen look. Stripping it here (old
        # behavior) permanently reverted the source clip's correction and
        # made chained matching converge on the raw ungraded color.

        # For right: use last frame of ref (boundary toward targets)
        # For left:  use first frame of ref (boundary toward targets)
        ref_which = "last" if going_right else "first"
        self._status_label.setText("Match Grade: extracting reference frame...")
        ref_frame = self._extract_analysis_frame(ref_clip, ref_which)
        if ref_frame is None:
            self._status_label.setText("Match Grade: failed to extract reference frame")
            return

        dlg = MatchGradeDialog(ref_frame, parent=self)
        if dlg.exec() != QDialog.Accepted:
            self._status_label.setText("Match Grade cancelled")
            return
        crop_rect = dlg.crop_rect()
        next_only = dlg.next_only()
        chain = dlg.chain()
        color_only = dlg.color_only()
        bake_permanent = dlg.bake_permanent()
        logger.info("Match Grade %s crop=%s next_only=%s chain=%s color_only=%s bake=%s",
                     arrow, crop_rect, next_only, chain, color_only, bake_permanent)

        if next_only:
            target_indices = target_indices[:1]

        effects_snapshot = {
            i: copy.deepcopy(clips[i].effects)
            for i in target_indices
        }

        total = len(target_indices)
        self._progress.setVisible(True)
        self._progress.setRange(0, total)
        self._progress.setValue(0)

        lut_dir = str(self.project_path / "luts")
        worker = _MatchGradeWorker(
            ref_idx, clips, lut_dir=lut_dir, crop_rect=crop_rect,
            target_indices=target_indices, chain=chain,
            direction=direction, parent=self,
        )

        def _on_progress(frac, msg):
            self._status_label.setText(msg)
            self._progress.setValue(int(frac * total))

        def _on_error(msg):
            self._status_label.setText(f"Match Grade: {msg}")
            self._progress.setVisible(False)

        worker.progress.connect(_on_progress)
        worker.finished_ok.connect(
            lambda matched: self._on_match_grade_done(
                matched, track_id, effects_snapshot,
                target_indices=target_indices,
                bake_permanent=bake_permanent,
            )
        )
        worker.error.connect(_on_error)
        worker.finished.connect(worker.deleteLater)
        self._match_grade_worker = worker
        worker.start()

    def _on_match_grade_done(
        self, count: int, track_id: str, effects_snapshot: dict[int, list],
        target_indices: list[int] | None = None,
        bake_permanent: bool = False,
    ) -> None:
        logger.info("Match Grade: applied to %d clip(s)", count)
        self._progress.setVisible(False)
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._save_state()
        if self._gl_preview:
            self._gl_preview.seek(self._multitrack.playhead)

        if bake_permanent and count > 0 and target_indices:
            # Bake the LUT into each affected clip so the look survives export,
            # combine, and any send-to. Use right-click → Unbake to revert.
            self._match_grade_undo = None
            track = self._multitrack.get_track(track_id)
            queue: list[tuple[str, int]] = []
            if track:
                for i in target_indices:
                    if not (0 <= i < len(track.clips)):
                        continue
                    c = track.clips[i]
                    if c.bake_data:
                        continue
                    has_match_lut = any(
                        fx.get("_match_grade") for fx in (c.effects or [])
                    )
                    if has_match_lut:
                        queue.append((track_id, i))
            if queue:
                self._status_label.setText(
                    f"Match grade applied — baking {len(queue)} clip(s) permanently…"
                )
                self._bake_effects_multi(queue)
            else:
                self._status_label.setText(
                    f"Match grade applied to {count} clip(s)"
                )
        else:
            self._match_grade_undo = (track_id, effects_snapshot)
            self._status_label.setText(
                f"Match grade applied to {count} clip(s) — Ctrl+Z to undo"
            )

    def _undo_match_grade(self) -> bool:
        """Undo the last match grade. Returns True if there was something to undo."""
        if not hasattr(self, "_match_grade_undo") or self._match_grade_undo is None:
            return False
        track_id, snapshot = self._match_grade_undo
        track = self._multitrack.get_track(track_id)
        if not track:
            return False
        for idx, effects in snapshot.items():
            if idx < len(track.clips):
                track.clips[idx].effects = effects
        self._match_grade_undo = None
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._save_state()
        if self._gl_preview:
            self._gl_preview.seek(self._multitrack.playhead)
        self._status_label.setText("Match grade undone")
        return True

    @staticmethod
    def _extract_analysis_frame(clip, which: str = "first") -> "np.ndarray | None":
        from sdqt.utils.grade_compute import extract_frame_from_clip
        return extract_frame_from_clip(clip, which)

    @staticmethod
    def _crop_region(frame: "np.ndarray", crop: tuple | None) -> "np.ndarray":
        from sdqt.utils.grade_compute import crop_region
        return crop_region(frame, crop)

    @staticmethod
    def _compute_grade(ref: "np.ndarray", target: "np.ndarray") -> dict | None:
        from sdqt.utils.grade_compute import compute_grade
        return compute_grade(ref, target)

    @staticmethod
    def _apply_grade(clip, corrections: dict) -> None:
        """Apply computed corrections as non-destructive effects on the clip."""
        from sdqt.models.effects import make_effect

        clip.effects = [fx for fx in clip.effects if not fx.get("_match_grade")]

        if "brightness" in corrections or "contrast" in corrections:
            fx = make_effect(
                "brightness_contrast",
                brightness=corrections.get("brightness", 0.0),
                contrast=corrections.get("contrast", 1.0),
            )
            fx["_match_grade"] = True
            clip.effects.insert(0, fx)

        if "temperature" in corrections or "tint" in corrections:
            fx = make_effect(
                "color_temperature",
                temperature=corrections.get("temperature", 0.0),
                tint=corrections.get("tint", 0.0),
            )
            fx["_match_grade"] = True
            clip.effects.insert(0, fx)

        if "rgb_gain" in corrections:
            r, g, b = corrections["rgb_gain"]
            fx = make_effect("channel_mixer", rr=r, gg=g, bb=b)
            fx["_match_grade"] = True
            clip.effects.insert(0, fx)

        if "saturation" in corrections:
            fx = make_effect("hue_sat", saturation=corrections["saturation"])
            fx["_match_grade"] = True
            clip.effects.insert(0, fx)

        if "black_point" in corrections or "white_point" in corrections:
            fx = make_effect(
                "levels",
                black_point=corrections.get("black_point", 0.0),
                white_point=corrections.get("white_point", 255.0),
                gamma=corrections.get("gamma", 1.0),
            )
            fx["_match_grade"] = True
            clip.effects.insert(0, fx)

        if "shadows" in corrections or "highlights" in corrections:
            fx = make_effect(
                "shadows_highlights",
                shadows=corrections.get("shadows", 0.0),
                highlights=corrections.get("highlights", 0.0),
            )
            fx["_match_grade"] = True
            clip.effects.insert(0, fx)

    # ── Match All To ─────────────────────────────────────────────────────

    @staticmethod
    def _apply_grade_lab(clip, corrections: dict) -> None:
        """Apply measured corrections as non-destructive effects.

        Uses ONLY ``channel_mixer`` + ``hue_sat``.  No other effects.
        channel_mixer is a linear per-channel multiply — it controls
        color AND luminance in one filter with zero nonlinear interaction.
        Adding brightness/contrast/levels on top would cause compounding
        that prevents convergence.

        Tagged ``_match_all_to: True`` for undo.
        """
        from sdqt.models.effects import make_effect

        # Strip previous match-all-to effects
        clip.effects = [
            fx for fx in clip.effects if not fx.get("_match_all_to")
        ]

        new_effects: list[dict] = []

        # 1. Channel mixer — THE correction tool (linear multiply = exact)
        if "rgb_gain" in corrections:
            r, g, b = corrections["rgb_gain"]
            fx = make_effect("channel_mixer", rr=r, gg=g, bb=b)
            fx["_match_all_to"] = True
            new_effects.append(fx)

        # 2. Saturation (orthogonal to channel_mixer)
        if "saturation" in corrections:
            fx = make_effect(
                "hue_sat", saturation=corrections["saturation"],
            )
            fx["_match_all_to"] = True
            new_effects.append(fx)

        # Insert at beginning (before user effects)
        clip.effects = new_effects + clip.effects

    @staticmethod
    def _apply_lut3d(clip, lut_path: str, tag: str = "_match_grade") -> None:
        """Apply a 3D LUT effect to a clip, replacing any previous match effects."""
        from sdqt.models.effects import make_effect

        clip.effects = [fx for fx in clip.effects if not fx.get(tag)]
        fx = make_effect("lut3d", lut_file=lut_path)
        fx[tag] = True
        clip.effects.insert(0, fx)

    def _match_similar_to(self, track_id: str, clip_idx: int) -> None:
        """Color-match all clips with the same name to the leftmost one."""
        import copy

        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return

        clicked_clip = track.clips[clip_idx]
        target_name = clicked_clip.name

        # Collect all clips with the same name across all tracks
        similar: list[tuple[str, int, object]] = []
        for t in self._multitrack.tracks:
            for i, c in enumerate(t.clips):
                if c.name == target_name:
                    similar.append((t.id, i, c))

        if len(similar) < 2:
            self._status_label.setText(
                f"Only one \"{target_name}\" clip found — nothing to match"
            )
            return

        # Use the leftmost clip (earliest start_time) as reference
        similar.sort(key=lambda x: x[2].start_time)
        ref_track_id, ref_idx, ref_clip = similar[0]

        # Strip previous match-similar effects from the reference
        ref_clip.effects = [
            fx for fx in ref_clip.effects if not fx.get("_match_similar")
        ]

        # Targets are everything except the reference
        targets = [(tid, idx, c) for tid, idx, c in similar[1:]]

        effects_snapshot: dict[tuple[str, int], list] = {}
        for tid, idx, c in targets:
            effects_snapshot[(tid, idx)] = copy.deepcopy(c.effects)

        total = len(targets)
        logger.info(
            "Match Similar: ref=%s (track %s), %d target(s)",
            ref_clip.name, ref_track_id, total,
        )
        self._status_label.setText(
            f"Match Similar: matching {total} \"{target_name}\" clip(s)..."
        )
        self._progress.setVisible(True)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)

        lut_dir = str(self.project_path / "luts")
        worker = _MatchAllToWorker(
            ref_clip, targets, lut_dir=lut_dir,
            tag="_match_similar", parent=self,
        )

        def _on_progress(frac, msg):
            self._progress.setValue(int(frac * 100))
            if msg:
                self._status_label.setText(msg)

        def _on_error(msg):
            self._status_label.setText(f"Match Similar: {msg}")
            self._progress.setVisible(False)

        worker.progress.connect(_on_progress)
        worker.finished_ok.connect(
            lambda matched: self._on_match_similar_done(
                matched, target_name, effects_snapshot,
            )
        )
        worker.error.connect(_on_error)
        worker.finished.connect(worker.deleteLater)
        self._match_similar_worker = worker
        worker.start()

    def _on_match_similar_done(
        self, count: int, name: str,
        effects_snapshot: dict[tuple[str, int], list],
    ) -> None:
        logger.info("Match Similar: applied to %d clip(s)", count)
        self._match_similar_undo = effects_snapshot
        self._progress.setVisible(False)
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._save_state()
        if self._gl_preview:
            self._gl_preview.seek(self._multitrack.playhead)
        self._status_label.setText(
            f"Matched {count} \"{name}\" clip(s) to leftmost — Ctrl+Z to undo"
        )

    def _undo_match_similar(self) -> bool:
        """Undo the last Match Similar. Returns True if undone."""
        if not hasattr(self, "_match_similar_undo") or self._match_similar_undo is None:
            return False
        snapshot = self._match_similar_undo
        for (track_id, idx), effects in snapshot.items():
            track = self._multitrack.get_track(track_id)
            if track and idx < len(track.clips):
                track.clips[idx].effects = effects
        self._match_similar_undo = None
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._save_state()
        if self._gl_preview:
            self._gl_preview.seek(self._multitrack.playhead)
        self._status_label.setText("Match Similar undone")
        return True

    def _match_all_to_playhead(self) -> None:
        """Color-match ALL clips to the exact frame under the playhead.

        The reference is the frame the user is LOOKING AT: the clip under the
        playhead on the topmost visible video track, rendered at the playhead
        position with its effects applied. Deterministic and visible — park
        the playhead on a good frame, run this, everything matches it.
        """
        import copy
        from sdqt.widgets.timeline_track import TrackType

        pos = float(self._multitrack.playhead)

        # Find the clip under the playhead — topmost visible video track wins
        # (matches what the GL preview shows on screen).
        ref_track = ref_clip = None
        ref_idx = -1
        for t in reversed(self._multitrack.tracks):
            if getattr(t, "track_type", None) != TrackType.VIDEO or not t.visible:
                continue
            for i, c in enumerate(t.clips):
                if c.start_time <= pos < c.start_time + c.duration:
                    ref_track, ref_clip, ref_idx = t, c, i
                    break
            if ref_clip is not None:
                break

        if ref_clip is None:
            self._status_label.setText(
                "Match to Playhead: no clip under the playhead"
            )
            return

        # Render the EXACT playhead frame, with the clip's effects applied
        # (excluding prior match LUTs so re-matching converges) — this is the
        # same full-range-RGB extraction the LUT domain is built on.
        from sdqt.utils.grade_compute import render_frame_with_effects

        def _is_match(fx: dict) -> bool:
            return bool(
                fx.get("_match_grade")
                or fx.get("_match_all_to")
                or fx.get("_match_similar")
            )

        active_fx = [
            fx for fx in (ref_clip.effects or [])
            if fx.get("enabled", True) and not _is_match(fx)
        ]
        media_ts = ref_clip.media_offset + (pos - ref_clip.start_time)
        self._status_label.setText(
            f"Match to Playhead: extracting frame at {pos:.2f}s "
            f"from {ref_clip.name}…"
        )
        ref_frame = render_frame_with_effects(
            ref_clip.path, active_fx, media_offset=max(media_ts, 0.0),
        )
        if ref_frame is None:
            self._status_label.setText(
                "Match to Playhead: failed to extract the playhead frame"
            )
            return

        # Strip previous match-all-to effects from the reference clip so its
        # own look stays the anchor, then target every other clip.
        ref_clip.effects = [
            fx for fx in ref_clip.effects if not fx.get("_match_all_to")
        ]

        targets: list[tuple[str, int, object]] = []
        effects_snapshot: dict[tuple[str, int], list] = {}
        for t in self._multitrack.tracks:
            for i, c in enumerate(t.clips):
                if t.id == ref_track.id and i == ref_idx:
                    continue
                targets.append((t.id, i, c))
                effects_snapshot[(t.id, i)] = copy.deepcopy(c.effects)

        if not targets:
            self._status_label.setText("No other clips to match")
            return

        total = len(targets)
        logger.info(
            "Match to Playhead: ref frame @ %.3fs from %s, %d target(s)",
            pos, ref_clip.name, total,
        )
        self._status_label.setText(
            f"Match to Playhead: matching {total} clip(s) to the frame at "
            f"{pos:.2f}s…"
        )
        self._progress.setVisible(True)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)

        lut_dir = str(self.project_path / "luts")
        worker = _MatchAllToWorker(
            ref_clip, targets, lut_dir=lut_dir,
            ref_frame=ref_frame, parent=self,
        )

        def _on_progress(frac, msg):
            self._progress.setValue(int(frac * 100))
            if msg:
                self._status_label.setText(msg)

        def _on_error(msg):
            self._status_label.setText(f"Match to Playhead: {msg}")
            self._progress.setVisible(False)

        worker.progress.connect(_on_progress)
        worker.finished_ok.connect(
            lambda matched: self._on_match_all_to_done(
                matched, effects_snapshot,
            )
        )
        worker.error.connect(_on_error)
        worker.finished.connect(worker.deleteLater)
        self._match_all_to_worker = worker
        worker.start()

    def _match_all_to(self, track_id: str, clip_idx: int) -> None:
        """Color-match ALL clips on ALL tracks to the given reference clip."""
        import copy

        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return

        ref_clip = track.clips[clip_idx]
        logger.info(
            "Match All To: reference clip %s (track %s, idx %d)",
            ref_clip.name, track_id, clip_idx,
        )

        # Strip any previous match-all-to effects from the reference clip
        # so we match against its actual appearance, not a cascaded LUT
        ref_clip.effects = [
            fx for fx in ref_clip.effects if not fx.get("_match_all_to")
        ]

        # Collect all clips across all tracks, excluding the reference
        targets: list[tuple[str, int, object]] = []
        effects_snapshot: dict[tuple[str, int], list] = {}

        for t in self._multitrack.tracks:
            for i, c in enumerate(t.clips):
                if t.id == track_id and i == clip_idx:
                    continue  # skip the reference clip itself
                targets.append((t.id, i, c))
                effects_snapshot[(t.id, i)] = copy.deepcopy(c.effects)

        if not targets:
            self._status_label.setText("No other clips to match")
            return

        total = len(targets)
        self._status_label.setText(
            f"Match All To: analysing {total} clip(s) across "
            f"{len(self._multitrack.tracks)} track(s)..."
        )
        self._progress.setVisible(True)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)

        lut_dir = str(self.project_path / "luts")
        worker = _MatchAllToWorker(ref_clip, targets, lut_dir=lut_dir, parent=self)

        def _on_progress(frac, msg):
            self._progress.setValue(int(frac * 100))
            if msg:
                self._status_label.setText(msg)

        def _on_error(msg):
            self._status_label.setText(f"Match All To: {msg}")
            self._progress.setVisible(False)

        worker.progress.connect(_on_progress)
        worker.finished_ok.connect(
            lambda matched: self._on_match_all_to_done(
                matched, effects_snapshot,
            )
        )
        worker.error.connect(_on_error)
        worker.finished.connect(worker.deleteLater)
        self._match_all_to_worker = worker
        worker.start()

    def _on_match_all_to_done(
        self, count: int, effects_snapshot: dict[tuple[str, int], list],
    ) -> None:
        logger.info("Match All To: applied to %d clip(s)", count)
        self._match_all_to_undo = effects_snapshot
        self._progress.setVisible(False)
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._save_state()
        if self._gl_preview:
            self._gl_preview.seek(self._multitrack.playhead)
        self._status_label.setText(
            f"Match All To applied to {count} clip(s) — Ctrl+Z to undo"
        )

    def _undo_match_all_to(self) -> bool:
        """Undo the last Match All To. Returns True if undone."""
        if not hasattr(self, "_match_all_to_undo") or self._match_all_to_undo is None:
            return False
        snapshot = self._match_all_to_undo
        for (track_id, idx), effects in snapshot.items():
            track = self._multitrack.get_track(track_id)
            if track and idx < len(track.clips):
                track.clips[idx].effects = effects
        self._match_all_to_undo = None
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._save_state()
        if self._gl_preview:
            self._gl_preview.seek(self._multitrack.playhead)
        self._status_label.setText("Match All To undone")
        return True

    # ── Normalize Track Colors ───────────────────────────────────────────

    def _normalize_track_colors(self, track_id: str) -> None:
        """Run ffmpeg normalize across all clips on a track to smooth color."""
        import copy

        track = self._multitrack.get_track(track_id)
        if not track or not track.clips:
            self._status_label.setText("No clips on this track")
            return

        clips = sorted(track.clips, key=lambda c: c.start_time)

        # Snapshot original paths for undo
        self._normalize_undo = (
            track_id,
            {i: copy.deepcopy(c.path) for i, c in enumerate(track.clips)},
        )

        total = len(clips)
        self._status_label.setText(
            f"Normalizing {total} clips on {track.name}..."
        )
        self._progress.setVisible(True)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)

        output_dir = str(self.project_path / "clips")
        worker = _NormalizeTrackWorker(
            clips, output_dir=output_dir, parent=self,
        )

        def _on_progress(frac, msg):
            self._progress.setValue(int(frac * 100))
            if msg:
                self._status_label.setText(msg)

        def _on_error(msg):
            self._status_label.setText(f"Normalize: {msg}")
            self._progress.setVisible(False)

        def _on_done(results):
            self._progress.setVisible(False)
            replaced = 0
            fc = getattr(self, "_frame_cache", None)
            for i, clip in enumerate(clips):
                if i < len(results) and results[i]:
                    # Drop cached decoders/frames for BOTH the old path (now
                    # unused) and the new path — a previous normalize may have
                    # cached the same `_norm.mp4` name, and a stale decoder on
                    # an overwritten file shows black until app restart.
                    if fc is not None:
                        try:
                            fc.release_clip(clip.path)
                            fc.release_clip(results[i])
                        except Exception:
                            pass
                    clip.path = results[i]
                    replaced += 1
            self._multitrack._sync_canvas()
            self._multitrack.clips_changed.emit()
            self._save_state()
            if self._gl_preview:
                self._gl_preview.seek(self._multitrack.playhead)
            self._status_label.setText(
                f"Normalized {replaced} clip(s) — Ctrl+Z to undo"
            )

        worker.progress.connect(_on_progress)
        worker.finished_ok.connect(_on_done)
        worker.error.connect(_on_error)
        worker.finished.connect(worker.deleteLater)
        self._normalize_worker = worker
        worker.start()

    def _undo_normalize(self) -> bool:
        """Undo the last normalize. Returns True if undone."""
        if not hasattr(self, "_normalize_undo") or self._normalize_undo is None:
            return False
        track_id, path_snapshot = self._normalize_undo
        track = self._multitrack.get_track(track_id)
        if not track:
            return False
        for idx, path in path_snapshot.items():
            if idx < len(track.clips):
                track.clips[idx].path = path
        self._normalize_undo = None
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._save_state()
        if self._gl_preview:
            self._gl_preview.seek(self._multitrack.playhead)
        self._status_label.setText("Normalize undone")
        return True
