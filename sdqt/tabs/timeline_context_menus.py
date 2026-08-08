"""Timeline context menus mixin — right-click menus for clips, tracks, library."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QDialog, QInputDialog, QMenu

from sdqt.utils.naming import cap_stem
from sdqt.widgets.timeline_track import TimelineClip, TimelineTrack, TrackType

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TimelineContextMenus:
    """Mixin providing context menu methods for TimelineTab."""

    _copied_effects: list[dict] | None = None  # class-level clipboard

    def _on_track_context_menu(self, track_id: str, clip_idx: int, global_pos: QPoint) -> None:
        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        clip = track.clips[clip_idx]
        self._show_clip_context_menu(
            clip.path, global_pos, track_id=track_id, track_clip_idx=clip_idx,
            media_offset=getattr(clip, "media_offset", 0.0) or 0.0,
            clip_duration=clip.duration,
        )

    def _on_multi_clip_context_menu(self, global_pos: QPoint) -> None:
        """Right-click with multiple clips selected."""
        selection = self._multitrack.multi_selection
        if len(selection) < 2:
            return
        menu = QMenu(self)
        can_combine = self._can_combine(selection)
        if can_combine:
            combine_act = menu.addAction(f"Combine ({len(selection)} clips)")
            combine_act.triggered.connect(lambda: self._combine_clips(selection[:]))
            menu.addSeparator()
        close_act = menu.addAction(f"Close Gaps ({len(selection)} clips)")
        close_act.triggered.connect(lambda: self._close_gaps(selection[:]))
        # -- Effects submenu --
        fx_menu = menu.addMenu("Effects")
        if TimelineContextMenus._copied_effects is not None:
            paste_multi_act = fx_menu.addAction(f"Paste Effects ({len(selection)} clips)")
            paste_multi_act.triggered.connect(
                lambda: self._paste_effects_multi(selection[:])
            )
        clear_multi_act = fx_menu.addAction(f"Clear Effects ({len(selection)} clips)")
        clear_multi_act.triggered.connect(
            lambda: self._clear_effects_multi(selection[:])
        )
        # Check if any selected clips have bakeable effects
        has_bakeable = False
        for tid, cidx in selection:
            t = self._multitrack.get_track(tid)
            if t and 0 <= cidx < len(t.clips):
                c = t.clips[cidx]
                if c.effects and not c.bake_data:
                    has_bakeable = True
                    break
        if has_bakeable:
            fx_menu.addSeparator()
            bake_multi_act = fx_menu.addAction(f"Bake Effects ({len(selection)} clips)")
            bake_multi_act.triggered.connect(
                lambda: self._bake_effects_multi(selection[:])
            )
        menu.addSeparator()
        qe_act = menu.addAction(f"Quick Export {len(selection)} clips…")
        qe_act.setToolTip(
            "Batch quick-export the selected clips — each clip goes through "
            "the same WYSIWYG bake + denoise + sharpen + face restore + "
            "lanczos 1.5x + RIFE 2x pipeline as single-clip Quick Export."
        )
        qe_act.triggered.connect(lambda: self._quick_export_multi(selection[:]))

        menu.addSeparator()
        remove_act = menu.addAction(f"Remove {len(selection)} clips")
        remove_act.triggered.connect(lambda: self._remove_multi(selection[:]))
        menu.exec(global_pos)

    def _quick_export_multi(self, selection: list[tuple[str, int]]) -> None:
        """Quick-export a batch of selected timeline clips to a chosen folder."""
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        items: list[dict] = []
        for tid, cidx in selection:
            track = self._multitrack.get_track(tid)
            if not track or cidx < 0 or cidx >= len(track.clips):
                continue
            clip = track.clips[cidx]
            items.append({
                "path": clip.path,
                "effects": list(clip.effects or []),
                "media_offset": getattr(clip, "media_offset", 0.0) or 0.0,
                "duration": clip.duration,
                "name": clip.name or Path(clip.path).stem,
            })
        if not items:
            return

        default_dir = self.state.global_config.last_export_dir or str(Path.home())
        out_dir = QFileDialog.getExistingDirectory(
            self, f"Quick Export {len(items)} clips — choose output folder",
            default_dir,
        )
        if not out_dir:
            return

        try:
            self.state.global_config.last_export_dir = out_dir
            self.state.global_config.save()
        except Exception:
            pass

        from sdqt.tabs.clip_menu_actions import run_quick_export_batch
        run_quick_export_batch(self, self.state, items, out_dir)

    def _on_library_context_menu(self, path: str, global_pos: QPoint) -> None:
        self._show_clip_context_menu(path, global_pos, from_library=True)

    def _on_track_header_context_menu(self, track_id: str, global_pos: QPoint) -> None:
        track = self._multitrack.get_track(track_id)
        if not track:
            return
        menu = QMenu(self)
        rename_act = menu.addAction("Rename Track")
        rename_act.triggered.connect(lambda: self._rename_track(track_id))
        dup_act = menu.addAction("Duplicate Track")
        dup_act.triggered.connect(lambda: self._duplicate_track(track_id))
        menu.addSeparator()
        select_all_act = menu.addAction("Select All Clips")
        select_all_act.triggered.connect(lambda: self._select_all_on_track(track_id))
        close_gaps_act = menu.addAction("Close Gaps")
        close_gaps_act.triggered.connect(lambda: self._close_gaps_on_track(track_id))
        menu.addSeparator()
        tracks = self._multitrack.tracks
        idx = next((i for i, t in enumerate(tracks) if t.id == track_id), -1)
        if idx > 0:
            up_act = menu.addAction("Move Track Up")
            up_act.triggered.connect(lambda: self._move_track(track_id, -1))
        if idx < len(tracks) - 1:
            down_act = menu.addAction("Move Track Down")
            down_act.triggered.connect(lambda: self._move_track(track_id, 1))
        menu.addSeparator()
        del_act = menu.addAction("Delete Track")
        del_act.triggered.connect(lambda: self._delete_specific_track(track_id))
        menu.exec(global_pos)

    def _show_clip_context_menu(
        self, clip_path: str, global_pos: QPoint,
        track_id: str = "", track_clip_idx: int = -1, from_library: bool = False,
        media_offset: float = 0.0, clip_duration: float = 0.0,
    ) -> None:
        from sdqt.workers.timeline import _probe_has_audio

        menu = QMenu(self)

        # Resolve the exact clicked clip instance. Path-based lookup is
        # ambiguous after Split/Duplicate (siblings share one source file), so
        # every action below that needs effects/trim/position gets this object.
        menu_clip = None
        if track_id and track_clip_idx >= 0:
            _mt = self._multitrack.get_track(track_id)
            if _mt and 0 <= track_clip_idx < len(_mt.clips):
                menu_clip = _mt.clips[track_clip_idx]

        if from_library:
            add_act = menu.addAction("Add to Timeline")
            add_act.triggered.connect(lambda: self._add_library_clip_to_timeline(clip_path))
            lib_rename_act = menu.addAction("Rename Clip")
            lib_rename_act.triggered.connect(lambda: self._rename_clip_file(clip_path))
            rem_act = menu.addAction("Remove from Library")
            rem_act.triggered.connect(lambda: self._library.remove_clip(clip_path))
            menu.addSeparator()
        elif track_id and track_clip_idx >= 0:
            rem_act = menu.addAction("Remove from Timeline")
            rem_act.triggered.connect(
                lambda: self._multitrack.remove_clip_from_track(track_id, track_clip_idx)
            )
            ripple_del_act = menu.addAction("Ripple Delete")
            ripple_del_act.setToolTip("Remove clip and shift subsequent clips left")
            ripple_del_act.triggered.connect(
                lambda: self._ripple_delete_clip(track_id, track_clip_idx)
            )
            _mute_clip = self._multitrack.get_track(track_id).clips[track_clip_idx] if track_id else None
            if _mute_clip:
                is_muted = getattr(_mute_clip, "muted", False)
                mute_act = menu.addAction("Unmute Clip" if is_muted else "Mute Clip")
                mute_act.triggered.connect(
                    lambda checked, tid=track_id, ci=track_clip_idx:
                        self._toggle_clip_mute(tid, ci)
                )
                rename_act = menu.addAction("Rename Clip")
                rename_act.triggered.connect(
                    lambda checked, tid=track_id, ci=track_clip_idx:
                        self._rename_clip(tid, ci)
                )
            menu.addSeparator()

        # Send clip to
        def _send_clip(key, path=clip_path, mo=media_offset, dur=clip_duration):
            trimmed = self._export_trimmed_clip(path, mo, dur, clip=menu_clip) if dur > 0 else path
            self.send_clip_to.emit(key, trimmed)

        if self._clip_targets:
            from sdqt.widgets.send_targets import build_target_menu
            build_target_menu(menu, "Send clip to", self._clip_targets, _send_clip)

        # Send frame to (current/first/last — each as its own categorised submenu)
        if self._frame_targets:
            from sdqt.widgets.send_targets import build_target_menu as _btm, DISABLED_TARGETS as _DT
            mo = media_offset
            dur = clip_duration

            for label, which in [
                ("Send current frame to", "current"),
                ("Send first frame to", "first"),
                ("Send last frame to", "last"),
            ]:
                frame_menu = menu.addMenu(label)
                # I2V first/last frame shortcuts at top
                act_f = frame_menu.addAction("I2V First Frame")
                act_f.triggered.connect(
                    lambda checked, p=clip_path, w=which: self._extract_and_send_frame(
                        p, "img2vid_first", w, mo, dur, clip=menu_clip)
                )
                act_l = frame_menu.addAction("I2V Last Frame")
                act_l.triggered.connect(
                    lambda checked, p=clip_path, w=which: self._extract_and_send_frame(
                        p, "img2vid_last", w, mo, dur, clip=menu_clip)
                )
                # "Set as Neutralizer Reference" — extracts the chosen frame
                # and stores it as the global neutralize reference image so
                # subsequent extracted frames anchor to its color balance.
                act_neut = frame_menu.addAction("Set as Neutralizer Reference")
                act_neut.setToolTip(
                    "Save this frame as the color reference used by the frame "
                    "neutralizer (Settings → Color Drift Neutralization)."
                )
                act_neut.triggered.connect(
                    lambda checked, p=clip_path, w=which: self._set_as_neutralizer_reference(
                        p, w, mo, dur, clip=menu_clip)
                )
                frame_menu.addSeparator()

                # Populate with categorised image targets
                def _frame_callback(key, _w=which):
                    self._extract_and_send_frame(clip_path, key, _w, mo, dur, clip=menu_clip)
                from sdqt.widgets.send_targets import _populate_menu
                _populate_menu(frame_menu, self._frame_targets, _frame_callback)

        # Final frame targets
        if self._final_frame_targets:
            from sdqt.widgets.send_targets import build_target_menu as _btm2
            _btm2(menu, "Final Frame", self._final_frame_targets,
                   lambda k, p=clip_path: self._extract_and_send_final_frame(p, k, clip=menu_clip))

        # Guide video targets
        if self._guide_video_targets:
            from sdqt.widgets.send_targets import build_target_menu as _btm3
            _btm3(menu, "Guide Video", self._guide_video_targets,
                   lambda k, p=clip_path: self._extract_and_send_guide_video(p, k, clip=menu_clip))

        # Goto first/last frame
        if track_id and track_clip_idx >= 0:
            goto_menu = menu.addMenu("Goto")
            goto_menu.addAction("First Frame").triggered.connect(
                lambda checked: self._seek_to_clip_edge(menu_clip, "first"))
            goto_menu.addAction("Last Frame").triggered.connect(
                lambda checked: self._seek_to_clip_edge(menu_clip, "last"))

        # Clip Info / Prompt
        info_act = menu.addAction("Clip Info")
        info_act.triggered.connect(lambda checked, p=clip_path: self._show_clip_info(p))

        # Edit actions
        menu.addSeparator()

        if track_id and track_clip_idx >= 0:
            split_act = menu.addAction("Split at Playhead")
            split_act.triggered.connect(
                lambda: self._split_clip_on_track(track_id, track_clip_idx)
            )
            reverse_act = menu.addAction("Reverse")
            reverse_act.triggered.connect(
                lambda: self._reverse_clip_on_track(track_id, track_clip_idx)
            )
            speed_act = menu.addAction("Speed...")
            speed_act.triggered.connect(
                lambda: self._speed_clip_on_track(track_id, track_clip_idx)
            )
            # -- Effects submenu --
            fx_menu = menu.addMenu("Effects")
            effects_act = fx_menu.addAction("Edit Effects...")
            effects_act.triggered.connect(
                lambda: self._effects_clip_on_track(track_id, track_clip_idx)
            )
            _eff_clip = self._multitrack.get_track(track_id).clips[track_clip_idx]
            if _eff_clip.effects:
                copy_fx_act = fx_menu.addAction("Copy Effects")
                copy_fx_act.triggered.connect(
                    lambda: self._copy_effects(track_id, track_clip_idx)
                )
            if TimelineContextMenus._copied_effects is not None:
                paste_fx_act = fx_menu.addAction("Paste Effects")
                paste_fx_act.triggered.connect(
                    lambda: self._paste_effects(track_id, track_clip_idx)
                )
            if _eff_clip.effects:
                clear_fx_act = fx_menu.addAction("Clear Effects")
                clear_fx_act.triggered.connect(
                    lambda: self._clear_effects(track_id, track_clip_idx)
                )
            if _eff_clip.effects and not _eff_clip.bake_data:
                fx_menu.addSeparator()
                bake_act = fx_menu.addAction("Bake Effects")
                bake_act.setToolTip("Render effects into the clip file permanently (reversible)")
                bake_act.triggered.connect(
                    lambda: self._bake_effects(track_id, track_clip_idx)
                )
            if _eff_clip.bake_data:
                unbake_act = fx_menu.addAction("Unbake Effects")
                unbake_act.setToolTip("Restore original clip and effects")
                unbake_act.triggered.connect(
                    lambda: self._unbake_effects(track_id, track_clip_idx)
                )
            postproc_act = menu.addAction("Post Processing...")
            postproc_act.triggered.connect(
                lambda: self._postprocess_clip_on_track(track_id, track_clip_idx)
            )
            ls_menu = menu.addMenu("Lip Sync")
            for ls_key, ls_label in [
                ("ltx_a2vid", "LTX Audio-to-Video..."),
                ("musetalk", "MuseTalk..."),
                ("vace_multitalk", "VACE MultiTalk..."),
                ("latentsync", "LatentSync..."),
                ("video_retalking", "Video Retalking..."),
            ]:
                ls_act = ls_menu.addAction(ls_label)
                ls_act.triggered.connect(
                    lambda checked, k=ls_key: self._lipsync_clip_on_track(track_id, track_clip_idx, engine=k)
                )
            faceswap_act = menu.addAction("Face Swap...")
            faceswap_act.triggered.connect(
                lambda: self._faceswap_clip_on_track(track_id, track_clip_idx)
            )
            # Sequences submenu
            seq_menu = menu.addMenu("Sequences")
            char_replace_act = seq_menu.addAction("Character Replacement...")
            char_replace_act.triggered.connect(
                lambda: self._sequence_character_replacement(track_id, track_clip_idx)
            )
            # Corrections submenu
            corr_menu = menu.addMenu("Corrections")
            matchgrade_right = corr_menu.addAction("Match Grade →")
            matchgrade_right.setToolTip("Color-match all clips to the right")
            matchgrade_right.triggered.connect(
                lambda: self._match_grade_from(track_id, track_clip_idx, direction="right")
            )
            matchgrade_left = corr_menu.addAction("Match Grade ←")
            matchgrade_left.setToolTip("Color-match all clips to the left")
            matchgrade_left.triggered.connect(
                lambda: self._match_grade_from(track_id, track_clip_idx, direction="left")
            )
            matchall_act = corr_menu.addAction("Match All To")
            matchall_act.setToolTip("Color-match ALL clips on ALL tracks to this one")
            matchall_act.triggered.connect(
                lambda: self._match_all_to(track_id, track_clip_idx)
            )
            matchplayhead_act = corr_menu.addAction("Match All to Playhead Frame")
            matchplayhead_act.setToolTip(
                "Color-match ALL clips to the exact frame currently under "
                "the playhead — park the playhead on a good frame first"
            )
            matchplayhead_act.triggered.connect(
                lambda: self._match_all_to_playhead()
            )
            matchsimilar_act = corr_menu.addAction("Match Similar Clips")
            matchsimilar_act.setToolTip(
                "Color-match all clips with this name to the leftmost one"
            )
            matchsimilar_act.triggered.connect(
                lambda: self._match_similar_to(track_id, track_clip_idx)
            )
            corr_menu.addSeparator()
            normalize_act = corr_menu.addAction("Normalize Track Colors")
            normalize_act.setToolTip(
                "Smooth out color differences across all clips on this track"
            )
            normalize_act.triggered.connect(
                lambda: self._normalize_track_colors(track_id)
            )
            corr_menu.addSeparator()
            cc_ref_act = corr_menu.addAction("Set as Color Reference")
            cc_ref_act.setToolTip("Use this clip's frame as reference in Color Correct tab")
            cc_ref_act.triggered.connect(
                lambda: self._send_clip_as_color_ref(clip_path, media_offset)
            )
            cc_source_act = corr_menu.addAction("Color Correct This Clip")
            cc_source_act.setToolTip("Send to Color Correct tab — corrected result can replace this clip")
            cc_source_act.triggered.connect(
                lambda: self.send_cc_source_from_timeline.emit(clip_path)
            )

            _clip = self._multitrack.get_track(track_id).clips[track_clip_idx] if track_id else None
            cur_rot = getattr(_clip, "rotation", 0) % 360
            rot_label = f" [{cur_rot}°]" if cur_rot else ""
            rotate_cw = menu.addAction(f"Rotate 90° CW{rot_label}")
            rotate_cw.triggered.connect(
                lambda: self._rotate_clip_on_track(track_id, track_clip_idx, "cw")
            )
            rotate_ccw = menu.addAction(f"Rotate 90° CCW{rot_label}")
            rotate_ccw.triggered.connect(
                lambda: self._rotate_clip_on_track(track_id, track_clip_idx, "ccw")
            )

            resize_act = menu.addAction("Resize...")
            resize_act.setToolTip(
                "Re-encode the clip to a chosen resolution (presets or manual). "
                "Useful for matching source clips to AI-generated clip dimensions."
            )
            resize_act.triggered.connect(
                lambda: self._resize_clip_dialog(track_id, track_clip_idx)
            )

            resize_left_act = menu.addAction("Resize to ←")
            resize_left_act.setToolTip(
                "Re-encode this clip to match the resolution/fps of the clip "
                "to its left on the same track"
            )
            resize_left_act.triggered.connect(
                lambda: self._resize_clip_to_neighbor(track_id, track_clip_idx, "left")
            )
            resize_right_act = menu.addAction("Resize to →")
            resize_right_act.setToolTip(
                "Re-encode this clip to match the resolution/fps of the clip "
                "to its right on the same track"
            )
            resize_right_act.triggered.connect(
                lambda: self._resize_clip_to_neighbor(track_id, track_clip_idx, "right")
            )

            save_lib_act = menu.addAction("Save to Library")
            save_lib_act.triggered.connect(
                lambda: self._save_clip_to_library(track_id, track_clip_idx)
            )

            resync_act = menu.addAction("Resync Audio")
            resync_act.triggered.connect(self._resync_audio)

            _track_obj = self._multitrack.get_track(track_id)
            if (_track_obj and _track_obj.track_type == TrackType.VIDEO
                    and _clip and _clip.has_audio):
                sep_menu = menu.addMenu("Separate Audio")
                audio_tracks = [t for t in self._multitrack.tracks
                                if t.track_type == TrackType.AUDIO]
                for at in audio_tracks:
                    act = sep_menu.addAction(at.name)
                    act.triggered.connect(
                        lambda checked, tid=track_id, ci=track_clip_idx, atid=at.id:
                            self._separate_audio(tid, ci, atid)
                    )
                if audio_tracks:
                    sep_menu.addSeparator()
                new_act = sep_menu.addAction("New Audio Track")
                new_act.triggered.connect(
                    lambda checked, tid=track_id, ci=track_clip_idx:
                        self._separate_audio(tid, ci, "")
                )

            other_tracks = [t for t in self._multitrack.tracks if t.id != track_id]
            if other_tracks:
                move_menu = menu.addMenu("Move to Track")
                for t in other_tracks:
                    act = move_menu.addAction(t.name)
                    act.triggered.connect(
                        lambda checked, tid=t.id, ci=track_clip_idx, stid=track_id:
                            self._move_clip_to_track(stid, ci, tid)
                    )

        elif from_library:
            reverse_act = menu.addAction("Reverse")
            reverse_act.triggered.connect(
                lambda checked, p=clip_path: self._reverse_library_clip(p)
            )
            speed_act = menu.addAction("Speed...")
            speed_act.triggered.connect(
                lambda checked, p=clip_path: self._speed_library_clip(p)
            )

        # Export / Quick Export / Create Loop — shared across all clip menus.
        # For timeline/multitrack clips, pass the effect stack so Quick Export
        # can bake the same LUTs / colour correction / rotations the user sees
        # in the timeline preview.
        _clip_effects: list[dict] | None = None
        if track_id and track_clip_idx >= 0:
            _t = self._multitrack.get_track(track_id)
            if _t and 0 <= track_clip_idx < len(_t.clips):
                _clip_effects = list(_t.clips[track_clip_idx].effects or [])

        from sdqt.tabs.clip_menu_actions import add_clip_export_actions
        add_clip_export_actions(
            menu, self, self.state, clip_path,
            media_offset=media_offset, duration=clip_duration,
            clip_name=Path(clip_path).stem,
            effects=_clip_effects,
        )

        menu.exec(global_pos)

    # -- Multi-clip helpers --------------------------------------------------

    def _can_combine(self, selection: list[tuple[str, int]]) -> bool:
        """Check if selected clips are adjacent on the same track (by position, not index)."""
        if len(selection) < 2:
            return False
        track_ids = {tid for tid, _ in selection}
        if len(track_ids) != 1:
            return False
        track_id = track_ids.pop()
        track = self._multitrack.get_track(track_id)
        if not track:
            return False
        indices = [idx for _, idx in selection]
        clips_by_time = sorted(
            [(track.clips[i].start_time, i) for i in indices],
            key=lambda x: x[0],
        )
        for i in range(len(clips_by_time) - 1):
            _, idx_a = clips_by_time[i]
            _, idx_b = clips_by_time[i + 1]
            c1 = track.clips[idx_a]
            c2 = track.clips[idx_b]
            gap = abs(c2.start_time - (c1.start_time + c1.duration))
            if gap > 0.1:
                return False
        return True

    def _combine_clips(self, selection: list[tuple[str, int]]) -> None:
        """Combine selected contiguous clips into one."""
        from sdqt.workers.timeline import _probe_has_audio
        from sdqt.utils.codec import configured_codec_args as _codec_args

        track_id = selection[0][0]
        track = self._multitrack.get_track(track_id)
        if not track:
            return

        # Sort by visual position (start_time), not by add order (index)
        sel_indices = [idx for _, idx in selection]
        sorted_pairs = sorted(
            [(track.clips[i].start_time, i) for i in sel_indices],
            key=lambda x: x[0],
        )
        indices = [i for _, i in sorted_pairs]
        clips = [track.clips[i] for i in indices]

        same_source = len(set(c.path for c in clips)) == 1
        contiguous = True
        if same_source and len(clips) > 1:
            for i in range(len(clips) - 1):
                a, b = clips[i], clips[i + 1]
                expected_offset = a.media_offset + a.duration
                if abs(b.media_offset - expected_offset) > 0.05:
                    contiguous = False
                    break

        # The metadata-only fast path keeps just clips[0]'s effect stack and
        # discards the rest. When per-clip effects (e.g. match-grade LUTs)
        # differ, that silently loses grades on later clips and the GL preview
        # shifts colors after combine. Force the re-encode path in that case
        # so each clip's effects get baked into its own segment.
        all_effects_equal = all(
            (c.effects or []) == (clips[0].effects or []) for c in clips
        )

        if same_source and contiguous and all_effects_equal:
            self._combine_metadata(track, indices, clips)
        else:
            self._combine_reencode(track, indices, clips)

    def _combine_metadata(self, track, indices: list[int], clips: list) -> None:
        """Instant non-destructive combine: merge metadata only."""
        import copy

        first = clips[0]
        total_dur = sum(c.duration for c in clips)

        combined = TimelineClip(
            path=first.path,
            duration=total_dur,
            start_time=first.start_time,
            name=first.name.rstrip("_A_B").rstrip("_AB") or Path(first.path).stem,
            color=first.color,
            volume=first.volume,
            has_audio=first.has_audio,
            waveform=first.waveform,
            media_duration=first.media_duration,
            media_offset=first.media_offset,
            fade_in=first.fade_in,
            fade_out=clips[-1].fade_out,
            effects=copy.deepcopy(first.effects),
            rotation=first.rotation,
            group_id=first.group_id,
        )

        for idx in sorted(indices, reverse=True):
            track.clips.pop(idx)
        track.clips.insert(indices[0], combined)

        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._status_label.setText(f"Combined {len(clips)} clips (non-destructive)")

    def _combine_reencode(self, track, indices: list[int], clips: list) -> None:
        """Combine clips from different sources via ffmpeg background thread."""
        import subprocess
        from sdqt.workers.timeline import _probe_has_audio
        from sdqt.utils.codec import configured_codec_args as _codec_args, pix_fmt_args, configured_encoder_name

        logger.info("Combine (re-encode): %d clips from %s", len(clips), track.name)
        self._status_label.setText("Combining clips (encoding)...")

        def _probe_color_range(path: str) -> str:
            """Return ffmpeg in_range tag for *path* — 'tv' if limited, 'full' if full.

            AI-generated clips from this app are written as ``yuv420p`` (limited
            range) but the combine filter treated everything as full-range
            until now, which silently lifts blacks and desaturates colors.
            """
            try:
                r = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=color_range,pix_fmt",
                     "-of", "default=nw=1", path],
                    capture_output=True, text=True, timeout=15,
                )
                # Parse by key. ffprobe emits fields in stream order (pix_fmt
                # before color_range), NOT the order requested, so positional
                # parsing would swap them — misdetecting full-range (pc) clips
                # as tv and crushing them when the combine retags them tv.
                fields = {}
                for ln in r.stdout.splitlines():
                    k, sep, v = ln.strip().lower().partition("=")
                    if sep:
                        fields[k] = v
                color_range = fields.get("color_range", "")
                pix_fmt = fields.get("pix_fmt", "")
                if color_range in ("pc", "full"):
                    return "full"
                if color_range in ("tv", "limited"):
                    return "tv"
                # No tag — infer from pixel format. yuvj* / *p10le / rgb* are
                # commonly full; plain yuv420p with no tag is overwhelmingly
                # limited-range bt709 (which is what we encode).
                if pix_fmt.startswith("yuvj") or pix_fmt.startswith("rgb"):
                    return "full"
                return "tv"
            except Exception:
                return "tv"

        # Probe the first clip's native dimensions so we don't downscale
        # everything to a hardcoded 1280x720. All clips get scaled+padded
        # into the first clip's frame size.
        target_w, target_h = 1280, 720
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
                 clips[0].path],
                capture_output=True, text=True, check=True,
            )
            dims = r.stdout.strip().split("x")
            if len(dims) >= 2:
                target_w, target_h = int(dims[0]), int(dims[1])
        except Exception as exc:
            logger.warning("Combine: failed to probe first clip dims, using 1280x720: %s", exc)

        from sdqt.workers.video_effects import _build_video_filters
        from sdqt.utils.codec import matrix_convert_filter, _resolve_profile

        # Convert every clip into the ACTIVE PROFILE's range — the encode
        # below stamps the profile's tags (tv by default), so scaling to
        # out_range=full here (the old behavior) produced full-range data
        # tagged tv → crushed blacks on playback. Range AND matrix now both
        # follow the profile, with per-clip probes on the input side.
        _profile = _resolve_profile()
        _out_range = _profile.output_range()

        input_args = []
        filter_parts = []
        for i, clip in enumerate(clips):
            input_args.extend(["-i", clip.path])
            mo = getattr(clip, "media_offset", 0.0) or 0.0
            has_audio = _probe_has_audio(clip.path) if clip.has_audio else False
            vtag = f"cv{i}"
            # Probe the source's actual color range so limited sources get a
            # real conversion into the profile range, not a silent retag.
            in_range = _probe_color_range(clip.path)
            range_args = f"in_range={in_range}:out_range={_out_range}"
            # True matrix conversion for e.g. BT.601 imports (colorspace
            # filter — scale's matrix options are no-ops for YUV→YUV).
            conv = matrix_convert_filter(clip.path, _profile)
            conv = f"{conv}," if conv else ""
            # Geometry effects (crop_zoom, fractions of the SOURCE frame) must
            # run BEFORE the scale+pad that normalizes every segment to the
            # target size — applied after, they change the segment dimensions
            # and concat aborts with "Failed to inject frame into filter
            # network". Color effects stay after scale/pad so they operate on
            # the final-resolution frame the way the preview renders them.
            fx_list = clip.effects or []
            crop_filter = _build_video_filters(
                [f for f in fx_list if f.get("type") == "crop_zoom"])
            fx_filter = _build_video_filters(
                [f for f in fx_list if f.get("type") != "crop_zoom"])
            crop_prefix = f"{crop_filter}," if crop_filter else ""
            fx_suffix = f",{fx_filter}" if fx_filter else ""
            filter_parts.append(
                f"[{i}:v]trim={mo}:{mo + clip.duration},setpts=PTS-STARTPTS,"
                f"{conv}"
                f"{crop_prefix}"
                f"scale={target_w}:{target_h}:flags=lanczos:"
                f"force_original_aspect_ratio=decrease:"
                f"{range_args},"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,"
                f"setsar=1"
                f"{fx_suffix}[{vtag}]"
            )
            atag = f"ca{i}"
            if has_audio:
                filter_parts.append(
                    f"[{i}:a]aresample=44100,atrim={mo}:{mo + clip.duration},"
                    f"asetpts=PTS-STARTPTS[{atag}]"
                )
            else:
                filter_parts.append(
                    f"anullsrc=r=44100:cl=stereo:d={clip.duration},"
                    f"asetpts=PTS-STARTPTS[{atag}]"
                )

        n = len(clips)
        v_inputs = "".join(f"[cv{i}]" for i in range(n))
        a_inputs = "".join(f"[ca{i}]" for i in range(n))
        filter_parts.append(f"{v_inputs}concat=n={n}:v=1:a=0[outv]")
        filter_parts.append(f"{a_inputs}concat=n={n}:v=0:a=1[outa]")

        out_dir = self.project_path / "clips" / "combined"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = cap_stem(f"combined_{clips[0].name}_{len(clips)}clips") + ".mp4"
        out_path = str(out_dir / out_name)

        cmd = ["ffmpeg", "-y"] + input_args
        cmd += ["-filter_complex", ";".join(filter_parts)]
        cmd += ["-map", "[outv]", "-map", "[outa]"]
        cmd += [*_codec_args(), "-c:a", "aac", "-b:a", "192k", *pix_fmt_args(configured_encoder_name()), out_path]

        from sdqt.workers.base import BaseWorker

        class _CombineWorker(BaseWorker):
            def __init__(self, cmd, parent=None):
                super().__init__(parent)
                self._cmd = cmd
            def run(self):
                try:
                    self.progress.emit(0.1, "Encoding combined clip...")
                    r = subprocess.run(self._cmd, capture_output=True, text=True, timeout=300)
                    if r.returncode != 0:
                        self.error.emit(f"Combine failed: {r.stderr[-300:]}")
                        return
                    self.finished_ok.emit(out_path)
                except Exception as e:
                    self.error.emit(str(e))

        worker = _CombineWorker(cmd, parent=self)
        _track = track
        _indices = indices
        _clips = clips

        def _on_done(result_path):
            dur = self._probe_duration(result_path)
            thumb = self._generate_thumbnail(result_path)
            self._library.add_clip(result_path, dur, thumb)
            combined = TimelineClip(
                path=result_path, duration=dur,
                start_time=_clips[0].start_time,
                name=Path(result_path).stem, has_audio=True,
            )
            combined.thumbnail = thumb
            for idx in sorted(_indices, reverse=True):
                _track.clips.pop(idx)
            _track.clips.insert(_indices[0], combined)
            self._multitrack._sync_canvas()
            self._multitrack.clips_changed.emit()
            self._status_label.setText(
                f"Combined {len(_clips)} clips → {Path(result_path).name}"
            )

        worker.finished_ok.connect(_on_done)
        worker.error.connect(lambda msg: self._status_label.setText(f"Error: {msg}"))
        worker.progress.connect(lambda f, d: self._status_label.setText(d))
        worker.finished.connect(worker.deleteLater)
        self._combine_worker = worker
        worker.start()

    def _close_gaps(self, selection: list[tuple[str, int]]) -> None:
        self._multitrack.close_gaps(selection)
        self._multitrack._canvas.clear_selection()
        self._status_label.setText(f"Closed gaps on {len(selection)} clips")

    def _remove_multi(self, selection: list[tuple[str, int]]) -> None:
        """Remove multiple selected clips."""
        by_track: dict[str, list[int]] = {}
        for track_id, idx in selection:
            by_track.setdefault(track_id, []).append(idx)
        for track_id, indices in by_track.items():
            for idx in sorted(indices, reverse=True):
                self._multitrack.remove_clip_from_track(track_id, idx)
        self._multitrack._canvas.clear_selection()
        self._status_label.setText(f"Removed {len(selection)} clips")

    # -- Track management helpers -------------------------------------------

    def _select_all_on_track(self, track_id: str) -> None:
        track = self._multitrack.get_track(track_id)
        if not track:
            return
        canvas = self._multitrack._canvas
        canvas._multi_selection.clear()
        for i in range(len(track.clips)):
            canvas._multi_selection.append((track_id, i))
        if track.clips:
            canvas._selected_track_id = track_id
            canvas._selected_clip_idx = 0
        canvas.update()

    def _close_gaps_on_track(self, track_id: str) -> None:
        track = self._multitrack.get_track(track_id)
        if not track:
            return
        track.close_all_gaps()
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._status_label.setText(f"Gaps closed on {track.name}")

    def _move_track(self, track_id: str, direction: int) -> None:
        tracks = self._multitrack.tracks
        idx = next((i for i, t in enumerate(tracks) if t.id == track_id), -1)
        if idx < 0:
            return
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(tracks):
            return
        tracks[idx], tracks[new_idx] = tracks[new_idx], tracks[idx]
        self._multitrack._sync_canvas()
        self._multitrack._rebuild_headers()
        self._multitrack.clips_changed.emit()

    def _rename_track(self, track_id: str) -> None:
        track = self._multitrack.get_track(track_id)
        if not track:
            return
        name, ok = QInputDialog.getText(self, "Rename Track", "Track name:", text=track.name)
        if ok and name:
            track.name = name
            header = self._multitrack._headers.get(track_id)
            if header:
                header.set_name(name)
            self._multitrack.clips_changed.emit()

    def _delete_specific_track(self, track_id: str) -> None:
        if len(self._multitrack.tracks) <= 1:
            self._status_label.setText("Cannot delete the only track")
            return
        track = self._multitrack.get_track(track_id)
        name = track.name if track else "Track"
        self._multitrack.remove_track(track_id)
        self._status_label.setText(f"Deleted: {name}")

    def _duplicate_track(self, track_id: str) -> None:
        track = self._multitrack.get_track(track_id)
        if not track:
            return
        new_track = self._multitrack.add_track(track.track_type, name=f"{track.name} (copy)")
        for clip in track.clips:
            new_clip = TimelineClip(
                path=clip.path, duration=clip.duration, start_time=clip.start_time,
                name=clip.name, volume=clip.volume, has_audio=clip.has_audio,
                waveform=clip.waveform,
            )
            new_track.clips.append(new_clip)
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()

    # -- Bake / Unbake effects ------------------------------------------------

    def _bake_effects(self, track_id: str, clip_idx: int) -> None:
        """Render effects into the clip file. Stash originals for unbake."""
        import copy
        import subprocess
        import tempfile

        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        clip = track.clips[clip_idx]
        if not clip.effects:
            self._status_label.setText("No effects to bake")
            return

        from sdqt.workers.timeline import _build_effects_filter
        from sdqt.utils.codec import (
            configured_codec_args,
            pix_fmt_args,
            project_input_filter,
        )

        vf = _build_effects_filter(clip.effects)
        if not vf:
            self._status_label.setText("No active effects to bake")
            return
        # Probe the source's actual color range and convert it to the project's
        # output range FIRST. The previous code used full_range_filter() which
        # hardcodes in_range=full — wrong for TV-range generated clips, which
        # got their values double-compressed (blacks lifted, whites crushed).
        in_filter = project_input_filter(
            project_cfg=getattr(self, "project_config", None),
            global_cfg=getattr(self.state, "global_config", None) if hasattr(self, "state") else None,
            source_path=clip.path,
            add_setsar=False,  # bake doesn't need to enforce SAR
        )
        vf = f"{in_filter},{vf.lstrip(',')}"

        # Stash original state. On RE-bake (clip already baked once, new
        # effects added since) keep the FIRST bake's origin so unbake restores
        # the true pre-bake clip, and fold the newly baked effects into the
        # restore stack. Previously a baked clip silently ignored Bake —
        # "bake isn't saving my effects".
        if clip.bake_data:
            bake_data = dict(clip.bake_data)
            bake_data["original_effects"] = (
                copy.deepcopy(bake_data.get("original_effects") or [])
                + copy.deepcopy(clip.effects)
            )
        else:
            bake_data = {
                "original_path": clip.path,
                "original_effects": copy.deepcopy(clip.effects),
                "original_media_offset": clip.media_offset,
                "original_media_duration": clip.media_duration,
            }

        out_dir = self.project_path / "clips" / "baked"
        out_dir.mkdir(parents=True, exist_ok=True)
        _stem = cap_stem(clip.name)
        _out_p = out_dir / f"{_stem}_baked.mp4"
        _n = 1
        while _out_p.exists():
            # Never overwrite an existing bake — another clip (or this clip's
            # own current media) may reference it.
            _out_p = out_dir / f"{_stem}_baked_{_n}.mp4"
            _n += 1
        out_path = str(_out_p)

        self._status_label.setText(f"Baking effects on {clip.name}...")

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(clip.media_offset), "-i", clip.path,
            "-t", str(clip.duration),
            "-vf", vf,
            *configured_codec_args(),
            "-c:a", "aac", "-b:a", "192k",
            *pix_fmt_args(),
            out_path,
        ]

        from sdqt.workers.base import BaseWorker

        class _BakeWorker(BaseWorker):
            def __init__(self, cmd, parent=None):
                super().__init__(parent)
                self._cmd = cmd
            def run(self):
                try:
                    r = subprocess.run(self._cmd, capture_output=True, text=True, timeout=300)
                    if r.returncode != 0:
                        self.error.emit(f"Bake failed: {r.stderr[-300:]}")
                        return
                    self.finished_ok.emit(True)
                except Exception as e:
                    self.error.emit(str(e))

        worker = _BakeWorker(cmd, parent=self)

        def _on_done(_):
            clip.bake_data = bake_data
            clip.path = out_path
            clip.effects = []
            clip.media_offset = 0.0
            dur = self._probe_duration(out_path)
            # Keep the timeline duration (clamped to the new media) — the
            # encoder rounds to whole frames / pads audio, and adopting the
            # re-probed length grows the clip over its snapped neighbor.
            clip.duration = min(clip.duration, dur)
            clip.media_duration = dur
            self._multitrack._sync_canvas()
            self._multitrack.clips_changed.emit()
            self._save_state(force=True)
            self._status_label.setText(f"Baked: {clip.name}")

        worker.finished_ok.connect(_on_done)
        worker.error.connect(lambda msg: self._status_label.setText(f"Error: {msg}"))
        worker.finished.connect(worker.deleteLater)
        self._bake_worker = worker
        worker.start()

    def _unbake_effects(self, track_id: str, clip_idx: int) -> None:
        """Restore original clip path and effects from bake_data."""
        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        clip = track.clips[clip_idx]
        if not clip.bake_data:
            return

        bd = clip.bake_data
        clip.path = bd["original_path"]
        clip.effects = bd["original_effects"]
        clip.media_offset = bd["original_media_offset"]
        clip.media_duration = bd["original_media_duration"]
        clip.duration = clip.media_duration - clip.media_offset
        clip.bake_data = None

        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._save_state(force=True)
        self._status_label.setText(f"Unbaked: {clip.name}")

    # -- Clip rename --------------------------------------------------------

    def _rename_clip(self, track_id: str, clip_idx: int) -> None:
        """Rename a timeline clip's file — updates canvas, library, and disk."""
        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        self._rename_clip_file(track.clips[clip_idx].path)

    def _rename_clip_file(self, old_path: str) -> None:
        """Rename the clip's file on disk and propagate the new name to
        every timeline clip referencing it and to the library card."""
        from PySide6.QtWidgets import QMessageBox

        p = Path(old_path)
        if not p.is_file():
            QMessageBox.warning(self, "Rename Clip", f"File not found:\n{old_path}")
            return
        name, ok = QInputDialog.getText(
            self, "Rename Clip",
            "New name (renames the file on disk):",
            text=p.stem,
        )
        if not ok:
            return
        name = name.strip().strip(".")
        for ch in '/\\:*?"<>|\n\t':
            name = name.replace(ch, "_")
        if not name or name == p.stem:
            return
        new_path = p.with_name(name + p.suffix)
        if new_path.exists():
            QMessageBox.warning(
                self, "Rename Clip",
                f"A file named {new_path.name} already exists in that folder.",
            )
            return
        try:
            p.rename(new_path)
        except OSError as exc:
            QMessageBox.warning(self, "Rename Clip", f"Rename failed:\n{exc}")
            return

        new_str = str(new_path)
        for track in self._multitrack.tracks:
            for clip in track.clips:
                if clip.path == old_path:
                    clip.path = new_str
                    clip.name = name
                if clip.bake_data and clip.bake_data.get("original_path") == old_path:
                    clip.bake_data["original_path"] = new_str
        self._library.rename_clip(old_path, new_str)

        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._save_state(force=True)
        self._status_label.setText(f"Renamed to {new_path.name}")

    # -- Copy / Paste effects -------------------------------------------------

    def _copy_effects(self, track_id: str, clip_idx: int) -> None:
        """Copy effects from a clip to the class-level clipboard."""
        import copy
        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        clip = track.clips[clip_idx]
        TimelineContextMenus._copied_effects = copy.deepcopy(clip.effects)
        self._status_label.setText(
            f"Copied {len(clip.effects)} effect(s) from {clip.name}"
        )

    def _paste_effects(self, track_id: str, clip_idx: int) -> None:
        """Paste copied effects onto a single clip, replacing existing effects."""
        import copy
        if TimelineContextMenus._copied_effects is None:
            return
        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        clip = track.clips[clip_idx]
        clip.effects = copy.deepcopy(TimelineContextMenus._copied_effects)
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._status_label.setText(
            f"Pasted {len(clip.effects)} effect(s) onto {clip.name}"
        )

    def _paste_effects_multi(self, selection: list[tuple[str, int]]) -> None:
        """Paste copied effects onto multiple selected clips."""
        import copy
        if TimelineContextMenus._copied_effects is None:
            return
        count = 0
        for track_id, clip_idx in selection:
            track = self._multitrack.get_track(track_id)
            if not track or clip_idx < 0 or clip_idx >= len(track.clips):
                continue
            track.clips[clip_idx].effects = copy.deepcopy(
                TimelineContextMenus._copied_effects
            )
            count += 1
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._status_label.setText(
            f"Pasted {len(TimelineContextMenus._copied_effects)} effect(s) "
            f"onto {count} clip(s)"
        )

    def _clear_effects(self, track_id: str, clip_idx: int) -> None:
        """Remove all effects from a single clip."""
        track = self._multitrack.get_track(track_id)
        if not track or clip_idx < 0 or clip_idx >= len(track.clips):
            return
        clip = track.clips[clip_idx]
        clip.effects = []
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._save_state(force=True)
        self._status_label.setText(f"Cleared effects from {clip.name}")

    def _clear_effects_multi(self, selection: list[tuple[str, int]]) -> None:
        """Remove all effects from multiple selected clips."""
        count = 0
        for track_id, clip_idx in selection:
            track = self._multitrack.get_track(track_id)
            if not track or clip_idx < 0 or clip_idx >= len(track.clips):
                continue
            track.clips[clip_idx].effects = []
            count += 1
        self._multitrack._sync_canvas()
        self._multitrack.clips_changed.emit()
        self._save_state(force=True)
        self._status_label.setText(f"Cleared effects from {count} clip(s)")

    def _bake_effects_multi(self, selection: list[tuple[str, int]]) -> None:
        """Bake effects on multiple selected clips sequentially."""
        queue = []
        for track_id, clip_idx in selection:
            track = self._multitrack.get_track(track_id)
            if not track or clip_idx < 0 or clip_idx >= len(track.clips):
                continue
            clip = track.clips[clip_idx]
            if clip.effects:
                queue.append((track_id, clip_idx))
        if not queue:
            self._status_label.setText("No clips with bakeable effects")
            return
        self._bake_queue = queue
        self._bake_queue_total = len(queue)
        self._bake_next_in_queue()

    def _bake_next_in_queue(self) -> None:
        """Bake the next clip in the batch queue."""
        if not self._bake_queue:
            done = self._bake_queue_total
            self._status_label.setText(f"Baked effects on {done} clip(s)")
            return
        track_id, clip_idx = self._bake_queue.pop(0)
        remaining = len(self._bake_queue)
        total = self._bake_queue_total
        self._status_label.setText(
            f"Baking {total - remaining}/{total}..."
        )
        # Reuse single-clip bake, hooking into its completion
        self._bake_effects(track_id, clip_idx)
        # Chain: when the bake worker finishes, bake the next one
        if hasattr(self, "_bake_worker") and self._bake_worker is not None:
            self._bake_worker.finished.connect(self._bake_next_in_queue)

    def _send_clip_as_color_ref(self, clip_path: str, media_offset: float) -> None:
        """Extract a frame from the clip and send to Color Correct tab."""
        import tempfile
        from sdqt.utils.grade_compute import extract_frame_from_path
        from supremediffusion.utils.video import probe_video

        try:
            info = probe_video(clip_path)
            fps = info.get("fps", 16)
            frame_num = int(media_offset * fps)
        except Exception:
            frame_num = 0

        frame = extract_frame_from_path(clip_path, frame_num)
        if frame is not None:
            from PIL import Image
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            Image.fromarray(frame).save(tmp.name)
            self.send_cc_ref_to.emit(tmp.name)
