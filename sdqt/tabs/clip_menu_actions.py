"""Shared right-click menu helpers for Export / Quick Export / Create Loop.

Used by timeline, browser, and clip library context menus so all three surfaces
get the same actions without duplicating wiring.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
)

from sdqt.utils.naming import cap_stem

logger = logging.getLogger(__name__)


def _load_project_config(parent):
    """Best-effort load of the active project's ProjectConfig.

    Workers receive this so their thread-local color profile gets set to
    the project's profile instead of falling back to the global default.
    """
    try:
        from supremediffusion.config.project_config import ProjectConfig
        project_path = getattr(parent, "project_path", None)
        if project_path:
            return ProjectConfig.load(project_path)
    except Exception:
        logger.debug("clip_menu_actions: project_config load failed", exc_info=True)
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def add_clip_export_actions(
    menu: QMenu,
    parent,
    state,
    clip_path: str,
    *,
    media_offset: float = 0.0,
    duration: float = 0.0,
    clip_name: str = "",
    effects: list[dict] | None = None,
    on_loop_saved: Callable[[str], None] | None = None,
) -> None:
    """Append Export / Quick Export / Create Loop actions to *menu*.

    Parameters
    ----------
    menu :
        The QMenu to append to. A separator is added first.
    parent :
        The parent widget for dialogs.
    state :
        AppState instance (for global_config.last_export_dir + model paths).
    clip_path :
        Source video file path.
    media_offset, duration :
        Optional clip trim. If ``duration`` is 0, the worker will treat the
        whole file as the segment.
    clip_name :
        Label used to default output filenames.
    on_loop_saved :
        Optional callback invoked with the saved loop path when the Create
        Loop dialog's Save button is clicked.
    """
    if not clip_name:
        clip_name = Path(clip_path).stem

    menu.addSeparator()

    export_act = menu.addAction("Export…")
    export_act.triggered.connect(
        lambda _checked=False, p=clip_path, mo=media_offset, d=duration, n=clip_name:
            run_export_dialog(parent, state, p, mo, d, n)
    )

    quick_act = menu.addAction("Quick Export…")
    if effects:
        quick_act.setToolTip(
            "Export the clip exactly as shown in the timeline preview "
            "(LUTs, effects, trim all baked in) — no prompts."
        )
    else:
        quick_act.setToolTip(
            "RIFE 2x + denoise + sharpen + face restore + lanczos 1.5x — no prompts"
        )
    _fx = list(effects) if effects else None
    quick_act.triggered.connect(
        lambda _checked=False, p=clip_path, mo=media_offset, d=duration, n=clip_name, fx=_fx:
            run_quick_export(parent, state, p, mo, d, n, effects=fx)
    )

    if _is_video(clip_path):
        loop_act = menu.addAction("Create Loop…")
        loop_act.triggered.connect(
            lambda _checked=False, p=clip_path, mo=media_offset, d=duration, n=clip_name:
                run_create_loop(parent, state, p, mo, d, n, on_loop_saved)
        )


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def run_export_dialog(
    parent,
    state,
    clip_path: str,
    media_offset: float,
    duration: float,
    clip_name: str,
) -> None:
    """Open the Timeline export dialog wrapping a single clip as a 1-track list."""
    from sdqt.tabs.timeline_export import ExportDialog
    from sdqt.workers.timeline import TimelineExportWorker
    from sdqt.widgets.timeline_track import TimelineTrack, TimelineClip, TrackType

    default_dir = state.global_config.last_export_dir or str(Path.home())
    default_out = str(Path(default_dir) / f"{cap_stem(clip_name)}_export.mp4")

    dlg = ExportDialog(default_out, parent=parent)
    if dlg.exec() != QDialog.Accepted:
        return
    settings = dlg.get_settings()
    output = settings["output"]
    if not output:
        return
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    # Persist the directory
    try:
        state.global_config.last_export_dir = str(Path(output).parent)
        state.global_config.save()
    except Exception:
        pass

    # Build a synthetic single-clip track
    dur = duration if duration > 0 else _probe_duration(clip_path)
    if dur <= 0:
        _error(parent, "Export", "Could not determine clip duration.")
        return

    track = TimelineTrack(name=clip_name, track_type=TrackType.VIDEO)
    track.clips.append(TimelineClip(
        path=clip_path,
        duration=dur,
        start_time=0.0,
        media_offset=media_offset,
        media_duration=max(dur, media_offset + dur),
        has_audio=True,
        name=clip_name,
    ))

    worker = TimelineExportWorker(
        output=output,
        tracks=[track],
        resolution=settings.get("resolution", ""),
        rife_mode=settings.get("rife_mode", ""),
        codec=settings.get("codec", ""),
        parent=parent,
    )
    _run_with_progress(parent, worker, "Export", output)


def _ask_apply_post_process(parent, count: int = 1) -> bool | None:
    """Prompt the user to toggle the AI post-process preset for Quick Export.

    Returns True (apply preset), False (skip preset — bake+encode only),
    or None if the user cancels.
    """
    box = QMessageBox(parent)
    box.setWindowTitle("Quick Export")
    noun = "clip" if count == 1 else f"{count} clips"
    box.setText(f"Use Post Processing on the {noun}?")
    box.setInformativeText(
        "<b>Yes</b> runs denoise + sharpen + face restore + lanczos 1.5x + "
        "RIFE 2x on top of the baked timeline effects.<br><br>"
        "<b>No</b> only bakes the timeline effects and trim — use this if "
        "the source has already been post-processed (running it twice "
        "compounds upscaling, frame-rate, and sharpening)."
    )
    box.setIcon(QMessageBox.Question)
    godot_cb = QCheckBox("Godot (yuv420p, tv range, bt709)")
    godot_cb.setToolTip("Export with limited-range color for Godot/game engine import")
    box.setCheckBox(godot_cb)
    yes_btn = box.addButton("Yes", QMessageBox.YesRole)
    no_btn = box.addButton("No", QMessageBox.NoRole)
    box.addButton(QMessageBox.Cancel)
    box.setDefaultButton(yes_btn)
    box.exec()
    clicked = box.clickedButton()
    if clicked == yes_btn:
        return True, godot_cb.isChecked()
    if clicked == no_btn:
        return False, godot_cb.isChecked()
    return None, False


def run_quick_export(
    parent,
    state,
    clip_path: str,
    media_offset: float,
    duration: float,
    clip_name: str,
    *,
    effects: list[dict] | None = None,
) -> None:
    """Run Quick Export.

    When called from a timeline/multitrack context, ``effects`` is the
    clip's effect stack; the worker bakes those effects + the trim in a
    single pass so the output matches what's shown in the timeline
    preview exactly. The user is asked whether to apply the AI post-
    process preset (denoise/sharpen/face/lanczos/RIFE2x) on top — "No"
    gives a pure WYSIWYG bake so already-post-processed source clips
    don't get double-processed.
    """
    from sdqt.workers.clip_quick_export import ClipPostProcessWorker

    result = _ask_apply_post_process(parent, count=1)
    apply_pp, godot_mode = result
    if apply_pp is None:
        return  # user cancelled

    default_dir = state.global_config.last_export_dir or str(Path.home())
    default_out = str(Path(default_dir) / f"{cap_stem(clip_name)}_quick.mp4")
    output, _ = QFileDialog.getSaveFileName(
        parent, "Quick Export", default_out,
        "Video (*.mp4);;All Files (*)",
    )
    if not output:
        return
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    try:
        state.global_config.last_export_dir = str(Path(output).parent)
        state.global_config.save()
    except Exception:
        pass

    upscaler_dir = state.global_config.model_paths.get("upscaler_dir", "")
    face_models_dir = state.global_config.model_paths.get("face_models_dir", "")

    # If the caller is the TimelineTab and it has a GL preview available,
    # the WYSIWYG bake goes through the GL render pipeline — we temporarily
    # swap gl_preview's tracks to a single-clip list, drive frame-by-frame
    # rendering via GLExportController (which samples the same shader path
    # the timeline preview uses), then chain the AI post-process onto the
    # rendered result. This is the only way to guarantee the output matches
    # exactly what's shown in the preview.
    gl_preview = getattr(parent, "_gl_preview", None)
    if (
        effects is not None
        and gl_preview is not None
        and getattr(gl_preview, "available", False)
        and hasattr(parent, "_multitrack")
    ):
        _run_quick_export_via_gl(
            parent, state, clip_path, media_offset, duration, clip_name,
            effects, output, apply_pp, upscaler_dir, face_models_dir,
            godot_mode=godot_mode,
        )
        return

    # Non-timeline fallback: use the ffmpeg lut3d bake path inside the worker.
    tl_fps = getattr(parent, "timeline_fps", 0)
    project_cfg = _load_project_config(parent)
    worker = ClipPostProcessWorker(
        source=clip_path,
        output=output,
        media_offset=media_offset,
        duration=duration,
        effects=effects,
        enable_denoise=apply_pp,
        enable_sharpen=apply_pp,
        enable_face=apply_pp,
        enable_upscale=False,
        lanczos_mode="lanczos1.5" if apply_pp else "",
        rife_mode="x2" if apply_pp else "",
        upscaler_dir=upscaler_dir,
        face_models_dir=face_models_dir,
        godot_mode=godot_mode,
        target_fps=tl_fps,
        project_config=project_cfg,
        global_config=state.global_config,
        parent=parent,
    )
    _run_with_progress(parent, worker, "Quick Export", output)


def _run_quick_export_via_gl(
    parent,
    state,
    clip_path: str,
    media_offset: float,
    duration: float,
    clip_name: str,
    effects: list[dict],
    output: str,
    apply_pp: bool,
    upscaler_dir: str,
    face_models_dir: str,
    godot_mode: bool = False,
) -> None:
    """GL-rendered Quick Export — captures the exact pixels the timeline
    preview shows, then optionally runs the AI post-process on them."""
    import tempfile
    import shutil
    from sdqt.widgets.timeline_track import TimelineTrack, TimelineClip, TrackType
    from sdqt.workers.gl_export import GLExportController
    from sdqt.workers.clip_quick_export import ClipPostProcessWorker

    gl = parent._gl_preview
    # Save the timeline's real track list so we can restore it when we're done
    saved_tracks = list(parent._multitrack.tracks)

    # Probe source dimensions; use timeline FPS for the render target
    try:
        w, h, _ = _probe_wh_fps(clip_path)
    except Exception as exc:
        QMessageBox.critical(parent, "Quick Export", f"Could not probe source: {exc}")
        return
    fps = getattr(parent, "timeline_fps", None) or _probe_wh_fps(clip_path)[2]

    # Build a synthetic single-clip timeline with the full effect stack so
    # gl_preview's shaders apply everything (LUTs, effects, trim) as it
    # does for the normal timeline preview. start_time=0 so GLExportController
    # iterates the clip from its first frame.
    dur = duration if duration > 0 else 0.0
    if dur <= 0:
        dur = _probe_duration(clip_path)
    if dur <= 0:
        QMessageBox.critical(parent, "Quick Export", "Could not determine clip duration.")
        return

    synthetic_clip = TimelineClip(
        path=clip_path,
        duration=dur,
        start_time=0.0,
        media_offset=media_offset,
        media_duration=max(dur, media_offset + dur),
        has_audio=True,
        name=clip_name or Path(clip_path).stem,
        effects=list(effects),
    )
    synthetic_track = TimelineTrack(name="quick_export", track_type=TrackType.VIDEO)
    synthetic_track.clips.append(synthetic_clip)

    # Swap gl_preview state
    try:
        gl.set_timeline_data([synthetic_track])
    except Exception as exc:
        QMessageBox.critical(parent, "Quick Export", f"GL preview swap failed: {exc}")
        return

    # Render to a temp file — uses GL shaders → pipes raw RGB → ffmpeg
    # yuvj420p encode (via pix_fmt_args()).
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, prefix="clip_qe_gl_")
    tmp.close()

    audio_tracks = [{"path": clip_path, "media_offset": media_offset}]

    ctrl = GLExportController(
        gl_preview=gl,
        output=tmp.name,
        width=w, height=h,
        fps=float(fps),
        codec="",
        rife_mode="",
        audio_tracks=audio_tracks,
        godot_mode=godot_mode,
        parent=parent,
    )

    prog = QProgressDialog("GL render: starting…", "Cancel", 0, 100, parent)
    prog.setWindowTitle("Quick Export (GL)")
    prog.setAutoClose(False)
    prog.setAutoReset(False)
    prog.setMinimumDuration(0)
    prog.setValue(0)

    def _restore_gl():
        try:
            gl.set_timeline_data(saved_tracks)
        except Exception:
            logger.warning("Failed to restore gl_preview tracks", exc_info=True)

    def _on_gl_progress(frac: float, desc: str) -> None:
        # GL render gets the first 50% of the bar when PP is on, else 100%.
        total = frac * (0.5 if apply_pp else 1.0)
        prog.setValue(int(total * 100))
        prog.setLabelText(f"Quick Export: {desc}")

    def _on_gl_done(gl_path: str) -> None:
        _restore_gl()
        if not apply_pp:
            # Move rendered file straight to the user's chosen output.
            try:
                shutil.move(gl_path, output)
            except Exception as exc:
                prog.close()
                QMessageBox.critical(parent, "Quick Export", f"Move failed: {exc}")
                return
            prog.setValue(100)
            prog.close()
            _result_dialog(parent, "Quick Export", output)
            return

        # PP requested — chain the AI pipeline on the rendered file.
        # effects=None so the post-process worker doesn't re-bake (we
        # already did that through GL).
        prog.setLabelText("Quick Export: starting post-process…")
        project_cfg_pp = _load_project_config(parent)
        worker = ClipPostProcessWorker(
            source=gl_path,
            output=output,
            media_offset=0.0,
            duration=0.0,
            effects=None,
            enable_denoise=True,
            enable_sharpen=True,
            enable_face=True,
            enable_upscale=False,
            lanczos_mode="lanczos1.5",
            rife_mode="x2",
            upscaler_dir=upscaler_dir,
            face_models_dir=face_models_dir,
            godot_mode=godot_mode,
            target_fps=float(fps),
            project_config=project_cfg_pp,
            global_config=state.global_config,
            parent=parent,
        )

        def _on_pp_progress(frac: float, desc: str) -> None:
            total = 0.5 + frac * 0.5
            prog.setValue(int(total * 100))
            prog.setLabelText(f"Quick Export: {desc}")

        def _on_pp_done(_result) -> None:
            prog.close()
            worker.deleteLater()
            try:
                Path(gl_path).unlink(missing_ok=True)
            except Exception:
                pass
            _result_dialog(parent, "Quick Export", output)

        def _on_pp_error(msg: str) -> None:
            prog.close()
            worker.deleteLater()
            QMessageBox.critical(parent, "Quick Export Failed", msg)

        worker.progress.connect(_on_pp_progress)
        worker.finished_ok.connect(_on_pp_done)
        worker.error.connect(_on_pp_error)
        worker.start()

    def _on_gl_error(msg: str) -> None:
        _restore_gl()
        prog.close()
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except Exception:
            pass
        QMessageBox.critical(parent, "Quick Export (GL) Failed", msg)

    def _on_cancel() -> None:
        if ctrl._aborted:
            return
        try:
            ctrl.abort()
        except Exception:
            pass
        _restore_gl()

    ctrl.progress.connect(_on_gl_progress)
    ctrl.finished.connect(_on_gl_done)
    ctrl.error.connect(_on_gl_error)
    prog.canceled.connect(_on_cancel)
    ctrl.start()


def _probe_wh_fps(path: str) -> tuple[int, int, float]:
    import subprocess
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "csv=p=0:s=x", path],
        capture_output=True, text=True, check=True,
    )
    parts = r.stdout.strip().split("x")
    w, h = int(parts[0]), int(parts[1])
    fps_expr = parts[2] if len(parts) > 2 else "24/1"
    if "/" in fps_expr:
        num, den = fps_expr.split("/")
        fps = float(num) / float(den) if float(den) else 24.0
    else:
        fps = float(fps_expr)
    return w, h, fps


def run_quick_export_batch(
    parent,
    state,
    items: list[dict],
    output_dir: str,
) -> None:
    """Sequentially quick-export a list of timeline clips into *output_dir*.

    Each ``item`` is a dict with ``path``, ``effects``, ``media_offset``,
    ``duration``, ``name``. One ``QProgressDialog`` tracks the whole batch;
    cancelling it aborts the current worker and stops the chain.
    """
    from sdqt.workers.clip_quick_export import ClipPostProcessWorker

    if not items:
        return

    apply_pp, godot_mode = _ask_apply_post_process(parent, count=len(items))
    if apply_pp is None:
        return  # user cancelled

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    total = len(items)
    prog = QProgressDialog(f"Quick Export — 0/{total}", "Cancel", 0, 100, parent)
    prog.setWindowTitle(f"Quick Export {total} clips")
    prog.setAutoClose(False)
    prog.setAutoReset(False)
    prog.setMinimumDuration(0)
    prog.setValue(0)

    upscaler_dir = state.global_config.model_paths.get("upscaler_dir", "")
    face_models_dir = state.global_config.model_paths.get("face_models_dir", "")

    # Mutable state for the chain (captured by nested callbacks)
    state_obj = {
        "index": 0,
        "ok": 0,
        "errored": 0,
        "aborted": False,
        "worker": None,
        "last_output": "",
    }

    def _unique_path(base_name: str) -> Path:
        stem = cap_stem(Path(base_name).stem if base_name else "clip")
        dest = out_root / f"{stem}.mp4"
        n = 1
        while dest.exists():
            dest = out_root / f"{stem}_{n}.mp4"
            n += 1
        return dest

    def _start_next():
        i = state_obj["index"]
        if state_obj["aborted"] or i >= total:
            _finish()
            return
        item = items[i]
        output_path = str(_unique_path(item.get("name", "")))
        state_obj["last_output"] = output_path

        prog.setLabelText(
            f"Quick Export — {i + 1}/{total}: {Path(output_path).name}"
        )
        prog.setValue(int((i / total) * 100))

        try:
            project_cfg_batch = _load_project_config(parent)
            worker = ClipPostProcessWorker(
                source=item["path"],
                output=output_path,
                media_offset=float(item.get("media_offset", 0.0) or 0.0),
                duration=float(item.get("duration", 0.0) or 0.0),
                effects=list(item.get("effects") or []),
                enable_denoise=apply_pp,
                enable_sharpen=apply_pp,
                enable_face=apply_pp,
                enable_upscale=False,
                lanczos_mode="lanczos1.5" if apply_pp else "",
                rife_mode="x2" if apply_pp else "",
                upscaler_dir=upscaler_dir,
                face_models_dir=face_models_dir,
                godot_mode=godot_mode,
                project_config=project_cfg_batch,
                global_config=state.global_config,
                parent=parent,
            )
        except Exception as exc:
            logger.exception("Batch quick-export worker init failed")
            state_obj["errored"] += 1
            state_obj["index"] += 1
            _start_next()
            return

        def _on_progress(frac: float, desc: str, idx=i) -> None:
            # Map this item's 0..1 progress onto its slice of the total bar
            total_frac = (idx + max(0.0, min(1.0, frac))) / total
            prog.setValue(int(total_frac * 100))
            prog.setLabelText(
                f"Quick Export — {idx + 1}/{total}: {desc}"
            )

        def _on_done(result) -> None:
            state_obj["ok"] += 1
            state_obj["worker"] = None
            state_obj["index"] += 1
            _start_next()

        def _on_error(msg: str) -> None:
            logger.warning("Batch quick-export item %d failed: %s", i + 1, msg)
            state_obj["errored"] += 1
            state_obj["worker"] = None
            state_obj["index"] += 1
            _start_next()

        worker.progress.connect(_on_progress)
        worker.finished_ok.connect(_on_done)
        worker.error.connect(_on_error)
        worker.finished.connect(worker.deleteLater)
        state_obj["worker"] = worker
        worker.start()

    def _on_cancel():
        state_obj["aborted"] = True
        w = state_obj["worker"]
        if w is not None:
            try:
                w.abort()
            except Exception:
                pass

    def _finish():
        prog.close()
        ok, err = state_obj["ok"], state_obj["errored"]
        msg = f"Quick Export batch complete: {ok} ok"
        if err:
            msg += f", {err} errored"
        if state_obj["aborted"]:
            msg += " (aborted)"
        box = QMessageBox(parent)
        box.setWindowTitle("Quick Export")
        box.setText(msg + f"\n\nOutput: {output_dir}")
        open_btn = box.addButton("Open Folder", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Ok)
        box.exec()
        if box.clickedButton() == open_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir))

    prog.canceled.connect(_on_cancel)
    _start_next()


def run_create_loop(
    parent,
    state,
    clip_path: str,
    media_offset: float,
    duration: float,
    clip_name: str,
    on_loop_saved: Callable[[str], None] | None,
) -> None:
    from sdqt.tabs.timeline_loop_dialog import LoopDialog

    dur = duration if duration > 0 else _probe_duration(clip_path)
    dlg = LoopDialog(
        source=clip_path,
        media_offset=media_offset,
        duration=dur,
        clip_name=clip_name,
        state=state,
        parent=parent,
    )
    if on_loop_saved is not None:
        dlg.loop_saved.connect(on_loop_saved)
    dlg.exec()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _is_video(path: str) -> bool:
    return Path(path).suffix.lower() in _VIDEO_EXTS


def _probe_duration(path: str) -> float:
    try:
        from supremediffusion.utils.video import probe_video
        info = probe_video(path)
        dur = info.get("duration")
        if dur:
            return float(dur)
        frames = info.get("num_frames")
        fps = info.get("fps")
        if frames and fps:
            return float(frames) / float(fps)
    except Exception:
        pass
    # ffprobe format.duration fallback (audio-only safe)
    try:
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip() or 0.0)
    except Exception:
        return 0.0


def _run_with_progress(parent, worker, title: str, output_path: str) -> None:
    """Run *worker* with a modal QProgressDialog, show result message when done."""
    prog = QProgressDialog(f"{title}…", "Cancel", 0, 100, parent)
    prog.setWindowTitle(title)
    prog.setAutoClose(False)
    prog.setAutoReset(False)
    prog.setMinimumDuration(0)
    prog.setValue(0)

    def _on_progress(frac: float, desc: str) -> None:
        prog.setValue(int(max(0.0, min(1.0, frac)) * 100))
        prog.setLabelText(f"{title}: {desc}")

    def _on_done(result) -> None:
        prog.close()
        worker.deleteLater()
        _result_dialog(parent, title, str(result) if result else output_path)

    def _on_error(msg: str) -> None:
        prog.close()
        worker.deleteLater()
        QMessageBox.critical(parent, f"{title} Failed", msg)

    def _on_cancel() -> None:
        worker.abort()

    worker.progress.connect(_on_progress)
    worker.finished_ok.connect(_on_done)
    worker.error.connect(_on_error)
    prog.canceled.connect(_on_cancel)
    worker.start()


def _result_dialog(parent, title: str, path: str) -> None:
    box = QMessageBox(parent)
    box.setWindowTitle(f"{title} Complete")
    box.setText(f"{title} complete:\n{path}")
    box.setIcon(QMessageBox.Information)
    open_btn = box.addButton("Open Folder", QMessageBox.ActionRole)
    box.addButton(QMessageBox.Ok)
    box.exec()
    if box.clickedButton() == open_btn:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))


def _error(parent, title: str, msg: str) -> None:
    QMessageBox.warning(parent, title, msg)
