"""Keyframe animation timeline — data model + compact timeline bar widget."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QMouseEvent
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class Keyframe:
    """A single keyframe storing transform state at a given frame."""
    frame: int
    rot_x: float = 0.0
    rot_y: float = 0.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    zoom: float = 3.0
    # Mesh transforms (for mesh tracks)
    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_z: float = 0.0
    rot_mx: float = 0.0
    rot_my: float = 0.0
    rot_mz: float = 0.0
    scale: float = 1.0


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _angle_lerp(a: float, b: float, t: float) -> float:
    """Shortest-path interpolation for angles in degrees."""
    diff = (b - a) % 360
    if diff > 180:
        diff -= 360
    return a + diff * t


@dataclass
class AnimationTrack:
    """A sequence of keyframes for one target (camera or mesh index)."""
    target: str  # "camera" or "mesh_0", "mesh_1", etc.
    keyframes: list[Keyframe] = field(default_factory=list)

    def add_keyframe(self, kf: Keyframe) -> None:
        """Insert or replace a keyframe at the given frame number."""
        # Remove existing keyframe at same frame
        self.keyframes = [k for k in self.keyframes if k.frame != kf.frame]
        self.keyframes.append(kf)
        self.keyframes.sort(key=lambda k: k.frame)

    def remove_keyframe(self, frame: int) -> None:
        self.keyframes = [k for k in self.keyframes if k.frame != frame]

    def interpolate(self, frame: float) -> Keyframe:
        """Interpolate between surrounding keyframes."""
        if not self.keyframes:
            return Keyframe(frame=int(frame))
        if len(self.keyframes) == 1:
            return self.keyframes[0]

        # Clamp to range
        if frame <= self.keyframes[0].frame:
            return self.keyframes[0]
        if frame >= self.keyframes[-1].frame:
            return self.keyframes[-1]

        # Find surrounding keyframes
        prev = self.keyframes[0]
        nxt = self.keyframes[-1]
        for i in range(len(self.keyframes) - 1):
            if self.keyframes[i].frame <= frame <= self.keyframes[i + 1].frame:
                prev = self.keyframes[i]
                nxt = self.keyframes[i + 1]
                break

        # Compute t
        span = nxt.frame - prev.frame
        t = (frame - prev.frame) / span if span > 0 else 0.0

        return Keyframe(
            frame=int(frame),
            # Camera
            rot_x=_lerp(prev.rot_x, nxt.rot_x, t),
            rot_y=_angle_lerp(prev.rot_y, nxt.rot_y, t),
            pan_x=_lerp(prev.pan_x, nxt.pan_x, t),
            pan_y=_lerp(prev.pan_y, nxt.pan_y, t),
            zoom=_lerp(prev.zoom, nxt.zoom, t),
            # Mesh
            pos_x=_lerp(prev.pos_x, nxt.pos_x, t),
            pos_y=_lerp(prev.pos_y, nxt.pos_y, t),
            pos_z=_lerp(prev.pos_z, nxt.pos_z, t),
            rot_mx=_angle_lerp(prev.rot_mx, nxt.rot_mx, t),
            rot_my=_angle_lerp(prev.rot_my, nxt.rot_my, t),
            rot_mz=_angle_lerp(prev.rot_mz, nxt.rot_mz, t),
            scale=_lerp(prev.scale, nxt.scale, t),
        )


class AnimationState:
    """Container for all animation tracks + timing."""

    def __init__(self, fps: int = 24, duration_seconds: float = 5.0) -> None:
        self.fps = fps
        self.duration_seconds = duration_seconds
        self.tracks: dict[str, AnimationTrack] = {}

    @property
    def duration_frames(self) -> int:
        return int(self.fps * self.duration_seconds)

    def get_or_create_track(self, target: str) -> AnimationTrack:
        if target not in self.tracks:
            self.tracks[target] = AnimationTrack(target=target)
        return self.tracks[target]

    def evaluate(self, frame: int) -> dict[str, Keyframe]:
        """Evaluate all tracks at a given frame. Returns {target: Keyframe}."""
        return {target: track.interpolate(frame) for target, track in self.tracks.items()}

    def clear(self) -> None:
        self.tracks.clear()

    def has_keyframes(self) -> bool:
        return any(len(t.keyframes) > 0 for t in self.tracks.values())


# ── Animation presets ────────────────────────────────────────────────────────

def preset_turntable(state: AnimationState, cam_state: dict) -> None:
    """360° camera orbit around the scene."""
    track = state.get_or_create_track("camera")
    track.keyframes.clear()
    total = state.duration_frames
    base_rot_x = cam_state.get("rot_x", -20)
    base_zoom = cam_state.get("zoom", 3.0)
    base_pan_x = cam_state.get("pan_x", 0.0)
    base_pan_y = cam_state.get("pan_y", 0.0)

    track.add_keyframe(Keyframe(
        frame=0, rot_x=base_rot_x, rot_y=0,
        pan_x=base_pan_x, pan_y=base_pan_y, zoom=base_zoom,
    ))
    track.add_keyframe(Keyframe(
        frame=total, rot_x=base_rot_x, rot_y=360,
        pan_x=base_pan_x, pan_y=base_pan_y, zoom=base_zoom,
    ))


def preset_dolly_in(state: AnimationState, cam_state: dict) -> None:
    """Camera zooms from far to close."""
    track = state.get_or_create_track("camera")
    track.keyframes.clear()
    total = state.duration_frames
    rot_x = cam_state.get("rot_x", -20)
    rot_y = cam_state.get("rot_y", 45)

    track.add_keyframe(Keyframe(
        frame=0, rot_x=rot_x, rot_y=rot_y, zoom=8.0,
    ))
    track.add_keyframe(Keyframe(
        frame=total, rot_x=rot_x, rot_y=rot_y, zoom=1.5,
    ))


def preset_object_spin(state: AnimationState, mesh_idx: int = 0) -> None:
    """Selected mesh rotates 360° in place."""
    target = f"mesh_{mesh_idx}"
    track = state.get_or_create_track(target)
    track.keyframes.clear()
    total = state.duration_frames

    track.add_keyframe(Keyframe(frame=0, rot_my=0))
    track.add_keyframe(Keyframe(frame=total, rot_my=360))


def preset_reveal(state: AnimationState, cam_state: dict) -> None:
    """Camera starts close, pulls back to reveal full scene."""
    track = state.get_or_create_track("camera")
    track.keyframes.clear()
    total = state.duration_frames
    rot_x = cam_state.get("rot_x", -20)
    rot_y = cam_state.get("rot_y", 45)

    track.add_keyframe(Keyframe(
        frame=0, rot_x=rot_x - 10, rot_y=rot_y, zoom=1.2, pan_y=0.3,
    ))
    track.add_keyframe(Keyframe(
        frame=total, rot_x=rot_x, rot_y=rot_y, zoom=cam_state.get("zoom", 3.0), pan_y=0.0,
    ))


def preset_orbit_rise(state: AnimationState, cam_state: dict) -> None:
    """Camera spirals upward while orbiting."""
    track = state.get_or_create_track("camera")
    track.keyframes.clear()
    total = state.duration_frames
    base_zoom = cam_state.get("zoom", 3.0)

    n_keys = 5
    for i in range(n_keys + 1):
        t = i / n_keys
        track.add_keyframe(Keyframe(
            frame=int(t * total),
            rot_x=-10 - t * 40,  # rise from -10 to -50
            rot_y=t * 360,
            zoom=base_zoom,
        ))


# ── Timeline bar widget ──────────────────────────────────────────────────────

class _TimelineBar(QWidget):
    """Custom-painted timeline bar showing tracks, keyframes, and playhead."""

    frame_clicked = Signal(int)  # user clicked on a frame position

    TRACK_HEIGHT = 24
    HEADER_HEIGHT = 20
    DIAMOND_SIZE = 8

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._anim: AnimationState | None = None
        self._current_frame = 0
        self._dragging_playhead = False
        self.setMinimumHeight(self.HEADER_HEIGHT + self.TRACK_HEIGHT * 2 + 4)
        self.setMouseTracking(True)

    def set_animation(self, anim: AnimationState) -> None:
        self._anim = anim
        self.update()

    def set_current_frame(self, frame: int) -> None:
        self._current_frame = frame
        self.update()

    def _frame_to_x(self, frame: int) -> float:
        if not self._anim or self._anim.duration_frames <= 0:
            return 0
        return 40 + (self.width() - 50) * frame / self._anim.duration_frames

    def _x_to_frame(self, x: float) -> int:
        if not self._anim or self._anim.duration_frames <= 0:
            return 0
        f = (x - 40) / max(1, self.width() - 50) * self._anim.duration_frames
        return max(0, min(self._anim.duration_frames, int(f)))

    def paintEvent(self, event) -> None:
        if not self._anim:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        total = self._anim.duration_frames

        # Background
        p.fillRect(0, 0, w, h, QColor(30, 30, 30))

        # Frame ruler (header)
        p.setPen(QPen(QColor(100, 100, 100), 1))
        step = max(1, total // 20)
        for f in range(0, total + 1, step):
            x = self._frame_to_x(f)
            p.drawLine(int(x), 0, int(x), self.HEADER_HEIGHT)
            if f % (step * 2) == 0:
                p.setPen(QColor(160, 160, 160))
                p.drawText(int(x) - 10, self.HEADER_HEIGHT - 3, f"{f}")
                p.setPen(QPen(QColor(100, 100, 100), 1))

        # Track lanes
        y = self.HEADER_HEIGHT
        track_names = sorted(self._anim.tracks.keys())
        for i, name in enumerate(track_names):
            track = self._anim.tracks[name]
            ty = y + i * self.TRACK_HEIGHT

            # Track label
            p.setPen(QColor(140, 140, 140))
            label = "Cam" if name == "camera" else name.replace("mesh_", "M")
            p.drawText(2, ty + 16, label)

            # Lane background
            p.fillRect(40, ty, w - 50, self.TRACK_HEIGHT, QColor(40, 40, 40))

            # Keyframe diamonds
            for kf in track.keyframes:
                kx = self._frame_to_x(kf.frame)
                d = self.DIAMOND_SIZE
                p.setPen(Qt.PenStyle.NoPen)
                color = QColor(255, 180, 50) if name == "camera" else QColor(100, 200, 255)
                p.setBrush(QBrush(color))
                # Diamond shape
                pts = [
                    (kx, ty + self.TRACK_HEIGHT // 2 - d // 2),
                    (kx + d // 2, ty + self.TRACK_HEIGHT // 2),
                    (kx, ty + self.TRACK_HEIGHT // 2 + d // 2),
                    (kx - d // 2, ty + self.TRACK_HEIGHT // 2),
                ]
                from PySide6.QtCore import QPointF
                from PySide6.QtGui import QPolygonF
                poly = QPolygonF([QPointF(px, py) for px, py in pts])
                p.drawPolygon(poly)

        # Playhead
        px = self._frame_to_x(self._current_frame)
        p.setPen(QPen(QColor(255, 80, 80), 2))
        p.drawLine(int(px), 0, int(px), h)

        p.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_playhead = True
            frame = self._x_to_frame(event.position().x())
            self._current_frame = frame
            self.frame_clicked.emit(frame)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging_playhead:
            frame = self._x_to_frame(event.position().x())
            self._current_frame = frame
            self.frame_clicked.emit(frame)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging_playhead = False


class AnimationTimelineWidget(QWidget):
    """Complete animation timeline: controls + painted timeline bar.

    Signals:
        frame_changed(int): emitted when playhead moves (scrub or playback)
        keyframe_requested(int): emitted when user wants to add a keyframe at a frame
    """

    frame_changed = Signal(int)
    keyframe_requested = Signal(int)
    render_requested = Signal()  # user clicked Render Sequence

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._anim = AnimationState(fps=24, duration_seconds=5.0)
        self._playing = False
        self._current_frame = 0
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Controls row
        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)

        self._play_btn = QPushButton("\u25b6")  # ▶
        self._play_btn.setFixedWidth(32)
        self._play_btn.setToolTip("Play / Pause (Space)")
        self._play_btn.clicked.connect(self._toggle_play)
        ctrl.addWidget(self._play_btn)

        self._stop_btn = QPushButton("\u25a0")  # ■
        self._stop_btn.setFixedWidth(32)
        self._stop_btn.setToolTip("Stop (rewind to 0)")
        self._stop_btn.clicked.connect(self._stop)
        ctrl.addWidget(self._stop_btn)

        ctrl.addWidget(QLabel("Frame:"))
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, 9999)
        self._frame_spin.setValue(0)
        self._frame_spin.setFixedWidth(70)
        self._frame_spin.valueChanged.connect(self._on_frame_spin)
        ctrl.addWidget(self._frame_spin)

        ctrl.addWidget(QLabel("FPS:"))
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(8, 60)
        self._fps_spin.setValue(24)
        self._fps_spin.setFixedWidth(50)
        self._fps_spin.valueChanged.connect(self._on_fps_changed)
        ctrl.addWidget(self._fps_spin)

        ctrl.addWidget(QLabel("Duration:"))
        self._dur_spin = QSpinBox()
        self._dur_spin.setRange(1, 30)
        self._dur_spin.setValue(5)
        self._dur_spin.setSuffix("s")
        self._dur_spin.setFixedWidth(55)
        self._dur_spin.valueChanged.connect(self._on_duration_changed)
        ctrl.addWidget(self._dur_spin)

        ctrl.addWidget(QLabel("  "))

        self._add_kf_btn = QPushButton("+ Key")
        self._add_kf_btn.setToolTip("Add keyframe at current frame")
        self._add_kf_btn.clicked.connect(lambda: self.keyframe_requested.emit(self._current_frame))
        ctrl.addWidget(self._add_kf_btn)

        self._del_kf_btn = QPushButton("- Key")
        self._del_kf_btn.setToolTip("Delete all keyframes at the current frame")
        self._del_kf_btn.clicked.connect(self._delete_keyframe_at_playhead)
        ctrl.addWidget(self._del_kf_btn)

        self._clear_anim_btn = QPushButton("Clear All")
        self._clear_anim_btn.setToolTip("Remove all keyframes from all tracks")
        self._clear_anim_btn.clicked.connect(self._clear_all_keyframes)
        ctrl.addWidget(self._clear_anim_btn)

        ctrl.addWidget(QLabel("Preset:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItems(["(none)", "Turntable", "Dolly In", "Object Spin", "Reveal", "Orbit + Rise"])
        self._preset_combo.setFixedWidth(110)
        ctrl.addWidget(self._preset_combo)

        self._apply_preset_btn = QPushButton("Apply")
        self._apply_preset_btn.clicked.connect(self._on_apply_preset)
        ctrl.addWidget(self._apply_preset_btn)

        ctrl.addWidget(QLabel("  "))

        self._render_btn = QPushButton("Render Sequence")
        self._render_btn.setStyleSheet(
            "QPushButton { font-weight: bold; color: #fff; background: #0078d4;"
            " border-radius: 3px; padding: 4px 12px; } "
            "QPushButton:hover { background: #005a9e; }")
        self._render_btn.clicked.connect(self.render_requested.emit)
        ctrl.addWidget(self._render_btn)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Timeline bar
        self._bar = _TimelineBar()
        self._bar.set_animation(self._anim)
        self._bar.frame_clicked.connect(self._on_bar_frame)
        layout.addWidget(self._bar)

        # Playback timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)

    # -- Properties --------------------------------------------------------

    @property
    def animation(self) -> AnimationState:
        return self._anim

    @property
    def current_frame(self) -> int:
        return self._current_frame

    @property
    def preset_combo(self) -> QComboBox:
        return self._preset_combo

    # -- Playback ----------------------------------------------------------

    def _toggle_play(self) -> None:
        if self._playing:
            self._pause()
        else:
            self._play()

    def _play(self) -> None:
        self._playing = True
        self._play_btn.setText("\u23f8")  # ⏸
        interval = max(1, int(1000 / self._anim.fps))
        self._timer.start(interval)

    def _pause(self) -> None:
        self._playing = False
        self._play_btn.setText("\u25b6")  # ▶
        self._timer.stop()

    def _stop(self) -> None:
        self._pause()
        self._set_frame(0)

    def _advance_frame(self) -> None:
        nxt = self._current_frame + 1
        if nxt >= self._anim.duration_frames:
            nxt = 0  # loop
        self._set_frame(nxt)

    def _set_frame(self, frame: int) -> None:
        self._current_frame = frame
        self._frame_spin.blockSignals(True)
        self._frame_spin.setValue(frame)
        self._frame_spin.blockSignals(False)
        self._bar.set_current_frame(frame)
        self.frame_changed.emit(frame)

    def _on_bar_frame(self, frame: int) -> None:
        self._set_frame(frame)

    def _on_frame_spin(self, val: int) -> None:
        self._set_frame(val)

    def _on_fps_changed(self, val: int) -> None:
        self._anim.fps = val
        self._frame_spin.setMaximum(self._anim.duration_frames)
        self._bar.update()

    def _on_duration_changed(self, val: int) -> None:
        self._anim.duration_seconds = float(val)
        self._frame_spin.setMaximum(self._anim.duration_frames)
        self._bar.update()

    def _on_apply_preset(self) -> None:
        """Apply the selected preset — emits keyframe_requested so the tab
        can provide the current camera/mesh state."""
        # The actual preset application is handled by the Model3DTab
        # which has access to the viewport camera state.
        self.keyframe_requested.emit(-1)  # -1 signals "apply preset"

    def _delete_keyframe_at_playhead(self) -> None:
        """Remove all keyframes at the current frame across all tracks."""
        f = self._current_frame
        removed = 0
        for track in self._anim.tracks.values():
            before = len(track.keyframes)
            track.remove_keyframe(f)
            removed += before - len(track.keyframes)
        self._bar.update()

    def _clear_all_keyframes(self) -> None:
        """Remove all keyframes from all tracks."""
        self._anim.clear()
        self._bar.update()

    def reset(self) -> None:
        """Clear all animation data and reset to defaults."""
        self._pause()
        self._anim.clear()
        self._set_frame(0)
        self._bar.update()
