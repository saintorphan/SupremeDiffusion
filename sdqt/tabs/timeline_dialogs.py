"""Timeline dialogs — SpeedDialog and PostProcessDialog."""

from __future__ import annotations

from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)


def _settings() -> QSettings:
    return QSettings("SupremeDiffusion", "sdqt")


class SpeedDialog(QDialog):
    """Small popup with a slider to adjust clip playback speed."""

    def __init__(self, clip_duration: float = 0.0, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Clip Speed")
        self.setMinimumWidth(360)
        self._clip_duration = clip_duration
        self._updating = False

        layout = QVBoxLayout(self)

        self._label = QLabel("Speed: 1.00x")
        self._label.setStyleSheet("font-size: 13px; font-weight: bold;")
        layout.addWidget(self._label)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(10, 400)
        self._slider.setValue(100)
        self._slider.setTickInterval(50)
        self._slider.setTickPosition(QSlider.TicksBelow)
        self._slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self._slider)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        for val in ("0.25x", "0.5x", "1.0x", "1.5x", "2.0x", "4.0x"):
            btn = QPushButton(val)
            btn.setMinimumHeight(28)
            btn.clicked.connect(lambda checked, v=val: self._set_preset(v))
            preset_row.addWidget(btn)
        layout.addLayout(preset_row)

        # Target duration input
        dur_row = QHBoxLayout()
        dur_row.setSpacing(6)
        dur_row.addWidget(QLabel("Target duration:"))
        self._target_dur = QDoubleSpinBox()
        self._target_dur.setMinimumWidth(95)
        self._target_dur.setRange(0.1, 600.0)
        self._target_dur.setDecimals(2)
        self._target_dur.setSingleStep(0.1)
        self._target_dur.setSuffix("s")
        self._target_dur.setValue(clip_duration if clip_duration > 0 else 1.0)
        self._target_dur.valueChanged.connect(self._on_target_changed)
        dur_row.addWidget(self._target_dur)
        if clip_duration > 0:
            dur_row.addWidget(QLabel(f"(current: {clip_duration:.2f}s)"))
        dur_row.addStretch()
        layout.addLayout(dur_row)

        self._reverse = QCheckBox("Reverse playback")
        self._reverse.setToolTip("Reverse the clip (plays backwards)")
        layout.addWidget(self._reverse)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Ok).setText("Apply")
        layout.addWidget(buttons)

    def _on_slider(self, val: int) -> None:
        speed = val / 100.0
        self._label.setText(f"Speed: {speed:.2f}x")
        if not self._updating and self._clip_duration > 0:
            self._updating = True
            self._target_dur.setValue(self._clip_duration / speed)
            self._updating = False

    def _on_target_changed(self, target: float) -> None:
        if not self._updating and self._clip_duration > 0 and target > 0:
            self._updating = True
            speed = self._clip_duration / target
            self._slider.setValue(int(round(speed * 100)))
            self._updating = False

    def _set_preset(self, text: str) -> None:
        speed = float(text.rstrip("x"))
        self._slider.setValue(int(speed * 100))

    def speed_value(self) -> float:
        return self._slider.value() / 100.0

    def reverse(self) -> bool:
        return self._reverse.isChecked()


class PostProcessDialog(QDialog):
    """Post-processing options: RIFE temporal upsampling, Lanczos spatial upsampling, film grain."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Post Processing")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)

        note = QLabel(
            "Steps apply top-to-bottom. Creates a new clip file\n"
            "(original is preserved in the library)."
        )
        note.setStyleSheet("color: #999; font-size: 13px; font-style: italic; padding-bottom: 2px;")
        layout.addWidget(note)

        # ── AI Enhancement (neural net, frame-by-frame) ──────────────
        from PySide6.QtWidgets import QSpinBox

        ai_label = QLabel("<b>AI Enhancement</b>")
        ai_label.setStyleSheet("color: #aaa; padding-top: 2px;")
        layout.addWidget(ai_label)

        def _ai_row(label, items, tooltip):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"  {label}:"))
            combo = QComboBox()
            combo.addItems(items)
            combo.setMinimumWidth(180)
            combo.setToolTip(tooltip)
            row.addWidget(combo)
            row.addStretch()
            layout.addLayout(row)
            return combo

        self._ai_denoise_combo = _ai_row(
            "1. Denoise", ["Disabled", "SCUNet"],
            "Remove grain/noise while preserving detail")
        self._ai_sharpen_combo = _ai_row(
            "2. Sharpen", ["Disabled", "RealESRGAN 2x"],
            "Neural detail recovery (upscale 2x then downscale)")
        self._ai_enhance_combo = _ai_row(
            "3. Upscale", ["Disabled", "RealESRGAN 4x"],
            "4x neural upscale — adds real detail to low-res clips")
        self._ai_face_combo = _ai_row(
            "4. Face Restore", ["Disabled", "GFPGAN v1.4"],
            "Fix AI-generated face artifacts (runs at final resolution)")

        tile_row = QHBoxLayout()
        tile_row.addWidget(QLabel("  Tile Size:"))
        self._ai_tile = QSpinBox()
        self._ai_tile.setRange(0, 1024)
        self._ai_tile.setValue(512)
        self._ai_tile.setSingleStep(128)
        self._ai_tile.setSuffix("px")
        self._ai_tile.setMinimumWidth(90)
        self._ai_tile.setToolTip("0 = no tiling (fastest, most VRAM). Lower = less VRAM")
        tile_row.addWidget(self._ai_tile)
        tile_row.addStretch()
        layout.addLayout(tile_row)

        # ── Traditional post-processing ──────────────────────────────
        trad_label = QLabel("<b>Traditional</b>")
        trad_label.setStyleSheet("color: #aaa; padding-top: 6px;")
        layout.addWidget(trad_label)

        temporal_row = QHBoxLayout()
        temporal_row.addWidget(QLabel("  5. Frame Interp:"))
        self._temporal_combo = QComboBox()
        self._temporal_combo.addItems(["Disabled", "RIFE x2", "RIFE x4"])
        self._temporal_combo.setMinimumWidth(180)
        temporal_row.addWidget(self._temporal_combo)
        temporal_row.addStretch()
        layout.addLayout(temporal_row)

        spatial_row = QHBoxLayout()
        spatial_row.addWidget(QLabel("  6. Lanczos Upscale:"))
        self._spatial_combo = QComboBox()
        self._spatial_combo.addItems(["Disabled", "Lanczos 1.5x", "Lanczos 2.0x"])
        self._spatial_combo.setMinimumWidth(180)
        spatial_row.addWidget(self._spatial_combo)
        spatial_row.addStretch()
        layout.addLayout(spatial_row)

        grain_row = QHBoxLayout()
        grain_row.addWidget(QLabel("  7. Film Grain:"))
        self._grain_slider = QSlider(Qt.Horizontal)
        self._grain_slider.setRange(0, 100)
        self._grain_slider.setValue(0)
        self._grain_slider.valueChanged.connect(self._on_grain)
        grain_row.addWidget(self._grain_slider, 1)
        self._grain_label = QLabel("0%")
        self._grain_label.setMinimumWidth(40)
        grain_row.addWidget(self._grain_label)
        layout.addLayout(grain_row)

        grain_sat_row = QHBoxLayout()
        grain_sat_row.addWidget(QLabel("Grain Color:"))
        self._grain_sat_slider = QSlider(Qt.Horizontal)
        self._grain_sat_slider.setRange(0, 100)
        self._grain_sat_slider.setValue(50)
        grain_sat_row.addWidget(self._grain_sat_slider, 1)
        self._grain_sat_label = QLabel("50%")
        self._grain_sat_label.setMinimumWidth(40)
        self._grain_sat_slider.valueChanged.connect(
            lambda v: self._grain_sat_label.setText(f"{v}%")
        )
        grain_sat_row.addWidget(self._grain_sat_label)
        layout.addLayout(grain_sat_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Ok).setText("Apply")
        layout.addWidget(buttons)

        self._load_settings()

    _SETTINGS_GROUP = "PostProcessDialog"

    def _load_settings(self) -> None:
        s = _settings()
        s.beginGroup(self._SETTINGS_GROUP)
        for combo, key in (
            (self._ai_denoise_combo, "ai_denoise"),
            (self._ai_sharpen_combo, "ai_sharpen"),
            (self._ai_enhance_combo, "ai_enhance"),
            (self._ai_face_combo, "ai_face"),
            (self._temporal_combo, "temporal"),
            (self._spatial_combo, "spatial"),
        ):
            text = s.value(key, "", type=str)
            if text:
                idx = combo.findText(text)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
        self._ai_tile.setValue(s.value("tile", self._ai_tile.value(), type=int))
        self._grain_slider.setValue(s.value("grain", self._grain_slider.value(), type=int))
        self._grain_sat_slider.setValue(s.value("grain_sat", self._grain_sat_slider.value(), type=int))
        s.endGroup()

    def _save_settings(self) -> None:
        s = _settings()
        s.beginGroup(self._SETTINGS_GROUP)
        s.setValue("ai_denoise", self._ai_denoise_combo.currentText())
        s.setValue("ai_sharpen", self._ai_sharpen_combo.currentText())
        s.setValue("ai_enhance", self._ai_enhance_combo.currentText())
        s.setValue("ai_face", self._ai_face_combo.currentText())
        s.setValue("temporal", self._temporal_combo.currentText())
        s.setValue("spatial", self._spatial_combo.currentText())
        s.setValue("tile", self._ai_tile.value())
        s.setValue("grain", self._grain_slider.value())
        s.setValue("grain_sat", self._grain_sat_slider.value())
        s.endGroup()

    def _on_accept(self) -> None:
        self._save_settings()
        self.accept()

    def _on_grain(self, val: int) -> None:
        self._grain_label.setText(f"{val}%")

    def temporal(self) -> str:
        text = self._temporal_combo.currentText()
        return {"RIFE x2": "rife2", "RIFE x4": "rife4"}.get(text, "")

    def spatial(self) -> str:
        text = self._spatial_combo.currentText()
        return {"Lanczos 1.5x": "lanczos1.5", "Lanczos 2.0x": "lanczos2"}.get(text, "")

    def grain_intensity(self) -> float:
        return self._grain_slider.value() / 100.0

    def grain_saturation(self) -> float:
        return self._grain_sat_slider.value() / 100.0

    def ai_enhance(self) -> bool:
        return self._ai_enhance_combo.currentIndex() > 0

    def ai_denoise(self) -> bool:
        return self._ai_denoise_combo.currentIndex() > 0

    def ai_sharpen(self) -> bool:
        return self._ai_sharpen_combo.currentIndex() > 0

    def ai_face_restore(self) -> bool:
        return self._ai_face_combo.currentIndex() > 0

    def ai_tile_size(self) -> int:
        return self._ai_tile.value()


class ImagePostProcessDialog(QDialog):
    """AI enhancement options for still images (subset of PostProcessDialog)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Post-Process Image")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)

        from PySide6.QtWidgets import QSpinBox

        ai_label = QLabel("<b>AI Enhancement</b>")
        ai_label.setStyleSheet("color: #aaa; padding-top: 2px;")
        layout.addWidget(ai_label)

        def _ai_row(label, items, tooltip):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"  {label}:"))
            combo = QComboBox()
            combo.addItems(items)
            combo.setMinimumWidth(180)
            combo.setToolTip(tooltip)
            row.addWidget(combo)
            row.addStretch()
            layout.addLayout(row)
            return combo

        self._ai_denoise_combo = _ai_row(
            "1. Denoise", ["Disabled", "SCUNet"],
            "Remove grain/noise while preserving detail")
        self._ai_sharpen_combo = _ai_row(
            "2. Sharpen", ["Disabled", "RealESRGAN 2x"],
            "Neural detail recovery (upscale 2x then downscale)")
        self._ai_enhance_combo = _ai_row(
            "3. Upscale", ["Disabled", "RealESRGAN 4x"],
            "4x neural upscale — adds real detail to low-res images")
        self._ai_face_combo = _ai_row(
            "4. Face Restore", ["Disabled", "GFPGAN v1.4"],
            "Fix AI-generated face artifacts")

        tile_row = QHBoxLayout()
        tile_row.addWidget(QLabel("  Tile Size:"))
        self._ai_tile = QSpinBox()
        self._ai_tile.setRange(0, 1024)
        self._ai_tile.setValue(512)
        self._ai_tile.setSingleStep(128)
        self._ai_tile.setSuffix("px")
        self._ai_tile.setMinimumWidth(90)
        self._ai_tile.setToolTip("0 = no tiling (fastest, most VRAM). Lower = less VRAM")
        tile_row.addWidget(self._ai_tile)
        tile_row.addStretch()
        layout.addLayout(tile_row)

        # Traditional spatial upscale
        spatial_row = QHBoxLayout()
        spatial_row.addWidget(QLabel("  5. Lanczos Upscale:"))
        self._spatial_combo = QComboBox()
        self._spatial_combo.addItems(["Disabled", "Lanczos 1.5x", "Lanczos 2.0x"])
        self._spatial_combo.setMinimumWidth(180)
        spatial_row.addWidget(self._spatial_combo)
        spatial_row.addStretch()
        layout.addLayout(spatial_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Ok).setText("Apply")
        layout.addWidget(buttons)

        self._load_settings()

    _SETTINGS_GROUP = "ImagePostProcessDialog"

    def _load_settings(self) -> None:
        s = _settings()
        s.beginGroup(self._SETTINGS_GROUP)
        for combo, key in (
            (self._ai_denoise_combo, "ai_denoise"),
            (self._ai_sharpen_combo, "ai_sharpen"),
            (self._ai_enhance_combo, "ai_enhance"),
            (self._ai_face_combo, "ai_face"),
            (self._spatial_combo, "spatial"),
        ):
            text = s.value(key, "", type=str)
            if text:
                idx = combo.findText(text)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
        self._ai_tile.setValue(s.value("tile", self._ai_tile.value(), type=int))
        s.endGroup()

    def _save_settings(self) -> None:
        s = _settings()
        s.beginGroup(self._SETTINGS_GROUP)
        s.setValue("ai_denoise", self._ai_denoise_combo.currentText())
        s.setValue("ai_sharpen", self._ai_sharpen_combo.currentText())
        s.setValue("ai_enhance", self._ai_enhance_combo.currentText())
        s.setValue("ai_face", self._ai_face_combo.currentText())
        s.setValue("spatial", self._spatial_combo.currentText())
        s.setValue("tile", self._ai_tile.value())
        s.endGroup()

    def _on_accept(self) -> None:
        self._save_settings()
        self.accept()

    def ai_enhance(self) -> bool:
        return self._ai_enhance_combo.currentIndex() > 0

    def ai_denoise(self) -> bool:
        return self._ai_denoise_combo.currentIndex() > 0

    def ai_sharpen(self) -> bool:
        return self._ai_sharpen_combo.currentIndex() > 0

    def ai_face_restore(self) -> bool:
        return self._ai_face_combo.currentIndex() > 0

    def ai_tile_size(self) -> int:
        return self._ai_tile.value()

    def spatial(self) -> str:
        text = self._spatial_combo.currentText()
        return {"Lanczos 1.5x": "lanczos1.5", "Lanczos 2.0x": "lanczos2"}.get(text, "")

    def has_any_enabled(self) -> bool:
        return (self.ai_denoise() or self.ai_sharpen() or
                self.ai_enhance() or self.ai_face_restore() or
                bool(self.spatial()))


class ResizeDialog(QDialog):
    """Resize a clip to a chosen resolution — presets or manual entry."""

    PRESETS = [
        ("HD 1080p", 1920, 1080),
        ("HD 720p", 1280, 720),
        ("LTX wide", 1024, 576),
        ("Wan 480p", 832, 480),
        ("V. 720p", 720, 1280),
        ("V. 480p", 480, 832),
        ("Square 1024", 1024, 1024),
        ("Square 768", 768, 768),
        ("4K", 3840, 2160),
    ]

    def __init__(self, current_w: int = 0, current_h: int = 0, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Resize Clip")
        self.setMinimumWidth(420)
        self._current_w = current_w
        self._current_h = current_h
        self._src_aspect = (current_w / current_h) if (current_w > 0 and current_h > 0) else 1.0
        self._updating = False

        layout = QVBoxLayout(self)

        cur_text = (
            f"Current: {current_w}×{current_h}" if current_w > 0 else "Current: unknown"
        )
        info = QLabel(cur_text)
        info.setStyleSheet("color: #999; font-size: 13px; font-style: italic;")
        layout.addWidget(info)

        preset_label = QLabel("<b>Presets</b>")
        preset_label.setStyleSheet("color: #aaa; padding-top: 4px;")
        layout.addWidget(preset_label)

        grid = QGridLayout()
        grid.setSpacing(4)
        for i, (name, w, h) in enumerate(self.PRESETS):
            btn = QPushButton(f"{name}\n{w}×{h}")
            btn.setMinimumHeight(44)
            btn.clicked.connect(lambda checked=False, ww=w, hh=h: self._set_dims(ww, hh))
            grid.addWidget(btn, i // 3, i % 3)
        layout.addLayout(grid)

        manual_label = QLabel("<b>Manual</b>")
        manual_label.setStyleSheet("color: #aaa; padding-top: 6px;")
        layout.addWidget(manual_label)

        wh_row = QHBoxLayout()
        wh_row.setSpacing(6)
        wh_row.addWidget(QLabel("Width:"))
        self._w_spin = QSpinBox()
        self._w_spin.setRange(16, 8192)
        self._w_spin.setSingleStep(8)
        self._w_spin.setMinimumWidth(90)
        self._w_spin.setValue(current_w if current_w > 0 else 1280)
        self._w_spin.valueChanged.connect(self._on_w_changed)
        wh_row.addWidget(self._w_spin)
        wh_row.addWidget(QLabel("Height:"))
        self._h_spin = QSpinBox()
        self._h_spin.setRange(16, 8192)
        self._h_spin.setSingleStep(8)
        self._h_spin.setMinimumWidth(90)
        self._h_spin.setValue(current_h if current_h > 0 else 720)
        self._h_spin.valueChanged.connect(self._on_h_changed)
        wh_row.addWidget(self._h_spin)
        wh_row.addStretch()
        layout.addLayout(wh_row)

        self._lock_aspect = QCheckBox("Lock aspect ratio (from source)")
        self._lock_aspect.setToolTip(
            "When checked, editing one dimension auto-fills the other "
            "from the source clip's aspect ratio."
        )
        layout.addWidget(self._lock_aspect)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        mode_row.addWidget(QLabel("Scale mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Stretch", "Fit (letterbox)", "Fill (crop)"])
        self._mode_combo.setMinimumWidth(160)
        self._mode_combo.setToolTip(
            "Stretch: distort to exact dimensions (no bars, source aspect lost).\n"
            "Fit: preserve aspect, pad with black bars.\n"
            "Fill: preserve aspect, crop edges to fill."
        )
        mode_row.addWidget(self._mode_combo)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        self._load_settings()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Ok).setText("Resize")
        layout.addWidget(buttons)

    def _set_dims(self, w: int, h: int) -> None:
        self._updating = True
        self._w_spin.setValue(w)
        self._h_spin.setValue(h)
        self._updating = False

    def _on_w_changed(self, w: int) -> None:
        if self._updating or not self._lock_aspect.isChecked() or self._src_aspect <= 0:
            return
        self._updating = True
        self._h_spin.setValue(max(16, int(round(w / self._src_aspect))))
        self._updating = False

    def _on_h_changed(self, h: int) -> None:
        if self._updating or not self._lock_aspect.isChecked() or self._src_aspect <= 0:
            return
        self._updating = True
        self._w_spin.setValue(max(16, int(round(h * self._src_aspect))))
        self._updating = False

    def _load_settings(self) -> None:
        s = _settings()
        s.beginGroup("resize")
        mode = s.value("mode", "Stretch")
        idx = self._mode_combo.findText(mode)
        if idx >= 0:
            self._mode_combo.setCurrentIndex(idx)
        s.endGroup()

    def _save_settings(self) -> None:
        s = _settings()
        s.beginGroup("resize")
        s.setValue("mode", self._mode_combo.currentText())
        s.endGroup()

    def _on_accept(self) -> None:
        self._save_settings()
        self.accept()

    def target_width(self) -> int:
        return self._w_spin.value()

    def target_height(self) -> int:
        return self._h_spin.value()

    def scale_mode(self) -> str:
        return {
            "Stretch": "stretch",
            "Fit (letterbox)": "fit",
            "Fill (crop)": "fill",
        }.get(self._mode_combo.currentText(), "stretch")
