"""Reusable post-processing controls widget.

Mirrors the pp_* block in the Timeline Export dialog so other dialogs
(Batch Generate, etc.) can embed the same set of controls.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class PostProcessControls(QWidget):
    """Denoise / Sharpen / Upscale / Face Restore / Tile / Lanczos / Grain."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        def _combo_row(label_text: str, items: list[str], tooltip: str) -> QComboBox:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"  {label_text}:"))
            combo = QComboBox()
            combo.addItems(items)
            combo.setFixedWidth(160)
            combo.setToolTip(tooltip)
            row.addWidget(combo)
            row.addStretch()
            layout.addLayout(row)
            return combo

        self.pp_denoise = _combo_row(
            "Denoise", ["Disabled", "SCUNet"],
            "Remove grain/noise while preserving detail")
        self.pp_sharpen = _combo_row(
            "Sharpen", ["Disabled", "RealESRGAN 2x"],
            "Neural detail recovery (upscale 2x then downscale)")
        self.pp_upscale = _combo_row(
            "Upscale", ["Disabled", "RealESRGAN 4x"],
            "4x neural upscale")
        self.pp_face = _combo_row(
            "Face Restore", ["Disabled", "GFPGAN v1.4"],
            "Fix AI-generated face artifacts")

        tile_row = QHBoxLayout()
        tile_row.addWidget(QLabel("  AI Tile Size:"))
        self.pp_tile = QSpinBox()
        self.pp_tile.setRange(0, 1024)
        self.pp_tile.setValue(512)
        self.pp_tile.setSingleStep(128)
        self.pp_tile.setSuffix("px")
        self.pp_tile.setFixedWidth(80)
        tile_row.addWidget(self.pp_tile)
        tile_row.addStretch()
        layout.addLayout(tile_row)

        self.pp_lanczos = _combo_row(
            "Lanczos Upscale", ["Disabled", "Lanczos 1.5x", "Lanczos 2.0x"],
            "Traditional spatial upscale")

        grain_row = QHBoxLayout()
        grain_row.addWidget(QLabel("  Film Grain:"))
        self.pp_grain = QSlider(Qt.Horizontal)
        self.pp_grain.setRange(0, 100)
        self.pp_grain.setValue(0)
        grain_row.addWidget(self.pp_grain, 1)
        self._grain_label = QLabel("0%")
        self._grain_label.setFixedWidth(35)
        self.pp_grain.valueChanged.connect(
            lambda v: self._grain_label.setText(f"{v}%")
        )
        grain_row.addWidget(self._grain_label)
        layout.addLayout(grain_row)

    # ------------------------------------------------------------------

    def get_settings(self) -> dict:
        """Return the same pp_* dict shape as ExportDialog.get_settings()."""
        return {
            "pp_denoise": self.pp_denoise.currentIndex() > 0,
            "pp_sharpen": self.pp_sharpen.currentIndex() > 0,
            "pp_upscale": self.pp_upscale.currentIndex() > 0,
            "pp_face": self.pp_face.currentIndex() > 0,
            "pp_tile": self.pp_tile.value(),
            "pp_lanczos": {
                "Lanczos 1.5x": "1.5",
                "Lanczos 2.0x": "2.0",
            }.get(self.pp_lanczos.currentText(), ""),
            "pp_grain": self.pp_grain.value() / 100.0,
            # Preserve the combo text so restore_settings() can round-trip it
            "pp_denoise_text": self.pp_denoise.currentText(),
            "pp_sharpen_text": self.pp_sharpen.currentText(),
            "pp_upscale_text": self.pp_upscale.currentText(),
            "pp_face_text": self.pp_face.currentText(),
            "pp_lanczos_text": self.pp_lanczos.currentText(),
            "pp_grain_pct": self.pp_grain.value(),
        }

    def restore_settings(self, saved: dict | None) -> None:
        if not saved:
            return

        def _set_combo(combo: QComboBox, text_key: str, bool_key: str,
                       on_text: str) -> None:
            txt = saved.get(text_key)
            if txt:
                idx = combo.findText(txt)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                    return
            # Fallback: bool key → map True to the on_text, False to Disabled
            if bool_key in saved:
                combo.setCurrentIndex(1 if saved[bool_key] else 0)

        _set_combo(self.pp_denoise, "pp_denoise_text", "pp_denoise", "SCUNet")
        _set_combo(self.pp_sharpen, "pp_sharpen_text", "pp_sharpen", "RealESRGAN 2x")
        _set_combo(self.pp_upscale, "pp_upscale_text", "pp_upscale", "RealESRGAN 4x")
        _set_combo(self.pp_face, "pp_face_text", "pp_face", "GFPGAN v1.4")

        if "pp_lanczos_text" in saved and saved["pp_lanczos_text"]:
            idx = self.pp_lanczos.findText(saved["pp_lanczos_text"])
            if idx >= 0:
                self.pp_lanczos.setCurrentIndex(idx)
        elif saved.get("pp_lanczos") == "1.5":
            self.pp_lanczos.setCurrentIndex(1)
        elif saved.get("pp_lanczos") == "2.0":
            self.pp_lanczos.setCurrentIndex(2)

        if "pp_tile" in saved:
            try:
                self.pp_tile.setValue(int(saved["pp_tile"]))
            except (TypeError, ValueError):
                pass

        if "pp_grain_pct" in saved:
            try:
                self.pp_grain.setValue(int(saved["pp_grain_pct"]))
            except (TypeError, ValueError):
                pass
        elif "pp_grain" in saved:
            try:
                self.pp_grain.setValue(int(float(saved["pp_grain"]) * 100))
            except (TypeError, ValueError):
                pass
