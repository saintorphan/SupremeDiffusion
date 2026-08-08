"""ADetailer settings panel — drops into any image-gen tab.

Exposes every ADetailerProcessor knob so the user can dial in the post-step
on a per-project basis. Settings persist via project config keys named
``adetailer_*``.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from sdqt.widgets.collapsible_section import CollapsibleSection


class ADetailerPanel(QWidget):
    """Collapsible panel exposing all ADetailer post-step knobs."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._wire_signals()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        self._section = CollapsibleSection(
            "ADetailer (face refinement post-step)", collapsed=True
        )
        body_layout = QVBoxLayout()
        body_layout.setContentsMargins(6, 4, 6, 4)
        body_layout.setSpacing(4)

        # Row 1: enable + detector + threshold
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        self._enable = QCheckBox("Enable")
        self._enable.setToolTip(
            "After image generation, detect faces in the result and re-inpaint each "
            "face region at higher detail using the same SD pipeline."
        )
        row1.addWidget(self._enable)

        row1.addWidget(QLabel("Detector:"))
        self._detector = QComboBox()
        self._detector.addItems(["insightface", "yolov8"])
        self._detector.setFixedWidth(100)
        self._detector.setToolTip(
            "insightface = buffalo_l (same as face_swap). yolov8 = face_yolov8s.pt — "
            "better on tough angles. Falls back to insightface if YOLO weights missing."
        )
        row1.addWidget(self._detector)

        row1.addWidget(QLabel("Threshold:"))
        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(0.05, 0.95)
        self._threshold.setSingleStep(0.05)
        self._threshold.setDecimals(2)
        self._threshold.setValue(0.30)
        self._threshold.setFixedWidth(80)
        self._threshold.setToolTip("Detection confidence threshold. Lower = catches more faces.")
        row1.addWidget(self._threshold)
        row1.addStretch()
        body_layout.addLayout(row1)

        # Row 2: mask shape / feather
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        row2.addWidget(QLabel("Dilation %:"))
        self._dilation = QDoubleSpinBox()
        self._dilation.setRange(0.0, 60.0)
        self._dilation.setSingleStep(5.0)
        self._dilation.setDecimals(1)
        self._dilation.setValue(20.0)
        self._dilation.setFixedWidth(80)
        self._dilation.setToolTip(
            "Expand the detected bbox by this percentage before masking. "
            "Higher = includes more of forehead/chin/jaw. 20 is a good default."
        )
        row2.addWidget(self._dilation)

        row2.addWidget(QLabel("Feather px:"))
        self._feather = QSpinBox()
        self._feather.setRange(0, 64)
        self._feather.setSingleStep(2)
        self._feather.setValue(12)
        self._feather.setFixedWidth(75)
        self._feather.setToolTip(
            "Gaussian blur radius on the mask edge. Reduces visible seam where the "
            "inpainted face joins the original."
        )
        row2.addWidget(self._feather)
        row2.addStretch()
        body_layout.addLayout(row2)

        # Row 3: denoise / steps / inpaint size
        row3 = QHBoxLayout()
        row3.setSpacing(6)
        row3.addWidget(QLabel("Denoise:"))
        self._denoise = QDoubleSpinBox()
        self._denoise.setRange(0.0, 1.0)
        self._denoise.setSingleStep(0.05)
        self._denoise.setDecimals(2)
        self._denoise.setValue(0.40)
        self._denoise.setFixedWidth(80)
        self._denoise.setToolTip(
            "Inpaint denoising strength. 0.30-0.45 = subtle refinement; "
            "0.50+ = stronger redraw (more identity drift)."
        )
        row3.addWidget(self._denoise)

        row3.addWidget(QLabel("+Steps:"))
        self._steps_add = QSpinBox()
        self._steps_add.setRange(0, 40)
        self._steps_add.setValue(6)
        self._steps_add.setFixedWidth(75)
        self._steps_add.setToolTip("Additional sampling steps for the inpaint pass on top of the base step count.")
        row3.addWidget(self._steps_add)

        row3.addWidget(QLabel("Inpaint size:"))
        self._inpaint_size = QComboBox()
        self._inpaint_size.addItems(["512", "768", "1024", "1280", "1536"])
        self._inpaint_size.setCurrentText("1024")
        self._inpaint_size.setFixedWidth(100)
        self._inpaint_size.setToolTip(
            "Resolution the face crop is upscaled to before inpainting. Higher = sharper detail "
            "but slower. 1024 is the standard ADetailer default."
        )
        row3.addWidget(self._inpaint_size)
        row3.addStretch()
        body_layout.addLayout(row3)

        # Row 4: prompt override
        row4 = QHBoxLayout()
        row4.setSpacing(6)
        row4.addWidget(QLabel("Prompt override:"))
        self._prompt_override = QLineEdit()
        self._prompt_override.setPlaceholderText(
            "Empty = reuse main prompt. Common: 'detailed face, sharp eyes, skin texture'"
        )
        row4.addWidget(self._prompt_override, 1)
        body_layout.addLayout(row4)

        row5 = QHBoxLayout()
        row5.setSpacing(6)
        row5.addWidget(QLabel("Negative override:"))
        self._neg_override = QLineEdit()
        self._neg_override.setPlaceholderText("Empty = reuse main negative prompt.")
        row5.addWidget(self._neg_override, 1)
        body_layout.addLayout(row5)

        self._section.add_layout(body_layout)
        root.addWidget(self._section)

    # ------------------------------------------------------------------
    def _wire_signals(self) -> None:
        for w in (
            self._enable, self._detector, self._threshold, self._dilation,
            self._feather, self._denoise, self._steps_add, self._inpaint_size,
            self._prompt_override, self._neg_override,
        ):
            for sig_name in ("toggled", "currentTextChanged", "valueChanged", "textChanged"):
                sig = getattr(w, sig_name, None)
                if sig is not None and hasattr(sig, "connect"):
                    sig.connect(self.changed)
                    break

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        """Snapshot of all settings, ready to pass to ADetailerProcessor.process()."""
        return {
            "enabled": self._enable.isChecked(),
            "detector": self._detector.currentText(),
            "threshold": self._threshold.value(),
            "dilation_pct": self._dilation.value(),
            "feather_px": self._feather.value(),
            "denoise": self._denoise.value(),
            "steps_add": self._steps_add.value(),
            "inpaint_size": int(self._inpaint_size.currentText()),
            "prompt_override": self._prompt_override.text(),
            "neg_prompt_override": self._neg_override.text(),
        }

    def load_from_cfg(self, cfg: Any) -> None:
        """Restore settings from project config attributes (uses safe getattr)."""
        self.blockSignals(True)
        try:
            self._enable.setChecked(bool(getattr(cfg, "adetailer_enabled", False)))
            det = getattr(cfg, "adetailer_detector", "insightface") or "insightface"
            idx = self._detector.findText(det)
            if idx >= 0:
                self._detector.setCurrentIndex(idx)
            self._threshold.setValue(float(getattr(cfg, "adetailer_threshold", 0.30)))
            self._dilation.setValue(float(getattr(cfg, "adetailer_dilation_pct", 20.0)))
            self._feather.setValue(int(getattr(cfg, "adetailer_feather_px", 12)))
            self._denoise.setValue(float(getattr(cfg, "adetailer_denoise", 0.40)))
            self._steps_add.setValue(int(getattr(cfg, "adetailer_steps_add", 6)))
            size = str(getattr(cfg, "adetailer_inpaint_size", 1024))
            idx = self._inpaint_size.findText(size)
            if idx >= 0:
                self._inpaint_size.setCurrentIndex(idx)
            self._prompt_override.setText(getattr(cfg, "adetailer_prompt_override", "") or "")
            self._neg_override.setText(getattr(cfg, "adetailer_neg_prompt_override", "") or "")
        finally:
            self.blockSignals(False)

    def save_to_cfg(self, cfg: Any) -> None:
        """Write current settings to project config attributes (caller saves the cfg)."""
        cfg.adetailer_enabled = self._enable.isChecked()
        cfg.adetailer_detector = self._detector.currentText()
        cfg.adetailer_threshold = self._threshold.value()
        cfg.adetailer_dilation_pct = self._dilation.value()
        cfg.adetailer_feather_px = self._feather.value()
        cfg.adetailer_denoise = self._denoise.value()
        cfg.adetailer_steps_add = self._steps_add.value()
        cfg.adetailer_inpaint_size = int(self._inpaint_size.currentText())
        cfg.adetailer_prompt_override = self._prompt_override.text()
        cfg.adetailer_neg_prompt_override = self._neg_override.text()
