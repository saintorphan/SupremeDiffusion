"""Reusable prompt enhancement widget — row of buttons for LLM-powered prompt improvement."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

# Image style options (SD-family)
_IMAGE_STYLES = [
    ("SD 1.5", "sd15"),
    ("SDXL", "sdxl"),
    ("SD3 / SD3.5", "sd3"),
    ("FLUX", "flux"),
    ("Pony: Real", "pony_real"),
    ("Pony: Anime", "pony_anime"),
    ("Illustrious", "illustrious"),
    ("NoobAI", "noobai"),
]


class PromptEnhanceWidget(QWidget):
    """Row widget: [Style ▼] [Enhance POS] [Enhance NEG].

    For video tabs, style is auto-determined from model dropdown — no style combo.
    For image tabs, a style dropdown is shown.
    Clicking enhance when transformers is not installed triggers auto-install.
    """

    enhance_requested = Signal(str, str, str)  # (which, current_text, style)

    def __init__(self, mode: str = "image", parent=None) -> None:
        super().__init__(parent)
        self._mode = mode  # "image" or "video"
        self._fixed_style = "sdxl"  # auto-set from model family
        self._build_ui()

    def _build_ui(self) -> None:
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        self._label = QLabel("Qwen 4B:")
        self._label.setStyleSheet("font-size: 12px; font-weight: bold; color: #999;")
        row.addWidget(self._label)

        # Style combo kept for backward compat but hidden — family auto-detected
        self._style_combo = QComboBox()
        for display, key in _IMAGE_STYLES:
            self._style_combo.addItem(display, userData=key)
        self._style_combo.setFixedWidth(120)
        self._style_combo.setVisible(False)
        row.addWidget(self._style_combo)

        self._pos_btn = QPushButton("Enhance +")
        self._pos_btn.setFixedHeight(24)
        self._pos_btn.setToolTip("Enhance the positive prompt using local LLM")
        self._pos_btn.clicked.connect(lambda: self._emit("pos"))
        row.addWidget(self._pos_btn)

        self._neg_btn = QPushButton("Enhance \u2013")
        self._neg_btn.setFixedHeight(24)
        self._neg_btn.setToolTip("Generate an appropriate negative prompt using local LLM")
        self._neg_btn.clicked.connect(lambda: self._emit("neg"))
        row.addWidget(self._neg_btn)

        row.addStretch()

    def _emit(self, which: str) -> None:
        # Check transformers is installed — offer to install if not
        from sdqt.deps import check_and_install
        if not check_and_install("transformers", self):
            return
        style = self._current_style()
        self.enhance_requested.emit(which, "", style)

    def _current_style(self) -> str:
        if self._style_combo is not None and self._style_combo.isVisible():
            return self._style_combo.currentData() or "sdxl"
        return self._fixed_style

    def set_video_style(self, style: str) -> None:
        """Set the video style (called when model dropdown changes)."""
        self._fixed_style = style

    def set_family(self, strategy) -> None:
        """Auto-set prompt style from model family strategy."""
        family = getattr(strategy, "family", "sdxl")
        # Map family to prompt enhance style
        style_map = {
            "sd15": "sd15",
            "sdxl": "sdxl",
            "flux": "flux",
            "zimage": "sdxl",  # Z-Image uses SDXL-style prompts
        }
        self._fixed_style = style_map.get(family, "sdxl")

    def set_enabled_buttons(self, enabled: bool) -> None:
        """Enable/disable the enhance buttons during processing."""
        self._pos_btn.setEnabled(enabled)
        self._neg_btn.setEnabled(enabled)

    def set_fixed_style(self, style: str) -> None:
        """Fix the style dropdown to a specific value (for FLUX tabs)."""
        if self._style_combo is not None:
            idx = self._style_combo.findData(style)
            if idx >= 0:
                self._style_combo.setCurrentIndex(idx)
                self._style_combo.setEnabled(False)

    @staticmethod
    def is_available() -> bool:
        """Return True if transformers is installed."""
        import importlib.util
        return importlib.util.find_spec("transformers") is not None
