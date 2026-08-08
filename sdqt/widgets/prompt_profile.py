"""Reusable prompt profile widget for saving/loading generation presets.

Stores per-project profiles as JSON. Each profile contains:
  prompt, negative_prompt, loras (list), cfg, denoising
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

logger = logging.getLogger(__name__)


class PromptProfileWidget(QWidget):
    """Dropdown + Save/Load/Delete for prompt profiles.

    Parameters
    ----------
    filename : str
        JSON file name stored inside the project directory,
        e.g. ``"flux_profiles.json"`` or ``"zimage_profiles.json"``.
    """

    def __init__(self, filename: str = "profiles.json", parent=None) -> None:
        super().__init__(parent)
        self._filename = filename
        self._project_path: Path | None = None
        self._collect_fn: Any = None  # callable returning dict
        self._apply_fn: Any = None    # callable accepting dict
        self._status_fn: Any = None   # callable accepting str
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Profile:"))
        self._combo = QComboBox()
        self._combo.setEditable(True)
        self._combo.setMinimumWidth(140)
        layout.addWidget(self._combo, 1)

        save_btn = QPushButton("Save")
        save_btn.setFixedWidth(50)
        save_btn.clicked.connect(self._on_save)
        layout.addWidget(save_btn)

        load_btn = QPushButton("Load")
        load_btn.setFixedWidth(50)
        load_btn.clicked.connect(self._on_load)
        layout.addWidget(load_btn)

        del_btn = QPushButton("Del")
        del_btn.setFixedWidth(40)
        del_btn.clicked.connect(self._on_delete)
        layout.addWidget(del_btn)

    # -- Public API ---------------------------------------------------------

    def set_project_path(self, project_path: Path) -> None:
        self._project_path = project_path
        self.refresh()

    def set_callbacks(
        self,
        collect: Any,
        apply: Any,
        status: Any = None,
    ) -> None:
        """Register callbacks for profile save/load.

        collect() -> dict   : gather current UI state into a profile dict
        apply(dict) -> None : restore UI state from a profile dict
        status(str) -> None : optional status message callback
        """
        self._collect_fn = collect
        self._apply_fn = apply
        self._status_fn = status

    def refresh(self) -> None:
        self._combo.clear()
        profiles = self._read()
        self._combo.addItems(sorted(profiles.keys()))

    # -- Internal -----------------------------------------------------------

    def _file_path(self) -> Path | None:
        if self._project_path is None:
            return None
        return self._project_path / self._filename

    def _read(self) -> dict:
        fp = self._file_path()
        if fp and fp.is_file():
            try:
                return json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _write(self, data: dict) -> None:
        fp = self._file_path()
        if fp:
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    def _msg(self, text: str) -> None:
        if self._status_fn:
            self._status_fn(text)

    @Slot()
    def _on_save(self) -> None:
        name = self._combo.currentText().strip()
        if not name:
            self._msg("Enter a profile name first.")
            return
        if self._collect_fn is None:
            return
        profiles = self._read()
        profiles[name] = self._collect_fn()
        self._write(profiles)
        self.refresh()
        self._combo.setCurrentText(name)
        self._msg(f"Saved profile '{name}'.")

    @Slot()
    def _on_load(self) -> None:
        name = self._combo.currentText().strip()
        profiles = self._read()
        if name not in profiles:
            self._msg(f"Profile '{name}' not found.")
            return
        if self._apply_fn is None:
            return
        self._apply_fn(profiles[name])
        self._msg(f"Loaded profile '{name}'.")

    @Slot()
    def _on_delete(self) -> None:
        name = self._combo.currentText().strip()
        profiles = self._read()
        if name in profiles:
            del profiles[name]
            self._write(profiles)
            self.refresh()
            self._msg(f"Deleted profile '{name}'.")
