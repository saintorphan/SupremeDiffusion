"""Grouped checkpoint selector — displays checkpoints organized by subfolder/family.

Uses QComboBox with QStandardItemModel to show bold, non-selectable group headers
(e.g., "PonyXL", "IllustriousXL", "SDXL") with selectable checkpoint items underneath.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from supremediffusion.models.sd_models import CheckpointInfo

logger = logging.getLogger(__name__)

# Display order for checkpoint groups
_GROUP_ORDER = [
    "PonyXL", "IllustriousXL", "NoobAI", "SDXL",
    "SD 1.5", "Chroma", "Flux", "Z-Image", "Other",
]

# Map model_type to fallback display group when no subfolder prefix
_TYPE_TO_GROUP = {
    "sd15": "SD 1.5",
    "sdxl": "SDXL",
    "flux": "Flux",
    "zimage": "Z-Image",
}


def _derive_group(ci: CheckpointInfo) -> str:
    """Derive a display group from a CheckpointInfo."""
    # Use explicit group if set
    if ci.group:
        return ci.group
    # Try subfolder prefix from name (e.g., "PonyXL/model_name")
    if "/" in ci.name:
        prefix = ci.name.split("/", 1)[0]
        if prefix:
            return prefix
    # Fallback to model_type mapping
    return _TYPE_TO_GROUP.get(ci.model_type, "Other")


class GroupedCheckpointCombo(QWidget):
    """Checkpoint selector with grouped display.

    Signals:
        checkpoint_changed(object): Emitted when selection changes.
            The payload is the selected CheckpointInfo or None.
    """

    checkpoint_changed = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._checkpoints: list[CheckpointInfo] = []
        self._index_to_info: dict[int, CheckpointInfo] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        lbl = QLabel("Checkpoint:")
        lbl.setFixedWidth(90)
        lbl.setStyleSheet("font-size: 13px; color: #e0e0e0; font-weight: bold;")
        layout.addWidget(lbl)
        self._combo = QComboBox()
        self._combo.setMinimumWidth(250)
        self._combo.setStyleSheet("font-size: 13px;")
        self._combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._combo.currentIndexChanged.connect(self._on_index_changed)
        layout.addWidget(self._combo, 1)

    def populate(self, checkpoints: list[CheckpointInfo]) -> None:
        """Rebuild the dropdown from a list of CheckpointInfo objects."""
        self._combo.blockSignals(True)
        self._checkpoints = list(checkpoints)
        self._index_to_info.clear()

        model = QStandardItemModel()
        bold_font = QFont()
        bold_font.setBold(True)

        # Group checkpoints
        grouped: dict[str, list[CheckpointInfo]] = defaultdict(list)
        for ci in self._checkpoints:
            grouped[_derive_group(ci)].append(ci)

        # Sort groups in display order
        ordered = sorted(
            grouped.keys(),
            key=lambda g: _GROUP_ORDER.index(g) if g in _GROUP_ORDER else 99,
        )

        combo_idx = 0
        for group_name in ordered:
            items = grouped[group_name]
            if not items:
                continue

            # Group header (non-selectable)
            header = QStandardItem(f"── {group_name} ({len(items)}) ──")
            header.setFont(bold_font)
            header.setEnabled(False)
            header.setSelectable(False)
            model.appendRow(header)
            combo_idx += 1

            # Checkpoint items
            for ci in sorted(items, key=lambda c: c.name.lower()):
                # Show just the filename part (after subfolder prefix)
                display = ci.name.split("/", 1)[-1] if "/" in ci.name else ci.name
                item = QStandardItem(f"  {display}")
                item.setData(ci, Qt.ItemDataRole.UserRole)
                model.appendRow(item)
                self._index_to_info[combo_idx] = ci
                combo_idx += 1

        self._combo.setModel(model)

        # Select first valid item
        for idx in range(self._combo.count()):
            if idx in self._index_to_info:
                self._combo.setCurrentIndex(idx)
                break

        self._combo.blockSignals(False)
        # Emit initial selection
        self._on_index_changed(self._combo.currentIndex())

    def selected_info(self) -> CheckpointInfo | None:
        """Return the currently selected CheckpointInfo, or None."""
        return self._index_to_info.get(self._combo.currentIndex())

    def selected_family(self) -> str:
        """Return the model_type of the current selection (e.g., 'sdxl')."""
        info = self.selected_info()
        return info.model_type if info else "sdxl"

    def set_current_by_name(self, name: str) -> None:
        """Select a checkpoint by its name field."""
        for idx, ci in self._index_to_info.items():
            if ci.name == name:
                self._combo.setCurrentIndex(idx)
                return
        # Try partial match (filename without subfolder)
        for idx, ci in self._index_to_info.items():
            if ci.name.endswith(name) or name in ci.name:
                self._combo.setCurrentIndex(idx)
                return

    def _on_index_changed(self, index: int) -> None:
        info = self._index_to_info.get(index)
        if info is not None:
            self.checkpoint_changed.emit(info)
