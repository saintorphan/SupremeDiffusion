"""Download confirmation dialog — shows missing models with sizes."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from .registry import ModelEntry


class ModelDownloadDialog(QDialog):
    """Modal dialog listing models that need to be downloaded.

    Shows a table of model names and sizes, with OK/Cancel buttons.
    """

    def __init__(
        self,
        feature_label: str,
        missing_models: list[ModelEntry],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Download Required Models")
        self.setMinimumWidth(520)
        self._missing = missing_models
        self._build_ui(feature_label)

    def _build_ui(self, feature_label: str) -> None:
        layout = QVBoxLayout(self)

        # Header
        header = QLabel(
            f"<b>{feature_label}</b> requires the following models to be downloaded:"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # Table
        table = QTableWidget(len(self._missing), 2)
        table.setHorizontalHeaderLabels(["Model", "Size"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)

        for row, model in enumerate(self._missing):
            name_item = QTableWidgetItem(model.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 0, name_item)

            size_item = QTableWidgetItem(model.size_display)
            size_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            size_item.setFlags(size_item.flags() & ~Qt.ItemIsEditable)
            table.setItem(row, 1, size_item)

        layout.addWidget(table)

        # Total size
        total_bytes = sum(m.size_bytes for m in self._missing)
        total_gb = total_bytes / (1024 ** 3)
        if total_gb >= 1.0:
            total_str = f"{total_gb:.1f} GB"
        else:
            total_str = f"{total_bytes / (1024**2):.0f} MB"
        total_label = QLabel(f"<b>Total download size: {total_str}</b>")
        total_label.setAlignment(Qt.AlignRight)
        layout.addWidget(total_label)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
