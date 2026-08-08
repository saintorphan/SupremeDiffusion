"""Compact 'Send to: [dropdown] [Send]' widget replacing groups of send-to buttons."""

from __future__ import annotations

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QWidget


class SendToWidget(QWidget):
    """Compact send-to selector: [Label] [QComboBox] [Send button].

    Signals:
        send_requested(str): emitted with the selected target key when Send is clicked
    """

    send_requested = Signal(str)

    def __init__(
        self,
        label: str = "Send to:",
        targets: list[tuple[str, str]] | None = None,
        parent=None,
    ) -> None:
        """Create a send-to widget.

        Args:
            label: Display label (e.g. "Send clip to:", "Send frame to:")
            targets: List of (key, display_name) tuples.
                     key is used in signal, display_name is shown in dropdown.
            parent: Parent widget
        """
        super().__init__(parent)
        self._targets: dict[str, str] = {}  # display_name -> key

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._label = QLabel(label)
        layout.addWidget(self._label)

        self._combo = QComboBox()
        self._combo.setMinimumWidth(100)
        self._combo.setMaxVisibleItems(3)
        layout.addWidget(self._combo, 1)

        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedWidth(50)
        self._send_btn.clicked.connect(self._on_send)
        layout.addWidget(self._send_btn)

        if targets:
            self.set_targets(targets)

    def set_targets(self, targets: list[tuple[str, str]]) -> None:
        """Set the available targets. Each is (key, display_name)."""
        self._combo.clear()
        self._targets.clear()
        for key, display_name in targets:
            self._targets[display_name] = key
            self._combo.addItem(display_name)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the entire widget."""
        self._combo.setEnabled(enabled)
        self._send_btn.setEnabled(enabled)

    @Slot()
    def _on_send(self) -> None:
        display_name = self._combo.currentText()
        key = self._targets.get(display_name)
        if key:
            self.send_requested.emit(key)
