"""Per-LoRA metadata storage and config dialog.

Stores activation_text, trigger_phrase, base_strength, description, keywords,
and family per LoRA filename in a global JSON file
(~/.supremediffusion/lora_metadata.json).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

logger = logging.getLogger(__name__)

_META_DIR = Path.home() / ".supremediffusion"
_META_FILE = _META_DIR / "lora_metadata.json"

LORA_FAMILIES: list[str] = [
    "SD 1.5",
    "SDXL",
    "Pony",
    "Illustrious",
    "Flux",
    "Z-Image",
    "Wan",
    "LTX",
    "Other",
]
UNCATEGORIZED = "Uncategorized"


def _load_all() -> dict:
    if _META_FILE.is_file():
        try:
            return json.loads(_META_FILE.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to read lora_metadata.json", exc_info=True)
    return {}


def _save_all(data: dict) -> None:
    _META_DIR.mkdir(parents=True, exist_ok=True)
    _META_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_lora_meta(filename: str) -> dict:
    """Return metadata dict for a LoRA file.

    Keys: activation_text, trigger_phrase, base_strength, description,
    keywords, family.
    """
    all_meta = _load_all()
    return all_meta.get(filename, {})


def set_lora_meta(
    filename: str,
    activation_text: str,
    trigger_phrase: str,
    base_strength: float,
) -> None:
    """Persist activation/trigger/strength for a LoRA file (legacy signature)."""
    update_lora_meta(
        filename,
        activation_text=activation_text,
        trigger_phrase=trigger_phrase,
        base_strength=base_strength,
    )


def update_lora_meta(filename: str, **fields) -> None:
    """Partial update — merges *fields* into the entry for *filename*."""
    all_meta = _load_all()
    entry = dict(all_meta.get(filename, {}))
    entry.update(fields)
    all_meta[filename] = entry
    _save_all(all_meta)


def get_lora_family(filename: str) -> str:
    """Return the configured family for *filename*, or empty string."""
    return get_lora_meta(filename).get("family", "") or ""


def get_lora_keywords(filename: str) -> str:
    """Return the configured keywords text for *filename*, or empty string."""
    return get_lora_meta(filename).get("keywords", "") or ""


def lora_prompt_tag(filename: str) -> str:
    """Build prompt text for a checked LoRA.

    Format: ``<lora:activation_text:strength> <keywords>``.
    Falls back to legacy ``trigger_phrase`` if ``keywords`` is unset.
    """
    meta = get_lora_meta(filename)
    stem = Path(filename).stem
    act = meta.get("activation_text", "") or stem
    strength = meta.get("base_strength", 1.0)
    appended = (meta.get("keywords", "") or meta.get("trigger_phrase", "") or "").strip()
    tag = f"<lora:{act}:{strength}>"
    if appended:
        tag += f" {appended}"
    return tag


def populate_lora_list_grouped(
    list_widget: QListWidget,
    names: Iterable[str],
    activated: Iterable[str] | None = None,
) -> None:
    """Clear *list_widget* and populate with checkable LoRAs grouped by family.

    Group headers use ``Qt.NoItemFlags`` so they cannot be selected or checked.
    Items appear in family order (``LORA_FAMILIES`` first, then
    ``Uncategorized``); within each family, names are case-insensitive sorted.
    """
    activated_set = set(activated or [])
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        fam = get_lora_family(name) or UNCATEGORIZED
        groups[fam].append(name)

    list_widget.clear()
    family_order = LORA_FAMILIES + [UNCATEGORIZED]
    seen: set[str] = set()
    ordered_families = [f for f in family_order if f in groups]
    for fam in groups:
        if fam not in family_order:
            ordered_families.append(fam)
    for fam in ordered_families:
        if fam in seen:
            continue
        seen.add(fam)
        header = QListWidgetItem(f"— {fam} —")
        header.setFlags(Qt.NoItemFlags)
        font = header.font()
        font.setBold(True)
        font.setItalic(True)
        header.setFont(font)
        list_widget.addItem(header)
        for name in sorted(groups[fam], key=str.lower):
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if name in activated_set else Qt.CheckState.Unchecked
            )
            list_widget.addItem(item)


class LoRAConfigDialog(QDialog):
    """Popup for editing per-LoRA metadata (activation text, trigger phrase, strength)."""

    def __init__(self, filename: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("LoRA Settings")
        self.setMinimumWidth(400)

        meta = get_lora_meta(filename)
        stem = Path(filename).stem

        layout = QVBoxLayout(self)

        title = QLabel(f"<b>{filename}</b>")
        title.setStyleSheet("font-size: 14px; padding: 4px 0;")
        layout.addWidget(title)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Activation Text:"))
        self._activation = QLineEdit()
        self._activation.setText(meta.get("activation_text", "") or stem)
        self._activation.setPlaceholderText(stem)
        row1.addWidget(self._activation, 1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Trigger Phrase:"))
        self._trigger = QLineEdit()
        self._trigger.setText(meta.get("trigger_phrase", ""))
        self._trigger.setPlaceholderText("(optional)")
        row2.addWidget(self._trigger, 1)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Base Strength:"))
        self._strength = QDoubleSpinBox()
        self._strength.setRange(0.0, 5.0)
        self._strength.setDecimals(2)
        self._strength.setSingleStep(0.05)
        self._strength.setValue(meta.get("base_strength", 1.0))
        self._strength.setFixedWidth(80)
        row3.addWidget(self._strength)
        row3.addStretch()
        layout.addLayout(row3)

        layout.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self._filename = filename

    def accept(self) -> None:
        update_lora_meta(
            self._filename,
            activation_text=self._activation.text().strip(),
            trigger_phrase=self._trigger.text().strip(),
            base_strength=self._strength.value(),
        )
        super().accept()
