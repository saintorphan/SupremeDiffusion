"""LoRA Manager tab — browse, categorize, and annotate installed image LoRAs."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sdqt.tabs.base import BaseTab
from sdqt.widgets.lora_metadata import (
    LORA_FAMILIES,
    UNCATEGORIZED,
    get_lora_meta,
    update_lora_meta,
)

logger = logging.getLogger(__name__)

_FILTER_ALL = "All"


class LoRAManagerTab(BaseTab):
    """List installed image LoRAs and edit their description / keywords / family.

    The metadata is shared with the global lora_metadata.json store, so the
    keywords entered here are appended automatically by every LoRA list whose
    tickbox triggers ``lora_prompt_tag()``. The family selection drives the
    "— Category —" grouping in LoRA dropdowns elsewhere in the app.
    """

    def __init__(self, state, parent=None) -> None:
        super().__init__(state, parent)
        self._all_loras: list[str] = []
        self._current: str | None = None
        self._dirty: bool = False
        self._loading_fields: bool = False
        self._build_ui()
        self._refresh_loras()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # Top toolbar
        top = QHBoxLayout()
        top.setSpacing(6)
        top.addWidget(QLabel("Filter:"))
        self._filter_combo = QComboBox()
        self._filter_combo.setFixedWidth(140)
        self._filter_combo.addItem(_FILTER_ALL)
        self._filter_combo.addItems(LORA_FAMILIES)
        self._filter_combo.addItem(UNCATEGORIZED)
        self._filter_combo.currentIndexChanged.connect(self._apply_filter)
        top.addWidget(self._filter_combo)

        top.addWidget(QLabel("Search:"))
        self._search = QLineEdit()
        self._search.setPlaceholderText("name contains…")
        self._search.textChanged.connect(self._apply_filter)
        top.addWidget(self._search, 1)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Rescan the SD LoRA directory")
        refresh_btn.clicked.connect(self._refresh_loras)
        top.addWidget(refresh_btn)
        root.addLayout(top)

        # Splitter: list on left, editor on right
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_selection_changed)
        splitter.addWidget(self._list)

        editor = QWidget()
        ev = QVBoxLayout(editor)
        ev.setContentsMargins(8, 4, 4, 4)
        ev.setSpacing(6)

        self._title = QLabel("(select a LoRA)")
        self._title.setStyleSheet("font-size: 13px; font-weight: bold;")
        self._title.setWordWrap(True)
        ev.addWidget(self._title)

        # Family
        fam_row = QHBoxLayout()
        fam_row.setSpacing(6)
        fam_row.addWidget(QLabel("Family:"))
        self._family = QComboBox()
        self._family.setFixedWidth(160)
        self._family.addItem("(unset)")
        self._family.addItems(LORA_FAMILIES)
        self._family.currentIndexChanged.connect(self._mark_dirty)
        fam_row.addWidget(self._family)
        fam_row.addStretch()
        ev.addLayout(fam_row)

        # Activation text + base strength
        act_row = QHBoxLayout()
        act_row.setSpacing(6)
        act_row.addWidget(QLabel("Activation:"))
        self._activation = QLineEdit()
        self._activation.setToolTip(
            "Token used inside <lora:NAME:strength>. Defaults to filename stem."
        )
        self._activation.textChanged.connect(self._mark_dirty)
        act_row.addWidget(self._activation, 1)
        act_row.addWidget(QLabel("Strength:"))
        self._strength = QDoubleSpinBox()
        self._strength.setRange(0.0, 5.0)
        self._strength.setDecimals(2)
        self._strength.setSingleStep(0.05)
        self._strength.setValue(1.0)
        self._strength.setFixedWidth(80)
        self._strength.valueChanged.connect(self._mark_dirty)
        act_row.addWidget(self._strength)
        ev.addLayout(act_row)

        # Keywords
        ev.addWidget(QLabel("Keywords (appended to prompt after the LoRA tag):"))
        self._keywords = QPlainTextEdit()
        self._keywords.setPlaceholderText(
            "e.g. masterpiece, intricate detail, cinematic lighting"
        )
        self._keywords.setMaximumHeight(80)
        self._keywords.textChanged.connect(self._mark_dirty)
        ev.addWidget(self._keywords)

        # Description
        ev.addWidget(QLabel("Description (notes only — not added to the prompt):"))
        self._description = QPlainTextEdit()
        self._description.setPlaceholderText("Free-form notes about this LoRA…")
        self._description.textChanged.connect(self._mark_dirty)
        ev.addWidget(self._description, 1)

        # Save / Revert
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._status = QLabel("")
        self._status.setStyleSheet("color: #888;")
        btn_row.addWidget(self._status, 1)
        self._revert_btn = QPushButton("Revert")
        self._revert_btn.clicked.connect(self._revert)
        btn_row.addWidget(self._revert_btn)
        self._save_btn = QPushButton("Save")
        self._save_btn.setStyleSheet("font-weight: bold;")
        self._save_btn.clicked.connect(self._save)
        btn_row.addWidget(self._save_btn)
        ev.addLayout(btn_row)

        splitter.addWidget(editor)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([260, 520])
        root.addWidget(splitter, 1)

        self._set_editor_enabled(False)

    # ── Population ────────────────────────────────────────────────────────────

    @Slot()
    def _refresh_loras(self) -> None:
        cfg = self.state.global_config
        lora_dir = cfg.model_paths.get("sd_lora_dir", "")
        names: list[str] = []
        if lora_dir:
            try:
                from supremediffusion.models.sd_models import scan_loras
                names = scan_loras(lora_dir)
            except Exception as exc:
                logger.warning("Failed to scan SD LoRAs: %s", exc)
        self._all_loras = names
        self._apply_filter()

    def populate_options(self, loras: list[str] | None = None, **_kwargs) -> None:
        """Receive the LoRA list shared by the rest of the Image suite."""
        if loras is not None:
            self._all_loras = list(loras)
            self._apply_filter()

    def _apply_filter(self) -> None:
        target_family = self._filter_combo.currentText()
        query = self._search.text().strip().lower()
        self._list.blockSignals(True)
        self._list.clear()
        for name in self._all_loras:
            if query and query not in name.lower():
                continue
            fam = (get_lora_meta(name).get("family") or UNCATEGORIZED)
            if target_family != _FILTER_ALL and fam != target_family:
                continue
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setToolTip(f"{name}\nFamily: {fam}")
            self._list.addItem(item)
        self._list.blockSignals(False)
        if self._list.count() > 0:
            self._list.setCurrentRow(0)
        else:
            self._current = None
            self._set_editor_enabled(False)
            self._title.setText("(no LoRAs match)")
            self._clear_fields()

    # ── Selection ─────────────────────────────────────────────────────────────

    def _on_selection_changed(self, current: QListWidgetItem | None,
                              previous: QListWidgetItem | None) -> None:
        if self._dirty and previous is not None:
            self._save(silent=True)
        if current is None:
            self._current = None
            self._set_editor_enabled(False)
            self._title.setText("(select a LoRA)")
            self._clear_fields()
            return
        name = current.data(Qt.ItemDataRole.UserRole) or current.text()
        self._current = name
        self._set_editor_enabled(True)
        self._load_fields(name)

    def _load_fields(self, filename: str) -> None:
        meta = get_lora_meta(filename)
        stem = Path(filename).stem
        self._loading_fields = True
        try:
            self._title.setText(filename)
            fam = meta.get("family", "") or ""
            idx = self._family.findText(fam) if fam else 0
            self._family.setCurrentIndex(idx if idx >= 0 else 0)
            self._activation.setText(meta.get("activation_text", "") or stem)
            self._activation.setPlaceholderText(stem)
            self._strength.setValue(float(meta.get("base_strength", 1.0) or 1.0))
            keywords = meta.get("keywords", "")
            if not keywords:
                # Surface legacy single-line trigger_phrase as a starting point
                keywords = meta.get("trigger_phrase", "") or ""
            self._keywords.setPlainText(keywords)
            self._description.setPlainText(meta.get("description", "") or "")
        finally:
            self._loading_fields = False
        self._dirty = False
        self._status.setText("")

    def _clear_fields(self) -> None:
        self._loading_fields = True
        try:
            self._family.setCurrentIndex(0)
            self._activation.clear()
            self._activation.setPlaceholderText("")
            self._strength.setValue(1.0)
            self._keywords.clear()
            self._description.clear()
        finally:
            self._loading_fields = False
        self._dirty = False
        self._status.setText("")

    def _set_editor_enabled(self, enabled: bool) -> None:
        for w in (self._family, self._activation, self._strength,
                  self._keywords, self._description, self._save_btn,
                  self._revert_btn):
            w.setEnabled(enabled)

    # ── Save / Revert ─────────────────────────────────────────────────────────

    def _mark_dirty(self, *_args) -> None:
        if self._loading_fields:
            return
        self._dirty = True
        self._status.setText("● unsaved")

    @Slot()
    def _save(self, silent: bool = False) -> None:
        if not self._current or not self._dirty:
            return
        family_idx = self._family.currentIndex()
        family = "" if family_idx == 0 else self._family.currentText()
        update_lora_meta(
            self._current,
            family=family,
            activation_text=self._activation.text().strip(),
            base_strength=self._strength.value(),
            keywords=self._keywords.toPlainText().strip(),
            description=self._description.toPlainText().strip(),
        )
        self._dirty = False
        if not silent:
            self._status.setText("Saved.")
        # Refresh tooltip / category for the active row
        item = self._list.currentItem()
        if item is not None:
            fam = family or UNCATEGORIZED
            item.setToolTip(f"{self._current}\nFamily: {fam}")

    @Slot()
    def _revert(self) -> None:
        if not self._current:
            return
        self._load_fields(self._current)
        self._status.setText("Reverted.")
