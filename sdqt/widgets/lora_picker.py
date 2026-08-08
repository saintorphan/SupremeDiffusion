"""Reusable LoRA picker widget with staging area.

Layout:
  1. Browser — checkable list grouped by base model, filter combo
  2. Staging Area — each checked LoRA gets its own row with:
       [name] [---weight slider---] [±0.70] [Auto]
       trigger words as clickable insert buttons
  3. Actions — "Insert All to Prompt", "Weave Triggers (Qwen)"

Signals:
  insert_to_prompt(str): text to insert into the active prompt field
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from sdqt.widgets.lora_metadata import LoRAConfigDialog, lora_prompt_tag

logger = logging.getLogger(__name__)

# Base-model labels returned by civitai .info files, mapped to display groups
_BASE_MODEL_MAP = {
    "sd 1.5": "SD 1.5",
    "sd1.5": "SD 1.5",
    "sd15": "SD 1.5",
    "sdxl 1.0": "SDXL",
    "sdxl": "SDXL",
    "sdxl 0.9": "SDXL",
    "pony": "Pony",
    "pony diffusion": "Pony",
    "illustrious": "Illustrious",
    "noobai": "NoobAI",
    "flux.1 d": "Flux",
    "flux.1 s": "Flux",
    "flux": "Flux",
    "sd 3": "SD3",
    "sd3": "SD3",
}

# Display order for model groups
_GROUP_ORDER = ["Pony", "SDXL", "Illustrious", "NoobAI", "Flux", "SD 1.5", "SD3", "Other"]


def _normalize_base_model(raw: str) -> str:
    """Map a raw base_model string to a display group name."""
    if not raw:
        return "Other"
    low = raw.lower().strip()
    if low in _BASE_MODEL_MAP:
        return _BASE_MODEL_MAP[low]
    for key, group in _BASE_MODEL_MAP.items():
        if key in low:
            return group
    return "Other"


def _detect_lora_base_model(lora_file: Path) -> str:
    """Try to detect the base model from .civitai.info sidecar."""
    base = lora_file.with_suffix("")
    for pattern in [f"{base}.civitai.info", f"{base}_civitai.info"]:
        p = Path(pattern)
        if p.is_file():
            try:
                import json
                data = json.loads(p.read_text(encoding="utf-8"))
                raw = data.get("baseModel", "")
                if raw:
                    return _normalize_base_model(raw)
            except Exception:
                pass
    parent = lora_file.parent.name.lower()
    for key, group in _BASE_MODEL_MAP.items():
        if key in parent:
            return group
    return "Other"


# ---------------------------------------------------------------------------
# Staging row — one per checked LoRA
# ---------------------------------------------------------------------------

class _LoRAStagingRow(QFrame):
    """A single LoRA in the staging area: name, weight slider, trigger buttons."""

    remove_clicked = Signal(str)            # filename
    insert_trigger = Signal(str)            # single trigger word to insert
    insert_lora_tag = Signal(str)           # "<lora:name:weight>"

    def __init__(self, filename: str, meta: dict, recommended_weight: float,
                 parent=None) -> None:
        super().__init__(parent)
        self.filename = filename
        self._meta = meta
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QFrame { background: #1a1a1a; border: 1px solid #333; "
            "border-radius: 6px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(5)

        # ── Row 1: name + weight slider + remove ──
        top = QHBoxLayout()
        top.setSpacing(6)

        stem = Path(filename).stem
        name_lbl = QLabel(f"<b>{stem}</b>")
        name_lbl.setStyleSheet("color: #ddd; font-size: 12px;")
        name_lbl.setMaximumWidth(180)
        name_lbl.setMinimumWidth(60)
        name_lbl.setToolTip(filename)
        from PySide6.QtWidgets import QSizePolicy
        name_lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        top.addWidget(name_lbl)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(-200, 200)
        self._slider.setValue(int(recommended_weight * 100))
        self._slider.setMinimumWidth(100)
        self._slider.setToolTip(
            "LoRA weight (-2.00 to 2.00)\n"
            "Positive = apply effect, Negative = suppress/invert"
        )
        self._slider.valueChanged.connect(self._on_slider)
        top.addWidget(self._slider, 2)

        self._spin = QDoubleSpinBox()
        self._spin.setRange(-2.0, 2.0)
        self._spin.setDecimals(2)
        self._spin.setSingleStep(0.05)
        self._spin.setValue(recommended_weight)
        self._spin.setFixedWidth(65)
        self._spin.valueChanged.connect(self._on_spin)
        top.addWidget(self._spin)

        auto_btn = QPushButton("Auto")
        auto_btn.setFixedSize(38, 22)
        auto_btn.setToolTip("Reset to sidecar-recommended weight")
        auto_btn.setStyleSheet("font-size: 10px;")
        auto_btn.clicked.connect(lambda: self.set_weight(recommended_weight))
        top.addWidget(auto_btn)

        insert_btn = QPushButton("Ins")
        insert_btn.setFixedSize(32, 22)
        insert_btn.setToolTip("Insert <lora:name:weight> tag into prompt")
        insert_btn.setStyleSheet("font-size: 10px; color: #8f8;")
        insert_btn.clicked.connect(self._emit_lora_tag)
        top.addWidget(insert_btn)

        rm_btn = QPushButton("x")
        rm_btn.setFixedSize(22, 22)
        rm_btn.setToolTip("Remove from staging")
        rm_btn.setStyleSheet(
            "QPushButton { color: #f66; font-weight: bold; font-size: 12px; "
            "background: transparent; border: none; }"
            "QPushButton:hover { color: #f00; }"
        )
        rm_btn.clicked.connect(lambda: self.remove_clicked.emit(self.filename))
        top.addWidget(rm_btn)

        layout.addLayout(top)

        # ── Row 2: source info ──
        sources = meta.get("metadata_sources", [])
        rec_w = meta.get("recommended_weight", 0.7)
        if sources:
            src_lbl = QLabel(f"Recommended: {rec_w:.2f} (from {', '.join(sources)})")
        else:
            src_lbl = QLabel("No sidecar data — default 0.70")
        src_lbl.setStyleSheet("color: #7a7; font-size: 10px;")
        layout.addWidget(src_lbl)

        # ── Row 3: trigger word buttons ──
        triggers = meta.get("trigger_words", [])
        if triggers:
            trig_row = QHBoxLayout()
            trig_row.setSpacing(3)
            trig_lbl = QLabel("Triggers:")
            trig_lbl.setStyleSheet("color: #89b; font-size: 10px;")
            trig_lbl.setFixedWidth(50)
            trig_row.addWidget(trig_lbl)

            for tw in triggers[:10]:
                btn = QPushButton(tw)
                btn.setFixedHeight(20)
                btn.setStyleSheet(
                    "QPushButton { background: #2a3a4a; color: #8bf; font-size: 10px; "
                    "border: 1px solid #456; border-radius: 3px; padding: 1px 5px; }"
                    "QPushButton:hover { background: #3a5a7a; color: #adf; }"
                )
                btn.setToolTip(f"Insert \"{tw}\" into prompt")
                btn.clicked.connect(lambda checked, t=tw: self.insert_trigger.emit(t))
                trig_row.addWidget(btn)

            if len(triggers) > 10:
                more = QLabel(f"+{len(triggers) - 10}")
                more.setStyleSheet("color: #666; font-size: 10px;")
                trig_row.addWidget(more)

            trig_row.addStretch()
            layout.addLayout(trig_row)

    def get_weight(self) -> float:
        return self._spin.value()

    def set_weight(self, w: float) -> None:
        self._slider.blockSignals(True)
        self._spin.blockSignals(True)
        self._slider.setValue(int(w * 100))
        self._spin.setValue(w)
        self._slider.blockSignals(False)
        self._spin.blockSignals(False)

    def get_lora_tag(self) -> str:
        stem = Path(self.filename).stem
        return f"<lora:{stem}:{self.get_weight():.2f}>"

    def get_trigger_words(self) -> list[str]:
        return self._meta.get("trigger_words", [])

    def _on_slider(self, val: int) -> None:
        self._spin.blockSignals(True)
        self._spin.setValue(val / 100.0)
        self._spin.blockSignals(False)

    def _on_spin(self, val: float) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(int(val * 100))
        self._slider.blockSignals(False)

    def _emit_lora_tag(self) -> None:
        self.insert_lora_tag.emit(self.get_lora_tag())


# ---------------------------------------------------------------------------
# Main picker widget
# ---------------------------------------------------------------------------

class LoRAPickerWidget(QGroupBox):
    """LoRA picker with staging area for per-LoRA weight control.

    Signals:
        lora_toggled(str, bool): Legacy — emitted on check/uncheck.
        insert_to_prompt(str): Text to insert into the active prompt field.
    """

    lora_toggled = Signal(str, bool)
    insert_to_prompt = Signal(str)

    def __init__(self, title: str = "LoRA", parent=None) -> None:
        super().__init__(title, parent)
        self._lora_dir: str = ""
        self._suppressing_signals = False
        self._all_loras: list[tuple[str, str]] = []  # [(filename, group), ...]
        self._lora_meta_cache: dict[str, dict] = {}
        self._staging_rows: dict[str, _LoRAStagingRow] = {}  # filename → row
        self._state = None  # set by caller for Qwen access
        self._build_ui()

    def set_state(self, state) -> None:
        """Provide AppState reference for Qwen trigger weaving."""
        self._state = state

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(8, 8, 8, 8)

        # ── Splitter: browser on top, staging on bottom — drag to resize ──
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.setChildrenCollapsible(True)
        self._splitter.setHandleWidth(6)
        self._splitter.setStyleSheet(
            "QSplitter::handle { background: #3a3a3a; border-radius: 2px; }"
            "QSplitter::handle:hover { background: #c9a0ff; }"
        )

        # ── Top pane: browser ──
        browser_pane = QWidget()
        browser_layout = QVBoxLayout(browser_pane)
        browser_layout.setContentsMargins(0, 0, 0, 0)
        browser_layout.setSpacing(3)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(4)
        filter_row.addWidget(QLabel("Browse LoRAs"))
        filter_row.addStretch()
        self._filter_combo = QComboBox()
        self._filter_combo.setMinimumWidth(100)
        self._filter_combo.setToolTip(
            "Filter LoRAs by base model.\n"
            "Auto-detected from .civitai.info sidecars or parent folder name."
        )
        self._filter_combo.addItem("All Models")
        self._filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self._filter_combo)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedWidth(55)
        refresh_btn.clicked.connect(self.refresh)
        filter_row.addWidget(refresh_btn)
        browser_layout.addLayout(filter_row)

        self._list = QListWidget()
        self._list.setMinimumHeight(60)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._list.itemChanged.connect(self._on_item_changed)
        browser_layout.addWidget(self._list)

        self._splitter.addWidget(browser_pane)

        # ── Bottom pane: staging area ──
        staging_pane = QWidget()
        staging_layout = QVBoxLayout(staging_pane)
        staging_layout.setContentsMargins(0, 0, 0, 0)
        staging_layout.setSpacing(3)

        self._staging_label = QLabel("Active LoRAs (0)")
        self._staging_label.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: #c9a0ff;"
        )
        staging_layout.addWidget(self._staging_label)

        self._staging_scroll = QScrollArea()
        self._staging_scroll.setWidgetResizable(True)
        self._staging_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._staging_scroll.setMinimumHeight(40)
        self._staging_container = QWidget()
        self._staging_layout = QVBoxLayout(self._staging_container)
        self._staging_layout.setContentsMargins(0, 0, 0, 0)
        self._staging_layout.setSpacing(4)
        self._staging_layout.addStretch()
        self._staging_scroll.setWidget(self._staging_container)

        self._staging_empty_label = QLabel("Check a LoRA above to add it here")
        self._staging_empty_label.setStyleSheet("color: #555; font-size: 11px;")
        self._staging_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._staging_layout.insertWidget(0, self._staging_empty_label)

        staging_layout.addWidget(self._staging_scroll)
        self._splitter.addWidget(staging_pane)

        # Default split: 55% browser, 45% staging
        self._splitter.setStretchFactor(0, 55)
        self._splitter.setStretchFactor(1, 45)

        layout.addWidget(self._splitter, 1)

        # ── Trigger option ──
        self._include_triggers_cb = QCheckBox("Include Trigger Words in Compose")
        self._include_triggers_cb.setChecked(True)
        self._include_triggers_cb.setToolTip(
            "When checked, trigger words from LoRA metadata\n"
            "(parsed from .civitai.info / .json sidecars)\n"
            "are included when composing the prompt.\n\n"
            "Uncheck to insert only <lora:name:weight> tags."
        )
        self._include_triggers_cb.setStyleSheet(
            "QCheckBox { color: #8bf; font-size: 11px; }"
        )
        layout.addWidget(self._include_triggers_cb)

        # ── Action buttons ──
        action_row = QHBoxLayout()
        action_row.setSpacing(4)

        self._insert_all_btn = QPushButton("Insert All to Prompt")
        self._insert_all_btn.setToolTip(
            "Insert all staged LoRA tags (and trigger words if checked) into the prompt"
        )
        self._insert_all_btn.setStyleSheet(
            "QPushButton { background: #2a4a2a; color: #8f8; font-size: 11px; "
            "padding: 4px 10px; border-radius: 3px; }"
            "QPushButton:hover { background: #3a6a3a; }"
        )
        self._insert_all_btn.clicked.connect(self._on_insert_all)
        action_row.addWidget(self._insert_all_btn)

        self._weave_btn = QPushButton("Weave Triggers (Qwen)")
        self._weave_btn.setToolTip(
            "Use Qwen AI to naturally incorporate trigger words into your prompt.\n"
            "Reads the current prompt, adds trigger words in a natural way,\n"
            "and emits the rewritten prompt."
        )
        self._weave_btn.setStyleSheet(
            "QPushButton { background: #3a2a4a; color: #c9a0ff; font-size: 11px; "
            "padding: 4px 10px; border-radius: 3px; }"
            "QPushButton:hover { background: #5a3a7a; }"
        )
        self._weave_btn.clicked.connect(self._on_weave_triggers)
        action_row.addWidget(self._weave_btn)

        action_row.addStretch()
        layout.addLayout(action_row)

    # -- Staging count ---------------------------------------------------------

    def _update_staging_count(self) -> None:
        n = len(self._staging_rows)
        self._staging_label.setText(f"Active LoRAs ({n})")

    # -- Directory + scan ----------------------------------------------------

    def set_lora_dir(self, path: str) -> None:
        self._lora_dir = path
        self.refresh()

    def refresh(self) -> None:
        checked = set()
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) and \
               item.checkState() == Qt.CheckState.Checked:
                checked.add(item.text())

        self._all_loras.clear()
        groups_seen: set[str] = set()

        if self._lora_dir:
            lora_path = Path(self._lora_dir)
            if lora_path.is_dir():
                for f in sorted(lora_path.rglob("*.safetensors")):
                    group = _detect_lora_base_model(f)
                    self._all_loras.append((f.name, group))
                    groups_seen.add(group)

        self._filter_combo.blockSignals(True)
        current_filter = self._filter_combo.currentText()
        self._filter_combo.clear()
        self._filter_combo.addItem("All Models")
        for g in _GROUP_ORDER:
            if g in groups_seen:
                self._filter_combo.addItem(g)
        idx = self._filter_combo.findText(current_filter)
        self._filter_combo.setCurrentIndex(max(0, idx))
        self._filter_combo.blockSignals(False)

        self._rebuild_list(checked)

    def _rebuild_list(self, checked: set[str] | None = None) -> None:
        if checked is None:
            checked = set()
            for i in range(self._list.count()):
                item = self._list.item(i)
                if item and item.data(Qt.ItemDataRole.UserRole) and \
                   item.checkState() == Qt.CheckState.Checked:
                    checked.add(item.text())

        active_filter = self._filter_combo.currentText()
        show_all = (active_filter == "All Models")

        grouped: dict[str, list[str]] = defaultdict(list)
        for filename, group in self._all_loras:
            if show_all or group == active_filter:
                grouped[group].append(filename)

        self._suppressing_signals = True
        self._list.clear()

        ordered_groups = sorted(
            grouped.keys(),
            key=lambda g: _GROUP_ORDER.index(g) if g in _GROUP_ORDER else 99,
        )
        bold_font = QFont()
        bold_font.setBold(True)

        for group in ordered_groups:
            files = grouped[group]
            if not files:
                continue
            if show_all or len(ordered_groups) > 1:
                header = QListWidgetItem(f"── {group} ({len(files)}) ──")
                header.setFlags(Qt.ItemFlag.NoItemFlags)
                header.setFont(bold_font)
                header.setForeground(Qt.GlobalColor.gray)
                self._list.addItem(header)

            for fname in files:
                item = QListWidgetItem(fname)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setData(Qt.ItemDataRole.UserRole, True)
                item.setCheckState(
                    Qt.CheckState.Checked if fname in checked else Qt.CheckState.Unchecked
                )
                self._list.addItem(item)

        self._suppressing_signals = False

    def _on_filter_changed(self, _idx: int) -> None:
        self._rebuild_list()

    # -- Metadata cache ------------------------------------------------------

    def _get_meta(self, filename: str) -> dict:
        if filename in self._lora_meta_cache:
            return self._lora_meta_cache[filename]
        if not self._lora_dir:
            return {}
        lora_path = Path(self._lora_dir)
        matches = list(lora_path.rglob(filename))
        if not matches:
            return {}
        try:
            from sdqt.widgets.lora_auto_prompter import parse_lora_metadata
            meta = parse_lora_metadata(str(matches[0]))
            self._lora_meta_cache[filename] = meta
            return meta
        except Exception:
            logger.debug("Failed to parse metadata for %s", filename, exc_info=True)
            return {}

    # -- Staging area management ---------------------------------------------

    def _add_to_staging(self, filename: str) -> None:
        if filename in self._staging_rows:
            return
        meta = self._get_meta(filename)
        rec_weight = meta.get("recommended_weight", 0.7) if meta else 0.7

        row = _LoRAStagingRow(filename, meta, rec_weight, parent=self._staging_container)
        row.remove_clicked.connect(self._on_staging_remove)
        row.insert_trigger.connect(self._on_insert_trigger)
        row.insert_lora_tag.connect(self._on_insert_lora_tag)

        self._staging_rows[filename] = row
        # Insert before the stretch
        idx = self._staging_layout.count() - 1
        self._staging_layout.insertWidget(idx, row)
        self._staging_empty_label.hide()
        self._update_staging_count()

    def _remove_from_staging(self, filename: str) -> None:
        row = self._staging_rows.pop(filename, None)
        if row:
            self._staging_layout.removeWidget(row)
            row.deleteLater()
        if not self._staging_rows:
            self._staging_empty_label.show()
        self._update_staging_count()

    @Slot(str)
    def _on_staging_remove(self, filename: str) -> None:
        """User clicked X on a staging row — uncheck in browser too."""
        self._remove_from_staging(filename)
        # Uncheck in the list
        self._suppressing_signals = True
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) and item.text() == filename:
                item.setCheckState(Qt.CheckState.Unchecked)
                break
        self._suppressing_signals = False

    @Slot(str)
    def _on_insert_trigger(self, trigger_word: str) -> None:
        """User clicked a trigger word button — emit for prompt insertion."""
        self.insert_to_prompt.emit(trigger_word)

    @Slot(str)
    def _on_insert_lora_tag(self, tag: str) -> None:
        """User clicked Ins on a staging row — emit the lora tag."""
        self.insert_to_prompt.emit(tag)

    # -- Check toggle --------------------------------------------------------

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._suppressing_signals:
            return
        if not item.data(Qt.ItemDataRole.UserRole):
            return
        filename = item.text()
        checked = item.checkState() == Qt.CheckState.Checked

        if checked:
            self._add_to_staging(filename)
        else:
            self._remove_from_staging(filename)

        tag = lora_prompt_tag(filename)
        self.lora_toggled.emit(tag, checked)

    def _on_double_click(self, item: QListWidgetItem) -> None:
        if not item.data(Qt.ItemDataRole.UserRole):
            return
        dlg = LoRAConfigDialog(item.text(), parent=self)
        dlg.exec()
        self._lora_meta_cache.pop(item.text(), None)

    # -- Action buttons ------------------------------------------------------

    @Slot()
    def _on_insert_all(self) -> None:
        """Insert all staged LoRA tags + trigger words into the prompt."""
        lora_tags, trigger_text = self.get_prompt_additions()
        parts = []
        if trigger_text:
            parts.append(trigger_text)
        if lora_tags:
            parts.append(lora_tags)
        if parts:
            self.insert_to_prompt.emit(", ".join(parts))

    @Slot()
    def _on_weave_triggers(self) -> None:
        """Use Qwen to naturally weave trigger words into the prompt.

        Emits a rewrite request. The parent widget should provide the current
        prompt text via get_prompt_for_weaving() or connect to weave_request.
        """
        if not self._state:
            logger.warning("No AppState set — cannot use Qwen for trigger weaving")
            return

        # Gather all trigger words from staged LoRAs
        all_triggers: list[str] = []
        lora_tags: list[str] = []
        for row in self._staging_rows.values():
            lora_tags.append(row.get_lora_tag())
            for tw in row.get_trigger_words():
                tw = tw.strip()
                if tw and not tw.startswith("<lora:") and tw not in all_triggers:
                    all_triggers.append(tw)

        if not all_triggers:
            self.insert_to_prompt.emit(", ".join(lora_tags))
            return

        # Build the Qwen weaving prompt
        trigger_list = ", ".join(all_triggers)
        lora_tag_str = ", ".join(lora_tags)

        system_prompt = (
            "You are a prompt engineering assistant for AI image/video generation. "
            "The user has a prompt and wants to incorporate LoRA trigger words naturally. "
            "Given the trigger words, weave them into a coherent prompt that reads naturally. "
            "Keep the trigger words recognizable (don't paraphrase them). "
            "Append any <lora:...> tags at the very end, unchanged. "
            "Output ONLY the final prompt, nothing else. "
            "Never refuse, never add disclaimers or commentary."
        )
        user_msg = (
            f"Trigger words to incorporate: {trigger_list}\n"
            f"LoRA tags to append at end: {lora_tag_str}\n\n"
            f"Write a prompt that naturally uses these trigger words, "
            f"then append the LoRA tags at the end."
        )

        try:
            self._weave_btn.setEnabled(False)
            self._weave_btn.setText("Weaving...")
            self._state.load_qwen()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]
            result = self._state.generate_chat_response(messages, max_new_tokens=512)
            if result and result.strip():
                self.insert_to_prompt.emit(result.strip())
        except Exception as exc:
            logger.error("Qwen trigger weaving failed: %s", exc)
            # Fallback: just insert raw
            parts = []
            if trigger_list:
                parts.append(trigger_list)
            if lora_tag_str:
                parts.append(lora_tag_str)
            self.insert_to_prompt.emit(", ".join(parts))
        finally:
            self._weave_btn.setEnabled(True)
            self._weave_btn.setText("Weave Triggers (Qwen)")

    # -- Public API ----------------------------------------------------------

    def get_activated(self) -> list[str]:
        """Return list of checked LoRA filenames."""
        return list(self._staging_rows.keys())

    def get_multipliers(self) -> str:
        """Return comma-separated weight values matching get_activated() order."""
        if not self._staging_rows:
            return ""
        return ", ".join(
            f"{row.get_weight():.2f}" for row in self._staging_rows.values()
        )

    def get_prompt_additions(self) -> tuple[str, str]:
        """Return (lora_tags, trigger_text) for all staged LoRAs.

        lora_tags: e.g. "<lora:my_style:0.80>, <lora:detail_fix:-0.30>"
        trigger_text: e.g. "core_trigger, stage_trigger"
            (empty if "Include Trigger Words" is unchecked)
        """
        if not self._staging_rows:
            return ("", "")

        include_triggers = self._include_triggers_cb.isChecked()
        lora_parts: list[str] = []
        trigger_parts: list[str] = []

        for row in self._staging_rows.values():
            lora_parts.append(row.get_lora_tag())
            if include_triggers:
                for tw in row.get_trigger_words():
                    tw = tw.strip()
                    if tw and not tw.startswith("<lora:") and tw not in trigger_parts:
                        trigger_parts.append(tw)

        return (", ".join(lora_parts), ", ".join(trigger_parts))

    def set_state_obj(self, activated: list[str], multipliers: str = "") -> None:
        """Restore LoRA selection state from config."""
        activated_set = set(activated) if activated else set()
        self._suppressing_signals = True
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole):
                should_check = item.text() in activated_set
                item.setCheckState(
                    Qt.CheckState.Checked if should_check else Qt.CheckState.Unchecked
                )
                if should_check:
                    self._add_to_staging(item.text())
        self._suppressing_signals = False

    def activate_by_stem(self, stem: str, weight: float = 1.0) -> bool:
        """Activate a LoRA by its filename stem (e.g. from ``<lora:name:w>``).

        Searches the list for a file whose stem matches *stem*
        (case-insensitive).  Returns ``True`` if a match was found and
        activated.
        """
        stem_lower = stem.lower()
        for i in range(self._list.count()):
            item = self._list.item(i)
            if not item or not item.data(Qt.ItemDataRole.UserRole):
                continue
            if Path(item.text()).stem.lower() == stem_lower:
                if item.checkState() != Qt.CheckState.Checked:
                    item.setCheckState(Qt.CheckState.Checked)
                # Set weight on staging row if it exists
                row = self._staging_rows.get(item.text())
                if row:
                    row.set_weight(weight)
                return True
        return False
