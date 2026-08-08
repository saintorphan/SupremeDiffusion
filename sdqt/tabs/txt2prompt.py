"""Txt2Prompt tab — LLM-written, model-formatted video prompts.

Describe an idea, pick the target video model, and the abliterated Qwen GGUF
(loaded via AppState's memory-managed prompt LLM) writes a correctly-formatted
prompt — single, per-second timed, or per-window. Each segment can be copied or
AI-refined, and "Send to" pushes the prompt + model + frames + window + fps + mode
straight into the Generate or Video Extender tab.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton,
    QScrollArea, QSpinBox, QVBoxLayout, QWidget, QCheckBox, QFrame,
)

from sdqt.state import AppState
from sdqt.widgets.generation_params import _MODEL_TYPES, get_video_backend
from supremediffusion.core import prompt_writer as pw
from sdqt.workers.prompt_write import PromptWriteWorker, PromptRefineWorker

from .base import BaseTab

_STORE = Path.home() / ".supremediffusion" / "txt2prompt.json"
_WAN_FRAMES = [17, 25, 33, 41, 49, 65, 81]
_LTX_FRAMES = [9, 25, 57, 121, 161, 241, 321]


class Txt2PromptTab(BaseTab):
    """Write model-formatted video prompts with the embedded prompt LLM."""

    # (target, payload) -> wired by MainWindow to Generate / Video Extender
    send_to_requested = Signal(str, dict)

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._worker = None
        self._refine_workers: list = []
        self._last_result: dict | None = None
        self._build_ui()
        self._on_model_changed()
        self._refresh_saved()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 0, 2, 2)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        root = QWidget()
        scroll.setWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(8)

        # Model + caps
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel("Video model:"))
        self._model = QComboBox()
        self._model.setMinimumWidth(280)
        for label, key in _MODEL_TYPES:
            self._model.addItem(label, key)
        self._model.currentIndexChanged.connect(self._on_model_changed)
        row.addWidget(self._model)
        row.addStretch()
        layout.addLayout(row)
        self._caps = QLabel()
        self._caps.setStyleSheet("color:#8c95a4; font-size:12px;")
        layout.addWidget(self._caps)

        # Breakdown mode
        mrow = QHBoxLayout()
        mrow.setSpacing(6)
        mrow.addWidget(QLabel("Breakdown:"))
        self._mode = QComboBox()
        self._mode.setMinimumWidth(160)
        self._mode.currentIndexChanged.connect(self._on_mode_changed)
        mrow.addWidget(self._mode)
        mrow.addStretch()
        layout.addLayout(mrow)

        # per-second controls
        self._ps_box = QWidget()
        ps = QHBoxLayout(self._ps_box)
        ps.setContentsMargins(0, 0, 0, 0); ps.setSpacing(6)
        ps.addWidget(QLabel("Duration:"))
        self._seconds = QSpinBox(); self._seconds.setRange(1, 60); self._seconds.setValue(5)
        self._seconds.setFixedWidth(75); self._seconds.valueChanged.connect(self._update_readout)
        ps.addWidget(self._seconds); ps.addWidget(QLabel("s"))
        ps.addStretch()
        layout.addWidget(self._ps_box)

        # per-window controls
        self._win_box = QWidget()
        wb = QHBoxLayout(self._win_box)
        wb.setContentsMargins(0, 0, 0, 0); wb.setSpacing(6)
        wb.addWidget(QLabel("Windows:"))
        self._win_count = QSpinBox(); self._win_count.setRange(1, 30); self._win_count.setValue(2)
        self._win_count.setFixedWidth(70)
        self._win_count.valueChanged.connect(self._update_readout)
        wb.addWidget(self._win_count)
        wb.addWidget(QLabel("Window:"))
        self._window = QSpinBox(); self._window.setRange(17, 1000); self._window.setSingleStep(8)
        self._window.setValue(121); self._window.setFixedWidth(80)
        self._window.valueChanged.connect(self._update_readout)
        wb.addWidget(self._window)
        wb.addWidget(QLabel("fps:"))
        self._fps = QSpinBox(); self._fps.setRange(1, 60); self._fps.setValue(30); self._fps.setFixedWidth(60)
        self._fps.valueChanged.connect(self._update_readout)
        wb.addWidget(self._fps)
        self._readout = QLabel(); self._readout.setStyleSheet("color:#8c95a4; font-size:12px;")
        wb.addWidget(self._readout)
        wb.addStretch()
        layout.addWidget(self._win_box)

        # idea + negative
        layout.addWidget(QLabel("Main prompt / idea:"))
        self._idea = QPlainTextEdit()
        self._idea.setPlaceholderText("Describe the video — subject, action, mood, camera. The LLM expands & formats it for the selected model.")
        self._idea.setMinimumHeight(100)  # ~4+ rows; QPlainTextEdit word-wraps
        layout.addWidget(self._idea)
        layout.addWidget(QLabel("Negative prompt:"))
        self._neg = QPlainTextEdit()
        self._neg.setPlaceholderText("Things to avoid (or let the LLM write one).")
        self._neg.setMinimumHeight(90)
        layout.addWidget(self._neg)
        self._gen_neg = QCheckBox("Let the LLM write the negative prompt")
        self._gen_neg.setChecked(True)
        layout.addWidget(self._gen_neg)

        # actions
        act = QHBoxLayout(); act.setSpacing(8)
        self._btn_gen = QPushButton("✨ Generate")
        self._btn_gen.setProperty("class", "primary")
        self._btn_gen.clicked.connect(self._generate)
        act.addWidget(self._btn_gen)
        self._btn_abort = QPushButton("■ Stop"); self._btn_abort.setVisible(False)
        self._btn_abort.clicked.connect(self._abort)
        act.addWidget(self._btn_abort)
        self._status = QLabel(""); self._status.setStyleSheet("color:#8c95a4; font-size:12px;")
        act.addWidget(self._status); act.addStretch()
        layout.addLayout(act)

        # output
        head = QHBoxLayout()
        head.addWidget(QLabel("<b>Generated prompt</b>"))
        b = QPushButton("Copy"); b.setFixedWidth(60); b.clicked.connect(lambda: self._copy(self._out_prompt.toPlainText()))
        head.addStretch(); head.addWidget(b)
        layout.addLayout(head)
        self._note = QLabel(""); self._note.setStyleSheet("color:#e0b34a; font-size:12px;"); self._note.setWordWrap(True)
        self._note.setVisible(False); layout.addWidget(self._note)
        self._out_prompt = QPlainTextEdit(); self._out_prompt.setMinimumHeight(100)
        layout.addWidget(self._out_prompt)

        # per-segment breakdown rows
        self._seg_frame = QFrame()
        self._seg_layout = QVBoxLayout(self._seg_frame)
        self._seg_layout.setContentsMargins(0, 0, 0, 0); self._seg_layout.setSpacing(4)
        self._seg_frame.setVisible(False)
        layout.addWidget(self._seg_frame)

        nhead = QHBoxLayout()
        nhead.addWidget(QLabel("<b>Negative prompt</b>"))
        nb = QPushButton("Copy"); nb.setFixedWidth(60); nb.clicked.connect(lambda: self._copy(self._out_neg.toPlainText()))
        nhead.addStretch(); nhead.addWidget(nb)
        layout.addLayout(nhead)
        self._out_neg = QPlainTextEdit(); self._out_neg.setMinimumHeight(90)
        layout.addWidget(self._out_neg)

        # save / load / delete
        srow = QHBoxLayout(); srow.setSpacing(6)
        self._save_name = QLineEdit(); self._save_name.setPlaceholderText("Name to save…")
        srow.addWidget(self._save_name)
        sb = QPushButton("💾 Save"); sb.clicked.connect(self._save); srow.addWidget(sb)
        self._saved = QComboBox(); self._saved.setMinimumWidth(180); srow.addWidget(self._saved)
        lb = QPushButton("Load"); lb.clicked.connect(self._load); srow.addWidget(lb)
        db = QPushButton("Delete"); db.clicked.connect(self._delete); srow.addWidget(db)
        layout.addLayout(srow)

        # send-to
        send = QHBoxLayout(); send.setSpacing(8)
        send.addWidget(QLabel("Send to:"))
        g = QPushButton("→ Img2Vid"); g.clicked.connect(lambda: self._send("generate")); send.addWidget(g)
        g2 = QPushButton("→ Img2Vid Mode 2 (first/last)"); g2.clicked.connect(self._send_mode2); send.addWidget(g2)
        v = QPushButton("→ Video Extender"); v.clicked.connect(lambda: self._send("video_extender")); send.addWidget(v)
        send.addStretch()
        layout.addLayout(send)
        layout.addStretch()

    # ---------- model / mode ----------
    def _model_key(self) -> str:
        return self._model.currentData() or ""

    def _natural_mode(self, key: str) -> str:
        if pw.supports_per_second(key):
            return "per-second"
        if pw.supports_sliding(key):
            return "per-window"
        return "single"

    @Slot()
    def _on_model_changed(self) -> None:
        key = self._model_key()
        be = get_video_backend(key)
        # backend defaults
        self._fps.setValue(30 if be == "ltx" else 16)
        self._window.setValue(121 if be == "ltx" else 81)
        self._win_count.setValue(2)
        caps = [f"backend: {be}"]
        if pw.supports_per_second(key):
            caps.append("per-second ✓")
        if pw.supports_sliding(key):
            caps.append("sliding-window ✓")
        self._caps.setText("  •  ".join(caps))
        # rebuild mode options
        self._mode.blockSignals(True)
        self._mode.clear()
        self._mode.addItem("Single prompt", "single")
        if pw.supports_per_second(key):
            self._mode.addItem("Per-second", "per-second")
        if pw.supports_sliding(key):
            self._mode.addItem("Sliding window", "per-window")
        idx = self._mode.findData(self._natural_mode(key))
        self._mode.setCurrentIndex(max(0, idx))
        self._mode.blockSignals(False)
        self._on_mode_changed()

    @Slot()
    def _on_mode_changed(self) -> None:
        mode = self._mode.currentData() or "single"
        self._ps_box.setVisible(mode == "per-second")
        self._win_box.setVisible(mode == "per-window")
        self._update_readout()

    def _update_readout(self) -> None:
        if (self._mode.currentData() or "single") == "per-window":
            total = self._win_count.value() * self._window.value()
            spw = pw.seconds_per_window(self._window.value(), self._fps.value())
            secs = total / max(self._fps.value(), 1)
            self._readout.setText(f"= {total} frames · ~{spw:.1f}s/window · ~{secs:.1f}s total")

    # ---------- generate ----------
    def _build_req(self) -> dict:
        key = self._model_key()
        mode = self._mode.currentData() or "single"
        fps = self._fps.value()
        req = {
            "model_key": key,
            "model_label": self._model.currentText(),
            "idea": self._idea.toPlainText(),
            "negative": self._neg.toPlainText(),
            "generate_negative": self._gen_neg.isChecked(),
            "mode": mode,
            "fps": fps,
            "frames": 0,
            "window_size": self._window.value(),
            "overlap": 0,
        }
        if mode == "per-second":
            req["frames"] = self._seconds.value() * fps
        elif mode == "per-window":
            req["frames"] = self._win_count.value() * self._window.value()
        return req

    @Slot()
    def _generate(self) -> None:
        if not self._idea.toPlainText().strip():
            self._show_status("Enter a main prompt / idea first.")
            return
        self._btn_gen.setEnabled(False)
        self._btn_abort.setVisible(True)
        self._status.setText("Working…")
        self._worker = PromptWriteWorker(self.state, self._build_req(), parent=self)
        self._worker.status.connect(self._status.setText)
        self._worker.error.connect(self._on_error)
        self._worker.finished_ok.connect(self._on_result)
        self._worker.finished.connect(lambda: (self._btn_gen.setEnabled(True), self._btn_abort.setVisible(False)))
        self._worker.start()

    def _abort(self) -> None:
        if self._worker:
            self._worker.abort()

    def _on_error(self, msg: str) -> None:
        self._status.setText("")
        self._show_status(f"Prompt LLM error: {msg}")

    def _on_result(self, result: dict) -> None:
        self._last_result = result
        self._out_prompt.setPlainText(result.get("prompt", ""))
        self._out_neg.setPlainText(result.get("negative_prompt", ""))
        note = result.get("note", "")
        self._note.setText(note); self._note.setVisible(bool(note))
        self._render_segments(result.get("segments", []), result.get("mode", "single"))
        self._status.setText(f"Done · {result.get('mode')} · {result.get('model_label')}")

    # ---------- per-segment rows ----------
    def _clear_segments(self) -> None:
        while self._seg_layout.count():
            it = self._seg_layout.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()

    def _render_segments(self, segments: list, mode: str) -> None:
        self._clear_segments()
        if not segments or len(segments) <= 1:
            self._seg_frame.setVisible(False)
            return
        for seg in segments:
            row = QWidget()
            rl = QVBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(2)
            top = QHBoxLayout(); top.setSpacing(6)
            lab = QLabel(seg["label"]); lab.setFixedWidth(56); lab.setStyleSheet("color:#6aa3ff;")
            top.addWidget(lab, 0, Qt.AlignmentFlag.AlignTop)
            field = QPlainTextEdit(seg["text"]); field.setMinimumHeight(84)  # ~4 rows, wraps
            top.addWidget(field, 1)
            cp = QPushButton("Copy"); cp.setFixedWidth(54)
            cp.clicked.connect(lambda _=False, f=field: self._copy(f.toPlainText()))
            top.addWidget(cp, 0, Qt.AlignmentFlag.AlignTop)
            rl.addLayout(top)
            bot = QHBoxLayout(); bot.setSpacing(6)
            bot.addSpacing(56)
            instr = QLineEdit(); instr.setPlaceholderText(f"Tell the AI how to change {seg['label']} (optional)…")
            bot.addWidget(instr, 1)
            wand = QPushButton("🪄"); wand.setFixedWidth(40)
            wand.clicked.connect(lambda _=False, f=field, i=instr, w=wand, lb=seg["label"]: self._refine(f, i, w, lb))
            bot.addWidget(wand)
            rl.addLayout(bot)
            field.textChanged.connect(self._rebuild_from_segments)
            self._seg_layout.addWidget(row)
        self._seg_frame.setVisible(True)

    def _segment_fields(self) -> list:
        fields = []
        for i in range(self._seg_layout.count()):
            w = self._seg_layout.itemAt(i).widget()
            if w:
                pe = w.findChildren(QPlainTextEdit)  # prompt box (instr is a QLineEdit)
                if pe:
                    fields.append(pe[0])
        return fields

    def _rebuild_from_segments(self) -> None:
        if not self._last_result:
            return
        mode = self._last_result.get("mode", "single")
        texts = [f.toPlainText() for f in self._segment_fields()]
        if mode == "per-second":
            segs = self._last_result.get("segments", [])
            out = [
                f"(at {seg['label'].replace('s', '')} seconds: {t})"
                for seg, t in zip(segs, texts)
            ]
            self._out_prompt.setPlainText("\n".join(out))
        else:
            self._out_prompt.setPlainText("\n".join(texts))

    def _refine(self, field: QPlainTextEdit, instr: QLineEdit, wand: QPushButton, label: str) -> None:
        text = field.toPlainText().strip()
        if not text:
            self._show_status("Nothing to refine in that segment.")
            return
        wand.setEnabled(False); wand.setText("…")
        req = {
            "model_key": self._model_key(),
            "model_label": self._model.currentText(),
            "mode": self._mode.currentData() or "single",
            "label": label,
            "text": text,
            "instruction": instr.text(),
        }
        w = PromptRefineWorker(self.state, req, parent=self)
        w.finished_ok.connect(lambda t, f=field: (f.setPlainText(t), self._show_status(f"Refined {label}.")))
        w.error.connect(lambda m: self._show_status(f"Refine error: {m}"))
        w.finished.connect(lambda b=wand: (b.setEnabled(True), b.setText("🪄")))
        self._refine_workers.append(w)
        w.start()

    # ---------- send-to ----------
    def _payload(self) -> dict:
        mode = self._mode.currentData() or "single"
        fps = self._fps.value()
        frames = 0
        if mode == "per-second":
            frames = self._seconds.value() * fps
        elif mode == "per-window":
            frames = self._win_count.value() * self._window.value()
        return {
            "model_key": self._model_key(),
            "prompt": self._out_prompt.toPlainText(),
            "negative_prompt": self._out_neg.toPlainText(),
            "frames": frames,
            "fps": fps,
            "window_size": self._window.value(),
            "mode": mode,
        }

    def _send(self, target: str) -> None:
        if not self._out_prompt.toPlainText().strip():
            self._show_status("Generate a prompt first.")
            return
        self.send_to_requested.emit(target, self._payload())
        self._show_status(f"Sent to {target.replace('_', ' ')}.")

    def _send_mode2(self) -> None:
        """Send just the overall prompt into Img2Vid Mode 2 (first/last frame).
        Mode 2 is a single first->last transition, so no per-window breakdown."""
        if not self._out_prompt.toPlainText().strip():
            self._show_status("Generate a prompt first.")
            return
        self.send_to_requested.emit("generate", {
            "model_key": self._model_key(),
            "prompt": self._out_prompt.toPlainText(),
            "negative_prompt": self._out_neg.toPlainText(),
            "target_mode": 2,
            "single_prompt": True,
        })
        self._show_status("Sent to Img2Vid Mode 2 (first/last frame).")

    # ---------- clipboard ----------
    def _copy(self, text: str) -> None:
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(text or "")
        self._show_status("Copied.")

    # ---------- save / load / delete ----------
    def _read_store(self) -> list:
        try:
            return json.loads(_STORE.read_text())
        except Exception:
            return []

    def _write_store(self, data: list) -> None:
        _STORE.parent.mkdir(parents=True, exist_ok=True)
        _STORE.write_text(json.dumps(data, indent=2))

    def _refresh_saved(self) -> None:
        self._saved.clear()
        for rec in self._read_store():
            self._saved.addItem(rec.get("name", "?"), rec)

    def _save(self) -> None:
        name = self._save_name.text().strip()
        if not name:
            self._show_status("Enter a name to save.")
            return
        rec = {
            "name": name,
            "model_key": self._model_key(),
            "mode": self._mode.currentData() or "single",
            "idea": self._idea.toPlainText(),
            "prompt": self._out_prompt.toPlainText(),
            "negative_prompt": self._out_neg.toPlainText(),
            "win_count": self._win_count.value(), "fps": self._fps.value(),
            "seconds": self._seconds.value(), "window": self._window.value(),
            "segments": (self._last_result or {}).get("segments", []),
        }
        data = [r for r in self._read_store() if r.get("name") != name]
        data.append(rec)
        self._write_store(data)
        self._refresh_saved()
        self._show_status(f"Saved '{name}'.")

    def _load(self) -> None:
        rec = self._saved.currentData()
        if not rec:
            return
        key = rec.get("model_key")
        i = self._model.findData(key)
        if i >= 0:
            self._model.setCurrentIndex(i)  # triggers _on_model_changed
        mi = self._mode.findData(rec.get("mode", "single"))
        if mi >= 0:
            self._mode.setCurrentIndex(mi)
        if rec.get("win_count"): self._win_count.setValue(int(rec["win_count"]))
        if rec.get("fps"): self._fps.setValue(int(rec["fps"]))
        if rec.get("seconds"): self._seconds.setValue(int(rec["seconds"]))
        if rec.get("window"): self._window.setValue(int(rec["window"]))
        self._idea.setPlainText(rec.get("idea", ""))
        self._out_prompt.setPlainText(rec.get("prompt", ""))
        self._out_neg.setPlainText(rec.get("negative_prompt", ""))
        self._neg.setPlainText(rec.get("negative_prompt", ""))
        self._last_result = {"segments": rec.get("segments", []), "mode": rec.get("mode", "single")}
        self._render_segments(rec.get("segments", []), rec.get("mode", "single"))
        self._save_name.setText(rec.get("name", ""))
        self._show_status(f"Loaded '{rec.get('name')}'.")

    def _delete(self) -> None:
        rec = self._saved.currentData()
        if not rec:
            return
        data = [r for r in self._read_store() if r.get("name") != rec.get("name")]
        self._write_store(data)
        self._refresh_saved()
        self._show_status("Deleted.")
