"""Longshot tab -- long-form video via chunk + gapfill + assemble."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from supremediffusion.config.project_config import ProjectConfig
from supremediffusion.utils.video import probe_video

from sdqt.state import AppState
from sdqt.widgets.generation_params import GenerationParamsWidget
from sdqt.widgets.range_slider import RangeSliderWidget
from sdqt.widgets.video_player import VideoPlayerWidget
from sdqt.workers.inference import LongshotRenderWorker
from sdqt.workers.video import AcceptEditsWorker, AssembleWorker, ChunkWorker

from .base import BaseTab

logger = logging.getLogger(__name__)

_GAPFILL_SECONDS = {"1s": 1.0, "2s": 2.0, "3s": 3.0}

_PRESETS = {
    "(No Preset)": {},
    "Slow Down": {
        "subdivisions_per_sec": 5,  # dynamic: 5 per second of selected range
        "min_subdivisions": 4,
        "max_subdivisions": 20,
        "gapfill": "3s",
        "prompt": "smooth continuous motion, natural fluid movement, seamless transition between frames",
        "negative_prompt": "abrupt changes, new objects, scene change, jump cut, flickering",
    },
}


class GapfillItemWidget(QWidget):
    """Custom widget for a single gapfill entry in the list."""

    def __init__(self, index: int, gapfill: dict, parent=None) -> None:
        super().__init__(parent)
        self._gf = gapfill

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        # Index label
        idx_label = QLabel(f"<b>#{index}</b>")
        idx_label.setFixedWidth(40)
        layout.addWidget(idx_label)

        # Frame A thumbnail
        self._thumb_a = QLabel()
        self._thumb_a.setFixedSize(100, 56)
        self._thumb_a.setScaledContents(True)
        if gapfill.get("frame_a_path") and Path(gapfill["frame_a_path"]).is_file():
            self._thumb_a.setPixmap(QPixmap(gapfill["frame_a_path"]))
        layout.addWidget(self._thumb_a)

        # Arrow
        layout.addWidget(QLabel("\u2192"))

        # Frame B thumbnail
        self._thumb_b = QLabel()
        self._thumb_b.setFixedSize(100, 56)
        self._thumb_b.setScaledContents(True)
        if gapfill.get("frame_b_path") and Path(gapfill["frame_b_path"]).is_file():
            self._thumb_b.setPixmap(QPixmap(gapfill["frame_b_path"]))
        layout.addWidget(self._thumb_b)

        # Per-gapfill prompt override (blank = use the global/preset prompt).
        self._prompt_edit = QLineEdit(gapfill.get("prompt", ""))
        self._prompt_edit.setPlaceholderText("Prompt override (optional)")
        self._prompt_edit.setToolTip(
            "Leave blank to use the global/preset prompt for this gapfill."
        )
        self._prompt_edit.textChanged.connect(self._on_prompt_changed)
        layout.addWidget(self._prompt_edit, 1)

        # Status badge
        self._status_label = QLabel()
        self._update_badge(gapfill.get("status", "pending"))
        layout.addWidget(self._status_label)

    def _on_prompt_changed(self, text: str) -> None:
        self._gf["prompt"] = text

    def update_status(self, status: str) -> None:
        self._gf["status"] = status
        self._update_badge(status)

    def _update_badge(self, status: str) -> None:
        colors = {
            "pending": "#888888",
            "rendering": "#4a9eff",
            "done": "#4caf50",
            "error": "#f44336",
        }
        color = colors.get(status, "#888888")
        self._status_label.setText(f"  {status.upper()}  ")
        self._status_label.setStyleSheet(
            f"background-color: {color}; color: white; border-radius: 4px; "
            f"font-weight: bold; font-size: 13px; padding: 3px 8px;"
        )


class LongshotTab(BaseTab):
    """Long-form video creation: chunk a clip, render gapfills, assemble."""

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)

        self._chunks: list[dict] = []
        self._gapfills: list[dict] = []
        self._source_info: dict = {}
        self._worker: ChunkWorker | LongshotRenderWorker | AssembleWorker | AcceptEditsWorker | None = None
        self._preset_prompt: str = ""
        self._preset_neg_prompt: str = ""
        # Sequential gapfill render-chain state (one worker per gapfill so each
        # can carry its own prompt override).
        self._render_pipeline = None
        self._render_cfg: ProjectConfig | None = None
        self._render_ls_dir: Path | None = None
        self._render_base_prompt: str = ""
        self._render_aborting: bool = False
        self._render_pending: list[int] = []
        self._render_total: int = 0
        self._render_done_count: int = 0

        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 0, 2, 2)
        main_layout.setSpacing(1)

        # -- Top row: three equal video previews ------------------------------
        preview_splitter = QSplitter(Qt.Horizontal)

        self._source_player = VideoPlayerWidget("Source Clip")
        self._source_player.video_loaded.connect(self._on_source_dropped)
        preview_splitter.addWidget(self._source_player)

        self._gapfill_player = VideoPlayerWidget("Gapfill Render")
        preview_splitter.addWidget(self._gapfill_player)

        self._output_player = VideoPlayerWidget("Final Output")
        preview_splitter.addWidget(self._output_player)

        preview_splitter.setStretchFactor(0, 1)
        preview_splitter.setStretchFactor(1, 1)
        preview_splitter.setStretchFactor(2, 1)

        main_layout.addWidget(preview_splitter, 1)

        # -- Range slider (physical draggable brackets) -----------------------
        range_group = QGroupBox("Range")
        range_gl = QVBoxLayout(range_group)
        range_gl.setContentsMargins(4, 0, 4, 0)
        range_gl.setSpacing(0)

        self._range_slider = RangeSliderWidget()
        self._range_slider.range_changed.connect(self._on_range_changed)
        self._range_slider.begin_dragging.connect(self._on_begin_drag)
        self._range_slider.end_dragging.connect(self._on_end_drag)
        range_gl.addWidget(self._range_slider)

        range_row = QHBoxLayout()
        # Begin thumb + label
        self._begin_thumb = QLabel("IN")
        self._begin_thumb.setFixedSize(80, 45)
        self._begin_thumb.setAlignment(Qt.AlignCenter)
        self._begin_thumb.setStyleSheet("background: #1a1a1a; border: 1px solid #444; color: #555;")
        self._begin_thumb.setScaledContents(True)
        range_row.addWidget(self._begin_thumb)
        self._range_begin_label = QLabel("Begin: 0.0s")
        range_row.addWidget(self._range_begin_label)
        range_row.addStretch()
        self._range_duration_label = QLabel("Duration: 0.0s")
        range_row.addWidget(self._range_duration_label)
        range_row.addStretch()
        self._range_end_label = QLabel("End: 0.0s")
        range_row.addWidget(self._range_end_label)
        # End thumb
        self._end_thumb = QLabel("OUT")
        self._end_thumb.setFixedSize(80, 45)
        self._end_thumb.setAlignment(Qt.AlignCenter)
        self._end_thumb.setStyleSheet("background: #1a1a1a; border: 1px solid #444; color: #555;")
        self._end_thumb.setScaledContents(True)
        range_row.addWidget(self._end_thumb)
        range_gl.addLayout(range_row)

        main_layout.addWidget(range_group)

        # -- Controls row -----------------------------------------------------
        controls_row = QHBoxLayout()

        self._info_label = QLabel("No source loaded.")
        controls_row.addWidget(self._info_label, 1)

        # Chunking — split into two rows so the preset selector and the
        # subdivisions/gapfill controls don't crowd a single line.
        opts_group = QGroupBox("Chunking")
        opts_layout = QVBoxLayout(opts_group)
        opts_layout.setContentsMargins(8, 6, 8, 6)
        opts_layout.setSpacing(8)

        # Row 1: Preset selector
        opts_row1 = QHBoxLayout()
        opts_row1.setSpacing(8)
        opts_row1.addWidget(QLabel("Preset:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItems(list(_PRESETS.keys()))
        self._preset_combo.setMinimumWidth(160)
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        opts_row1.addWidget(self._preset_combo)
        opts_row1.addStretch()
        opts_layout.addLayout(opts_row1)

        # Row 2: Subdivisions + Gapfill duration
        opts_row2 = QHBoxLayout()
        opts_row2.setSpacing(8)
        opts_row2.addWidget(QLabel("Subdivisions:"))
        self._subdivisions = QSpinBox()
        self._subdivisions.setRange(2, 100)
        self._subdivisions.setValue(4)
        self._subdivisions.setMinimumWidth(90)
        opts_row2.addWidget(self._subdivisions)
        opts_row2.addWidget(QLabel("Gapfill:"))
        self._gf_group = QButtonGroup(self)
        for label in ("1s", "2s", "3s"):
            rb = QRadioButton(label)
            if label == "2s":
                rb.setChecked(True)
            self._gf_group.addButton(rb)
            opts_row2.addWidget(rb)
        opts_row2.addStretch()
        opts_layout.addLayout(opts_row2)
        controls_row.addWidget(opts_group)

        main_layout.addLayout(controls_row)

        # -- Generation params (model / sampler / LoRA / quality) -------------
        # Longshot previously had no params of its own and relied on whatever
        # the Generate / Img2Vid path last saved to ProjectConfig. Embed the
        # same shared widget the other video tabs use so the gapfill render is
        # configured explicitly. The values are collected into ProjectConfig
        # in _on_render before the worker starts.
        self._params = GenerationParamsWidget(show_lora=True)
        self._params.set_lora_refresh_callback(self._get_available_loras)
        main_layout.addWidget(self._params)

        # -- Action buttons ---------------------------------------------------
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._chunk_btn = QPushButton("Chunk Clip")
        self._chunk_btn.clicked.connect(self._on_chunk)
        btn_row.addWidget(self._chunk_btn)

        # Primary action — rendered by the global #primary rule (44px accent).
        self._render_btn = QPushButton("Begin Longshot Render")
        self._render_btn.setObjectName("primary")
        self._render_btn.clicked.connect(self._on_render)
        btn_row.addWidget(self._render_btn)

        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setVisible(False)
        self._abort_btn.clicked.connect(self._on_abort)
        btn_row.addWidget(self._abort_btn)

        self._assemble_btn = QPushButton("Assemble Final Video")
        self._assemble_btn.setVisible(False)
        self._assemble_btn.clicked.connect(self._on_assemble)
        btn_row.addWidget(self._assemble_btn)

        self._accept_btn = QPushButton("Accept Edits")
        self._accept_btn.setToolTip("Stitch longshot output back into original source video")
        self._accept_btn.setStyleSheet("font-weight: bold; color: #4caf50;")
        self._accept_btn.setVisible(False)
        self._accept_btn.clicked.connect(self._on_accept_edits)
        btn_row.addWidget(self._accept_btn)

        # Frame capture
        self._dl_frame_btn = QPushButton("Download Frame")
        self._dl_frame_btn.clicked.connect(self._on_download_frame)
        btn_row.addWidget(self._dl_frame_btn)
        self._save_frame_btn = QPushButton("Save to Project")
        self._save_frame_btn.clicked.connect(self._on_save_frame)
        btn_row.addWidget(self._save_frame_btn)

        main_layout.addLayout(btn_row)

        self._status_label_main = QLabel("")
        main_layout.addWidget(self._status_label_main)

        # -- Bottom: Gapfill Queue -------------------------------------------
        queue_label = QLabel("<b>Gapfill Queue</b>")
        main_layout.addWidget(queue_label)

        self._gapfill_list = QListWidget()
        # Was hard-capped at 120px (~2 rows). Give it real height so the queue
        # is usable past a couple of items; the list scrolls internally.
        self._gapfill_list.setMinimumHeight(280)
        self._gapfill_list.setStyleSheet("QListWidget { background: #1a1a1a; }")
        main_layout.addWidget(self._gapfill_list)

    # -- Range slider events -----------------------------------------------

    def _on_range_changed(self, begin: float, end: float) -> None:
        """Update labels when range slider moves."""
        duration = self._source_info.get("duration", 0.0)
        b_sec = begin * duration
        e_sec = end * duration
        self._range_begin_label.setText(f"Begin: {b_sec:.1f}s")
        self._range_end_label.setText(f"End: {e_sec:.1f}s")
        self._range_duration_label.setText(f"Duration: {e_sec - b_sec:.1f}s")
        # Recalculate preset subdivisions if a dynamic preset is active
        self._apply_preset_subdivisions(e_sec - b_sec)

    def _on_begin_drag(self, val: float) -> None:
        """Seek video and update thumbnail while dragging begin handle."""
        duration = self._source_info.get("duration", 0.0)
        sec = val * duration
        self._source_player._player.setPosition(int(sec * 1000))
        self._extract_thumb(sec, self._begin_thumb)

    def _on_end_drag(self, val: float) -> None:
        """Seek video and update thumbnail while dragging end handle."""
        duration = self._source_info.get("duration", 0.0)
        sec = val * duration
        self._source_player._player.setPosition(int(sec * 1000))
        self._extract_thumb(sec, self._end_thumb)

    def _extract_thumb(self, sec: float, label: QLabel) -> None:
        """Extract a frame at the given second and show it in the label."""
        source = self._source_info.get("_path")
        if not source:
            return
        try:
            from supremediffusion.utils.video import extract_single_frame
            fps = self._source_info.get("fps", 16)
            frame_num = max(0, int(sec * fps))
            img = extract_single_frame(source, frame_num)
            # Convert PIL to QPixmap
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(tmp.name)
            tmp.close()
            pixmap = QPixmap(tmp.name)
            if not pixmap.isNull():
                label.setPixmap(pixmap)
            # Clean up temp file
            Path(tmp.name).unlink(missing_ok=True)
        except Exception:
            pass

    # -- Source loading ----------------------------------------------------

    def _on_source_dropped(self, path: str) -> None:
        """Handle video dropped onto the source player."""
        self._on_source_loaded(path)

    def load_source(self, path: str) -> None:
        """Load a source video (called externally for cross-tab send)."""
        self._output_player.clear_video()
        self._source_player.load_video(path)
        self._on_source_loaded(path)

    def _on_source_loaded(self, path: str) -> None:
        if not path or not Path(path).is_file():
            self._info_label.setText("No source loaded.")
            self._source_info = {}
            return

        persisted = self.persist_file(path, str(Path("clips") / "longshot"))
        try:
            cfg = ProjectConfig.load(self.project_path)
            cfg.ls_source_path = persisted
            cfg.save(self.project_path)
        except Exception:
            pass

        try:
            info = probe_video(path)
            info["_path"] = path
            self._source_info = info
            self._info_label.setText(
                f"Duration: {info['duration']:.1f}s  |  "
                f"FPS: {info['fps']:.1f}  |  "
                f"Resolution: {info['width']}\u00d7{info['height']}"
            )
            # Reset range slider to full duration
            self._range_slider.set_range(0.0, 1.0)
            self._on_range_changed(0.0, 1.0)
            # Extract initial thumbnails
            self._extract_thumb(0.0, self._begin_thumb)
            self._extract_thumb(info["duration"], self._end_thumb)
        except Exception as exc:
            self._info_label.setText(f"Error probing video: {exc}")
            self._source_info = {}

        self._chunks = []
        self._gapfills = []
        self._gapfill_list.clear()

    # -- Chunk clip --------------------------------------------------------

    def _get_range_seconds(self) -> tuple[float, float]:
        """Convert slider 0..1 range to seconds."""
        duration = self._source_info.get("duration", 0.0)
        return (
            self._range_slider.begin() * duration,
            self._range_slider.end() * duration,
        )

    @Slot()
    def _on_chunk(self) -> None:
        if not self._source_info:
            self._set_status("No source video loaded.")
            return

        cfg = ProjectConfig.load(self.project_path)
        source_path = cfg.ls_source_path
        if not source_path or not Path(source_path).is_file():
            self._set_status("Source video not found in project.")
            return

        r_start, r_end = self._get_range_seconds()
        n_subdivs = self._subdivisions.value()

        if r_end <= r_start:
            self._set_status("End must be greater than Begin.")
            return

        ls_dir = self.project_path / "clips" / "longshot"

        self._chunk_btn.setEnabled(False)
        self._set_status("Chunking...")

        worker = ChunkWorker(
            source_path, ls_dir, r_start, r_end, n_subdivs, parent=self,
        )
        worker.progress.connect(lambda f, d: self._set_status(d))
        worker.finished_ok.connect(self._on_chunk_done)
        worker.error.connect(lambda e: self._on_chunk_error(e))
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_chunk_done(self, result: dict) -> None:
        self._chunks = result["chunks"]
        self._gapfills = result["gapfills"]
        # Each gapfill may carry its own prompt override (blank = global).
        for gf in self._gapfills:
            gf.setdefault("prompt", "")
        self._chunk_btn.setEnabled(True)
        self._assemble_btn.setVisible(False)
        self._accept_btn.setVisible(False)

        # Save settings
        r_start, r_end = self._get_range_seconds()
        gf_label = self._gf_group.checkedButton().text() if self._gf_group.checkedButton() else "2s"
        try:
            cfg = ProjectConfig.load(self.project_path)
            cfg.ls_range_start = r_start
            cfg.ls_range_end = r_end
            cfg.ls_subdivisions = self._subdivisions.value()
            cfg.ls_gapfill_duration = _GAPFILL_SECONDS.get(gf_label, 2.0)
            cfg.save(self.project_path)
        except Exception:
            pass

        self._rebuild_gapfill_list()
        n = len(self._gapfills)
        chunk_len = (r_end - r_start) / self._subdivisions.value()
        self._set_status(
            f"Chunked into {self._subdivisions.value()} segments ({chunk_len:.1f}s each). "
            f"{n} gapfill{'s' if n != 1 else ''} queued."
        )

    def _on_chunk_error(self, msg: str) -> None:
        self._chunk_btn.setEnabled(True)
        self._set_status(f"Error: {msg}")

    # -- Render gapfills ---------------------------------------------------

    @Slot()
    def _on_render(self) -> None:
        if not self._gapfills:
            self._set_status("No gapfills to render. Chunk a clip first.")
            return

        from sdqt.models.manager import check_and_prompt_download
        from sdqt.widgets.generation_params import MODEL_TYPE_TO_FEATURE
        # Collect the embedded params widget into ProjectConfig so the gapfill
        # render uses the model / sampler / LoRA / quality chosen here rather
        # than whatever another tab last saved.
        cfg = ProjectConfig.load(self.project_path)
        self._params.collect_to_config(cfg)
        cfg.save(self.project_path)
        model_key = self._params.model_type.currentData() or cfg.model_type or "i2v_2_2"
        feature = MODEL_TYPE_TO_FEATURE.get(model_key, "video_gen")
        if not check_and_prompt_download(
            feature, self.state.model_registry, self,
            on_progress=lambda f, d: self._set_status(d),
        ):
            return

        from sdqt.widgets.generation_params import get_video_backend
        backend = get_video_backend(model_key)
        pipeline = self.state.pipeline
        if pipeline is None or self.state.active_video_backend != backend:
            self._set_status(f"Loading {backend.upper()} video pipelines...")
            try:
                self.state.load_video_pipeline_for_backend(backend)
                pipeline = self.state.pipeline
            except Exception as exc:
                self._set_status(f"Failed to load: {exc}")
                return
        if pipeline is None:
            self._set_status("Pipeline not loaded. Check Settings.")
            return

        # Apply preset prompt overrides (temporary, not saved to project)
        if self._preset_prompt:
            cfg.prompt = self._preset_prompt
        if self._preset_neg_prompt:
            cfg.negative_prompt = self._preset_neg_prompt

        fps = self._source_info.get("fps", cfg.fps or 16)
        gf_label = self._gf_group.checkedButton().text() if self._gf_group.checkedButton() else "2s"
        gf_duration = _GAPFILL_SECONDS.get(gf_label, 2.0)
        cfg.video_length = int(gf_duration * fps)
        cfg.fps = int(fps)

        ls_dir = self.project_path / "clips" / "longshot"
        ls_dir.mkdir(parents=True, exist_ok=True)

        # The global prompt (preset override already applied above) every
        # gapfill falls back to when its per-gap override is blank.
        self._render_pipeline = pipeline
        self._render_cfg = cfg
        self._render_ls_dir = ls_dir
        self._render_base_prompt = cfg.prompt
        self._render_aborting = False
        self._render_pending = [
            i for i, gf in enumerate(self._gapfills) if gf["status"] == "pending"
        ]
        self._render_total = len(self._render_pending)
        self._render_done_count = 0
        if self._render_total == 0:
            self._set_status("No pending gapfills to render.")
            return

        self._render_btn.setVisible(False)
        self._abort_btn.setVisible(True)
        self._assemble_btn.setVisible(False)

        # Render one gapfill at a time so each can carry its own prompt. The
        # batch worker is reused with a single-element list per launch; the gf
        # dicts are shared references, so status/render_path mutations land
        # directly on self._gapfills.
        self._render_next_gapfill()

    def _render_next_gapfill(self) -> None:
        """Launch the next pending gapfill with its own (or global) prompt."""
        if self._render_aborting or not self._render_pending:
            self._finish_render_chain()
            return

        idx = self._render_pending.pop(0)
        gf = self._gapfills[idx]

        # Per-gap prompt override; blank falls back to the global/preset prompt.
        cfg = self._render_cfg
        cfg.prompt = (gf.get("prompt") or "").strip() or self._render_base_prompt

        step = self._render_done_count + 1
        self._set_status(f"Rendering gapfill {step}/{self._render_total}...")

        worker = LongshotRenderWorker(
            self._render_pipeline, self.project_name, cfg, [gf],
            self._render_ls_dir, parent=self,
        )
        worker.progress.connect(self._on_render_progress)
        worker.finished_ok.connect(self._on_one_gapfill_done)
        worker.error.connect(self._on_render_error)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_render_progress(self, frac: float, desc: str) -> None:
        self._set_status(desc)
        self._rebuild_gapfill_list()
        # Show last completed gapfill in the middle preview
        for gf in reversed(self._gapfills):
            vid = gf.get("render_path")
            if gf.get("status") == "done" and vid and Path(vid).is_file():
                self._gapfill_player.load_video(vid)
                break

    def _on_one_gapfill_done(self, _gapfills: list[dict]) -> None:
        # gf dicts are shared references — self._gapfills is already updated.
        self._render_done_count += 1
        self._rebuild_gapfill_list()
        if self._render_aborting:
            self._finish_render_chain()
            return
        self._render_next_gapfill()

    def _finish_render_chain(self) -> None:
        self._worker = None
        self._render_btn.setVisible(True)
        self._abort_btn.setVisible(False)
        self._rebuild_gapfill_list()

        n_done = sum(1 for gf in self._gapfills if gf["status"] == "done")
        n_err = sum(1 for gf in self._gapfills if gf["status"] == "error")
        all_done = n_done == len(self._gapfills)
        self._assemble_btn.setVisible(all_done)

        if self._render_aborting:
            self._set_status(
                f"Render aborted. {n_done} done, {len(self._gapfills) - n_done} remaining."
            )
            return

        msg = f"Render complete. {n_done} done"
        if n_err:
            msg += f", {n_err} failed"
        self._set_status(msg + ".")

    def _on_render_error(self, msg: str) -> None:
        # BaseWorker emits error("Aborted.") on cancellation — route that
        # through the clean abort summary instead of an error banner.
        if msg == "Aborted." or self._render_aborting:
            self._render_aborting = True
            self._worker = None
            self._finish_render_chain()
            return
        self._render_aborting = True
        self._worker = None
        self._render_btn.setVisible(True)
        self._abort_btn.setVisible(False)
        self._rebuild_gapfill_list()
        self._set_status(f"Render error: {msg}")

    @Slot()
    def _on_abort(self) -> None:
        # Stop the chain after the in-flight gapfill returns.
        self._render_aborting = True
        if self._worker is not None:
            self._worker.abort()

    # -- Assemble ----------------------------------------------------------

    @Slot()
    def _on_assemble(self) -> None:
        ls_dir = self.project_path / "clips" / "longshot"
        self._assemble_btn.setEnabled(False)
        self._set_status("Assembling...")

        worker = AssembleWorker(self._chunks, self._gapfills, ls_dir, parent=self)
        worker.progress.connect(lambda f, d: self._set_status(d))
        worker.finished_ok.connect(self._on_assemble_done)
        worker.error.connect(lambda e: self._on_assemble_error(e))
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_assemble_done(self, final_path: str) -> None:
        self._assemble_btn.setEnabled(True)
        self._accept_btn.setVisible(True)
        self._output_player.load_video(final_path)
        self._set_status("Assembly complete! Use 'Accept Edits' to stitch back into source.")

        try:
            cfg = ProjectConfig.load(self.project_path)
            cfg.ls_final_video_path = final_path
            cfg.save(self.project_path)
        except Exception:
            pass

    def _on_assemble_error(self, msg: str) -> None:
        self._assemble_btn.setEnabled(True)
        self._set_status(f"Assembly error: {msg}")

    # -- Frame capture -----------------------------------------------------

    @Slot()
    def _on_download_frame(self) -> None:
        # Try output player first, fall back to source
        player = self._output_player if self._output_player.video_path else self._source_player
        path = player.capture_frame()
        if path:
            self.download_frame(path)

    @Slot()
    def _on_save_frame(self) -> None:
        player = self._output_player if self._output_player.video_path else self._source_player
        path = player.capture_frame()
        if path:
            saved = self.save_frame_to_project(path)
            if saved:
                self._set_status("Saved frame to project.")

    # -- Project change ----------------------------------------------------

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        # Clear all media
        self._source_player.clear_video()
        self._gapfill_player.clear_video()
        self._output_player.clear_video()
        self._source_info = {}

        try:
            cfg = ProjectConfig.load(self.project_path)
        except Exception:
            return

        if cfg.ls_source_path and Path(cfg.ls_source_path).is_file():
            self._source_player.load_video(cfg.ls_source_path)
            self._on_source_loaded(cfg.ls_source_path)

        # Restore range from saved seconds
        duration = self._source_info.get("duration", 0.0)
        if duration > 0:
            b = cfg.ls_range_start / duration
            e = cfg.ls_range_end / duration if cfg.ls_range_end > 0 else 1.0
            self._range_slider.set_range(b, e)
            self._on_range_changed(b, e)

        self._subdivisions.setValue(cfg.ls_subdivisions)

        # Restore gapfill radio
        for btn in self._gf_group.buttons():
            if _GAPFILL_SECONDS.get(btn.text()) == cfg.ls_gapfill_duration:
                btn.setChecked(True)
                break

        if cfg.ls_final_video_path and Path(cfg.ls_final_video_path).is_file():
            self._output_player.load_video(cfg.ls_final_video_path)

        # Restore the embedded generation params + LoRA list
        self._params.restore_from_config(cfg)
        try:
            self._params.populate_loras(self._get_available_loras(), cfg.activated_loras)
        except Exception:
            pass

    # -- Presets -----------------------------------------------------------

    @Slot(str)
    def _on_preset_changed(self, name: str) -> None:
        preset = _PRESETS.get(name, {})
        if not preset:
            self._preset_prompt = ""
            self._preset_neg_prompt = ""
            return

        self._preset_prompt = preset.get("prompt", "")
        self._preset_neg_prompt = preset.get("negative_prompt", "")

        # Set gapfill radio
        gf = preset.get("gapfill", "")
        for btn in self._gf_group.buttons():
            if btn.text() == gf:
                btn.setChecked(True)
                break

        # Compute dynamic subdivisions from current range
        r_start, r_end = self._get_range_seconds()
        self._apply_preset_subdivisions(r_end - r_start)

    def _apply_preset_subdivisions(self, range_duration: float) -> None:
        """Recalculate subdivisions if a dynamic preset is active."""
        name = self._preset_combo.currentText()
        preset = _PRESETS.get(name, {})
        sps = preset.get("subdivisions_per_sec")
        if sps is None or range_duration <= 0:
            return
        mn = preset.get("min_subdivisions", 4)
        mx = preset.get("max_subdivisions", 20)
        subs = max(mn, min(mx, int(range_duration * sps)))
        self._subdivisions.setValue(subs)

    # -- Accept Edits ------------------------------------------------------

    @Slot()
    def _on_accept_edits(self) -> None:
        """Stitch longshot output back into original source, replacing selected range."""
        source = self._source_info.get("_path")
        if not source or not Path(source).is_file():
            self._set_status("Original source video not found.")
            return

        cfg = ProjectConfig.load(self.project_path)
        longshot = cfg.ls_final_video_path
        if not longshot or not Path(longshot).is_file():
            self._set_status("No assembled longshot output found.")
            return

        duration = self._source_info.get("duration", 0.0)
        ls_dir = self.project_path / "clips" / "longshot"

        self._accept_btn.setEnabled(False)
        self._set_status("Stitching edits into original...")

        worker = AcceptEditsWorker(
            source, longshot,
            cfg.ls_range_start, cfg.ls_range_end, duration,
            ls_dir, parent=self,
        )
        worker.progress.connect(lambda f, d: self._set_status(d))
        worker.finished_ok.connect(self._on_accept_done)
        worker.error.connect(self._on_accept_error)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_accept_done(self, final_path: str) -> None:
        self._accept_btn.setEnabled(True)
        self._output_player.load_video(final_path)
        self._set_status(f"Edits accepted! Final video: {Path(final_path).name}")

        try:
            cfg = ProjectConfig.load(self.project_path)
            cfg.ls_final_video_path = final_path
            cfg.save(self.project_path)
        except Exception:
            pass

    def _on_accept_error(self, msg: str) -> None:
        self._accept_btn.setEnabled(True)
        self._set_status(f"Accept edits error: {msg}")

    # -- Helpers -----------------------------------------------------------

    def _get_available_loras(self) -> list[str]:
        lora_mgr = self.state.lora_manager
        if lora_mgr is None:
            return []
        return lora_mgr.list_available()

    def _set_status(self, msg: str) -> None:
        self._status_label_main.setText(msg)

    def _rebuild_gapfill_list(self) -> None:
        self._gapfill_list.clear()
        for i, gf in enumerate(self._gapfills):
            item = QListWidgetItem(self._gapfill_list)
            widget = GapfillItemWidget(i, gf)
            item.setSizeHint(widget.sizeHint())
            self._gapfill_list.setItemWidget(item, widget)
