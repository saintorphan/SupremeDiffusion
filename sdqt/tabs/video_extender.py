"""Video Extender tab -- extend a video clip (continuation)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from supremediffusion.config.project_config import ProjectConfig

from sdqt.state import AppState
from sdqt.widgets.collapsible_section import CollapsibleSection
from sdqt.widgets.generation_params import GenerationParamsWidget
from sdqt.widgets.image_drop import ImageDropWidget
from sdqt.widgets.video_drop import VideoDropWidget
from sdqt.widgets.video_player import VideoPlayerWidget
from sdqt.workers.inference import PostProcessWorker, VideoExtendWorker

from .base import BaseTab

logger = logging.getLogger(__name__)

# The shared GenerationParamsWidget defaults denoising_strength to 1.0, which
# fully re-noises the latent and disables source-as-guidance continuity — fine
# for fresh generation but wrong for an *extension*, where the source frames
# are meant to anchor the continuation. When a project has never customised the
# slider (still the global 1.0 default), VE substitutes this continuity-friendly
# value so extensions stay visually coherent with the source.
_VE_DEFAULT_DENOISE = 0.75


class VideoExtenderTab(BaseTab):
    """Extend a video clip by generating continuation frames."""

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(state, parent)
        self._worker: VideoExtendWorker | None = None
        self._last_frame_path: str | None = None  # lossless PNG for continue
        # When the user clicks "From Source Frame" the captured PIL image is
        # stashed here. At generate-time it is handed to VideoExtendWorker as
        # guidance_image, which builds the tensor in-memory and skips the
        # ffmpeg encode/decode roundtrip that would otherwise tint colors.
        self._static_guidance_pil = None  # type: ignore[assignment]
        # The ProjectConfig actually used for the last generation. Reused by
        # Save Clip so post-processing runs with the params the clip was made
        # with, not whatever the UI happens to show after the run finished.
        self._gen_cfg: ProjectConfig | None = None
        # Auto-chain state: number of remaining automatic continuations and the
        # 1-based index of the link currently running (for progress labels).
        self._chain_remaining: int = 0
        self._chain_total: int = 1
        self._chain_index: int = 0
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 2)
        layout.setSpacing(1)

        # ── Top: Two equal video players side by side ────────────────
        video_splitter = QSplitter(Qt.Horizontal)
        self._source_player = VideoPlayerWidget("Source Video")
        self._source_player.video_loaded.connect(self._on_source_dropped)
        self._source_player.video_cleared.connect(self._on_source_cleared)
        video_splitter.addWidget(self._source_player)
        self._preview = VideoPlayerWidget("Generated Preview")
        video_splitter.addWidget(self._preview)
        video_splitter.setStretchFactor(0, 1)
        video_splitter.setStretchFactor(1, 1)
        # ── Bottom: Controls in a scrollable area ────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        controls = QWidget()
        ctrl_layout = QVBoxLayout(controls)
        ctrl_layout.setContentsMargins(8, 6, 8, 6)
        ctrl_layout.setSpacing(8)

        # Action buttons + status. Primary action (Generate) on its own row;
        # the secondary controls (Chain count, Paint Mask) move to a second row
        # so the action row isn't overpacked.
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        # Primary action — styled by the global #primary rule (44px accent).
        self._gen_btn = QPushButton("Generate")
        self._gen_btn.setObjectName("primary")
        self._gen_btn.clicked.connect(self._on_generate)
        btn_row.addWidget(self._gen_btn)
        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setVisible(False)
        self._abort_btn.clicked.connect(self._on_abort)
        btn_row.addWidget(self._abort_btn)
        self._continue_btn = QPushButton("Continue Clip")
        self._continue_btn.setEnabled(False)
        self._continue_btn.clicked.connect(self._on_continue)
        btn_row.addWidget(self._continue_btn)
        btn_row.addStretch()
        ctrl_layout.addLayout(btn_row)

        # Secondary action row: auto-chain count + mask editor.
        opts_row = QHBoxLayout()
        opts_row.setSpacing(8)
        # Auto-chain: after an extension finishes, automatically re-run extend
        # from the new last frame this many times. 1 = current behavior.
        opts_row.addWidget(QLabel("Chain:"))
        self._chain_count = QSpinBox()
        self._chain_count.setRange(1, 50)
        self._chain_count.setValue(1)
        self._chain_count.setMinimumWidth(90)
        self._chain_count.setToolTip(
            "Number of continuations to run automatically. 1 = single extension."
        )
        opts_row.addWidget(self._chain_count)
        # Open the popup video-mask editor for LTX inpaint LoRAs.
        self._mask_btn = QPushButton("🎨 Paint Mask")
        self._mask_btn.setToolTip(
            "Paint a static inpaint mask over the source video — "
            "used by the LTX inpaint LoRAs."
        )
        self._mask_btn.clicked.connect(self._on_paint_mask)
        opts_row.addWidget(self._mask_btn)
        opts_row.addStretch()
        ctrl_layout.addLayout(opts_row)

        self._status = QLabel("")
        ctrl_layout.addWidget(self._status)

        # Prompt
        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout(prompt_group)
        self._prompt = QPlainTextEdit()
        # Was capped at 80px — too short for multi-line continuation prompts.
        self._prompt.setMinimumHeight(120)
        self._prompt.setMaximumHeight(150)
        self._prompt.setPlaceholderText("Describe the continuation...")
        prompt_layout.addWidget(self._prompt)
        self._neg_prompt = QLineEdit()
        self._neg_prompt.setPlaceholderText("Negative prompt (optional)")
        prompt_layout.addWidget(self._neg_prompt)
        ctrl_layout.addWidget(prompt_group)

        # ── Guidance Video & Final Frame (shared collapsible) ────────
        # Expanded by default so these features are discoverable instead of
        # hidden behind a collapsed header.
        self._gf_section = gf_section = CollapsibleSection(
            "Guidance Video / Final Frame", collapsed=False
        )

        side_by_side = QHBoxLayout()

        # Left: Guidance Video
        guid_col = QVBoxLayout()
        self._use_guidance = QCheckBox("Enable Guidance Video")
        self._use_guidance.toggled.connect(self._on_guidance_toggled)
        guid_col.addWidget(self._use_guidance)

        self._guidance_container = QWidget()
        gc_layout = QVBoxLayout(self._guidance_container)
        gc_layout.setContentsMargins(0, 0, 0, 0)
        gc_layout.setSpacing(4)
        self._guidance_video = VideoDropWidget("Guidance Video", thumb_height=180)
        # If the user drops a real video file, invalidate any stashed PIL so
        # the file path wins.
        self._guidance_video.file_loaded.connect(self._on_guidance_file_loaded)
        self._guidance_video.file_cleared.connect(self._on_guidance_file_cleared)
        gc_layout.addWidget(self._guidance_video)
        gc_btn_row = QHBoxLayout()
        gv_static = QPushButton("From Source Frame")
        gv_static.setToolTip("Generate static guidance video from source video frame")
        gv_static.clicked.connect(self._on_generate_static_guidance)
        gc_btn_row.addWidget(gv_static)
        gc_btn_row.addWidget(QLabel("Frame:"))
        self._guidance_frame = QComboBox()
        self._guidance_frame.addItems(["first", "last"])
        gc_btn_row.addWidget(self._guidance_frame)
        gc_btn_row.addStretch()
        gc_layout.addLayout(gc_btn_row)
        self._guidance_container.setVisible(False)
        guid_col.addWidget(self._guidance_container)
        guid_col.addStretch()
        side_by_side.addLayout(guid_col, 1)

        # Right: Final Frame
        ff_col = QVBoxLayout()
        self._use_final_frame = QCheckBox("Enable Final Frame")
        self._use_final_frame.setToolTip("Provide a destination frame the extension will aim toward")
        self._use_final_frame.toggled.connect(self._on_final_frame_toggled)
        ff_col.addWidget(self._use_final_frame)

        self._final_frame_container = QWidget()
        ffc_layout = QVBoxLayout(self._final_frame_container)
        ffc_layout.setContentsMargins(0, 0, 0, 0)
        self._final_frame = ImageDropWidget("Destination Frame", thumb_height=180)
        ffc_layout.addWidget(self._final_frame)
        self._final_frame_container.setVisible(False)
        ff_col.addWidget(self._final_frame_container)
        ff_col.addStretch()
        side_by_side.addLayout(ff_col, 1)

        gf_section.add_layout(side_by_side)
        ctrl_layout.addWidget(gf_section)

        # Shared generation params
        self._params = GenerationParamsWidget(show_lora=True)
        self._params.set_lora_refresh_callback(self._get_available_loras)
        ctrl_layout.addWidget(self._params)

        # Save clip + frame capture actions
        actions_row = QHBoxLayout()

        save_clip_group = QGroupBox("Save clip to project")
        sc_layout = QHBoxLayout(save_clip_group)
        self._clip_name = QLineEdit()
        self._clip_name.setPlaceholderText("Clip name...")
        sc_layout.addWidget(self._clip_name, 1)
        self._save_clip_btn = QPushButton("Save Clip")
        self._save_clip_btn.setEnabled(False)
        self._save_clip_btn.clicked.connect(self._on_save_clip)
        sc_layout.addWidget(self._save_clip_btn)
        actions_row.addWidget(save_clip_group)

        capture_group = QGroupBox("Capture frame at playhead")
        capture_layout = QHBoxLayout(capture_group)
        self._dl_frame_btn = QPushButton("Download Frame")
        self._dl_frame_btn.setEnabled(False)
        self._dl_frame_btn.clicked.connect(self._on_download_frame)
        capture_layout.addWidget(self._dl_frame_btn)
        self._save_frame_btn = QPushButton("Save to Project")
        self._save_frame_btn.setEnabled(False)
        self._save_frame_btn.clicked.connect(self._on_save_frame)
        capture_layout.addWidget(self._save_frame_btn)
        actions_row.addWidget(capture_group)

        ctrl_layout.addLayout(actions_row)

        ctrl_layout.addStretch()
        scroll.setWidget(controls)

        # Vertical splitter: video on top, controls on bottom — drag to resize
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(5)
        main_splitter.setStyleSheet(
            "QSplitter::handle { background: #3a3a3a; }"
            "QSplitter::handle:hover { background: #888; }"
        )
        main_splitter.addWidget(video_splitter)
        main_splitter.addWidget(scroll)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)
        layout.addWidget(main_splitter, 1)

    # -- Checkbox toggles --------------------------------------------------

    @Slot(bool)
    def _on_guidance_toggled(self, checked: bool) -> None:
        self._guidance_container.setVisible(checked)

    @Slot(str)
    def _on_guidance_file_loaded(self, path: str) -> None:
        # When the loaded path is the thumbnail MP4 we just wrote from a
        # captured PIL frame, keep the stashed image. Otherwise, the user
        # dropped/picked a different video — fall back to file-based path.
        guidance_dir = self.project_path / "guidance"
        if Path(path).resolve() != (guidance_dir / "ve_static_guidance.mp4").resolve():
            self._static_guidance_pil = None

    @Slot()
    def _on_guidance_file_cleared(self) -> None:
        self._static_guidance_pil = None

    @Slot(bool)
    def _on_final_frame_toggled(self, checked: bool) -> None:
        self._final_frame_container.setVisible(checked)

    # -- Actions -----------------------------------------------------------

    def _on_source_dropped(self, path: str) -> None:
        """Handle video dropped onto the source player."""
        self._last_frame_path = None  # fresh source — extract from video
        try:
            cfg = ProjectConfig.load(self.project_path)
            cfg.ve_video_path = path
            cfg.save(self.project_path)
        except Exception:
            pass

    def _on_source_cleared(self) -> None:
        """Handle source video cleared via X button."""
        self._last_frame_path = None
        self._continue_btn.setEnabled(False)
        try:
            cfg = ProjectConfig.load(self.project_path)
            cfg.ve_video_path = ""
            cfg.save(self.project_path)
        except Exception:
            pass

    @Slot()
    def _on_generate_static_guidance(self) -> None:
        """Capture a still from the source player and prep it as static guidance.

        The captured PIL image is stashed on ``self._static_guidance_pil`` and
        passed straight into the inference worker at generate-time, so the
        guidance tensor is built in-memory with zero ffmpeg encode/decode.
        A PNG-codec MP4 is also written purely as a thumbnail for the drop
        widget — PNG codec is lossless RGB, so even loading that thumbnail
        back later wouldn't introduce a color shift.
        """
        frame_path = self._source_player.capture_frame()
        if not frame_path:
            self._show_status("No source video loaded for static guidance.")
            return
        try:
            from PIL import Image
            from supremediffusion.utils.video import generate_static_guidance_video

            img = Image.open(frame_path).convert("RGB")
            self._static_guidance_pil = img

            out_dir = self.project_path / "guidance"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Persist the captured frame as a lossless PNG sidecar so the
            # next session can re-stash it without going through the player.
            png_path = out_dir / "ve_static_guidance.png"
            img.save(png_path)

            # Thumbnail-only MP4 (PNG codec, no YUV conversion). This is just
            # for the VideoDropWidget preview; the actual guidance tensor is
            # built from `self._static_guidance_pil` at generate-time.
            thumb_path = generate_static_guidance_video(
                str(png_path), num_frames=81, fps=24,
                width=img.width, height=img.height,
            )
            mp4_dest = out_dir / "ve_static_guidance.mp4"
            shutil.move(thumb_path, mp4_dest)
            self._guidance_video.load_file(str(mp4_dest))
            self._show_status(
                "Static guidance ready (in-memory tensor — no codec roundtrip)"
            )
        except Exception as e:
            logger.warning("Static guidance error", exc_info=True)
            self._show_status(f"Static guidance error: {e}")

    def load_source(self, path: str) -> None:
        """Load a source video (called externally for cross-tab send)."""
        self._preview.clear_video()
        self._source_player.load_video(path)

    def apply_prompt_payload(self, payload: dict) -> None:
        """Receive a prompt + params from the Txt2Prompt tab: set the prompt and
        (via GenerationParamsWidget) the model, frames, window size, and fps."""
        self._prompt.setPlainText(payload.get("prompt", ""))
        neg = payload.get("negative_prompt")
        if neg is not None:
            self._neg_prompt.setText(neg)
        p = self._params
        key = payload.get("model_key")
        if key:
            idx = p.model_type.findData(key)
            if idx >= 0:
                p.model_type.setCurrentIndex(idx)
        fps = payload.get("fps")
        if fps:
            p.fps.setValue(int(fps))
        frames = payload.get("frames")
        if frames:
            ci = p.duration_combo.findText("Custom")
            if ci >= 0:
                p.duration_combo.setCurrentIndex(ci)
            p._duration_custom_frames.setValue(int(frames))
        win = payload.get("window_size")
        if win and hasattr(p, "sliding_window_size"):
            p.sliding_window_size.setValue(int(win))
        self._show_status("Prompt applied from Txt2Prompt.")

    @Slot()
    def _on_generate(self, _checked: bool = False, *, chained: bool = False) -> None:
        # Initialise the auto-chain counters on a fresh (user-clicked) run.
        # Auto-chained re-runs pass chained=True so they don't reset progress.
        if not chained:
            self._chain_total = self._chain_count.value()
            self._chain_remaining = self._chain_total - 1
            self._chain_index = 1

        from sdqt.models.manager import check_and_prompt_download
        from sdqt.widgets.generation_params import MODEL_TYPE_TO_FEATURE
        model_key = self._params.model_type.currentData() or "i2v_2_2"
        feature = MODEL_TYPE_TO_FEATURE.get(model_key, "video_gen")
        if not check_and_prompt_download(
            feature, self.state.model_registry, self,
            on_progress=lambda f, d: self._show_status(d),
        ):
            return

        from sdqt.widgets.generation_params import get_video_backend
        backend = get_video_backend(model_key)
        pipeline = self.state.pipeline
        if pipeline is None or self.state.active_video_backend != backend:
            self._show_status(f"Loading {backend.upper()} video pipelines...")
            try:
                self.state.load_video_pipeline_for_backend(backend)
                pipeline = self.state.pipeline
            except Exception as exc:
                self._show_status(f"Failed to load video pipelines: {exc}")
                return
        if pipeline is None:
            self._show_status("Pipeline not loaded. Check Settings.")
            return

        vid = self._source_player.video_path
        if not vid:
            self._show_status("No source video selected.")
            return

        cfg = ProjectConfig.load(self.project_path)
        cfg.prompt = self._prompt.toPlainText()
        cfg.negative_prompt = self._neg_prompt.text()
        cfg.ve_video_path = vid

        # Guidance video (only if enabled)
        if self._use_guidance.isChecked():
            cfg.ve_guidance_video_path = self._guidance_video.file_path or ""
            cfg.ve_guidance_video_frame = self._guidance_frame.currentText()
        else:
            cfg.ve_guidance_video_path = ""

        # Final frame path
        final_frame = None
        if self._use_final_frame.isChecked():
            final_frame = self._final_frame.image_path
            cfg.ve_final_frame_path = final_frame or ""
        else:
            cfg.ve_final_frame_path = ""

        self._params.collect_to_config(cfg)
        cfg.save(self.project_path)
        # Remember the exact config this run used so Save Clip can post-process
        # with the same params even if the UI is edited afterwards.
        self._gen_cfg = cfg

        self._gen_btn.setVisible(False)
        self._abort_btn.setVisible(True)
        if self._chain_total > 1:
            self._show_status(
                f"Starting extension (link {self._chain_index}/{self._chain_total})..."
            )
        else:
            self._show_status("Starting extension...")

        guidance_path = self._guidance_video.file_path if self._use_guidance.isChecked() else None
        # When the static-guidance flow stashed a PIL image, hand it to the
        # worker directly so the tensor is built in-memory (no codec roundtrip).
        guidance_image = (
            self._static_guidance_pil if self._use_guidance.isChecked() else None
        )

        worker = VideoExtendWorker(
            pipeline=pipeline,
            project_name=self.project_name,
            project_config=cfg,
            video_path=vid,
            final_frame_path=final_frame,
            guidance_video_path=guidance_path,
            guidance_image=guidance_image,
            start_image_path=self._last_frame_path,
            parent=self,
        )
        worker.progress.connect(self._on_gen_progress)
        worker.finished_ok.connect(self._on_gen_done)
        worker.error.connect(self._on_gen_error)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _on_gen_progress(self, _frac: float, desc: str) -> None:
        if self._chain_total > 1:
            self._show_status(f"[link {self._chain_index}/{self._chain_total}] {desc}")
        else:
            self._show_status(desc)

    def _on_gen_done(self, result: dict) -> None:
        self._gen_btn.setVisible(True)
        self._abort_btn.setVisible(False)

        result_path = result["video_path"]
        self._last_frame_path = result.get("last_frame_path")

        # Stitch source + generated extension using stream-copy concat
        # to avoid re-encoding the accumulated video (which causes
        # generational quality and colour loss over multiple continuations).
        # The backend already colour-corrects extension frames against the
        # source's last frame, so a crossfade blend is unnecessary.
        source_vid = self._source_player.video_path
        stitched_path = result_path
        if source_vid and Path(source_vid).is_file():
            try:
                from supremediffusion.utils.video import concat_videos
                import tempfile
                stitched = tempfile.NamedTemporaryFile(
                    suffix=".mp4", prefix="ve_stitched_", delete=False,
                )
                stitched.close()
                concat_videos(source_vid, result_path, stitched.name)
                stitched_path = stitched.name
                logger.info("Stitched (stream copy) source + extension: %s", stitched_path)
            except Exception as exc:
                logger.warning("Stitch failed, showing extension only: %s", exc)

        self._preview.load_video(stitched_path, auto_play=True)
        self._continue_btn.setEnabled(True)
        self._save_clip_btn.setEnabled(True)
        self._dl_frame_btn.setEnabled(True)
        self._save_frame_btn.setEnabled(True)
        if self._chain_remaining > 0:
            self._show_status(
                f"Link {self._chain_index}/{self._chain_total} complete."
            )
        else:
            self._show_status("Extension complete.")

        try:
            cfg = ProjectConfig.load(self.project_path)
            cfg.ve_preview_path = stitched_path
            cfg.save(self.project_path)
        except Exception:
            pass

        # Embed generation metadata
        try:
            from supremediffusion.utils.video import write_video_metadata
            cfg = ProjectConfig.load(self.project_path)
            meta = {
                "prompt": cfg.prompt or "",
                "negative_prompt": cfg.negative_prompt or "",
                "model_type": cfg.model_type or "",
                "resolution": cfg.resolution or "",
                "fps": cfg.fps,
                "video_length": cfg.video_length,
                "steps": cfg.num_inference_steps,
                "guidance_scale": cfg.guidance_scale,
                "seed": cfg.seed,
            }
            write_video_metadata(stitched_path, meta)
        except Exception:
            pass

        # Auto-chain: feed the stitched result back in as the new source and
        # re-run the extension from the lossless last frame. Reuses the same
        # continuity path as the manual "Continue Clip" button.
        if self._chain_remaining > 0:
            self._chain_remaining -= 1
            self._chain_index += 1
            saved_frame = self._last_frame_path
            self._source_player.blockSignals(True)
            self._source_player.load_video(stitched_path)
            self._source_player.blockSignals(False)
            # Preserve the lossless PNG last frame so the next link anchors on
            # it instead of re-extracting from the compressed stitched video.
            self._last_frame_path = saved_frame
            self._on_generate(chained=True)

    def _on_gen_error(self, msg: str) -> None:
        # Abort or failure stops the chain.
        self._chain_remaining = 0
        self._chain_total = 1
        self._gen_btn.setVisible(True)
        self._abort_btn.setVisible(False)
        self._show_status(f"Error: {msg}")

    @Slot()
    def _on_abort(self) -> None:
        # Stop any pending auto-chain links, then cancel the in-flight worker.
        self._chain_remaining = 0
        self._chain_total = 1
        if self._worker:
            self._worker.abort()

    @Slot()
    def _on_paint_mask(self) -> None:
        """Open the static-mask paint dialog over the loaded source video."""
        from sdqt.widgets.video_mask_dialog import VideoMaskDialog
        vid = self._source_player.video_path
        if not vid or not Path(vid).is_file():
            self._show_status("Load a source video first.")
            return
        guidance_dir = Path(self.project_path) / "guidance"
        guidance_dir.mkdir(parents=True, exist_ok=True)
        save_path = guidance_dir / "ltx_inpaint_mask.png"
        dlg = VideoMaskDialog(vid, save_path, frame_index=0, parent=self)
        if dlg.exec():
            try:
                cfg = ProjectConfig.load(self.project_path)
                cfg.ltx_inpaint_mask_path = str(dlg.mask_path or "")
                cfg.save(self.project_path)
            except Exception:
                pass
            self._show_status(f"Mask saved: {dlg.mask_path}")

    @Slot()
    def _on_continue(self) -> None:
        """Load the preview result as the new source for another extension.

        Preserves _last_frame_path so the next generation uses the lossless
        PNG instead of extracting the start frame from the compressed video.
        """
        pv = self._preview.video_path
        if pv and Path(pv).is_file():
            saved_frame = self._last_frame_path
            self._source_player.blockSignals(True)
            self._source_player.load_video(pv)
            self._source_player.blockSignals(False)
            self._last_frame_path = saved_frame
            self._show_status("Loaded result as new source.")

    @Slot()
    def _on_download_frame(self) -> None:
        path = self._preview.capture_frame()
        if path:
            self.download_frame(path)

    @Slot()
    def _on_save_frame(self) -> None:
        path = self._preview.capture_frame()
        if path:
            saved = self.save_frame_to_project(path)
            if saved:
                self._show_status("Saved frame to project.")

    @Slot()
    def _on_save_clip(self) -> None:
        vid = self._preview.video_path
        if not vid or not Path(vid).is_file():
            self._show_status("No clip to save.")
            return
        name = self._clip_name.text().strip()
        if not name:
            name = Path(vid).stem
        clips_dir = self.project_path / "clips" / "extend"
        clips_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(vid).suffix
        dest = clips_dir / f"{name}{ext}"
        idx = 1
        while dest.exists():
            dest = clips_dir / f"{name}_{idx}{ext}"
            idx += 1

        # If post-processing is configured, run it on the final clip now.
        # Reuse the config the clip was generated with so post-gen UI edits
        # (steps/guidance/quality etc.) don't leak into the post-processing
        # step. Fall back to the live UI only if no generation happened yet.
        cfg = self._gen_cfg
        if cfg is None:
            cfg = ProjectConfig.load(self.project_path)
            self._params.collect_to_config(cfg)
        from sdqt.workers.inference import _needs_postprocessing
        if _needs_postprocessing(cfg):
            self._save_clip_btn.setEnabled(False)
            self._show_status("Post-processing before save...")
            worker = PostProcessWorker(vid, str(dest), cfg, parent=self)
            worker.progress.connect(lambda f, d: self._show_status(d))
            worker.finished_ok.connect(
                lambda path: self._on_save_clip_done(Path(path).name)
            )
            worker.error.connect(
                lambda msg: self._on_save_clip_error(msg)
            )
            worker.finished.connect(worker.deleteLater)
            self._pp_worker = worker
            worker.start()
        else:
            shutil.copy2(vid, dest)
            self._show_status(f"Saved clip: {dest.name}")

    def _on_save_clip_done(self, clip_name: str) -> None:
        self._save_clip_btn.setEnabled(True)
        self._show_status(f"Saved clip: {clip_name}")

    def _on_save_clip_error(self, msg: str) -> None:
        self._save_clip_btn.setEnabled(True)
        self._show_status(f"Post-process error: {msg}")

    # -- LoRA helpers ------------------------------------------------------

    def _get_available_loras(self) -> list[str]:
        lora_mgr = self.state.lora_manager
        if lora_mgr is None:
            return []
        return lora_mgr.list_available()

    # -- Project change ----------------------------------------------------

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)

        # Load config BEFORE clearing — clear_video() emits video_cleared
        # which triggers _on_source_cleared, overwriting the saved path.
        try:
            cfg = ProjectConfig.load(self.project_path)
        except Exception:
            cfg = None

        # Block signals during clear so handlers don't wipe saved config
        self._source_player.blockSignals(True)
        self._preview.blockSignals(True)
        self._source_player.clear_video()
        self._preview.clear_video()
        self._source_player.blockSignals(False)
        self._preview.blockSignals(False)
        self._guidance_video.clear_file()
        self._use_guidance.setChecked(False)
        self._final_frame.clear_image()
        self._use_final_frame.setChecked(False)

        if cfg is None:
            return

        vid = cfg.ve_video_path
        if vid and Path(vid).is_file():
            self.load_source(vid)

        pv = getattr(cfg, "ve_preview_path", "") or ""
        if pv and Path(pv).is_file():
            self._preview.load_video(pv)

        self._prompt.setPlainText(cfg.prompt or "")
        self._neg_prompt.setText(cfg.negative_prompt or "")

        # Guidance video
        gv = cfg.ve_guidance_video_path or ""
        if gv and Path(gv).is_file():
            self._guidance_video.load_file(gv)
            self._use_guidance.setChecked(True)
        idx = self._guidance_frame.findText(cfg.ve_guidance_video_frame or "first")
        if idx >= 0:
            self._guidance_frame.setCurrentIndex(idx)

        # Final frame
        ff = getattr(cfg, "ve_final_frame_path", "") or ""
        if ff and Path(ff).is_file():
            self._final_frame.load_image(ff)
            self._use_final_frame.setChecked(True)

        # Auto-expand the Guidance/Final Frame section when either is in use so
        # the restored controls are visible.
        if self._use_guidance.isChecked() or self._use_final_frame.isChecked():
            self._gf_section.set_collapsed(False)

        self._params.restore_from_config(cfg)

        # Extension continuity: if the project never set a custom denoise value
        # (still the global 1.0 default), drop to a source-anchoring default so
        # the continuation doesn't fully re-noise away from the source.
        if abs(float(getattr(cfg, "denoising_strength", 1.0)) - 1.0) < 1e-6:
            self._params.denoising_strength.setValue(_VE_DEFAULT_DENOISE)

        # Refresh LoRAs — always include ltx_lora_dir so the LTX system-LoRA
        # checkboxes light up regardless of which manager is active.
        try:
            available: list[str] = []
            ltx_dir = self.state.global_config.model_paths.get("ltx_lora_dir", "")
            if ltx_dir:
                p = Path(ltx_dir)
                if p.is_dir():
                    available = sorted(
                        f.name for f in p.iterdir()
                        if f.suffix.lower() in (".safetensors", ".pt", ".pth")
                    )
            lora_mgr = self.state.lora_manager
            if lora_mgr:
                seen = set(available)
                for name in lora_mgr.list_available():
                    if name not in seen:
                        available.append(name)
                        seen.add(name)
            self._params.populate_loras(available, cfg.activated_loras)
        except Exception:
            pass

        pv = cfg.ve_preview_path
        if pv and Path(pv).is_file():
            self._preview.load_video(pv)
            self._continue_btn.setEnabled(True)
            self._save_clip_btn.setEnabled(True)
            self._dl_frame_btn.setEnabled(True)
            self._save_frame_btn.setEnabled(True)
