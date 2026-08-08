"""QwenAlyzer sub-tab for Image Suite.

Combines PNG metadata reading/editing with LLM-powered image analysis.
Drop any image (PNG, JPEG, WebP, etc.) to:
  - View PNG metadata (if available)
  - QwenAlyze: generate a prompt + negative from the image via Qwen 3.5 4B
  - Send the image + generated prompt to any generation tab
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from supremediffusion.config.project_config import ProjectConfig
from sdqt.utils.file_dialog import get_open_filename, get_save_filename
from sdqt.widgets.image_drop import ImageDropWidget
from sdqt.widgets.send_targets import DISABLED_TARGETS
from sdqt.tabs.base import BaseTab

logger = logging.getLogger(__name__)

_W_COMBO = 120

# Image style options — same keys as prompt_enhance.py
_IMAGE_STYLES = [
    ("SD 1.5", "sd15"),
    ("SDXL", "sdxl"),
    ("SD3 / SD3.5", "sd3"),
    ("FLUX", "flux"),
    ("Pony: Real", "pony_real"),
    ("Pony: Anime", "pony_anime"),
    ("Illustrious", "illustrious"),
    ("NoobAI", "noobai"),
]

# System prompt for image captioning / analysis
_CAPTION_SYSTEM = (
    "You are an expert image analyst for AI image generation. "
    "Given an image description, generate a detailed prompt that would reproduce "
    "this image using the specified model family. "
    "Describe the subject, composition, lighting, colors, style, and mood. "
    "Be specific and detailed. Output only the prompt text, nothing else."
)

_CAPTION_USER_TEMPLATE = (
    "Analyze this image and generate a {style_name} prompt to reproduce it.\n\n"
    "Image details: {description}\n\n"
    "Generate the prompt in the format appropriate for {style_name}."
)

# Send-to targets that accept both an image and a prompt
_SEND_TARGETS = [
    ("txt2img", "Txt2Img"),
    ("img2img", "Img2Img"),
    ("inpaint", "Inpaint"),
    ("imgedit", "Draw"),
    ("bodydouble_src", "Body Double (source)"),
    ("bodydouble_tgt", "Body Double (target)"),
    ("repose_src", "RePose (source)"),
    ("repose_pose", "RePose (pose ref)"),
    ("controlnet_cond", "ControlNet (condition)"),
    ("controlnet_src", "ControlNet (source)"),
    ("cropzoom", "Crop/Zoom"),
    ("model3d", "3D Modeling"),
]


class QwenAlyzerWorker:
    """Inline import to avoid circular deps — actual class below."""
    pass


class PngInfoTab(BaseTab):
    """QwenAlyzer — image metadata + LLM-powered image analysis and prompt generation."""

    # Signal: (target_key, image_path, prompt, negative_prompt)
    send_with_prompt = Signal(str, str, str, str)

    def __init__(self, state, parent=None) -> None:
        super().__init__(state, parent)
        self._current_path: str | None = None
        self._raw_info: dict[str, str] = {}
        self._qwen_worker = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 0, 2, 2)
        splitter = QSplitter(Qt.Horizontal)

        # ── Left: image preview (drag-drop with send-to) ─────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)

        src_row = QHBoxLayout()
        self._path = QLineEdit()
        self._path.setReadOnly(True)
        self._path.setPlaceholderText("Drop or browse any image...")
        src_row.addWidget(self._path, 1)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        src_row.addWidget(browse)
        left_layout.addLayout(src_row)

        self._image_drop = ImageDropWidget("Drop image here", thumb_height=400)
        self._image_drop.image_loaded.connect(self._on_image_dropped)
        self._image_drop.image_cleared.connect(self._on_image_cleared)
        left_layout.addWidget(self._image_drop, 1)

        # Right-click "Send Image + Prompt" on the preview
        self._image_drop.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._image_drop.customContextMenuRequested.connect(self._on_preview_context)

        splitter.addWidget(left)

        # ── Right: metadata + QwenAlyze ──────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # --- PNG Metadata section (collapsible via groupbox) ---
        meta_group = QGroupBox("PNG Metadata")
        meta_layout = QVBoxLayout(meta_group)
        meta_layout.setContentsMargins(4, 4, 4, 4)

        self._info_text = QTextEdit()
        self._info_text.setReadOnly(True)
        self._info_text.setMaximumHeight(120)
        self._info_text.setPlaceholderText("PNG metadata will appear here...")
        meta_layout.addWidget(self._info_text)

        meta_layout.addWidget(QLabel("<b>Generation Parameters (editable)</b>"))
        self._params_text = QPlainTextEdit()
        self._params_text.setMaximumHeight(80)
        self._params_text.setPlaceholderText(
            "A1111-style parameters (paste or edit)..."
        )
        meta_layout.addWidget(self._params_text)

        meta_btn_row = QHBoxLayout()
        self._save_meta_btn = QPushButton("Save Metadata")
        self._save_meta_btn.setToolTip("Write edited parameters back into the image file")
        self._save_meta_btn.clicked.connect(self._on_save_metadata)
        meta_btn_row.addWidget(self._save_meta_btn)
        self._save_as_btn = QPushButton("Save As...")
        self._save_as_btn.setToolTip("Save a copy with edited metadata")
        self._save_as_btn.clicked.connect(self._on_save_as)
        meta_btn_row.addWidget(self._save_as_btn)
        self._inject_btn = QPushButton("Inject into Image...")
        self._inject_btn.setToolTip("Pick any image and inject the parameters text as PNG metadata")
        self._inject_btn.clicked.connect(self._on_inject)
        meta_btn_row.addWidget(self._inject_btn)
        meta_layout.addLayout(meta_btn_row)

        right_layout.addWidget(meta_group)

        # --- QwenAlyze section ---
        qwen_group = QGroupBox("QwenAlyze")
        qg_layout = QVBoxLayout(qwen_group)
        qg_layout.setContentsMargins(4, 4, 4, 4)
        qg_layout.setSpacing(4)

        # Controls row: Model Family + QwenAlyze button
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(6)
        ctrl_row.addWidget(QLabel("Model Family:"))
        self._style_combo = QComboBox()
        self._style_combo.setFixedWidth(_W_COMBO)
        for display, key in _IMAGE_STYLES:
            self._style_combo.addItem(display, userData=key)
        # Default to SDXL
        idx = self._style_combo.findData("sdxl")
        if idx >= 0:
            self._style_combo.setCurrentIndex(idx)
        ctrl_row.addWidget(self._style_combo)

        self._qwenalyze_btn = QPushButton("QwenAlyze")
        self._qwenalyze_btn.setStyleSheet("font-weight: bold;")
        self._qwenalyze_btn.setToolTip(
            "Analyze the image with Qwen 3.5 4B and generate a reproduction prompt"
        )
        self._qwenalyze_btn.clicked.connect(self._on_qwenalyze)
        ctrl_row.addWidget(self._qwenalyze_btn)
        ctrl_row.addStretch()
        qg_layout.addLayout(ctrl_row)

        # Prompt output
        pos_label_row = QHBoxLayout()
        pos_label_row.setSpacing(4)
        pos_label_row.addWidget(QLabel("<b>Prompt:</b>"))
        pos_label_row.addStretch()
        self._reroll_pos_btn = QPushButton("Re-Roll")
        self._reroll_pos_btn.setFixedHeight(22)
        self._reroll_pos_btn.setToolTip("Re-generate positive prompt with Qwen")
        self._reroll_pos_btn.clicked.connect(self._on_reroll_pos)
        self._reroll_pos_btn.setEnabled(False)
        pos_label_row.addWidget(self._reroll_pos_btn)
        qg_layout.addLayout(pos_label_row)

        self._qwen_prompt = QPlainTextEdit()
        self._qwen_prompt.setPlaceholderText("Generated prompt will appear here...")
        qg_layout.addWidget(self._qwen_prompt, 1)

        # Negative prompt output
        neg_label_row = QHBoxLayout()
        neg_label_row.setSpacing(4)
        neg_label_row.addWidget(QLabel("<b>Negative:</b>"))
        neg_label_row.addStretch()
        self._reroll_neg_btn = QPushButton("Re-Roll")
        self._reroll_neg_btn.setFixedHeight(22)
        self._reroll_neg_btn.setToolTip("Re-generate negative prompt with Qwen")
        self._reroll_neg_btn.clicked.connect(self._on_reroll_neg)
        self._reroll_neg_btn.setEnabled(False)
        neg_label_row.addWidget(self._reroll_neg_btn)
        qg_layout.addLayout(neg_label_row)

        self._qwen_negative = QPlainTextEdit()
        self._qwen_negative.setMaximumHeight(60)
        self._qwen_negative.setPlaceholderText("Generated negative prompt...")
        qg_layout.addWidget(self._qwen_negative)

        # Send-to row
        send_row = QHBoxLayout()
        send_row.setSpacing(6)
        self._send_btn = QPushButton("Send To...")
        self._send_btn.setToolTip("Send image + generated prompt to a generation tab")
        self._send_btn.clicked.connect(self._on_send_menu)
        self._send_btn.setEnabled(False)
        send_row.addWidget(self._send_btn)
        send_row.addStretch()
        qg_layout.addLayout(send_row)

        right_layout.addWidget(qwen_group, 1)

        self._status = QLabel("")
        right_layout.addWidget(self._status)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter)

    # ── Browse / load ────────────────────────────────────────────────────

    def _browse(self) -> None:
        path, _ = get_open_filename(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.tif)"
        )
        if path:
            self._image_drop.load_image(path)

    def load_image(self, path: str) -> None:
        """Public API — load an image from another tab."""
        self._image_drop.load_image(path)

    def _on_image_dropped(self, path: str) -> None:
        """Called when ImageDropWidget loads an image (drop, browse, or load_image)."""
        self._path.setText(path)
        self._load_info(path)
        self._save_state()

    def _on_image_cleared(self) -> None:
        self._path.clear()
        self._current_path = None
        self._info_text.clear()
        self._params_text.clear()
        self._qwen_prompt.clear()
        self._qwen_negative.clear()
        self._send_btn.setEnabled(False)
        self._save_state()

    def _on_preview_context(self, pos) -> None:
        """Right-click on the image preview: send image + generated prompts."""
        if not self._current_path:
            return
        menu = QMenu(self)
        prompt = self._qwen_prompt.toPlainText()
        neg = self._qwen_negative.toPlainText()
        has_prompt = bool(prompt.strip())

        for key, label in _SEND_TARGETS:
            if has_prompt:
                action = menu.addAction(f"{label} (+ prompt)")
            else:
                action = menu.addAction(label)
            action.setData(key)
            if key in DISABLED_TARGETS:
                action.setEnabled(False)

        chosen = menu.exec(self._image_drop.mapToGlobal(pos))
        if chosen:
            target = chosen.data()
            self.send_with_prompt.emit(
                target,
                self._current_path or "",
                prompt,
                neg,
            )

    def _load_info(self, path: str) -> None:
        self._current_path = path

        # PNG metadata — only for PNG files
        is_png = path.lower().endswith(".png")
        if is_png:
            try:
                from supremediffusion.utils.png_info import (
                    read_png_info,
                    format_info_html,
                )
                self._raw_info = read_png_info(path)
                self._info_text.setHtml(format_info_html(self._raw_info))
                params_text = self._raw_info.get("parameters", "")
                if params_text:
                    self._params_text.setPlainText(params_text)
                else:
                    self._params_text.clear()
            except Exception as exc:
                self._info_text.setPlainText(f"Error reading metadata: {exc}")
                self._params_text.clear()
                self._raw_info = {}
        else:
            self._info_text.setPlainText(
                f"<i>Not a PNG file — no embedded metadata.</i><br>"
                f"Format: {Path(path).suffix.upper().lstrip('.')}"
            )
            self._params_text.clear()
            self._raw_info = {}

        # Enable send if we have an image
        self._send_btn.setEnabled(True)
        self._show_status("")

    # ── QwenAlyze ────────────────────────────────────────────────────────

    @Slot()
    def _on_qwenalyze(self) -> None:
        """Run Qwen analysis on the loaded image — generates both pos and neg prompts."""
        if not self._current_path:
            self._show_status("Load an image first.")
            return

        # Ensure VL deps + model are available
        from sdqt.deps import check_and_install
        if not check_and_install("qwen_vl", self):
            return
        if not check_and_install("transformers", self):
            return

        from sdqt.models.manager import check_and_prompt_download
        if not check_and_prompt_download(
            "qwen_vl", self.state.model_registry, self,
            on_progress=lambda f, d: self._show_status(d),
        ):
            return
        if not check_and_prompt_download(
            "prompt_enhance", self.state.model_registry, self,
            on_progress=lambda f, d: self._show_status(d),
        ):
            return

        self._run_qwen("both")

    @Slot()
    def _on_reroll_pos(self) -> None:
        """Re-roll just the positive prompt."""
        if not self._current_path:
            self._show_status("Load an image first.")
            return
        self._run_qwen("pos")

    @Slot()
    def _on_reroll_neg(self) -> None:
        """Re-roll just the negative prompt."""
        if not self._current_path:
            self._show_status("Load an image first.")
            return
        self._run_qwen("neg")

    def _run_qwen(self, which: str) -> None:
        """Launch Qwen VL + text worker for pos, neg, or both."""
        style_key = self._style_combo.currentData() or "sdxl"
        style_name = self._style_combo.currentText()

        self._qwenalyze_btn.setEnabled(False)
        self._reroll_pos_btn.setEnabled(False)
        self._reroll_neg_btn.setEnabled(False)

        from sdqt.workers.qwenalyzer import QwenAlyzerWorker
        worker = QwenAlyzerWorker(
            image_path=self._current_path,
            style_key=style_key,
            style_name=style_name,
            which=which,
            state=self.state,
            parent=self,
        )
        worker.finished_ok.connect(lambda r: self._on_qwen_done(which, r))
        worker.error.connect(self._on_qwen_error)
        worker.status.connect(lambda s: self._show_status(s))
        worker.finished.connect(worker.deleteLater)
        self._qwen_worker = worker
        worker.start()

    def _build_description(self) -> str:
        """Build an image description from available metadata and file info."""
        parts = []

        # Use existing PNG parameters if available
        params = self._params_text.toPlainText().strip()
        if params:
            parts.append(f"Existing generation parameters: {params}")

        # Add file info
        if self._current_path:
            p = Path(self._current_path)
            parts.append(f"Filename: {p.name}")
            try:
                from PIL import Image as PILImage
                img = PILImage.open(self._current_path)
                parts.append(f"Dimensions: {img.width}x{img.height}")
                parts.append(f"Mode: {img.mode}")
            except Exception:
                pass

        if not parts:
            return "(No image information available — generate a creative prompt)"

        return "\n".join(parts)

    def _on_qwen_done(self, which: str, result: dict) -> None:
        self._qwenalyze_btn.setEnabled(True)
        self._reroll_pos_btn.setEnabled(True)
        self._reroll_neg_btn.setEnabled(True)

        if which in ("pos", "both"):
            pos = result.get("pos", "")
            if pos:
                self._qwen_prompt.setPlainText(pos)
        if which in ("neg", "both"):
            neg = result.get("neg", "")
            if neg:
                self._qwen_negative.setPlainText(neg)

        self._send_btn.setEnabled(True)
        self._show_status("QwenAlysis complete.")
        self._save_state()

    def _on_qwen_error(self, msg: str) -> None:
        self._qwenalyze_btn.setEnabled(True)
        self._reroll_pos_btn.setEnabled(True)
        self._reroll_neg_btn.setEnabled(True)
        self._show_status(f"QwenAlyze error: {msg}")

    # ── Send-to menu ─────────────────────────────────────────────────────

    @Slot()
    def _on_send_menu(self) -> None:
        """Show a context menu with send-to targets."""
        menu = QMenu(self)
        for key, label in _SEND_TARGETS:
            action = menu.addAction(label)
            action.setData(key)

        chosen = menu.exec(self._send_btn.mapToGlobal(self._send_btn.rect().bottomLeft()))
        if chosen:
            target = chosen.data()
            self.send_with_prompt.emit(
                target,
                self._current_path or "",
                self._qwen_prompt.toPlainText(),
                self._qwen_negative.toPlainText(),
            )

    # ── Metadata writing helpers ─────────────────────────────────────────

    @staticmethod
    def _write_png_metadata(image_path: str, parameters: str) -> str:
        """Write A1111-style parameters into a PNG file's text chunks."""
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo

        img = Image.open(image_path)
        png_info = PngInfo()

        # Preserve existing text chunks except "parameters"
        existing = getattr(img, "info", {}) or {}
        text = getattr(img, "text", {}) or {}
        for d in (existing, text):
            for k, v in d.items():
                if isinstance(v, str) and k != "parameters":
                    png_info.add_text(k, v)

        if parameters.strip():
            png_info.add_text("parameters", parameters)

        out_path = image_path
        if not out_path.lower().endswith(".png"):
            out_path = str(Path(out_path).with_suffix(".png"))

        if img.mode not in ("RGB", "RGBA", "L", "LA", "P"):
            img = img.convert("RGBA")

        img.save(out_path, "PNG", pnginfo=png_info)
        return out_path

    @Slot()
    def _on_save_metadata(self) -> None:
        if not self._current_path:
            self._show_status("No image loaded.")
            return
        if not Path(self._current_path).is_file():
            self._show_status("Image file not found.")
            return

        params = self._params_text.toPlainText()
        try:
            out = self._write_png_metadata(self._current_path, params)
            self._show_status(f"Metadata saved to {Path(out).name}")
            self._load_info(out)
        except Exception as exc:
            self._show_status(f"Error saving: {exc}")

    @Slot()
    def _on_save_as(self) -> None:
        if not self._current_path or not Path(self._current_path).is_file():
            self._show_status("No image loaded.")
            return

        stem = Path(self._current_path).stem
        dest, _ = get_save_filename(
            self, "Save Image With Metadata", f"{stem}_meta.png", "PNG (*.png)"
        )
        if not dest:
            return

        params = self._params_text.toPlainText()
        try:
            shutil.copy2(self._current_path, dest)
            self._write_png_metadata(dest, params)
            self._show_status(f"Saved: {Path(dest).name}")
        except Exception as exc:
            self._show_status(f"Error: {exc}")

    @Slot()
    def _on_inject(self) -> None:
        params = self._params_text.toPlainText().strip()
        if not params:
            self._show_status("No parameters to inject. Type or paste metadata first.")
            return

        path, _ = get_open_filename(
            self, "Select Image to Inject Metadata Into",
            "", "Images (*.png *.jpg *.jpeg *.webp)",
        )
        if not path:
            return

        try:
            out = self._write_png_metadata(path, params)
            self._show_status(f"Injected metadata into {Path(out).name}")
            self._load_info(out)
        except Exception as exc:
            self._show_status(f"Injection error: {exc}")

    # ── Project persistence ──────────────────────────────────────────────

    def on_project_changed(self, project_name: str) -> None:
        super().on_project_changed(project_name)
        self._restoring = True
        try:
            cfg = ProjectConfig.load(self.project_path)
        except Exception:
            self._restoring = False
            return
        # Restore image
        path = getattr(cfg, "qwenalyzer_image_path", "") or ""
        if path and Path(path).is_file():
            self._image_drop.load_image(path)
        else:
            self._image_drop.clear_image()
            self._current_path = None
            self._path.clear()
        # Restore style
        style = getattr(cfg, "qwenalyzer_style", "sdxl") or "sdxl"
        idx = self._style_combo.findData(style)
        if idx >= 0:
            self._style_combo.setCurrentIndex(idx)
        # Restore prompts
        self._qwen_prompt.setPlainText(getattr(cfg, "qwenalyzer_prompt", "") or "")
        self._qwen_negative.setPlainText(getattr(cfg, "qwenalyzer_negative", "") or "")
        self._restoring = False

    def _save_state(self) -> None:
        """Persist QwenAlyzer state to project config."""
        if self._restoring:
            return
        try:
            cfg = ProjectConfig.load(self.project_path)
            cfg.qwenalyzer_image_path = self._current_path or ""
            cfg.qwenalyzer_style = self._style_combo.currentData() or "sdxl"
            cfg.qwenalyzer_prompt = self._qwen_prompt.toPlainText()
            cfg.qwenalyzer_negative = self._qwen_negative.toPlainText()
            cfg.save(self.project_path)
        except Exception:
            pass
