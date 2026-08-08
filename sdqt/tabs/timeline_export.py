"""Timeline export dialog and mixin — export settings and progress handling."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Slot, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from sdqt.utils.codec import configured_codec_args as _codec_args, codec_labels, get_codec_args
from sdqt.workers.timeline import TimelineExportWorker

logger = logging.getLogger(__name__)


class ExportDialog(QDialog):
    """Export settings dialog for the timeline."""

    def __init__(self, default_path: str = "", parent=None, saved: dict | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Timeline")
        self.setMinimumWidth(500)
        self._saved = saved or {}

        layout = QVBoxLayout(self)

        # Resolution
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Resolution:"))
        self.resolution = QComboBox()
        self.resolution.addItems([
            "Source (no change)", "512x512", "832x480",
            "1024x1024", "1280x720", "1920x1080",
        ])
        self.resolution.setMinimumWidth(180)
        res_row.addWidget(self.resolution)
        res_row.addStretch()
        layout.addLayout(res_row)

        # GL Render
        gl_row = QHBoxLayout()
        self.gl_render = QCheckBox("GL Render (pixel-identical to preview)")
        self.gl_render.setToolTip("Render each frame through the GL preview pipeline instead of ffmpeg filters")
        self.gl_render.setChecked(True)
        gl_row.addWidget(self.gl_render)
        gl_row.addStretch()
        layout.addLayout(gl_row)

        # RIFE
        rife_row = QHBoxLayout()
        self.rife_enabled = QCheckBox("RIFE Temporal Upsampling")
        rife_row.addWidget(self.rife_enabled)
        self.rife_mode = QComboBox()
        self.rife_mode.addItems(["RIFE x2", "RIFE x4"])
        self.rife_mode.setFixedWidth(100)
        self.rife_mode.setEnabled(False)
        self.rife_enabled.toggled.connect(self.rife_mode.setEnabled)
        rife_row.addWidget(self.rife_mode)
        rife_row.addStretch()
        layout.addLayout(rife_row)

        # Codec
        codec_row = QHBoxLayout()
        codec_row.addWidget(QLabel("Codec:"))
        self.codec = QComboBox()
        self.codec.addItems(codec_labels())
        self.codec.setMinimumWidth(180)
        codec_row.addWidget(self.codec)
        codec_row.addStretch()
        layout.addLayout(codec_row)

        # ── Post-Processing (collapsed by default — advanced/optional) ──
        from sdqt.widgets.collapsible_section import CollapsibleSection
        pp_section = CollapsibleSection(
            "Post-Processing (applied to final export)", collapsed=True)
        layout.addWidget(pp_section)
        pp_layout = pp_section.content_layout

        def _pp_row(label_text, items, tooltip):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"  {label_text}:"))
            combo = QComboBox()
            combo.addItems(items)
            combo.setMinimumWidth(180)
            combo.setToolTip(tooltip)
            row.addWidget(combo)
            row.addStretch()
            pp_layout.addLayout(row)
            return combo

        self.pp_denoise = _pp_row(
            "Denoise", ["Disabled", "SCUNet"],
            "Remove grain/noise while preserving detail")
        self.pp_sharpen = _pp_row(
            "Sharpen", ["Disabled", "RealESRGAN 2x"],
            "Neural detail recovery (upscale 2x then downscale)")
        self.pp_upscale = _pp_row(
            "Upscale", ["Disabled", "RealESRGAN 4x"],
            "4x neural upscale")
        self.pp_face = _pp_row(
            "Face Restore", ["Disabled", "GFPGAN v1.4"],
            "Fix AI-generated face artifacts")

        tile_row = QHBoxLayout()
        tile_row.addWidget(QLabel("  AI Tile Size:"))
        self.pp_tile = QSpinBox()
        self.pp_tile.setRange(0, 1024)
        self.pp_tile.setValue(512)
        self.pp_tile.setSingleStep(128)
        self.pp_tile.setSuffix("px")
        self.pp_tile.setFixedWidth(80)
        tile_row.addWidget(self.pp_tile)
        tile_row.addStretch()
        pp_layout.addLayout(tile_row)

        self.pp_lanczos = _pp_row(
            "Lanczos Upscale", ["Disabled", "Lanczos 1.5x", "Lanczos 2.0x"],
            "Traditional spatial upscale")

        grain_row = QHBoxLayout()
        grain_row.addWidget(QLabel("  Film Grain:"))
        self.pp_grain = QSlider(Qt.Horizontal)
        self.pp_grain.setRange(0, 100)
        self.pp_grain.setValue(0)
        grain_row.addWidget(self.pp_grain, 1)
        self._grain_label = QLabel("0%")
        self._grain_label.setFixedWidth(35)
        self.pp_grain.valueChanged.connect(lambda v: self._grain_label.setText(f"{v}%"))
        grain_row.addWidget(self._grain_label)
        pp_layout.addLayout(grain_row)

        # Godot-compatible output
        godot_row = QHBoxLayout()
        self.godot_mode = QCheckBox("Godot (yuv420p, tv range, bt709)")
        self.godot_mode.setToolTip(
            "Export with limited-range color space that Godot and standard\n"
            "players interpret consistently. Use this for game engine import."
        )
        if self._saved.get("godot_mode"):
            self.godot_mode.setChecked(True)
        godot_row.addWidget(self.godot_mode)
        godot_row.addStretch()
        layout.addLayout(godot_row)

        # Filename
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Output:"))
        self.filename = QLineEdit(default_path)
        file_row.addWidget(self.filename, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Ok).setText("Export")
        layout.addWidget(buttons)

        # Restore saved settings
        if self._saved:
            s = self._saved
            if s.get("resolution"):
                idx = self.resolution.findText(s["resolution"])
                if idx >= 0:
                    self.resolution.setCurrentIndex(idx)
            if s.get("codec"):
                idx = self.codec.findText(s["codec"])
                if idx >= 0:
                    self.codec.setCurrentIndex(idx)
            if s.get("rife_mode"):
                self.rife_enabled.setChecked(True)
                rife_text = f"RIFE {s['rife_mode']}"
                idx = self.rife_mode.findText(rife_text)
                if idx >= 0:
                    self.rife_mode.setCurrentIndex(idx)
            if "gl_render" in s:
                self.gl_render.setChecked(bool(s["gl_render"]))
            # Post-process combos persisted as their displayed text
            for key, combo in (
                ("pp_denoise_text", self.pp_denoise),
                ("pp_sharpen_text", self.pp_sharpen),
                ("pp_upscale_text", self.pp_upscale),
                ("pp_face_text", self.pp_face),
                ("pp_lanczos_text", self.pp_lanczos),
            ):
                val = s.get(key)
                if val:
                    idx = combo.findText(val)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
            if "pp_tile" in s:
                try:
                    self.pp_tile.setValue(int(s["pp_tile"]))
                except (TypeError, ValueError):
                    pass
            if "pp_grain_pct" in s:
                try:
                    self.pp_grain.setValue(int(s["pp_grain_pct"]))
                except (TypeError, ValueError):
                    pass

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Video", self.filename.text(),
            "Videos (*.mp4);;All Files (*)",
        )
        if path:
            self.filename.setText(path)

    def get_settings(self) -> dict:
        res = self.resolution.currentText()
        if res.startswith("Source"):
            res = ""
        codec = self.codec.currentText()
        rife = ""
        if self.rife_enabled.isChecked():
            rife = "x2" if "x2" in self.rife_mode.currentText() else "x4"
        return {
            "resolution": res,
            "rife_mode": rife,
            "codec": codec,
            "output": self.filename.text(),
            "gl_render": self.gl_render.isChecked(),
            "pp_denoise": self.pp_denoise.currentIndex() > 0,
            "pp_sharpen": self.pp_sharpen.currentIndex() > 0,
            "pp_upscale": self.pp_upscale.currentIndex() > 0,
            "pp_face": self.pp_face.currentIndex() > 0,
            "pp_tile": self.pp_tile.value(),
            "pp_lanczos": {"Lanczos 1.5x": "1.5", "Lanczos 2.0x": "2.0"}.get(
                self.pp_lanczos.currentText(), ""),
            "pp_grain": self.pp_grain.value() / 100.0,
            "godot_mode": self.godot_mode.isChecked(),
        }


class TimelineExportMixin:
    """Mixin providing export methods for TimelineTab."""

    @Slot()
    def _on_export(self) -> None:
        from supremediffusion.config.project_config import ProjectConfig

        tracks = self._multitrack.tracks
        has_clips = any(t.clips for t in tracks)
        if not has_clips:
            self._status_label.setText("No clips to export")
            return

        # Load saved export settings
        try:
            cfg = ProjectConfig.load(self.project_path)
            saved = cfg.tl_export_settings or {}
        except Exception:
            saved = {}

        # Default output path: last saved output if available, else project clips
        last_out = saved.get("output") if isinstance(saved, dict) else None
        if last_out:
            last_dir = Path(last_out).parent
            if last_dir.exists():
                default_out = str(last_dir / Path(last_out).name)
            else:
                default_out = str(self.project_path / "clips" / "timeline" / "timeline_export.mp4")
        else:
            default_out = str(self.project_path / "clips" / "timeline" / "timeline_export.mp4")

        dlg = ExportDialog(default_out, parent=self, saved=saved)
        if dlg.exec() != QDialog.Accepted:
            return

        settings = dlg.get_settings()

        # Save export settings for next time — persist every dialog field so
        # the user's choices stick across sessions.
        try:
            cfg = ProjectConfig.load(self.project_path)
            cfg.tl_export_settings = {
                "resolution": settings.get("resolution") or dlg.resolution.currentText(),
                "rife_mode": settings.get("rife_mode", ""),
                "codec": settings.get("codec", ""),
                "gl_render": bool(settings.get("gl_render", False)),
                "pp_denoise_text": dlg.pp_denoise.currentText(),
                "pp_sharpen_text": dlg.pp_sharpen.currentText(),
                "pp_upscale_text": dlg.pp_upscale.currentText(),
                "pp_face_text": dlg.pp_face.currentText(),
                "pp_lanczos_text": dlg.pp_lanczos.currentText(),
                "pp_tile": dlg.pp_tile.value(),
                "pp_grain_pct": dlg.pp_grain.value(),
                "output": settings.get("output", ""),
            }
            cfg.save(self.project_path)
        except Exception:
            pass
        output = settings["output"]
        if not output:
            return

        Path(output).parent.mkdir(parents=True, exist_ok=True)

        self._btn_export.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status_label.setText("Exporting...")

        if settings.get("gl_render") and self._gl_preview and self._gl_preview.available:
            # GL Render path — pixel-identical to preview
            from sdqt.workers.gl_export import GLExportController

            # Parse resolution
            res = settings["resolution"]
            if res:
                try:
                    w, h = (int(x) for x in res.split("x"))
                except ValueError:
                    w, h = 1280, 720
            else:
                # Use first clip's native resolution
                w, h = 1280, 720
                for t in tracks:
                    for c in t.clips:
                        from supremediffusion.utils.video import probe_video
                        try:
                            info = probe_video(c.path)
                            w = info.get("width", 1280)
                            h = info.get("height", 720)
                            break
                        except Exception:
                            pass
                    if w != 1280:
                        break

            self._gl_export = GLExportController(
                gl_preview=self._gl_preview,
                output=output,
                width=w, height=h,
                fps=self._gl_preview.fps,
                codec=settings["codec"],
                rife_mode=settings["rife_mode"],
                godot_mode=settings.get("godot_mode", False),
                parent=self,
            )
            self._gl_export.progress.connect(self._on_export_progress)
            self._gl_export.finished.connect(self._on_export_done)
            self._gl_export.error.connect(self._on_export_error)
            self._gl_export.start()
        else:
            # Standard ffmpeg filter path
            self._export_worker = TimelineExportWorker(
                tracks=tracks,
                output=output,
                resolution=settings["resolution"],
                rife_mode=settings["rife_mode"],
                codec=settings["codec"],
                parent=self,
            )
            self._export_worker.progress.connect(self._on_export_progress)
            self._export_worker.finished_ok.connect(self._on_export_done)
            self._export_worker.error.connect(self._on_export_error)
            self._export_worker.start()

    @Slot(float, str)
    def _on_export_progress(self, frac: float, desc: str) -> None:
        self._progress.setValue(int(frac * 100))
        self._status_label.setText(desc)

    @Slot(object)
    def _on_export_done(self, path) -> None:
        if hasattr(self, "_export_worker") and self._export_worker:
            self._export_worker.deleteLater()
            self._export_worker = None
        if hasattr(self, "_gl_export") and self._gl_export:
            self._gl_export.deleteLater()
            self._gl_export = None
        self._btn_export.setEnabled(True)
        self._progress.setVisible(False)
        self._status_label.setText(f"Exported: {Path(path).name}")

        # Show result dialog with Open Folder option
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Export Complete")
        dlg.setText(f"Export complete:\n{path}")
        dlg.setIcon(QMessageBox.Information)
        open_btn = dlg.addButton("Open Folder", QMessageBox.ActionRole)
        dlg.addButton(QMessageBox.Ok)
        dlg.exec()
        if dlg.clickedButton() == open_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))

    @Slot(str)
    def _on_export_error(self, msg: str) -> None:
        if hasattr(self, "_export_worker") and self._export_worker:
            self._export_worker.deleteLater()
            self._export_worker = None
        if hasattr(self, "_gl_export") and self._gl_export:
            self._gl_export.deleteLater()
            self._gl_export = None
        self._btn_export.setEnabled(True)
        self._progress.setVisible(False)
        self._status_label.setText(f"Export failed: {msg}")
        logger.warning("Timeline export failed: %s", msg)
