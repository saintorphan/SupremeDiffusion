"""Sequential Processing dialog — pair-based img2vid (Mode 2) batch runner.

Requires every input image to be named as a bare integer (``0.png``,
``1.png``, …) and the full set must form a contiguous run. Iterates
sliding pairs (N, N+1), (N+1, N+2), … and feeds each pair to Mode 2.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from sdqt.widgets.post_process_controls import PostProcessControls

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_REF_EXTS = _IMAGE_EXTS | _VIDEO_EXTS


class SequentialProcessDialog(QDialog):
    """Collect Sequential Processing parameters for Mode 2."""

    def __init__(self, state, last_settings: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sequential Processing")
        self.setMinimumWidth(540)
        self._state = state
        self._selected_files: list[str] = []
        self._build_ui()
        self._restore(last_settings or {})
        self._sync_source_enabled()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(QLabel(
            "<b>Source</b> — files must be named as bare integers "
            "(e.g. <code>0.png 1.png 2.png</code>) and form a contiguous run."
        ))

        self._src_group = QButtonGroup(self)
        self._rb_dir = QRadioButton("Input Directory")
        self._rb_files = QRadioButton("Input Files")
        self._rb_dir.setChecked(True)
        self._src_group.addButton(self._rb_dir, 0)
        self._src_group.addButton(self._rb_files, 1)
        self._src_group.idToggled.connect(lambda _id, _on: self._sync_source_enabled())

        radio_row = QHBoxLayout()
        radio_row.addWidget(self._rb_dir)
        radio_row.addWidget(self._rb_files)
        radio_row.addStretch()
        layout.addLayout(radio_row)

        # Directory row
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Directory:"))
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("Folder containing integer-named images")
        dir_row.addWidget(self._dir_edit, 1)
        self._dir_browse = QPushButton("Browse…")
        self._dir_browse.clicked.connect(self._browse_dir)
        dir_row.addWidget(self._dir_browse)
        layout.addLayout(dir_row)

        # Files row
        files_row = QHBoxLayout()
        files_row.addWidget(QLabel("Files:"))
        self._files_edit = QLineEdit()
        self._files_edit.setReadOnly(True)
        self._files_edit.setPlaceholderText("0 files selected")
        files_row.addWidget(self._files_edit, 1)
        self._files_browse = QPushButton("Browse…")
        self._files_browse.clicked.connect(self._browse_files)
        files_row.addWidget(self._files_browse)
        layout.addLayout(files_row)

        layout.addWidget(_hline())

        # Output dir
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output Directory:"))
        self._out_edit = QLineEdit()
        out_row.addWidget(self._out_edit, 1)
        out_browse = QPushButton("Browse…")
        out_browse.clicked.connect(self._browse_output)
        out_row.addWidget(out_browse)
        layout.addLayout(out_row)

        # Prefix
        prefix_row = QHBoxLayout()
        prefix_row.addWidget(QLabel("Clip File Prefix:"))
        self._prefix_edit = QLineEdit("seq_")
        self._prefix_edit.setMaxLength(64)
        prefix_row.addWidget(self._prefix_edit, 1)
        layout.addLayout(prefix_row)

        # Guidance row with First/Last dropdown
        guide_row = QHBoxLayout()
        self._cb_guidance = QCheckBox("Generate/Use guidance vid")
        self._cb_guidance.setToolTip(
            "Synthesize a static guidance video from one of the pair's "
            "frames and pass it to the img2vid pipeline."
        )
        self._cb_guidance.toggled.connect(self._sync_guide_enabled)
        guide_row.addWidget(self._cb_guidance)

        guide_row.addWidget(QLabel("Guidance Frame:"))
        self._guide_frame = QComboBox()
        self._guide_frame.addItems(["First", "Last"])
        self._guide_frame.setToolTip(
            "Which of the pair's two frames is used to build the static "
            "guidance tensor for that pair."
        )
        guide_row.addWidget(self._guide_frame)
        guide_row.addStretch()
        layout.addLayout(guide_row)

        # Color Correct
        self._cb_cc = QCheckBox("Color Correct (match against reference)")
        self._cb_cc.setToolTip(
            "Build a 3D LUT from the reference frame and apply it to each "
            "generated clip before post-processing."
        )
        self._cb_cc.toggled.connect(self._sync_cc_enabled)
        layout.addWidget(self._cb_cc)

        cc_row = QHBoxLayout()
        cc_row.addWidget(QLabel("  Reference:"))
        self._cc_ref_edit = QLineEdit()
        self._cc_ref_edit.setPlaceholderText("Image or video reference file")
        cc_row.addWidget(self._cc_ref_edit, 1)
        self._cc_ref_browse = QPushButton("Browse…")
        self._cc_ref_browse.clicked.connect(self._browse_cc_ref)
        cc_row.addWidget(self._cc_ref_browse)
        layout.addLayout(cc_row)
        self._cc_row_widgets = (self._cc_ref_edit, self._cc_ref_browse)
        self._sync_cc_enabled(False)

        # Post-process
        layout.addWidget(_hline())
        self._cb_pp = QCheckBox("Apply Post-Process")
        self._cb_pp.toggled.connect(self._sync_pp_enabled)
        layout.addWidget(self._cb_pp)

        self._pp = PostProcessControls()
        self._pp.setEnabled(False)
        layout.addWidget(self._pp)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Begin")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._sync_guide_enabled(self._cb_guidance.isChecked())

    # ------------------------------------------------------------------

    def _browse_dir(self) -> None:
        start = self._dir_edit.text() or str(Path.home())
        path = QFileDialog.getExistingDirectory(
            self, "Select Input Directory", start,
        )
        if path:
            self._dir_edit.setText(path)
            self._rb_dir.setChecked(True)

    def _browse_files(self) -> None:
        start = (
            str(Path(self._selected_files[0]).parent)
            if self._selected_files else str(Path.home())
        )
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", start,
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;All Files (*)",
        )
        if paths:
            self._selected_files = list(paths)
            self._files_edit.setText(f"{len(paths)} file(s) selected")
            self._rb_files.setChecked(True)

    def _browse_output(self) -> None:
        start = self._out_edit.text() or str(Path.home())
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", start,
        )
        if path:
            self._out_edit.setText(path)

    def _browse_cc_ref(self) -> None:
        start = self._cc_ref_edit.text() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Color Reference", start,
            "Images & Videos (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.mp4 *.mov *.mkv *.webm *.avi);;All Files (*)",
        )
        if path:
            self._cc_ref_edit.setText(path)
            self._cb_cc.setChecked(True)

    def _sync_source_enabled(self) -> None:
        dir_mode = self._rb_dir.isChecked()
        self._dir_edit.setEnabled(dir_mode)
        self._dir_browse.setEnabled(dir_mode)
        self._files_edit.setEnabled(not dir_mode)
        self._files_browse.setEnabled(not dir_mode)

    def _sync_guide_enabled(self, on: bool) -> None:
        self._guide_frame.setEnabled(on)

    def _sync_cc_enabled(self, on: bool) -> None:
        for w in getattr(self, "_cc_row_widgets", ()):
            w.setEnabled(on)

    def _sync_pp_enabled(self, on: bool) -> None:
        self._pp.setEnabled(on)

    # ------------------------------------------------------------------

    def _collect_candidates(self) -> list[Path]:
        """Resolve the user's source selection to a list of image paths."""
        if self._rb_dir.isChecked():
            src_dir = self._dir_edit.text().strip()
            if not src_dir or not Path(src_dir).is_dir():
                raise ValueError("Please select a valid input directory.")
            items = [
                p for p in sorted(Path(src_dir).iterdir())
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
            ]
        else:
            items = [
                Path(p) for p in self._selected_files
                if Path(p).is_file() and Path(p).suffix.lower() in _IMAGE_EXTS
            ]
        if not items:
            raise ValueError("No images found in the selected source.")
        return items

    @staticmethod
    def _parse_integer_stems(paths: list[Path]) -> list[tuple[int, Path]]:
        """Map each path to (int_stem, path). Raise if any stem isn't an integer."""
        bad: list[str] = []
        out: list[tuple[int, Path]] = []
        for p in paths:
            stem = p.stem
            try:
                n = int(stem)
            except ValueError:
                bad.append(p.name)
                continue
            out.append((n, p))
        if bad:
            raise ValueError(
                "All files must be named as bare integers. Bad names: "
                + ", ".join(bad[:6])
                + (f" (+{len(bad) - 6} more)" if len(bad) > 6 else "")
            )
        return out

    @staticmethod
    def _check_contiguous(numbered: list[tuple[int, Path]]) -> list[int]:
        """Return the sorted integer list if contiguous; raise otherwise."""
        nums = sorted(n for n, _ in numbered)
        if len(nums) < 2:
            raise ValueError(
                "Need at least two numbered frames to build pairs."
            )
        if len(set(nums)) != len(nums):
            raise ValueError("Duplicate frame numbers detected.")
        expected = list(range(nums[0], nums[0] + len(nums)))
        if nums != expected:
            shown = ",".join(str(n) for n in nums[:10])
            if len(nums) > 10:
                shown += "…"
            raise ValueError(
                f"Files must be contiguous (got {shown}). "
                f"Expected range {nums[0]}..{nums[0] + len(nums) - 1}."
            )
        return nums

    def _on_accept(self) -> None:
        prefix = self._prefix_edit.text().strip()
        if not prefix:
            QMessageBox.warning(self, "Sequential", "Clip File Prefix is required.")
            return

        out_dir = self._out_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "Sequential", "Output directory is required.")
            return
        out_path = Path(out_dir)
        try:
            out_path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.critical(self, "Sequential", f"Cannot create output dir: {exc}")
            return

        try:
            candidates = self._collect_candidates()
            numbered = self._parse_integer_stems(candidates)
            self._check_contiguous(numbered)
        except ValueError as exc:
            QMessageBox.warning(self, "Sequential", str(exc))
            return

        numbered.sort(key=lambda t: t[0])
        nums = [n for n, _ in numbered]
        paths_by_num = {n: str(p) for n, p in numbered}

        # Sliding window of pairs. Each item is ((first_num, first_path),
        # (last_num, last_path)) so the controller has both numbers for
        # the zero-padded filename.
        pairs: list[dict] = []
        for i in range(len(nums) - 1):
            a, b = nums[i], nums[i + 1]
            pairs.append({
                "first_num": a,
                "last_num": b,
                "first_path": paths_by_num[a],
                "last_path": paths_by_num[b],
            })

        do_cc = self._cb_cc.isChecked()
        cc_ref = self._cc_ref_edit.text().strip()
        if do_cc:
            if not cc_ref or not Path(cc_ref).is_file():
                QMessageBox.warning(
                    self, "Sequential",
                    "Color Correct is enabled but the reference file is missing.",
                )
                return
            if Path(cc_ref).suffix.lower() not in _REF_EXTS:
                QMessageBox.warning(
                    self, "Sequential",
                    "Reference must be an image or video.",
                )
                return

        self._result = {
            "pairs": pairs,
            "max_num": nums[-1],
            "output_dir": str(out_path),
            "prefix": prefix,
            "use_guidance": self._cb_guidance.isChecked(),
            "guidance_frame": self._guide_frame.currentText().lower(),  # "first"|"last"
            "do_color_correct": do_cc,
            "color_correct_ref": cc_ref,
            "do_pp": self._cb_pp.isChecked(),
            "pp": self._pp.get_settings(),
            "source_mode": "dir" if self._rb_dir.isChecked() else "files",
            "source_dir": self._dir_edit.text().strip(),
            "source_files": list(self._selected_files),
        }
        self.accept()

    def get_settings(self) -> dict:
        return getattr(self, "_result", {})

    # ------------------------------------------------------------------

    def _restore(self, saved: dict) -> None:
        if not saved:
            return
        mode = saved.get("source_mode", "dir")
        if mode == "files":
            self._rb_files.setChecked(True)
        else:
            self._rb_dir.setChecked(True)
        self._dir_edit.setText(saved.get("source_dir", ""))
        files = saved.get("source_files") or []
        if files:
            self._selected_files = [str(p) for p in files]
            self._files_edit.setText(f"{len(self._selected_files)} file(s) selected")
        self._out_edit.setText(saved.get("output_dir", ""))
        self._prefix_edit.setText(saved.get("prefix", "seq_"))
        self._cb_guidance.setChecked(bool(saved.get("use_guidance", False)))
        gf = str(saved.get("guidance_frame", "first")).lower()
        self._guide_frame.setCurrentIndex(1 if gf == "last" else 0)
        self._sync_guide_enabled(self._cb_guidance.isChecked())
        self._cb_cc.setChecked(bool(saved.get("do_color_correct", False)))
        self._cc_ref_edit.setText(saved.get("color_correct_ref", ""))
        self._sync_cc_enabled(self._cb_cc.isChecked())
        self._cb_pp.setChecked(bool(saved.get("do_pp", False)))
        self._sync_pp_enabled(self._cb_pp.isChecked())
        if "pp" in saved and isinstance(saved["pp"], dict):
            self._pp.restore_settings(saved["pp"])


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line
