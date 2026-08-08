"""Reusable wizard dialog — QDialog + QStackedWidget + nav + worker management."""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from sdqt.workers.base import BaseWorker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WizardPage base
# ---------------------------------------------------------------------------


class WizardPage(QWidget):
    """Base class for a single page inside a SequenceWizard."""

    def on_enter(self) -> None:
        """Called when this page becomes visible."""

    def on_leave(self) -> None:
        """Called when leaving this page."""

    def validate(self) -> bool:
        """Return True if the user may advance past this page."""
        return True

    def can_skip(self) -> bool:
        """If True the wizard auto-skips this page (both forward and back)."""
        return False


# ---------------------------------------------------------------------------
# SequenceWizard base dialog
# ---------------------------------------------------------------------------


class SequenceWizard(QDialog):
    """Multi-page wizard with back/next navigation and worker management.

    Subclasses populate ``self._pages`` before calling ``_finish_setup()``.
    An optional *header_widget* can be provided to sit above the page stack
    (e.g. a model dropdown).
    """

    status_message = Signal(str)

    def __init__(
        self,
        title: str,
        state,
        *,
        header_widget: QWidget | None = None,
        footer_widget: QWidget | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        # Keep the minimum modest so the dialog fits small screens; each page is
        # wrapped in a scroll area (see _finish_setup) so taller content never
        # gets crushed. Open at a roomy default so most pages need no scrolling.
        self.setMinimumSize(760, 560)
        self.resize(900, 820)
        self.setAttribute(Qt.WA_DeleteOnClose)

        self._state = state
        self._pages: list[WizardPage] = []
        self._current_idx: int = 0
        self._worker: BaseWorker | None = None

        # -- layout ---------------------------------------------------------
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(6)

        # Optional header
        if header_widget is not None:
            root.addWidget(header_widget)

        # Page indicator
        self._page_label = QLabel()
        self._page_label.setAlignment(Qt.AlignRight)
        self._page_label.setStyleSheet("color: #aaa; font-size: 12px;")
        root.addWidget(self._page_label)

        # Stacked pages
        self._stack = QStackedWidget()
        root.addWidget(self._stack, 1)

        # Progress bar (hidden by default)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setTextVisible(True)
        self._progress.hide()
        root.addWidget(self._progress)

        # Status label
        self._status = QLabel()
        self._status.setStyleSheet("color: #aaa; font-size: 11px;")
        root.addWidget(self._status)

        # Optional persistent footer (e.g. generation settings shown on every page)
        if footer_widget is not None:
            root.addWidget(footer_widget)

        # Navigation buttons
        nav = QHBoxLayout()
        nav.addStretch()
        self._back_btn = QPushButton("Back")
        self._back_btn.setFixedWidth(90)
        self._back_btn.clicked.connect(self._go_back)
        nav.addWidget(self._back_btn)

        self._next_btn = QPushButton("Next")
        self._next_btn.setFixedWidth(90)
        self._next_btn.setDefault(True)
        self._next_btn.clicked.connect(self._go_next)
        nav.addWidget(self._next_btn)
        root.addLayout(nav)

    # -- Setup (call after populating self._pages) --------------------------

    def _finish_setup(self) -> None:
        """Register all pages in the stack and show the first one.

        Every page is wrapped in a scroll area so content that exceeds the
        dialog height scrolls instead of being squashed below its natural size.
        Pages that already manage their own inner scroll are unaffected — the
        page simply fills the resizable viewport and its inner scroll does the
        work.
        """
        for page in self._pages:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll.setWidget(page)
            self._stack.addWidget(scroll)
        self._current_idx = 0
        self._show_page(0)

    # -- Navigation ---------------------------------------------------------

    def _show_page(self, idx: int) -> None:
        # Skip pages that declare can_skip
        direction = 1 if idx >= self._current_idx else -1
        while 0 <= idx < len(self._pages) and self._pages[idx].can_skip():
            idx += direction
        idx = max(0, min(idx, len(self._pages) - 1))

        if 0 <= self._current_idx < len(self._pages):
            self._pages[self._current_idx].on_leave()

        self._current_idx = idx
        self._stack.setCurrentIndex(idx)
        self._page_label.setText(f"Page {idx + 1} of {len(self._pages)}")
        self._back_btn.setEnabled(idx > 0)
        is_last = idx == len(self._pages) - 1
        self._next_btn.setText("Finish" if is_last else "Next")
        self._pages[idx].on_enter()

    @Slot()
    def _go_back(self) -> None:
        if self._current_idx > 0:
            self._show_page(self._current_idx - 1)

    @Slot()
    def _go_next(self) -> None:
        page = self._pages[self._current_idx]
        if not page.validate():
            return
        if self._current_idx < len(self._pages) - 1:
            self._show_page(self._current_idx + 1)
        else:
            self._on_finish()

    def _on_finish(self) -> None:
        """Override in subclass for final save logic."""
        self.accept()

    # -- Worker management --------------------------------------------------

    def _run_worker(
        self,
        worker: BaseWorker,
        on_done=None,
        on_error=None,
    ) -> None:
        """Run *worker* on a background thread, showing progress."""
        self._worker = worker
        self.set_busy(True)
        self._progress.setValue(0)
        self._progress.show()

        worker.progress.connect(self._on_worker_progress)
        worker.status.connect(self._on_worker_status)
        if on_done:
            worker.finished_ok.connect(on_done)
        if on_error:
            worker.error.connect(on_error)
        else:
            worker.error.connect(self._on_worker_error_default)
        worker.finished_ok.connect(lambda _: self.set_busy(False))
        worker.error.connect(lambda _: self.set_busy(False))
        worker.aborted.connect(lambda: self.set_busy(False))
        worker.start()

    @Slot(float, str)
    def _on_worker_progress(self, frac: float, desc: str) -> None:
        self._progress.setValue(int(frac * 100))
        self._status.setText(desc)

    @Slot(str)
    def _on_worker_status(self, msg: str) -> None:
        self._status.setText(msg)

    @Slot(str)
    def _on_worker_error_default(self, msg: str) -> None:
        self._progress.hide()
        self._status.setText(f"Error: {msg}")

    def set_busy(self, busy: bool) -> None:
        """Disable/enable navigation while async work runs."""
        self._back_btn.setEnabled(not busy and self._current_idx > 0)
        self._next_btn.setEnabled(not busy)
        if not busy:
            self._progress.hide()

    # -- Close guard --------------------------------------------------------

    def closeEvent(self, event) -> None:
        if self._worker and self._worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Abort?",
                "A generation is running. Abort and close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            self._worker.abort()
            self._worker.wait(3000)
        super().closeEvent(event)
