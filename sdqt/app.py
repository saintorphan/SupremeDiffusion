"""Application entry point."""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QComboBox, QMessageBox

from sdqt.main_window import MainWindow


class _ScrollSafeApp(QApplication):
    """QApplication subclass that blocks scroll-wheel on unfocused spinboxes/combos.

    Overrides notify() to intercept wheel events before they reach any widget.
    This is more reliable than an event filter, which only works on the
    specific objects it's installed on.
    """

    def notify(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            if isinstance(obj, (QAbstractSpinBox, QComboBox)):
                if obj.focusPolicy() != Qt.FocusPolicy.StrongFocus:
                    obj.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
                if not obj.hasFocus():
                    # Pass the event to the parent so scroll areas still work
                    parent = obj.parent()
                    if parent:
                        return super().notify(parent, event)
                    return True
        return super().notify(obj, event)

_APP_ROOT = Path(__file__).resolve().parent.parent
_LOCK_FILE = _APP_ROOT / ".sdqt.lock"

def _find_backend_venvs() -> list[Path]:
    """Discover backend venv python paths relative to common install locations."""
    if os.name == "nt":
        py = "Scripts" + os.sep + "python.exe"
    else:
        py = "bin" + os.sep + "python"
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "venv" / py,
        root / ".venv" / py,
        Path("~/LatentSync/venv").expanduser() / py,
        Path("~/Wan2GP/venv").expanduser() / py,
        Path("~/MuseTalk/venv").expanduser() / py,
    ]
    return [p for p in candidates if p.is_file()]

_BACKEND_VENVS = _find_backend_venvs()


def _kill_orphan_gpu_processes() -> None:
    """Kill any leftover backend venv processes sitting on the GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",", 1)]
            if len(parts) != 2:
                continue
            pid_str, name = parts
            # Kill processes from known backend venvs (but not ourselves)
            for venv_python in _BACKEND_VENVS:
                if str(venv_python) in name:
                    pid = int(pid_str)
                    if pid != os.getpid():
                        try:
                            if os.name == "nt":
                                os.kill(pid, signal.SIGBREAK)
                            else:
                                os.kill(pid, signal.SIGTERM)
                            logging.getLogger(__name__).info(
                                "Killed orphan GPU process %d (%s)", pid, name,
                            )
                        except OSError:
                            pass
                    break
    except Exception:
        pass  # nvidia-smi not available or other error — non-fatal


def _acquire_lock() -> bool:
    """Try to acquire a file lock. Returns False if another instance is running."""
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _LOCK_FILE.exists():
        try:
            old_pid = int(_LOCK_FILE.read_text().strip())
            # Check if that process is still alive
            if os.name == "nt":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x100000, False, old_pid)  # SYNCHRONIZE
                if handle:
                    kernel32.CloseHandle(handle)
                    return False  # process is alive
                # handle == 0 means process doesn't exist
            else:
                os.kill(old_pid, 0)
                return False  # process is alive — another instance running
        except (ValueError, OSError):
            # Stale lock file (process dead) — safe to take over
            pass

    _LOCK_FILE.write_text(str(os.getpid()))
    return True


def _release_lock() -> None:
    """Remove the lock file on exit."""
    try:
        if _LOCK_FILE.exists():
            pid = int(_LOCK_FILE.read_text().strip())
            if pid == os.getpid():
                _LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _setup_logging() -> None:
    """Configure logging with colorized terminal output and a rotating log file.

    The file log ensures the app keeps running even if the CMD window is closed.
    Terminal colors: ERROR=red, WARNING=yellow, SUCCESS keywords=green, INFO=default.
    """
    import sys
    from pathlib import Path

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    _LOG_FMT = "%(asctime)s  %(name)-30s  %(levelname)-8s  %(message)s"
    _LOG_DATEFMT = "%H:%M:%S"

    # ── File handler (always active, survives CMD close) ──
    try:
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            str(log_dir / "supremediffusion.log"),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_LOG_DATEFMT))
        root.addHandler(file_handler)
    except Exception:
        pass  # file logging is best-effort

    # ── Terminal handler with ANSI colors ──
    class _ColorTerminalHandler(logging.StreamHandler):
        """StreamHandler that adds ANSI color codes by log level."""

        # ANSI escape codes
        RESET = "\033[0m"
        BOLD = "\033[1m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        GREEN = "\033[92m"
        CYAN = "\033[96m"
        DIM = "\033[90m"

        _LEVEL_COLORS = {
            logging.DEBUG:    DIM,
            logging.INFO:     "",           # default terminal color
            logging.WARNING:  BOLD + YELLOW,
            logging.ERROR:    BOLD + RED,
            logging.CRITICAL: BOLD + RED,
        }

        # Keywords that should appear green
        _SUCCESS_WORDS = {
            "success", "complete", "completed", "generated", "done",
            "finished", "saved", "loaded", "ready",
        }

        def emit(self, record: logging.LogRecord) -> None:
            try:
                msg = self.format(record)
                color = self._LEVEL_COLORS.get(record.levelno, "")

                # Override: success keywords get green
                msg_lower = msg.lower()
                if any(w in msg_lower for w in self._SUCCESS_WORDS):
                    color = self.BOLD + self.GREEN

                if color:
                    msg = f"{color}{msg}{self.RESET}"

                stream = self.stream
                stream.write(msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)

    # Only add terminal handler if stdout is available (not detached)
    try:
        if sys.stdout is not None and hasattr(sys.stdout, "write"):
            term_handler = _ColorTerminalHandler(sys.stdout)
            term_handler.setLevel(logging.DEBUG)
            term_handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_LOG_DATEFMT))
            root.addHandler(term_handler)
    except Exception:
        # If terminal is gone (e.g., CMD closed), file handler keeps working
        pass

    # Suppress verbose third-party libraries
    logging.getLogger("PIL").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("bitsandbytes").setLevel(logging.WARNING)


def main() -> None:
    _setup_logging()

    logger = logging.getLogger(__name__)

    # Kill orphan GPU processes from previous runs
    _kill_orphan_gpu_processes()

    # Single-instance check
    if not _acquire_lock():
        # Need a QApplication to show the dialog
        _app = QApplication(sys.argv)
        QMessageBox.warning(
            None, "Already Running",
            "Supreme Diffusion is already running.\n"
            "Only one instance is allowed at a time.",
        )
        sys.exit(1)

    atexit.register(_release_lock)

    app = _ScrollSafeApp(sys.argv)
    app.setApplicationName("Supreme Diffusion")
    app.setDesktopFileName("supreme-diffusion-qt")
    app.setStyle("Fusion")

    # Global font size increase
    from PySide6.QtGui import QColor, QFont, QPalette
    font = app.font()
    font.setPointSize(12)
    app.setFont(font)

    # Global stylesheet: base sizes + card-style containers.
    # Design system (2026-05): comfortable 15px scale, 13px floor (no text
    # smaller than 13px anywhere — de-emphasize with COLOR, not size), 32px
    # input/button heights, and a prominent #primary accent button class for
    # the main action on each page (Generate/Render/Export/Save/Analyze/Apply/
    # Trim/Crop/Grab/Swap). Set btn.setObjectName("primary") to opt in.
    app.setStyleSheet("""
        /* ── Base font sizes (15px primary, 13px floor) ── */
        QLabel { font-size: 15px; }
        QPushButton { font-size: 15px; min-height: 32px; padding: 4px 14px; }
        QToolButton { font-size: 15px; min-height: 30px; }
        QComboBox { font-size: 15px; min-height: 32px; padding: 2px 8px; }
        QSpinBox, QDoubleSpinBox { font-size: 15px; min-height: 32px; padding: 2px 6px; }
        QCheckBox { font-size: 15px; spacing: 8px; }
        QRadioButton { font-size: 15px; spacing: 8px; }
        QLineEdit { font-size: 15px; min-height: 32px; padding: 2px 8px; }
        QPlainTextEdit { font-size: 15px; }
        QTextEdit { font-size: 15px; }
        QSlider { min-height: 28px; }
        QListWidget { font-size: 15px; }
        QTreeWidget { font-size: 15px; }
        QStatusBar { font-size: 13px; }
        QMenu { font-size: 15px; }
        QMenuBar { font-size: 15px; }

        /* ── Primary action button (opt-in via objectName "primary") ── */
        QPushButton#primary {
            min-height: 44px;
            font-size: 16px;
            font-weight: bold;
            padding: 6px 20px;
            color: #ffffff;
            background: #0a84d4;
            border: 1px solid #0a84d4;
            border-radius: 6px;
        }
        QPushButton#primary:hover { background: #1f97e0; border-color: #1f97e0; }
        QPushButton#primary:pressed { background: #0769ab; }
        QPushButton#primary:disabled { background: #3a4654; color: #8a96a4; border-color: #3a4654; }

        /* ── Card-style QGroupBox ── */
        QGroupBox {
            font-size: 14px;
            font-weight: bold;
            color: #9dbcd4;
            background: #1e1e1e;
            border: 1px solid #333;
            border-radius: 6px;
            margin-top: 10px;
            padding: 16px 10px 10px 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 2px 10px;
            background: #252525;
            border: 1px solid #333;
            border-radius: 4px;
            left: 10px;
        }

        /* ── Tabs (legible labels; reflow + scroll when many tabs) ── */
        QTabWidget::pane {
            border: 1px solid #333;
            border-radius: 4px;
            background: #1a1a1a;
            top: -1px;
        }
        QTabBar::tab {
            font-size: 14px;
            min-width: 96px;
            padding: 8px 16px;
            background: #222;
            border: 1px solid #333;
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            margin-right: 2px;
            color: #888;
        }
        QTabBar::tab:selected {
            background: #1a1a1a;
            color: #ddd;
            border-bottom: 2px solid #0a84d4;
        }
        QTabBar::tab:hover:!selected {
            background: #2a2a2a;
            color: #bbb;
        }
        QTabBar::scroller { width: 24px; }

        /* ── Scroll areas (visible scrollbar affordance) ── */
        QScrollArea { border: none; }
        QScrollBar:vertical {
            width: 12px;
            background: transparent;
            border: none;
        }
        QScrollBar::handle:vertical {
            background: #555;
            border-radius: 6px;
            min-height: 28px;
        }
        QScrollBar::handle:vertical:hover { background: #777; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QScrollBar:horizontal {
            height: 12px;
            background: transparent;
            border: none;
        }
        QScrollBar::handle:horizontal {
            background: #555;
            border-radius: 6px;
            min-width: 28px;
        }
        QScrollBar::handle:horizontal:hover { background: #777; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

        /* ── Tooltips ── */
        QToolTip {
            font-size: 13px;
            background: #2a2a2a;
            color: #ddd;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 6px 8px;
        }

        /* ── Splitter handles ── */
        QSplitter::handle {
            background: #333;
            border-radius: 2px;
        }
        QSplitter::handle:hover { background: #0a84d4; }
    """)

    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(40, 40, 40))
    palette.setColor(QPalette.ToolTipBase, QColor(50, 50, 50))
    palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
    palette.setColor(QPalette.Link, QColor(100, 149, 237))
    palette.setColor(QPalette.Highlight, QColor(70, 130, 220))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(100, 100, 100))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(100, 100, 100))
    app.setPalette(palette)

    # Choose UI layout based on config
    from supremediffusion.config.global_config import GlobalConfig
    _cfg = GlobalConfig.load()
    if _cfg.ui_layout == "modern":
        from sdqt.main_window_v2 import MainWindowV2
        window = MainWindowV2()
    else:
        window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
