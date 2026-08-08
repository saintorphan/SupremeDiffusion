"""Orchestrator — check for models and prompt download if missing."""

from __future__ import annotations

import logging
from typing import Callable, Optional

from PySide6.QtWidgets import QWidget

from .registry import ModelRegistry

logger = logging.getLogger(__name__)


def check_and_prompt_download(
    feature: str,
    registry: ModelRegistry,
    parent: QWidget,
    *,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> bool:
    """Check if models for *feature* are present; prompt to download if not.

    Args:
        feature: Feature key (e.g. "video_gen", "lipsync").
        registry: The app's ModelRegistry instance.
        parent: Parent widget for the dialog.
        on_progress: Optional callback ``(fraction, description)`` for download progress.

    Returns:
        True if all models are present (or were successfully downloaded).
        False if the user cancelled or download failed.
    """
    missing = registry.get_missing_models(feature)
    if not missing:
        return True

    fm = registry.get_feature_models(feature)
    label = fm.label if fm else feature

    # Show confirmation dialog
    from .download_dialog import ModelDownloadDialog
    dialog = ModelDownloadDialog(label, missing, parent=parent)
    if dialog.exec() != ModelDownloadDialog.Accepted:
        return False

    # Run download in a blocking event loop (worker thread + local event processing)
    from .download_worker import ModelDownloadWorker
    from PySide6.QtCore import QEventLoop

    worker = ModelDownloadWorker(
        models=missing,
        models_root=registry.models_root,
        parent=parent,
    )

    result_holder: dict = {"ok": False, "results": [], "error": ""}

    def _on_done(results):
        result_holder["ok"] = True
        result_holder["results"] = results

    def _on_error(msg):
        result_holder["error"] = msg

    def _on_progress(f: float, d: str) -> None:
        logger.info("[Download] %s", d)
        # Update the main window status bar if reachable
        _update_global_status(parent, d)
        if on_progress:
            on_progress(f, d)

    loop = QEventLoop()
    worker.finished_ok.connect(_on_done)
    worker.error.connect(_on_error)
    worker.finished.connect(loop.quit)
    worker.progress.connect(_on_progress)
    worker.status.connect(lambda s: logger.info("[Download] %s", s))

    worker.start()
    loop.exec()

    if not result_holder["ok"]:
        err = result_holder["error"]
        logger.error("Model download failed: %s", err)
        if on_progress:
            on_progress(0.0, f"Download failed: {err}")
        return False

    # Update config paths for downloaded models and persist
    _sync_config_paths(registry, result_holder["results"])
    try:
        from sdqt.state import AppState
        # Find the AppState instance through the parent widget chain
        widget = parent
        while widget is not None:
            if hasattr(widget, "state") and isinstance(widget.state, AppState):
                widget.state.sync_config_after_download()
                break
            widget = widget.parent() if hasattr(widget, "parent") else None
    except Exception:
        logger.debug("Could not auto-save config after download", exc_info=True)

    if on_progress:
        on_progress(1.0, "Models ready.")
    return True


def _update_global_status(widget: QWidget, message: str) -> None:
    """Walk up the widget tree to find the MainWindow and update its status bar."""
    try:
        w = widget
        while w is not None:
            # MainWindow has _status_bar and _pipeline_label
            if hasattr(w, "_status_bar") and hasattr(w, "_pipeline_label"):
                w._status_bar.showMessage(message, 10000)
                return
            w = w.parent() if hasattr(w, "parent") else None
    except Exception:
        pass


def _sync_config_paths(
    registry: ModelRegistry,
    results: list[tuple],
) -> None:
    """Update GlobalConfig.model_paths with newly downloaded model locations."""
    from sdqt.models.registry import ModelEntry

    for entry, local_path in results:
        if not isinstance(entry, ModelEntry):
            continue
        if entry.config_key:
            registry._config_paths[entry.config_key] = local_path
            logger.info(
                "Config updated: %s = %s", entry.config_key, local_path,
            )
