"""Shared canvas keyboard shortcuts for drawing/editing tabs."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut


def setup_canvas_shortcuts(
    widget,
    canvas,
    *,
    paste_callback=None,
    extra: list[tuple[str, object]] | None = None,
) -> list[QShortcut]:
    """Wire up standard canvas keyboard shortcuts on *widget*.

    Parameters
    ----------
    widget : QWidget
        The parent widget that owns the shortcuts.
    canvas : DrawCanvas
        The canvas instance with undo/redo/copy/cut/delete/paste/selection.
    paste_callback : callable, optional
        Custom paste handler. Defaults to ``canvas.paste_clipboard``.
    extra : list of (key_sequence_str, callable), optional
        Additional shortcuts to register (e.g. Shift+F for fill-to-mask).

    Returns
    -------
    list[QShortcut]
        The created shortcuts (kept alive by parent widget).
    """
    _ctx = Qt.ShortcutContext.WidgetWithChildrenShortcut
    paste = paste_callback or canvas.paste_clipboard

    shortcuts = [
        QShortcut(QKeySequence("Ctrl+Z"), widget, canvas.undo, context=_ctx),
        QShortcut(QKeySequence("Ctrl+Y"), widget, canvas.redo, context=_ctx),
        QShortcut(QKeySequence("Ctrl+Shift+Z"), widget, canvas.redo, context=_ctx),
        QShortcut(QKeySequence("Ctrl+C"), widget, canvas.copy_selection, context=_ctx),
        QShortcut(QKeySequence("Ctrl+X"), widget, canvas.cut_selection, context=_ctx),
        QShortcut(QKeySequence("Ctrl+V"), widget, paste, context=_ctx),
        QShortcut(QKeySequence("Delete"), widget, canvas.delete_selection, context=_ctx),
        QShortcut(
            QKeySequence("Ctrl+A"), widget,
            lambda: canvas.selection.select_all(canvas.scene().sceneRect()),
            context=_ctx,
        ),
        QShortcut(QKeySequence("Ctrl+D"), widget, canvas.selection.clear, context=_ctx),
        QShortcut(
            QKeySequence("Ctrl+Shift+I"), widget,
            lambda: canvas.selection.invert(canvas.scene().sceneRect()),
            context=_ctx,
        ),
        QShortcut(QKeySequence("M"), widget, canvas.toggle_flip_view, context=_ctx),
        QShortcut(
            QKeySequence("Ctrl+Shift+A"), widget,
            lambda: (canvas.selection.select_opaque(canvas.active_layer)
                     if canvas.active_layer else None),
            context=_ctx,
        ),
    ]

    for key_seq, callback in (extra or []):
        shortcuts.append(QShortcut(QKeySequence(key_seq), widget, callback, context=_ctx))

    return shortcuts
