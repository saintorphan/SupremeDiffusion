"""Shared constants and styles for Draw / Inpaint toolbars."""

from PySide6.QtGui import QColor

TOOL_SZ = 32  # square tool-button size (px)

TOOL_BTN_STYLE = (
    f"QToolButton {{ border: 1px solid #555; padding: 0px;"
    f" color: #ccc; background: #333; border-radius: 3px;"
    f" min-width: {TOOL_SZ}px; max-width: {TOOL_SZ}px;"
    f" min-height: {TOOL_SZ}px; max-height: {TOOL_SZ}px; }} "
    "QToolButton:hover { color: #fff; background: #555; } "
    "QToolButton:checked { color: #fff; background: #0078d4; border-color: #0078d4; }"
)

BTN_STYLE = (
    "QPushButton { border: 1px solid #555; padding: 4px 10px; font-size: 13px;"
    " color: #ccc; background: #333; border-radius: 3px; } "
    "QPushButton:hover { color: #fff; background: #555; }"
)

MASK_BTN_STYLE = (
    "QPushButton { border: 1px solid #555; padding: 4px 10px; font-size: 13px;"
    " color: #ff8888; background: #333; border-radius: 3px; font-weight: bold; } "
    "QPushButton:hover { color: #fff; background: #663333; }"
)

ROW_LABEL_STYLE = (
    "font-size: 13px; font-weight: bold; color: #999;"
    " padding: 0 6px 0 2px; min-width: 42px;"
)

TOGGLE_STYLE = (
    "QToolButton { border: 1px solid #555; padding: 4px 10px; font-size: 13px;"
    " color: #ccc; background: #333; border-radius: 3px; } "
    "QToolButton:hover { color: #fff; background: #555; } "
    "QToolButton:checked { color: #fff; background: #0078d4; border-color: #0078d4; }"
)

SEND_BTN_STYLE = (
    "QPushButton { border: 1px solid #555; padding: 4px 10px; font-size: 13px;"
    " color: #fff; background: #0078d4; border-radius: 3px; font-weight: bold; } "
    "QPushButton:hover { background: #005a9e; }"
)

MASK_COLOR = QColor(255, 0, 0, 128)

PRESET_COLORS = [
    "#000000", "#ffffff", "#ff0000", "#00cc00", "#0000ff", "#ffff00",
    "#ff8800", "#8800ff", "#ff66aa", "#00cccc", "#885522", "#888888",
]

MAX_RECENT_COLORS = 8
