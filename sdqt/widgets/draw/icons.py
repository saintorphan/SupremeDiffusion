"""Shared tool icon painter for Draw / Inpaint toolbars."""

from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap


def make_tool_icon(key: str, size: int = 24) -> QPixmap:
    """Paint a simple tool icon onto a pixmap."""
    pm = QPixmap(size, size)
    pm.fill(QColor(0, 0, 0, 0))
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    c = QColor(210, 210, 210)
    pen = QPen(c, 1.5)
    p.setPen(pen)

    if key == "brush":
        p.setBrush(c)
        p.drawEllipse(10, 10, 7, 7)
        p.drawLine(5, 5, 10, 10)
    elif key == "eraser":
        p.setBrush(QColor(180, 100, 100))
        p.drawRoundedRect(4, 6, 12, 9, 2, 2)
        p.setPen(QPen(QColor(210, 210, 210), 1))
        p.drawLine(4, 10, 16, 10)
    elif key == "grab":
        p.drawLine(6, 10, 6, 5)
        p.drawLine(8, 9, 8, 3)
        p.drawLine(10, 9, 10, 2)
        p.drawLine(12, 9, 12, 4)
        p.drawLine(14, 10, 14, 6)
        p.drawArc(5, 9, 10, 8, 0, 180 * 16)
    elif key == "line":
        p.setPen(QPen(c, 2))
        p.drawLine(3, 17, 17, 3)
    elif key == "ellipse":
        p.setBrush(QColor(0, 0, 0, 0))
        p.drawEllipse(3, 5, 14, 10)
    elif key == "rect":
        p.setBrush(QColor(0, 0, 0, 0))
        p.drawRect(3, 4, 14, 12)
    elif key == "freehand":
        path = QPainterPath()
        path.moveTo(4, 10)
        path.cubicTo(4, 3, 10, 2, 14, 6)
        path.cubicTo(18, 10, 14, 17, 10, 16)
        path.cubicTo(6, 15, 4, 13, 4, 10)
        p.setPen(QPen(c, 1.2, Qt.PenStyle.DashLine))
        p.drawPath(path)
    elif key == "lasso":
        path = QPainterPath()
        path.moveTo(10, 2)
        path.lineTo(18, 16)
        path.lineTo(2, 16)
        path.closeSubpath()
        p.setPen(QPen(c, 1.2, Qt.PenStyle.DashLine))
        p.drawPath(path)
    elif key == "magnetic":
        p.setPen(QPen(QColor(0, 200, 200), 1.5))
        p.drawArc(5, 2, 10, 10, 0, 180 * 16)
        p.setPen(QPen(QColor(200, 60, 60), 1.5))
        p.drawLine(5, 7, 5, 16)
        p.setPen(QPen(QColor(60, 60, 200), 1.5))
        p.drawLine(15, 7, 15, 16)
    elif key == "wand":
        p.setPen(QPen(QColor(255, 220, 80), 2))
        p.drawLine(3, 17, 10, 10)
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            x1 = 12 + 2 * math.cos(rad)
            y1 = 8 + 2 * math.sin(rad)
            x2 = 12 + 5 * math.cos(rad)
            y2 = 8 + 5 * math.sin(rad)
            p.drawLine(int(x1), int(y1), int(x2), int(y2))
    elif key == "transform":
        p.drawRect(5, 5, 10, 10)
        p.setPen(QPen(c, 1.2))
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            cx = 10 + dx * 5
            cy = 10 + dy * 5
            p.drawLine(cx, cy, cx + dx * 3, cy)
            p.drawLine(cx, cy, cx, cy + dy * 3)
    elif key == "eyedropper":
        p.setPen(QPen(c, 1.5))
        p.drawLine(6, 14, 12, 8)
        p.setBrush(c)
        p.drawEllipse(11, 3, 6, 6)
    elif key == "fill":
        p.setPen(QPen(c, 1.5))
        p.setBrush(QColor(100, 150, 255))
        p.drawRect(4, 8, 10, 8)
        p.setBrush(QColor(0, 0, 0, 0))
        p.drawArc(8, 2, 8, 10, 0, 180 * 16)
    elif key == "smudge":
        # Water drop shape
        p.setPen(QPen(QColor(150, 180, 255), 1.5))
        p.setBrush(QColor(150, 180, 255, 120))
        path = QPainterPath()
        path.moveTo(10, 3)
        path.cubicTo(10, 3, 16, 10, 16, 14)
        path.cubicTo(16, 18, 4, 18, 4, 14)
        path.cubicTo(4, 10, 10, 3, 10, 3)
        p.drawPath(path)
    elif key == "text":
        # Letter A
        p.setPen(QPen(c, 2))
        p.drawLine(6, 18, 10, 4)
        p.drawLine(10, 4, 14, 18)
        p.drawLine(7, 14, 13, 14)
    elif key == "gradient":
        # Gradient bar
        from PySide6.QtGui import QLinearGradient, QBrush as _QBrush
        grad = QLinearGradient(4, 10, 16, 10)
        grad.setColorAt(0, QColor(255, 255, 255))
        grad.setColorAt(1, QColor(60, 60, 60))
        p.setPen(QPen(c, 1))
        p.setBrush(_QBrush(grad))
        p.drawRoundedRect(4, 6, 12, 8, 2, 2)
    else:
        p.drawText(3, 15, "?")

    p.end()
    return pm
