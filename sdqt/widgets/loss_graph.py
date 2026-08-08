"""Loss curve graph widget for LoRA training visualization."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen, QFont
from PySide6.QtWidgets import QWidget


class LossGraphWidget(QWidget):
    """Custom-painted loss curve graph.

    Call add_point(step, loss) to append data. The graph auto-scales.
    Shows raw values as dots and a rolling average as a line.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._points: list[tuple[int, float]] = []  # (step, loss)
        self._avg_window = 20
        self.setMinimumHeight(120)
        self.setStyleSheet("background: #1a1a1a; border: 1px solid #333;")

    def add_point(self, step: int, loss: float) -> None:
        self._points.append((step, loss))
        self.update()

    def clear(self) -> None:
        self._points.clear()
        self.update()

    def paintEvent(self, event) -> None:
        if not self._points:
            p = QPainter(self)
            p.setPen(QColor(100, 100, 100))
            p.setFont(QFont("sans-serif", 11))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Loss graph — waiting for data...")
            p.end()
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        margin_l, margin_r, margin_t, margin_b = 50, 10, 10, 25

        plot_w = w - margin_l - margin_r
        plot_h = h - margin_t - margin_b

        # Data range
        steps = [pt[0] for pt in self._points]
        losses = [pt[1] for pt in self._points]
        min_step, max_step = min(steps), max(steps)
        min_loss = min(losses) * 0.9
        max_loss = max(losses) * 1.1
        if max_step == min_step:
            max_step = min_step + 1
        if max_loss == min_loss:
            max_loss = min_loss + 0.01

        def to_px(step: float, loss: float) -> tuple[int, int]:
            x = margin_l + int((step - min_step) / (max_step - min_step) * plot_w)
            y = margin_t + int((1.0 - (loss - min_loss) / (max_loss - min_loss)) * plot_h)
            return x, y

        # Background
        p.fillRect(margin_l, margin_t, plot_w, plot_h, QColor(25, 25, 30))

        # Grid lines
        p.setPen(QPen(QColor(50, 50, 55), 1))
        for i in range(5):
            y = margin_t + int(i / 4 * plot_h)
            p.drawLine(margin_l, y, w - margin_r, y)

        # Axis labels
        p.setPen(QColor(120, 120, 120))
        p.setFont(QFont("sans-serif", 9))
        for i in range(5):
            loss_val = max_loss - i / 4 * (max_loss - min_loss)
            y = margin_t + int(i / 4 * plot_h)
            p.drawText(2, y + 4, f"{loss_val:.4f}")

        # Step labels
        for i in range(5):
            step_val = min_step + i / 4 * (max_step - min_step)
            x = margin_l + int(i / 4 * plot_w)
            p.drawText(x - 15, h - 3, f"{int(step_val)}")

        # Raw data points (small dots)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(80, 140, 220, 100))
        for step, loss in self._points:
            x, y = to_px(step, loss)
            p.drawEllipse(x - 1, y - 1, 3, 3)

        # Rolling average line
        if len(self._points) > 1:
            avg_pts = self._rolling_average()
            p.setPen(QPen(QColor(255, 160, 50), 2))
            for i in range(1, len(avg_pts)):
                x1, y1 = to_px(*avg_pts[i - 1])
                x2, y2 = to_px(*avg_pts[i])
                p.drawLine(x1, y1, x2, y2)

        # Current value label
        if self._points:
            last_step, last_loss = self._points[-1]
            p.setPen(QColor(220, 220, 220))
            p.setFont(QFont("sans-serif", 10))
            p.drawText(margin_l + 5, margin_t + 15, f"Loss: {last_loss:.5f}  Step: {last_step}")

        p.end()

    def _rolling_average(self) -> list[tuple[int, float]]:
        """Compute rolling average of loss values."""
        result = []
        window = self._avg_window
        for i in range(len(self._points)):
            start = max(0, i - window + 1)
            subset = self._points[start:i + 1]
            avg_loss = sum(pt[1] for pt in subset) / len(subset)
            result.append((self._points[i][0], avg_loss))
        return result
