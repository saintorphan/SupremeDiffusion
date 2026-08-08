"""Image adjustment filters for the Draw editor."""

from __future__ import annotations

from PySide6.QtGui import QColor, QImage, QPainter


def adjust_brightness_contrast(image: QImage, brightness: int, contrast: int) -> QImage:
    """Apply brightness (-255..255) and contrast (-255..255) to an ARGB32 image.

    Uses raw pixel access via constBits/bits for performance.
    """
    img = image.convertToFormat(QImage.Format.Format_ARGB32)
    w, h = img.width(), img.height()
    ptr = img.bits()
    if ptr is None:
        return img
    bpl = img.bytesPerLine()
    data = bytearray(bytes(ptr))

    # Contrast factor: map -255..255 to 0..~4x multiplier
    if contrast != 0:
        f = (259.0 * (contrast + 255)) / (255.0 * (259 - contrast))
    else:
        f = 1.0

    for y in range(h):
        row_off = y * bpl
        for x in range(w):
            off = row_off + x * 4
            # BGRA layout on little-endian
            for c in range(3):  # B, G, R channels (skip A at index 3)
                val = data[off + c]
                # Apply contrast
                val = int(f * (val - 128) + 128)
                # Apply brightness
                val = val + brightness
                data[off + c] = max(0, min(255, val))

    result = QImage(bytes(data), w, h, bpl, QImage.Format.Format_ARGB32).copy()
    return result


def adjust_hue_saturation(image: QImage, hue_shift: int, saturation: int) -> QImage:
    """Apply hue shift (-180..180 degrees) and saturation (-100..100) to an ARGB32 image.

    Uses QColor HSL conversion per pixel.
    """
    img = image.convertToFormat(QImage.Format.Format_ARGB32)
    w, h = img.width(), img.height()
    ptr = img.bits()
    if ptr is None:
        return img
    bpl = img.bytesPerLine()
    data = bytearray(bytes(ptr))

    sat_factor = 1.0 + saturation / 100.0

    for y in range(h):
        row_off = y * bpl
        for x in range(w):
            off = row_off + x * 4
            b, g, r, a = data[off], data[off + 1], data[off + 2], data[off + 3]
            if a == 0:
                continue
            c = QColor(r, g, b, a)
            h_val, s_val, l_val, _ = c.getHslF()
            # Shift hue
            h_val = (h_val + hue_shift / 360.0) % 1.0
            if h_val < 0:
                h_val += 1.0
            # Adjust saturation
            s_val = max(0.0, min(1.0, s_val * sat_factor))
            c.setHslF(h_val, s_val, l_val, a / 255.0)
            data[off] = c.blue()
            data[off + 1] = c.green()
            data[off + 2] = c.red()

    result = QImage(bytes(data), w, h, bpl, QImage.Format.Format_ARGB32).copy()
    return result


def invert_colors(image: QImage) -> QImage:
    """Invert RGB channels (preserve alpha)."""
    img = image.convertToFormat(QImage.Format.Format_ARGB32)
    w, h = img.width(), img.height()
    ptr = img.bits()
    if ptr is None:
        return img
    bpl = img.bytesPerLine()
    data = bytearray(bytes(ptr))

    for y in range(h):
        row_off = y * bpl
        for x in range(w):
            off = row_off + x * 4
            data[off] = 255 - data[off]          # B
            data[off + 1] = 255 - data[off + 1]  # G
            data[off + 2] = 255 - data[off + 2]  # R
            # Alpha unchanged

    result = QImage(bytes(data), w, h, bpl, QImage.Format.Format_ARGB32).copy()
    return result


def desaturate(image: QImage) -> QImage:
    """Convert to greyscale (preserve alpha)."""
    img = image.convertToFormat(QImage.Format.Format_ARGB32)
    w, h = img.width(), img.height()
    ptr = img.bits()
    if ptr is None:
        return img
    bpl = img.bytesPerLine()
    data = bytearray(bytes(ptr))

    for y in range(h):
        row_off = y * bpl
        for x in range(w):
            off = row_off + x * 4
            b, g, r = data[off], data[off + 1], data[off + 2]
            grey = int(0.299 * r + 0.587 * g + 0.114 * b)
            data[off] = grey
            data[off + 1] = grey
            data[off + 2] = grey

    result = QImage(bytes(data), w, h, bpl, QImage.Format.Format_ARGB32).copy()
    return result
