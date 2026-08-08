""".sdraw file format -- ZIP containing layer PNGs + manifest.json."""

from __future__ import annotations

import json
import logging
import tempfile
import zipfile
from pathlib import Path

from PySide6.QtGui import QColor, QImage

from sdqt.widgets.draw.layer_model import Layer, LayerStack

logger = logging.getLogger(__name__)


def save_sdraw(
    path: str,
    layer_stack: LayerStack,
    active_uid: str | None = None,
    canvas_size: tuple[int, int] | None = None,
) -> None:
    """Save the layer stack to a .sdraw file (ZIP archive)."""
    if canvas_size:
        cw, ch = canvas_size
    elif len(layer_stack) > 0:
        cw = max(l.image.width() for l in layer_stack)
        ch = max(l.image.height() for l in layer_stack)
    else:
        cw, ch = 1024, 1024

    manifest = {
        "version": 1,
        "canvas_width": cw,
        "canvas_height": ch,
        "layers": [],
        "active_layer_uid": active_uid or layer_stack.active_uid,
    }

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for layer in layer_stack:
            layer_meta = {
                "uid": layer.uid,
                "name": layer.name,
                "visible": layer.visible,
                "opacity": layer.opacity,
                "locked": layer.locked,
                "offset_x": layer.offset_x,
                "offset_y": layer.offset_y,
                "rotation": layer.rotation,
                "scale_x": layer.scale_x,
                "scale_y": layer.scale_y,
                "flip_h": layer.flip_h,
                "flip_v": layer.flip_v,
                "blend_mode": layer.blend_mode,
            }
            manifest["layers"].append(layer_meta)

            # Save layer image as PNG inside zip
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                layer.image.save(tmp.name, "PNG")
                zf.write(tmp.name, f"layers/{layer.uid}.png")
                Path(tmp.name).unlink(missing_ok=True)

        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    logger.info("Saved .sdraw: %s (%d layers)", path, len(layer_stack))


def load_sdraw(path: str, layer_stack: LayerStack) -> tuple[int, int]:
    """Load a .sdraw file into the given LayerStack.

    Returns (canvas_width, canvas_height).
    """
    layer_stack.clear()

    with zipfile.ZipFile(path, "r") as zf:
        manifest_data = zf.read("manifest.json")
        manifest = json.loads(manifest_data)

        canvas_w = manifest.get("canvas_width", 1024)
        canvas_h = manifest.get("canvas_height", 1024)

        for layer_meta in manifest.get("layers", []):
            uid = layer_meta["uid"]
            png_path = f"layers/{uid}.png"

            img = QImage()
            if png_path in zf.namelist():
                png_data = zf.read(png_path)
                img.loadFromData(png_data)
                if img.format() != QImage.Format.Format_ARGB32:
                    img = img.convertToFormat(QImage.Format.Format_ARGB32)
            else:
                # Missing layer image -- create transparent placeholder
                img = QImage(canvas_w, canvas_h, QImage.Format.Format_ARGB32)
                img.fill(QColor(0, 0, 0, 0))

            layer = Layer(
                uid=uid,
                name=layer_meta.get("name", "Layer"),
                image=img,
                visible=layer_meta.get("visible", True),
                opacity=layer_meta.get("opacity", 1.0),
                locked=layer_meta.get("locked", False),
                offset_x=layer_meta.get("offset_x", 0.0),
                offset_y=layer_meta.get("offset_y", 0.0),
                rotation=layer_meta.get("rotation", 0.0),
                scale_x=layer_meta.get("scale_x", 1.0),
                scale_y=layer_meta.get("scale_y", 1.0),
                flip_h=layer_meta.get("flip_h", False),
                flip_v=layer_meta.get("flip_v", False),
                blend_mode=layer_meta.get("blend_mode", "normal"),
            )
            layer_stack._layers.append(layer)

        active_uid = manifest.get("active_layer_uid")
        if active_uid and layer_stack.get(active_uid):
            layer_stack._active_uid = active_uid
        elif len(layer_stack) > 0:
            layer_stack._active_uid = layer_stack[-1].uid

        layer_stack.layers_changed.emit()
        if layer_stack._active_uid:
            layer_stack.active_changed.emit(layer_stack._active_uid)

    logger.info("Loaded .sdraw: %s (%d layers)", path, len(layer_stack))
    return canvas_w, canvas_h
