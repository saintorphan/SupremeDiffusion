"""Real-time GL preview pipeline for timeline: PyAV decoding + OpenGL compositing."""

from __future__ import annotations

# Optional dependency flags
HAS_PYAV = False
HAS_OPENGL = False

try:
    import av  # noqa: F401
    HAS_PYAV = True
except ImportError:
    pass

try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget  # noqa: F401
    from OpenGL import GL  # noqa: F401
    HAS_OPENGL = True
except ImportError:
    pass
