"""OpenGL 3D viewport widget for loading, viewing, and compositing meshes."""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QVBoxLayout, QWidget

logger = logging.getLogger(__name__)

# Try to import OpenGL -- will be None if not installed
try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from OpenGL import GL
    from OpenGL import GLU
    HAS_OPENGL = True
except ImportError:
    HAS_OPENGL = False
    QOpenGLWidget = None

# Try to import trimesh for loading models
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    trimesh = None


class _GLCanvas(QOpenGLWidget if HAS_OPENGL else QWidget):
    """Internal OpenGL canvas for 3D rendering.

    Supports:
    - Orbit camera (left-drag), Pan (middle/shift+left), Zoom (scroll)
    - Mesh rendering with flat shading, vertex colors, optional wireframe
    - Per-mesh transforms (position, rotation, scale)
    - Configurable lighting (direction, color, intensity, ambient)
    - Background image plate
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Camera state
        self._rot_x = -30.0
        self._rot_y = 45.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._zoom = 3.0

        # Mouse tracking
        self._last_mouse_pos = None
        self._mouse_button = None

        # Mesh data
        self._meshes: list[dict] = []

        # Grid / wireframe
        self._show_grid = True
        self._show_wireframe = False

        # Background image plate
        self._bg_image_path: str | None = None
        self._bg_texture_id: int | None = None

        # Lighting
        self._light_azimuth = 45.0
        self._light_elevation = 45.0
        self._light_color = (1.0, 1.0, 1.0)
        self._light_intensity = 0.8
        self._ambient = 0.3

        # Gizmo / selection state
        self._selected_mesh_idx: int = -1
        self._interaction_mode: str = "orbit"  # orbit, translate, rotate, scale
        self._gizmo_drag_axis: str | None = None  # "x", "y", "z", or None
        self._gizmo_drag_start: tuple | None = None

    # -- Mesh loading --------------------------------------------------

    def load_mesh(self, path: str) -> bool:
        """Load a mesh file (OBJ, GLB, PLY, STL) into the viewport."""
        if not HAS_TRIMESH:
            logger.error("trimesh not installed -- cannot load meshes")
            return False

        try:
            scene_or_mesh = trimesh.load(path)

            if isinstance(scene_or_mesh, trimesh.Scene):
                for name, geom in scene_or_mesh.geometry.items():
                    if isinstance(geom, trimesh.Trimesh):
                        self._add_trimesh(geom, name, source_path=path)
            elif isinstance(scene_or_mesh, trimesh.Trimesh):
                self._add_trimesh(scene_or_mesh, Path(path).stem, source_path=path)
            else:
                logger.warning("Unsupported mesh type: %s", type(scene_or_mesh))
                return False

            self.update()
            return True

        except Exception as exc:
            logger.error("Failed to load mesh %s: %s", path, exc)
            return False

    def _add_trimesh(self, mesh: 'trimesh.Trimesh', name: str,
                     source_path: str = "") -> None:
        """Convert a trimesh to our internal format."""
        import numpy as np

        # Center the mesh
        mesh.vertices -= mesh.centroid

        # Scale to unit bounding box
        extents = mesh.extents
        max_extent = max(extents) if max(extents) > 0 else 1.0
        mesh.vertices /= max_extent

        # Extract vertex colors if available
        colors = None
        if mesh.visual and hasattr(mesh.visual, 'vertex_colors'):
            try:
                vc = mesh.visual.vertex_colors
                if vc is not None and len(vc) == len(mesh.vertices):
                    colors = (vc[:, :3].astype(np.float32) / 255.0)
            except Exception:
                pass

        # Extract UV coordinates if available
        uvs = None
        if mesh.visual and hasattr(mesh.visual, 'uv'):
            try:
                uv = mesh.visual.uv
                if uv is not None and len(uv) == len(mesh.vertices):
                    uvs = uv.astype(np.float32)
            except Exception:
                pass

        # Look for albedo texture alongside the source mesh file
        texture_path = None
        if source_path:
            from pathlib import Path as _P
            stem = _P(source_path).stem
            parent = _P(source_path).parent
            for candidate in [
                parent / f"{stem}_albedo.png",
                parent / f"{stem}.png",
                parent / "albedo.png",
            ]:
                if candidate.is_file():
                    texture_path = str(candidate)
                    break

        self._meshes.append({
            "vertices": mesh.vertices.astype(np.float32),
            "faces": mesh.faces.astype(np.int32),
            "normals": mesh.vertex_normals.astype(np.float32),
            "colors": colors,
            "uvs": uvs,
            "texture_path": texture_path,
            "texture_id": None,  # GL texture ID, loaded lazily
            "source_path": source_path,
            "name": name,
            "visible": True,
            "position": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "user_scale": 1.0,
        })

    def clear_meshes(self) -> None:
        """Remove all loaded meshes and free their GL textures."""
        if HAS_OPENGL:
            for m in self._meshes:
                tid = m.get("texture_id")
                if tid is not None:
                    try:
                        GL.glDeleteTextures([tid])
                    except Exception:
                        pass
        self._meshes.clear()
        self.update()

    def reload_texture(self, mesh_idx: int = 0) -> None:
        """Reload the texture for a mesh (e.g. after texture refinement).

        Scans for ``<stem>_albedo.png`` next to the mesh source file and
        forces a GL texture reload on next paint.
        """
        if mesh_idx < 0 or mesh_idx >= len(self._meshes):
            return
        m = self._meshes[mesh_idx]
        src = m.get("source_path", "")
        if not src:
            return
        from pathlib import Path as _P
        stem = _P(src).stem
        parent = _P(src).parent
        for candidate in [
            parent / f"{stem}_albedo.png",
            parent / f"{stem}.png",
            parent / "albedo.png",
        ]:
            if candidate.is_file():
                m["texture_path"] = str(candidate)
                # Force GL texture re-upload
                if HAS_OPENGL and m.get("texture_id") is not None:
                    try:
                        GL.glDeleteTextures([m["texture_id"]])
                    except Exception:
                        pass
                m["texture_id"] = None
                self.update()
                return

    # -- Lighting API --------------------------------------------------

    def set_light(self, azimuth: float, elevation: float,
                  color: tuple = (1.0, 1.0, 1.0),
                  intensity: float = 0.8, ambient: float = 0.3) -> None:
        """Configure the directional light."""
        self._light_azimuth = azimuth
        self._light_elevation = elevation
        self._light_color = color
        self._light_intensity = intensity
        self._ambient = ambient
        self.update()

    # -- Camera state API ----------------------------------------------

    def get_camera_state(self) -> dict:
        """Return current camera position/rotation/zoom for keyframing."""
        return {
            "rot_x": self._rot_x,
            "rot_y": self._rot_y,
            "pan_x": self._pan_x,
            "pan_y": self._pan_y,
            "zoom": self._zoom,
        }

    def set_camera_state(self, state: dict) -> None:
        """Set camera from animation system."""
        self._rot_x = state.get("rot_x", self._rot_x)
        self._rot_y = state.get("rot_y", self._rot_y)
        self._pan_x = state.get("pan_x", self._pan_x)
        self._pan_y = state.get("pan_y", self._pan_y)
        self._zoom = state.get("zoom", self._zoom)
        self.update()

    # -- OpenGL --------------------------------------------------------

    def initializeGL(self) -> None:
        if not HAS_OPENGL:
            return

        GL.glClearColor(0.12, 0.12, 0.14, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)

        self._update_light()

    def _update_light(self) -> None:
        """Apply current light settings to GL_LIGHT0."""
        if not HAS_OPENGL:
            return
        # Convert azimuth/elevation to direction vector
        az = math.radians(self._light_azimuth)
        el = math.radians(self._light_elevation)
        x = math.cos(el) * math.sin(az)
        y = math.sin(el)
        z = math.cos(el) * math.cos(az)
        pos = [x, y, z, 0.0]  # directional light (w=0)

        r, g, b = self._light_color
        i = self._light_intensity
        a = self._ambient

        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, pos)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, [r * i, g * i, b * i, 1.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, [r * i * 0.5, g * i * 0.5, b * i * 0.5, 1.0])
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, [a, a, a, 1.0])

    def resizeGL(self, w: int, h: int) -> None:
        if not HAS_OPENGL or h == 0:
            return
        GL.glViewport(0, 0, w, h)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GLU.gluPerspective(45.0, w / h, 0.1, 100.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def paintGL(self) -> None:
        if not HAS_OPENGL:
            return
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # Background image (before 3D scene)
        if self._bg_image_path:
            self._draw_background()

        GL.glLoadIdentity()
        self._update_light()

        # Camera
        GL.glTranslatef(self._pan_x, self._pan_y, -self._zoom)
        GL.glRotatef(self._rot_x, 1, 0, 0)
        GL.glRotatef(self._rot_y, 0, 1, 0)

        # Grid
        if self._show_grid:
            self._draw_grid()

        # Meshes
        for i, mesh_data in enumerate(self._meshes):
            if not mesh_data["visible"]:
                continue
            # Highlight selected mesh with a subtle tint
            self._draw_mesh(mesh_data)

        # Gizmo (drawn last, on top)
        self._draw_gizmo()

    def _draw_background(self) -> None:
        """Render background image as a fullscreen textured quad."""
        if self._bg_texture_id is None and self._bg_image_path:
            self._load_bg_texture()
        if self._bg_texture_id is None:
            return

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        GL.glOrtho(0, 1, 0, 1, -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glLoadIdentity()

        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._bg_texture_id)

        GL.glColor3f(1.0, 1.0, 1.0)
        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0, 0); GL.glVertex2f(0, 0)
        GL.glTexCoord2f(1, 0); GL.glVertex2f(1, 0)
        GL.glTexCoord2f(1, 1); GL.glVertex2f(1, 1)
        GL.glTexCoord2f(0, 1); GL.glVertex2f(0, 1)
        GL.glEnd()

        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_LIGHTING)

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopMatrix()

        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)

    def _load_bg_texture(self) -> None:
        """Load background image as an OpenGL texture."""
        try:
            from PIL import Image as PILImage
            import numpy as np

            img = PILImage.open(self._bg_image_path).convert("RGB")
            img = img.transpose(PILImage.FLIP_TOP_BOTTOM)
            data = np.array(img, dtype=np.uint8)

            tex_id = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
                data.shape[1], data.shape[0], 0,
                GL.GL_RGB, GL.GL_UNSIGNED_BYTE, data,
            )
            self._bg_texture_id = tex_id

        except Exception as exc:
            logger.warning("Failed to load background texture: %s", exc)
            self._bg_texture_id = None

    def _draw_grid(self) -> None:
        """Draw a ground-plane grid with axis indicators."""
        GL.glDisable(GL.GL_LIGHTING)
        GL.glBegin(GL.GL_LINES)
        GL.glColor3f(0.3, 0.3, 0.3)
        extent = 5
        for i in range(-extent, extent + 1):
            GL.glVertex3f(i * 0.5, 0, -extent * 0.5)
            GL.glVertex3f(i * 0.5, 0, extent * 0.5)
            GL.glVertex3f(-extent * 0.5, 0, i * 0.5)
            GL.glVertex3f(extent * 0.5, 0, i * 0.5)
        GL.glEnd()

        # Axis indicators
        GL.glLineWidth(2.0)
        GL.glBegin(GL.GL_LINES)
        GL.glColor3f(0.8, 0.2, 0.2); GL.glVertex3f(0, 0, 0); GL.glVertex3f(1, 0, 0)
        GL.glColor3f(0.2, 0.8, 0.2); GL.glVertex3f(0, 0, 0); GL.glVertex3f(0, 1, 0)
        GL.glColor3f(0.2, 0.2, 0.8); GL.glVertex3f(0, 0, 0); GL.glVertex3f(0, 0, 1)
        GL.glEnd()
        GL.glLineWidth(1.0)
        GL.glEnable(GL.GL_LIGHTING)

    def _load_mesh_texture(self, mesh_data: dict) -> int | None:
        """Load a mesh's albedo texture into OpenGL. Returns texture ID or None."""
        tex_path = mesh_data.get("texture_path")
        if not tex_path:
            return None
        try:
            from PIL import Image as PILImage
            import numpy as np

            img = PILImage.open(tex_path).convert("RGB")
            img = img.transpose(PILImage.FLIP_TOP_BOTTOM)
            data = np.array(img, dtype=np.uint8)

            tex_id = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
                data.shape[1], data.shape[0], 0,
                GL.GL_RGB, GL.GL_UNSIGNED_BYTE, data,
            )
            logger.debug("Loaded mesh texture: %s (%dx%d)", tex_path, data.shape[1], data.shape[0])
            return tex_id
        except Exception as exc:
            logger.warning("Failed to load mesh texture %s: %s", tex_path, exc)
            return None

    def _draw_mesh(self, mesh_data: dict) -> None:
        """Render a mesh with textures, vertex colors, or flat shading."""
        verts = mesh_data["vertices"]
        faces = mesh_data["faces"]
        norms = mesh_data["normals"]
        colors = mesh_data.get("colors")
        uvs = mesh_data.get("uvs")
        pos = mesh_data["position"]
        rot = mesh_data["rotation"]
        scale = mesh_data.get("user_scale", 1.0)

        # Lazy-load texture if we have UVs + texture path but no GL ID yet
        has_texture = False
        if uvs is not None and mesh_data.get("texture_path") and mesh_data.get("texture_id") is None:
            mesh_data["texture_id"] = self._load_mesh_texture(mesh_data)
        if uvs is not None and mesh_data.get("texture_id") is not None:
            has_texture = True

        GL.glPushMatrix()
        GL.glTranslatef(*pos)
        GL.glRotatef(rot[0], 1, 0, 0)
        GL.glRotatef(rot[1], 0, 1, 0)
        GL.glRotatef(rot[2], 0, 0, 1)
        GL.glScalef(scale, scale, scale)

        # Enable texturing if available
        if has_texture:
            GL.glEnable(GL.GL_TEXTURE_2D)
            GL.glBindTexture(GL.GL_TEXTURE_2D, mesh_data["texture_id"])
            GL.glColor3f(1.0, 1.0, 1.0)  # modulate with white so texture shows true color

        # Solid pass
        GL.glBegin(GL.GL_TRIANGLES)
        for face in faces:
            for vi in face:
                GL.glNormal3fv(norms[vi])
                if has_texture:
                    GL.glTexCoord2fv(uvs[vi])
                elif colors is not None:
                    GL.glColor3fv(colors[vi])
                else:
                    GL.glColor3f(0.7, 0.7, 0.75)
                GL.glVertex3fv(verts[vi])
        GL.glEnd()

        if has_texture:
            GL.glDisable(GL.GL_TEXTURE_2D)

        # Wireframe overlay
        if self._show_wireframe:
            GL.glDisable(GL.GL_LIGHTING)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            GL.glColor3f(0.1, 0.1, 0.1)
            GL.glBegin(GL.GL_TRIANGLES)
            for face in faces:
                for vi in face:
                    GL.glVertex3fv(verts[vi])
            GL.glEnd()
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            GL.glEnable(GL.GL_LIGHTING)

        GL.glPopMatrix()

    # -- Gizmo rendering -----------------------------------------------

    _GIZMO_LEN = 0.5
    _GIZMO_COLORS = {"x": (0.9, 0.2, 0.2), "y": (0.2, 0.9, 0.2), "z": (0.2, 0.2, 0.9)}

    def _draw_gizmo(self) -> None:
        """Draw translate/rotate/scale gizmo at the selected mesh's position."""
        if self._selected_mesh_idx < 0 or self._selected_mesh_idx >= len(self._meshes):
            return
        if self._interaction_mode == "orbit":
            return

        mesh = self._meshes[self._selected_mesh_idx]
        pos = mesh["position"]
        scale = mesh.get("user_scale", 1.0)

        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_DEPTH_TEST)  # gizmos always on top
        GL.glLineWidth(3.0)

        GL.glPushMatrix()
        GL.glTranslatef(*pos)

        length = self._GIZMO_LEN

        if self._interaction_mode == "translate":
            self._draw_gizmo_arrows(length)
        elif self._interaction_mode == "rotate":
            self._draw_gizmo_rings(length)
        elif self._interaction_mode == "scale":
            self._draw_gizmo_arrows(length, cubes=True)

        GL.glPopMatrix()

        GL.glLineWidth(1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_LIGHTING)

    def _draw_gizmo_arrows(self, length: float, cubes: bool = False) -> None:
        """Draw XYZ axis arrows (for translate) or arrows with cubes (for scale)."""
        axes = [
            ("x", (length, 0, 0)),
            ("y", (0, length, 0)),
            ("z", (0, 0, length)),
        ]
        for axis_name, end in axes:
            r, g, b = self._GIZMO_COLORS[axis_name]
            # Highlight dragged axis
            if self._gizmo_drag_axis == axis_name:
                r, g, b = 1.0, 1.0, 0.3
            GL.glColor3f(r, g, b)
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3f(0, 0, 0)
            GL.glVertex3f(*end)
            GL.glEnd()
            # Arrowhead or cube at tip
            GL.glPushMatrix()
            GL.glTranslatef(*end)
            if cubes:
                s = 0.04
                self._draw_cube(s)
            else:
                self._draw_cone(axis_name, 0.03, 0.1)
            GL.glPopMatrix()

    def _draw_gizmo_rings(self, length: float) -> None:
        """Draw rotation rings around XYZ axes."""
        import numpy as np
        segments = 32
        radius = length * 0.8
        for axis_name in ("x", "y", "z"):
            r, g, b = self._GIZMO_COLORS[axis_name]
            if self._gizmo_drag_axis == axis_name:
                r, g, b = 1.0, 1.0, 0.3
            GL.glColor3f(r, g, b)
            GL.glBegin(GL.GL_LINE_LOOP)
            for i in range(segments):
                angle = 2.0 * math.pi * i / segments
                c, s = math.cos(angle) * radius, math.sin(angle) * radius
                if axis_name == "x":
                    GL.glVertex3f(0, c, s)
                elif axis_name == "y":
                    GL.glVertex3f(c, 0, s)
                else:
                    GL.glVertex3f(c, s, 0)
            GL.glEnd()

    @staticmethod
    def _draw_cube(half: float) -> None:
        """Draw a small solid cube (for scale gizmo tips)."""
        GL.glBegin(GL.GL_QUADS)
        for dx in (-1, 1):
            for dy in (-1, 1):
                for dz in (-1, 1):
                    GL.glVertex3f(dx * half, dy * half, dz * half)
        GL.glEnd()

    @staticmethod
    def _draw_cone(axis: str, radius: float, height: float) -> None:
        """Draw a small cone pointing along the given axis."""
        segments = 8
        GL.glBegin(GL.GL_TRIANGLE_FAN)
        # Tip
        if axis == "x":
            GL.glVertex3f(height, 0, 0)
        elif axis == "y":
            GL.glVertex3f(0, height, 0)
        else:
            GL.glVertex3f(0, 0, height)
        # Base circle
        for i in range(segments + 1):
            angle = 2.0 * math.pi * i / segments
            c, s = math.cos(angle) * radius, math.sin(angle) * radius
            if axis == "x":
                GL.glVertex3f(0, c, s)
            elif axis == "y":
                GL.glVertex3f(c, 0, s)
            else:
                GL.glVertex3f(c, s, 0)
        GL.glEnd()

    # -- Gizmo hit testing / dragging ----------------------------------

    def _screen_to_world_ray(self, sx: int, sy: int) -> tuple:
        """Convert screen pixel to a world-space ray (origin, direction)."""
        import numpy as np

        viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
        modelview = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
        projection = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
        sy_gl = viewport[3] - sy  # flip Y

        near = GLU.gluUnProject(sx, sy_gl, 0.0, modelview, projection, viewport)
        far = GLU.gluUnProject(sx, sy_gl, 1.0, modelview, projection, viewport)
        origin = np.array(near, dtype=np.float64)
        direction = np.array(far, dtype=np.float64) - origin
        length = np.linalg.norm(direction)
        if length > 0:
            direction /= length
        return origin, direction

    def _hit_test_gizmo(self, sx: int, sy: int) -> str | None:
        """Test if screen point hits a gizmo axis handle. Returns 'x','y','z' or None."""
        if self._selected_mesh_idx < 0 or self._interaction_mode == "orbit":
            return None

        mesh = self._meshes[self._selected_mesh_idx]
        center = mesh["position"]

        try:
            self.makeCurrent()
            origin, direction = self._screen_to_world_ray(sx, sy)
            self.doneCurrent()
        except Exception:
            return None

        import numpy as np
        center = np.array(center, dtype=np.float64)
        length = self._GIZMO_LEN
        threshold = 0.08  # hit radius

        # Test each axis: closest point on the ray to the axis line
        for axis_name, axis_vec in [("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1])]:
            axis = np.array(axis_vec, dtype=np.float64)
            # Parameterize: point on axis = center + t*axis, point on ray = origin + s*direction
            # Find closest approach
            w = origin - center
            a = float(np.dot(direction, direction))
            b = float(np.dot(direction, axis))
            c = float(np.dot(axis, axis))
            d = float(np.dot(direction, w))
            e = float(np.dot(axis, w))
            denom = a * c - b * b
            if abs(denom) < 1e-10:
                continue
            s = (b * e - c * d) / denom
            t = (a * e - b * d) / denom
            if t < 0 or t > length:
                continue  # outside the arrow
            closest_ray = origin + s * direction
            closest_axis = center + t * axis
            dist = float(np.linalg.norm(closest_ray - closest_axis))
            if dist < threshold:
                return axis_name
        return None

    def _hit_test_mesh(self, sx: int, sy: int) -> int:
        """Test if screen point hits any mesh AABB. Returns mesh index or -1."""
        try:
            self.makeCurrent()
            origin, direction = self._screen_to_world_ray(sx, sy)
            self.doneCurrent()
        except Exception:
            return -1

        import numpy as np
        best_dist = float("inf")
        best_idx = -1

        for i, mesh in enumerate(self._meshes):
            if not mesh["visible"]:
                continue
            pos = np.array(mesh["position"], dtype=np.float64)
            scale = mesh.get("user_scale", 1.0)
            half = 0.5 * scale  # unit bounding box scaled
            bmin = pos - half
            bmax = pos + half
            # Ray-AABB intersection
            t = self._ray_aabb(origin, direction, bmin, bmax)
            if t is not None and t < best_dist:
                best_dist = t
                best_idx = i
        return best_idx

    @staticmethod
    def _ray_aabb(origin, direction, bmin, bmax) -> float | None:
        """Ray-AABB intersection test. Returns t or None."""
        import numpy as np
        tmin = -float("inf")
        tmax = float("inf")
        for i in range(3):
            if abs(direction[i]) < 1e-10:
                if origin[i] < bmin[i] or origin[i] > bmax[i]:
                    return None
            else:
                t1 = (bmin[i] - origin[i]) / direction[i]
                t2 = (bmax[i] - origin[i]) / direction[i]
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmin > tmax:
                    return None
        if tmax < 0:
            return None
        return tmin if tmin >= 0 else tmax

    # -- Multi-angle rendering -----------------------------------------

    def render_view(self, rot_x: float, rot_y: float, width: int = 512, height: int = 512) -> 'QImage':
        """Render the scene from a specific camera angle, return QImage."""
        old_state = self.get_camera_state()
        self._rot_x = rot_x
        self._rot_y = rot_y
        self.makeCurrent()
        # Resize the framebuffer temporarily
        self.resizeGL(width, height)
        self.paintGL()
        img = self.grabFramebuffer()
        # Restore
        self.set_camera_state(old_state)
        self.resizeGL(self.width(), self.height())
        self.doneCurrent()
        return img

    # -- Mouse interaction ---------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last_mouse_pos = event.position().toPoint()
        self._mouse_button = event.button()
        self._gizmo_drag_axis = None

        if event.button() == Qt.MouseButton.LeftButton and self._interaction_mode != "orbit":
            sx, sy = int(event.position().x()), int(event.position().y())
            # Test gizmo first
            axis = self._hit_test_gizmo(sx, sy)
            if axis:
                self._gizmo_drag_axis = axis
                self._gizmo_drag_start = (sx, sy)
                return
            # Test mesh selection
            idx = self._hit_test_mesh(sx, sy)
            if idx >= 0:
                self._selected_mesh_idx = idx
                self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_mouse_pos is None:
            return

        pos = event.position().toPoint()
        dx = pos.x() - self._last_mouse_pos.x()
        dy = pos.y() - self._last_mouse_pos.y()
        self._last_mouse_pos = pos

        # Gizmo dragging
        if self._gizmo_drag_axis and self._selected_mesh_idx >= 0:
            mesh = self._meshes[self._selected_mesh_idx]
            axis = self._gizmo_drag_axis
            sensitivity = 0.005 * self._zoom

            if self._interaction_mode == "translate":
                delta = dx * sensitivity if axis != "y" else -dy * sensitivity
                idx = {"x": 0, "y": 1, "z": 2}[axis]
                mesh["position"][idx] += delta

            elif self._interaction_mode == "rotate":
                delta = dx * 0.5
                idx = {"x": 0, "y": 1, "z": 2}[axis]
                mesh["rotation"][idx] += delta

            elif self._interaction_mode == "scale":
                delta = dx * 0.005
                mesh["user_scale"] = max(0.05, mesh.get("user_scale", 1.0) + delta)

            self.update()
            return

        # Normal camera controls
        modifiers = event.modifiers()

        if self._mouse_button == Qt.MouseButton.LeftButton and not (modifiers & Qt.KeyboardModifier.ShiftModifier):
            # In non-orbit modes, left-drag without gizmo = orbit anyway
            self._rot_y += dx * 0.5
            self._rot_x += dy * 0.5
        elif self._mouse_button == Qt.MouseButton.MiddleButton or (
            self._mouse_button == Qt.MouseButton.LeftButton and modifiers & Qt.KeyboardModifier.ShiftModifier
        ):
            self._pan_x += dx * 0.005 * self._zoom
            self._pan_y -= dy * 0.005 * self._zoom

        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._gizmo_drag_axis = None
        self._gizmo_drag_start = None
        self._last_mouse_pos = None
        self._mouse_button = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        self._zoom *= 0.9 if delta > 0 else 1.1
        self._zoom = max(0.5, min(50.0, self._zoom))
        self.update()


class GLViewportWidget(QWidget):
    """High-level 3D viewport widget with mesh loading API.

    Wraps _GLCanvas with a fallback message if OpenGL is unavailable.
    """

    mesh_loaded = Signal(str)
    mesh_selected = Signal(int)  # index of selected mesh (-1 = none)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if HAS_OPENGL:
            self._canvas = _GLCanvas()
            layout.addWidget(self._canvas)
        else:
            from PySide6.QtWidgets import QLabel
            self._canvas = None
            lbl = QLabel(
                "OpenGL not available.\n\n"
                "Install PyOpenGL:\n"
                "pip install PyOpenGL"
            )
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #888; font-size: 14px;")
            layout.addWidget(lbl)

    def load_mesh(self, path: str) -> bool:
        if self._canvas is None:
            return False
        ok = self._canvas.load_mesh(path)
        if ok:
            self.mesh_loaded.emit(path)
        return ok

    def clear(self) -> None:
        if self._canvas:
            self._canvas.clear_meshes()

    def set_wireframe(self, enabled: bool) -> None:
        if self._canvas:
            self._canvas._show_wireframe = enabled
            self._canvas.update()

    def set_grid(self, enabled: bool) -> None:
        if self._canvas:
            self._canvas._show_grid = enabled
            self._canvas.update()

    def set_background_image(self, path: str | None) -> None:
        if self._canvas:
            self._canvas._bg_image_path = path
            self._canvas._bg_texture_id = None
            self._canvas.update()

    def set_light(self, azimuth: float, elevation: float,
                  color: tuple = (1.0, 1.0, 1.0),
                  intensity: float = 0.8, ambient: float = 0.3) -> None:
        """Configure directional light."""
        if self._canvas:
            self._canvas.set_light(azimuth, elevation, color, intensity, ambient)

    def render_views(self, angles: list[tuple[float, float]] | None = None,
                     width: int = 512, height: int = 512) -> list['QImage']:
        """Render the scene from multiple camera angles.

        Args:
            angles: List of (rot_x, rot_y) tuples. Defaults to 4 standard views.
            width: Render width.
            height: Render height.

        Returns:
            List of QImages, one per angle.
        """
        if self._canvas is None:
            return []
        if angles is None:
            angles = [
                (-20, 0),     # Front
                (-20, 90),    # Right
                (-20, 180),   # Back
                (-20, 270),   # Left
            ]
        return [self._canvas.render_view(rx, ry, width, height) for rx, ry in angles]

    def reload_texture(self, mesh_idx: int = 0) -> None:
        """Reload texture for a mesh after external texture refinement."""
        if self._canvas:
            self._canvas.reload_texture(mesh_idx)

    def get_mesh_source_path(self, mesh_idx: int = 0) -> str:
        """Return the source file path for a mesh, or empty string."""
        if self._canvas and 0 <= mesh_idx < len(self._canvas._meshes):
            return self._canvas._meshes[mesh_idx].get("source_path", "")
        return ""

    def render_single_view(self, rot_x: float, rot_y: float,
                           width: int = 512, height: int = 512):
        """Render a single view — convenience for the texture refiner."""
        if self._canvas is None:
            return None
        return self._canvas.render_view(rot_x, rot_y, width, height)

    def set_interaction_mode(self, mode: str) -> None:
        """Set interaction mode: 'orbit', 'translate', 'rotate', 'scale'."""
        if self._canvas:
            self._canvas._interaction_mode = mode
            self._canvas.update()

    def set_selected_mesh(self, idx: int) -> None:
        """Set the selected mesh index for gizmo display."""
        if self._canvas:
            self._canvas._selected_mesh_idx = idx
            self._canvas.update()

    @property
    def mesh_count(self) -> int:
        if self._canvas:
            return len(self._canvas._meshes)
        return 0
