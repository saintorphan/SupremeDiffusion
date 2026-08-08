"""PBR OpenGL viewport for Daz2Supreme — modern shader-based 3D renderer.

Provides:
- Orbital camera (drag rotate, scroll zoom, middle-click pan)
- PBR metallic-roughness shading with up to 2 lights
- Skinned mesh rendering (LBS with 4 bone influences)
- Bone overlay (lines)
- Offscreen render to image at arbitrary resolution
- Click-to-select bone
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QImage, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QVBoxLayout, QWidget, QLabel

try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
    from OpenGL import GL
    _HAS_GL = True
except ImportError:
    _HAS_GL = False
    QOpenGLWidget = QWidget  # fallback for type hints

if TYPE_CHECKING:
    from sdqt.models.scene3d import Camera, Figure, Light, Prop, Scene3D

logger = logging.getLogger(__name__)

# ── Shader sources ────────────────────────────────────────────────────────────

_VERT_SRC = """
#version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in ivec4 a_bone_idx;
layout(location = 4) in vec4 a_bone_wt;

uniform mat4 u_mvp;
uniform mat4 u_model;
uniform mat3 u_normal_mat;
uniform mat4 u_bones[128];
uniform bool u_skinned;

out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;
out vec3 v_tangent;

void main() {
    vec4 pos = vec4(a_position, 1.0);
    vec3 norm = a_normal;

    if (u_skinned) {
        mat4 skin = u_bones[a_bone_idx.x] * a_bone_wt.x
                   + u_bones[a_bone_idx.y] * a_bone_wt.y
                   + u_bones[a_bone_idx.z] * a_bone_wt.z
                   + u_bones[a_bone_idx.w] * a_bone_wt.w;
        pos = skin * pos;
        norm = mat3(skin) * norm;
    }

    v_world_pos = (u_model * pos).xyz;
    v_normal = normalize(u_normal_mat * norm);
    v_uv = a_uv;
    // Approximate tangent from UV direction
    vec3 t = cross(norm, vec3(0.0, 1.0, 0.0));
    if (length(t) < 0.001) t = cross(norm, vec3(1.0, 0.0, 0.0));
    v_tangent = normalize(u_normal_mat * t);
    gl_Position = u_mvp * pos;
}
"""

_FRAG_SRC = """
#version 330 core

in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;
in vec3 v_tangent;

uniform vec4 u_base_color;
uniform float u_metallic;
uniform float u_roughness;
uniform vec3 u_cam_pos;

uniform int u_num_lights;
uniform vec3 u_light_dir[2];
uniform vec3 u_light_color[2];
uniform float u_light_intensity[2];
uniform vec3 u_ambient;

uniform bool u_has_diffuse_map;
uniform sampler2D u_diffuse_map;
uniform bool u_has_normal_map;
uniform sampler2D u_normal_map;
uniform bool u_has_roughness_map;
uniform sampler2D u_roughness_map;
uniform bool u_has_metallic_map;
uniform sampler2D u_metallic_map;
uniform bool u_has_opacity_map;
uniform sampler2D u_opacity_map;

uniform bool u_has_shadow_map;
uniform sampler2D u_shadow_map;
uniform mat4 u_light_vp;

out vec4 frag_color;

const float PI = 3.14159265359;

float distribution_ggx(vec3 N, vec3 H, float rough) {
    float a = rough * rough;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 0.0001);
}

float geometry_schlick(float NdotV, float rough) {
    float k = (rough + 1.0) * (rough + 1.0) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

vec3 fresnel_schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float shadow_calc(vec3 world_pos) {
    if (!u_has_shadow_map) return 1.0;
    vec4 light_space = u_light_vp * vec4(world_pos, 1.0);
    vec3 proj = light_space.xyz / light_space.w;
    proj = proj * 0.5 + 0.5;
    if (proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0)
        return 1.0;
    // PCF 3x3
    float shadow = 0.0;
    vec2 texel = 1.0 / vec2(textureSize(u_shadow_map, 0));
    float bias = 0.002;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float depth = texture(u_shadow_map, proj.xy + vec2(x, y) * texel).r;
            shadow += (proj.z - bias > depth) ? 0.0 : 1.0;
        }
    }
    return shadow / 9.0;
}

void main() {
    vec3 N = normalize(v_normal);

    // Normal mapping via TBN
    if (u_has_normal_map) {
        vec3 T = normalize(v_tangent - dot(v_tangent, N) * N);
        vec3 B = cross(N, T);
        mat3 TBN = mat3(T, B, N);
        vec3 nmap = texture(u_normal_map, v_uv).rgb * 2.0 - 1.0;
        N = normalize(TBN * nmap);
    }

    vec3 V = normalize(u_cam_pos - v_world_pos);

    vec4 albedo = u_base_color;
    if (u_has_diffuse_map) {
        albedo *= texture(u_diffuse_map, v_uv);
    }

    float roughness = u_roughness;
    if (u_has_roughness_map) {
        roughness *= texture(u_roughness_map, v_uv).g; // green channel (glTF convention)
    }
    roughness = clamp(roughness, 0.04, 1.0);

    float metallic = u_metallic;
    if (u_has_metallic_map) {
        metallic *= texture(u_metallic_map, v_uv).b; // blue channel (glTF convention)
    }

    float opacity = albedo.a;
    if (u_has_opacity_map) {
        opacity *= texture(u_opacity_map, v_uv).r;
    }
    if (opacity < 0.01) discard;

    float shadow = shadow_calc(v_world_pos);

    vec3 F0 = mix(vec3(0.04), albedo.rgb, metallic);
    vec3 color = u_ambient * albedo.rgb;

    for (int i = 0; i < u_num_lights && i < 2; i++) {
        vec3 L = normalize(-u_light_dir[i]);
        vec3 H = normalize(V + L);
        float NdotL = max(dot(N, L), 0.0);
        float NdotV = max(dot(N, V), 0.001);

        float D = distribution_ggx(N, H, roughness);
        float G = geometry_schlick(NdotV, roughness)
                * geometry_schlick(NdotL, roughness);
        vec3 F = fresnel_schlick(max(dot(H, V), 0.0), F0);

        vec3 spec = (D * G * F) / (4.0 * NdotV * NdotL + 0.0001);
        vec3 kD = (1.0 - F) * (1.0 - metallic);
        float light_shadow = (i == 0) ? shadow : 1.0; // shadow only on primary light
        color += (kD * albedo.rgb / PI + spec)
               * u_light_color[i] * u_light_intensity[i] * NdotL * light_shadow;
    }

    color = color / (color + 1.0);
    color = pow(color, vec3(1.0 / 2.2));
    frag_color = vec4(color, opacity);
}
"""

# Shadow depth pass shader
_SHADOW_VERT_SRC = """
#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 3) in ivec4 a_bone_idx;
layout(location = 4) in vec4 a_bone_wt;

uniform mat4 u_light_mvp;
uniform mat4 u_bones[128];
uniform bool u_skinned;

void main() {
    vec4 pos = vec4(a_position, 1.0);
    if (u_skinned) {
        mat4 skin = u_bones[a_bone_idx.x] * a_bone_wt.x
                   + u_bones[a_bone_idx.y] * a_bone_wt.y
                   + u_bones[a_bone_idx.z] * a_bone_wt.z
                   + u_bones[a_bone_idx.w] * a_bone_wt.w;
        pos = skin * pos;
    }
    gl_Position = u_light_mvp * pos;
}
"""

_SHADOW_FRAG_SRC = """
#version 330 core
void main() {
    // Depth is written automatically
}
"""

_LINE_VERT_SRC = """
#version 330 core
layout(location = 0) in vec3 a_position;
uniform mat4 u_mvp;
void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
}
"""

_LINE_FRAG_SRC = """
#version 330 core
uniform vec4 u_color;
out vec4 frag_color;
void main() {
    frag_color = u_color;
}
"""

_GRID_VERT_SRC = """
#version 330 core
layout(location = 0) in vec3 a_position;
uniform mat4 u_mvp;
out float v_dist;
void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    v_dist = length(a_position.xz);
}
"""

_GRID_FRAG_SRC = """
#version 330 core
in float v_dist;
out vec4 frag_color;
void main() {
    float alpha = clamp(1.0 - v_dist / 10.0, 0.05, 0.3);
    frag_color = vec4(0.4, 0.4, 0.4, alpha);
}
"""

# HDRI skybox — equirectangular projection on a fullscreen quad
_SKY_VERT_SRC = """
#version 330 core
out vec2 v_uv;
void main() {
    // Fullscreen triangle
    vec2 pos = vec2((gl_VertexID & 1) * 4.0 - 1.0,
                    (gl_VertexID & 2) * 2.0 - 1.0);
    v_uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.9999, 1.0);
}
"""

_SKY_FRAG_SRC = """
#version 330 core
in vec2 v_uv;
uniform sampler2D u_hdri;
uniform mat4 u_inv_vp;
out vec4 frag_color;

const float PI = 3.14159265359;

void main() {
    // Reconstruct view ray from UV
    vec4 clip = vec4(v_uv * 2.0 - 1.0, 1.0, 1.0);
    vec4 world = u_inv_vp * clip;
    vec3 dir = normalize(world.xyz / world.w);

    // Equirectangular mapping
    float phi = atan(dir.z, dir.x);
    float theta = asin(clamp(dir.y, -1.0, 1.0));
    vec2 uv = vec2(phi / (2.0 * PI) + 0.5, theta / PI + 0.5);

    vec3 color = texture(u_hdri, uv).rgb;
    // Tonemap
    color = color / (color + 1.0);
    color = pow(color, vec3(1.0 / 2.2));
    frag_color = vec4(color, 1.0);
}
"""

# ── Matrix helpers ────────────────────────────────────────────────────────────

def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-8)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-8)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def _rotation_matrix(angles_deg: np.ndarray) -> np.ndarray:
    """Euler XYZ rotation -> 4x4 matrix."""
    rx, ry, rz = np.radians(angles_deg.astype(np.float64))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = cy * cz
    m[0, 1] = -cy * sz
    m[0, 2] = sy
    m[1, 0] = sx * sy * cz + cx * sz
    m[1, 1] = -sx * sy * sz + cx * cz
    m[1, 2] = -sx * cy
    m[2, 0] = -cx * sy * cz + sx * sz
    m[2, 1] = cx * sy * sz + sx * cz
    m[2, 2] = cx * cy
    return m


def _translation_matrix(t: np.ndarray) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 3] = t[0]
    m[1, 3] = t[1]
    m[2, 3] = t[2]
    return m


# ── GPU buffer wrapper ────────────────────────────────────────────────────────

class _MeshBuffers:
    __slots__ = ("vao", "vbo_pos", "vbo_norm", "vbo_uv", "vbo_bi", "vbo_bw",
                 "ebo", "num_indices", "valid")

    def __init__(self) -> None:
        self.vao = self.vbo_pos = self.vbo_norm = self.vbo_uv = 0
        self.vbo_bi = self.vbo_bw = self.ebo = 0
        self.num_indices = 0
        self.valid = False


def _load_texture_to_gl(path: str) -> int:
    """Load an image file as an OpenGL texture. Returns texture ID or 0."""
    if not _HAS_GL or not path:
        return 0
    from pathlib import Path
    p = Path(path)
    if not p.is_file():
        return 0
    try:
        img = QImage(str(p))
        if img.isNull():
            return 0
        img = img.convertToFormat(QImage.Format.Format_RGBA8888).mirrored(False, True)
        w, h = img.width(), img.height()
        ptr = img.constBits()
        data = bytes(ptr)

        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return tex
    except Exception as exc:
        logger.debug("Failed to load texture %s: %s", path, exc)
        return 0


# ── Internal canvas ──────────────────────────────────────────────────────────

class _PBRCanvas(QOpenGLWidget if _HAS_GL else QWidget):
    """Shader-based PBR canvas with skeletal skinning."""

    bone_clicked = Signal(str, str)  # (figure_id, bone_name)
    ik_drag = Signal(str, str, float, float, float)  # (figure_id, bone_name, tx, ty, tz)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._scene: Scene3D | None = None
        self._figure_buffers: dict[str, _MeshBuffers] = {}
        # IK drag state
        self._ik_dragging = False
        self._ik_fig_id = ""
        self._ik_bone_name = ""
        self._ik_plane_point = np.zeros(3, dtype=np.float32)
        self._ik_plane_normal = np.zeros(3, dtype=np.float32)

        # Orbital camera
        self._cam_yaw = 0.0
        self._cam_pitch = 20.0
        self._cam_dist = 3.0
        self._cam_target = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._cam_fov = 45.0

        # Mouse tracking
        self._last_mouse: QPoint | None = None
        self._mouse_button: Qt.MouseButton = Qt.MouseButton.NoButton

        # GL handles
        self._pbr_prog = 0
        self._line_prog = 0
        self._grid_prog = 0
        self._sky_prog = 0
        self._grid_vao = 0
        self._grid_vbo = 0
        self._grid_count = 0
        self._bone_vao = 0
        self._bone_vbo = 0
        self._sky_vao = 0
        self._hdri_tex = 0
        self._hdri_path = ""
        self._shadow_prog = 0
        self._shadow_fbo = 0
        self._shadow_tex = 0
        self._shadow_size = 2048
        self._initialized = False

        # Pose matrices: figure_id -> list[np.ndarray(4,4)]
        self._pose_matrices: dict[str, list[np.ndarray]] = {}

        # Texture cache: path -> GL texture ID
        self._texture_cache: dict[str, int] = {}

        # Scene lights (mutable from UI)
        self._lights: list["Light"] = []

        self.show_bones = True
        self.show_grid = True

    # -- Scene management --------------------------------------------------

    def set_scene(self, scene: "Scene3D") -> None:
        self._scene = scene
        if self._initialized:
            self._rebuild_buffers()
            self.update()

    def update_morphs(self, figure_id: str) -> None:
        """Recompute morphed vertices and re-upload to GPU."""
        if not self._scene:
            return
        fig = self._scene.figure_by_id(figure_id)
        if not fig or fig.mesh.vertices.size == 0:
            return

        bufs = self._figure_buffers.get(figure_id)
        if not bufs or not bufs.valid:
            return

        # Start from rest vertices, apply active morphs
        verts = fig.mesh.vertices.copy()
        for morph in fig.morphs:
            if abs(morph.value) < 1e-6 or not morph.deltas:
                continue
            for vi, (dx, dy, dz) in morph.deltas.items():
                if vi < len(verts):
                    verts[vi] += np.array([dx, dy, dz], dtype=np.float32) * morph.value

        # Recompute normals
        from sdqt.models.daz_parser import DazParser
        normals = DazParser._compute_normals(verts, fig.mesh.indices)

        # Re-upload
        self.makeCurrent()
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, bufs.vbo_pos)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, bufs.vbo_norm)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, normals.nbytes, normals)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        self.doneCurrent()
        self.update()

    def update_pose(
        self, figure_id: str, bone_poses: dict[str, tuple[float, float, float]]
    ) -> None:
        if not self._scene:
            return
        fig = self._scene.figure_by_id(figure_id)
        if not fig:
            return
        self._pose_matrices[figure_id] = self._compute_pose_matrices(
            fig, bone_poses
        )
        self.update()

    def set_lights(self, lights: list["Light"]) -> None:
        self._lights = list(lights)
        self.update()

    def set_hdri(self, path: str) -> None:
        """Load an HDRI equirectangular image for the skybox."""
        if path == self._hdri_path:
            return
        self._hdri_path = path
        if not path or not _HAS_GL or not self._initialized:
            if self._hdri_tex:
                self.makeCurrent()
                GL.glDeleteTextures(1, [self._hdri_tex])
                self.doneCurrent()
                self._hdri_tex = 0
            self.update()
            return
        # Load as texture
        self.makeCurrent()
        tex = _load_texture_to_gl(path)
        if self._hdri_tex:
            GL.glDeleteTextures(1, [self._hdri_tex])
        self._hdri_tex = tex
        self.doneCurrent()
        self.update()

    def _get_texture(self, path: str) -> int:
        """Get or load a texture, returning GL texture ID (0 = none)."""
        if not path:
            return 0
        if path in self._texture_cache:
            return self._texture_cache[path]
        tex = _load_texture_to_gl(path)
        self._texture_cache[path] = tex
        return tex

    # -- Camera state ------------------------------------------------------

    def camera_position(self) -> np.ndarray:
        yaw = math.radians(self._cam_yaw)
        pitch = math.radians(self._cam_pitch)
        x = self._cam_dist * math.cos(pitch) * math.sin(yaw)
        y = self._cam_dist * math.sin(pitch)
        z = self._cam_dist * math.cos(pitch) * math.cos(yaw)
        return self._cam_target + np.array([x, y, z], dtype=np.float32)

    def set_camera_from(self, cam: "Camera") -> None:
        diff = cam.position - cam.target
        self._cam_dist = float(np.linalg.norm(diff))
        self._cam_target = cam.target.copy()
        self._cam_fov = cam.fov
        if self._cam_dist > 1e-4:
            d = diff / self._cam_dist
            self._cam_pitch = math.degrees(math.asin(float(np.clip(d[1], -1, 1))))
            self._cam_yaw = math.degrees(math.atan2(float(d[0]), float(d[2])))
        self.update()

    def get_camera_state(self) -> "Camera":
        from sdqt.models.scene3d import Camera
        return Camera(
            position=self.camera_position().copy(),
            target=self._cam_target.copy(),
            fov=self._cam_fov,
        )

    # -- Offscreen render --------------------------------------------------

    def render_to_image(self, width: int, height: int) -> QImage | None:
        if not _HAS_GL or not self._initialized:
            return None

        self.makeCurrent()
        try:
            fbo = GL.glGenFramebuffers(1)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)

            color_tex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, color_tex)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height,
                            0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                       GL.GL_TEXTURE_2D, color_tex, 0)

            depth_rb = GL.glGenRenderbuffers(1)
            GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth_rb)
            GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, width, height)
            GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
                                          GL.GL_RENDERBUFFER, depth_rb)

            if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
                logger.error("FBO incomplete")
                return None

            GL.glViewport(0, 0, width, height)
            self._render_scene(width, height)

            data = GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
            img = QImage(data, width, height, QImage.Format.Format_RGBA8888).mirrored(False, True)
            img = img.copy()

            GL.glDeleteFramebuffers(1, [fbo])
            GL.glDeleteTextures(1, [color_tex])
            GL.glDeleteRenderbuffers(1, [depth_rb])
            return img
        finally:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            self.doneCurrent()

    # -- GL lifecycle ------------------------------------------------------

    def initializeGL(self) -> None:
        if not _HAS_GL:
            return
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glClearColor(0.12, 0.12, 0.14, 1.0)

        self._pbr_prog = self._compile_program(_VERT_SRC, _FRAG_SRC)
        self._line_prog = self._compile_program(_LINE_VERT_SRC, _LINE_FRAG_SRC)
        self._grid_prog = self._compile_program(_GRID_VERT_SRC, _GRID_FRAG_SRC)
        self._sky_prog = self._compile_program(_SKY_VERT_SRC, _SKY_FRAG_SRC)
        self._shadow_prog = self._compile_program(_SHADOW_VERT_SRC, _SHADOW_FRAG_SRC)

        # Empty VAO for fullscreen triangle (no attributes, uses gl_VertexID)
        self._sky_vao = GL.glGenVertexArrays(1)

        # Shadow map FBO
        self._shadow_fbo = GL.glGenFramebuffers(1)
        self._shadow_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._shadow_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_DEPTH_COMPONENT24,
                        self._shadow_size, self._shadow_size, 0,
                        GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_COMPARE_MODE, GL.GL_NONE)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._shadow_fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT,
                                   GL.GL_TEXTURE_2D, self._shadow_tex, 0)
        GL.glDrawBuffer(GL.GL_NONE)
        GL.glReadBuffer(GL.GL_NONE)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        self._build_grid()

        self._bone_vao = GL.glGenVertexArrays(1)
        self._bone_vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._bone_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._bone_vbo)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 12, None)
        GL.glEnableVertexAttribArray(0)
        GL.glBindVertexArray(0)

        self._initialized = True
        if self._scene:
            self._rebuild_buffers()

    def paintGL(self) -> None:
        if not _HAS_GL:
            return
        self._render_scene(self.width(), self.height())

    def resizeGL(self, w: int, h: int) -> None:
        if _HAS_GL:
            GL.glViewport(0, 0, w, h)

    # -- Core render -------------------------------------------------------

    def _render_shadow_pass(self, light_vp: np.ndarray) -> None:
        """Render scene depth from light's POV into shadow map."""
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._shadow_fbo)
        GL.glViewport(0, 0, self._shadow_size, self._shadow_size)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glUseProgram(self._shadow_prog)

        if self._scene:
            for fig in self._scene.figures:
                bufs = self._figure_buffers.get(fig.id)
                if not bufs or not bufs.valid:
                    continue
                model = _translation_matrix(fig.position) @ _rotation_matrix(fig.rotation)
                mvp = light_vp @ model
                self._set_mat4(self._shadow_prog, "u_light_mvp", mvp)

                has_skin = fig.id in self._pose_matrices and len(self._pose_matrices[fig.id]) > 0
                self._set_bool(self._shadow_prog, "u_skinned", has_skin)
                if has_skin:
                    for i, mat in enumerate(self._pose_matrices[fig.id][:128]):
                        loc = GL.glGetUniformLocation(self._shadow_prog, f"u_bones[{i}]")
                        if loc >= 0:
                            GL.glUniformMatrix4fv(loc, 1, GL.GL_TRUE, mat.astype(np.float32))

                GL.glBindVertexArray(bufs.vao)
                GL.glDrawElements(GL.GL_TRIANGLES, bufs.num_indices, GL.GL_UNSIGNED_INT, None)
                GL.glBindVertexArray(0)

            for prop in self._scene.props:
                if not prop.visible:
                    continue
                bufs = self._figure_buffers.get(prop.id)
                if not bufs or not bufs.valid:
                    continue
                model = _translation_matrix(prop.position) @ _rotation_matrix(prop.rotation)
                scale_mat = np.eye(4, dtype=np.float32)
                scale_mat[0, 0] = prop.scale[0]
                scale_mat[1, 1] = prop.scale[1]
                scale_mat[2, 2] = prop.scale[2]
                model = model @ scale_mat
                mvp = light_vp @ model
                self._set_mat4(self._shadow_prog, "u_light_mvp", mvp)

                has_skin = bool(prop.parent_figure_id and prop.parent_figure_id in self._pose_matrices)
                self._set_bool(self._shadow_prog, "u_skinned", has_skin)
                if has_skin:
                    for i, mat in enumerate(self._pose_matrices[prop.parent_figure_id][:128]):
                        loc = GL.glGetUniformLocation(self._shadow_prog, f"u_bones[{i}]")
                        if loc >= 0:
                            GL.glUniformMatrix4fv(loc, 1, GL.GL_TRUE, mat.astype(np.float32))

                GL.glBindVertexArray(bufs.vao)
                GL.glDrawElements(GL.GL_TRIANGLES, bufs.num_indices, GL.GL_UNSIGNED_INT, None)
                GL.glBindVertexArray(0)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def _compute_light_vp(self) -> np.ndarray:
        """Compute orthographic light-space VP matrix for shadow mapping."""
        from sdqt.models.scene3d import Light as LightClass
        lights = self._lights or [LightClass()]
        d = lights[0].direction
        light_dir = np.array([d[0], d[1], d[2]], dtype=np.float32)
        light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-8)

        # Place light far along its reverse direction, looking at scene center
        light_pos = self._cam_target - light_dir * 15.0
        up = np.array([0, 1, 0], dtype=np.float32)
        if abs(np.dot(light_dir, up)) > 0.99:
            up = np.array([1, 0, 0], dtype=np.float32)
        light_view = _look_at(light_pos, self._cam_target, up)
        # Orthographic projection covering the scene
        s = 8.0  # half-extents
        light_proj = np.zeros((4, 4), dtype=np.float32)
        light_proj[0, 0] = 1.0 / s
        light_proj[1, 1] = 1.0 / s
        light_proj[2, 2] = -2.0 / 30.0
        light_proj[2, 3] = -(30.0 + 0.1) / (30.0 - 0.1)
        light_proj[3, 3] = 1.0
        return light_proj @ light_view

    def _render_scene(self, w: int, h: int) -> None:
        # Shadow pass
        light_vp = self._compute_light_vp()
        if self._shadow_prog and self._shadow_fbo and self._scene:
            self._render_shadow_pass(light_vp)

        GL.glViewport(0, 0, w, h)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        aspect = w / max(h, 1)
        proj = _perspective(self._cam_fov, aspect, 0.01, 100.0)
        cam_pos = self.camera_position()
        view = _look_at(cam_pos, self._cam_target, np.array([0, 1, 0], dtype=np.float32))
        vp = proj @ view

        # Draw HDRI skybox (behind everything)
        if self._hdri_tex and self._sky_prog:
            GL.glDepthMask(GL.GL_FALSE)
            GL.glUseProgram(self._sky_prog)
            inv_vp = np.linalg.inv(vp).astype(np.float32)
            self._set_mat4(self._sky_prog, "u_inv_vp", inv_vp)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._hdri_tex)
            GL.glUniform1i(GL.glGetUniformLocation(self._sky_prog, "u_hdri"), 0)
            GL.glBindVertexArray(self._sky_vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
            GL.glBindVertexArray(0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            GL.glDepthMask(GL.GL_TRUE)

        if self.show_grid:
            self._draw_grid(vp)

        if self._scene:
            # Opaque pass first
            for fig in self._scene.figures:
                self._draw_figure(fig, vp, cam_pos, transparent=False)
                if self.show_bones:
                    self._draw_bones(fig, vp)
            for prop in self._scene.props:
                if prop.visible:
                    self._draw_prop(prop, vp, cam_pos, transparent=False)
            # Transparent pass (depth write off, blending on)
            GL.glDepthMask(GL.GL_FALSE)
            for fig in self._scene.figures:
                self._draw_figure(fig, vp, cam_pos, transparent=True)
            for prop in self._scene.props:
                if prop.visible:
                    self._draw_prop(prop, vp, cam_pos, transparent=True)
            GL.glDepthMask(GL.GL_TRUE)

    def _draw_figure(self, fig: "Figure", vp: np.ndarray, cam_pos: np.ndarray,
                     transparent: bool = False) -> None:
        bufs = self._figure_buffers.get(fig.id)
        if not bufs or not bufs.valid:
            return

        model = _translation_matrix(fig.position) @ _rotation_matrix(fig.rotation)
        mvp = vp @ model
        normal_mat = np.linalg.inv(model[:3, :3]).T.astype(np.float32)

        GL.glUseProgram(self._pbr_prog)
        self._set_mat4(self._pbr_prog, "u_mvp", mvp)
        self._set_mat4(self._pbr_prog, "u_model", model)
        self._set_mat3(self._pbr_prog, "u_normal_mat", normal_mat)
        self._set_vec3(self._pbr_prog, "u_cam_pos", cam_pos)

        # Bone matrices
        has_skin = fig.id in self._pose_matrices and len(self._pose_matrices[fig.id]) > 0
        self._set_bool(self._pbr_prog, "u_skinned", has_skin)
        if has_skin:
            for i, mat in enumerate(self._pose_matrices[fig.id][:128]):
                loc = GL.glGetUniformLocation(self._pbr_prog, f"u_bones[{i}]")
                if loc >= 0:
                    GL.glUniformMatrix4fv(loc, 1, GL.GL_TRUE, mat.astype(np.float32))

        # Lights — prefer explicitly set lights, fall back to scene/defaults
        from sdqt.models.scene3d import Light as LightClass
        lights = self._lights or [LightClass()]
        if not self._lights and self._scene and self._scene.instances:
            inst_lights = self._scene.instances[0].lights
            if inst_lights:
                lights = inst_lights

        num_lights = min(len(lights), 2)
        GL.glUniform1i(GL.glGetUniformLocation(self._pbr_prog, "u_num_lights"), num_lights)
        for i, lt in enumerate(lights[:2]):
            self._set_vec3(self._pbr_prog, f"u_light_dir[{i}]",
                           np.asarray(lt.direction, dtype=np.float32))
            self._set_vec3(self._pbr_prog, f"u_light_color[{i}]",
                           np.array(lt.color, dtype=np.float32))
            GL.glUniform1f(GL.glGetUniformLocation(self._pbr_prog, f"u_light_intensity[{i}]"),
                           lt.intensity)

        amb = self._scene.environment.get("ambient_color", (0.15, 0.15, 0.18)) if self._scene else (0.15, 0.15, 0.18)
        self._set_vec3(self._pbr_prog, "u_ambient", np.array(amb, dtype=np.float32))

        self._bind_shadow_map()

        # Draw submeshes (opaque or transparent pass)
        GL.glBindVertexArray(bufs.vao)
        for sm in fig.mesh.submeshes:
            mat = fig.materials.get(sm.material_id)
            has_opacity = bool(mat and (mat.opacity_map or mat.base_color[3] < 0.99))
            if transparent != has_opacity:
                continue
            if mat:
                GL.glUniform4f(GL.glGetUniformLocation(self._pbr_prog, "u_base_color"),
                               *mat.base_color)
                GL.glUniform1f(GL.glGetUniformLocation(self._pbr_prog, "u_metallic"),
                               mat.metallic)
                GL.glUniform1f(GL.glGetUniformLocation(self._pbr_prog, "u_roughness"),
                               mat.roughness)
            else:
                GL.glUniform4f(GL.glGetUniformLocation(self._pbr_prog, "u_base_color"),
                               0.7, 0.7, 0.7, 1.0)
                GL.glUniform1f(GL.glGetUniformLocation(self._pbr_prog, "u_metallic"), 0.0)
                GL.glUniform1f(GL.glGetUniformLocation(self._pbr_prog, "u_roughness"), 0.5)

            self._bind_material_textures(mat)

            GL.glDrawElements(GL.GL_TRIANGLES, sm.index_count,
                              GL.GL_UNSIGNED_INT, GL.ctypes.c_void_p(sm.index_offset * 4))

            self._unbind_textures()
        GL.glBindVertexArray(0)

    def _bind_material_textures(self, mat) -> None:
        """Bind all PBR texture maps for a material."""
        tex_slots = [
            ("diffuse_map",   "u_diffuse_map",   "u_has_diffuse_map",   0),
            ("normal_map",    "u_normal_map",    "u_has_normal_map",    1),
            ("roughness_map", "u_roughness_map", "u_has_roughness_map", 2),
            ("metallic_map",  "u_metallic_map",  "u_has_metallic_map",  3),
            ("opacity_map",   "u_opacity_map",   "u_has_opacity_map",   4),
        ]
        for attr, sampler, flag, unit in tex_slots:
            path = getattr(mat, attr, "") if mat else ""
            tex = self._get_texture(path) if path else 0
            if tex:
                GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
                GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
                GL.glUniform1i(GL.glGetUniformLocation(self._pbr_prog, sampler), unit)
                self._set_bool(self._pbr_prog, flag, True)
            else:
                self._set_bool(self._pbr_prog, flag, False)

    def _bind_shadow_map(self) -> None:
        """Bind shadow map texture and light VP for the PBR shader."""
        if self._shadow_tex:
            GL.glActiveTexture(GL.GL_TEXTURE5)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._shadow_tex)
            GL.glUniform1i(GL.glGetUniformLocation(self._pbr_prog, "u_shadow_map"), 5)
            self._set_bool(self._pbr_prog, "u_has_shadow_map", True)
            light_vp = self._compute_light_vp()
            self._set_mat4(self._pbr_prog, "u_light_vp", light_vp)
        else:
            self._set_bool(self._pbr_prog, "u_has_shadow_map", False)

    @staticmethod
    def _unbind_textures() -> None:
        for unit in range(5, -1, -1):
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def _draw_prop(self, prop: "Prop", vp: np.ndarray, cam_pos: np.ndarray,
                   transparent: bool = False) -> None:
        """Draw a static (unskinned) prop mesh."""
        bufs = self._figure_buffers.get(prop.id)
        if not bufs or not bufs.valid:
            return

        # Build scale matrix manually for props
        model = _translation_matrix(prop.position) @ _rotation_matrix(prop.rotation)
        scale_mat = np.eye(4, dtype=np.float32)
        scale_mat[0, 0] = prop.scale[0]
        scale_mat[1, 1] = prop.scale[1]
        scale_mat[2, 2] = prop.scale[2]
        model = model @ scale_mat

        mvp = vp @ model
        normal_mat = np.linalg.inv(model[:3, :3]).T.astype(np.float32)

        GL.glUseProgram(self._pbr_prog)
        self._set_mat4(self._pbr_prog, "u_mvp", mvp)
        self._set_mat4(self._pbr_prog, "u_model", model)
        self._set_mat3(self._pbr_prog, "u_normal_mat", normal_mat)
        self._set_vec3(self._pbr_prog, "u_cam_pos", cam_pos)

        # Clothing conforming: use parent figure's bone matrices
        has_skin = False
        if prop.parent_figure_id and prop.parent_figure_id in self._pose_matrices:
            pose_mats = self._pose_matrices[prop.parent_figure_id]
            if pose_mats and prop.mesh.bone_weights.size > 0:
                has_skin = True
                for i, mat in enumerate(pose_mats[:128]):
                    loc = GL.glGetUniformLocation(self._pbr_prog, f"u_bones[{i}]")
                    if loc >= 0:
                        GL.glUniformMatrix4fv(loc, 1, GL.GL_TRUE, mat.astype(np.float32))
        self._set_bool(self._pbr_prog, "u_skinned", has_skin)

        # Lights (same as figures)
        from sdqt.models.scene3d import Light as LightClass
        lights = self._lights or [LightClass()]
        if not self._lights and self._scene and self._scene.instances:
            inst_lights = self._scene.instances[0].lights
            if inst_lights:
                lights = inst_lights

        num_lights = min(len(lights), 2)
        GL.glUniform1i(GL.glGetUniformLocation(self._pbr_prog, "u_num_lights"), num_lights)
        for i, lt in enumerate(lights[:2]):
            self._set_vec3(self._pbr_prog, f"u_light_dir[{i}]",
                           np.asarray(lt.direction, dtype=np.float32))
            self._set_vec3(self._pbr_prog, f"u_light_color[{i}]",
                           np.array(lt.color, dtype=np.float32))
            GL.glUniform1f(GL.glGetUniformLocation(self._pbr_prog, f"u_light_intensity[{i}]"),
                           lt.intensity)

        amb = self._scene.environment.get("ambient_color", (0.15, 0.15, 0.18)) if self._scene else (0.15, 0.15, 0.18)
        self._set_vec3(self._pbr_prog, "u_ambient", np.array(amb, dtype=np.float32))

        self._bind_shadow_map()

        GL.glBindVertexArray(bufs.vao)
        for sm in prop.mesh.submeshes:
            mat = prop.materials.get(sm.material_id)
            has_opacity = bool(mat and (mat.opacity_map or mat.base_color[3] < 0.99))
            if transparent != has_opacity:
                continue
            if mat:
                GL.glUniform4f(GL.glGetUniformLocation(self._pbr_prog, "u_base_color"),
                               *mat.base_color)
                GL.glUniform1f(GL.glGetUniformLocation(self._pbr_prog, "u_metallic"),
                               mat.metallic)
                GL.glUniform1f(GL.glGetUniformLocation(self._pbr_prog, "u_roughness"),
                               mat.roughness)
            else:
                GL.glUniform4f(GL.glGetUniformLocation(self._pbr_prog, "u_base_color"),
                               0.7, 0.7, 0.7, 1.0)
                GL.glUniform1f(GL.glGetUniformLocation(self._pbr_prog, "u_metallic"), 0.0)
                GL.glUniform1f(GL.glGetUniformLocation(self._pbr_prog, "u_roughness"), 0.5)

            self._bind_material_textures(mat)

            GL.glDrawElements(GL.GL_TRIANGLES, sm.index_count,
                              GL.GL_UNSIGNED_INT, GL.ctypes.c_void_p(sm.index_offset * 4))

            self._unbind_textures()
        GL.glBindVertexArray(0)

    def _draw_bones(self, fig: "Figure", vp: np.ndarray) -> None:
        if not fig.skeleton.bones:
            return
        positions = self._get_bone_world_positions(fig)
        if not positions:
            return

        lines: list[float] = []
        for i, bone in enumerate(fig.skeleton.bones):
            if bone.parent_index >= 0 and bone.parent_index < len(positions):
                p, c = positions[bone.parent_index], positions[i]
                lines.extend([p[0], p[1], p[2], c[0], c[1], c[2]])
        if not lines:
            return

        data = np.array(lines, dtype=np.float32)
        GL.glUseProgram(self._line_prog)
        self._set_mat4(self._line_prog, "u_mvp", vp)
        GL.glUniform4f(GL.glGetUniformLocation(self._line_prog, "u_color"),
                       0.0, 0.8, 1.0, 0.9)

        GL.glBindVertexArray(self._bone_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._bone_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_DYNAMIC_DRAW)
        GL.glLineWidth(2.0)
        GL.glDrawArrays(GL.GL_LINES, 0, len(lines) // 3)
        GL.glBindVertexArray(0)

    def _draw_grid(self, vp: np.ndarray) -> None:
        GL.glUseProgram(self._grid_prog)
        self._set_mat4(self._grid_prog, "u_mvp", vp)
        GL.glBindVertexArray(self._grid_vao)
        GL.glDrawArrays(GL.GL_LINES, 0, self._grid_count)
        GL.glBindVertexArray(0)

    # -- Buffer management -------------------------------------------------

    def _rebuild_buffers(self) -> None:
        for bufs in self._figure_buffers.values():
            if bufs.valid:
                GL.glDeleteVertexArrays(1, [bufs.vao])
                GL.glDeleteBuffers(5, [bufs.vbo_pos, bufs.vbo_norm, bufs.vbo_uv,
                                       bufs.vbo_bi, bufs.vbo_bw])
                GL.glDeleteBuffers(1, [bufs.ebo])
        self._figure_buffers.clear()

        if not self._scene:
            return

        for fig in self._scene.figures:
            if fig.mesh.vertices.size == 0:
                continue
            bufs = _MeshBuffers()
            bufs.vao = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(bufs.vao)

            nv = len(fig.mesh.vertices)

            # Position (loc 0)
            bufs.vbo_pos = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, bufs.vbo_pos)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, fig.mesh.vertices.nbytes,
                            fig.mesh.vertices, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(0)

            # Normal (loc 1)
            bufs.vbo_norm = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, bufs.vbo_norm)
            norms = fig.mesh.normals if fig.mesh.normals.size else np.zeros_like(fig.mesh.vertices)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, norms.nbytes, norms, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(1)

            # UV (loc 2)
            bufs.vbo_uv = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, bufs.vbo_uv)
            uvs = fig.mesh.uvs if fig.mesh.uvs.size else np.zeros((nv, 2), dtype=np.float32)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(2)

            # Bone indices (loc 3)
            bufs.vbo_bi = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, bufs.vbo_bi)
            bi = fig.mesh.bone_indices if fig.mesh.bone_indices.size else np.zeros((nv, 4), dtype=np.int32)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, bi.nbytes, bi, GL.GL_STATIC_DRAW)
            GL.glVertexAttribIPointer(3, 4, GL.GL_INT, 0, None)
            GL.glEnableVertexAttribArray(3)

            # Bone weights (loc 4)
            bufs.vbo_bw = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, bufs.vbo_bw)
            bw = fig.mesh.bone_weights if fig.mesh.bone_weights.size else np.zeros((nv, 4), dtype=np.float32)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, bw.nbytes, bw, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(4, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(4)

            # EBO
            bufs.ebo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, bufs.ebo)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, fig.mesh.indices.nbytes,
                            fig.mesh.indices, GL.GL_STATIC_DRAW)

            bufs.num_indices = len(fig.mesh.indices)
            bufs.valid = True
            GL.glBindVertexArray(0)
            self._figure_buffers[fig.id] = bufs

        # Props (same buffer layout, just no meaningful bone data)
        for prop in self._scene.props:
            if prop.mesh.vertices.size == 0:
                continue
            bufs = _MeshBuffers()
            bufs.vao = GL.glGenVertexArrays(1)
            GL.glBindVertexArray(bufs.vao)

            nv = len(prop.mesh.vertices)

            bufs.vbo_pos = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, bufs.vbo_pos)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, prop.mesh.vertices.nbytes,
                            prop.mesh.vertices, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(0)

            bufs.vbo_norm = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, bufs.vbo_norm)
            norms = prop.mesh.normals if prop.mesh.normals.size else np.zeros_like(prop.mesh.vertices)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, norms.nbytes, norms, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(1)

            bufs.vbo_uv = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, bufs.vbo_uv)
            uvs = prop.mesh.uvs if prop.mesh.uvs.size else np.zeros((nv, 2), dtype=np.float32)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, uvs.nbytes, uvs, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(2)

            # Bone indices/weights — use real data if conforming, else zeros
            bufs.vbo_bi = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, bufs.vbo_bi)
            bi = prop.mesh.bone_indices if prop.mesh.bone_indices.size else np.zeros((nv, 4), dtype=np.int32)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, bi.nbytes, bi, GL.GL_STATIC_DRAW)
            GL.glVertexAttribIPointer(3, 4, GL.GL_INT, 0, None)
            GL.glEnableVertexAttribArray(3)

            bufs.vbo_bw = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, bufs.vbo_bw)
            bw = prop.mesh.bone_weights if prop.mesh.bone_weights.size else np.zeros((nv, 4), dtype=np.float32)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, bw.nbytes, bw, GL.GL_STATIC_DRAW)
            GL.glVertexAttribPointer(4, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(4)

            bufs.ebo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, bufs.ebo)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, prop.mesh.indices.nbytes,
                            prop.mesh.indices, GL.GL_STATIC_DRAW)

            bufs.num_indices = len(prop.mesh.indices)
            bufs.valid = True
            GL.glBindVertexArray(0)
            self._figure_buffers[prop.id] = bufs

    def _build_grid(self) -> None:
        lines: list[float] = []
        extent = 10
        step = 0.5
        y = 0.0
        x = float(-extent)
        while x <= extent + 0.01:
            lines.extend([x, y, -extent, x, y, extent])
            lines.extend([-extent, y, x, extent, y, x])
            x += step

        data = np.array(lines, dtype=np.float32)
        self._grid_count = len(lines) // 3

        self._grid_vao = GL.glGenVertexArrays(1)
        self._grid_vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._grid_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._grid_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 12, None)
        GL.glEnableVertexAttribArray(0)
        GL.glBindVertexArray(0)

    # -- Pose computation --------------------------------------------------

    @staticmethod
    def _compute_pose_matrices(
        fig: "Figure", bone_poses: dict[str, tuple[float, float, float]]
    ) -> list[np.ndarray]:
        skel = fig.skeleton
        n = len(skel.bones)
        local_mats = []
        world_mats = [np.eye(4, dtype=np.float32) for _ in range(n)]

        for bone in skel.bones:
            t = _translation_matrix(bone.rest_position)
            r = _rotation_matrix(bone.rest_rotation)
            pose_rot = bone_poses.get(bone.name)
            if pose_rot is not None:
                r = r @ _rotation_matrix(np.array(pose_rot, dtype=np.float32))
            local_mats.append(t @ r)

        for i, bone in enumerate(skel.bones):
            if bone.parent_index >= 0:
                world_mats[i] = world_mats[bone.parent_index] @ local_mats[i]
            else:
                world_mats[i] = local_mats[i]

        rest_world = [np.eye(4, dtype=np.float32) for _ in range(n)]
        for i, bone in enumerate(skel.bones):
            rest_local = _translation_matrix(bone.rest_position) @ _rotation_matrix(bone.rest_rotation)
            if bone.parent_index >= 0:
                rest_world[i] = rest_world[bone.parent_index] @ rest_local
            else:
                rest_world[i] = rest_local

        return [world_mats[i] @ np.linalg.inv(rest_world[i]) for i in range(n)]

    def _get_bone_world_positions(self, fig: "Figure") -> list[np.ndarray]:
        skel = fig.skeleton
        n = len(skel.bones)
        if n == 0:
            return []
        positions: list[np.ndarray] = []
        world_mats = [np.eye(4, dtype=np.float32) for _ in range(n)]
        for i, bone in enumerate(skel.bones):
            local = _translation_matrix(bone.rest_position) @ _rotation_matrix(bone.rest_rotation)
            if bone.parent_index >= 0:
                world_mats[i] = world_mats[bone.parent_index] @ local
            else:
                world_mats[i] = local
            positions.append(world_mats[i][:3, 3] + fig.position)
        return positions

    # -- Mouse interaction -------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._last_mouse = event.position().toPoint()
        self._mouse_button = event.button()

        # Right-click: start IK drag if near a bone
        if event.button() == Qt.MouseButton.RightButton and self._scene:
            mx, my = event.position().toPoint().x(), event.position().toPoint().y()
            fig_id, bone_name = self._find_nearest_bone(mx, my, threshold=25.0)
            if bone_name:
                self._ik_dragging = True
                self._ik_fig_id = fig_id
                self._ik_bone_name = bone_name
                # Set up drag plane: camera-facing plane through the bone
                fig = self._scene.figure_by_id(fig_id)
                if fig:
                    positions = self._get_bone_world_positions(fig)
                    bi = fig.skeleton.bone_index(bone_name)
                    if 0 <= bi < len(positions):
                        self._ik_plane_point = positions[bi].copy()
                        cam_pos = self.camera_position()
                        self._ik_plane_normal = cam_pos - self._ik_plane_point
                        norm = np.linalg.norm(self._ik_plane_normal)
                        if norm > 1e-8:
                            self._ik_plane_normal /= norm

        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_mouse is None:
            return
        pos = event.position().toPoint()
        dx = pos.x() - self._last_mouse.x()
        dy = pos.y() - self._last_mouse.y()
        self._last_mouse = pos

        if self._ik_dragging and self._mouse_button == Qt.MouseButton.RightButton:
            # IK drag: project mouse to world via camera-facing plane
            from sdqt.models.ik_solver import screen_to_world_ray, project_to_plane
            w, h = self.width(), self.height()
            aspect = w / max(h, 1)
            proj_mat = _perspective(self._cam_fov, aspect, 0.01, 100.0)
            cam_pos = self.camera_position()
            view_mat = _look_at(cam_pos, self._cam_target, np.array([0, 1, 0], dtype=np.float32))
            vp_mat = proj_mat @ view_mat

            ray_o, ray_d = screen_to_world_ray(pos.x(), pos.y(), w, h, vp_mat, cam_pos)
            target = project_to_plane(ray_o, ray_d, self._ik_plane_point, self._ik_plane_normal)
            self.ik_drag.emit(self._ik_fig_id, self._ik_bone_name,
                              float(target[0]), float(target[1]), float(target[2]))
        elif self._mouse_button == Qt.MouseButton.LeftButton:
            self._cam_yaw += dx * 0.5
            self._cam_pitch = max(-89, min(89, self._cam_pitch + dy * 0.5))
            self.update()
        elif self._mouse_button == Qt.MouseButton.MiddleButton:
            scale = self._cam_dist * 0.002
            yaw = math.radians(self._cam_yaw)
            right = np.array([math.cos(yaw), 0, -math.sin(yaw)], dtype=np.float32)
            up = np.array([0, 1, 0], dtype=np.float32)
            self._cam_target -= right * dx * scale
            self._cam_target += up * dy * scale
            self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        # Detect click (no drag) for bone picking
        if (self._last_mouse is not None
                and event.button() == Qt.MouseButton.LeftButton):
            pos = event.position().toPoint()
            dx = abs(pos.x() - self._last_mouse.x())
            dy = abs(pos.y() - self._last_mouse.y())
            if dx < 4 and dy < 4:
                self._pick_bone(pos.x(), pos.y())
        self._ik_dragging = False
        self._last_mouse = None
        self._mouse_button = Qt.MouseButton.NoButton
        event.accept()

    def _find_nearest_bone(self, mx: int, my: int,
                           threshold: float = 20.0) -> tuple[str, str]:
        """Find nearest bone to screen position. Returns (figure_id, bone_name)."""
        if not self._scene:
            return "", ""
        w, h = self.width(), self.height()
        if w < 1 or h < 1:
            return "", ""

        aspect = w / h
        proj = _perspective(self._cam_fov, aspect, 0.01, 100.0)
        cam_pos = self.camera_position()
        view = _look_at(cam_pos, self._cam_target, np.array([0, 1, 0], dtype=np.float32))
        vp = proj @ view

        best_dist = threshold
        best_fig_id = ""
        best_bone_name = ""

        for fig in self._scene.figures:
            positions = self._get_bone_world_positions(fig)
            for i, bone in enumerate(fig.skeleton.bones):
                if i >= len(positions):
                    break
                wp = np.array([*positions[i], 1.0], dtype=np.float32)
                clip = vp @ wp
                if clip[3] <= 0:
                    continue
                ndc = clip[:3] / clip[3]
                sx = (ndc[0] * 0.5 + 0.5) * w
                sy = (1.0 - (ndc[1] * 0.5 + 0.5)) * h
                dist = math.sqrt((sx - mx) ** 2 + (sy - my) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_fig_id = fig.id
                    best_bone_name = bone.name

        return best_fig_id, best_bone_name

    def _pick_bone(self, mx: int, my: int) -> None:
        """Find nearest bone to screen click and emit bone_clicked."""
        fig_id, bone_name = self._find_nearest_bone(mx, my)
        if bone_name:
            self.bone_clicked.emit(fig_id, bone_name)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        self._cam_dist = max(0.1, min(50.0, self._cam_dist * factor))
        self.update()
        event.accept()

    # -- Shader helpers ----------------------------------------------------

    @staticmethod
    def _compile_program(vert_src: str, frag_src: str) -> int:
        vert = GL.glCreateShader(GL.GL_VERTEX_SHADER)
        GL.glShaderSource(vert, vert_src)
        GL.glCompileShader(vert)
        if not GL.glGetShaderiv(vert, GL.GL_COMPILE_STATUS):
            logger.error("Vertex shader error: %s", GL.glGetShaderInfoLog(vert).decode())
            return 0
        frag = GL.glCreateShader(GL.GL_FRAGMENT_SHADER)
        GL.glShaderSource(frag, frag_src)
        GL.glCompileShader(frag)
        if not GL.glGetShaderiv(frag, GL.GL_COMPILE_STATUS):
            logger.error("Fragment shader error: %s", GL.glGetShaderInfoLog(frag).decode())
            return 0
        prog = GL.glCreateProgram()
        GL.glAttachShader(prog, vert)
        GL.glAttachShader(prog, frag)
        GL.glLinkProgram(prog)
        if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
            logger.error("Program link error: %s", GL.glGetProgramInfoLog(prog).decode())
            return 0
        GL.glDeleteShader(vert)
        GL.glDeleteShader(frag)
        return prog

    @staticmethod
    def _set_mat4(prog: int, name: str, mat: np.ndarray) -> None:
        loc = GL.glGetUniformLocation(prog, name)
        if loc >= 0:
            GL.glUniformMatrix4fv(loc, 1, GL.GL_TRUE, mat.astype(np.float32))

    @staticmethod
    def _set_mat3(prog: int, name: str, mat: np.ndarray) -> None:
        loc = GL.glGetUniformLocation(prog, name)
        if loc >= 0:
            GL.glUniformMatrix3fv(loc, 1, GL.GL_TRUE, mat.astype(np.float32))

    @staticmethod
    def _set_vec3(prog: int, name: str, v: np.ndarray) -> None:
        loc = GL.glGetUniformLocation(prog, name)
        if loc >= 0:
            GL.glUniform3f(loc, float(v[0]), float(v[1]), float(v[2]))

    @staticmethod
    def _set_bool(prog: int, name: str, val: bool) -> None:
        loc = GL.glGetUniformLocation(prog, name)
        if loc >= 0:
            GL.glUniform1i(loc, int(val))


# ── Public widget wrapper ────────────────────────────────────────────────────

class PBRViewportWidget(QWidget):
    """High-level PBR 3D viewport for Daz2Supreme.

    Wraps _PBRCanvas with a fallback message if OpenGL is unavailable.
    """

    bone_selected = Signal(str, str)  # (figure_id, bone_name)
    ik_drag = Signal(str, str, float, float, float)  # (fig_id, bone, tx, ty, tz)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if _HAS_GL:
            self._canvas = _PBRCanvas()
            self._canvas.bone_clicked.connect(self.bone_selected)
            self._canvas.ik_drag.connect(self.ik_drag)
            layout.addWidget(self._canvas)
        else:
            self._canvas = None
            lbl = QLabel(
                "OpenGL not available.\n\n"
                "Install PyOpenGL:\n"
                "pip install PyOpenGL"
            )
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #888; font-size: 14px;")
            layout.addWidget(lbl)

    def set_scene(self, scene: "Scene3D") -> None:
        if self._canvas:
            self._canvas.set_scene(scene)

    def update_pose(self, figure_id: str, bone_poses: dict[str, tuple[float, float, float]]) -> None:
        if self._canvas:
            self._canvas.update_pose(figure_id, bone_poses)

    def update_morphs(self, figure_id: str) -> None:
        if self._canvas:
            self._canvas.update_morphs(figure_id)

    def set_camera_from(self, cam: "Camera") -> None:
        if self._canvas:
            self._canvas.set_camera_from(cam)

    def get_camera_state(self) -> "Camera | None":
        if self._canvas:
            return self._canvas.get_camera_state()
        return None

    def render_to_image(self, width: int, height: int) -> QImage | None:
        if self._canvas:
            return self._canvas.render_to_image(width, height)
        return None

    def set_lights(self, lights: list) -> None:
        if self._canvas:
            self._canvas.set_lights(lights)

    def set_hdri(self, path: str) -> None:
        if self._canvas:
            self._canvas.set_hdri(path)

    def set_show_bones(self, show: bool) -> None:
        if self._canvas:
            self._canvas.show_bones = show
            self._canvas.update()

    def set_show_grid(self, show: bool) -> None:
        if self._canvas:
            self._canvas.show_grid = show
            self._canvas.update()
