"""Iterative texture refinement — project SD-enhanced renders back onto UV maps.

Pipeline:
    1. For each camera angle, render the mesh to get a raw image
    2. Run SD img2img at low denoising to add surface detail
    3. Back-project enhanced pixels onto the UV texture map
    4. Mark painted UV regions to avoid overwriting
    5. Repeat from next angle
    6. Fill gaps with nearest-neighbor interpolation
    7. Save final albedo texture alongside the mesh
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _build_mvp(rot_x: float, rot_y: float, zoom: float,
               fov: float, aspect: float, near: float = 0.1, far: float = 100.0):
    """Build Model-View-Projection matrix matching the GL viewport camera."""
    # Perspective projection (matches gluPerspective)
    f = 1.0 / math.tan(math.radians(fov) / 2.0)
    proj = np.zeros((4, 4), dtype=np.float64)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2 * far * near) / (near - far)
    proj[3, 2] = -1.0

    # View: translate(0, 0, -zoom) → rotate(rot_x, X) → rotate(rot_y, Y)
    def _rot_x(angle):
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=np.float64)

    def _rot_y(angle):
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=np.float64)

    def _translate(x, y, z):
        m = np.eye(4, dtype=np.float64)
        m[0, 3] = x
        m[1, 3] = y
        m[2, 3] = z
        return m

    view = _translate(0, 0, -zoom) @ _rot_x(rot_x) @ _rot_y(rot_y)
    return proj @ view, view


def _project_vertices(verts: np.ndarray, mvp: np.ndarray,
                      width: int, height: int) -> np.ndarray:
    """Project 3D vertices to 2D screen coordinates.

    Returns (N, 3) array: [pixel_x, pixel_y, depth].
    """
    n = len(verts)
    hom = np.ones((n, 4), dtype=np.float64)
    hom[:, :3] = verts
    clip = (mvp @ hom.T).T  # (N, 4)
    w = clip[:, 3:4]
    w = np.where(np.abs(w) < 1e-8, 1e-8, w)
    ndc = clip[:, :3] / w  # normalized device coords [-1, 1]
    screen = np.empty((n, 3), dtype=np.float64)
    screen[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * width
    screen[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height  # flip Y
    screen[:, 2] = ndc[:, 2]  # depth for visibility
    return screen


class TextureRefiner:
    """Project SD-enhanced renders back onto a mesh's UV texture map.

    Usage::

        refiner = TextureRefiner(mesh_path, tex_size=1024)
        for angle in angles:
            raw_img = render_func(angle)
            enhanced_img = sd_enhance_func(raw_img)
            refiner.project_view(enhanced_img, angle, zoom=3.0, aspect=1.0)
        refiner.fill_gaps()
        output_path = refiner.save(output_dir)
    """

    def __init__(self, mesh_path: str, tex_size: int = 1024) -> None:
        import trimesh

        self._mesh_path = mesh_path
        self._tex_size = tex_size

        # Load mesh
        loaded = trimesh.load(mesh_path)
        if isinstance(loaded, trimesh.Scene):
            geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not geoms:
                raise ValueError(f"No meshes found in {mesh_path}")
            self._mesh = geoms[0]
        elif isinstance(loaded, trimesh.Trimesh):
            self._mesh = loaded
        else:
            raise ValueError(f"Unsupported mesh type: {type(loaded)}")

        # Center and scale to unit box (same as GLCanvas._add_trimesh)
        self._mesh.vertices -= self._mesh.centroid
        extents = self._mesh.extents
        max_ext = max(extents) if max(extents) > 0 else 1.0
        self._mesh.vertices /= max_ext

        # Ensure UVs exist
        if not hasattr(self._mesh.visual, 'uv') or self._mesh.visual.uv is None:
            raise ValueError("Mesh has no UV coordinates. Run xatlas UV unwrapping first.")

        self._uvs = np.array(self._mesh.visual.uv, dtype=np.float64)
        self._verts = np.array(self._mesh.vertices, dtype=np.float64)
        self._faces = np.array(self._mesh.faces, dtype=np.int32)

        # Texture accumulator: RGB float + weight for blending
        self._texture = np.zeros((tex_size, tex_size, 3), dtype=np.float64)
        self._weights = np.zeros((tex_size, tex_size), dtype=np.float64)

        # Build ray intersector for visibility testing
        self._ray_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(self._mesh)

    def project_view(
        self,
        image: np.ndarray,
        rot_x: float,
        rot_y: float,
        zoom: float = 3.0,
        aspect: float = 1.0,
        fov: float = 45.0,
    ) -> int:
        """Project an enhanced image onto the UV texture map.

        Args:
            image: RGB uint8 array (H, W, 3) — the SD-enhanced render.
            rot_x: Camera elevation angle (degrees).
            rot_y: Camera azimuth angle (degrees).
            zoom: Camera distance.
            aspect: Viewport aspect ratio.
            fov: Camera field of view.

        Returns:
            Number of UV texels written.
        """
        h, w = image.shape[:2]
        mvp, view = _build_mvp(rot_x, rot_y, zoom, fov, aspect)

        # Project all vertices to screen space
        screen = _project_vertices(self._verts, mvp, w, h)

        texels_written = 0
        ts = self._tex_size

        for face_idx in range(len(self._faces)):
            vi = self._faces[face_idx]
            s0, s1, s2 = screen[vi[0]], screen[vi[1]], screen[vi[2]]
            uv0, uv1, uv2 = self._uvs[vi[0]], self._uvs[vi[1]], self._uvs[vi[2]]

            # Face normal in view space (back-face culling)
            v0_view = (view @ np.append(self._verts[vi[0]], 1))[:3]
            v1_view = (view @ np.append(self._verts[vi[1]], 1))[:3]
            v2_view = (view @ np.append(self._verts[vi[2]], 1))[:3]
            face_normal = np.cross(v1_view - v0_view, v2_view - v0_view)
            if face_normal[2] >= 0:  # facing away from camera
                continue

            # Bounding box of the triangle in UV space
            uv_all = np.array([uv0, uv1, uv2])
            uv_min = np.floor(uv_all.min(axis=0) * ts).astype(int)
            uv_max = np.ceil(uv_all.max(axis=0) * ts).astype(int)
            uv_min = np.clip(uv_min, 0, ts - 1)
            uv_max = np.clip(uv_max, 0, ts - 1)

            # Iterate UV texels in this triangle's bounding box
            for ty in range(uv_min[1], uv_max[1] + 1):
                for tx in range(uv_min[0], uv_max[0] + 1):
                    # UV coordinate of this texel center
                    u = (tx + 0.5) / ts
                    v = (ty + 0.5) / ts

                    # Barycentric coords in UV space
                    bary = _barycentric_2d(u, v, uv0, uv1, uv2)
                    if bary is None:
                        continue

                    # Interpolate screen position
                    sx = bary[0] * s0[0] + bary[1] * s1[0] + bary[2] * s2[0]
                    sy = bary[0] * s0[1] + bary[1] * s1[1] + bary[2] * s2[1]

                    px = int(round(sx))
                    py = int(round(sy))
                    if px < 0 or px >= w or py < 0 or py >= h:
                        continue

                    # Sample the enhanced image
                    color = image[py, px].astype(np.float64) / 255.0

                    # Write to texture with blending (flipped V)
                    tex_y = ts - 1 - ty
                    self._texture[tex_y, tx] += color
                    self._weights[tex_y, tx] += 1.0
                    texels_written += 1

        return texels_written

    def fill_gaps(self) -> None:
        """Fill unpainted UV regions with nearest-neighbor interpolation."""
        mask = self._weights > 0
        if mask.all() or not mask.any():
            return

        from scipy.ndimage import distance_transform_edt

        # Normalize painted texels
        painted = mask.copy()
        for c in range(3):
            self._texture[:, :, c] = np.where(
                mask, self._texture[:, :, c] / self._weights, 0
            )

        # Use distance transform to find nearest painted pixel
        dist, indices = distance_transform_edt(~painted, return_distances=True, return_indices=True)
        for c in range(3):
            self._texture[:, :, c] = self._texture[indices[0], indices[1], c]

    def fill_gaps_simple(self) -> None:
        """Fallback gap-fill without scipy — iterative dilation."""
        mask = self._weights > 0
        if mask.all() or not mask.any():
            return

        # Normalize painted texels
        for c in range(3):
            self._texture[:, :, c] = np.where(
                mask, self._texture[:, :, c] / self._weights, 0
            )

        # Iterative dilation: spread painted pixels outward
        for _ in range(max(self._tex_size // 4, 32)):
            unpainted = ~mask
            if not unpainted.any():
                break
            # For each unpainted pixel, average its painted neighbors
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                shifted_mask = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
                can_fill = unpainted & shifted_mask
                if not can_fill.any():
                    continue
                for c in range(3):
                    shifted = np.roll(np.roll(self._texture[:, :, c], dy, axis=0), dx, axis=1)
                    self._texture[:, :, c] = np.where(can_fill, shifted, self._texture[:, :, c])
                mask |= can_fill

    def save(self, output_dir: str | None = None) -> str:
        """Save the texture as a PNG alongside the mesh.

        Returns the path to the saved texture.
        """
        # Use gap fill (try scipy, fall back to simple)
        try:
            self.fill_gaps()
        except ImportError:
            self.fill_gaps_simple()

        # Clamp and convert to uint8
        tex_uint8 = (np.clip(self._texture, 0, 1) * 255).astype(np.uint8)

        from PIL import Image
        tex_img = Image.fromarray(tex_uint8)

        if output_dir is None:
            output_dir = str(Path(self._mesh_path).parent)

        stem = Path(self._mesh_path).stem
        tex_path = Path(output_dir) / f"{stem}_albedo.png"
        tex_img.save(str(tex_path))
        logger.info("Refined texture saved: %s (%dx%d, coverage %.1f%%)",
                     tex_path, self._tex_size, self._tex_size,
                     100.0 * (self._weights > 0).mean())
        return str(tex_path)

    @property
    def coverage(self) -> float:
        """Fraction of UV texels that have been painted (0-1)."""
        return float((self._weights > 0).mean())


def _barycentric_2d(px, py, a, b, c):
    """Compute barycentric coordinates of point (px, py) in triangle (a, b, c).

    Returns (w0, w1, w2) or None if outside triangle.
    """
    v0 = c - a
    v1 = b - a
    v2 = np.array([px - a[0], py - a[1]], dtype=np.float64)

    dot00 = v0[0] * v0[0] + v0[1] * v0[1]
    dot01 = v0[0] * v1[0] + v0[1] * v1[1]
    dot02 = v0[0] * v2[0] + v0[1] * v2[1]
    dot11 = v1[0] * v1[0] + v1[1] * v1[1]
    dot12 = v1[0] * v2[0] + v1[1] * v2[1]

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-12:
        return None

    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv

    if u < -1e-6 or v < -1e-6 or (u + v) > 1.0 + 1e-6:
        return None

    return (1.0 - u - v, v, u)
