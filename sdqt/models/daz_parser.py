"""Parse Daz3D .duf/.dsf files into our internal Scene3D model.

Daz files are gzipped JSON (or plain JSON).  The parser extracts:
  - Geometry: vertices, polygons (quads→tris), UVs, polygon groups
  - Skeleton: bone hierarchy, rest transforms, rotation limits
  - Skin weights: general vertex weights → LBS bone_indices/bone_weights
  - Materials: Iray/3Delight surface nodes → PBR approximation
  - Morphs: vertex deltas with dial metadata

External .dsf asset references are resolved relative to a Daz content
directory if provided.

Usage:
    parser = DazParser(content_dirs=["/path/to/Daz3D Library"])
    figure = parser.parse_file("/path/to/Genesis9.duf")
"""

from __future__ import annotations

import gzip
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from sdqt.models.scene3d import (
    Bone,
    Camera,
    Figure,
    Light,
    Mesh,
    Morph,
    PBRMaterial,
    Prop,
    Skeleton,
    SubMesh,
)

logger = logging.getLogger(__name__)

# Daz bone name → body region for UI grouping
_REGION_MAP: dict[str, str] = {}
for _names, _region in [
    (("hip", "pelvis"), "hip"),
    (("abdomenLower", "abdomenUpper", "abdomen2", "chestLower", "chestUpper", "chest", "neckLower", "neckUpper", "neck"), "spine"),
    (("head",), "head"),
    (("lEye", "rEye", "lEyelidUpper", "lEyelidLower", "rEyelidUpper", "rEyelidLower",
      "lowerJaw", "upperJaw", "tongue", "tongueBase", "tongueTip"), "face"),
    (("lCollar", "lShldrBend", "lShldrTwist", "lForearmBend", "lForearmTwist", "lHand"), "lArm"),
    (("rCollar", "rShldrBend", "rShldrTwist", "rForearmBend", "rForearmTwist", "rHand"), "rArm"),
    (("lThighBend", "lThighTwist", "lShin", "lFoot", "lToe", "lMetatarsals"), "lLeg"),
    (("rThighBend", "rThighTwist", "rShin", "rFoot", "rToe", "rMetatarsals"), "rLeg"),
]:
    for _n in _names:
        _REGION_MAP[_n] = _region
# Finger bones
for _side in ("l", "r"):
    for _finger in ("Thumb", "Index", "Mid", "Ring", "Pinky"):
        for _seg in ("1", "2", "3"):
            _REGION_MAP[f"{_side}{_finger}{_seg}"] = f"{_side}Hand"


class DazParser:
    """Parse .duf / .dsf files into our internal Figure model."""

    def __init__(self, content_dirs: list[str | Path] | None = None) -> None:
        self.content_dirs: list[Path] = [Path(d) for d in (content_dirs or [])]
        self._asset_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse_file(self, path: str | Path) -> Figure:
        """Parse a .duf or .dsf and return a single Figure."""
        path = Path(path)
        data = self._load_json(path)
        if data is None:
            raise ValueError(f"Failed to load Daz file: {path}")

        data["_file_path"] = str(path)
        self._asset_cache["_current_file"] = str(path)

        fig = Figure(
            id=uuid.uuid4().hex[:12],
            name=path.stem,
            source_path=str(path),
        )

        # Geometry library — find the first geometry definition
        geom_lib = data.get("geometry_library", [])
        geom_data = geom_lib[0] if geom_lib else None

        # If this is a scene file (.duf), geometry may be referenced via url
        scene_nodes = data.get("scene", {}).get("nodes", [])
        if not geom_data and scene_nodes:
            geom_data, fig.name = self._resolve_geometry_from_scene(data, scene_nodes)

        if geom_data:
            fig.mesh = self._parse_geometry(geom_data)

        # Skeleton (node_library or scene nodes)
        node_lib = data.get("node_library", [])
        if node_lib:
            fig.skeleton = self._parse_skeleton_from_lib(node_lib)
        elif scene_nodes:
            fig.skeleton = self._parse_skeleton_from_scene(scene_nodes)

        # Skin binding (modifier_library)
        modifier_lib = data.get("modifier_library", [])
        self._apply_skin_weights(fig, modifier_lib)

        # Materials
        mat_lib = data.get("material_library", [])
        if mat_lib:
            fig.materials = self._parse_materials(mat_lib)

        # Morphs
        for mod in modifier_lib:
            morph = self._parse_morph(mod)
            if morph:
                fig.morphs.append(morph)

        # Assign bone regions
        for bone in fig.skeleton.bones:
            bone.region = self._guess_region(bone.name)

        logger.info(
            "Parsed %s: %d verts, %d tris, %d bones, %d morphs, %d materials",
            fig.name,
            len(fig.mesh.vertices),
            len(fig.mesh.indices) // 3,
            len(fig.skeleton.bones),
            len(fig.morphs),
            len(fig.materials),
        )
        return fig

    def parse_scene(self, path: str | Path) -> dict:
        """Parse a .duf scene file and return all objects.

        Returns dict with keys:
            figures: list[Figure]  — rigged characters
            props:   list[Prop]    — static mesh objects
            cameras: list[Camera]  — scene cameras
            lights:  list[Light]   — scene lights
        """
        path = Path(path)
        data = self._load_json(path)
        if data is None:
            raise ValueError(f"Failed to load Daz file: {path}")

        # Store file path for asset resolution
        data["_file_path"] = str(path)

        scene_nodes = data.get("scene", {}).get("nodes", [])
        geom_lib = data.get("geometry_library", [])
        node_lib = data.get("node_library", [])
        modifier_lib = data.get("modifier_library", [])
        mat_lib = data.get("material_library", [])

        # Build geometry lookup: id → geom_data
        geom_lookup: dict[str, dict] = {}
        for g in geom_lib:
            gid = g.get("id", "")
            if gid:
                geom_lookup[gid] = g

        figures: list[Figure] = []
        props: list[Prop] = []
        cameras: list[Camera] = []
        lights: list[Light] = []

        # Parse materials once (shared across objects)
        all_materials: dict[str, PBRMaterial] = {}
        if mat_lib:
            all_materials = self._parse_materials(mat_lib)

        # Classify scene nodes
        for node in scene_nodes:
            node_type = node.get("type", "")
            node_name = node.get("label", node.get("name", "Object"))

            if node_type == "camera":
                cam = self._parse_camera_node(node)
                if cam:
                    cameras.append(cam)
                continue

            if node_type == "light":
                lt = self._parse_light_node(node)
                if lt:
                    lights.append(lt)
                continue

            # Geometry node — figure or prop
            geom_url = ""
            if node.get("geometries"):
                geom_url = node["geometries"][0].get("url", "")

            if not geom_url:
                # Check if it's a figure node referencing geometry via node_library
                if node_type == "figure":
                    # Try to parse as single figure (legacy path)
                    pass
                continue

            # Resolve geometry
            geom_data = self._resolve_asset(geom_url, path)
            if not geom_data:
                # Try geom_lookup by fragment
                frag = geom_url.split("#")[-1] if "#" in geom_url else ""
                geom_data = geom_lookup.get(frag)
            if not geom_data:
                continue

            mesh = self._parse_geometry(geom_data)
            if mesh.vertices.size == 0:
                continue

            # Extract node transform
            pos = self._get_node_transform(node, "translation")
            rot = self._get_node_transform(node, "rotation")
            scl = self._get_node_transform(node, "scale", default=1.0)

            # Determine if this is a figure (has skeleton) or prop
            is_figure = node_type == "figure" or any(
                cn.get("type") == "bone" for cn in node.get("nodes", [])
            )

            if is_figure:
                fig = Figure(
                    id=uuid.uuid4().hex[:12],
                    name=node_name,
                    source_path=str(path),
                    mesh=mesh,
                    position=np.array(pos, dtype=np.float32),
                    rotation=np.array(rot, dtype=np.float32),
                    scale=np.array(scl, dtype=np.float32),
                )
                # Parse skeleton from child bone nodes
                bone_nodes = self._collect_bone_nodes(node)
                if bone_nodes:
                    fig.skeleton = self._parse_skeleton_from_lib(bone_nodes)
                elif node_lib:
                    fig.skeleton = self._parse_skeleton_from_lib(node_lib)
                # Skin weights
                self._apply_skin_weights(fig, modifier_lib)
                # Materials — match submesh IDs to parsed materials
                fig.materials = self._match_materials(mesh, all_materials)
                # Morphs
                for mod in modifier_lib:
                    morph = self._parse_morph(mod)
                    if morph:
                        fig.morphs.append(morph)
                # Assign bone regions
                for bone in fig.skeleton.bones:
                    bone.region = self._guess_region(bone.name)
                figures.append(fig)
                logger.info(
                    "Parsed figure %s: %d verts, %d tris, %d bones",
                    fig.name, len(fig.mesh.vertices),
                    len(fig.mesh.indices) // 3, len(fig.skeleton.bones),
                )
            else:
                prop = Prop(
                    id=uuid.uuid4().hex[:12],
                    name=node_name,
                    source_path=str(path),
                    mesh=mesh,
                    position=np.array(pos, dtype=np.float32),
                    rotation=np.array(rot, dtype=np.float32),
                    scale=np.array(scl, dtype=np.float32),
                )
                prop.materials = self._match_materials(mesh, all_materials)
                props.append(prop)
                logger.info(
                    "Parsed prop %s: %d verts, %d tris",
                    prop.name, len(prop.mesh.vertices),
                    len(prop.mesh.indices) // 3,
                )

        # Fallback: if no scene nodes parsed anything, try single-figure parse
        if not figures and not props:
            fig = self.parse_file(path)
            if fig.mesh.vertices.size > 0:
                figures.append(fig)

        return {
            "figures": figures,
            "props": props,
            "cameras": cameras,
            "lights": lights,
        }

    # ------------------------------------------------------------------
    # Scene node helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_node_transform(node: dict, key: str,
                            default: float = 0.0) -> list[float]:
        """Extract XYZ transform values from a scene node."""
        t = node.get(key, {})
        if isinstance(t, dict):
            return [
                t.get("x", {}).get("value", default) if isinstance(t.get("x"), dict) else default,
                t.get("y", {}).get("value", default) if isinstance(t.get("y"), dict) else default,
                t.get("z", {}).get("value", default) if isinstance(t.get("z"), dict) else default,
            ]
        return [default, default, default]

    @staticmethod
    def _collect_bone_nodes(node: dict) -> list[dict]:
        """Recursively collect bone child nodes from a figure node."""
        bones = []
        for child in node.get("nodes", []):
            if child.get("type") == "bone":
                bones.append(child)
                bones.extend(DazParser._collect_bone_nodes(child))
        return bones

    def _parse_camera_node(self, node: dict) -> Camera | None:
        """Extract camera from a Daz scene camera node."""
        pos = self._get_node_transform(node, "translation")
        rot = self._get_node_transform(node, "rotation")
        # Daz stores cameras with rotation; convert to position + target
        import math
        dist = 3.0
        ry = math.radians(rot[1])
        rx = math.radians(rot[0])
        target = np.array(pos, dtype=np.float32)
        eye = target + np.array([
            dist * math.cos(rx) * math.sin(ry),
            dist * math.sin(rx),
            dist * math.cos(rx) * math.cos(ry),
        ], dtype=np.float32)

        # FOV from camera properties
        fov = 45.0
        extras = node.get("extra", [])
        for extra in extras if isinstance(extras, list) else [extras]:
            if isinstance(extra, dict):
                fov_val = extra.get("focal_length", extra.get("fov", None))
                if isinstance(fov_val, (int, float)):
                    # Convert focal length (mm) to FOV (degrees) for 36mm sensor
                    if fov_val > 10:  # likely focal length in mm
                        fov = math.degrees(2 * math.atan(18.0 / fov_val))
                    else:
                        fov = fov_val

        return Camera(position=eye, target=target, fov=fov)

    def _parse_light_node(self, node: dict) -> Light | None:
        """Extract light from a Daz scene light node."""
        pos = self._get_node_transform(node, "translation")
        rot = self._get_node_transform(node, "rotation")

        # Determine light type
        kind = "directional"
        extras = node.get("extra", [])
        color = (1.0, 1.0, 1.0)
        intensity = 1.0
        for extra in extras if isinstance(extras, list) else [extras]:
            if isinstance(extra, dict):
                lt = extra.get("type", extra.get("light_type", ""))
                if "point" in str(lt).lower():
                    kind = "point"
                elif "spot" in str(lt).lower():
                    kind = "point"  # treat spot as point
                c = extra.get("color", None)
                if isinstance(c, list) and len(c) >= 3:
                    color = (c[0], c[1], c[2])
                i = extra.get("intensity", extra.get("luminous_flux", None))
                if isinstance(i, (int, float)):
                    intensity = float(i)

        import math
        ry = math.radians(rot[1])
        rx = math.radians(rot[0])
        direction = np.array([
            -math.cos(rx) * math.sin(ry),
            -math.sin(rx),
            -math.cos(rx) * math.cos(ry),
        ], dtype=np.float32)

        return Light(
            kind=kind,
            color=color,
            intensity=min(intensity, 5.0),  # clamp extreme values
            position=np.array(pos, dtype=np.float32),
            direction=direction,
        )

    @staticmethod
    def _match_materials(mesh: Mesh,
                         all_materials: dict[str, PBRMaterial]) -> dict[str, PBRMaterial]:
        """Match submesh material IDs to parsed materials."""
        matched: dict[str, PBRMaterial] = {}
        for sm in mesh.submeshes:
            if sm.material_id in all_materials:
                matched[sm.material_id] = all_materials[sm.material_id]
            else:
                # Try partial match
                for mid, mat in all_materials.items():
                    if sm.material_id in mid or mid in sm.material_id:
                        matched[sm.material_id] = mat
                        break
        return matched

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _load_json(self, path: Path) -> dict | None:
        """Load a .duf/.dsf — try gzip first, fall back to plain JSON."""
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        except (gzip.BadGzipFile, OSError):
            pass
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Cannot read %s: %s", path, exc)
            return None

    def _resolve_asset(self, url: str, base_path: Path) -> dict | None:
        """Resolve a Daz asset URL like '/data/DAZ 3D/Genesis9/Genesis9.dsf#geometry'.

        Returns the referenced object or the full file data.
        """
        if url in self._asset_cache:
            return self._asset_cache[url]

        # Split URL into file path and fragment
        parts = url.split("#", 1)
        rel_path = parts[0].lstrip("/")
        fragment = parts[1] if len(parts) > 1 else ""

        # Search content directories, then relative to base file
        candidates = [base_path.parent / rel_path]
        for cd in self.content_dirs:
            candidates.append(cd / rel_path)
            candidates.append(cd / "data" / rel_path)

        data = None
        for candidate in candidates:
            if candidate.is_file():
                data = self._load_json(candidate)
                if data:
                    break

        if not data:
            logger.debug("Could not resolve asset: %s", url)
            return None

        # Resolve fragment (e.g. #geometry → look in geometry_library by id)
        result = data
        if fragment:
            result = self._resolve_fragment(data, fragment)
            if result is None:
                result = data

        self._asset_cache[url] = result
        return result

    def _resolve_fragment(self, data: dict, fragment: str) -> dict | None:
        """Find an object by id within a Daz JSON file."""
        for lib_key in ("geometry_library", "node_library", "modifier_library",
                        "material_library", "image_library", "uv_set_library"):
            for item in data.get(lib_key, []):
                item_id = item.get("id", "")
                if item_id == fragment or item_id.endswith(f"/{fragment}"):
                    return item
        return None

    # ------------------------------------------------------------------
    # Geometry parsing
    # ------------------------------------------------------------------

    def _resolve_geometry_from_scene(
        self, data: dict, nodes: list[dict]
    ) -> tuple[dict | None, str]:
        """From scene nodes, resolve the geometry asset reference."""
        for node in nodes:
            geom_url = node.get("geometries", [{}])[0].get("url", "") if node.get("geometries") else ""
            if not geom_url:
                continue
            base_path = Path(data.get("_file_path", ""))
            geom = self._resolve_asset(geom_url, base_path)
            name = node.get("label", node.get("name", "Figure"))
            if geom:
                return geom, name
        return None, "Figure"

    def _parse_geometry(self, geom: dict) -> Mesh:
        """Extract vertices, faces, UVs from a Daz geometry object."""
        mesh = Mesh()

        # Vertices
        verts = geom.get("vertices", {}).get("values", [])
        if verts:
            mesh.vertices = np.array(verts, dtype=np.float32).reshape(-1, 3)

        # Polygons — Daz uses mixed quad/tri lists
        poly_list = geom.get("polylist", geom.get("polygon_list", {}))
        raw_polys = poly_list.get("values", [])
        poly_groups = poly_list.get("groups", [])

        indices = []
        group_tri_ranges: list[tuple[str, int, int]] = []  # (name, start_idx, count)

        for gi, group in enumerate(poly_groups):
            group_name = group if isinstance(group, str) else group.get("id", f"group_{gi}")
            start = len(indices)
            poly_indices = []
            # Get polygon indices for this group
            if isinstance(group, dict) and "polygons" in group:
                poly_indices = group["polygons"]
            elif isinstance(group, dict) and "polygon_range" in group:
                r = group["polygon_range"]
                poly_indices = list(range(r[0], r[0] + r[1]))

            for pi in poly_indices:
                if pi >= len(raw_polys):
                    continue
                poly = raw_polys[pi]
                self._triangulate_polygon(poly, indices)

            count = len(indices) - start
            if count > 0:
                group_tri_ranges.append((group_name, start, count))

        # If no groups, just triangulate all polygons
        if not group_tri_ranges and raw_polys:
            for poly in raw_polys:
                self._triangulate_polygon(poly, indices)

        if indices:
            mesh.indices = np.array(indices, dtype=np.uint32)

        # Build submeshes from polygon groups
        for gname, start, count in group_tri_ranges:
            mesh.submeshes.append(SubMesh(
                material_id=gname,
                index_offset=start,
                index_count=count,
            ))
        if not mesh.submeshes and len(mesh.indices) > 0:
            mesh.submeshes.append(SubMesh(
                material_id="default",
                index_offset=0,
                index_count=len(mesh.indices),
            ))

        # UVs
        uv_set = geom.get("default_uv_set", {})
        if isinstance(uv_set, str):
            # It's a reference URL — resolve it
            base_path = Path(self._asset_cache.get("_current_file", ""))
            resolved = self._resolve_asset(uv_set, base_path)
            if resolved and isinstance(resolved, dict):
                uv_values = resolved.get("values", [])
                if uv_values:
                    mesh.uvs = np.array(uv_values, dtype=np.float32).reshape(-1, 2)
        elif isinstance(uv_set, dict):
            uv_values = uv_set.get("values", [])
            if uv_values:
                mesh.uvs = np.array(uv_values, dtype=np.float32).reshape(-1, 2)

        # Compute normals if not provided
        if len(mesh.normals) == 0 and len(mesh.vertices) > 0 and len(mesh.indices) > 0:
            mesh.normals = self._compute_normals(mesh.vertices, mesh.indices)

        return mesh

    @staticmethod
    def _triangulate_polygon(poly: list[int], out: list[int]) -> None:
        """Fan-triangulate a polygon (works for tris and quads)."""
        n = len(poly)
        if n < 3:
            return
        for i in range(1, n - 1):
            out.extend([poly[0], poly[i], poly[i + 1]])

    @staticmethod
    def _compute_normals(vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Compute per-vertex normals from face normals."""
        normals = np.zeros_like(vertices)
        tris = indices.reshape(-1, 3)
        v0 = vertices[tris[:, 0]]
        v1 = vertices[tris[:, 1]]
        v2 = vertices[tris[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        # Accumulate face normals to vertices
        for i in range(3):
            np.add.at(normals, tris[:, i], face_normals)
        # Normalize
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths[lengths < 1e-8] = 1.0
        normals /= lengths
        return normals.astype(np.float32)

    # ------------------------------------------------------------------
    # Skeleton parsing
    # ------------------------------------------------------------------

    def _parse_skeleton_from_lib(self, node_lib: list[dict]) -> Skeleton:
        """Parse skeleton from node_library entries."""
        bones: list[Bone] = []
        name_to_idx: dict[str, int] = {}

        # First pass: collect all bone nodes
        bone_nodes = [n for n in node_lib if n.get("type", "") in ("bone", "figure")]
        # Sort: figures (roots) first, then bones
        bone_nodes.sort(key=lambda n: 0 if n.get("type") == "figure" else 1)

        for node in bone_nodes:
            name = node.get("name", node.get("id", f"bone_{len(bones)}"))
            parent_name = node.get("parent", "")
            # Strip URL prefix from parent ref
            if "/" in parent_name:
                parent_name = parent_name.rsplit("/", 1)[-1]
            if "#" in parent_name:
                parent_name = parent_name.rsplit("#", 1)[-1]

            parent_idx = name_to_idx.get(parent_name, -1)

            # Rest transform
            center = node.get("center_point", {}).get("values", [0, 0, 0])
            rotation = node.get("rotation", {})
            rot_vals = [
                rotation.get("x", {}).get("value", 0),
                rotation.get("y", {}).get("value", 0),
                rotation.get("z", {}).get("value", 0),
            ]
            # Rotation limits
            rot_min = rot_max = None
            if "x" in rotation and "min" in rotation["x"]:
                rot_min = np.array([
                    rotation.get("x", {}).get("min", -180),
                    rotation.get("y", {}).get("min", -180),
                    rotation.get("z", {}).get("min", -180),
                ], dtype=np.float32)
                rot_max = np.array([
                    rotation.get("x", {}).get("max", 180),
                    rotation.get("y", {}).get("max", 180),
                    rotation.get("z", {}).get("max", 180),
                ], dtype=np.float32)

            bone = Bone(
                name=name,
                parent_index=parent_idx,
                rest_position=np.array(center, dtype=np.float32),
                rest_rotation=np.array(rot_vals, dtype=np.float32),
                rotation_min=rot_min,
                rotation_max=rot_max,
            )
            name_to_idx[name] = len(bones)
            bones.append(bone)

        return Skeleton(bones=bones)

    def _parse_skeleton_from_scene(self, nodes: list[dict]) -> Skeleton:
        """Parse skeleton from scene node entries (same structure, different location)."""
        return self._parse_skeleton_from_lib(nodes)

    # ------------------------------------------------------------------
    # Skin weights
    # ------------------------------------------------------------------

    def _apply_skin_weights(self, fig: Figure, modifier_lib: list[dict]) -> None:
        """Extract skin binding weights and apply to mesh."""
        if not fig.mesh.vertices.size or not fig.skeleton.bones:
            return

        num_verts = len(fig.mesh.vertices)
        # Collect per-vertex weight lists: vert_idx → [(bone_idx, weight), ...]
        vert_weights: list[list[tuple[int, float]]] = [[] for _ in range(num_verts)]

        for mod in modifier_lib:
            if mod.get("type") != "skin":
                # Also check for "skin_binding" in older formats
                if "skin" not in mod.get("id", "").lower():
                    continue

            joints = mod.get("skin", {}).get("joints", [])
            if not joints:
                joints = mod.get("joints", [])

            for joint in joints:
                bone_name = joint.get("id", joint.get("node", ""))
                if "/" in bone_name:
                    bone_name = bone_name.rsplit("/", 1)[-1]
                if "#" in bone_name:
                    bone_name = bone_name.rsplit("#", 1)[-1]
                if ":" in bone_name:
                    bone_name = bone_name.rsplit(":", 1)[-1]

                bone_idx = fig.skeleton.bone_index(bone_name)
                if bone_idx < 0:
                    continue

                # Node weights
                node_weights = joint.get("node_weights", {})
                values = node_weights.get("values", [])
                for entry in values:
                    if isinstance(entry, list) and len(entry) >= 2:
                        vi = int(entry[0])
                        w = float(entry[1])
                        if 0 <= vi < num_verts and w > 0:
                            vert_weights[vi].append((bone_idx, w))

                # Local weights (alternative format)
                local_weights = joint.get("local_weights", {})
                for _axis, axis_data in local_weights.items():
                    values = axis_data.get("values", []) if isinstance(axis_data, dict) else []
                    for entry in values:
                        if isinstance(entry, list) and len(entry) >= 2:
                            vi = int(entry[0])
                            w = float(entry[1])
                            if 0 <= vi < num_verts and w > 0:
                                vert_weights[vi].append((bone_idx, w))

        # Convert to fixed 4-influence arrays
        bone_indices = np.zeros((num_verts, 4), dtype=np.int32)
        bone_weights_arr = np.zeros((num_verts, 4), dtype=np.float32)

        for vi, wlist in enumerate(vert_weights):
            if not wlist:
                continue
            # Sort by weight descending, take top 4
            wlist.sort(key=lambda x: x[1], reverse=True)
            top = wlist[:4]
            total = sum(w for _, w in top)
            if total < 1e-8:
                continue
            for j, (bi, w) in enumerate(top):
                bone_indices[vi, j] = bi
                bone_weights_arr[vi, j] = w / total  # normalize

        fig.mesh.bone_indices = bone_indices
        fig.mesh.bone_weights = bone_weights_arr

    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------

    def _parse_materials(self, mat_lib: list[dict]) -> dict[str, PBRMaterial]:
        """Extract PBR material approximations from Daz material library."""
        materials: dict[str, PBRMaterial] = {}

        for mat_group in mat_lib:
            for mat_data in mat_group.get("materials", [mat_group]):
                mat_id = mat_data.get("id", mat_data.get("name", f"mat_{len(materials)}"))
                mat = PBRMaterial(name=mat_id)

                # Diffuse / base color
                diffuse = mat_data.get("diffuse", mat_data.get("base_color", {}))
                if isinstance(diffuse, dict):
                    color = diffuse.get("value", diffuse.get("color", None))
                    if isinstance(color, list) and len(color) >= 3:
                        mat.base_color = (color[0], color[1], color[2], 1.0)
                    img = diffuse.get("image", diffuse.get("map", ""))
                    if isinstance(img, dict):
                        img = img.get("value", img.get("url", ""))
                    if img:
                        mat.diffuse_map = str(img)

                # Roughness / glossiness
                rough = mat_data.get("specular_roughness",
                        mat_data.get("glossy_roughness",
                        mat_data.get("roughness", {})))
                if isinstance(rough, dict):
                    val = rough.get("value", 0.5)
                    if isinstance(val, (int, float)):
                        mat.roughness = float(val)

                # Metallic
                metallic = mat_data.get("metallic_weight",
                           mat_data.get("metallicity", {}))
                if isinstance(metallic, dict):
                    val = metallic.get("value", 0.0)
                    if isinstance(val, (int, float)):
                        mat.metallic = float(val)

                # Normal map
                normal = mat_data.get("normal_map",
                         mat_data.get("bump", {}))
                if isinstance(normal, dict):
                    img = normal.get("image", normal.get("map", ""))
                    if isinstance(img, dict):
                        img = img.get("value", img.get("url", ""))
                    if img:
                        mat.normal_map = str(img)

                materials[mat_id] = mat

        return materials

    # ------------------------------------------------------------------
    # Morphs
    # ------------------------------------------------------------------

    def _parse_morph(self, mod: dict) -> Morph | None:
        """Parse a morph modifier into our Morph model."""
        if mod.get("type") not in ("morph", None):
            # Check for morph-like modifiers
            if "morph" not in mod.get("id", "").lower():
                return None

        morph_data = mod.get("morph", {})
        deltas_raw = morph_data.get("deltas", {}).get("values", [])
        if not deltas_raw:
            return None

        name = mod.get("name", mod.get("id", "morph"))
        group = mod.get("group", mod.get("region", ""))

        deltas: dict[int, tuple[float, float, float]] = {}
        for entry in deltas_raw:
            if isinstance(entry, list) and len(entry) >= 4:
                vi = int(entry[0])
                deltas[vi] = (float(entry[1]), float(entry[2]), float(entry[3]))

        channel = mod.get("channel", {})
        min_val = channel.get("min", 0.0)
        max_val = channel.get("max", 1.0)
        cur_val = channel.get("value", 0.0)

        return Morph(
            name=name,
            group=group,
            deltas=deltas,
            value=cur_val,
            min_value=min_val,
            max_value=max_val,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _guess_region(bone_name: str) -> str:
        """Map a bone name to a body region for UI grouping."""
        if bone_name in _REGION_MAP:
            return _REGION_MAP[bone_name]
        # Heuristic fallbacks
        lower = bone_name.lower()
        if "thumb" in lower or "index" in lower or "mid" in lower or "ring" in lower or "pinky" in lower:
            return "lHand" if lower.startswith("l") else "rHand"
        if "eye" in lower or "jaw" in lower or "lip" in lower or "brow" in lower or "nose" in lower:
            return "face"
        if "collar" in lower or "shldr" in lower or "forearm" in lower or "hand" in lower:
            return "lArm" if lower.startswith("l") else "rArm"
        if "thigh" in lower or "shin" in lower or "foot" in lower or "toe" in lower:
            return "lLeg" if lower.startswith("l") else "rLeg"
        if "spine" in lower or "abdomen" in lower or "chest" in lower or "neck" in lower:
            return "spine"
        if "head" in lower:
            return "head"
        if "hip" in lower or "pelvis" in lower:
            return "hip"
        return "other"
