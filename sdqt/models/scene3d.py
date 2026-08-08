"""Internal 3D scene model — modern representations for imported Daz assets.

Geometry, skeleton, materials, morphs, and poses are stored as clean,
standard dataclasses.  No Daz-specific cruft leaks past the parser.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass
class SubMesh:
    """A contiguous range of triangles sharing one material."""

    material_id: str = ""
    index_offset: int = 0   # into parent Mesh.indices
    index_count: int = 0


@dataclass
class Mesh:
    """Indexed triangle mesh with optional UV and normals.

    All arrays are numpy for GPU upload efficiency.
    """

    vertices: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    normals: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    uvs: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    indices: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.uint32))

    # Per-vertex bone weights (max 4 influences)
    bone_indices: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=np.int32))
    bone_weights: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=np.float32))

    submeshes: list[SubMesh] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Skeleton / Bones
# ---------------------------------------------------------------------------

@dataclass
class Bone:
    """Single bone in a skeleton hierarchy."""

    name: str = ""
    parent_index: int = -1          # -1 = root
    rest_position: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    rest_rotation: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)  # Euler XYZ degrees
    )
    rest_scale: np.ndarray = field(
        default_factory=lambda: np.ones(3, dtype=np.float32)
    )
    # Limits (degrees) — None means unconstrained
    rotation_min: np.ndarray | None = None   # (3,) XYZ
    rotation_max: np.ndarray | None = None   # (3,) XYZ

    # Body region tag for UI grouping (head, spine, lArm, rArm, lLeg, rLeg, etc.)
    region: str = ""


@dataclass
class Skeleton:
    """Ordered list of bones forming a hierarchy."""

    bones: list[Bone] = field(default_factory=list)

    def bone_by_name(self, name: str) -> Bone | None:
        for b in self.bones:
            if b.name == name:
                return b
        return None

    def bone_index(self, name: str) -> int:
        for i, b in enumerate(self.bones):
            if b.name == name:
                return i
        return -1

    @property
    def bone_names(self) -> list[str]:
        return [b.name for b in self.bones]

    def children_of(self, index: int) -> list[int]:
        return [i for i, b in enumerate(self.bones) if b.parent_index == index]


# ---------------------------------------------------------------------------
# Materials (PBR metallic-roughness)
# ---------------------------------------------------------------------------

@dataclass
class PBRMaterial:
    """Standard PBR material — maps directly to a simple GL shader."""

    name: str = "default"
    base_color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    metallic: float = 0.0
    roughness: float = 0.5
    # Texture paths (empty = no texture)
    diffuse_map: str = ""
    normal_map: str = ""
    roughness_map: str = ""
    metallic_map: str = ""
    opacity_map: str = ""


# ---------------------------------------------------------------------------
# Morphs (blendshapes)
# ---------------------------------------------------------------------------

@dataclass
class Morph:
    """Sparse vertex deltas driven by a single float dial."""

    name: str = ""
    group: str = ""  # e.g. "Head", "Body", "Expressions"
    # Sparse deltas: dict mapping vertex index → (dx, dy, dz)
    deltas: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    value: float = 0.0  # current dial value 0..1
    min_value: float = 0.0
    max_value: float = 1.0


# ---------------------------------------------------------------------------
# Figure (mesh + skeleton + materials + morphs = one importable asset)
# ---------------------------------------------------------------------------

@dataclass
class Figure:
    """A rigged, morphable 3D figure (e.g. Genesis 9, a prop, clothing)."""

    id: str = ""
    name: str = ""
    source_path: str = ""       # original .duf/.dsf path
    mesh: Mesh = field(default_factory=Mesh)
    skeleton: Skeleton = field(default_factory=Skeleton)
    materials: dict[str, PBRMaterial] = field(default_factory=dict)
    morphs: list[Morph] = field(default_factory=list)
    # World transform
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))

    def morph_by_name(self, name: str) -> Morph | None:
        for m in self.morphs:
            if m.name == name:
                return m
        return None


# ---------------------------------------------------------------------------
# Prop (static mesh — no skeleton, no morphs)
# ---------------------------------------------------------------------------

@dataclass
class Prop:
    """A static mesh object (environment piece, furniture, clothing prop)."""

    id: str = ""
    name: str = ""
    source_path: str = ""
    mesh: Mesh = field(default_factory=Mesh)
    materials: dict[str, PBRMaterial] = field(default_factory=dict)
    visible: bool = True
    # Clothing conforming: if set, this prop follows a figure's skeleton
    parent_figure_id: str = ""
    # World transform
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    rotation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    scale: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=np.float32))


# ---------------------------------------------------------------------------
# Lights / Camera
# ---------------------------------------------------------------------------

@dataclass
class Light:
    """Scene light (point, directional, or ambient)."""

    kind: str = "directional"  # "point", "directional", "ambient"
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    position: np.ndarray = field(default_factory=lambda: np.array([2, 4, 3], dtype=np.float32))
    direction: np.ndarray = field(default_factory=lambda: np.array([-0.3, -0.8, -0.5], dtype=np.float32))


@dataclass
class Camera:
    """Orbital camera state."""

    position: np.ndarray = field(default_factory=lambda: np.array([0, 1.2, 3.0], dtype=np.float32))
    target: np.ndarray = field(default_factory=lambda: np.array([0, 1.0, 0], dtype=np.float32))
    fov: float = 45.0
    near: float = 0.01
    far: float = 100.0


# ---------------------------------------------------------------------------
# Scene & Scene Instance
# ---------------------------------------------------------------------------

@dataclass
class SceneInstance:
    """A snapshot of poses, camera, and lights — a 'saved state' of a scene."""

    name: str = "default"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: float = field(default_factory=time.time)
    camera: Camera = field(default_factory=Camera)
    lights: list[Light] = field(default_factory=lambda: [Light()])
    # figure_id → {bone_name: (rx, ry, rz) in degrees}
    figure_poses: dict[str, dict[str, tuple[float, float, float]]] = field(default_factory=dict)
    # figure_id → {morph_name: value}
    figure_morphs: dict[str, dict[str, float]] = field(default_factory=dict)
    render_width: int = 1024
    render_height: int = 1024
    thumbnail: str = ""  # relative path to thumbnail image


@dataclass
class Scene3D:
    """Top-level scene: figures + environment + saved instances."""

    name: str = "Untitled Scene"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    figures: list[Figure] = field(default_factory=list)
    props: list[Prop] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=lambda: {
        "ground_plane": True,
        "hdri": "",
        "ambient_color": (0.15, 0.15, 0.18),
    })
    instances: list[SceneInstance] = field(default_factory=list)

    def figure_by_id(self, fid: str) -> Figure | None:
        for f in self.figures:
            if f.id == fid:
                return f
        return None

    def prop_by_id(self, pid: str) -> Prop | None:
        for p in self.props:
            if p.id == pid:
                return p
        return None

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize scene metadata (not heavy mesh data — that goes to .npz)."""
        return {
            "name": self.name,
            "id": self.id,
            "figures": [
                {
                    "id": f.id,
                    "name": f.name,
                    "source_path": f.source_path,
                    "position": f.position.tolist(),
                    "rotation": f.rotation.tolist(),
                    "scale": f.scale.tolist(),
                    "materials": {
                        mid: {
                            "name": m.name,
                            "base_color": list(m.base_color),
                            "metallic": m.metallic,
                            "roughness": m.roughness,
                            "diffuse_map": m.diffuse_map,
                            "normal_map": m.normal_map,
                            "roughness_map": m.roughness_map,
                            "metallic_map": m.metallic_map,
                            "opacity_map": m.opacity_map,
                        }
                        for mid, m in f.materials.items()
                    },
                    "morphs": [
                        {
                            "name": mo.name,
                            "group": mo.group,
                            "min_value": mo.min_value,
                            "max_value": mo.max_value,
                        }
                        for mo in f.morphs
                    ],
                }
                for f in self.figures
            ],
            "props": [
                {
                    "id": p.id,
                    "name": p.name,
                    "source_path": p.source_path,
                    "visible": p.visible,
                    "parent_figure_id": p.parent_figure_id,
                    "position": p.position.tolist(),
                    "rotation": p.rotation.tolist(),
                    "scale": p.scale.tolist(),
                    "materials": {
                        mid: {
                            "name": m.name,
                            "base_color": list(m.base_color),
                            "metallic": m.metallic,
                            "roughness": m.roughness,
                            "diffuse_map": m.diffuse_map,
                            "normal_map": m.normal_map,
                            "roughness_map": m.roughness_map,
                            "metallic_map": m.metallic_map,
                            "opacity_map": m.opacity_map,
                        }
                        for mid, m in p.materials.items()
                    },
                }
                for p in self.props
            ],
            "environment": self.environment,
        }

    def save_meta(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    def save_figure_data(self, scene_dir: Path) -> None:
        """Save heavy figure data (mesh, skeleton, morph deltas) to .npz."""
        fig_dir = scene_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        for fig in self.figures:
            if fig.mesh.vertices.size == 0:
                continue
            arrays: dict[str, Any] = {
                "vertices": fig.mesh.vertices,
                "normals": fig.mesh.normals,
                "uvs": fig.mesh.uvs,
                "indices": fig.mesh.indices,
                "bone_indices": fig.mesh.bone_indices,
                "bone_weights": fig.mesh.bone_weights,
            }
            # Skeleton as structured arrays
            if fig.skeleton.bones:
                bone_names = [b.name for b in fig.skeleton.bones]
                bone_parents = np.array([b.parent_index for b in fig.skeleton.bones], dtype=np.int32)
                bone_positions = np.array([b.rest_position for b in fig.skeleton.bones], dtype=np.float32)
                bone_rotations = np.array([b.rest_rotation for b in fig.skeleton.bones], dtype=np.float32)
                bone_scales = np.array([b.rest_scale for b in fig.skeleton.bones], dtype=np.float32)
                bone_regions = [b.region for b in fig.skeleton.bones]
                arrays["bone_names"] = np.array(bone_names)
                arrays["bone_parents"] = bone_parents
                arrays["bone_positions"] = bone_positions
                arrays["bone_rotations"] = bone_rotations
                arrays["bone_scales"] = bone_scales
                arrays["bone_regions"] = np.array(bone_regions)
                # Rotation limits (pack into one array, -999 = unconstrained)
                rot_mins = []
                rot_maxs = []
                for b in fig.skeleton.bones:
                    rot_mins.append(b.rotation_min if b.rotation_min is not None
                                    else np.full(3, -999, dtype=np.float32))
                    rot_maxs.append(b.rotation_max if b.rotation_max is not None
                                    else np.full(3, -999, dtype=np.float32))
                arrays["bone_rot_mins"] = np.array(rot_mins, dtype=np.float32)
                arrays["bone_rot_maxs"] = np.array(rot_maxs, dtype=np.float32)
            # Submeshes
            if fig.mesh.submeshes:
                sm_ids = [s.material_id for s in fig.mesh.submeshes]
                sm_offsets = np.array([s.index_offset for s in fig.mesh.submeshes], dtype=np.int64)
                sm_counts = np.array([s.index_count for s in fig.mesh.submeshes], dtype=np.int64)
                arrays["submesh_ids"] = np.array(sm_ids)
                arrays["submesh_offsets"] = sm_offsets
                arrays["submesh_counts"] = sm_counts
            # Morph deltas (sparse — pack as index + delta arrays per morph)
            for mi, morph in enumerate(fig.morphs):
                if morph.deltas:
                    idxs = np.array(list(morph.deltas.keys()), dtype=np.int32)
                    vals = np.array(list(morph.deltas.values()), dtype=np.float32)
                    arrays[f"morph_{mi}_idx"] = idxs
                    arrays[f"morph_{mi}_val"] = vals

            np.savez_compressed(fig_dir / f"{fig.id}.npz", **arrays)

        # Save prop mesh data
        for prop in self.props:
            if prop.mesh.vertices.size == 0:
                continue
            arrays: dict[str, Any] = {
                "vertices": prop.mesh.vertices,
                "normals": prop.mesh.normals,
                "uvs": prop.mesh.uvs,
                "indices": prop.mesh.indices,
                "bone_indices": prop.mesh.bone_indices,
                "bone_weights": prop.mesh.bone_weights,
            }
            if prop.mesh.submeshes:
                sm_ids = [s.material_id for s in prop.mesh.submeshes]
                sm_offsets = np.array([s.index_offset for s in prop.mesh.submeshes], dtype=np.int64)
                sm_counts = np.array([s.index_count for s in prop.mesh.submeshes], dtype=np.int64)
                arrays["submesh_ids"] = np.array(sm_ids)
                arrays["submesh_offsets"] = sm_offsets
                arrays["submesh_counts"] = sm_counts
            np.savez_compressed(fig_dir / f"prop_{prop.id}.npz", **arrays)

    @classmethod
    def load_meta(cls, path: Path) -> "Scene3D":
        data = json.loads(path.read_text())
        scene = cls(name=data.get("name", ""), id=data.get("id", ""))
        scene.environment = data.get("environment", scene.environment)
        fig_dir = path.parent / "figures"

        for fd in data.get("figures", []):
            fig = Figure(
                id=fd["id"],
                name=fd.get("name", ""),
                source_path=fd.get("source_path", ""),
                position=np.array(fd.get("position", [0, 0, 0]), dtype=np.float32),
                rotation=np.array(fd.get("rotation", [0, 0, 0]), dtype=np.float32),
                scale=np.array(fd.get("scale", [1, 1, 1]), dtype=np.float32),
            )
            # Restore materials from JSON
            for mid, md in fd.get("materials", {}).items():
                fig.materials[mid] = PBRMaterial(
                    name=md.get("name", mid),
                    base_color=tuple(md.get("base_color", [0.8, 0.8, 0.8, 1.0])),
                    metallic=md.get("metallic", 0.0),
                    roughness=md.get("roughness", 0.5),
                    diffuse_map=md.get("diffuse_map", ""),
                    normal_map=md.get("normal_map", ""),
                    roughness_map=md.get("roughness_map", ""),
                    metallic_map=md.get("metallic_map", ""),
                    opacity_map=md.get("opacity_map", ""),
                )
            # Restore morph metadata (deltas loaded from .npz)
            for mi, md in enumerate(fd.get("morphs", [])):
                fig.morphs.append(Morph(
                    name=md.get("name", ""),
                    group=md.get("group", ""),
                    min_value=md.get("min_value", 0.0),
                    max_value=md.get("max_value", 1.0),
                ))
            # Load heavy data from .npz
            npz_path = fig_dir / f"{fig.id}.npz"
            if npz_path.is_file():
                _load_figure_npz(fig, npz_path)

            scene.figures.append(fig)

        # Restore props
        for pd in data.get("props", []):
            prop = Prop(
                id=pd["id"],
                name=pd.get("name", ""),
                source_path=pd.get("source_path", ""),
                visible=pd.get("visible", True),
                parent_figure_id=pd.get("parent_figure_id", ""),
                position=np.array(pd.get("position", [0, 0, 0]), dtype=np.float32),
                rotation=np.array(pd.get("rotation", [0, 0, 0]), dtype=np.float32),
                scale=np.array(pd.get("scale", [1, 1, 1]), dtype=np.float32),
            )
            for mid, md in pd.get("materials", {}).items():
                prop.materials[mid] = PBRMaterial(
                    name=md.get("name", mid),
                    base_color=tuple(md.get("base_color", [0.8, 0.8, 0.8, 1.0])),
                    metallic=md.get("metallic", 0.0),
                    roughness=md.get("roughness", 0.5),
                    diffuse_map=md.get("diffuse_map", ""),
                    normal_map=md.get("normal_map", ""),
                    roughness_map=md.get("roughness_map", ""),
                    metallic_map=md.get("metallic_map", ""),
                    opacity_map=md.get("opacity_map", ""),
                )
            npz_path = fig_dir / f"prop_{prop.id}.npz"
            if npz_path.is_file():
                _load_prop_npz(prop, npz_path)
            scene.props.append(prop)

        return scene


def save_instance(inst: SceneInstance, path: Path) -> None:
    """Serialize a scene instance to JSON."""
    data = {
        "name": inst.name,
        "id": inst.id,
        "created": inst.created,
        "camera": {
            "position": inst.camera.position.tolist(),
            "target": inst.camera.target.tolist(),
            "fov": inst.camera.fov,
        },
        "lights": [
            {
                "kind": lt.kind,
                "color": list(lt.color),
                "intensity": lt.intensity,
                "position": lt.position.tolist(),
                "direction": lt.direction.tolist(),
            }
            for lt in inst.lights
        ],
        "figure_poses": {
            fid: {bone: list(rot) for bone, rot in poses.items()}
            for fid, poses in inst.figure_poses.items()
        },
        "figure_morphs": inst.figure_morphs,
        "render_width": inst.render_width,
        "render_height": inst.render_height,
        "thumbnail": inst.thumbnail,
    }
    path.write_text(json.dumps(data, indent=2))


def _load_figure_npz(fig: Figure, npz_path: Path) -> None:
    """Restore heavy mesh/skeleton data from a cached .npz file."""
    d = np.load(npz_path, allow_pickle=True)
    fig.mesh.vertices = d["vertices"]
    fig.mesh.normals = d["normals"]
    fig.mesh.uvs = d["uvs"]
    fig.mesh.indices = d["indices"]
    fig.mesh.bone_indices = d["bone_indices"]
    fig.mesh.bone_weights = d["bone_weights"]
    # Submeshes
    if "submesh_ids" in d:
        sm_ids = d["submesh_ids"]
        sm_offsets = d["submesh_offsets"]
        sm_counts = d["submesh_counts"]
        fig.mesh.submeshes = [
            SubMesh(material_id=str(sm_ids[i]), index_offset=int(sm_offsets[i]),
                    index_count=int(sm_counts[i]))
            for i in range(len(sm_ids))
        ]
    # Skeleton
    if "bone_names" in d:
        names = d["bone_names"]
        parents = d["bone_parents"]
        positions = d["bone_positions"]
        rotations = d["bone_rotations"]
        scales = d["bone_scales"]
        regions = d["bone_regions"]
        rot_mins = d.get("bone_rot_mins")
        rot_maxs = d.get("bone_rot_maxs")
        bones = []
        for i in range(len(names)):
            rmin = rot_mins[i] if rot_mins is not None else None
            rmax = rot_maxs[i] if rot_maxs is not None else None
            if rmin is not None and rmin[0] == -999:
                rmin = None
            if rmax is not None and rmax[0] == -999:
                rmax = None
            bones.append(Bone(
                name=str(names[i]),
                parent_index=int(parents[i]),
                rest_position=positions[i].astype(np.float32),
                rest_rotation=rotations[i].astype(np.float32),
                rest_scale=scales[i].astype(np.float32),
                rotation_min=rmin.astype(np.float32) if rmin is not None else None,
                rotation_max=rmax.astype(np.float32) if rmax is not None else None,
                region=str(regions[i]),
            ))
        fig.skeleton = Skeleton(bones=bones)
    # Morph deltas
    for mi, morph in enumerate(fig.morphs):
        idx_key = f"morph_{mi}_idx"
        val_key = f"morph_{mi}_val"
        if idx_key in d and val_key in d:
            idxs = d[idx_key]
            vals = d[val_key]
            morph.deltas = {int(idxs[j]): (float(vals[j, 0]), float(vals[j, 1]), float(vals[j, 2]))
                           for j in range(len(idxs))}


def _load_prop_npz(prop: Prop, npz_path: Path) -> None:
    """Restore prop mesh data from a cached .npz file."""
    d = np.load(npz_path, allow_pickle=True)
    prop.mesh.vertices = d["vertices"]
    prop.mesh.normals = d["normals"]
    prop.mesh.uvs = d["uvs"]
    prop.mesh.indices = d["indices"]
    if "bone_indices" in d:
        prop.mesh.bone_indices = d["bone_indices"]
    if "bone_weights" in d:
        prop.mesh.bone_weights = d["bone_weights"]
    if "submesh_ids" in d:
        sm_ids = d["submesh_ids"]
        sm_offsets = d["submesh_offsets"]
        sm_counts = d["submesh_counts"]
        prop.mesh.submeshes = [
            SubMesh(material_id=str(sm_ids[i]), index_offset=int(sm_offsets[i]),
                    index_count=int(sm_counts[i]))
            for i in range(len(sm_ids))
        ]


def load_instance(path: Path) -> SceneInstance:
    """Deserialize a scene instance from JSON."""
    data = json.loads(path.read_text())
    cam_data = data.get("camera", {})
    camera = Camera(
        position=np.array(cam_data.get("position", [0, 1.2, 3]), dtype=np.float32),
        target=np.array(cam_data.get("target", [0, 1, 0]), dtype=np.float32),
        fov=cam_data.get("fov", 45.0),
    )
    lights = []
    for ld in data.get("lights", []):
        lights.append(Light(
            kind=ld.get("kind", "directional"),
            color=tuple(ld.get("color", [1, 1, 1])),
            intensity=ld.get("intensity", 1.0),
            position=np.array(ld.get("position", [2, 4, 3]), dtype=np.float32),
            direction=np.array(ld.get("direction", [-0.3, -0.8, -0.5]), dtype=np.float32),
        ))
    figure_poses = {}
    for fid, poses in data.get("figure_poses", {}).items():
        figure_poses[fid] = {bone: tuple(rot) for bone, rot in poses.items()}

    return SceneInstance(
        name=data.get("name", "default"),
        id=data.get("id", uuid.uuid4().hex[:12]),
        created=data.get("created", time.time()),
        camera=camera,
        lights=lights or [Light()],
        figure_poses=figure_poses,
        figure_morphs=data.get("figure_morphs", {}),
        render_width=data.get("render_width", 1024),
        render_height=data.get("render_height", 1024),
        thumbnail=data.get("thumbnail", ""),
    )
