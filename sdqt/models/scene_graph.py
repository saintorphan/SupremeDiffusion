"""Scene Graph — shot tree data model for the divide-and-conquer workflow.

Each node is either an **image** (still frame) or a **clip** (short video).
Edges represent generation steps: img2img, i2v, still extraction, inpaint, upscale.

The tree enforces the wide-to-narrow workflow:
  Root (wide establishing shot)
    └─ Orbit clip (2-3 sec)
         ├─ Still @ 0.5s
         │    └─ Zoom-in clip
         │         ├─ Best frame @ 1.2s
         │         │    └─ Scene clip ...
         │         └─ Best frame @ 2.0s
         ├─ Still @ 1.2s
         └─ Still @ 2.8s

Persistence: ``SceneGraph.save(path)`` / ``SceneGraph.load(path)``
writes a single ``scene_graph.json`` per project.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    IMAGE = "image"
    CLIP = "clip"


class EdgeType(str, Enum):
    """How a child was derived from its parent."""
    IMPORT = "import"           # Imported from external source (Daz, file)
    IMG2IMG = "img2img"         # Realism pass / style transfer
    INPAINT = "inpaint"         # Inpainted region
    I2V = "i2v"                 # Image-to-video generation
    STILL_EXTRACT = "still"     # Frame extracted from a clip
    UPSCALE = "upscale"         # Upscaled version
    EXTEND = "extend"           # Video extender continuation
    MANUAL = "manual"           # User manually linked


@dataclass
class GenerationParams:
    """Snapshot of the settings used to produce a node."""
    prompt: str = ""
    negative_prompt: str = ""
    model_type: str = ""
    seed: int = -1
    steps: int = 0
    guidance_scale: float = 1.0
    denoise_strength: float = 1.0
    resolution: str = ""
    duration_frames: int = 0
    fps: int = 16
    loras: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> GenerationParams:
        # Handle unknown keys gracefully
        known = {f.name for f in cls.__dataclass_fields__.values()}
        extra = d.pop("extra", {})
        overflow = {k: v for k, v in d.items() if k not in known}
        extra.update(overflow)
        filtered = {k: v for k, v in d.items() if k in known}
        filtered["extra"] = extra
        return cls(**filtered)


@dataclass
class ShotNode:
    """A single node in the scene graph."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    node_type: NodeType = NodeType.IMAGE
    file_path: str = ""                     # Absolute path to image or clip
    thumbnail_path: str = ""                # Cached thumbnail (generated on demand)
    parent_id: str | None = None
    edge_type: EdgeType = EdgeType.IMPORT   # How this node was derived from parent
    children_ids: list[str] = field(default_factory=list)
    params: GenerationParams = field(default_factory=GenerationParams)
    created_at: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)       # User labels: "hero", "trash", "angle-left"
    starred: bool = False                               # Quick-pick flag
    notes: str = ""                                     # Free-form user notes
    # For still extractions — which timestamp in the parent clip
    extract_timestamp: float = 0.0

    # ── Quality metrics (populated by Smart Still Picker) ──
    sharpness_score: float = 0.0    # Laplacian variance
    face_count: int = 0             # Detected faces in frame
    composition_score: float = 0.0  # Rule-of-thirds / centroid score

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type.value,
            "file_path": self.file_path,
            "thumbnail_path": self.thumbnail_path,
            "parent_id": self.parent_id,
            "edge_type": self.edge_type.value,
            "children_ids": list(self.children_ids),
            "params": self.params.to_dict(),
            "created_at": self.created_at,
            "tags": list(self.tags),
            "starred": self.starred,
            "notes": self.notes,
            "extract_timestamp": self.extract_timestamp,
            "sharpness_score": self.sharpness_score,
            "face_count": self.face_count,
            "composition_score": self.composition_score,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ShotNode:
        return cls(
            id=d["id"],
            name=d.get("name", ""),
            node_type=NodeType(d.get("node_type", "image")),
            file_path=d.get("file_path", ""),
            thumbnail_path=d.get("thumbnail_path", ""),
            parent_id=d.get("parent_id"),
            edge_type=EdgeType(d.get("edge_type", "import")),
            children_ids=d.get("children_ids", []),
            params=GenerationParams.from_dict(d.get("params", {})),
            created_at=d.get("created_at", 0.0),
            tags=d.get("tags", []),
            starred=d.get("starred", False),
            notes=d.get("notes", ""),
            extract_timestamp=d.get("extract_timestamp", 0.0),
            sharpness_score=d.get("sharpness_score", 0.0),
            face_count=d.get("face_count", 0),
            composition_score=d.get("composition_score", 0.0),
        )


class SceneGraph:
    """The full shot tree for a project.

    Supports multiple root nodes (e.g. different establishing shots for
    different locations in the same project).
    """

    def __init__(self) -> None:
        self._nodes: dict[str, ShotNode] = {}
        self._root_ids: list[str] = []

    # ── Node access ──────────────────────────────────────────────────────

    def get(self, node_id: str) -> ShotNode | None:
        return self._nodes.get(node_id)

    def roots(self) -> list[ShotNode]:
        return [self._nodes[rid] for rid in self._root_ids if rid in self._nodes]

    def children(self, node_id: str) -> list[ShotNode]:
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [self._nodes[cid] for cid in node.children_ids if cid in self._nodes]

    def parent(self, node_id: str) -> ShotNode | None:
        node = self._nodes.get(node_id)
        if node and node.parent_id:
            return self._nodes.get(node.parent_id)
        return None

    def ancestors(self, node_id: str) -> list[ShotNode]:
        """Walk up to root. Returns [parent, grandparent, ..., root]."""
        result: list[ShotNode] = []
        current = self._nodes.get(node_id)
        while current and current.parent_id:
            parent = self._nodes.get(current.parent_id)
            if parent:
                result.append(parent)
            current = parent
        return result

    def all_nodes(self) -> list[ShotNode]:
        return list(self._nodes.values())

    def node_count(self) -> int:
        return len(self._nodes)

    # ── Mutation ─────────────────────────────────────────────────────────

    def add_root(self, node: ShotNode) -> ShotNode:
        """Add a root node (establishing shot / imported base)."""
        node.parent_id = None
        self._nodes[node.id] = node
        if node.id not in self._root_ids:
            self._root_ids.append(node.id)
        return node

    def add_child(self, parent_id: str, child: ShotNode) -> ShotNode:
        """Add a child node under an existing parent."""
        parent = self._nodes.get(parent_id)
        if not parent:
            raise KeyError(f"Parent node {parent_id} not found")
        child.parent_id = parent_id
        self._nodes[child.id] = child
        if child.id not in parent.children_ids:
            parent.children_ids.append(child.id)
        return child

    def remove_node(self, node_id: str, recursive: bool = True) -> list[str]:
        """Remove a node (and optionally all descendants).

        Returns list of removed node IDs.
        """
        node = self._nodes.get(node_id)
        if not node:
            return []

        removed: list[str] = []

        if recursive:
            # DFS remove children first
            for cid in list(node.children_ids):
                removed.extend(self.remove_node(cid, recursive=True))

        # Unlink from parent
        if node.parent_id:
            parent = self._nodes.get(node.parent_id)
            if parent and node_id in parent.children_ids:
                parent.children_ids.remove(node_id)

        # Remove from roots
        if node_id in self._root_ids:
            self._root_ids.remove(node_id)

        del self._nodes[node_id]
        removed.append(node_id)
        return removed

    def move_node(self, node_id: str, new_parent_id: str) -> None:
        """Re-parent a node under a different parent."""
        node = self._nodes.get(node_id)
        new_parent = self._nodes.get(new_parent_id)
        if not node or not new_parent:
            raise KeyError("Node or new parent not found")

        # Prevent cycles
        if new_parent_id == node_id:
            raise ValueError("Cannot parent a node under itself")
        ancestor_ids = {a.id for a in self.ancestors(new_parent_id)}
        if node_id in ancestor_ids:
            raise ValueError("Cannot create a cycle in the scene graph")

        # Unlink from old parent
        if node.parent_id:
            old_parent = self._nodes.get(node.parent_id)
            if old_parent and node_id in old_parent.children_ids:
                old_parent.children_ids.remove(node_id)
        if node_id in self._root_ids:
            self._root_ids.remove(node_id)

        # Link to new parent
        node.parent_id = new_parent_id
        if node_id not in new_parent.children_ids:
            new_parent.children_ids.append(node_id)

    # ── Query helpers ────────────────────────────────────────────────────

    def starred_nodes(self) -> list[ShotNode]:
        return [n for n in self._nodes.values() if n.starred]

    def images(self) -> list[ShotNode]:
        return [n for n in self._nodes.values() if n.node_type == NodeType.IMAGE]

    def clips(self) -> list[ShotNode]:
        return [n for n in self._nodes.values() if n.node_type == NodeType.CLIP]

    def find_by_tag(self, tag: str) -> list[ShotNode]:
        return [n for n in self._nodes.values() if tag in n.tags]

    def depth(self, node_id: str) -> int:
        """How many generations deep a node is from its root."""
        return len(self.ancestors(node_id))

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Write scene graph to JSON file."""
        path = Path(path)
        data = {
            "version": 1,
            "root_ids": list(self._root_ids),
            "nodes": {nid: node.to_dict() for nid, node in self._nodes.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(path)
        logger.info("Scene graph saved: %d nodes → %s", len(self._nodes), path)

    @classmethod
    def load(cls, path: str | Path) -> SceneGraph:
        """Load scene graph from JSON file."""
        path = Path(path)
        if not path.is_file():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        graph = cls()
        graph._root_ids = data.get("root_ids", [])
        for nid, nd in data.get("nodes", {}).items():
            graph._nodes[nid] = ShotNode.from_dict(nd)
        logger.info("Scene graph loaded: %d nodes from %s", len(graph._nodes), path)
        return graph

    # ── Convenience constructors ─────────────────────────────────────────

    @staticmethod
    def create_image_node(
        file_path: str,
        name: str = "",
        edge_type: EdgeType = EdgeType.IMPORT,
        params: GenerationParams | None = None,
        **kwargs,
    ) -> ShotNode:
        """Create an image node with sensible defaults."""
        return ShotNode(
            name=name or Path(file_path).stem,
            node_type=NodeType.IMAGE,
            file_path=file_path,
            edge_type=edge_type,
            params=params or GenerationParams(),
            **kwargs,
        )

    @staticmethod
    def create_clip_node(
        file_path: str,
        name: str = "",
        edge_type: EdgeType = EdgeType.I2V,
        params: GenerationParams | None = None,
        **kwargs,
    ) -> ShotNode:
        """Create a clip node with sensible defaults."""
        return ShotNode(
            name=name or Path(file_path).stem,
            node_type=NodeType.CLIP,
            file_path=file_path,
            edge_type=edge_type,
            params=params or GenerationParams(),
            **kwargs,
        )
