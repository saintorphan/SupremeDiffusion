"""Scene Tree Widget — visual tree view for the shot graph.

Shows thumbnails, node names, edge types, and quality badges.
Right-click context menu for generation actions.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot, QSize
from PySide6.QtGui import QIcon, QPixmap, QColor, QPainter, QFont, QAction
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from sdqt.models.scene_graph import (
    EdgeType,
    NodeType,
    SceneGraph,
    ShotNode,
)

logger = logging.getLogger(__name__)

# ── Edge type display labels ──
_EDGE_LABELS = {
    EdgeType.IMPORT: "Imported",
    EdgeType.IMG2IMG: "img2img",
    EdgeType.INPAINT: "Inpaint",
    EdgeType.I2V: "Video",
    EdgeType.STILL_EXTRACT: "Still",
    EdgeType.UPSCALE: "Upscale",
    EdgeType.EXTEND: "Extended",
    EdgeType.MANUAL: "Linked",
}

_EDGE_COLORS = {
    EdgeType.IMPORT: "#9e9e9e",
    EdgeType.IMG2IMG: "#4fc3f7",
    EdgeType.INPAINT: "#ffb74d",
    EdgeType.I2V: "#81c784",
    EdgeType.STILL_EXTRACT: "#ce93d8",
    EdgeType.UPSCALE: "#a1887f",
    EdgeType.EXTEND: "#4db6ac",
    EdgeType.MANUAL: "#e0e0e0",
}


def _make_thumbnail(file_path: str, size: int = 64) -> QPixmap:
    """Load and scale a thumbnail from an image or video frame."""
    pixmap = QPixmap()
    path = Path(file_path)
    if not path.exists():
        # Placeholder
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(40, 40, 40))
        return pixmap

    if path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        # Try to extract first frame via ffmpeg
        try:
            import subprocess
            import tempfile
            tmp = Path(tempfile.mktemp(suffix=".jpg"))
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(path), "-vframes", "1",
                 "-q:v", "2", str(tmp)],
                capture_output=True, timeout=5,
            )
            if tmp.exists():
                pixmap.load(str(tmp))
                tmp.unlink()
        except Exception:
            pass

    if pixmap.isNull():
        pixmap.load(str(path))

    if pixmap.isNull():
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(40, 40, 40))

    return pixmap.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)


class SceneTreeWidget(QWidget):
    """Visual tree view of the scene graph.

    Signals:
        node_selected(str) — emitted when user clicks a node (node ID)
        node_action(str, str) — emitted for context menu actions (node_id, action_name)
    """

    node_selected = Signal(str)
    node_action = Signal(str, str)  # (node_id, action)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._graph: SceneGraph | None = None
        self._item_map: dict[str, QTreeWidgetItem] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)
        lbl = QLabel("Shot Tree")
        lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
        toolbar.addWidget(lbl)
        toolbar.addStretch()

        self._btn_expand = QPushButton("Expand All")
        self._btn_expand.setFixedHeight(22)
        self._btn_expand.clicked.connect(lambda: self._tree.expandAll())
        toolbar.addWidget(self._btn_expand)

        self._btn_collapse = QPushButton("Collapse")
        self._btn_collapse.setFixedHeight(22)
        self._btn_collapse.clicked.connect(lambda: self._tree.collapseAll())
        toolbar.addWidget(self._btn_collapse)

        self._btn_starred = QPushButton("Starred Only")
        self._btn_starred.setFixedHeight(22)
        self._btn_starred.setCheckable(True)
        self._btn_starred.toggled.connect(self._on_starred_filter)
        toolbar.addWidget(self._btn_starred)

        layout.addLayout(toolbar)

        # Tree
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Shot", "Type", "Method", "Quality"])
        self._tree.setColumnWidth(0, 250)
        self._tree.setColumnWidth(1, 50)
        self._tree.setColumnWidth(2, 70)
        self._tree.setColumnWidth(3, 60)
        self._tree.setIconSize(QSize(48, 48))
        self._tree.setAlternatingRowColors(True)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        self._tree.currentItemChanged.connect(self._on_selection_changed)
        layout.addWidget(self._tree)

        # Info panel at bottom
        self._info = QLabel("")
        self._info.setWordWrap(True)
        self._info.setMaximumHeight(60)
        self._info.setStyleSheet("color: #aaa; font-size: 11px; padding: 4px;")
        layout.addWidget(self._info)

    # ── Public API ───────────────────────────────────────────────────────

    def set_graph(self, graph: SceneGraph) -> None:
        """Load a scene graph and rebuild the tree."""
        self._graph = graph
        self._rebuild_tree()

    def refresh(self) -> None:
        """Rebuild the tree from the current graph."""
        if self._graph:
            self._rebuild_tree()

    def select_node(self, node_id: str) -> None:
        """Programmatically select a node."""
        item = self._item_map.get(node_id)
        if item:
            self._tree.setCurrentItem(item)

    # ── Tree building ────────────────────────────────────────────────────

    def _rebuild_tree(self) -> None:
        self._tree.clear()
        self._item_map.clear()
        if not self._graph:
            return

        for root in self._graph.roots():
            root_item = self._make_item(root)
            self._tree.addTopLevelItem(root_item)
            self._build_subtree(root, root_item)

        self._tree.expandAll()

    def _build_subtree(self, node: ShotNode, tree_item: QTreeWidgetItem) -> None:
        if not self._graph:
            return
        for child in self._graph.children(node.id):
            child_item = self._make_item(child)
            tree_item.addChild(child_item)
            self._build_subtree(child, child_item)

    def _make_item(self, node: ShotNode) -> QTreeWidgetItem:
        item = QTreeWidgetItem()

        # Column 0: Name + thumbnail
        display_name = node.name or Path(node.file_path).stem if node.file_path else node.id[:8]
        if node.starred:
            display_name = f"* {display_name}"
        item.setText(0, display_name)
        item.setToolTip(0, f"ID: {node.id}\nPath: {node.file_path}\nNotes: {node.notes}")

        # Thumbnail
        if node.file_path:
            thumb = _make_thumbnail(node.file_path, 48)
            item.setIcon(0, QIcon(thumb))

        # Column 1: Type icon
        type_str = "IMG" if node.node_type == NodeType.IMAGE else "VID"
        item.setText(1, type_str)

        # Column 2: Edge type (how it was made)
        edge_label = _EDGE_LABELS.get(node.edge_type, "?")
        item.setText(2, edge_label)
        color = _EDGE_COLORS.get(node.edge_type, "#ffffff")
        item.setForeground(2, QColor(color))

        # Column 3: Quality badge
        if node.sharpness_score > 0:
            item.setText(3, f"{node.sharpness_score:.0f}")
        elif node.starred:
            item.setText(3, "PICK")
            item.setForeground(3, QColor("#ffd54f"))

        # Store node ID
        item.setData(0, Qt.ItemDataRole.UserRole, node.id)
        self._item_map[node.id] = item

        return item

    # ── Selection ────────────────────────────────────────────────────────

    @Slot()
    def _on_selection_changed(self, current, _previous) -> None:
        if not current or not self._graph:
            return
        node_id = current.data(0, Qt.ItemDataRole.UserRole)
        if not node_id:
            return

        node = self._graph.get(node_id)
        if not node:
            return

        # Update info panel
        parts = [f"{node.name or node.id[:8]}"]
        if node.file_path:
            parts.append(Path(node.file_path).name)
        parts.append(f"Type: {node.node_type.value} | Made by: {node.edge_type.value}")
        if node.params.prompt:
            prompt_preview = node.params.prompt[:120]
            if len(node.params.prompt) > 120:
                prompt_preview += "..."
            parts.append(f"Prompt: {prompt_preview}")
        depth = self._graph.depth(node_id)
        children_count = len(node.children_ids)
        parts.append(f"Depth: {depth} | Children: {children_count}")
        self._info.setText("\n".join(parts))

        self.node_selected.emit(node_id)

    # ── Context menu ─────────────────────────────────────────────────────

    @Slot()
    def _on_context_menu(self, pos) -> None:
        item = self._tree.itemAt(pos)
        if not item or not self._graph:
            return

        node_id = item.data(0, Qt.ItemDataRole.UserRole)
        node = self._graph.get(node_id)
        if not node:
            return

        menu = QMenu(self)

        # Actions depend on node type
        if node.node_type == NodeType.IMAGE:
            menu.addAction("Plan Shot (Start/End Frames)").setData("plan_shot")
            menu.addAction("Generate Orbit Clip (2-3s)").setData("orbit_clip")
            menu.addAction("Generate Zoom-In Clip").setData("zoom_clip")
            menu.addAction("Run Realism Pass (img2img)").setData("realism_pass")
            menu.addAction("Inpaint Region").setData("inpaint")
            menu.addAction("Upscale").setData("upscale")
            menu.addSeparator()
            menu.addAction("Use as Source Image").setData("use_as_source")

        elif node.node_type == NodeType.CLIP:
            menu.addAction("Extract Stills (Smart Pick)").setData("extract_stills")
            menu.addAction("Extract All Frames").setData("extract_all")
            menu.addAction("Extend Clip").setData("extend")
            menu.addSeparator()
            menu.addAction("Open in StillGrabber").setData("open_stillgrabber")

        menu.addSeparator()

        # Universal actions
        star_text = "Unstar" if node.starred else "Star as Hero Shot"
        menu.addAction(star_text).setData("toggle_star")
        menu.addAction("Rename").setData("rename")
        menu.addAction("Add Note").setData("add_note")
        menu.addSeparator()

        # Navigation
        if node.parent_id:
            menu.addAction("Go to Parent").setData("go_parent")
        ancestors = self._graph.ancestors(node_id)
        if ancestors:
            menu.addAction(f"Go to Root ({ancestors[-1].name or 'root'})").setData("go_root")

        menu.addSeparator()
        remove_action = menu.addAction("Remove Node")
        remove_action.setData("remove")
        if node.children_ids:
            remove_tree_action = menu.addAction(
                f"Remove Node + {len(node.children_ids)} Children"
            )
            remove_tree_action.setData("remove_tree")

        # Execute
        action = menu.exec(self._tree.viewport().mapToGlobal(pos))
        if action and action.data():
            action_name = action.data()

            # Handle navigation locally
            if action_name == "go_parent" and node.parent_id:
                self.select_node(node.parent_id)
                return
            elif action_name == "go_root" and ancestors:
                self.select_node(ancestors[-1].id)
                return
            elif action_name == "toggle_star":
                node.starred = not node.starred
                self.refresh()
                return

            # Emit for the wizard/tab to handle
            self.node_action.emit(node_id, action_name)

    # ── Filters ──────────────────────────────────────────────────────────

    @Slot(bool)
    def _on_starred_filter(self, starred_only: bool) -> None:
        """Show/hide non-starred nodes."""
        for node_id, item in self._item_map.items():
            if starred_only:
                node = self._graph.get(node_id) if self._graph else None
                if node and not node.starred:
                    # Check if any descendant is starred
                    has_starred_child = False
                    if self._graph:
                        stack = list(node.children_ids)
                        while stack:
                            cid = stack.pop()
                            child = self._graph.get(cid)
                            if child:
                                if child.starred:
                                    has_starred_child = True
                                    break
                                stack.extend(child.children_ids)
                    item.setHidden(not has_starred_child)
                else:
                    item.setHidden(False)
            else:
                item.setHidden(False)
