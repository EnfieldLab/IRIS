"""
Gating Hierarchy Tree Model
===========================
Tree data structure representing hierarchical cell populations in sequential
flow/IF manual gating workflows. Tracks population parentage, boolean cell masks,
display colors, gating criteria descriptors, and cellular yields.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class GatingHierarchyNode:
    """Represents a population node in a sequential hierarchical gating tree."""

    def __init__(
        self,
        name: str,
        parent: Optional['GatingHierarchyNode'] = None,
        cell_mask: Optional[np.ndarray] = None,
        color: str = "#00E5FF",
        markers: Tuple[str, ...] = (),
        criteria_desc: str = "",
        total_cells_count: int = 0,
    ):
        self.name = name
        self.parent = parent
        self.cell_mask = cell_mask
        self.color = color
        self.markers = markers
        self.criteria_desc = criteria_desc
        self.total_cells_count = total_cells_count
        self.children: List['GatingHierarchyNode'] = []
        if parent is not None:
            parent.children.append(self)

    @property
    def cell_count(self) -> int:
        """Returns the number of cells belonging to this population node."""
        if self.cell_mask is not None:
            return int(self.cell_mask.sum())
        return self.total_cells_count

    @property
    def parent_percentage(self) -> float:
        """Percentage of the immediate parent population."""
        if self.parent and self.parent.cell_count > 0:
            return (self.cell_count / self.parent.cell_count) * 100.0
        return 100.0

    @property
    def total_percentage(self) -> float:
        """Percentage of the total root population."""
        root = self.get_root()
        if root and root.cell_count > 0:
            return (self.cell_count / root.cell_count) * 100.0
        return 100.0

    def get_root(self) -> 'GatingHierarchyNode':
        """Traverses up the tree to return the top-level root node."""
        curr = self
        while curr.parent is not None:
            curr = curr.parent
        return curr

    def find_node(self, name: str) -> Optional['GatingHierarchyNode']:
        """Searches recursively for a node with the specified name."""
        if self.name == name:
            return self
        for child in self.children:
            found = child.find_node(name)
            if found:
                return found
        return None

    def get_all_nodes(self) -> List['GatingHierarchyNode']:
        """Returns a flat list of this node and all descendants."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.get_all_nodes())
        return nodes

    def get_path_string(self) -> str:
        """Returns the breadcrumb path string (e.g. 'Leukocytes / T cells / CD8+')."""
        parts = []
        curr: Optional['GatingHierarchyNode'] = self
        while curr is not None and curr.parent is not None:
            parts.append(curr.name)
            curr = curr.parent
        parts.reverse()
        return " / ".join(parts) if parts else self.name

    def remove_child(self, name: str) -> bool:
        """Removes a child node by name if present."""
        for i, child in enumerate(self.children):
            if child.name == name:
                child.parent = None
                self.children.pop(i)
                return True
        return False
