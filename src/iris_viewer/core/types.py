"""
IRIS-Suite Shared Types
Dataclasses and typed aliases used by both Viewer and Phenotyper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Marker Panel

@dataclass
class MarkerPanel:
    """
    Describes the channel layout of a multiplexed imaging experiment.

    Attributes
    ----------
    markers : list[str]
        Ordered list of marker names corresponding to image channels (C axis).
    panel_file : Path or None
        Source CSV/Excel file the panel was loaded from (for audit trail).
    nuclear_channel : int
        Index of the nuclear/DAPI channel (default: 0).
    """
    markers: List[str] = field(default_factory=list)
    panel_file: Optional[Path] = None
    nuclear_channel: int = 0

    @property
    def n_channels(self) -> int:
        return len(self.markers)

    def index_of(self, marker: str) -> int:
        """Return the channel index of *marker* (case-insensitive). Raises ValueError if absent."""
        for i, m in enumerate(self.markers):
            if m.strip().lower() == marker.strip().lower():
                return i
        raise ValueError(f"Marker '{marker}' not found in panel: {self.markers}")

    def subset(self, names: List[str]) -> List[int]:
        """Return channel indices for a list of marker names (ignores missing)."""
        return [self.index_of(m) for m in names if any(
            m.strip().lower() == x.strip().lower() for x in self.markers
        )]


# Cell Type Mapping

@dataclass
class CellTypeRule:
    """
    Single rule entry for the ASTIR / CELESTA marker-to-type mapping.

    Attributes
    ----------
    cell_type : str
        Human-readable cell type label (e.g. 'T cell', 'Macrophage').
    markers_positive : list[str]
        Markers expected to be HIGH for this type.
    markers_negative : list[str]
        Markers expected to be LOW for this type (used by CELESTA anchor table).
    parent : str or None
        Parent type in the CELESTA hierarchy (e.g. 'T cell' parent='Lymphocyte').
    """
    cell_type: str
    markers_positive: List[str] = field(default_factory=list)
    markers_negative: List[str] = field(default_factory=list)
    parent: Optional[str] = None


@dataclass
class CellTypePanel:
    """Collection of CellTypeRules defining a full phenotyping schema."""
    rules: List[CellTypeRule] = field(default_factory=list)
    panel_file: Optional[Path] = None

    @property
    def cell_types(self) -> List[str]:
        return [r.cell_type for r in self.rules]

    def to_astir_dict(self, marker_panel: MarkerPanel) -> Dict[str, Any]:
        """Convert to the ASTIR ``marker_dict`` format: ``{'cell_types': {cell_type: [marker, ...]}}``.

        Only includes markers present in the panel, matching the panel's exact casing.
        """
        result: Dict[str, List[str]] = {}
        panel_map = {m.lower(): m for m in marker_panel.markers}
        for rule in self.rules:
            valid = [panel_map[m.lower()] for m in rule.markers_positive if m.lower() in panel_map]
            if valid:
                result[rule.cell_type] = valid
        return {"cell_types": result}

    def to_celesta_anchor_matrix(self, marker_panel: MarkerPanel):
        """
        Convert to the CELESTA anchor marker DataFrame:
        rows = cell types, columns = markers, values = 1 (high) / 0 (low/ignore).
        """
        import pandas as pd
        markers = marker_panel.markers
        rows = []
        for rule in self.rules:
            pos_lower = {m.lower() for m in rule.markers_positive}
            neg_lower = {m.lower() for m in rule.markers_negative}
            row = {
                'cell_type': rule.cell_type,
                'parent': rule.parent or '',
            }
            for m in markers:
                if m.lower() in pos_lower:
                    row[m] = 1
                elif m.lower() in neg_lower:
                    row[m] = 0
                else:
                    row[m] = -1  # Ignored by CELESTA
            rows.append(row)
        df = pd.DataFrame(rows)
        df.set_index('cell_type', inplace=True)
        return df


# Segmentation / Pipeline types (re-exported from here for phenotyper use)

@dataclass
class BoundingBox:
    """Integer bounding box in (row, col) convention."""
    min_row: int
    min_col: int
    max_row: int
    max_col: int

    @property
    def height(self) -> int:
        return self.max_row - self.min_row

    @property
    def width(self) -> int:
        return self.max_col - self.min_col

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.min_row, self.min_col, self.max_row, self.max_col)

    def padded(self, px: int, max_h: int, max_w: int) -> "BoundingBox":
        return BoundingBox(
            max(0, self.min_row - px),
            max(0, self.min_col - px),
            min(max_h, self.max_row + px),
            min(max_w, self.max_col + px),
        )
