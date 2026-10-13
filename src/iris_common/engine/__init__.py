"""iris_common.engine — shared computational engines."""
from .geometry import compute_tissue_boundary_mask, recompute_area_from_vertices
from .neighborhood import identify_cellular_neighborhoods

__all__ = [
    "compute_tissue_boundary_mask",
    "recompute_area_from_vertices",
    "identify_cellular_neighborhoods",
]
