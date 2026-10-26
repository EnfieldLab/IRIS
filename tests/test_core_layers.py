"""
Unit tests for iris_viewer.core.layers module:
- get_layer_metadata: Safe metadata attribute lookup
- get_layer_base_name: Metadata-first base name resolution with string split fallback
- get_layer_sample_name: Metadata-first sample name resolution with bracket fallback
- get_layer_type: Semantic layer role classification with prefix fallback
"""
from __future__ import annotations

import types

from iris_viewer.core.layers import (
    get_layer_base_name,
    get_layer_metadata,
    get_layer_sample_name,
    get_layer_type,
)


def test_get_layer_metadata_lookup_and_default_fallback():
    """Verify get_layer_metadata extracts values when present and returns default otherwise."""
    layer_with_meta = types.SimpleNamespace(
        name="PanCK [Sample_1]",
        metadata={"sample_name": "Sample_1", "marker": "PanCK"},
    )
    assert get_layer_metadata(layer_with_meta, "marker") == "PanCK"
    assert get_layer_metadata(layer_with_meta, "sample_name") == "Sample_1"
    assert get_layer_metadata(layer_with_meta, "nonexistent", default="fallback") == "fallback"

    layer_no_meta = types.SimpleNamespace(name="CD3", metadata=None)
    assert get_layer_metadata(layer_no_meta, "marker", default="none") == "none"


def test_get_layer_base_name_prefers_metadata_over_string_split():
    """Verify get_layer_base_name uses structured metadata first before falling back to string parsing."""
    # Marker metadata
    channel_layer = types.SimpleNamespace(
        name="PanCK [Sample_A]",
        metadata={"marker": "PanCK", "sample_name": "Sample_A"},
    )
    assert get_layer_base_name(channel_layer) == "PanCK"

    # Cell-type metadata
    pheno_layer = types.SimpleNamespace(
        name="Tumor [Sample_B]",
        metadata={"cell_type": "Tumor", "sample_name": "Sample_B"},
    )
    assert get_layer_base_name(pheno_layer) == "Tumor"

    # Legacy string splitting fallback
    legacy_layer = types.SimpleNamespace(name="CD8 [Sample_C]", metadata={})
    assert get_layer_base_name(legacy_layer) == "CD8"

    # Plain name without bracket
    plain_layer = types.SimpleNamespace(name="DAPI", metadata=None)
    assert get_layer_base_name(plain_layer) == "DAPI"


def test_get_layer_sample_name_prefers_metadata_over_brackets():
    """Verify get_layer_sample_name uses metadata before falling back to bracket parsing."""
    # Metadata presence
    layer = types.SimpleNamespace(
        name="CD4 [TMA_Core_1]",
        metadata={"sample_name": "TMA_Core_1"},
    )
    assert get_layer_sample_name(layer) == "TMA_Core_1"

    # String fallback
    legacy = types.SimpleNamespace(name="CD4 [TMA_Core_2]", metadata={})
    assert get_layer_sample_name(legacy) == "TMA_Core_2"

    # No sample in name
    no_sample = types.SimpleNamespace(name="Global Mask", metadata=None)
    assert get_layer_sample_name(no_sample) is None


def test_get_layer_type_identifies_semantic_roles_with_prefix_fallback():
    """Verify get_layer_type detects layer roles from metadata or name patterns."""
    # Explicit metadata
    channel_lyr = types.SimpleNamespace(name="CD3", metadata={"layer_type": "channel"})
    assert get_layer_type(channel_lyr) == "channel"

    pheno_lyr = types.SimpleNamespace(name="Stroma", metadata={"layer_type": "phenotype"})
    assert get_layer_type(pheno_lyr) == "phenotype"

    # Prefix fallbacks
    bounds_lyr = types.SimpleNamespace(name="Bounds - Core_1", metadata={})
    assert get_layer_type(bounds_lyr) == "bounds"

    centroids_lyr = types.SimpleNamespace(name="Centroids - Core_1", metadata={})
    assert get_layer_type(centroids_lyr) == "centroids"

    niches_lyr = types.SimpleNamespace(name="Niches - Core_1", metadata={})
    assert get_layer_type(niches_lyr) == "niches"

    mask_lyr = types.SimpleNamespace(name="Nuclear Mask", metadata={})
    assert get_layer_type(mask_lyr) == "segmentation"
