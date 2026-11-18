"""Unit tests for the IRIS Nimbus phenotyping pipeline and adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import tifffile

from iris_common.types import CellTypePanel, CellTypeRule
from iris_phenotyper.base import PhenotypingPredictionResult
from iris_phenotyper.nimbus import (
    NimbusPhenotyper,
    aggregate_cell_positivity,
    assign_cell_types_from_panel,
    cluster_positivity_profiles,
    compute_fov_channel_quantiles,
    is_upstream_available,
    normalize_channel_image,
    prepare_binary_mask,
    prepare_input_data,
    prepare_normalization_dict,
)


def _make_synthetic_fov(
    height: int = 96,
    width: int = 96,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Create a 3-channel synthetic FOV with non-overlapping + touching cells."""
    mask = np.zeros((height, width), dtype=np.uint32)
    mask[10:30, 10:25] = 1
    mask[10:30, 25:40] = 2
    mask[50:70, 15:35] = 3
    mask[55:75, 55:75] = 4

    channels = ["CD3", "CD20", "Cytokeratin"]
    img = np.zeros((3, height, width), dtype=np.float32) + 2.0
    img[0, mask == 1] = 120.0
    img[0, mask == 3] = 110.0
    img[1, mask == 2] = 130.0
    img[2, mask == 4] = 140.0
    return img, mask, channels


def _save_mock_checkpoint(path: Path) -> Path:
    """Create a deterministic NimbusUNet checkpoint."""
    import torch

    from iris_phenotyper.nimbus import NimbusUNet

    torch.manual_seed(7)
    model = NimbusUNet(num_classes=1, padding="reflect")
    torch.save(model.state_dict(), path)
    return path


def test_prepare_binary_mask_separates_touching_cells() -> None:
    _, mask, _ = _make_synthetic_fov()
    binary = prepare_binary_mask(mask)

    assert binary.shape == mask.shape
    assert binary.dtype == np.float32
    assert set(np.unique(binary).tolist()).issubset({0.0, 1.0})
    assert np.all(binary[mask == 0] == 0.0)
    # Touching border pixels between cell 1 (col 24) and cell 2 (col 25) must be 0
    assert np.all(binary[12:28, 24] == 0.0)
    assert np.all(binary[12:28, 25] == 0.0)
    # Cell interiors must remain 1
    assert np.all(binary[14:26, 14:22] == 1.0)
    assert np.all(binary[14:26, 28:36] == 1.0)


def test_prepare_input_data_and_quantile_normalization() -> None:
    img, mask, channels = _make_synthetic_fov()
    q_val = compute_fov_channel_quantiles(img[0], quantile=0.999)
    assert q_val is not None and q_val > 100.0

    zero_img = np.zeros_like(img)
    norm_dict = prepare_normalization_dict(
        [img, zero_img], channel_names=channels, quantile=0.999, n_subset=2
    )
    assert set(norm_dict.keys()) == set(channels)
    for ch in channels:
        assert norm_dict[ch] > 0.0

    norm_cd3 = normalize_channel_image(img[0], norm_factor=norm_dict["CD3"])
    assert norm_cd3.dtype == np.float32
    assert float(norm_cd3.min()) >= 0.0
    assert float(norm_cd3.max()) <= 1.0

    tensor_in = prepare_input_data(norm_cd3, instance_mask=mask)
    assert tensor_in.shape == (1, 2, *mask.shape)
    assert tensor_in.dtype == np.float32


def test_aggregate_cell_positivity_schema() -> None:
    _, mask, _ = _make_synthetic_fov()
    prob_map = np.zeros(mask.shape, dtype=np.float32)
    prob_map[mask == 1] = 0.90
    prob_map[mask == 2] = 0.25
    prob_map[mask == 3] = 0.75
    prob_map[mask == 4] = 0.10

    df = aggregate_cell_positivity(mask, prob_map)
    assert list(df.columns) == ["ObjectNumber", "Pos_X", "Pos_Y", "area", "intensity_mean"]
    assert df["ObjectNumber"].tolist() == [1, 2, 3, 4]
    assert np.allclose(df["intensity_mean"].to_numpy(), [0.90, 0.25, 0.75, 0.10], atol=1e-5)

    empty_df = aggregate_cell_positivity(np.zeros_like(mask), prob_map)
    assert empty_df.empty
    assert list(empty_df.columns) == ["ObjectNumber", "Pos_X", "Pos_Y", "area", "intensity_mean"]


def test_adapter_contract_and_positivity_df_assignment() -> None:
    img, _, channels = _make_synthetic_fov()
    pheno = NimbusPhenotyper(positive_threshold=0.5, n_clusters=2)
    assert pheno.name == "Nimbus"
    assert pheno.input_modality == "image_and_mask"
    assert pheno.requires_training is False
    assert isinstance(pheno.is_available(), bool)

    defaults = pheno.get_default_params()
    assert defaults["quantile"] == 0.999
    assert defaults["model_magnification"] == 10.0

    panel = CellTypePanel(
        rules=[
            CellTypeRule(cell_type="T_cell", markers_positive=["CD3"], markers_negative=["CD20"]),
            CellTypeRule(cell_type="B_cell", markers_positive=["CD20"], markers_negative=["CD3"]),
            CellTypeRule(cell_type="Epithelial", markers_positive=["Cytokeratin"]),
        ]
    )
    pheno.fit(img, channel_names=channels, cell_type_panel=panel)
    assert pheno.normalization_dict_ is not None

    pos_df = pd.DataFrame(
        {
            "CD3": [0.92, 0.08, 0.85, 0.05],
            "CD20": [0.05, 0.91, 0.10, 0.04],
            "Cytokeratin": [0.03, 0.05, 0.06, 0.94],
        },
        index=pd.Index([1, 2, 3, 4], name="ObjectNumber"),
    )
    res_panel = pheno.predict(positivity_df=pos_df)
    assert isinstance(res_panel, PhenotypingPredictionResult)
    res_panel.validate()
    assert res_panel.labels.tolist() == ["T_cell", "B_cell", "T_cell", "Epithelial"]
    assert res_panel.marker_positivity is not None
    assert res_panel.marker_positivity.equals(pos_df)

    res_cluster = pheno.predict(positivity_df=pos_df, cell_type_panel=None, n_clusters=3)
    res_cluster.validate()
    assert res_cluster.labels.nunique() == 3
    assert res_cluster.probabilities is not None
    assert np.allclose(res_cluster.probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_downstream_clustering_edge_cases() -> None:
    pos_df = pd.DataFrame(
        {"CD3": [0.9, 0.1, 0.85], "CD20": [0.05, 0.92, 0.08]},
        index=pd.Index([1, 2, 3], name="ObjectNumber"),
    )
    labels, probs, conf = cluster_positivity_profiles(pos_df, n_clusters=2)
    assert len(labels) == 3
    assert probs.shape == (3, 2)
    assert np.all((conf >= 0.0) & (conf <= 1.0))

    single_labels, single_probs, single_conf = cluster_positivity_profiles(pos_df.iloc[:1])
    assert len(single_labels) == 1
    assert float(single_conf.iloc[0]) == 1.0
    assert single_probs.shape == (1, 1)

    p_labels, p_probs, p_conf = assign_cell_types_from_panel(
        pos_df, {"T_cell": ["CD3"], "B_cell": ["CD20"]}, positive_threshold=0.5
    )
    assert p_labels.tolist() == ["T_cell", "B_cell", "T_cell"]
    assert p_probs.shape == (3, 2)
    assert np.all((p_conf >= 0.0) & (p_conf <= 1.0))


def test_nimbus_unet_single_shot_and_tiling_forward(tmp_path: Path) -> None:
    ckpt_path = _save_mock_checkpoint(tmp_path / "mock_nimbus.pt")
    img, mask, channels = _make_synthetic_fov(height=288, width=288)

    pheno = NimbusPhenotyper(
        checkpoint=str(ckpt_path),
        input_shape=(256, 256),  # Forces valid-padded tile-and-stitch on 288x288 at 10x mag
        dataset_magnification=10.0,
        model_magnification=10.0,
        test_time_aug=False,
        device="cpu",
        batch_size=2,
        n_clusters=3,
    )
    assert pheno.is_available()

    res = pheno.predict(img, mask, channel_names=channels)
    assert isinstance(res, PhenotypingPredictionResult)
    res.validate()
    assert res.method_name == "Nimbus"
    assert res.marker_positivity is not None
    assert list(res.marker_positivity.columns) == channels
    assert res.marker_positivity.index.tolist() == [1, 2, 3, 4]
    vals = res.marker_positivity.to_numpy()
    assert np.all((vals >= 0.0) & (vals <= 1.0))


def test_nimbus_phenotyper_predict_contract_with_panel_and_clustering() -> None:
    import torch

    class _IntensityTrackingModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            marker = x[:, 0:1]
            interior = x[:, 1:2]
            return torch.clamp(marker * (0.2 + 0.8 * interior), 0.0, 1.0)

    img, mask, channels = _make_synthetic_fov(height=64, width=64)
    surrogate = _IntensityTrackingModel()

    panel = CellTypePanel(
        rules=[
            CellTypeRule(cell_type="T_cell", markers_positive=["CD3"], markers_negative=["CD20"]),
            CellTypeRule(cell_type="B_cell", markers_positive=["CD20"], markers_negative=["CD3"]),
            CellTypeRule(cell_type="Epithelial", markers_positive=["Cytokeratin"]),
        ]
    )

    pheno = NimbusPhenotyper(
        dataset_magnification=10.0,
        model_magnification=10.0,
        test_time_aug=True,
        device="cpu",
        positive_threshold=0.5,
        return_pixel_maps=True,
    )
    pheno.fit(img, channel_names=channels, cell_type_panel=panel)

    res_panel = pheno.predict(img, mask, model=surrogate)
    res_panel.validate()
    assert res_panel.marker_positivity is not None
    assert res_panel.labels.loc[1] == "T_cell"
    assert res_panel.labels.loc[2] == "B_cell"
    assert res_panel.labels.loc[3] == "T_cell"
    assert res_panel.labels.loc[4] == "Epithelial"
    assert "probability_maps" in res_panel.metadata
    assert res_panel.metadata["assignment_mode"] == "panel"

    pheno_cluster = NimbusPhenotyper(
        dataset_magnification=10.0,
        model_magnification=10.0,
        test_time_aug=False,
        device="cpu",
        clustering_method="consensus",
        n_clusters=3,
    )
    res_cluster = pheno_cluster.predict(img, mask, channel_names=channels, model=surrogate)
    res_cluster.validate()
    assert res_cluster.labels.nunique() == 3
    assert res_cluster.probabilities is not None
    assert np.allclose(res_cluster.probabilities.sum(axis=1), 1.0, atol=1e-5)

    res_thresh = pheno_cluster.predict(
        img, mask, channel_names=channels, model=surrogate, clustering_method="threshold"
    )
    res_thresh.validate()
    assert res_thresh.labels.loc[1] == "CD3+"
    assert res_thresh.labels.loc[2] == "CD20+"
    assert res_thresh.labels.loc[4] == "Cytokeratin+"


def test_nimbus_directory_inference_native_and_empty_mask(tmp_path: Path) -> None:
    ckpt_path = _save_mock_checkpoint(tmp_path / "mock_nimbus.pt")
    img, mask, channels = _make_synthetic_fov(height=64, width=64)

    fov_dir = tmp_path / "image_data" / "fov_0"
    mask_dir = tmp_path / "masks"
    fov_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    for c_idx, ch in enumerate(channels):
        tifffile.imwrite(fov_dir / f"{ch}.tiff", img[c_idx])
    tifffile.imwrite(mask_dir / "fov_0_whole_cell.tiff", mask)

    pheno = NimbusPhenotyper(
        checkpoint=str(ckpt_path),
        dataset_magnification=10.0,
        model_magnification=10.0,
        test_time_aug=False,
        device="cpu",
        use_upstream=False,
        n_clusters=2,
    )
    res = pheno.predict(
        fov_paths=[fov_dir],
        mask_dir=mask_dir,
        channel_names=channels,
        suffix=".tiff",
    )
    res.validate()
    assert res.marker_positivity is not None
    assert len(res.marker_positivity) == 4

    empty_res = pheno.predict(img, np.zeros_like(mask), channel_names=channels)
    empty_res.validate()
    assert len(empty_res.labels) == 0
    assert empty_res.marker_positivity is not None and empty_res.marker_positivity.empty


@pytest.mark.skipif(
    not is_upstream_available(),
    reason="nimbus_inference not installed in env",
)
def test_upstream_nimbus_directory_integration(tmp_path: Path) -> None:
    ckpt_path = _save_mock_checkpoint(tmp_path / "mock_nimbus_upstream.pt")
    img, mask, channels = _make_synthetic_fov(height=64, width=64)

    fov_dir = tmp_path / "image_data" / "fov_0"
    mask_dir = tmp_path / "masks"
    fov_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    for c_idx, ch in enumerate(channels):
        tifffile.imwrite(fov_dir / f"{ch}.tiff", img[c_idx])
    tifffile.imwrite(mask_dir / "fov_0_whole_cell.tiff", mask)

    pheno = NimbusPhenotyper(
        checkpoint=str(ckpt_path),
        dataset_magnification=10.0,
        model_magnification=10.0,
        test_time_aug=False,
        device="cpu",
        use_upstream=True,
        n_clusters=2,
    )
    res = pheno.predict(
        fov_paths=[fov_dir],
        mask_dir=mask_dir,
        channel_names=channels,
        suffix=".tiff",
    )
    res.validate()
    assert res.metadata.get("backend") == "nimbus_inference"
    assert res.marker_positivity is not None
    assert len(res.marker_positivity) == 4

    # Verify numerical parity between native PyTorch pipeline and upstream nimbus_inference
    pheno_native = NimbusPhenotyper(
        checkpoint=str(ckpt_path),
        dataset_magnification=10.0,
        model_magnification=10.0,
        test_time_aug=False,
        device="cpu",
        use_upstream=False,
        n_clusters=2,
    )
    res_native = pheno_native.predict(
        fov_paths=[fov_dir],
        mask_dir=mask_dir,
        channel_names=channels,
        suffix=".tiff",
    )
    res_native.validate()
    assert res_native.marker_positivity is not None
    max_diff = float(
        np.abs(res_native.marker_positivity.values - res.marker_positivity.values).max()
    )
    assert max_diff < 1e-5
