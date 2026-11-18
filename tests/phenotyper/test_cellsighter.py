"""Unit tests for iris_phenotyper.cellsighter."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from iris_phenotyper.base import PhenotyperMethodBase, PhenotypingPredictionResult
from iris_phenotyper.cellsighter import (
    CellCropsDataset,
    CellSighterNet,
    CellSighterPhenotyper,
    ChannelNormalizer,
    compute_sampling_weights,
    ensemble_predict_probabilities,
    extract_crops_from_fov,
    prepare_fov_data,
)


def _make_synthetic_fov(
    *,
    n_per_class: int = 8,
    size: int = 160,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
    """Generate a synthetic (C=6, H, W) multiplexed image, (H, W) mask, and labels."""
    rng = np.random.default_rng(seed)
    n_channels = 6
    image = rng.normal(0.05, 0.02, size=(n_channels, size, size)).clip(0.0, 1.0).astype(np.float32)
    mask = np.zeros((size, size), dtype=np.int32)

    class_profiles = {
        "B cell": np.array([1.0, 0.9, 0.05, 0.05, 0.05, 0.05], dtype=np.float32),
        "T cell": np.array([0.05, 0.05, 1.0, 0.9, 0.05, 0.05], dtype=np.float32),
        "Tumor": np.array([0.05, 0.05, 0.05, 0.05, 1.0, 0.9], dtype=np.float32),
    }

    labels_dict: Dict[int, str] = {}
    cell_id = 1
    classes = list(class_profiles.keys())
    total_cells = n_per_class * len(classes)
    grid_cols = int(np.ceil(np.sqrt(total_cells)))
    step = (size - 24) // max(1, grid_cols)

    for idx in range(total_cells):
        cls_name = classes[idx % len(classes)]
        row_idx = idx // grid_cols
        col_idx = idx % grid_cols
        cy = 14 + row_idx * step
        cx = 14 + col_idx * step
        yy, xx = np.ogrid[:size, :size]
        circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= 4**2
        mask[circle] = cell_id
        for ch_idx, val in enumerate(class_profiles[cls_name]):
            image[ch_idx, circle] = np.clip(val + rng.normal(0.0, 0.03), 0.0, 1.5)
        labels_dict[cell_id] = cls_name
        cell_id += 1

    labels_ser = pd.Series(labels_dict, name="cell_type")
    labels_ser.index = pd.Index(labels_ser.index, name="ObjectNumber", dtype=int)
    return image, mask, labels_ser


class TestCropExtractionAndNormalization:
    def test_crop_extraction_mask_channels_and_border_padding(self) -> None:
        img = np.zeros((4, 80, 80), dtype=np.float32)
        img[0, 1:7, 1:7] = 10.0  # border cell
        img[1, 35:45, 35:45] = 20.0  # center cell
        img[2, 46:54, 46:54] = 30.0  # neighbor cell

        mask = np.zeros((80, 80), dtype=np.int32)
        mask[1:7, 1:7] = 1
        mask[35:45, 35:45] = 2
        mask[46:54, 46:54] = 3

        fov_unpadded = prepare_fov_data(img, mask, crop_size=36, to_pad=False)
        crops_unpadded = extract_crops_from_fov(fov_unpadded, crop_size=36)
        assert [c.cell_id for c in crops_unpadded] == [2, 3]

        fov_padded = prepare_fov_data(img, mask, crop_size=36, to_pad=True)
        crops_padded = extract_crops_from_fov(fov_padded, crop_size=36)
        assert [c.cell_id for c in crops_padded] == [1, 2, 3]

        norm = ChannelNormalizer(mode="quantile", clip_quantile=0.99).fit(
            [fov_padded.image_hwc], channels_first=False
        )
        ds = CellCropsDataset(
            crops_padded, crop_input_size=28, normalizer=norm, augment=False
        )
        sample = ds[1]  # cell_id == 2
        assert sample["cell_id"] == 2
        assert sample["image"].shape == (5, 28, 28)  # 4 channels + all_cells_mask
        assert sample["mask"].shape == (1, 28, 28)  # target cell mask

        all_cells = sample["image"][-1].numpy()
        target_mask = sample["mask"][0].numpy()
        assert target_mask.sum() > 0
        assert all_cells.sum() > target_mask.sum()  # neighbor cell 3 falls inside crop
        assert float(sample["image"][:-1].max()) <= 1.0 + 1e-6

    def test_anscombe_and_none_normalizers(self) -> None:
        arr = np.array([[[0.0, 4.0], [16.0, 0.0]]], dtype=np.float32)
        ans = ChannelNormalizer(mode="anscombe").transform(arr)
        expected = 2.0 * np.sqrt(arr + 0.375)
        np.testing.assert_allclose(ans, expected, rtol=1e-5)

        raw = ChannelNormalizer(mode="none").transform(arr)
        np.testing.assert_allclose(raw, arr)


class TestModelAndSampling:
    def test_resnet50_forward_train_and_eval_modes(self) -> None:
        import torch

        net = CellSighterNet(in_channels=8, num_classes=4, backbone="resnet50")
        assert net.model.conv1.in_channels == 8
        x = torch.randn(3, 8, 60, 60)

        net.train()
        logits = net(x)
        assert logits.shape == (3, 4)

        net.eval()
        with torch.no_grad():
            probs = net(x)
        assert probs.shape == (3, 4)
        assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)
        np.testing.assert_allclose(probs.sum(dim=1).numpy(), np.ones(3), atol=1e-5)

    def test_hierarchical_sampling_weights(self) -> None:
        # 2 groups: "A" has classes 0 (10 cells) and 1 (2 cells); "B" has class 2 (4 cells).
        labels = [0] * 10 + [1] * 2 + [2] * 4
        hier = {"0": "A", "1": "A", "2": "B"}
        weights = compute_sampling_weights(labels, hierarchy_match=hier)
        w_arr = np.asarray(weights)
        total_group_a = w_arr[:12].sum()
        total_group_b = w_arr[12:].sum()
        assert total_group_a == pytest.approx(total_group_b, rel=1e-6)
        assert w_arr[:10].sum() == pytest.approx(w_arr[10:12].sum(), rel=1e-6)

    def test_ensemble_probability_averaging(self) -> None:
        import torch
        from torch.utils.data import DataLoader

        img, mask, labels = _make_synthetic_fov(n_per_class=2, size=96, seed=7)
        fov = prepare_fov_data(img, mask, crop_size=24, to_pad=True)
        crops = extract_crops_from_fov(fov, crop_size=24)
        ds = CellCropsDataset(crops, crop_input_size=20, augment=False)
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        m1 = CellSighterNet(in_channels=8, num_classes=3, backbone="resnet18")
        m2 = CellSighterNet(in_channels=8, num_classes=3, backbone="resnet18")
        torch.manual_seed(11)
        for p in m2.parameters():
            if p.requires_grad:
                p.data.add_(0.05 * torch.randn_like(p))

        dev = torch.device("cpu")
        p1 = ensemble_predict_probabilities([m1], loader, dev)
        p2 = ensemble_predict_probabilities([m2], loader, dev)
        p_ens = ensemble_predict_probabilities([m1, m2], loader, dev)
        np.testing.assert_allclose(p_ens, 0.5 * (p1 + p2), atol=1e-5)


class TestCellSighterAdapter:
    def test_fit_predict_schema_and_checkpoint_roundtrip(self, tmp_path: Path) -> None:
        train_img, train_mask, train_labels = _make_synthetic_fov(
            n_per_class=8, size=160, seed=10
        )
        test_img, test_mask, test_labels = _make_synthetic_fov(
            n_per_class=5, size=160, seed=21
        )

        ckpt = tmp_path / "cellsighter_ckpt.pt"
        model = CellSighterPhenotyper(
            checkpoint_path=ckpt,
            crop_size=24,
            crop_input_size=20,
            backbone="resnet18",
            epochs=6,
            batch_size=8,
            lr=2e-3,
            ensemble_size=2,
            val_fraction=0.0,
            augment=False,
            device="cpu",
            seed=42,
        )
        assert isinstance(model, PhenotyperMethodBase)
        assert model.name == "CellSighter"
        assert model.input_modality == "image_and_mask"
        assert model.requires_training is True
        assert model.is_available() is True
        assert "crop_input_size" in model.get_default_params()

        model.fit(train_img, train_mask, train_labels)
        assert ckpt.exists()

        res = model.predict(test_img, test_mask)
        assert isinstance(res, PhenotypingPredictionResult)
        res.validate()
        assert res.method_name == "CellSighter"
        assert list(res.labels.index) == sorted(test_labels.index.tolist())
        assert res.probabilities is not None
        assert list(res.probabilities.columns) == ["B cell", "T cell", "Tumor"]
        np.testing.assert_allclose(res.probabilities.sum(axis=1).to_numpy(), 1.0, atol=1e-5)
        assert float((res.labels == test_labels.loc[res.labels.index]).mean()) >= 0.75

        reloaded = CellSighterPhenotyper(checkpoint_path=ckpt, device="cpu")
        res_reloaded = reloaded.predict(test_img, test_mask)
        assert res_reloaded.probabilities is not None
        np.testing.assert_allclose(
            res_reloaded.probabilities.to_numpy(),
            res.probabilities.to_numpy(),
            atol=1e-5,
        )

    def test_directory_load_fit_predict_with_blacklist(self, tmp_path: Path) -> None:
        img, mask, labels = _make_synthetic_fov(n_per_class=6, size=128, seed=99)
        class_list = ["B cell", "T cell", "Tumor"]
        cls_map = {name: i for i, name in enumerate(class_list)}

        lbl_vec = np.full((int(mask.max()) + 1,), -1, dtype=np.int32)
        for cid, cname in labels.items():
            lbl_vec[int(cid)] = cls_map[cname]

        cell_types_dir = tmp_path / "CellTypes"
        (cell_types_dir / "data" / "images").mkdir(parents=True)
        (cell_types_dir / "cells").mkdir(parents=True)
        (cell_types_dir / "cells2labels").mkdir(parents=True)

        np.savez(cell_types_dir / "data" / "images" / "fov1.npz", data=np.moveaxis(img, 0, -1))
        np.savez(cell_types_dir / "cells" / "fov1.npz", data=mask)
        np.savez(cell_types_dir / "cells2labels" / "fov1.npz", data=lbl_vec)

        (tmp_path / "channels.txt").write_text(
            "CD20\nCD19\nCD3\nCD4\nCK\nSOX10\n", encoding="utf-8"
        )
        (tmp_path / "labels_strings.txt").write_text(
            "{0: 'B cell', 1: 'T cell', 2: 'Tumor'}\n", encoding="utf-8"
        )

        model = CellSighterPhenotyper(
            crop_size=24,
            crop_input_size=20,
            backbone="resnet18",
            epochs=3,
            batch_size=6,
            ensemble_size=1,
            val_fraction=0.0,
            augment=False,
            blacklist_channels=("CD19",),
            device="cpu",
        )
        model.fit(root_dir=tmp_path, train_set=["fov1"])
        assert model.trainer is not None
        assert model.trainer.in_channels == 7  # 5 unblocked markers + 2 binary masks

        res = model.predict(root_dir=tmp_path, image_names=["fov1"])
        res.validate()
        assert len(res.labels) == len(labels)
