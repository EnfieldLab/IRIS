"""Unit tests for iris_phenotyper.maps."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from iris_phenotyper.base import PhenotyperMethodBase, PhenotypingPredictionResult
from iris_phenotyper.maps import (
    MAPSMLP,
    MAPSPhenotyper,
    MAPSTabularDataset,
    MAPSTrainer,
    resolve_device,
    resolve_dtype,
    standardize_features,
)


def _make_synthetic_iris_df(
    n_per_class: int = 40,
    seed: int = 42,
    start_id: int = 101,
) -> pd.DataFrame:
    """Create a well-separated synthetic single-cell table using canonical IRIS columns."""
    rng = np.random.default_rng(seed)
    classes = ["B cell", "T cell", "Macrophage"]
    centers = {
        "B cell": np.array([0.85, 0.10, 0.10, 0.20, 220.0]),
        "T cell": np.array([0.10, 0.85, 0.15, 0.25, 180.0]),
        "Macrophage": np.array([0.15, 0.15, 0.85, 0.80, 480.0]),
    }
    rows = []
    obj_id = start_id
    for ct in classes:
        c = centers[ct]
        for _ in range(n_per_class):
            noise = rng.normal(0.0, 0.05, size=4)
            size_noise = rng.normal(0.0, 20.0)
            rows.append(
                {
                    "ObjectNumber": obj_id,
                    "Image": "sample_01",
                    "Pos_X": float(rng.uniform(0, 500)),
                    "Pos_Y": float(rng.uniform(0, 500)),
                    "CD20_mean": float(np.clip(c[0] + noise[0], 0.0, 1.0)),
                    "CD3_mean": float(np.clip(c[1] + noise[1], 0.0, 1.0)),
                    "CD68_mean": float(np.clip(c[2] + noise[2], 0.0, 1.0)),
                    "HLA_DR_mean": float(np.clip(c[3] + noise[3], 0.0, 1.0)),
                    "area": float(max(20.0, c[4] + size_noise)),
                    "perimeter": float(rng.uniform(30.0, 90.0)),
                    "eccentricity": float(rng.uniform(0.1, 0.9)),
                    "cell_type": ct,
                }
            )
            obj_id += 2
    return pd.DataFrame(rows)


def test_maps_contract_and_defaults() -> None:
    model = MAPSPhenotyper()
    assert isinstance(model, PhenotyperMethodBase)
    assert model.name == "MAPS"
    assert model.input_modality == "tabular"
    assert model.requires_training is True
    assert model.is_available() is True

    params = model.get_default_params()
    assert params["hidden_dim"] == 512
    assert params["scale_divisor"] == 255.0
    assert params["learning_rate"] == pytest.approx(0.001)
    assert params["include_cell_size"] is True


def test_maps_mlp_architecture_shapes_and_keys() -> None:
    import torch

    net = MAPSMLP(input_dim=10, hidden_dim=32, num_classes=4, dropout=0.2)
    keys = set(net.state_dict().keys())
    expected_keys = {
        "fc.0.weight",
        "fc.0.bias",
        "fc.3.weight",
        "fc.3.bias",
        "fc.6.weight",
        "fc.6.bias",
        "fc.9.weight",
        "fc.9.bias",
        "classifier.weight",
        "classifier.bias",
    }
    assert keys == expected_keys

    x = torch.randn(7, 10)
    logits, probs = net(x)
    assert logits.shape == (7, 4)
    assert probs.shape == (7, 4)
    np.testing.assert_allclose(probs.sum(dim=-1).detach().numpy(), np.ones(7), atol=1e-6)


def test_standardize_features_and_zero_variance_guard() -> None:
    X = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]], dtype=np.float64)
    scaled, mean, std = standardize_features(X, scale_divisor=255.0)
    assert np.all(np.isfinite(scaled))
    np.testing.assert_allclose(mean, [3.0, 5.0])
    np.testing.assert_allclose(std, [np.std([1.0, 3.0, 5.0]), 0.0])
    np.testing.assert_allclose(scaled[:, 1], [0.0, 0.0, 0.0])
    expected_col0 = ((X[:, 0] - 3.0) / np.std([1.0, 3.0, 5.0])) / 255.0
    np.testing.assert_allclose(scaled[:, 0], expected_col0)


def test_fit_predict_roundtrip_canonical_iris_schema() -> None:
    train_df = _make_synthetic_iris_df(n_per_class=45, seed=7, start_id=1)
    test_df = _make_synthetic_iris_df(n_per_class=15, seed=99, start_id=501)
    y_test = test_df["cell_type"].copy()

    model = MAPSPhenotyper(
        hidden_dim=64,
        dropout=0.1,
        batch_size=16,
        learning_rate=0.005,
        max_epochs=80,
        min_epochs=80,
        patience=20,
        device="cpu",
        seed=123,
    )
    fitted = model.fit(train_df, label_col="cell_type")
    assert fitted is model
    assert model.feature_names_ == [
        "CD20_mean",
        "CD3_mean",
        "CD68_mean",
        "HLA_DR_mean",
        "area",
    ]
    assert model.class_names_ == ["B cell", "Macrophage", "T cell"]

    res = model.predict(test_df.drop(columns=["cell_type"]))
    assert isinstance(res, PhenotypingPredictionResult)
    res.validate()

    assert res.method_name == "MAPS"
    assert list(res.labels.index) == test_df["ObjectNumber"].tolist()
    assert list(res.probabilities.columns) == ["B cell", "Macrophage", "T cell"]
    np.testing.assert_allclose(res.probabilities.sum(axis=1).to_numpy(), 1.0, atol=1e-6)
    np.testing.assert_allclose(
        res.confidence.to_numpy(),
        res.probabilities.max(axis=1).to_numpy(),
        atol=1e-8,
    )

    acc = float(np.mean(res.labels.to_numpy() == y_test.to_numpy()))
    assert acc >= 0.90


def test_explicit_valid_and_alias_bridging_and_checkpoint_roundtrip(tmp_path) -> None:
    train_raw = pd.DataFrame(
        {
            "CD20": [0.9, 0.85, 0.1, 0.12, 0.15, 0.1, 0.88, 0.12],
            "CD3": [0.1, 0.15, 0.9, 0.88, 0.12, 0.14, 0.1, 0.92],
            "cellSize": [210, 220, 175, 185, 470, 490, 205, 180],
            "cell_label": [0, 0, 1, 1, 2, 2, 0, 1],
        }
    )
    valid_raw = train_raw.iloc[:6].copy()
    class_names = ["B", "CD4 T", "Macrophage"]

    model = MAPSPhenotyper(
        hidden_dim=32,
        batch_size=4,
        learning_rate=0.03,
        max_epochs=20,
        min_epochs=10,
        patience=5,
        device="cpu",
    )
    model.fit(
        train_raw,
        label_col="cell_label",
        valid_features=valid_raw,
        class_names=class_names,
    )

    iris_query = pd.DataFrame(
        {
            "ObjectNumber": [10, 20, 30],
            "CD20_mean": [0.90, 0.10, 0.12],
            "CD3_mean": [0.10, 0.91, 0.11],
            "area": [215.0, 180.0, 480.0],
        }
    )
    res_orig = model.predict(iris_query)

    ckpt_path = model.save(tmp_path / "maps_roundtrip.pt")
    assert ckpt_path.exists()

    import torch

    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert {"model_parameters", "train_data_mean", "train_data_std"}.issubset(raw_ckpt.keys())

    reloaded = MAPSPhenotyper.from_checkpoint(ckpt_path, device="cpu")
    res_reloaded = reloaded.predict(iris_query)
    res_reloaded.validate()

    assert res_orig.labels.equals(res_reloaded.labels)
    np.testing.assert_allclose(
        res_orig.probabilities.to_numpy(),
        res_reloaded.probabilities.to_numpy(),
        atol=1e-7,
    )


def test_device_resolution_mps_and_edge_cases() -> None:
    import torch

    cpu_dev = resolve_device("cpu")
    assert cpu_dev.type == "cpu"
    assert resolve_dtype(cpu_dev, "auto") == torch.float64
    assert resolve_dtype(cpu_dev, "float32") == torch.float32

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        mps_dev = resolve_device("mps")
        assert mps_dev.type == "mps"
        assert resolve_dtype(mps_dev, "auto") == torch.float32
        with pytest.raises(ValueError, match="float64"):
            resolve_dtype(mps_dev, "float64")

        df = _make_synthetic_iris_df(n_per_class=20, seed=5)
        mps_model = MAPSPhenotyper(
            hidden_dim=32,
            batch_size=16,
            max_epochs=6,
            min_epochs=3,
            device="mps",
        )
        mps_model.fit(df, label_col="cell_type")
        res = mps_model.predict(df.drop(columns=["cell_type"]))
        res.validate()
        assert res.metadata["device"] == "mps"
        assert "float32" in res.metadata["dtype"]

    # Unfitted predict error
    unfitted = MAPSPhenotyper()
    with pytest.raises(RuntimeError):
        unfitted.predict(pd.DataFrame({"CD3_mean": [0.5], "area": [100.0]}))

    # Empty fit error
    with pytest.raises(ValueError, match="empty"):
        unfitted.fit(pd.DataFrame())

    # NaN feature error & missing feature error & empty predict
    df_small = _make_synthetic_iris_df(n_per_class=10, seed=11)
    fitted = MAPSPhenotyper(hidden_dim=16, batch_size=8, max_epochs=4, min_epochs=2, device="cpu")
    fitted.fit(df_small, label_col="cell_type")

    empty_res = fitted.predict(df_small.iloc[:0].drop(columns=["cell_type"]))
    empty_res.validate()
    assert len(empty_res.labels) == 0

    bad_nan = df_small.iloc[:2].drop(columns=["cell_type"]).copy()
    bad_nan.iloc[0, bad_nan.columns.get_loc("CD20_mean")] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fitted.predict(bad_nan)

    with pytest.raises(ValueError, match="Missing required"):
        fitted.predict(pd.DataFrame({"ObjectNumber": [1], "CD20_mean": [0.5]}))

    # MAPSTrainer direct error when calling predict before fit
    trainer = MAPSTrainer(num_features=3, num_classes=2)
    with pytest.raises(RuntimeError):
        trainer.predict(np.zeros((2, 3)))


def test_csv_dataset_and_result_conversions(tmp_path) -> None:
    df = _make_synthetic_iris_df(n_per_class=12, seed=21, start_id=5)
    csv_df = df[["CD20_mean", "CD3_mean", "CD68_mean", "HLA_DR_mean", "area"]].copy()
    csv_df["cell_label"] = df["cell_type"].map({"B cell": 0, "Macrophage": 1, "T cell": 2})
    csv_path = tmp_path / "sample_maps.csv"
    csv_df.to_csv(csv_path, index=False)

    ds = MAPSTabularDataset.from_csv(csv_path, is_train=True)
    assert len(ds) == len(csv_df)
    feat0, gt0 = ds[0]
    assert feat0.shape == (5,)
    assert gt0 in (0, 1, 2)

    # include_cell_size=False excludes area/cellSize from auto-detected features
    no_size_model = MAPSPhenotyper(
        include_cell_size=False,
        hidden_dim=16,
        batch_size=8,
        max_epochs=6,
        min_epochs=3,
        device="cpu",
    )
    no_size_model.fit(df, label_col="cell_type")
    assert no_size_model.feature_names_ == [
        "CD20_mean",
        "CD3_mean",
        "CD68_mean",
        "HLA_DR_mean",
    ]
    res = no_size_model.predict(df.drop(columns=["cell_type"]))
    res.validate()

    summary_df = res.to_dataframe(include_probabilities=True)
    assert "ObjectNumber" in summary_df.columns
    assert "cell_type" in summary_df.columns
    assert "prob_B cell" in summary_df.columns
    assert len(summary_df) == len(df)

    legacy = res.to_legacy_dict()
    assert len(legacy) == len(df)
    first_id = int(df["ObjectNumber"].iloc[0])
    assert pytest.approx(sum(legacy[first_id].values()), abs=1e-5) == 1.0
