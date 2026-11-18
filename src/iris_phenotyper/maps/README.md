# MAPS (`iris_phenotyper.maps`)

Faithful PyTorch implementation of **MAPS** (Machine learning for Analysis of Proteomics in Spatial biology; Shaban et al., *Nature Communications* 2024, DOI `10.1038/s41467-023-44188-w`, reference repo `mahmoodlab/MAPS`).

## Overview

MAPS is a supervised feed-forward neural network (4-layer MLP with `ReLU` and `Dropout` followed by a linear classification head) operating on single-cell marker expressions and cell size/area (`cellSize` or `area`) — **not** a spatial graph neural network.

Key properties faithfully matched to `mahmoodlab/MAPS`:
- Architecture (`MAPSMLP`): 4 hidden blocks (`Linear(hidden_dim=512) -> ReLU -> Dropout(p)`) + `Linear(512, num_classes)`. Checkpoint keys (`fc.*`, `classifier.*`) are directly compatible with official pretrained `.pt` checkpoints.
- Preprocessing (`standardize_features`): per-feature training mean and standard deviation z-score followed by `/ 255.0` scaling (`((X - mean) / std) / 255.0`).
- Training (`MAPSTrainer`): `WeightedRandomSampler` for class-balanced mini-batches, `Adam(lr=1e-3)`, `CrossEntropyLoss`, and validation-loss early stopping.
- Device & dtype: `float64` on CPU/CUDA (matching upstream `Trainer`), automatic `float32` on Apple Silicon `mps`.
- IRIS integration (`MAPSPhenotyper(PhenotyperMethodBase)`): accepts both canonical IRIS feature tables (`ObjectNumber`, `<Marker>_mean`, `area`, `cell_type`) and upstream MAPS CSV tables (`<Marker>`, `cellSize`, `cell_label`), returning a validated `PhenotypingPredictionResult`.

## Quick Usage

```python
from iris_phenotyper.maps import MAPSPhenotyper

model = MAPSPhenotyper(max_epochs=100, min_epochs=20, patience=15, dropout=0.25)
model.fit(train_df, label_col="cell_type", valid_features=valid_df)
result = model.predict(test_df)
```

Loading an official upstream checkpoint:

```python
model = MAPSPhenotyper.from_checkpoint(
    "/path/to/cHL_CODEX.pt",
    class_names=CHL_CODEX_CLASS_NAMES,
)
result = model.predict(valid_df)
```

## Paper Reproduction

Run `python -m iris_phenotyper.maps.reproduce_paper` (after downloading datasets into `/Users/wwu/Documents/IRIS_Benchmarks/maps/data/`) to evaluate both official pretrained checkpoints and freshly trained MAPS models against published benchmarks and write outputs to `/Users/wwu/Documents/IRIS_Benchmarks/maps/results/`.
