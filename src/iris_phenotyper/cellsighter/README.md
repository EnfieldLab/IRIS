# IRIS CellSighter Phenotyper

Faithful PyTorch implementation of **CellSighter** (Amitay et al., *Nature Communications* 2023,
DOI: `10.1038/s41467-023-40066-7`; upstream `KerenLab/CellSighter`) integrated into the
`iris_phenotyper` package.

## Overview

CellSighter classifies single cells directly from multiplexed tissue images and instance
segmentation masks without relying solely on tabular mean-expression summaries:

1. **Spatial Crop Extraction**: Extracts a `crop_size x crop_size` (default `128 x 128`) window
   centered on each cell bounding box (`scipy.ndimage.find_objects`), optionally zero-padding FOV
   borders (`to_pad=True`) so boundary cells are retained.
2. **Two Appended Binary Mask Channels**: Appends `all_cells_mask` (`mask > 0`, channel `-2`) and
   `target_cell_mask` (`mask == cell_id`, channel `-1`) to the `C` marker channels, yielding a
   `(C + 2, crop_input_size, crop_input_size)` input tensor (default `crop_input_size=60`).
3. **Channel Normalization**: Supports foreground-aware per-channel quantile clipping and scaling
   into `[0, 1]` (`normalization="quantile"`, `clip_quantile=0.99`), Anscombe variance
   stabilization (`normalization="anscombe"`), or raw intensities (`normalization="none"`).
4. **Data Augmentations**: Gaussian-blur Poisson count resampling (for raw ion-count data), random
   target/environment mask dilation, 360-degree rotation, channel-independent `ShiftAugmentation`,
   center crop to `crop_input_size`, and random flips.
5. **Backbone & Ensemble**: ResNet50 (default) with `conv1` replaced for `C + 2` channels and
   Kaiming normal initialization, trained with Adam + CrossEntropyLoss and optional hierarchical
   `WeightedRandomSampler` (`hierarchy_match`, `sample_batch`). Predictions across ensemble members
   are averaged in probability space.

## Usage

### In-Memory Arrays (`(C, H, W)` Image + `(H, W)` Mask)

```python
from iris_phenotyper.cellsighter import CellSighterPhenotyper

model = CellSighterPhenotyper(
    crop_size=128,
    crop_input_size=60,
    backbone="resnet50",
    epochs=20,
    ensemble_size=2,
)
model.fit(train_image_chw, train_mask_hw, train_labels_series)
result = model.predict(test_image_chw, test_mask_hw)
result.validate()
```

### CellSighter Directory Format (`.npz` / `.tiff`)

```python
model = CellSighterPhenotyper(epochs=20, ensemble_size=2)
model.fit(
    root_dir="/Users/wwu/Documents/IRIS_Benchmarks/cellsighter/data",
    train_set=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    val_set=["11", "12"],
)
result = model.predict(
    root_dir="/Users/wwu/Documents/IRIS_Benchmarks/cellsighter/data",
    image_names=["13", "14", "15", "16"],
)
```

## Paper Reproduction Script

```bash
/Users/wwu/.conda/envs/iris/bin/python -m iris_phenotyper.cellsighter.reproduce_paper \
    --epochs 20 --ensemble-size 2 --samples-per-epoch 10240
```

Outputs are written to `/Users/wwu/Documents/IRIS_Benchmarks/cellsighter/results/`.
