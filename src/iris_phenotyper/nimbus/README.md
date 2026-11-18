# Nimbus Phenotyping Module (`iris_phenotyper.nimbus`)

Zero-shot deep learning classification of single-cell marker positivity in multiplexed imaging data, faithfully implementing **Nimbus** within the IRIS phenotyping contract.

## Citation & Upstream Provenance

- **Paper**: Rumberger, J. L., Greenwald, N. F., Ranek, J. S., *et al.* "Automated classification of cellular expression in multiplexed imaging data with Nimbus." *Nature Methods* **22**, 2161-2170 (2025).
  - Preprint DOI: [`10.1101/2024.06.02.597062`](https://doi.org/10.1101/2024.06.02.597062)
  - Published DOI: [`10.1038/s41592-025-02826-9`](https://doi.org/10.1038/s41592-025-02826-9)
- **Upstream Code**: [`angelolab/Nimbus-Inference`](https://github.com/angelolab/Nimbus-Inference) (MIT License, commit `241281ef85ba34d9857cc03886cc45aedd0f0b34`)
- **Pretrained Checkpoint**: [`JLrumberger/Nimbus-Inference`](https://huggingface.co/JLrumberger/Nimbus-Inference) (`V1.pt`, Residual U-Net)
- **Benchmark Dataset**: [`JLrumberger/Pan-Multiplex-Gold-Standard`](https://huggingface.co/datasets/JLrumberger/Pan-Multiplex-Gold-Standard)

## Algorithmic Summary

1. **FOV Quantile Normalization (`prepare_normalization_dict`)**: For each marker channel across up to `n_subset=50` FOVs, computes the `0.999` quantile of strictly positive foreground pixels (`img > 0`), averages across FOVs, and scales/clips images to `[0, 1]`.
2. **2-Channel Input Construction (`prepare_binary_mask`, `prepare_input_data`)**: Pairs each normalized marker channel with a binary cell-interior mask whose inner segmentation boundaries (`skimage.segmentation.find_boundaries(..., mode="inner")`) are zeroed out to separate touching cells.
3. **Magnification Rescaling & Shift-Equivariant Inference (`predict_channel_map`)**: Rescales inputs from `dataset_magnification` (default `20.0`) to the model's `10.0` reference magnification, applies optional 6-view test-time augmentation (4 rotations + 2 flips), and runs either reflect-padded single-shot inference or valid-padded tile-and-stitch inference for large FOVs.
4. **Cell-Level Positivity Aggregation (`aggregate_cell_positivity`)**: Averages pixel-level predicted probabilities inside each segmented `ObjectNumber` instance via `skimage.measure.regionprops_table` (`intensity_mean` in `[0, 1]`).
5. **Downstream Cell Phenotyping**:
   - With a `CellTypePanel`: computes rule compatibility scores and normalized cell-type posteriors (`assign_cell_types_from_panel`).
   - Without a panel: runs z-scored/clipped (`[-3, 3]`) two-stage SOM-style prototype quantization + Ward hierarchical consensus clustering (`cluster_positivity_profiles`), mirroring `4_Generic_Cell_Clustering.ipynb`.

## Usage

```python
from iris_phenotyper.nimbus import NimbusPhenotyper

pheno = NimbusPhenotyper(
    dataset_magnification=20.0,
    test_time_aug=True,
    device="auto",
)

# 1. Direct in-memory (C, H, W) image + (H, W) instance mask
result = pheno.predict(image_chw, mask_hw, channel_names=markers)
positivity_df = result.marker_positivity  # ObjectNumber x markers in [0, 1]

# 2. Directory / OME-TIFF FOV workflow (uses nimbus_inference when installed)
result = pheno.predict(
    fov_paths=fov_dirs,
    mask_dir=masks_dir,
    approx_mask_match=True,
    channel_names=markers,
    suffix=".ome.tif",
)
```

## Environments

- **Dedicated Nimbus environment** (`/Users/wwu/.conda/envs/iris_nimbus`, Python 3.11): includes `Nimbus-Inference` (`numpy<2.0.0`), `torch`, `huggingface_hub`, `scanpy`, `leidenalg`, and `iris-viewer` (editable). Use this environment for full `nimbus_inference` parity and paper benchmark reproduction.
- **Core IRIS environment** (`/Users/wwu/.conda/envs/iris`): provides the pure-Python/PyTorch `NimbusPhenotyper` adapter and clustering utilities with lazy `torch` import.

## Reproducing the Paper Benchmark

```bash
/Users/wwu/.conda/envs/iris_nimbus/bin/python -m iris_phenotyper.nimbus.reproduce_paper
```
