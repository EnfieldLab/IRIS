# `iris_phenotyper.astir`

Faithful PyTorch variational inference implementation and `PhenotyperMethodBase` adapter for **ASTIR** (Automated aSsignmenT sIngle-cell pRoteomics; Geuenich et al., *Cell Systems* 2021, DOI `10.1016/j.cels.2021.08.012`).

## Architecture & Scientific Faithfulness

- **Recognition Networks**:
  - `TypeRecognitionNet`: 2-layer MLP (`G -> 20 -> LeakyReLU -> C + 1`) predicting variational posterior probabilities $\gamma = q(c \mid x)$ over $C$ defined cell types plus `"Other"`.
  - `StateRecognitionNet`: 2-hidden-layer MLP with Gaussian reparameterization predicting pathway activation mean $\mu_z$ and log-scale $\log \sigma_z$.
- **Generative Model & ELBO**:
  - Models expression $Y \mid c$ via `LowRankMultivariateNormal` parameterized by baseline expression $\mu$ (with optional batch design matrix $D$), cell-type marker indicator matrix $\rho \in \{0, 1\}^{G \times (C+1)}$ with over-expression weights $\tilde{\delta} = \exp(\text{log\_delta})$, rank-1 covariance factor $(\rho \odot \sigma(p))^T \sigma$, and `Dirichlet(C + 1)` mixing prior on $\alpha$.
  - Multi-start initialization (`n_init` restarts for `n_init_epochs` epochs) selecting the lowest ELBO restart before full convergence with Adam.
- **Preprocessing & Schema Support**:
  - Supports YAML marker files (`cell_types`, `cell_states`, `hierarchy`), nested or flat Python dictionaries, and `iris_common.types.CellTypePanel` / `MarkerPanel`.
  - Resolves canonical IRIS feature columns (`<marker>_mean` or exact marker names) via `iris_common.schema.resolve_marker_column` and supports `AnnData` or `pd.DataFrame`.
  - Supports `arcsinh(x / 5.0)` transformation (`apply_arcsinh=True | False | "auto"`) and optional percentile winsorization.
- **Post-Fit Outputs**:
  - Calibrated posterior probabilities `(N, C + 1)` including `"Other"`.
  - Probability-threshold gating (`threshold=0.7` or `0.5`) assigning low-confidence cells to `"Unknown"`.
  - Hierarchical lineage probability summation (`assign_celltype_hierarchy`), cell-type marker Welch/Student t-test diagnostics (`diagnostics_celltype`), and cell-state correlation diagnostics (`diagnostics_cellstate`).
- **Dual Execution Engine**:
  - `"iris"` (default when upstream `astir` is absent): PyTorch 2.x / NumPy 2.x compatible implementation running natively in `/Users/wwu/.conda/envs/iris`.
  - `"official"`: delegates directly to upstream `camlab-bioml/astir` when installed (e.g., `/Users/wwu/.conda/envs/iris_astir`).

## Usage

```python
from iris_phenotyper.astir import ASTIRPhenotyper

model = ASTIRPhenotyper(
    marker_config="/path/to/markers.yml",
    threshold=0.7,
    apply_arcsinh="auto",
    fit_states=True,
)
result = model.predict(cell_features_df)
result.validate()
```

## Paper Reproduction Benchmark

Dataset: `basel_22k_subset.h5ad` (22,717 cells x 38 markers, Jackson et al. 2020 breast cancer IMC subset) with `jackson-2020-markers-v4.yml` and Jackson et al. annotations from `astir-manuscript` (Zenodo `10.5281/zenodo.5156034`).

Run reproduction:

```bash
/Users/wwu/.conda/envs/iris/bin/python -m iris_phenotyper.astir.reproduce_paper
```

Observed benchmark metrics (`threshold=0.7` / `0.5`, `max_epochs=1000`, `lr=2e-3`, `batch_size=227`, `n_init=5`, `n_init_epochs=3`):

| Comparison | Accuracy | Macro F1 | Weighted F1 | Cohen's Kappa | Posterior r |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official `astir` vs IRIS ASTIR (`thresh=0.7`) | 0.9709 | 0.9726 | 0.9709 | 0.9642 | 0.9988 |
| Official `astir` vs IRIS ASTIR (`thresh=0.5`) | 0.9778 | 0.9475 | 0.9778 | 0.9717 | 0.9988 |
| IRIS ASTIR vs Jackson Coarse Assigned (`N=10,822`) | 0.9370 | 0.7746 | 0.9352 | 0.8525 | - |
| Official `astir` vs Jackson Coarse Assigned (`N=10,757`) | 0.9375 | 0.7730 | 0.9356 | 0.8542 | - |
| Vignette Published Counts vs IRIS (`N=22,717`) | - | - | - | - | 1.0000 (0.99997) |

## Files

- `__init__.py`: Subpackage exports and macOS OpenMP runtime guard.
- `pipeline.py`: `ASTIRDataset`, `ASTIRCellTypeModel`, `ASTIRCellStateModel`, `ASTIRPipeline`, diagnostics, and hierarchy helpers.
- `adapter.py`: `ASTIRPhenotyper(PhenotyperMethodBase)` returning `PhenotypingPredictionResult`.
- `reproduce_paper.py`: End-to-end Basel 22k official vs IRIS benchmark script.
