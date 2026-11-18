# iris_phenotyper

Unified, scientifically faithful cell phenotyping implementations and benchmarking utilities for multiplexed imaging data in IRIS (`iris_phenotyper` v`0.2.0`).

---

## Package Architecture

Every phenotyping method is packaged as a standalone subpackage under `src/iris_phenotyper/<method>/` with five standard modules:

- `__init__.py` -- public method exports
- `pipeline.py` -- faithful reproduction of the paper's preprocessing, normalization, architecture, training/optimization, and inference
- `adapter.py` -- `PhenotyperMethodBase` subclass bridging canonical IRIS inputs to `PhenotypingPredictionResult`
- `reproduce_paper.py` -- CLI/script driver for reproducing published benchmarks against external datasets in `/Users/wwu/Documents/IRIS_Benchmarks/`
- `README.md` -- method-specific usage notes, parameters, and benchmark details

Shared core interfaces live at the package root (`src/iris_phenotyper/__init__.py`):

- `PhenotyperMethodBase` (`base.py`) -- base interface implemented by all five adapters
- `PhenotypingPredictionResult` (`base.py`) -- standardized prediction container with fields:
  - `method_name: str`
  - `labels: pd.Series`
  - `probabilities: Optional[pd.DataFrame]`
  - `confidence: pd.Series`
  - `is_novel: Optional[pd.Series] = None`
  - `marker_positivity: Optional[pd.DataFrame] = None`
  - `metadata: Dict[str, Any]`
  - Helper methods: `.validate()`, `.to_dataframe(...)`, `.to_legacy_dict()`
- `EvaluationResult`, `evaluate_predictions`, `format_benchmark_report` (`evaluation.py`) -- standardized evaluation and report formatting utilities

### Supported Methods

| Adapter Class | Import Path | `name` | `input_modality` | `requires_training` | Subpackage Docs |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `MAPSPhenotyper` | `iris_phenotyper.maps` | `"MAPS"` | `"tabular"` | `True` | [`maps/README.md`](maps/README.md) |
| `STELLARPhenotyper` | `iris_phenotyper.stellar` | `"STELLAR"` | `"spatial_tabular"` | `True` | [`stellar/README.md`](stellar/README.md) |
| `CellSighterPhenotyper` | `iris_phenotyper.cellsighter` | `"CellSighter"` | `"image_and_mask"` | `True` | [`cellsighter/README.md`](cellsighter/README.md) |
| `ASTIRPhenotyper` | `iris_phenotyper.astir` | `"ASTIR"` | `"tabular"` | `False` | [`astir/README.md`](astir/README.md) |
| `NimbusPhenotyper` | `iris_phenotyper.nimbus` | `"Nimbus"` | `"image_and_mask"` | `False` | [`nimbus/README.md`](nimbus/README.md) |

Example headless import check:

```python
from iris_phenotyper import (
    EvaluationResult,
    PhenotyperMethodBase,
    PhenotypingPredictionResult,
    evaluate_predictions,
    format_benchmark_report,
)
from iris_phenotyper.astir import ASTIRPhenotyper
from iris_phenotyper.cellsighter import CellSighterPhenotyper
from iris_phenotyper.maps import MAPSPhenotyper
from iris_phenotyper.nimbus import NimbusPhenotyper
from iris_phenotyper.stellar import STELLARPhenotyper
```

---

## Benchmark Comparison Across Published Datasets

External datasets, cloned upstream repositories, SHA-256 manifests, and benchmark outputs are kept outside the `IRIS` repository under `/Users/wwu/Documents/IRIS_Benchmarks/`. See `/Users/wwu/Documents/IRIS_Benchmarks/BENCHMARK_VERIFICATION_REPORT.md` for full provenance and per-cohort breakdowns.

| Method | Modality | Dataset(s) & Link/DOI | Download Status | Published Paper Metric(s) | IRIS Reproduced Metric(s) | Verification Status | Any Manually Needed Files |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MAPS** | `tabular` | Zenodo `10.5281/zenodo.10067010` (`Source_Data.zip` w/ 143,730-cell `cHL_CODEX_MAPS.csv` & `Figure2B_F1_score.csv`; `cHL_2_MIBI.zip`; upstream `train.csv`/`valid.csv`, `N=28,746`) | `downloaded_and_verified` | `cHL_CODEX`: 5-fold Macro-F1 `0.8465 +/- 0.0022` (Fold 0 `0.8436`); `cHL2_MIBI`: 5-fold Macro-F1 `0.8121 +/- 0.0167` (Fold 0 `0.7914`) | `cHL_CODEX` official 5-fold Mean Macro-F1 `0.8465 +/- 0.0022` (Acc `0.8560`, ROC-AUC `0.9950`), Fresh IRIS Fold 0 Macro-F1 `0.8405` (Acc `0.8446`); `cHL2_MIBI` official 5-fold Mean Macro-F1 `0.8121 +/- 0.0167` (Acc `0.8387`), Fresh IRIS Fold 0 Macro-F1 `0.7907` (Acc `0.8299`) | **PASS** (`7 passed` in `iris`) | None for primary benchmarks (`cHL_CODEX.zip` 3.39 GB & `cHL_1_MIBI.zip` 11.88 GB raw archives skipped under disk guard) |
| **STELLAR** | `spatial_tabular` | Dryad `10.5061/dryad.g4f4qrfrc` (`B004_training_dryad.csv`, `B0056_unnanotated_dryad.csv`, `BE_Tonsil_l3_dryad.csv`, `README_CODEXdata_STELLAR.docx`) | `downloaded_and_verified` | Tonsil -> BE (`BE_Tonsil_l3_dryad.csv`) project site accuracy `0.93` | Tonsil -> BE (86,984 ref -> 45,958 BE) excl. rare `SmoothMuscle` Acc `0.9418` (open-world Acc `0.7569`, Bal-Acc `0.8492`, Novel F1 `0.9843`); B004 `reg1-3 -> reg4` (`N=25,063`) Acc `0.8918`, Macro ROC-AUC `0.9938`; B004 CL -> B005 CL (`N=57,904`) 20/20 types, mean `r = 0.9069` | **PASS** (`7 passed, 1 skipped` in `iris`; `8 passed` in `iris_stellar`) | None for primary benchmarks (`10.5061/dryad.pk0p2ngrf` 2.91 GB optional external archive not required) |
| **CellSighter** | `image_and_mask` | Mendeley `10.17632/zcz8743fcv.2` (CC BY 4.0, 50/50 files, `186,105,832` B) | `downloaded_and_verified` | >0.80 (~0.85 across major types; recalls ~0.93 CD8 T, ~0.88 CD4 T, ~0.83 CD4 Treg, ~0.82 Vessels) | Test FOVs 13-16 (`N=26,286`, 2-model ResNet50 ensemble, `mps`): Acc `0.7711`, Bal-Acc `0.7919`, Weighted-F1 `0.7712`, Kappa `0.7331`, Macro ROC-AUC `0.9795`; recalls CD8 T `0.9250`, CD4 Treg `0.8889`, Vessels `0.9055`, Tumor `0.9850` (F1 `0.9654`) | **PASS** (`7 passed` in `iris`) | None for primary benchmarks (`10.5281/zenodo.6004986` `full_data.zip` 17.93 GB skipped under disk guard) |
| **ASTIR** | `tabular` | `Astir-Vignette` (`basel_22k_subset.h5ad`, `jackson-2020-markers-v4.yml`) & Zenodo `10.5281/zenodo.5156034` (`astir-manuscript-v1.0.0.zip`) | `downloaded_and_verified` | Jackson 2020 `basel_22k_subset.h5ad` (22,717 cells) vignette counts & official `astir 0.2.0` (assigned `thresh=0.5`, `N=10,757`: Acc `0.9375`, Macro-F1 `0.7730`, Kappa `0.8542`) | Vignette count correlation `r = 0.999972`; Official vs IRIS posterior `r = 0.9988`, `rho = 0.9976`, MAE `0.00334`, label agreement `0.9778` (`thresh=0.5`); Jackson coarse assigned (`N=10,822`): Acc `0.9370`, Macro-F1 `0.7746`, Kappa `0.8525` | **PASS** (`8 passed` in `iris`; `8 passed` in `iris_astir`) | None |
| **Nimbus** | `image_and_mask` | HuggingFace `JLrumberger/Pan-Multiplex-Gold-Standard` (15 FOVs, 5 cohorts, 92,229 cells, 1,059,159 annotations, 2.8 GB unpacked) + `V1.pt` | `downloaded_and_verified` | `experimental_results.csv` (`V1.pt`, TTA=True, thresh=0.5, 50-FOV norm): Precision `0.8542`, Recall `0.8498`, Specificity `0.9854`, Positive F1 `0.8502` | `V1.pt`, TTA=True, thresh=0.5 (`N=1,059,159`, 3-FOV/dataset norm): Pos Prec `0.8995`, Recall `0.7815`, Specificity `0.9911`, Pos F1 `0.8364`, Acc `0.9718`, Bal-Acc `0.8863`, Binary Macro-F1 `0.9105`, Kappa `0.8210`, Pooled AUROC `0.9886`, AUPRC `0.9158`, Mean AUROC `0.9843` | **PASS** (`8 passed, 1 skipped` in `iris`; `9 passed` in `iris_nimbus`) | None for primary benchmarks (full `Pan-Multiplex` >200 GB corpus skipped under disk budget) |

---

## Environments & Running Tests

The primary `/Users/wwu/.conda/envs/iris` environment runs all five methods natively in pure PyTorch / NumPy / scikit-learn without requiring upstream forks or PyTorch Geometric. Optional extras declared in `pyproject.toml` (`astir`, `maps`, `stellar`, `nimbus`, `spatial`) and dedicated upstream parity environments (`iris_astir`, `iris_stellar`, `iris_nimbus`) enable cross-checking against reference packages.

Run all phenotyper tests and linter checks from the repository root:

```bash
/Users/wwu/.conda/envs/pytorch/bin/ruff check src/iris_phenotyper/ tests/phenotyper/
/Users/wwu/.conda/envs/iris/bin/pytest tests/phenotyper/ -v
```

Run upstream parity suites in their dedicated environments when installed:

```bash
PYTHONPATH=/Users/wwu/Documents/IRIS/src /Users/wwu/.conda/envs/iris_astir/bin/pytest tests/phenotyper/test_astir.py -v
PYTHONPATH=/Users/wwu/Documents/IRIS/src /Users/wwu/.conda/envs/iris_stellar/bin/pytest tests/phenotyper/test_stellar.py -v
PYTHONPATH=/Users/wwu/Documents/IRIS/src /Users/wwu/.conda/envs/iris_nimbus/bin/pytest tests/phenotyper/test_nimbus.py -v
```
