# STELLAR Spatial Single-Cell Phenotyper (`iris_phenotyper.stellar`)

Faithful IRIS implementation of **STELLAR** (*SpaTial cELl LeARning*) from Brbic et al., *Nature Methods* 2022 (`doi:10.1038/s41592-022-01651-8`, upstream `snap-stanford/stellar`).

## Algorithm Summary

1. **Spatial Graph Construction (`build_spatial_graph`)**:
   Constructs intra-region cellular adjacency graphs connecting cells closer than `distance_thres` (default `50.0` pixels/um, strict `< distance_thres`, no self-loops) in O(N log N) via `scipy.spatial.cKDTree`, matching upstream `pairwise_distances < distance_thres` edge-for-edge without dense O(N^2) memory allocation.
2. **GraphSAGE + Cosine Margin Encoder (`STELLAREncoder`)**:
   Maps single-cell protein/gene expression vectors through `Linear(x_dim, 128) -> ReLU -> SAGEConv(128, 128) -> NormedLinear(128, num_heads)` (cosine-normalized classifier with temperature scale `10.0`).
   - Pure-PyTorch `PureSAGEConv` matches `torch_geometric.nn.SAGEConv` numerically to `< 1e-6` (`0.0` forward diff) when `torch_geometric` is absent (e.g., main `iris` conda env).
   - Optional PyTorch Geometric backend (`backend="pyg"` or `"auto"`) uses `torch_geometric.nn.SAGEConv` and `ClusterData` / `ClusterLoader` (e.g., in `iris_stellar` env).
3. **Joint Transductive Loss**:
   Combines:
   - Pairwise neighborhood similarity BCE loss over cosine-similar cell pairs across reference + target batches.
   - Adaptive `MarginLoss(m=-mean_uncertainty, s=10.0)` cross-entropy over labeled reference cells and seeded novel target clusters.
   - Mean class-distribution entropy regularization (`-0.3 * H(mean(p))`).
4. **Novel Cell-Type Discovery (`num_seed_class > 0`)**:
   Clusters unlabeled target cells via k-NN Louvain/Leiden community detection, trains a linear cosine `FCNet` seed classifier on reference cells, selects the top `num_seed_class` highest-entropy clusters as novel class seeds, and returns `is_novel` (`pd.Series[bool]`) alongside `Novel_1`, ..., `Novel_K` predictions.

## Module Contents

- `pipeline.py`: `STELLARConfig`, `STELLARGraph`, `STELLARPipeline`, `STELLARResult`, `build_spatial_graph`, `partition_graph`.
- `adapter.py`: `STELLARPhenotyper(PhenotyperMethodBase)` (`input_modality="spatial_tabular"`, `requires_training=True`).
- `reproduce_paper.py`: CLI and programmatic reproduction of the Dryad CODEX benchmarks (`Tonsil -> Barrett's Esophagus`, `HuBMAP B004 Cross-Region`, `HuBMAP B004 -> B005 Cross-Donor`) plus PyG equivalence verification.

## Usage

```python
from iris_phenotyper.stellar import STELLARPhenotyper

pheno = STELLARPhenotyper(
    distance_thres=50.0,
    epochs=50,
    num_seed_class=3,
    seed=1,
)
result = pheno.fit_predict(reference_df, target_df)
# result.labels, result.probabilities, result.confidence, result.is_novel
```

Run paper reproduction in `/Users/wwu/.conda/envs/iris_stellar`:

```bash
/Users/wwu/.conda/envs/iris_stellar/bin/python -m iris_phenotyper.stellar.reproduce_paper
```
