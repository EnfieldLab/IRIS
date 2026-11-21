"""
STELLAR Core Training and Inference Pipeline
============================================
Faithful PyTorch implementation of STELLAR (SpaTial cELl LeARning) from
Brbic et al., Nature Methods 2022 (``snap-stanford/stellar``).

Supports both:
- Pure PyTorch backend (``PureSAGEConv`` + spatial KD-bisection partitioning),
  numerically equivalent to PyTorch Geometric's ``SAGEConv`` while requiring no
  external C++ PyG extensions.
- Optional PyTorch Geometric backend (``torch_geometric.nn.SAGEConv`` +
  ``ClusterData`` / ``ClusterLoader``) when ``torch_geometric`` and
  ``torch_sparse`` / ``pyg_lib`` are installed.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from iris_common.logging import iris_log


def _import_torch() -> Any:
    """Import PyTorch safely on macOS environments with multiple OpenMP runtimes."""
    if sys.platform == "darwin":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    if sys.platform == "darwin" and not getattr(torch, "_iris_threads_configured", False):
        try:
            # Avoid OpenMP thread-pool contention when dual libomp runtimes coexist on macOS.
            if int(torch.get_num_threads()) > 1 and "torch_geometric" not in sys.modules:
                torch.set_num_threads(1)
        except Exception:
            pass
        torch._iris_threads_configured = True
    return torch


def is_pyg_available() -> bool:
    """Return True if ``torch_geometric`` is importable."""
    try:
        _import_torch()
        import torch_geometric  # noqa: F401

        return True
    except ImportError:
        return False


def is_pyg_cluster_available() -> bool:
    """Return True if PyG ``ClusterData`` dependencies (``torch_sparse``/``pyg_lib``) exist."""
    if not is_pyg_available():
        return False
    try:
        import torch_sparse  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        import pyg_lib  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_device(device: str = "auto") -> str:
    """Resolve execution device string (``'cuda'``, ``'mps'``, or ``'cpu'``)."""
    torch = _import_torch()
    req = str(device).strip().lower()
    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if req == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if req == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        return "cpu"
    return req


@dataclass
class STELLARConfig:
    """Hyperparameters matching upstream ``snap-stanford/stellar``."""

    distance_thres: float = 50.0
    epochs: int = 20
    seed_epochs: int = 20
    lr: float = 1e-3
    wd: float = 5e-2
    hidden_dim: int = 128
    num_heads: Optional[int] = None
    num_seed_class: int = 0
    num_parts: int = 100
    entropy_weight: float = 0.3
    margin_scale: float = 10.0
    seed: int = 1
    device: str = "auto"
    backend: Literal["auto", "pyg", "pytorch"] = "auto"
    novel_prefix: str = "Novel_"
    clustering_resolution: float = 1.0


@dataclass
class STELLARGraph:
    """In-memory graph representation for STELLAR training and inference."""

    x: np.ndarray
    edge_index: np.ndarray
    coords: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    novel_label_seeds: Optional[np.ndarray] = None


@dataclass
class STELLARResult:
    """Outputs produced by ``STELLARPipeline``."""

    pred_indices: np.ndarray
    pred_labels: np.ndarray
    probabilities: np.ndarray
    confidence: np.ndarray
    is_novel: np.ndarray
    embeddings: np.ndarray
    class_names: List[str]
    known_classes: List[str]
    novel_classes: List[str]
    mean_uncertainty: float
    backend_used: str
    device_used: str
    train_time_sec: float
    history: Dict[str, List[float]] = field(default_factory=dict)


def build_spatial_graph(
    coords: np.ndarray,
    distance_thres: float = 50.0,
    regions: Optional[Sequence[Any]] = None,
) -> np.ndarray:
    """Build directed spatial graph edges for cells closer than ``distance_thres``.

    Matches ``datasets.get_hubmap_edge_index`` and ``datasets.get_tonsilbe_edge_index``
    (strict ``distance < distance_thres``, no self-loops, both ``(u, v)`` and ``(v, u)``),
    computed via ``scipy.spatial.cKDTree`` in O(N log N) memory/time instead of an O(N^2)
    dense distance matrix.
    """
    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"coords must have shape (N, >=2), got {pts.shape}.")
    if float(distance_thres) <= 0.0:
        raise ValueError(f"distance_thres must be positive, got {distance_thres}.")
    n_cells = pts.shape[0]
    if n_cells < 2:
        return np.zeros((2, 0), dtype=np.int64)
    if not np.all(np.isfinite(pts[:, :2])):
        raise ValueError("Spatial coordinates contain NaN or infinite values.")

    pts_2d = pts[:, :2]
    thres = float(distance_thres)
    thres_sq = thres * thres

    if regions is None:
        region_codes = np.zeros(n_cells, dtype=np.int64)
        n_regions = 1
    else:
        reg_arr = np.asarray(list(regions), dtype=object)
        if len(reg_arr) != n_cells:
            raise ValueError(
                f"regions length ({len(reg_arr)}) does not match coords ({n_cells})."
            )
        codes, uniques = pd.factorize(reg_arr, sort=False)
        region_codes = codes
        n_regions = len(uniques)

    src_chunks: List[np.ndarray] = []
    dst_chunks: List[np.ndarray] = []
    for reg_id in range(n_regions):
        locs = np.flatnonzero(region_codes == reg_id)
        if len(locs) < 2:
            continue
        pos_reg = pts_2d[locs]
        tree = cKDTree(pos_reg)
        pairs = tree.query_pairs(r=thres, output_type="ndarray")
        if len(pairs) == 0:
            continue
        diff = pos_reg[pairs[:, 0]] - pos_reg[pairs[:, 1]]
        dist_sq = np.einsum("ij,ij->i", diff, diff)
        pairs = pairs[dist_sq < thres_sq]
        if len(pairs) == 0:
            continue
        u = locs[pairs[:, 0]]
        v = locs[pairs[:, 1]]
        src_chunks.append(np.concatenate([u, v]))
        dst_chunks.append(np.concatenate([v, u]))

    if not src_chunks:
        return np.zeros((2, 0), dtype=np.int64)
    edge_index = np.vstack([np.concatenate(src_chunks), np.concatenate(dst_chunks)])
    return edge_index.astype(np.int64, copy=False)


def _partition_indices_kd(coords: np.ndarray, num_parts: int) -> List[np.ndarray]:
    """Recursively bisect node indices along alternating spatial medians."""
    n_nodes = len(coords)
    if num_parts <= 1 or n_nodes <= 1:
        return [np.arange(n_nodes, dtype=np.int64)]

    queue: List[Tuple[np.ndarray, int]] = [(np.arange(n_nodes, dtype=np.int64), 0)]
    while len(queue) < num_parts:
        # Split the largest remaining chunk along alternating spatial axes.
        largest_idx = max(range(len(queue)), key=lambda i: len(queue[i][0]))
        idxs, depth = queue[largest_idx]
        if len(idxs) < 2:
            break
        queue.pop(largest_idx)
        axis = depth % coords.shape[1]
        order = np.argsort(coords[idxs, axis], kind="mergesort")
        mid = len(order) // 2
        queue.append((idxs[order[:mid]], depth + 1))
        queue.append((idxs[order[mid:]], depth + 1))

    return [np.sort(part) for part, _ in queue if len(part) > 0]


def partition_graph(
    graph: STELLARGraph,
    num_parts: int,
    seed: int = 1,
) -> List[STELLARGraph]:
    """Partition a graph into ``num_parts`` induced subgraphs without PyG C++ extensions."""
    n_nodes = graph.x.shape[0]
    effective_parts = max(1, min(int(num_parts), max(1, n_nodes // 4)))
    if effective_parts <= 1:
        return [graph]

    if graph.coords is not None and graph.coords.shape[0] == n_nodes:
        parts = _partition_indices_kd(graph.coords[:, :2], effective_parts)
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_nodes)
        parts = [np.sort(chunk) for chunk in np.array_split(perm, effective_parts) if len(chunk) > 0]

    src = graph.edge_index[0] if graph.edge_index.shape[1] > 0 else np.empty(0, dtype=np.int64)
    dst = graph.edge_index[1] if graph.edge_index.shape[1] > 0 else np.empty(0, dtype=np.int64)
    remap = np.full(n_nodes, -1, dtype=np.int64)

    subgraphs: List[STELLARGraph] = []
    for part_nodes in parts:
        remap.fill(-1)
        remap[part_nodes] = np.arange(len(part_nodes), dtype=np.int64)
        if src.size > 0:
            mask = (remap[src] >= 0) & (remap[dst] >= 0)
            sub_edges = np.vstack([remap[src[mask]], remap[dst[mask]]]).astype(np.int64)
        else:
            sub_edges = np.zeros((2, 0), dtype=np.int64)

        subgraphs.append(
            STELLARGraph(
                x=graph.x[part_nodes],
                edge_index=sub_edges,
                coords=graph.coords[part_nodes] if graph.coords is not None else None,
                y=graph.y[part_nodes] if graph.y is not None else None,
                novel_label_seeds=(
                    graph.novel_label_seeds[part_nodes]
                    if graph.novel_label_seeds is not None
                    else None
                ),
            )
        )
    return subgraphs


def _build_model_classes() -> Dict[str, Any]:
    """Construct PyTorch neural-network classes lazily via ``_import_torch()``."""
    torch = _import_torch()
    import torch.nn as nn
    import torch.nn.functional as F

    class NormedLinear(nn.Module):
        """Cosine similarity classifier with temperature scale 10."""

        def __init__(self, in_features: int, out_features: int, scale: float = 10.0) -> None:
            super().__init__()
            self.scale = float(scale)
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
            self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        def forward(self, x: Any) -> Any:
            out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
            return self.scale * out

    class PureSAGEConv(nn.Module):
        """Pure-PyTorch GraphSAGE mean-aggregation layer matching PyG ``SAGEConv``."""

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.lin_l = nn.Linear(in_channels, out_channels, bias=True)
            self.lin_r = nn.Linear(in_channels, out_channels, bias=False)

        def forward(self, x: Any, edge_index: Any) -> Any:
            aggr = torch.zeros_like(x)
            if edge_index is not None and edge_index.numel() > 0:
                src, dst = edge_index[0], edge_index[1]
                aggr.index_add_(0, dst, x[src])
                deg = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
                deg.index_add_(0, dst, torch.ones(dst.size(0), dtype=x.dtype, device=x.device))
                aggr = aggr / deg.clamp(min=1.0).unsqueeze(1)
            return self.lin_l(aggr) + self.lin_r(x)

    class STELLAREncoder(nn.Module):
        """STELLAR 2-layer GraphSAGE + NormedLinear encoder (``models/net.py:Encoder``)."""

        def __init__(
            self,
            x_dim: int,
            num_cls: int,
            hidden_dim: int = 128,
            use_pyg: bool = False,
            margin_scale: float = 10.0,
        ) -> None:
            super().__init__()
            self.x_dim = int(x_dim)
            self.hidden_dim = int(hidden_dim)
            self.use_pyg = bool(use_pyg)
            self.conv1 = nn.Linear(self.x_dim, self.hidden_dim)
            if self.use_pyg:
                from torch_geometric.nn import SAGEConv

                self.conv2 = SAGEConv(self.hidden_dim, self.hidden_dim)
            else:
                self.conv2 = PureSAGEConv(self.hidden_dim, self.hidden_dim)
            self.relu = nn.ReLU()
            self.linear = NormedLinear(self.hidden_dim, int(num_cls), scale=margin_scale)

        def forward(self, data_or_x: Any, edge_index: Any = None) -> Tuple[Any, Any, Any]:
            if edge_index is None and hasattr(data_or_x, "x") and hasattr(data_or_x, "edge_index"):
                x = data_or_x.x
                edge_idx = data_or_x.edge_index
            else:
                x = data_or_x
                edge_idx = edge_index
            h = self.conv1(x)
            feat = h
            h = self.relu(h)
            h = self.conv2(h, edge_idx)
            out_feat = h
            out = self.linear(h)
            return out, feat, out_feat

    class FCNet(nn.Module):
        """Linear cosine classifier used for novel-class seed entropy estimation."""

        def __init__(self, x_dim: int, num_cls: int, margin_scale: float = 10.0) -> None:
            super().__init__()
            self.linear = NormedLinear(int(x_dim), int(num_cls), scale=margin_scale)

        def forward(self, data_or_x: Any, edge_index: Any = None) -> Tuple[Any, Any, Any]:
            x = data_or_x.x if hasattr(data_or_x, "x") else data_or_x
            out = self.linear(x)
            return out, x, x

    class MarginLoss(nn.Module):
        """Adaptive margin cross-entropy loss from ``snap-stanford/stellar/utils.py``."""

        def __init__(
            self,
            m: float = 0.2,
            weight: Optional[Any] = None,
            s: float = 10.0,
        ) -> None:
            super().__init__()
            self.m = float(m)
            self.s = float(s)
            self.weight = weight

        def forward(self, x: Any, target: Any) -> Any:
            index = torch.zeros_like(x, dtype=torch.bool)
            index.scatter_(1, target.data.view(-1, 1), 1)
            x_m = x - self.m * self.s
            output = torch.where(index, x_m, x)
            return F.cross_entropy(output, target, weight=self.weight)

    return {
        "NormedLinear": NormedLinear,
        "PureSAGEConv": PureSAGEConv,
        "STELLAREncoder": STELLAREncoder,
        "FCNet": FCNet,
        "MarginLoss": MarginLoss,
    }


def _entropy_loss(probs: Any) -> Any:
    """Compute Shannon entropy matching ``snap-stanford/stellar/utils.py:entropy``."""
    torch = _import_torch()
    eps = 1e-8
    clamped = torch.clamp(probs, min=eps)
    term = clamped * torch.log(clamped)
    if term.dim() == 2:
        return -term.sum(dim=1).mean()
    if term.dim() == 1:
        return -term.sum()
    raise ValueError(f"Expected 1D or 2D tensor for entropy, got {term.dim()}D.")


def _cluster_unlabeled_cells(
    unlabeled_x: np.ndarray,
    resolution: float = 1.0,
    seed: int = 1,
) -> np.ndarray:
    """Cluster unlabeled cells for novel-class seed estimation.

    Builds a k-NN graph compatible with SciPy >=1.15 / Scanpy >=1.11 (avoiding
    ``csr_array.getnnz`` incompatibilities) and clusters via Louvain (``scanpy``
    / ``louvain``, matching upstream ``STELLAR.py``), Leiden (``igraph``), or
    ``MiniBatchKMeans`` as a final fallback.
    """
    n_cells = unlabeled_x.shape[0]
    if n_cells < 3:
        return np.zeros(n_cells, dtype=np.int64)

    x_f32 = np.asarray(unlabeled_x, dtype=np.float32)
    n_neighbors = min(15, max(2, n_cells - 1))

    try:
        import anndata
        import scanpy as sc
        from scanpy.neighbors._connectivity import umap as umap_connectivity
        from scipy.sparse import csr_matrix
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
        nn.fit(x_f32)
        dists, idxs = nn.kneighbors(x_f32)
        conn = csr_matrix(
            umap_connectivity(idxs, dists, n_obs=n_cells, n_neighbors=n_neighbors)
        )
        n_extra = max(1, n_neighbors - 1)
        dist_mat = csr_matrix(
            (
                dists[:, 1:].ravel(),
                (np.repeat(np.arange(n_cells), n_extra), idxs[:, 1:].ravel()),
            ),
            shape=(n_cells, n_cells),
        )
        adata = anndata.AnnData(x_f32)
        adata.uns["neighbors"] = {
            "connectivities_key": "connectivities",
            "distances_key": "distances",
            "params": {"n_neighbors": n_neighbors, "method": "umap", "metric": "euclidean"},
        }
        adata.obsp["connectivities"] = conn
        adata.obsp["distances"] = dist_mat

        try:
            sc.tl.louvain(adata, resolution=resolution, random_state=seed)
            return adata.obs["louvain"].astype(int).to_numpy()
        except Exception:
            sc.tl.leiden(
                adata,
                resolution=resolution,
                flavor="igraph",
                n_iterations=2,
                random_state=seed,
            )
            return adata.obs["leiden"].astype(int).to_numpy()
    except Exception:
        from sklearn.cluster import MiniBatchKMeans

        n_clusters = min(max(2, int(np.sqrt(n_cells / 2))), 20, n_cells)
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
        return km.fit_predict(x_f32).astype(np.int64)


class STELLARPipeline:
    """End-to-end STELLAR training and spatial cell-type annotation pipeline."""

    def __init__(self, config: Optional[STELLARConfig] = None) -> None:
        self.config = config or STELLARConfig()
        self.model: Any = None
        self.known_classes: List[str] = []
        self.novel_classes: List[str] = []
        self.class_names: List[str] = []
        self.backend_used: str = "pytorch"
        self.device_used: str = "cpu"
        self.input_dim: int = 0
        self.num_heads: int = 0

    def _select_backend(self) -> str:
        req = self.config.backend
        if req == "pytorch":
            return "pytorch"
        if req == "pyg":
            if not is_pyg_available():
                raise ImportError(
                    "backend='pyg' requested, but torch_geometric is not installed."
                )
            return "pyg"
        return "pyg" if is_pyg_cluster_available() else "pytorch"

    def _build_subgraphs_loader(
        self,
        graph: STELLARGraph,
        num_parts: int,
        use_pyg_loader: bool,
        seed: int,
    ) -> List[Tuple[Any, Any, Any, Any]]:
        """Create per-epoch graph partition batches ``(x, edge_index, y, novel_seeds)``."""
        torch = _import_torch()
        n_nodes = graph.x.shape[0]
        eff_parts = max(1, min(int(num_parts), max(1, n_nodes // 4)))

        if use_pyg_loader and eff_parts > 1:
            from torch_geometric.data import Data
            from torch_geometric.loader import ClusterData, ClusterLoader

            data = Data(
                x=torch.as_tensor(graph.x, dtype=torch.float32),
                edge_index=torch.as_tensor(graph.edge_index, dtype=torch.long),
                y=(
                    torch.as_tensor(graph.y, dtype=torch.long)
                    if graph.y is not None
                    else torch.zeros(n_nodes, dtype=torch.long)
                ),
                novel_label_seeds=(
                    torch.as_tensor(graph.novel_label_seeds, dtype=torch.long)
                    if graph.novel_label_seeds is not None
                    else torch.zeros(n_nodes, dtype=torch.long)
                ),
            )
            cluster_data = ClusterData(data, num_parts=eff_parts, recursive=False)
            loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True, num_workers=0)
            batches = []
            for part in loader:
                if part.x.size(0) == 0:
                    continue
                batches.append((part.x, part.edge_index, part.y, part.novel_label_seeds))
            if batches:
                return batches

        parts = partition_graph(graph, eff_parts, seed=seed)
        return [
            (
                torch.as_tensor(p.x, dtype=torch.float32),
                torch.as_tensor(p.edge_index, dtype=torch.long),
                (
                    torch.as_tensor(p.y, dtype=torch.long)
                    if p.y is not None
                    else torch.zeros(p.x.shape[0], dtype=torch.long)
                ),
                (
                    torch.as_tensor(p.novel_label_seeds, dtype=torch.long)
                    if p.novel_label_seeds is not None
                    else torch.zeros(p.x.shape[0], dtype=torch.long)
                ),
            )
            for p in parts
            if p.x.shape[0] > 0
        ]

    def _estimate_novel_seeds(
        self,
        labeled_graph: STELLARGraph,
        unlabeled_graph: STELLARGraph,
        labeled_batches: List[Tuple[Any, Any, Any, Any]],
        classes_dict: Dict[str, Any],
        device: Any,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, List[float]]:
        """Estimate high-entropy Louvain clusters in unlabeled data as novel seeds."""
        torch = _import_torch()
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim

        n_unlabeled = unlabeled_graph.x.shape[0]
        if self.config.num_seed_class <= 0:
            return np.zeros(n_unlabeled, dtype=np.int64), []

        clusters = _cluster_unlabeled_cells(
            unlabeled_graph.x,
            resolution=self.config.clustering_resolution,
            seed=self.config.seed,
        )
        num_known = int(np.max(labeled_graph.y)) + 1
        seed_model = classes_dict["FCNet"](
            self.input_dim,
            num_known,
            margin_scale=self.config.margin_scale,
        ).to(device)
        seed_opt = optim.Adam(seed_model.parameters(), lr=1e-3, weight_decay=5e-2)
        ce_loss_fn = nn.CrossEntropyLoss()

        seed_losses: List[float] = []
        n_batches = max(1, len(labeled_batches))
        for _ in range(self.config.seed_epochs):
            seed_model.train()
            order = rng.permutation(len(labeled_batches))
            epoch_sum = 0.0
            for idx in order:
                bx, _, by, _ = labeled_batches[int(idx)]
                bx = bx.to(device)
                by = by.to(device)
                seed_opt.zero_grad()
                out, _, _ = seed_model(bx)
                loss = ce_loss_fn(out, by)
                loss.backward()
                seed_opt.step()
                epoch_sum += float(loss.item())
            seed_losses.append(epoch_sum / n_batches)

        seed_model.eval()
        with torch.no_grad():
            ux = torch.as_tensor(unlabeled_graph.x, dtype=torch.float32, device=device)
            u_out, _, _ = seed_model(ux)
            prob = F.softmax(u_out, dim=1)
            entr = -torch.sum(prob * torch.log(prob.clamp(min=1e-12)), dim=1).cpu().numpy()

        unique_clusters = np.unique(clusters)
        cluster_entrs = np.array(
            [float(np.mean(entr[clusters == cid])) for cid in unique_clusters],
            dtype=np.float64,
        )
        top_k = min(int(self.config.num_seed_class), len(unique_clusters))
        top_order = np.argsort(cluster_entrs)[-top_k:]
        novel_cluster_ids = unique_clusters[top_order]

        novel_seeds = np.zeros(n_unlabeled, dtype=np.int64)
        largest_seen_id = int(np.max(labeled_graph.y))
        for i, cid in enumerate(novel_cluster_ids):
            novel_seeds[clusters == cid] = largest_seen_id + i + 1
        return novel_seeds, seed_losses

    def fit_predict(
        self,
        labeled_x: np.ndarray,
        labeled_y: Sequence[Any],
        unlabeled_x: np.ndarray,
        labeled_coords: Optional[np.ndarray] = None,
        unlabeled_coords: Optional[np.ndarray] = None,
        labeled_regions: Optional[Sequence[Any]] = None,
        unlabeled_regions: Optional[Sequence[Any]] = None,
        labeled_edges: Optional[np.ndarray] = None,
        unlabeled_edges: Optional[np.ndarray] = None,
    ) -> STELLARResult:
        """Train STELLAR jointly on reference + target graphs and predict target cells."""
        t0 = time.perf_counter()
        torch = _import_torch()
        import torch.nn.functional as F
        import torch.optim as optim

        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        np.random.seed(self.config.seed)
        rng = np.random.default_rng(self.config.seed)

        lx = np.asarray(labeled_x, dtype=np.float32)
        ux = np.asarray(unlabeled_x, dtype=np.float32)
        if lx.ndim != 2 or ux.ndim != 2 or lx.shape[1] != ux.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: labeled {lx.shape} vs unlabeled {ux.shape}."
            )
        if lx.shape[0] == 0 or ux.shape[0] == 0:
            raise ValueError("Both labeled and unlabeled datasets must be non-empty.")

        raw_y = [str(v) for v in labeled_y]
        if len(raw_y) != lx.shape[0]:
            raise ValueError(
                f"labeled_y length ({len(raw_y)}) != labeled_x rows ({lx.shape[0]})."
            )

        known_classes = sorted(set(raw_y))
        class_to_id = {name: i for i, name in enumerate(known_classes)}
        ly_int = np.array([class_to_id[v] for v in raw_y], dtype=np.int64)

        if labeled_edges is None:
            if labeled_coords is None:
                raise ValueError("Provide either labeled_coords or labeled_edges.")
            l_edges = build_spatial_graph(
                labeled_coords, self.config.distance_thres, regions=labeled_regions
            )
        else:
            l_edges = np.asarray(labeled_edges, dtype=np.int64)

        if unlabeled_edges is None:
            if unlabeled_coords is None:
                raise ValueError("Provide either unlabeled_coords or unlabeled_edges.")
            u_edges = build_spatial_graph(
                unlabeled_coords, self.config.distance_thres, regions=unlabeled_regions
            )
        else:
            u_edges = np.asarray(unlabeled_edges, dtype=np.int64)

        num_known = len(known_classes)
        min_heads = num_known + max(0, int(self.config.num_seed_class))
        num_heads = int(self.config.num_heads) if self.config.num_heads is not None else min_heads
        if num_heads < min_heads:
            raise ValueError(
                f"num_heads ({num_heads}) must be >= known classes + num_seed_class ({min_heads})."
            )

        self.known_classes = known_classes
        n_novel_heads = num_heads - num_known
        self.novel_classes = [
            f"{self.config.novel_prefix}{i + 1}" for i in range(n_novel_heads)
        ]
        self.class_names = self.known_classes + self.novel_classes
        self.input_dim = lx.shape[1]
        self.num_heads = num_heads

        backend = self._select_backend()
        self.backend_used = backend
        self.device_used = resolve_device(self.config.device)
        device = torch.device(self.device_used)

        use_pyg_layer = backend == "pyg"
        use_pyg_cluster = backend == "pyg" and is_pyg_cluster_available()

        labeled_graph = STELLARGraph(
            x=lx,
            edge_index=l_edges,
            coords=np.asarray(labeled_coords, dtype=np.float64)
            if labeled_coords is not None
            else None,
            y=ly_int,
        )
        unlabeled_graph = STELLARGraph(
            x=ux,
            edge_index=u_edges,
            coords=np.asarray(unlabeled_coords, dtype=np.float64)
            if unlabeled_coords is not None
            else None,
        )

        classes_dict = _build_model_classes()
        self.model = classes_dict["STELLAREncoder"](
            self.input_dim,
            num_heads,
            hidden_dim=self.config.hidden_dim,
            use_pyg=use_pyg_layer,
            margin_scale=self.config.margin_scale,
        ).to(device)

        labeled_batches = self._build_subgraphs_loader(
            labeled_graph, self.config.num_parts, use_pyg_cluster, seed=self.config.seed
        )
        novel_seeds, seed_losses = self._estimate_novel_seeds(
            labeled_graph, unlabeled_graph, labeled_batches, classes_dict, device, rng
        )
        unlabeled_graph.novel_label_seeds = novel_seeds

        unlabeled_batches = self._build_subgraphs_loader(
            unlabeled_graph, self.config.num_parts, use_pyg_cluster, seed=self.config.seed + 17
        )

        optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.wd
        )
        MarginLoss = classes_dict["MarginLoss"]

        ux_full = torch.as_tensor(unlabeled_graph.x, dtype=torch.float32, device=device)
        u_edges_full = torch.as_tensor(
            unlabeled_graph.edge_index, dtype=torch.long, device=device
        )

        epoch_losses: List[float] = []
        mean_uncertainties: List[float] = []

        for _ in range(self.config.epochs):
            # Evaluate current mean uncertainty on unlabeled data for margin schedule.
            self.model.eval()
            with torch.no_grad():
                u_out_eval, _, _ = self.model(ux_full, u_edges_full)
                u_prob_eval = F.softmax(u_out_eval, dim=1)
                confs_eval, _ = u_prob_eval.max(dim=1)
                mean_uncert = float(1.0 - confs_eval.mean().item())
            mean_uncertainties.append(mean_uncert)

            self.model.train()
            margin_ce = MarginLoss(m=-mean_uncert, s=self.config.margin_scale)

            l_order = rng.permutation(len(labeled_batches))
            u_order = rng.permutation(len(unlabeled_batches))
            sum_loss = 0.0

            for step, l_idx in enumerate(l_order):
                u_idx = u_order[step % len(u_order)]
                lx_b, l_ei_b, ly_b, _ = labeled_batches[int(l_idx)]
                ux_b, u_ei_b, _, u_seeds_b = unlabeled_batches[int(u_idx)]

                unlabeled_ce_idx = torch.where(u_seeds_b > 0)[0]

                lx_b = lx_b.to(device)
                l_ei_b = l_ei_b.to(device)
                ly_b = ly_b.to(device)
                ux_b = ux_b.to(device)
                u_ei_b = u_ei_b.to(device)
                u_seeds_b = u_seeds_b.to(device)
                unlabeled_ce_idx = unlabeled_ce_idx.to(device)

                optimizer.zero_grad()
                l_out, l_feat, _ = self.model(lx_b, l_ei_b)
                u_out, u_feat, _ = self.model(ux_b, u_ei_b)

                labeled_len = l_out.size(0)
                unlabeled_len = u_out.size(0)
                batch_size = labeled_len + unlabeled_len

                output = torch.cat([l_out, u_out], dim=0)
                feat = torch.cat([l_feat, u_feat], dim=0)
                prob = F.softmax(output, dim=1)

                feat_norm = F.normalize(feat.detach(), p=2, dim=1)
                cosine_dist = torch.mm(feat_norm, feat_norm.t())

                pos_pairs: List[int] = []
                target_np = ly_b.detach().cpu().numpy()
                class_to_locs: Dict[int, np.ndarray] = {}
                for cls_val in np.unique(target_np):
                    class_to_locs[int(cls_val)] = np.flatnonzero(target_np == cls_val)

                for i in range(labeled_len):
                    locs = class_to_locs[int(target_np[i])]
                    if len(locs) == 1:
                        pos_pairs.append(int(locs[0]))
                    else:
                        pick = int(rng.choice(locs))
                        while pick == i:
                            pick = int(rng.choice(locs))
                        pos_pairs.append(pick)

                if unlabeled_len > 0:
                    if batch_size >= 2:
                        unlabel_cos = cosine_dist[labeled_len:, :]
                        _, top2 = torch.topk(unlabel_cos, 2, dim=1)
                        self_idx = torch.arange(
                            labeled_len, batch_size, device=device, dtype=top2.dtype
                        )
                        nearest = torch.where(top2[:, 0] == self_idx, top2[:, 1], top2[:, 0])
                        pos_pairs.extend(nearest.cpu().tolist())
                    else:
                        pos_pairs.append(0)

                pos_idx_t = torch.tensor(pos_pairs, dtype=torch.long, device=device)
                pos_prob = prob[pos_idx_t]
                pos_sim = torch.sum(prob * pos_prob, dim=1).clamp(1e-7, 1.0 - 1e-7)
                bce_loss = F.binary_cross_entropy(pos_sim, torch.ones_like(pos_sim))

                ce_idx = torch.cat(
                    [
                        torch.arange(labeled_len, device=device, dtype=torch.long),
                        labeled_len + unlabeled_ce_idx,
                    ]
                )
                all_targets = torch.cat([ly_b, u_seeds_b], dim=0)
                ce_loss = margin_ce(output[ce_idx], all_targets[ce_idx])
                ent_loss = _entropy_loss(torch.mean(prob, dim=0))

                loss = bce_loss + ce_loss - self.config.entropy_weight * ent_loss
                loss.backward()
                optimizer.step()
                sum_loss += float(loss.item())

            epoch_losses.append(sum_loss / max(1, len(l_order)))

        train_time_sec = time.perf_counter() - t0
        result = self.predict_graph(unlabeled_x=ux, unlabeled_edges=u_edges)
        result.train_time_sec = train_time_sec
        result.history = {
            "epoch_losses": epoch_losses,
            "mean_uncertainties": mean_uncertainties,
            "seed_losses": seed_losses,
        }
        iris_log(
            "IRIS-Phenotyper",
            "STELLAR",
            f"Completed graph transfer ({len(lx):,} ref -> {len(ux):,} target cells, "
            f"backend={self.backend_used}, device={self.device_used}, {train_time_sec:.2f}s).",
        )
        return result

    def predict_graph(
        self,
        unlabeled_x: np.ndarray,
        unlabeled_coords: Optional[np.ndarray] = None,
        unlabeled_regions: Optional[Sequence[Any]] = None,
        unlabeled_edges: Optional[np.ndarray] = None,
    ) -> STELLARResult:
        """Run forward inference on an unlabeled spatial graph using a fitted encoder."""
        if self.model is None:
            raise RuntimeError("STELLARPipeline model is not fitted yet.")
        torch = _import_torch()
        import torch.nn.functional as F

        ux = np.asarray(unlabeled_x, dtype=np.float32)
        if ux.ndim != 2 or ux.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected unlabeled_x with {self.input_dim} features, got {ux.shape}."
            )
        if unlabeled_edges is None:
            if unlabeled_coords is None:
                raise ValueError("Provide either unlabeled_coords or unlabeled_edges.")
            u_edges = build_spatial_graph(
                unlabeled_coords, self.config.distance_thres, regions=unlabeled_regions
            )
        else:
            u_edges = np.asarray(unlabeled_edges, dtype=np.int64)

        device = torch.device(self.device_used)
        self.model.eval()
        with torch.no_grad():
            ux_t = torch.as_tensor(ux, dtype=torch.float32, device=device)
            ei_t = torch.as_tensor(u_edges, dtype=torch.long, device=device)
            logits, _, embeddings_t = self.model(ux_t, ei_t)
            probs_t = F.softmax(logits, dim=1)
            conf_t, pred_t = probs_t.max(dim=1)

        probs = np.clip(probs_t.cpu().numpy().astype(np.float64), 0.0, 1.0)
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = np.divide(probs, row_sums, out=np.zeros_like(probs), where=row_sums > 0.0)
        pred_indices = pred_t.cpu().numpy().astype(np.int64)
        confidence = np.clip(conf_t.cpu().numpy().astype(np.float64), 0.0, 1.0)
        embeddings = embeddings_t.cpu().numpy().astype(np.float32)

        num_known = len(self.known_classes)
        is_novel = pred_indices >= num_known
        pred_labels = np.array(
            [self.class_names[int(idx)] for idx in pred_indices],
            dtype=object,
        )
        mean_uncert = float(1.0 - np.mean(confidence)) if len(confidence) > 0 else 0.0

        return STELLARResult(
            pred_indices=pred_indices,
            pred_labels=pred_labels,
            probabilities=probs,
            confidence=confidence,
            is_novel=is_novel,
            embeddings=embeddings,
            class_names=list(self.class_names),
            known_classes=list(self.known_classes),
            novel_classes=list(self.novel_classes),
            mean_uncertainty=mean_uncert,
            backend_used=self.backend_used,
            device_used=self.device_used,
            train_time_sec=0.0,
        )


__all__ = [
    "STELLARConfig",
    "STELLARGraph",
    "STELLARPipeline",
    "STELLARResult",
    "build_spatial_graph",
    "is_pyg_available",
    "is_pyg_cluster_available",
    "partition_graph",
    "resolve_device",
]
