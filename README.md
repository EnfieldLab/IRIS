# IRIS: Interactive Research viewer for Integrated Spatial omics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-56%20passed-brightgreen.svg)](tests/)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Windows-lightgrey.svg)]()
[![Framework: napari](https://img.shields.io/badge/napari-plugin-blueviolet.svg)](https://napari.org/)

**IRIS** is an interactive, high-performance spatial biology analytics tool designed for multiplexed tissue imaging. IRIS bridges deep learning segmentation, interactive curation, cell phenotyping, and spatial analysis within a single, unified graphical user interface.

<p align="center">
  <img src="image/iris_blink.gif" alt="IRIS Logo" width="700">
</p>

---

## Key Methodological Capabilities

### 1. Multi-Model Consensus Segmentation & Confidence Mapping
- **Deep Learning Segmentmentation**: Modular execution pipeline supporting Cellpose, StarDist, and pretrained deep learning backends.
- **Ensemble Consensus**: Probability-weighted consensus masks generated across multiple segmentation architectures to mitigate model-specific boundary artifacts.
- **Pixel-Level Uncertainty**: Per-cell and spatial confidence scoring highlighting ambiguous boundaries and over-segmented regions.

### 2. Interactive Review & Fine Tuning
- **Active Visual Audit**: Real-time inspection of segmented cell boundaries superimposed over multi-channel fluorescence signals.
- **Human-in-the-Loop Curation**: Direct split, merge, delete, and manual polygon boundary refinement on full-resolution imagery.
- **Dynamic Channel Management**: Live channel visibility, contrast adjustment, and synchronized multi-sample camera locking.

### 3. Hierarchical Single-Cell Phenotyping
- **Manual Gating Studio**: 2D biaxial scatter plotting with polygon lasso selection, quadrant breakdown, and real-time parent/cohort population statistics.
- **Probabilistic Annotation (ASTIR)**: Integration with automated probabilistic marker-to-cell-type association maps.
- **Microenvironment-Aware Phenotyping (MAPS)**: Spatial graph neural network modelling spatial context and neighbouring cell-type distributions.
- **Unsupervised Clustering**: High-throughput Leiden, PhenoGraph, and k-means clustering with integrated expression heatmaps.

### 4. Spatial Neighbourhoods & Microenvironment Analytics
- **Spatial Interaction Graphs**: k-nearest neighbours ($k$-NN), Delaunay triangulation, and fixed-radius spatial network construction.
- **Cellular Neighbourhood (CN) Clustering**: Unsupervised discovery of recurring cellular niches and tissue microenvironments.
- **Marker Expression Profiling**: Dynamic hierarchical clustering, dendrogram ordering, and publication-ready expression heatmaps.

### 5. Multi-Sample & Cohort Management
- **Tissue Microarray (TMA) Navigation**: Automated sample discovery, multi-core spatial layout, and synchronized cohort inspection.
- **Multi-Format I/O**: Direct import and export of OME-TIFF, pyramidal TIFF, CSV, TSV, Apache Parquet, Excel, and AnnData (`.h5ad`) tables.

### 6. Interactive Publication Figure & ROI Export
- **Canvas ROI Bounding Box**: Interactive click-and-drag tool to define square or rectangular regions directly on the viewing canvas with live resize and reposition handles.
- **Calibrated Publication Scale Bars**: Publication-standard scale bars (10, 20, 50, 100, 200 µm) burned into figures with subpixel precision, high-contrast boundaries, and customizable typography.
- **Journal-Grade DPI Metadata**: Native 300 DPI, 600 DPI, and super-sampled detail scaling with embedded DPI tags in lossless PNG, publication TIFF (LZW), vector PDF, and JPEG formats.
- **Live Preview & Clipboard Copy**: Interactive real-time preview capturing all active viewing canvas layers (fluorescence, masks, phenotyping points) and one-click copy to clipboard.

---

## Architectural Overview

The repository is organized as a unified, high-performance spatial biology framework centred in `src/iris_viewer/`:

```
src/iris_viewer/
├── app/                       # Application layer and service coordinators
│   ├── controller.py          # Thin MVC coordinator docking widgets and binding signals
│   └── services/              # Domain services (Image, Multi-Sample, Review, Batch, Curation)
├── core/                      # Application state and session persistence models
├── engine/                    # High-throughput computational algorithms
│   ├── consensus.py           # Multi-model consensus ensemble and agreement mapping
│   ├── feature_extraction.py  # Per-cell morphology and multi-channel intensity extraction
│   ├── pipeline.py            # Asynchronous segmentation pipeline execution engine
│   ├── geometry.py            # Convex hull and tissue boundary delineation
│   ├── neighbourhood.py        # Spatial interaction graphs and cellular niche clustering
│   ├── phenotyping/           # ASTIR probabilistic mapping, MAPS graph neural networks, ensemble
│   └── segmentation/          # Deep learning wrappers (Cellpose, StarDist, CellSAM, InstanSeg)
├── io/                        # Multi-format data adapters (OME-TIFF, CSV, Parquet, AnnData)
└── widgets/                   # Interactive docking panels, pop-up studios, and on-canvas overlays
```

---

## Installation

### Prerequisites
- Python >= 3.10
- Conda or Mamba
- Hardware acceleration (optional): Apple Silicon (Metal/MPS), NVIDIA GPU (CUDA >= 11.8), or multi-core CPU

### Option A: Conda / Mamba (Recommended)
```bash
git clone https://github.com/EnfieldLab/IRIS.git
cd IRIS

# Create environment with all dependencies
conda env create -f environment.yml
conda activate iris
```

### Option B: Pip / Virtual Environment
```bash
git clone https://github.com/EnfieldLab/IRIS.git
cd IRIS

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Or install with all deep learning extras
pip install -e ".[all]"
```

### Optional Dependency Bundles
| Bundle | Command | Included Tools |
| :--- | :--- | :--- |
| **Core** | `pip install -e .` | napari, numpy, scipy, pandas, scikit-image, scanpy |
| **Segmentation** | `pip install -e ".[cellpose,stardist]"` | Cellpose 3.0+, StarDist, CSBDeep |
| **Phenotyping** | `pip install -e ".[astir,maps]"` | ASTIR, PyTorch, PyG (Graph Neural Networks) |
| **All Features** | `pip install -e ".[all]"` | Full analytical and machine learning stack |
| **Development** | `pip install -e ".[dev]"` | pytest, pytest-qt, mypy, ruff |

---

## Quickstart

### Launching IRIS-Viewer
Launch the unified interactive desktop environment directly from the terminal:
```bash
iris-viewer
```

---

## End-to-End Workflow

1. **Load Image Data**:
   - In the **Control** panel, click **Select Image Directory…** to load multi-channel TIFFs or multi-core TMA folders.
   - Adjust channel colours, contrast limits, and physical scale ($\mu\text{m/pixel}$).
2. **Execute Segmentation & Consensus**:
   - In the **Segmentation** tab, select target channels (e.g., nuclear DAPI, membrane PanCK/CD45).
   - Run individual models or compute an ensemble **Consensus Mask** with user-defined overlap thresholds.
3. **Interactive Review**:
   - Toggle **Interactive Review** to inspect low-confidence cells flagged by the consensus scoring engine.
   - Refine masks using split/merge brushes or delete non-cellular debris.
4. **Phenotyping & Gating**:
   - Navigate to the **Phenotyper** panel.
   - Select markers for 2D biaxial gating (e.g., CD3 vs. CD8) or configure automated ASTIR/MAPS models.
   - Assign phenotypes and preview population fractions across active samples or the entire cohort.
5. **Spatial Analysis Analytics**:
   - In the **Neighbourhoods** tab, specify the spatial radius or $k$-nearest neighbors.
   - Compute cellular neighbourhoods to discover spatial niches and microenvironment compositions.
6. **Export Results**:
   - Export consensus label masks (`.ome.tif`, `.tiff`), feature tables (`.csv`, `.parquet`), and high-resolution figures.

---

## Automated Test Suite

Run the comprehensive test suite to verify backend detection, pipeline coordination, state serialization, and phenotyping algorithms:

```bash
pytest tests/ -v
```
