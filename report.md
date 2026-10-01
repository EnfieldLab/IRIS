# IRIS-Viewer Architectural Audit Report

## 1. Architecture

### Current Architecture
The current application uses a "God Class" pattern centered around `ViewerController` in `app/controller.py` (1,006 lines). This class is tightly coupled: it directly manages Napari viewer state, instantiates PyQt/Napari signals, orchestrates background thread workers, stores segmentation results in `ViewerAppState`, and spawns QMessageboxes. 

The data flow is highly monolithic:
`UI (Napari/Qt) -> Controller -> ThreadWorker -> Backend (Segmentation) -> Controller -> Napari Layer Addition`

### Recommended Target Architecture
We recommend a strict Model-View-ViewModel (MVVM) or multi-tier Service Architecture:
1.  **Napari View (UI):** Only handles Qt events, docking, and translating UI actions to Service requests.
2.  **Application Services:** `SegmentationService`, `ConsensusService`, `FeatureService`. These manage the thread queues, emit progress signals, and handle the business logic asynchronously.
3.  **Data/State Manager:** Stores the image arrays using lazy loading (Dask/Zarr) instead of holding NumPy arrays in memory.
4.  **Core Engines:** Pure Python functions for segmentation, mask consensus, and I/O. Strictly UI-agnostic.

**Benefit:** Eliminates UI freezing, simplifies unit testing (you can test `SegmentationService` without launching Napari), and allows IRIS-Viewer to be run headlessly or from a CLI.
**Migration effort:** Medium. The backend logic is already somewhat isolated in `engine/segmentation/base.py`, but `controller.py` needs to be aggressively refactored.

## 2. Repository Structure

**Problem:** The repository lacks standard Python packaging files (`pyproject.toml`, `setup.py`, `requirements.txt`, `README.md`) and a `tests/` directory.
**Recommended fix:** Adopt a modern `src/` layout:
```text
iris-viewer/
├── pyproject.toml
├── README.md
├── src/
│   └── iris_viewer/
│       ├── __init__.py
│       ├── ui/             # Napari widgets, layout, controllers
│       ├── core/           # State, IO, data models
│       └── engine/         # Segmentation backends, consensus, features
└── tests/
```

## 3. Segmentation Audit

| Backend | API Correct? | Parameters | Dependencies | GPU | Memory concerns | Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cellpose** | Yes (mostly) | `expand_radius` ignored | `cellpose` | Auto-detects | **HIGH:** No `tile=True` passed | Add tiling; fix param mapping |
| **StarDist** | Yes | Global norm hardcoded | `stardist`, `csbdeep` | N/A | **HIGH:** `n_tiles` not used | Add chunking |
| **InstanSeg** | Needs Check | `pixel_size` used | `instanseg` | MPS/CUDA | **LOW:** Uses `eval_medium_image` | Verify manual `[0,1]` norm |
| **CellSAM** | Version-dep | DeepCell token mixed in? | `cellSAM` | CUDA mapped | **MEDIUM** | Fix DeepCell token anomaly |

### Critical Issues

*   **File/function:** `app/controller.py` -> `_build_method_kwargs()`
    **Problem:** UI parameters are hardcoded to ignore `expand_radius`.
    **Evidence:** `kwargs['expand_radius'] = 0` is hardcoded for Cellpose, InstanSeg, and CellSAM regardless of user input or morphological needs.
    **Recommended fix:** Pass the actual UI-provided radius to the backends.

*   **File/function:** `engine/segmentation/cellsam_backend.py` -> `segment()`
    **Problem:** Potential package naming / API mismatch.
    **Evidence:** It imports `cellSAM` but requires an environment variable `DEEPCELL_ACCESS_TOKEN`. DeepCell (by Van Valen lab) powers Mesmer, not standard SAM. This suggests a custom internal wrapper is being used, which will fail for open-source users.
    **Recommended fix:** Document the exact provenance of `cellSAM` or switch to standard `segment-anything` or `deepcell`.

*   **File/function:** `engine/segmentation/instanseg_backend.py` -> `segment()`
    **Problem:** Manual `[0, 1]` percentile normalization before model inference.
    **Evidence:** It manually clips to the 1st-99.9th percentile.
    **Recommended fix:** **Needs runtime verification**. Check if the official `instanseg` model requires pre-normalized `[0,1]` inputs, or if this causes hallucinated background noise.

## 4. Dependency Audit

**Missing / Optional Dependencies:**
*   There is no dependency declaration. 
*   **Action:** Create a `pyproject.toml` with optional extras:
    ```toml
    [project.optional-dependencies]
    cellpose = ["cellpose>=3.0"]
    stardist = ["stardist", "csbdeep"]
    instanseg = ["instanseg"]
    all = ["cellpose", "stardist", "instanseg"]
    ```

## 5. Memory and Performance Issues

*   **File/function:** `io_utils/image_loaders.py` -> `load_multichannel_tiff`
    **Problem:** Full-image RAM loading.
    **Evidence:** `raw = tifffile.imread(str(filepath))` loads the entire file into memory as a NumPy array.
    **Recommended fix:** Use `dask.array.image.imread` or `tifffile.imread(..., aszarr=True)` wrapped in Dask to support lazy loading for large multiplexed datasets.

*   **File/function:** `engine/segmentation/cellpose_backend.py` -> `segment()`
    **Problem:** Out-of-memory on large images.
    **Evidence:** Passes the whole `input_cyto` array to `eval()` without enabling tiling (`tile=True`).
    **Recommended fix:** Add `tile=True, tile_overlap=0.1` to the `eval()` call.

*   **File/function:** `engine/consensus.py` -> `compute_consensus_mask`
    **Problem:** Mask arrays are allocated at full size multiple times.
    **Evidence:** Creates multiple `np.zeros((h, w))` arrays.
    **Recommended fix:** Chunk the consensus generation or use sparse coordinate lists until rendering is required.

## 6. Scientific Correctness

*   **File/function:** `io_utils/image_loaders.py` -> `load_multichannel_tiff`
    **Problem:** Discarding Z/T slices.
    **Evidence:** `return raw[0], raw[0].shape[0]`. If a user loads a Z-stack or Time-series (4D), it silently discards everything but the first slice.
    **Recommended fix:** Check dimensionality and prompt the user to select a slice, perform a Maximum Intensity Projection (MIP), or reject the file.
*   **File/function:** `io_utils/image_loaders.py`
    **Problem:** Physical pixel size ignored.
    **Evidence:** Fails to parse `PhysicalSizeX`/`Y` from OME-XML. Instead, `app/controller.py` relies on a manual UI `spin_scale.value()`.

## 7. Migration Plan

1.  **Phase 1: Packaging & Testing:** Add `pyproject.toml`, establish a standard `src/` layout, and create baseline integration tests for the segmentation backends using small dummy arrays.
2.  **Phase 2: I/O & Memory:** Replace full-RAM `tifffile.imread` with Dask lazy loading. Implement `tile=True` across all deep learning backends.
3.  **Phase 3: Controller Decoupling:** Extract the thread/queue management from `app/controller.py` into a headless `PipelineService`.
4.  **Phase 4: Scientific Fixes:** Add OME-XML physical pixel size parsing and proper Z-stack handling.
