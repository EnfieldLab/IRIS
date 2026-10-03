"""
IRIS-Viewer entry point.

All environment variables that control model storage paths are set here,
BEFORE any module that imports torch / cellpose / instanseg is loaded.
This is the single canonical location for these settings — controller.py
no longer sets them.
"""
import os
import sys
from pathlib import Path

# ── Model Storage Paths ────────────────────────────────────────────────────────
# Resolve to <package_root>/data/ so models are always stored inside the
# package directory tree, regardless of where the process is launched from.

_local_data_dir = Path(__file__).parent / "data"
_local_data_dir.mkdir(exist_ok=True)

# Only set if not already overridden in the environment (allows CI/test overrides)
def _setenv(var: str, default: Path) -> None:
    if var not in os.environ:
        default.mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(default)

_setenv("CELLPOSE_LOCAL_MODELS_PATH", _local_data_dir / "cellpose")
_setenv("INSTANSEG_MODEL_PATH",       _local_data_dir / "instanseg_models")
_setenv("INSTANSEG_BIOIMAGEIO_PATH",  _local_data_dir / "bioimageio")
_setenv("TORCH_HOME",                 _local_data_dir / "torch")
_setenv("HF_HOME",                    _local_data_dir / "huggingface")

# ── Deferred imports (after env vars are in place) ─────────────────────────────
import napari
from iris_viewer.app.controller import ViewerController


def main() -> None:
    """Launch IRIS-Viewer."""
    controller = ViewerController()

    # Optional: auto-load a session file passed on the command line
    # e.g. iris-viewer /path/to/session.json
    if len(sys.argv) > 1:
        session_path = Path(sys.argv[1])
        if session_path.is_file() and session_path.suffix.lower() == ".json":
            try:
                controller._load_session.__func__(controller)  # call private method directly
            except Exception:
                pass  # non-fatal: user can load via menu

    napari.run()


if __name__ == "__main__":
    main()