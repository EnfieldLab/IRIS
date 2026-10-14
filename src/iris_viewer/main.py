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
# Use standard user cache directory ~/.cache/iris (or IRIS_DATA_DIR) for downloaded weights,
# preserving bundled static resources within package data.

_user_cache_dir = Path(os.environ.get("IRIS_DATA_DIR", Path.home() / ".cache" / "iris"))
_pkg_data_dir = Path(__file__).parent / "data"

def _setenv(var: str, default: Path) -> None:
    """Set environment variable default if not explicitly provided."""
    if var not in os.environ:
        default.mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(default)

_bioimageio_dir = _pkg_data_dir / "bioimageio" if (_pkg_data_dir / "bioimageio").exists() else _user_cache_dir / "bioimageio"

_setenv("CELLPOSE_LOCAL_MODELS_PATH", _user_cache_dir / "cellpose")
_setenv("INSTANSEG_MODEL_PATH",       _user_cache_dir / "instanseg_models")
_setenv("INSTANSEG_BIOIMAGEIO_PATH",  _bioimageio_dir)
_setenv("TORCH_HOME",                 _user_cache_dir / "torch")
_setenv("HF_HOME",                    _user_cache_dir / "huggingface")

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