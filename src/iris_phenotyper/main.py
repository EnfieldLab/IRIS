"""
IRIS-Phenotyper entry point.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Environment setup (before any heavy imports) ──────────────────────────────
_local_data = Path.home() / ".iris" / "phenotyper_data"

def _setenv(var: str, default: Path) -> None:
    if var not in os.environ:
        default.mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(default)

_setenv("TORCH_HOME",  _local_data / "torch")
_setenv("HF_HOME",     _local_data / "huggingface")

# ── Deferred imports ──────────────────────────────────────────────────────────
import napari
from iris_phenotyper.app.controller import PhenotyperController


def main() -> None:
    """Launch IRIS-Phenotyper."""
    controller = PhenotyperController()

    # Optional: auto-load a session JSON from command line
    if len(sys.argv) > 1:
        session_path = Path(sys.argv[1])
        if session_path.is_file() and session_path.suffix.lower() == ".json":
            try:
                controller.state, _ = controller.state.load_session(session_path)
            except Exception:
                pass

    napari.run()


if __name__ == "__main__":
    main()
