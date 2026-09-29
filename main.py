import os
import sys
from pathlib import Path

local_data_dir = Path(__file__).parent / "data"
local_data_dir.mkdir(exist_ok=True)

os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = str(local_data_dir / "cellpose")
os.environ["INSTANSEG_MODEL_PATH"] = str(local_data_dir / "instanseg_models")
os.environ["INSTANSEG_BIOIMAGEIO_PATH"] = str(local_data_dir / "bioimageio")
os.environ["TORCH_HOME"] = str(local_data_dir / "torch")
os.environ["HF_HOME"] = str(local_data_dir / "huggingface")

import napari
from app.controller import ViewerController

def main():
    controller = ViewerController()
    napari.run()

if __name__ == "__main__":
    main()