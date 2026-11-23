import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "1.0.0"
__author__ = "IRIS (Interactive Research viewer for Integrated Spatial-omics) - Enfield Lab"
