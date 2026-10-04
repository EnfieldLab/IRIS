"""
iris-common — Shared library for the IRIS-Suite.

Provides:
  iris_common.io          — image loaders and exporters
  iris_common.types       — shared dataclasses (MarkerPanel, etc.)
  iris_common.engine      — backend ABCs (SegmentationBackend, PhenotypeBackend)
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("iris-common")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"
