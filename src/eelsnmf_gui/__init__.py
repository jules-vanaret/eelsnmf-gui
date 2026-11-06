from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("eelsnmf-gui")
except PackageNotFoundError:  # running from source without installation
    __version__ = "unknown"

from ._results import (
    BeamParams,
    DecoBundle,
    EdgeCollection,
    EdgeSeries,
    ImageCollection,
    MaskParams,
    MaskResult,
    PreprocessedData,
    PreprocessedSI,
    ProcessFoldersResult,
    ProcessingResult,
)
from ._widget import EELSProcessingWidget
from .session import Session

__all__ = (
    "EELSProcessingWidget",
    "Session",
    "BeamParams",
    "MaskParams",
    "PreprocessedData",
    "PreprocessedSI",
    "MaskResult",
    "DecoBundle",
    "ProcessingResult",
    "ProcessFoldersResult",
    "EdgeCollection",
    "EdgeSeries",
    "ImageCollection",
)
