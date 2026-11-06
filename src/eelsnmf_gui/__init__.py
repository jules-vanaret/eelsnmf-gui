try:
    from ._version import version as __version__
except ImportError:
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
