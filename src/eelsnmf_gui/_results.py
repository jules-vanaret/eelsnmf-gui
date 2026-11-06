from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Parameter value types (shared between the session, the engine and the GUI)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BeamParams:
    """Microscope parameters passed to the decomposition model."""

    E0_V: float
    alpha_rad: float
    beta_rad: float


@dataclass(frozen=True)
class MaskParams:
    """Parameters describing the spatial mask applied before decomposition."""

    enabled: bool = False
    dilation_iters: int = 1
    mode: str = "mean"  # energy aggregation mode: "mean" or "sum"


# ---------------------------------------------------------------------------
# Preprocessing results
# ---------------------------------------------------------------------------


@dataclass
class PreprocessedSI:
    """A single preprocessed spectrum image with numpy-friendly metadata.

    *data* is the processed core-loss cube (ny, nx, n_energy). *raw* keeps the
    backend-specific processed signal (e.g. a hyperspy signal) so the NMF
    engine can consume it; pure-numpy stages must not rely on it.
    """

    key: str  # loader key; basename matches the folder label
    label: str  # folder label
    data: np.ndarray  # (ny, nx, n_energy) float32
    raw: object
    original_image: np.ndarray  # (ny, nx) mean over energy of the raw data
    pixel_scale: float  # nm per pixel
    energy_offset: float  # energy of the first channel (eV)
    energy_scale: float  # dispersion (eV per channel)

    @property
    def spatial_shape(self) -> Tuple[int, int]:
        return self.data.shape[:2]

    @property
    def n_energy(self) -> int:
        return self.data.shape[2]


@dataclass
class PreprocessedData:
    """All preprocessed spectrum images of a session, in loader key order."""

    sis: List[PreprocessedSI]
    folder_index: List[int]  # folder index of each SI
    labels: List[str]  # folder labels, in folder order
    energy_axis: np.ndarray  # common axis across all SIs (joint scope)


@dataclass
class MaskResult:
    """Spatial masks for one set of mask parameters."""

    params: MaskParams
    per_si: List[np.ndarray]  # (ny, nx) boolean mask per SI
    flat: np.ndarray  # per_si raveled and concatenated
    counts: List[int]  # number of True pixels per SI
    shapes: List[Tuple[int, int]]  # spatial shape per SI


@dataclass
class DecoBundle:
    """An NMF decomposition object plus its ListOfSI wrapper."""

    deco: object
    list_of_si: object


# ---------------------------------------------------------------------------
# Final results (unchanged contract used by the GUI and export helpers)
# ---------------------------------------------------------------------------


@dataclass
class EdgeSeries:
    x: np.ndarray
    raw: List[np.ndarray]
    smoothed: List[np.ndarray]
    norm: List[np.ndarray]


@dataclass
class EdgeCollection:
    combined: EdgeSeries
    by_label: Dict[str, EdgeSeries]


@dataclass
class ImageCollection:
    mixed: List[np.ndarray] | None
    original: List[np.ndarray] | None
    components: List[List[np.ndarray]] | None
    mask: List[np.ndarray] | None = None


@dataclass
class ProcessingResult:
    images: ImageCollection
    edges: EdgeCollection
    pixel_scale: float | None
    energy_axis: np.ndarray
    list_keys: Sequence[str]
    W: np.ndarray
    deco: object
    edge_contributions_summary: List[Dict[str, float]] | None = None


@dataclass
class ProcessFoldersResult:
    individual: List[ProcessingResult]
    joint: ProcessingResult | None
    labels: List[str]
    labels_joint: List[str]
    deco_joint: object | None
    decos_individual: List[object]
