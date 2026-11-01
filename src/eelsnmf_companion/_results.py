from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


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
    mixed: Optional[List[np.ndarray]]
    original: Optional[List[np.ndarray]]
    components: Optional[List[List[np.ndarray]]]
    mask: Optional[List[np.ndarray]] = None


@dataclass
class ProcessingResult:
    images: ImageCollection
    edges: EdgeCollection
    pixel_scale: Optional[float]
    energy_axis: np.ndarray
    list_keys: Sequence[str]
    W: np.ndarray
    deco: object
    edge_contributions_summary: Optional[List[Dict[str, float]]] = None


@dataclass
class ProcessFoldersResult:
    individual: List[ProcessingResult]
    joint: Optional[ProcessingResult]
    labels: List[str]
    labels_joint: List[str]
    deco_joint: Optional[object]
    decos_individual: List[object]
