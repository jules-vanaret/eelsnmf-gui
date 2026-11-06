from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ._results import EdgeCollection, EdgeSeries, MaskResult

logger = logging.getLogger(__name__)

Smoother = Callable[[np.ndarray], np.ndarray]


def default_smoother(arr: np.ndarray) -> np.ndarray:
    """Whittaker smoother from the EML package (imported lazily)."""
    from EELSNMF.experimental import EML  # lazy heavy import

    return EML.we_smooth(arr)


def find_index(ax: np.ndarray, v):
    """Find nearest axis indices for one value or an iterable of values."""
    if hasattr(v, "__iter__"):
        return [abs(ax - iv).argmin() for iv in v]
    return abs(ax - v).argmin()


def _concat(chunks: List[np.ndarray]) -> np.ndarray:
    """Concatenate chunks into a single array, preserving empty cases."""
    if len(chunks) == 0:
        return np.array([])
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks)


# ---------------------------------------------------------------------------
# Edge contribution metrics
# ---------------------------------------------------------------------------


@dataclass
class EdgeTotals:
    """Per-component edge contribution metrics."""

    leave_out_loss: np.ndarray
    power: np.ndarray
    explained_variance: np.ndarray


def edge_totals(
    deco,
    fine_structure_ranges: Dict[str, Tuple[float, float]],
) -> EdgeTotals | None:
    """Per-component edge contribution metrics from a fitted decomposition.

    *deco* must expose X, G, W, H, get_model(), n_components and
    model._edge_slices (the EELSNMF decomposition object does).
    """
    if not hasattr(deco, "model") or not hasattr(deco.model, "_edge_slices"):
        return None

    n_components = int(getattr(deco, "n_components", 0))
    if n_components <= 0:
        return None

    edge_slices = deco.model._edge_slices
    X = np.asarray(deco.X)
    model = np.asarray(deco.get_model())
    norm_x = float((X**2).sum())
    residual = X - model

    G = np.asarray(deco.G)
    W = np.asarray(deco.W)
    H = np.asarray(deco.H)

    loo = np.zeros(n_components, dtype=float)
    power = np.zeros(n_components, dtype=float)
    ev = np.zeros(n_components, dtype=float)

    for edge in fine_structure_ranges:
        if edge not in edge_slices:
            continue
        idxs = edge_slices[edge]
        for k in range(n_components):
            wk = G[:, idxs] @ W[idxs, k]
            hk = H[k, :]

            comp_sq = np.sum(wk**2) * np.sum(hk**2)

            power_vals = comp_sq / norm_x if norm_x else 0.0

            interaction_x = (wk.T @ X) @ hk.T
            evk = (2 * interaction_x - comp_sq) / norm_x if norm_x else 0.0

            interaction_residual = wk.T @ residual @ hk.T
            ck = (comp_sq + 2 * interaction_residual) / norm_x if norm_x else 0.0

            loo[k] += ck
            power[k] += power_vals
            ev[k] += evk

    return EdgeTotals(
        leave_out_loss=loo, power=power, explained_variance=ev
    )


def order_by_power(
    totals: EdgeTotals | None, *, n_components: int
) -> List[int]:
    """Component indices ordered by decreasing power (identity fallback)."""
    if totals is None or not np.any(totals.power):
        return list(range(n_components))
    return np.argsort(totals.power)[::-1].tolist()


def summary_rows(totals: EdgeTotals, order: Sequence[int]) -> List[Dict[str, float]]:
    """Summary table rows for components listed in *order*."""
    totals_sum = {
        "Leave-Out Loss": float(np.sum(totals.leave_out_loss)),
        "Power": float(np.sum(totals.power)),
        "Explained Variance": float(np.sum(totals.explained_variance)),
    }
    rows: List[Dict[str, float]] = []
    for position, comp_idx in enumerate(order):
        rows.append(
            {
                "Component": position,
                "Leave-Out Loss": float(totals.leave_out_loss[comp_idx]),
                "Relative Leave-Out Loss (%)": (
                    100.0
                    * totals.leave_out_loss[comp_idx]
                    / totals_sum["Leave-Out Loss"]
                    if totals_sum["Leave-Out Loss"]
                    else 0.0
                ),
                "Power": float(totals.power[comp_idx]),
                "Relative Power (%)": (
                    100.0 * totals.power[comp_idx] / totals_sum["Power"]
                    if totals_sum["Power"]
                    else 0.0
                ),
                "Explained Variance": float(
                    totals.explained_variance[comp_idx]
                ),
                "Relative Explained Variance (%)": (
                    100.0
                    * totals.explained_variance[comp_idx]
                    / totals_sum["Explained Variance"]
                    if totals_sum["Explained Variance"]
                    else 0.0
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Edge spectra extraction
# ---------------------------------------------------------------------------


def extract_edge_collection(
    deco,
    fine_structure_ranges: Dict[str, Tuple[float, float]],
    n_components: int,
    *,
    order: Sequence[int] | None = None,
    smoother: Smoother | None = None,
) -> EdgeCollection:
    """Extract per-edge spectra for each component, combined and by label."""
    if smoother is None:
        smoother = default_smoother
    if order is None:
        order = list(range(n_components))

    by_label: Dict[str, EdgeSeries] = {}
    for edge_label in fine_structure_ranges:
        ii, ff = find_index(
            deco.energy_axis, deco.fine_structure_ranges[edge_label]
        )
        x_axis = deco.energy_axis[ii:ff]

        raw: List[np.ndarray] = []
        smoothed: List[np.ndarray] = []
        norm: List[np.ndarray] = []
        for comp_idx in order:
            edge_raw = np.asarray(
                deco.get_edge_from_component(comp_idx, edge_label).data
            )
            edge_smoothed = np.asarray(smoother(edge_raw.copy()))[ii:ff]
            denom = edge_smoothed.max() - edge_smoothed.min()
            if denom == 0:
                edge_norm = np.zeros_like(edge_smoothed)
            else:
                edge_norm = (edge_smoothed - edge_smoothed.min()) / denom

            raw.append(edge_raw[ii:ff])
            smoothed.append(edge_smoothed)
            norm.append(edge_norm)

        by_label[edge_label] = EdgeSeries(
            x=x_axis, raw=raw, smoothed=smoothed, norm=norm
        )

    return EdgeCollection(
        combined=_combine_edges(by_label, n_components),
        by_label=by_label,
    )


def _combine_edges(by_label: Dict[str, EdgeSeries], n_components: int) -> EdgeSeries:
    """Concatenate per-edge spectra sorted by edge onset energy."""
    ordered = sorted(
        by_label.values(),
        key=lambda edge: edge.x[0] if len(edge.x) > 0 else np.inf,
    )
    return EdgeSeries(
        x=_concat([edge.x for edge in ordered]),
        raw=[_concat([edge.raw[i] for edge in ordered]) for i in range(n_components)],
        smoothed=[
            _concat([edge.smoothed[i] for edge in ordered])
            for i in range(n_components)
        ],
        norm=[
            _concat([edge.norm[i] for edge in ordered])
            for i in range(n_components)
        ],
    )


# ---------------------------------------------------------------------------
# Component intensity maps
# ---------------------------------------------------------------------------


def _edge_weights(
    deco,
    fine_structure_ranges: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    """Per-component spectral weights restricted to fine-structure ranges."""
    edge_slices = deco.model._edge_slices
    idxs_list = [
        edge_slices[edge]
        for edge in fine_structure_ranges
        if edge in edge_slices
    ]
    if idxs_list:
        weights = np.zeros(deco.W.shape[1], dtype=float)
        for slc in idxs_list:
            weights += (deco.G[:, slc] @ deco.W[slc, :]).sum(0)
    else:
        weights = (deco.G @ deco.W).sum(0)
    return weights


def expand_masked_components(
    values: np.ndarray,
    mask_flat: np.ndarray,
    shapes: Sequence[Tuple[int, int]],
) -> List[List[np.ndarray]]:
    """Restore masked component vectors back to per-SI 2D images."""
    n_components, _ = values.shape
    total_pixels = sum(h * w for h, w in shapes)
    full = np.zeros((n_components, total_pixels), dtype=values.dtype)
    full[:, mask_flat] = values

    images: List[List[np.ndarray]] = [[] for _ in range(n_components)]
    offset = 0
    for height, width in shapes:
        count = height * width
        for comp_idx in range(n_components):
            chunk = full[comp_idx, offset : offset + count]
            images[comp_idx].append(chunk.reshape((height, width)))
        offset += count
    return images


def component_maps(
    deco,
    list_of_si,
    fine_structure_ranges: Dict[str, Tuple[float, float]],
    n_components: int,
    *,
    order: Sequence[int],
    mask_result: MaskResult | None,
) -> List[List[np.ndarray]]:
    """Normalized (uncolored) component intensity maps, one list per SI."""

    weights = _edge_weights(deco, fine_structure_ranges)
    normalized = np.asarray(deco.H) * weights[:, None]

    spatial_dims = list(list_of_si.spatial_dim_list)
    if len(spatial_dims) > 1:
        # Normalize per SI (or per masked chunk) so every image is comparable.
        normalized = normalized.copy()
        filled = 0
        sizes = (
            list(mask_result.counts) if mask_result is not None else spatial_dims
        )
        for size in sizes:
            chunk = normalized[:, filled : filled + size]
            denom = float(np.max(np.max(chunk, axis=1)))
            if denom > 0:
                normalized[:, filled : filled + size] = chunk / denom
            filled += size
    else:
        normalized = normalized / max(normalized.max(axis=1).tolist())

    ordered = normalized[list(order)]
    if mask_result is not None:
        return expand_masked_components(
            ordered, mask_result.flat, mask_result.shapes
        )
    return [list_of_si.fold_array(ordered[i]) for i in range(n_components)]


def component_maps_from_deco(deco, fine_structure_ranges):
    """Rebuild normalized component maps from a loaded deco object.

    Only possible for single-SI, unmasked decos (the per-SI split and the
    masked-pixel bookkeeping are not stored in the deco). Returns a list per
    component containing one (ny, nx) image, or None when reconstruction is
    not possible.
    """
    n_components = int(getattr(deco, "n_components", 0) or 0)
    if n_components < 1:
        return None
    shape = getattr(deco, "spatial_shape", None)
    if shape is None or len(shape) != 2:
        return None
    n_pix = int(shape[0]) * int(shape[1])
    H = np.asarray(deco.H)
    if H.ndim != 2 or H.shape[1] != n_pix:
        return None

    weights = _edge_weights(deco, fine_structure_ranges)
    normalized = np.asarray(deco.H) * weights[:, None]
    normalized = normalized / max(normalized.max(axis=1).tolist())
    return [[normalized[i].reshape(shape)] for i in range(n_components)]


def apply_colors(
    maps: List[List[np.ndarray]],
    colors: Sequence[np.ndarray],
) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    """Mix component maps into RGB images; also color each component map."""
    n_components = len(maps)
    mixed: List[np.ndarray] = []
    n_images = len(maps[0]) if maps else 0
    for idx in range(n_images):
        rgb = None
        for comp_idx in range(n_components):
            contribution = maps[comp_idx][idx][..., None] * np.asarray(
                colors[comp_idx]
            )
            rgb = contribution if rgb is None else rgb + contribution
        mixed.append(np.clip(rgb, 0.0, 1.0))

    colored = [
        [
            np.clip(img[..., None] * np.asarray(colors[i]), 0.0, 1.0)
            for img in comp_maps
        ]
        for i, comp_maps in enumerate(maps)
    ]
    return mixed, colored


def zero_outside_masks(
    mixed: List[np.ndarray],
    original: List[np.ndarray] | None,
    colored_maps: List[List[np.ndarray]],
    per_si_masks: Sequence[np.ndarray],
) -> None:
    """Zero out image values where the mask is False (in place)."""
    for idx, mask in enumerate(per_si_masks):
        if original is not None:
            original[idx] = np.where(mask, original[idx], 0.0)
        mixed[idx] = np.where(mask[..., None], mixed[idx], 0.0)
        for comp_idx in range(len(colored_maps)):
            colored_maps[comp_idx][idx] = np.where(
                mask[..., None], colored_maps[comp_idx][idx], 0.0
            )


# ---------------------------------------------------------------------------
# Joint-result helpers
# ---------------------------------------------------------------------------


def map_images_to_labels(images, keys, labels: Sequence[str]):
    """Reorder per-key images to match label order from folder names."""
    if images is None or keys is None:
        return []
    key_map = {os.path.basename(k): img for k, img in zip(keys, images)}
    return [key_map.get(label) for label in labels]
