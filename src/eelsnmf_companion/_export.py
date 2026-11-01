from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from ._results import EdgeSeries


def _upscale_nearest(img: np.ndarray, scale: int = 10) -> np.ndarray:
    """Upscale an image using nearest-neighbor sampling."""
    if scale <= 1:
        return img

    if img.ndim == 2:
        return zoom(img, (scale, scale), order=0)

    if img.ndim == 3 and img.shape[-1] in (3, 4):
        return zoom(img, (scale, scale, 1), order=0)

    return img


def _save_images(
    list_imgs: Sequence[Sequence[np.ndarray]],
    labels: Sequence[str],
    output_dir: str,
    *,
    cmap: str | None = None,
    scale: int = 10,
):
    """Save a set of images to disk with optional upscaling and colormap."""
    os.makedirs(output_dir, exist_ok=True)
    if len(list_imgs) > 0 and isinstance(list_imgs[0], np.ndarray):
        # One image per label: wrap each image individually.
        list_imgs = [[img] for img in list_imgs]
    for label, imgs in zip(labels, list_imgs):
        if imgs is None:
            continue
        for j, img in enumerate(imgs):
            img_filename = (
                f"{label}_image{j:02d}.png"
                if len(imgs) > 1
                else f"{label}.png"
            )
            img_path = os.path.join(output_dir, img_filename)
            upscaled = _upscale_nearest(np.asarray(img), scale=scale)
            if cmap is None:
                plt.imsave(img_path, upscaled)
            else:
                plt.imsave(img_path, upscaled, cmap=cmap)


def save_colored_images(
    list_imgs: Sequence[Sequence[np.ndarray]],
    labels: Sequence[str],
    output_dir: str,
):
    """Save mixed-color component images to disk."""
    _save_images(list_imgs, labels, output_dir, cmap=None, scale=10)


def save_original_images(
    list_imgs_original: Sequence[Sequence[np.ndarray]],
    labels: Sequence[str],
    output_dir: str,
):
    """Save original grayscale images to disk."""
    _save_images(list_imgs_original, labels, output_dir, cmap="gray", scale=10)


def save_mask_images(
    list_masks: Sequence[Sequence[np.ndarray]],
    labels: Sequence[str],
    output_dir: str,
):
    """Save mask images to disk."""
    _save_images(list_masks, labels, output_dir, cmap="gray", scale=10)


def save_component_images(
    list_imgs_component: Sequence[Sequence[np.ndarray]],
    labels: Sequence[str],
    output_dir: str,
    component_label: str,
):
    """Save per-component images into a component subdirectory."""
    component_dir = os.path.join(output_dir, component_label)
    _save_images(
        list_imgs_component, labels, component_dir, cmap=None, scale=10
    )


def save_summary_plot(fig, output_path: str, dpi: int = 300):
    """Save a matplotlib figure to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi)


def _build_edges_csv_data(
    energy_axis: np.ndarray,
    edge_raw: Sequence[np.ndarray],
    edge_smoothed: Sequence[np.ndarray],
    edge_smoothed_norm: Sequence[np.ndarray],
) -> Dict[str, np.ndarray]:
    """Assemble columnar edge data for CSV export."""
    data: Dict[str, np.ndarray] = {"Energy (eV)": energy_axis}
    for idx, values in enumerate(edge_raw):
        data[f"Component {idx + 1}"] = values
    for idx, values in enumerate(edge_smoothed):
        data[f"Component {idx + 1} Smoothed"] = values
    for idx, values in enumerate(edge_smoothed_norm):
        data[f"Component {idx + 1} Smoothed (Norm)"] = values
    return data


def export_edges_to_csv(
    full_energy_axis: np.ndarray,
    edge_raw: Sequence[np.ndarray],
    edge_smoothed: Sequence[np.ndarray],
    edge_smoothed_norm: Sequence[np.ndarray],
    output_path: str,
):
    """Export combined edge data to a CSV file."""
    data = _build_edges_csv_data(
        full_energy_axis,
        edge_raw,
        edge_smoothed,
        edge_smoothed_norm,
    )
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def export_edges_by_label_to_csv(
    edges_by_label: Dict[str, EdgeSeries],
    output_dir: str,
    *,
    filename_prefix: str = "",
):
    """Export per-label edge data to CSV files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    for label, edge in edges_by_label.items():
        data = _build_edges_csv_data(
            edge.x,
            edge.raw,
            edge.smoothed,
            edge.norm,
        )
        df = pd.DataFrame(data)
        filename = f"{filename_prefix}{label}.csv"
        df.to_csv(os.path.join(output_dir, filename), index=False)


def export_edge_contributions_summary_to_csv(
    summary_rows: List[Dict[str, float]],
    output_path: str,
):
    """Export edge contribution summaries to a CSV file."""
    if not summary_rows:
        return
    adjusted_rows: List[Dict[str, float]] = []
    for row in summary_rows:
        updated = dict(row)
        if "Component" in updated:
            try:
                updated["Component"] = int(updated["Component"]) + 1
            except (TypeError, ValueError):
                pass
        adjusted_rows.append(updated)
    df = pd.DataFrame(adjusted_rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
