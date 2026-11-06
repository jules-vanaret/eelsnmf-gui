from __future__ import annotations

import importlib
import logging
import os
import shutil
import tempfile
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np
from scipy.ndimage import median_filter
from skimage.filters import threshold_multiotsu
from skimage.morphology import dilation, disk

from ._results import MaskParams, MaskResult, PreprocessedSI
from ._util import HiddenPrints

logger = logging.getLogger(__name__)

# A loader turns (folders, labels, median_filter_size) into preprocessed SIs.
Loader = Callable[[Sequence[str], Sequence[str], int], List[PreprocessedSI]]


# ---------------------------------------------------------------------------
# Pure numpy helpers
# ---------------------------------------------------------------------------


def despike_array(data: np.ndarray, median_filter_size: int) -> np.ndarray:
    """Hampel-style spike removal along the energy axis of one SI.

    Outliers are detected as samples deviating from a moving median by more
    than 7 robust standard deviations (MAD-based) and are replaced by that
    median.
    """
    data = np.asarray(data, dtype=np.float32)
    med = median_filter(data, size=(1, 1, median_filter_size))
    diff = data - med
    np.abs(diff, out=diff)
    mad = median_filter(diff, size=(1, 1, median_filter_size))

    mask = diff > mad * 7.0
    filtered = data.copy()
    filtered[mask] = med[mask]
    return filtered


def common_energy_axis(
    energy_offsets: Sequence[float],
    energy_scales: Sequence[float],
    n_channels: Sequence[int],
) -> np.ndarray:
    """Compute a common energy axis (mean dispersion, overlapping window)."""
    offsets = np.asarray(energy_offsets, dtype=float)
    scales = np.asarray(energy_scales, dtype=float)
    ends = offsets + (np.asarray(n_channels, dtype=float) - 1) * scales

    scale = float(np.mean(scales))
    ei = float(np.max(offsets))
    ef = float(np.min(ends))

    n = int((ef - ei) // scale)
    if n < 2:
        raise ValueError("Invalid total range; resulting axis has <2 points.")

    return np.linspace(ei, ei + (n - 1) * scale, n)


def aggregate_energy(data: np.ndarray, mode: str) -> np.ndarray:
    """Aggregate the energy axis of one SI using mean or sum."""
    if mode == "sum":
        return np.sum(data, axis=-1)
    return np.mean(data, axis=-1)


def threshold_mask(
    img: np.ndarray,
    *,
    dilation_iters: int,
    filter_size: int = 3,
) -> np.ndarray:
    """Compute a boolean mask keeping the bright part of an aggregated image."""
    filtered = median_filter(img, size=filter_size)
    try:
        thresholds = threshold_multiotsu(filtered, classes=3)
        threshold = float(thresholds[0]) if len(thresholds) > 0 else 0.0
    except ValueError:
        threshold = float(np.min(filtered))
    mask = filtered >= threshold
    if dilation_iters > 0:
        structure = disk(1)
        for _ in range(dilation_iters):
            mask = dilation(mask, structure)
    return mask


def build_masks(
    datas: Sequence[np.ndarray],
    *,
    dilation_iters: int,
    mode: str,
) -> Tuple[List[np.ndarray], np.ndarray, List[int], List[Tuple[int, int]]]:
    """Build per-SI masks plus the flattened concatenation over all SIs."""
    per_si: List[np.ndarray] = []
    flats: List[np.ndarray] = []
    counts: List[int] = []
    shapes: List[Tuple[int, int]] = []

    for data in datas:
        data = np.asarray(data)
        shapes.append(tuple(data.shape[:2]))
        aggregated = aggregate_energy(data, mode)
        mask = threshold_mask(aggregated, dilation_iters=dilation_iters)
        per_si.append(mask)
        flats.append(mask.ravel())
        counts.append(int(mask.sum()))

    flat = np.concatenate(flats) if flats else np.array([])
    return per_si, flat, counts, shapes


def build_mask_result(sis: Sequence[PreprocessedSI], params: MaskParams) -> MaskResult:
    """Auto-build a MaskResult from preprocessed SIs."""
    per_si, flat, counts, shapes = build_masks(
        [si.data for si in sis],
        dilation_iters=params.dilation_iters,
        mode=params.mode,
    )
    return MaskResult(
        params=params, per_si=per_si, flat=flat, counts=counts, shapes=shapes
    )


def mask_result_from_user_masks(
    masks: Sequence[np.ndarray],
    shapes: Sequence[Tuple[int, int]],
    params: MaskParams,
) -> MaskResult:
    """Build a MaskResult from user-provided 2D masks (one per SI)."""
    if len(masks) != len(shapes):
        raise ValueError(
            f"Number of masks ({len(masks)}) must match number of "
            f"spectrum images ({len(shapes)})"
        )
    per_si: List[np.ndarray] = []
    for idx, mask in enumerate(masks):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != shapes[idx]:
            raise ValueError(
                f"Mask shape {mask.shape} for image {idx} does not "
                f"match spectrum image shape {shapes[idx]}"
            )
        per_si.append(mask)
    flat = np.concatenate([m.ravel() for m in per_si])
    counts = [int(m.sum()) for m in per_si]
    return MaskResult(
        params=params,
        per_si=per_si,
        flat=flat,
        counts=counts,
        shapes=list(shapes),
    )


def mask_flat_to_images(
    mask_flat: np.ndarray,
    shapes: Sequence[Tuple[int, int]],
) -> Tuple[List[np.ndarray], List[int]]:
    """Expand a flattened mask back into per-SI masks and counts."""
    mask_images: List[np.ndarray] = []
    mask_counts: List[int] = []
    offset = 0
    for height, width in shapes:
        count = height * width
        if offset + count > mask_flat.size:
            raise ValueError("mask_flat size does not match image shapes")
        mask = mask_flat[offset : offset + count].reshape((height, width))
        mask_images.append(mask)
        mask_counts.append(int(mask.sum()))
        offset += count
    return mask_images, mask_counts


# ---------------------------------------------------------------------------
# Default loader (EML / dm4 files)
# ---------------------------------------------------------------------------


def _visible_deprecation_warnings() -> tuple:
    """Collect VisibleDeprecationWarning classes when their packages exist."""
    classes = []
    for module_name, attr in (
        ("rsciio.exceptions", "VisibleDeprecationWarning"),
        ("hyperspy.exceptions", "VisibleDeprecationWarning"),
    ):
        try:
            module = importlib.import_module(module_name)
            classes.append(getattr(module, attr))
        except Exception:
            continue
    return tuple(classes)


def _process_one_eml(args):
    """Preprocess one spectrum image entry (thread-pool worker).

    With ``preprocess=False`` the core-loss signal is returned as loaded:
    the SIs are assumed already despiked and ZLP-aligned. Otherwise the SIs
    are ZLP-aligned and despiked with the EML algorithm by default, or with
    the median-based Hampel filter when ``despike_method="hampel"``.
    """
    from EELSNMF.experimental import EML  # lazy heavy import

    key, entry, median_filter_size, preprocess, despike_method = args

    # Split low-loss and core-loss signals.
    ll, cl = entry[0][-2:]
    if not preprocess:
        return key, cl

    p = EML.EML_Processing(cl.isig[2:-2], ll)
    p.zlp_align()

    if despike_method == "hampel":
        # Hampel filter for outlier removal, applied per spectrum.
        p.despiked = True
        p.core_loss.data = despike_array(
            np.asarray(p.core_loss.data), median_filter_size
        )
    elif despike_method == "eml":
        # EML despiking algorithm run on each image.
        p.long_despike = (
            10  # number of iterations the despiking algorithm is run
        )
        p.despike(p.core_loss)
    else:
        raise ValueError(f"Unknown despike method: {despike_method!r}")
    return key, p.core_loss


def load_eml_folders(
    folders: Sequence[str],
    labels: Sequence[str],
    *,
    median_filter_size: int,
    preprocess: bool = True,
    despike_method: str = "eml",
) -> List[PreprocessedSI]:
    """Default loader: copy folders to a temp dir, load and preprocess dm4s.

    When *preprocess* is False, the core-loss signals are used exactly as
    loaded (no channel cropping, ZLP alignment or despiking) and
    *median_filter_size* is ignored. Otherwise *despike_method* selects the
    despiking algorithm: "eml" (default, EML despike) or "hampel"
    (median-based Hampel filter).
    """
    from EELSNMF.experimental import EML  # lazy heavy import

    ignored = _visible_deprecation_warnings()
    with tempfile.TemporaryDirectory() as temp_root:
        for folder, label in zip(folders, labels):
            shutil.copytree(folder, os.path.join(temp_root, label))

        with warnings.catch_warnings():
            for warning_class in ignored:
                warnings.simplefilter("ignore", warning_class)
            with HiddenPrints():
                si_list = EML.load_results(temp_root, "STEM SI.dm4")
        if len(si_list) == 0:
            raise ValueError(f"No SI found in {temp_root}")

        keys = list(si_list.keys())
        jobs = [
            (
                key,
                si_list[key],
                median_filter_size,
                preprocess,
                despike_method,
            )
            for key in keys
        ]

        results = {}
        with HiddenPrints():
            max_workers = min(os.cpu_count() or 1, len(jobs))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_process_one_eml, job) for job in jobs]
                for future in as_completed(futures):
                    key, processed = future.result()
                    results[key] = processed

        sis: List[PreprocessedSI] = []
        for key in keys:
            entry = si_list[key]
            processed = results[key]
            unprocessed_cl = entry[0][-1]
            axis = processed.axes_manager[-1]
            sis.append(
                PreprocessedSI(
                    key=key,
                    label=os.path.basename(key),
                    data=np.asarray(processed.data, dtype=np.float32),
                    raw=processed,
                    original_image=np.asarray(
                        unprocessed_cl.data, dtype=np.float32
                    ).mean(-1),
                    pixel_scale=float(
                        unprocessed_cl.axes_manager.navigation_axes[0].scale
                    ),
                    energy_offset=float(axis.axis[0]),
                    energy_scale=float(axis.scale),
                )
            )
        return sis
