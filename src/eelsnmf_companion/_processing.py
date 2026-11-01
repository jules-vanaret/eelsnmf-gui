from __future__ import annotations

import hashlib
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import tempfile

import numpy as np
from scipy.interpolate import interp1d

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None
from EELSNMF.experimental import EML  # type: ignore[import-not-found]
import EELSNMF as enmf  # type: ignore[import-not-found]
import EELSNMF.utils as enmfu  # type: ignore[import-not-found]
import warnings

from rsciio.exceptions import VisibleDeprecationWarning  # type: ignore[import-not-found]

warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)

from hyperspy.exceptions import VisibleDeprecationWarning as HyperspyVisibleDeprecationWarning  # type: ignore[import-not-found]

warnings.filterwarnings("ignore", category=HyperspyVisibleDeprecationWarning)

from scipy.ndimage import median_filter
from skimage.filters import threshold_multiotsu
from skimage.morphology import dilation, disk
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ._results import (
    EdgeCollection,
    EdgeSeries,
    ImageCollection,
    ProcessFoldersResult,
    ProcessingResult,
)


def _detect_cupy_gpu_available() -> bool:
    """Return True if a usable CUDA device is available via CuPy."""
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


USE_CUPY = _detect_cupy_gpu_available()


class HiddenPrints:
    """Context manager that suppresses stdout/stderr inside the block."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._null = open(os.devnull, "w")
        sys.stdout = self._null
        sys.stderr = self._null

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        self._null.close()


@dataclass
class PreprocessResult:
    """Container for preprocessing outputs reused across pipeline stages."""

    processed_cl: Sequence
    energy_axis: np.ndarray
    list_keys: Sequence[str]
    mask_flat: Optional[np.ndarray] = None
    mask_params: Optional[Tuple[bool, int, str]] = None
    si_list: Optional[Dict[str, object]] = None


def _concat_arrays(chunks: List[np.ndarray]) -> np.ndarray:
    """Concatenate chunks into a single array, preserving empty cases."""
    if len(chunks) == 0:
        return np.array([])
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks)


def find_index(ax: np.ndarray, v):
    """Find nearest axis indices for one value or an iterable of values."""
    if hasattr(v, "__iter__"):
        return [abs(ax - iv).argmin() for iv in v]
    return abs(ax - v).argmin()


def _map_joint_images(
    imgs_joint,
    imgs_original_joint,
    list_keys,
    labels: Sequence[str],
):
    """Reorder joint images to match label order from folder names."""
    if imgs_joint is None or imgs_original_joint is None or list_keys is None:
        return [], []

    key_order = [os.path.basename(k) for k in list_keys]
    color_map = {key: img for key, img in zip(key_order, imgs_joint)}
    original_map = {
        key: img for key, img in zip(key_order, imgs_original_joint)
    }
    list_imgs_joint = [color_map.get(k) for k in labels]
    list_imgs_original_joint = [original_map.get(k) for k in labels]
    return list_imgs_joint, list_imgs_original_joint


def _map_joint_image_list(imgs, list_keys, labels: Sequence[str]):
    """Reorder a list of images to match label order from folder names."""
    if imgs is None or list_keys is None:
        return []

    key_order = [os.path.basename(k) for k in list_keys]
    image_map = {key: img for key, img in zip(key_order, imgs)}
    return [image_map.get(k) for k in labels]


def _copy_folders_to_temp(
    temp_root: str,
    source_folders: Sequence[str],
    labels: Sequence[str],
) -> None:
    """Copy source folders into a temporary root using label names."""
    for folder, label in zip(source_folders, labels):
        dst = os.path.join(temp_root, label)
        shutil.copytree(folder, dst)


def _process_one_preprocess(args):
    """Preprocess a single spectrum image entry for thread-pool use."""
    # Top-level function to keep it pickleable for ThreadPoolExecutor.
    key, SI, median_filter_size = args

    # Split low-loss and core-loss signals, then align the ZLP.
    ll, cl = SI[0][-2:]
    p = EML.EML_Processing(cl.isig[2:-2], ll)
    p.zlp_align()

    # Mark despiking as done; actual despike uses the median-based filter.
    p.despiked = True
    ### OLD DESPIKE
    # p.long_despike = (
    #     10  # number of iteration that the despiking algo is run on each image
    # )
    # p.despike(p.core_loss)
    # my_cl = p.core_loss

    ### NEW DESPIKE -> Hampel filter for outlier removal, applied to each spectrum independently
    my_cl = p.core_loss

    # Median filter in the energy dimension to remove spikes.
    data = np.asarray(my_cl.data, dtype=np.float32)

    median = median_filter(data, size=(1, 1, median_filter_size))
    diff = data - median
    np.abs(diff, out=diff)
    mad = median_filter(diff, size=(1, 1, median_filter_size))

    # Replace outliers by the local median.
    thr = mad * 7.0
    mask = diff > thr
    filtered = data.copy()
    filtered[mask] = median[mask]
    my_cl.data = filtered
    ###

    return key, my_cl


def _preprocess_cl(path_to_temp: str, *, median_filter_size: int):
    """Load spectral images from a temp path and apply preprocessing."""
    SI_list = EML.load_results(path_to_temp, "STEM SI.dm4")
    if len(SI_list) == 0:
        raise ValueError(f"No SI found in {path_to_temp}")

    list_keys = list(SI_list.keys())

    # Pack args once to reduce overhead in the thread pool.
    jobs = [(k, SI_list[k], median_filter_size) for k in list_keys]

    results = {}
    with HiddenPrints():
        max_workers = min(os.cpu_count(), len(jobs))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_process_one_preprocess, job) for job in jobs]
            for fut in as_completed(futures):
                k, my_cl = fut.result()
                results[k] = my_cl

    processed_cl = [results[k] for k in list_keys]
    return SI_list, processed_cl, list_keys


def _build_energy_axis(
    processed_cl: Sequence,
):
    """Compute a common energy axis for a list of spectra."""
    list_s = []
    for s in processed_cl:
        foo = s.axes_manager[-1]
        list_s.append((foo.axis[0], foo.axis[-1], foo.scale))
    list_s = np.array(list_s)

    # Use a single mean dispersion and the overlapping energy window.
    scale = np.mean(list_s[:, -1])

    ei = np.max(list_s[:, 0])
    ef = np.min(list_s[:, 1])

    n = int((ef - ei) // scale)
    if n < 2:
        raise ValueError("Invalid total range; resulting axis has <2 points.")

    energy_axis = np.linspace(ei, ei + (n - 1) * scale, n)
    return energy_axis, scale, ei, ef


def _build_deco(
    processed_cl: Sequence,
    energy_axis: np.ndarray,
    *,
    E0_V: float,
    alpha_rad: float,
    beta_rad: float,
    mask_flat: Optional[np.ndarray] = None,
):
    """Build an EELSNMF decomposition object and its ListOfSI wrapper."""
    sl = enmfu.ListOfSI(processed_cl, energy_axis)
    sl.unfolded_data = np.clip(sl.unfolded_data, 0, None)

    unfolded = sl.unfolded_si
    # Apply mask by reducing the unfolded data to masked pixels only.
    if mask_flat is not None:
        new_data = unfolded.data[mask_flat]
        unfolded = unfolded._deepcopy_with_new_data(new_data)
        unfolded.axes_manager.navigation_axes[0].size = new_data.shape[0]

    deco = enmf.EELSNMF(
        unfolded,
        E0=E0_V,
        alpha=alpha_rad,
        beta=beta_rad,
    )

    return deco, sl


def _aggregate_energy_axis(data: np.ndarray, mode: str) -> np.ndarray:
    """Aggregate the energy axis using mean or sum across spectra."""
    if mode == "sum":
        return np.sum(data, axis=-1)
    return np.mean(data, axis=-1)


def _build_threshold_mask(
    img: np.ndarray,
    *,
    dilation_iters: int,
    filter_size: int = 3,
) -> np.ndarray:
    """Compute a threshold mask from an aggregated image."""
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


def _build_mask_data(
    processed_cl: Sequence,
    *,
    dilation_iters: int,
    mode: str,
):
    """Build per-image masks and a flattened mask for all images."""
    masks: List[np.ndarray] = []
    mask_flat: List[np.ndarray] = []
    mask_counts: List[int] = []
    shapes: List[Tuple[int, int]] = []

    # Build per-image masks and flatten them for downstream masking.
    for cl in processed_cl:
        data = np.asarray(cl.data)
        shapes.append(tuple(data.shape[:2]))
        aggregated = _aggregate_energy_axis(data, mode)
        mask = _build_threshold_mask(
            aggregated,
            dilation_iters=dilation_iters,
        )
        flat = mask.ravel()
        masks.append(mask)
        mask_flat.append(flat)
        mask_counts.append(int(flat.sum()))

    combined = np.concatenate(mask_flat) if mask_flat else np.array([])
    return masks, combined, mask_counts, shapes


def _build_mask_flat(
    processed_cl: Sequence,
    *,
    mask_enabled: bool,
    mask_dilation_iters: int,
    mask_mode: str,
) -> Optional[np.ndarray]:
    """Compute a flattened mask over all spectra when masking is enabled."""
    if not mask_enabled:
        return None
    _, mask_flat, _, _ = _build_mask_data(
        processed_cl,
        dilation_iters=mask_dilation_iters,
        mode=mask_mode,
    )
    return mask_flat


def _mask_flat_to_images(
    mask_flat: np.ndarray,
    shapes: Sequence[Tuple[int, int]],
) -> Tuple[List[np.ndarray], List[int]]:
    """Expand a flattened mask back into per-image masks and counts."""
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


def _build_preprocess_result_for_folders(
    source_folders: Sequence[str],
    labels: Sequence[str],
    *,
    median_filter_size: int,
) -> PreprocessResult:
    """Copy folders to a temp location and build a PreprocessResult."""
    with tempfile.TemporaryDirectory() as temp_root:
        _copy_folders_to_temp(temp_root, source_folders, labels)

        si_list, processed_cl, list_keys = _preprocess_cl(
            temp_root, median_filter_size=median_filter_size
        )
        energy_axis, _, _, _ = _build_energy_axis(processed_cl)

    return PreprocessResult(
        processed_cl=processed_cl,
        energy_axis=energy_axis,
        list_keys=list_keys,
        si_list=si_list,
    )


def _cache_scope_info(scope: str, folders) -> str:
    """Summarize cache scope in a compact string for logging."""
    if isinstance(folders, (list, tuple)):
        count = len(folders)
    else:
        count = 1
    return f"scope={scope}, folders={count}"


def _log_cache_hit(stage: str, scope: str, folders) -> None:
    """Log a concise cache-hit message for the given stage."""
    print(f"Cache hit: {stage} ({_cache_scope_info(scope, folders)})")


def _get_preprocess_result(
    *,
    cache_key: Tuple,
    source_folders: Sequence[str],
    labels: Sequence[str],
    median_filter_size: int,
    preprocess_cache: Optional[Dict[Tuple, object]] = None,
) -> PreprocessResult:
    """Get or populate preprocessing results in the shared cache."""
    # Reuse preprocessing across plots and full runs when parameters match.
    cached = preprocess_cache.get(cache_key) if preprocess_cache else None
    if cached is None:
        cached = _build_preprocess_result_for_folders(
            source_folders,
            labels,
            median_filter_size=median_filter_size,
        )
        if preprocess_cache is not None:
            preprocess_cache[cache_key] = cached
    else:
        scope = cache_key[1] if len(cache_key) > 1 else "unknown"
        folders = cache_key[2] if len(cache_key) > 2 else source_folders
        _log_cache_hit("preprocess", scope, folders)

    return cached


def _get_mask_flat_from_cache(
    *,
    preprocess_key: Tuple,
    preprocess_result: PreprocessResult,
    mask_enabled: bool,
    mask_dilation_iters: int,
    mask_mode: str,
    preprocess_cache: Optional[Dict[Tuple, object]] = None,
) -> Optional[np.ndarray]:
    """Get or build a flattened mask and store it in the shared cache."""
    if not mask_enabled:
        return None

    mask_key = (
        "mask",
        preprocess_key,
        mask_dilation_iters,
        mask_mode,
    )
    # Mask is cached separately so it can be invalidated without recomputing spectra.
    mask_flat = preprocess_cache.get(mask_key) if preprocess_cache else None
    if mask_flat is None:
        mask_flat = _build_mask_flat(
            preprocess_result.processed_cl,
            mask_enabled=mask_enabled,
            mask_dilation_iters=mask_dilation_iters,
            mask_mode=mask_mode,
        )
        if preprocess_cache is not None:
            preprocess_cache[mask_key] = mask_flat
    else:
        scope = preprocess_key[1] if len(preprocess_key) > 1 else "unknown"
        folders = preprocess_key[2] if len(preprocess_key) > 2 else []
        _log_cache_hit("mask", scope, folders)

    preprocess_result.mask_flat = mask_flat
    preprocess_result.mask_params = (
        mask_enabled,
        mask_dilation_iters,
        mask_mode,
    )
    return mask_flat


def _ranges_cache_key(
    fine_structure_ranges: Dict[str, Tuple[float, float]],
) -> Tuple[Tuple[str, float, float], ...]:
    """Normalize fine-structure ranges into a stable cache key."""
    return tuple(
        sorted(
            (k, float(v[0]), float(v[1]))
            for k, v in fine_structure_ranges.items()
        )
    )


def _colors_cache_key(
    colors: Sequence[np.ndarray],
) -> Tuple[Tuple[float, float, float], ...]:
    """Normalize a list of RGB colors into a stable cache key."""
    return tuple(tuple(float(x) for x in c) for c in colors)


def _array_signature(
    arr: Optional[np.ndarray],
) -> Optional[Tuple[str, Tuple[int, ...], str]]:
    """Build a hash signature for numpy arrays used in cache keys."""
    if arr is None:
        return None
    digest = hashlib.md5(arr.tobytes()).hexdigest()
    return (digest, arr.shape, str(arr.dtype))


def _expand_masked_components(
    values: np.ndarray,
    mask_flat: np.ndarray,
    shapes: Sequence[Tuple[int, int]],
) -> List[List[np.ndarray]]:
    """Restore masked component vectors back to 2D images."""
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


def build_scree_plot_inputs(
    source_folders: Sequence[str],
    *,
    E0_V: float,
    alpha_rad: float,
    beta_rad: float,
    median_filter_size: int,
    process_individual: bool,
    process_joint: bool,
    mask_enabled: bool = False,
    mask_dilation_iters: int = 1,
    mask_mode: str = "mean",
    preprocess_cache: Optional[Dict[Tuple, object]] = None,
) -> List[Tuple[str, object]]:
    """Prepare ListOfSI objects for scree plots using cached preprocessing."""
    if len(source_folders) == 0:
        raise ValueError("No source folders provided")

    labels = [os.path.basename(folder) for folder in source_folders]

    results: List[Tuple[str, object]] = []

    # Joint path aggregates all folders into a single ListOfSI.
    if process_joint:
        cache_key = (
            "preprocess",
            "joint",
            tuple(source_folders),
            median_filter_size,
        )
        cached = _get_preprocess_result(
            cache_key=cache_key,
            source_folders=source_folders,
            labels=labels,
            median_filter_size=median_filter_size,
            preprocess_cache=preprocess_cache,
        )

        mask_flat = _get_mask_flat_from_cache(
            preprocess_key=cache_key,
            preprocess_result=cached,
            mask_enabled=mask_enabled,
            mask_dilation_iters=mask_dilation_iters,
            mask_mode=mask_mode,
            preprocess_cache=preprocess_cache,
        )

        processed_cl = cached.processed_cl
        energy_axis = cached.energy_axis
        mask_flat = mask_flat if mask_enabled else None
        _, sl = _build_deco(
            processed_cl,
            energy_axis,
            E0_V=E0_V,
            alpha_rad=alpha_rad,
            beta_rad=beta_rad,
            mask_flat=mask_flat,
        )
        results.append(("Joint", sl))

    # Individual path keeps a ListOfSI per folder.
    if process_individual:
        for folder, label in zip(source_folders, labels):
            cache_key = (
                "preprocess",
                "individual",
                folder,
                median_filter_size,
            )
            cached = _get_preprocess_result(
                cache_key=cache_key,
                source_folders=[folder],
                labels=[label],
                median_filter_size=median_filter_size,
                preprocess_cache=preprocess_cache,
            )

            mask_flat = _get_mask_flat_from_cache(
                preprocess_key=cache_key,
                preprocess_result=cached,
                mask_enabled=mask_enabled,
                mask_dilation_iters=mask_dilation_iters,
                mask_mode=mask_mode,
                preprocess_cache=preprocess_cache,
            )

            processed_cl = cached.processed_cl
            energy_axis = cached.energy_axis
            mask_flat = mask_flat if mask_enabled else None
            _, sl = _build_deco(
                processed_cl,
                energy_axis,
                E0_V=E0_V,
                alpha_rad=alpha_rad,
                beta_rad=beta_rad,
                mask_flat=mask_flat,
            )
            results.append((label, sl))

    return results


def _compute_edge_contributions_summary(
    deco,
    fine_structure_ranges: Dict[str, Tuple[float, float]],
    *,
    component_order: Optional[Sequence[int]] = None,
) -> List[Dict[str, float]]:
    """Compute per-component edge contribution metrics from a decomposition."""
    if not hasattr(deco, "model") or not hasattr(deco.model, "_edge_slices"):
        return []

    edge_slices = deco.model._edge_slices
    X = np.asarray(deco.X)
    model = np.asarray(deco.get_model())
    normX = float((X**2).sum())
    Emodel = X - model

    n_components = int(getattr(deco, "n_components", 0))
    if n_components <= 0:
        return []

    loo_totals = np.zeros(n_components, dtype=float)
    power_totals = np.zeros(n_components, dtype=float)
    ev_totals = np.zeros(n_components, dtype=float)
    G = np.asarray(deco.G)
    W = np.asarray(deco.W)
    H = np.asarray(deco.H)

    for edge in fine_structure_ranges.keys():
        if edge not in edge_slices:
            continue
        idxs = edge_slices[edge]
        for k in range(n_components):
            wk = G[:, idxs] @ W[idxs, k]
            hk = H[k, :]

            wk_sq = np.sum(wk**2)
            hk_sq = np.sum(hk**2)
            comp_sq = wk_sq * hk_sq

            power_vals = comp_sq / normX if normX else 0.0

            interaction_X = (wk.T @ X) @ hk.T
            evk = (2 * interaction_X - comp_sq) / normX if normX else 0.0

            interaction_Emodel = wk.T @ Emodel @ hk.T
            Ck = (comp_sq + 2 * interaction_Emodel) / normX if normX else 0.0

            loo_totals[k] += Ck
            power_totals[k] += power_vals
            ev_totals[k] += evk

    totals = {
        "Leave-Out Loss": float(np.sum(loo_totals)),
        "Power": float(np.sum(power_totals)),
        "Explained Variance": float(np.sum(ev_totals)),
    }

    order = (
        list(component_order)
        if component_order is not None
        else list(range(n_components))
    )
    rows: List[Dict[str, float]] = []
    for idx, comp_idx in enumerate(order):
        rows.append(
            {
                "Component": idx,
                "Leave-Out Loss": float(loo_totals[comp_idx]),
                "Relative Leave-Out Loss (%)": (
                    100.0 * loo_totals[comp_idx] / totals["Leave-Out Loss"]
                    if totals["Leave-Out Loss"]
                    else 0.0
                ),
                "Power": float(power_totals[comp_idx]),
                "Relative Power (%)": (
                    100.0 * power_totals[comp_idx] / totals["Power"]
                    if totals["Power"]
                    else 0.0
                ),
                "Explained Variance": float(ev_totals[comp_idx]),
                "Relative Explained Variance (%)": (
                    100.0 * ev_totals[comp_idx] / totals["Explained Variance"]
                    if totals["Explained Variance"]
                    else 0.0
                ),
            }
        )

    return rows


def _order_from_summary_power(
    summary_rows: Sequence[Dict[str, float]],
    *,
    n_components: int,
) -> Optional[List[int]]:
    """Return component indices ordered by decreasing power contribution."""
    if not summary_rows:
        return None

    power_values = np.zeros(n_components, dtype=float)
    for row in summary_rows:
        comp_idx = row.get("Component")
        if comp_idx is None:
            continue
        try:
            comp_idx = int(comp_idx)
        except (TypeError, ValueError):
            continue
        if 0 <= comp_idx < n_components:
            power_values[comp_idx] = float(row.get("Power", 0.0))

    if not np.any(power_values):
        return None

    return np.argsort(power_values)[::-1].tolist()


def processing(
    path_to_temp: Optional[str],
    *,
    E0_V: float,
    alpha_rad: float,
    beta_rad: float,
    max_iters: int,
    median_filter_size: int,
    fine_structure_ranges: Dict[str, Tuple[float, float]],
    colors: Sequence[np.ndarray],
    n_components: int,
    W_init: Optional[np.ndarray] = None,
    compute_images: bool = True,
    mask_enabled: bool = False,
    mask_dilation_iters: int = 1,
    mask_mode: str = "mean",
    masks: Optional[Sequence[np.ndarray]] = None,
    preprocess_result: Optional[PreprocessResult] = None,
):
    """Run EELSNMF decomposition and return computed images and edges.

    If *masks* is provided, it must contain one 2-D boolean array per
    preprocessed spectral image and *mask_enabled* must be True. The masks
    are used directly and pixels where the mask is False are excluded from
    the decomposition.
    """
    # Stage 1: preprocessing (load + despike + axis) or reuse cached result.
    if preprocess_result is None:
        if path_to_temp is None:
            raise ValueError(
                "path_to_temp is required when preprocess_result is None"
            )
        SI_list, processed_cl, list_keys = _preprocess_cl(
            path_to_temp, median_filter_size=median_filter_size
        )
        energy_axis, scale, ei, ef = _build_energy_axis(processed_cl)
        print(f"Energy axis from {ei} eV to {ef} eV with step {scale:.4f} eV")
    else:
        processed_cl = preprocess_result.processed_cl
        energy_axis = preprocess_result.energy_axis
        list_keys = preprocess_result.list_keys
        SI_list = preprocess_result.si_list or {}
        if energy_axis.size >= 2:
            scale = float(energy_axis[1] - energy_axis[0])
            ei = float(energy_axis[0])
            ef = float(energy_axis[-1])
            print(
                f"Energy axis from {ei} eV to {ef} eV with step {scale:.4f} eV"
            )

    mask_images = None
    mask_flat = None
    mask_counts = None
    spatial_shapes = [
        tuple(np.asarray(cl.data).shape[:2]) for cl in processed_cl
    ]
    # Stage 2: build or reuse mask and apply it to the unfolded data.
    if mask_enabled:
        if masks is not None:
            # User-provided masks take precedence over auto-generated masks.
            if len(masks) != len(processed_cl):
                raise ValueError(
                    f"Number of masks ({len(masks)}) must match number of "
                    f"spectral images ({len(processed_cl)})"
                )
            mask_images = [np.asarray(m, dtype=bool) for m in masks]
            for idx, (mask, shape) in enumerate(
                zip(mask_images, spatial_shapes)
            ):
                if mask.shape != shape:
                    raise ValueError(
                        f"Mask shape {mask.shape} for image {idx} does not "
                        f"match spectral image shape {shape}"
                    )
            mask_flat = np.concatenate([m.ravel() for m in mask_images])
            mask_counts = [int(m.sum()) for m in mask_images]
        else:
            mask_params = (mask_enabled, mask_dilation_iters, mask_mode)
            cached_mask = (
                preprocess_result.mask_flat
                if preprocess_result is not None
                and preprocess_result.mask_params == mask_params
                else None
            )
            if cached_mask is not None and cached_mask.size > 0:
                try:
                    mask_images, mask_counts = _mask_flat_to_images(
                        cached_mask, spatial_shapes
                    )
                    mask_flat = cached_mask
                except ValueError:
                    mask_images = None
                    mask_counts = None
                    mask_flat = None

            if mask_flat is None:
                (
                    mask_images,
                    mask_flat,
                    mask_counts,
                    spatial_shapes,
                ) = _build_mask_data(
                    processed_cl,
                    dilation_iters=mask_dilation_iters,
                    mode=mask_mode,
                )
        if mask_flat.size == 0 or not np.any(mask_flat):
            raise ValueError("Mask removed all pixels")

    # Stage 3: build EELSNMF object and its ListOfSI wrapper.
    deco, sl = _build_deco(
        processed_cl,
        energy_axis,
        E0_V=E0_V,
        alpha_rad=alpha_rad,
        beta_rad=beta_rad,
        mask_flat=mask_flat,
    )

    # Stage 4: build the spectral model (G) while suppressing library output.
    with HiddenPrints():
        deco.build_G(
            fine_structure_ranges=fine_structure_ranges,
            backgrounds=np.linspace(0.5, 8, 50),
        )

    # Stage 5: run decomposition, optionally interpolating the provided W init.
    if W_init is not None and W_init.shape[0] == n_components:

        s = deco.G.shape[1]
        # linear interpolation at this shape
        range_old = np.linspace(0, 1, W_init.shape[0])
        range_new = np.linspace(0, 1, s)
        f = interp1d(range_old, W_init, axis=0)
        W_init = f(range_new)

        deco.decomposition(
            n_components=n_components,
            max_iters=max_iters,
            error_skip_step=100,
            tol=1e-8,
            use_cupy=USE_CUPY,
            W_init=W_init,
            rescale_G=False,
        )
    else:
        if W_init is not None and W_init.shape[0] != n_components:
            print(
                "W_init component count mismatch; ignoring W_init for this run"
            )
        deco.decomposition(
            n_components=n_components,
            max_iters=max_iters,
            error_skip_step=100,
            tol=1e-8,
            use_cupy=USE_CUPY,
            rescale_G=False,
        )

    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    if len(colors) < n_components:
        raise ValueError(
            f"Need at least {n_components} colors; got {len(colors)}"
        )

    edge_chunks = []
    edges_by_label: Dict[str, EdgeSeries] = {}

    for edge_label in fine_structure_ranges.keys():
        ii, ff = find_index(
            deco.energy_axis, deco.fine_structure_ranges[edge_label]
        )
        x_axis_chunk = deco.energy_axis[ii:ff]

        raw_chunks: List[np.ndarray] = []
        smoothed_chunks: List[np.ndarray] = []
        norm_chunks: List[np.ndarray] = []

        for comp_idx in range(n_components):
            edge_raw = deco.get_edge_from_component(comp_idx, edge_label).data
            edge_smoothed = EML.we_smooth(edge_raw.copy())
            edge_smoothed_chunk = edge_smoothed[ii:ff]
            denom = edge_smoothed_chunk.max() - edge_smoothed_chunk.min()
            if denom == 0:
                edge_norm_chunk = np.zeros_like(edge_smoothed_chunk)
            else:
                edge_norm_chunk = (
                    edge_smoothed_chunk - edge_smoothed_chunk.min()
                ) / denom

            raw_chunks.append(edge_raw[ii:ff])
            smoothed_chunks.append(edge_smoothed_chunk)
            norm_chunks.append(edge_norm_chunk)

        edge_chunks.append(
            {
                "x_axis": x_axis_chunk,
                "raw": raw_chunks,
                "smoothed": smoothed_chunks,
                "norm": norm_chunks,
            }
        )
        edges_by_label[edge_label] = EdgeSeries(
            x=x_axis_chunk,
            raw=raw_chunks,
            smoothed=smoothed_chunks,
            norm=norm_chunks,
        )

    edge_chunks.sort(
        key=lambda item: (
            item["x_axis"][0] if len(item["x_axis"]) > 0 else np.inf
        )
    )

    x_axis_values = _concat_arrays([c["x_axis"] for c in edge_chunks])
    edge_raw = [
        _concat_arrays([c["raw"][i] for c in edge_chunks])
        for i in range(n_components)
    ]
    edge_smoothed = [
        _concat_arrays([c["smoothed"][i] for c in edge_chunks])
        for i in range(n_components)
    ]
    edge_norm = [
        _concat_arrays([c["norm"][i] for c in edge_chunks])
        for i in range(n_components)
    ]

    ims_components = None

    # Stage 6: compute spatial maps and component images if requested.
    if compute_images:

        ### This part masks energy values outside the fine structure ranges
        edge_slices = deco.model._edge_slices
        idxs_list = [
            edge_slices[edge]
            for edge in fine_structure_ranges.keys()
            if edge in edge_slices
        ]
        if idxs_list:
            weights = np.zeros(deco.W.shape[1], dtype=float)
            for slc in idxs_list:
                weights += (deco.G[:, slc] @ deco.W[slc, :]).sum(0)
        else:
            weights = (deco.G @ deco.W).sum(0)

        normalized_H = deco.H * weights[:, None]
        if len(sl.spatial_dim_list) > 1:
            normalized_H = normalized_H.copy()
            filled = 0
            if mask_enabled and mask_counts is not None:
                for count in mask_counts:
                    sample_chunk = normalized_H[:, filled : filled + count]
                    sample_max = np.max(sample_chunk, axis=1)
                    denom = float(np.max(sample_max))
                    if denom > 0:
                        normalized_H[:, filled : filled + count] = (
                            sample_chunk / denom
                        )
                    filled += count
            else:
                for n in sl.spatial_dim_list:
                    sample_chunk = normalized_H[:, filled : filled + n]
                    sample_max = np.max(sample_chunk, axis=1)
                    denom = float(np.max(sample_max))
                    if denom > 0:
                        normalized_H[:, filled : filled + n] = (
                            sample_chunk / denom
                        )
                    filled += n
        else:  # one element in sl.spatial_dim_list, so individual processing. or also for joint processing ?...
            normalized_H = normalized_H / max(
                normalized_H.max(axis=1).tolist()
            )

        if mask_enabled and mask_flat is not None:
            ims_components = _expand_masked_components(
                normalized_H,
                mask_flat,
                spatial_shapes,
            )
        else:
            ims_components = [
                sl.fold_array(normalized_H[i]) for i in range(n_components)
            ]

    summary_pre = _compute_edge_contributions_summary(
        deco,
        fine_structure_ranges,
        component_order=None,
    )
    order = _order_from_summary_power(
        summary_pre,
        n_components=n_components,
    ) or list(range(n_components))
    if order != list(range(n_components)):
        edge_raw = [edge_raw[i] for i in order]
        edge_smoothed = [edge_smoothed[i] for i in order]
        edge_norm = [edge_norm[i] for i in order]
        if compute_images and ims_components is not None:
            ims_components = [ims_components[i] for i in order]
        edges_by_label = {
            label: EdgeSeries(
                x=edge.x,
                raw=[edge.raw[i] for i in order],
                smoothed=[edge.smoothed[i] for i in order],
                norm=[edge.norm[i] for i in order],
            )
            for label, edge in edges_by_label.items()
        }

    ims_color = None
    ims_original = None
    if compute_images and ims_components is not None:
        ims_color = []
        n_images = len(ims_components[0]) if ims_components else 0
        for idx in range(n_images):
            rgb = None
            for comp_idx in range(n_components):
                img = ims_components[comp_idx][idx]
                color = np.asarray(colors[comp_idx])
                contribution = img[..., None] * color
                rgb = contribution if rgb is None else rgb + contribution
            ims_color.append(np.clip(rgb, 0.0, 1.0))

        ims_components = [
            [
                np.clip(img[..., None] * np.asarray(colors[i]), 0.0, 1.0)
                for img in comp_imgs
            ]
            for i, comp_imgs in enumerate(ims_components)
        ]

        ims_original = (
            [SI_list[k][0][-1].data.mean(-1) for k in SI_list.keys()]
            if SI_list
            else None
        )

        if mask_images is not None:
            for idx, mask in enumerate(mask_images):
                if ims_original is not None:
                    ims_original[idx] = np.where(mask, ims_original[idx], 0.0)
                ims_color[idx] = np.where(mask[..., None], ims_color[idx], 0.0)
                for comp_idx in range(n_components):
                    ims_components[comp_idx][idx] = np.where(
                        mask[..., None],
                        ims_components[comp_idx][idx],
                        0.0,
                    )

    pixel_scale = None
    if len(SI_list) > 0:
        first_key = next(iter(SI_list.keys()))
        pixel_scale = (
            SI_list[first_key][0][-1].axes_manager.navigation_axes[0].scale
        )

    combined_edges = EdgeSeries(
        x=x_axis_values,
        raw=edge_raw,
        smoothed=edge_smoothed,
        norm=edge_norm,
    )

    edge_contributions_summary = _compute_edge_contributions_summary(
        deco,
        fine_structure_ranges,
        component_order=order,
    )

    return ProcessingResult(
        images=ImageCollection(
            mixed=ims_color,
            original=ims_original,
            components=ims_components,
            mask=mask_images,
        ),
        edges=EdgeCollection(
            combined=combined_edges,
            by_label=edges_by_label,
        ),
        pixel_scale=pixel_scale,
        energy_axis=deco.energy_axis,
        list_keys=list_keys,
        W=deco.W.copy(),
        deco=deco,
        edge_contributions_summary=edge_contributions_summary,
    )


def process_folders(
    source_folders: Sequence[str],
    *,
    E0_V: float,
    alpha_rad: float,
    beta_rad: float,
    max_iters: int,
    median_filter_size: int,
    fine_structure_ranges: Dict[str, Tuple[float, float]],
    colors: Sequence[np.ndarray],
    n_components: int = 2,
    process_individual: bool = True,
    process_joint: bool = True,
    transmit_W_init: bool = True,
    mask_enabled: bool = False,
    mask_dilation_iters: int = 1,
    mask_mode: str = "mean",
    masks: Optional[Sequence[np.ndarray]] = None,
    preprocess_cache: Optional[Dict[Tuple, object]] = None,
):
    """Process one or more folders and return joint/individual results.

    If *masks* is provided, it must contain one 2-D boolean array per source
    folder (in the same order as *source_folders*) and *mask_enabled* must be
    True. The provided masks are used directly instead of auto-generating them.
    """
    if not (process_individual or process_joint):
        raise ValueError("Select at least one processing mode")

    labels = [os.path.basename(folder) for folder in source_folders]

    list_labels = []
    list_results: List[ProcessingResult] = []
    labels_joint = labels

    deco_joint = None
    list_decos_individual = []

    W_init = None

    joint_result: Optional[ProcessingResult] = None

    # Execute a single processing run with caching for this folder set.
    def _run_processing_for_folders(
        folders: Sequence[str],
        folder_labels: Sequence[str],
        *,
        W_init_local: Optional[np.ndarray],
        folder_masks: Optional[Sequence[np.ndarray]] = None,
    ) -> ProcessingResult:
        scope = "joint" if len(folders) > 1 else "individual"
        preprocess_key = (
            "preprocess",
            scope,
            tuple(folders) if scope == "joint" else folders[0],
            median_filter_size,
        )
        preprocess_result = _get_preprocess_result(
            cache_key=preprocess_key,
            source_folders=folders,
            labels=folder_labels,
            median_filter_size=median_filter_size,
            preprocess_cache=preprocess_cache,
        )

        if folder_masks is not None:
            # Use user-provided masks directly.
            if len(folder_masks) != len(folders):
                raise ValueError(
                    f"Number of masks ({len(folder_masks)}) must match number "
                    f"of folders ({len(folders)})"
                )
            mask_flat = np.concatenate(
                [np.asarray(m, dtype=bool).ravel() for m in folder_masks]
            )
            preprocess_result.mask_flat = mask_flat
            preprocess_result.mask_params = (
                mask_enabled,
                mask_dilation_iters,
                mask_mode,
            )
            masks_sig = _array_signature(mask_flat)
        else:
            mask_flat = _get_mask_flat_from_cache(
                preprocess_key=preprocess_key,
                preprocess_result=preprocess_result,
                mask_enabled=mask_enabled,
                mask_dilation_iters=mask_dilation_iters,
                mask_mode=mask_mode,
                preprocess_cache=preprocess_cache,
            )
            masks_sig = None

        ranges_key = _ranges_cache_key(fine_structure_ranges)
        colors_key = _colors_cache_key(colors)
        w_init_sig = _array_signature(W_init_local)
        run_key = (
            "run",
            scope,
            tuple(folders) if scope == "joint" else folders[0],
            E0_V,
            alpha_rad,
            beta_rad,
            max_iters,
            median_filter_size,
            ranges_key,
            n_components,
            mask_enabled,
            mask_dilation_iters,
            mask_mode,
            masks_sig,
            colors_key,
            w_init_sig,
        )
        cached_run = (
            preprocess_cache.get(run_key) if preprocess_cache else None
        )
        if cached_run is not None:
            _log_cache_hit("run", scope, folders)
            return cached_run

        # Store mask in the preprocessing result so downstream steps can reuse it.
        if mask_enabled and preprocess_result is not None and folder_masks is None:
            preprocess_result.mask_flat = mask_flat
            preprocess_result.mask_params = (
                mask_enabled,
                mask_dilation_iters,
                mask_mode,
            )

        result = processing(
            None,
            E0_V=E0_V,
            alpha_rad=alpha_rad,
            beta_rad=beta_rad,
            max_iters=max_iters,
            median_filter_size=median_filter_size,
            fine_structure_ranges=fine_structure_ranges,
            colors=colors,
            n_components=n_components,
            W_init=W_init_local,
            compute_images=True,
            mask_enabled=mask_enabled,
            mask_dilation_iters=mask_dilation_iters,
            mask_mode=mask_mode,
            masks=folder_masks,
            preprocess_result=preprocess_result,
        )
        if preprocess_cache is not None:
            preprocess_cache[run_key] = result
        return result

    # Joint run first to seed W_init for individual runs if enabled.
    if process_joint:
        joint_result = _run_processing_for_folders(
            source_folders,
            labels,
            W_init_local=None,
            folder_masks=masks,
        )

        list_keys = joint_result.list_keys
        joint_summary = joint_result.edge_contributions_summary
        W_init = joint_result.W
        deco_joint = joint_result.deco

        mapped_mixed, mapped_original = _map_joint_images(
            joint_result.images.mixed,
            joint_result.images.original,
            list_keys,
            labels,
        )
        mapped_components = None
        if joint_result.images.components is not None:
            mapped_components = [
                _map_joint_image_list(component, list_keys, labels)
                for component in joint_result.images.components
            ]
        mapped_masks = None
        if joint_result.images.mask is not None:
            mapped_masks = _map_joint_image_list(
                joint_result.images.mask,
                list_keys,
                labels,
            )

        joint_result = ProcessingResult(
            images=ImageCollection(
                mixed=mapped_mixed,
                original=mapped_original,
                components=mapped_components,
                mask=mapped_masks,
            ),
            edges=joint_result.edges,
            pixel_scale=joint_result.pixel_scale,
            energy_axis=joint_result.energy_axis,
            list_keys=joint_result.list_keys,
            W=joint_result.W,
            deco=joint_result.deco,
            edge_contributions_summary=joint_summary,
        )

    # Individual runs optionally reuse W_init from the joint decomposition.
    if process_individual:
        W_init = W_init if transmit_W_init else None
        print(
            f"transmit_W_init = {transmit_W_init}, W_init = {W_init is not None}"
        )
        for idx, (folder, label) in enumerate(zip(source_folders, labels)):
            folder_masks = [masks[idx]] if masks is not None else None
            result = _run_processing_for_folders(
                [folder],
                [label],
                W_init_local=W_init,
                folder_masks=folder_masks,
            )

            list_labels.append(label)
            list_results.append(result)
            list_decos_individual.append(result.deco)

    return ProcessFoldersResult(
        individual=list_results,
        joint=joint_result,
        labels=list_labels,
        labels_joint=labels_joint,
        deco_joint=deco_joint,
        decos_individual=list_decos_individual,
    )


def build_deco_for_folders_to_plot_er(
    source_folders: Sequence[str],
    *,
    E0_V: float,
    alpha_rad: float,
    beta_rad: float,
    median_filter_size: int,
    fine_structure_ranges: Dict[str, Tuple[float, float]],
    mask_enabled: bool = False,
    mask_dilation_iters: int = 1,
    mask_mode: str = "mean",
    preprocess_cache: Optional[Dict[Tuple, object]] = None,
):
    """Build EELSNMF objects with G for energy-range plots."""
    if len(source_folders) == 0:
        raise ValueError("No source folders provided")
    labels = [os.path.basename(folder) for folder in source_folders]
    list_decos = []

    for folder, label in zip(source_folders, labels):
        cache_key = ("preprocess", "individual", folder, median_filter_size)
        cached = _get_preprocess_result(
            cache_key=cache_key,
            source_folders=[folder],
            labels=[label],
            median_filter_size=median_filter_size,
            preprocess_cache=preprocess_cache,
        )

        mask_flat = _get_mask_flat_from_cache(
            preprocess_key=cache_key,
            preprocess_result=cached,
            mask_enabled=mask_enabled,
            mask_dilation_iters=mask_dilation_iters,
            mask_mode=mask_mode,
            preprocess_cache=preprocess_cache,
        )

        processed_cl = cached.processed_cl
        energy_axis = cached.energy_axis
        mask_flat = mask_flat if mask_enabled else None
        deco, _ = _build_deco(
            processed_cl,
            energy_axis,
            E0_V=E0_V,
            alpha_rad=alpha_rad,
            beta_rad=beta_rad,
            mask_flat=mask_flat,
        )

        with HiddenPrints():
            deco.build_G(
                fine_structure_ranges=fine_structure_ranges,
                backgrounds=np.linspace(0.5, 8, 50),
            )

        list_decos.append(deco)

    return list_decos


def build_mask_preview_for_folders(
    source_folders: Sequence[str],
    *,
    median_filter_size: int,
    dilation_iters: int,
    mode: str = "mean",
    preprocess_cache: Optional[Dict[Tuple, object]] = None,
):
    """Generate aggregated images and masks for a quick preview."""
    if len(source_folders) == 0:
        raise ValueError("No source folders provided")
    labels = [os.path.basename(folder) for folder in source_folders]

    previews: List[Tuple[str, np.ndarray, np.ndarray]] = []

    for folder, label in zip(source_folders, labels):
        cache_key = ("preprocess", "individual", folder, median_filter_size)
        cached = _get_preprocess_result(
            cache_key=cache_key,
            source_folders=[folder],
            labels=[label],
            median_filter_size=median_filter_size,
            preprocess_cache=preprocess_cache,
        )

        if not cached.processed_cl:
            continue
        for key, cl in zip(cached.list_keys, cached.processed_cl):
            data = np.asarray(cl.data)
            aggregated = _aggregate_energy_axis(data, mode)
            mask = _build_threshold_mask(
                aggregated,
                dilation_iters=dilation_iters,
            )
            display_label = os.path.basename(key) or label
            previews.append((display_label, aggregated, mask))

    return previews
