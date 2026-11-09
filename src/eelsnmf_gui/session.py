from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import (
    Dict,
    List,
    Protocol,
    Tuple,
)

import numpy as np

from ._analysis import (
    apply_colors,
    component_maps,
    edge_totals,
    extract_edge_collection,
    map_images_to_labels,
    order_by_power,
    summary_rows,
    zero_outside_masks,
)
from ._preprocess import (
    Loader,
    aggregate_energy,
    build_mask_result,
    common_energy_axis,
    load_eml_folders,
    threshold_mask,
)
from ._results import (
    BeamParams,
    DecoBundle,
    ImageCollection,
    MaskParams,
    MaskResult,
    PreprocessedData,
    PreprocessedSI,
    ProcessFoldersResult,
    ProcessingResult,
)
from ._util import array_signature

logger = logging.getLogger(__name__)


class NmfEngine(Protocol):
    """Minimal interface the Session needs from an NMF backend."""

    def list_of_si(
        self, sis: Sequence[PreprocessedSI], energy_axis: np.ndarray
    ): ...

    def build_deco(
        self,
        sis: Sequence[PreprocessedSI],
        energy_axis: np.ndarray,
        *,
        E0_V: float,
        alpha_rad: float,
        beta_rad: float,
        mask_flat: np.ndarray | None = None,
    ) -> DecoBundle: ...

    def build_g(
        self,
        bundle: DecoBundle,
        fine_structure_ranges: Dict[str, Tuple[float, float]],
        *,
        n_power_laws: int = 50,
    ) -> None: ...

    def decompose(
        self,
        bundle: DecoBundle,
        *,
        n_components: int,
        max_iters: int,
        w_init: np.ndarray | None = None,
        use_cupy: bool | None = None,
        tol: float = 1e-8,
        error_skip_step: int = 100,
        decomposition_method: str = "default_decomposition",
        regularization_tv: float | None = None,
        regularization_sr: float | None = None,
        fixed_weights: Tuple[Tuple[str, int, str], ...] | None = None,
    ) -> None: ...


def _default_engine() -> NmfEngine:
    # Imported lazily so that importing this module does not pull EELSNMF in.
    from ._nmf import EmlEngine

    return EmlEngine()


def _ranges_key(
    fine_structure_ranges: Dict[str, Tuple[float, float]],
) -> Tuple[Tuple[str, float, float], ...]:
    """Normalize fine-structure ranges into a stable cache key."""
    return tuple(
        sorted(
            (k, float(v[0]), float(v[1]))
            for k, v in fine_structure_ranges.items()
        )
    )


def _fixed_weights_key(
    fixed_weights: Sequence[Tuple[str, int, str]] | None,
) -> Tuple[Tuple[str, int, str], ...]:
    """Normalize fixed-weight constraints into a stable cache key."""
    if not fixed_weights:
        return ()
    return tuple(
        sorted((str(edge), int(comp), str(mode)) for edge, comp, mode in fixed_weights)
    )


@dataclass(frozen=True)
class DecompositionSpec:
    """Identity of a decomposition, used to reuse fitted models."""

    beam: BeamParams
    ranges: Tuple[Tuple[str, float, float], ...]
    mask: MaskParams | None
    mask_signature: tuple | None
    n_components: int
    max_iters: int
    w_init_signature: tuple | None
    use_cupy: bool | None
    n_power_laws: int
    tol: float
    error_skip_step: int
    decomposition_method: str
    regularization_tv: float | None
    regularization_sr: float | None
    fixed_weights: tuple


class Session:
    """Owns folders and parameters, caches pipeline stages, self-invalidates.

    Stage accessors (``preprocessed``, ``mask_result``, ``run``, ...) return
    cached results whenever their inputs did not change:

    - changing folders or the median filter size invalidates everything;
    - auto masks are cached per :class:`MaskParams`;
    - fitted decompositions are cached per :class:`DecompositionSpec`
      (colors are *not* part of the spec: recoloring only re-renders).

    The GUI may call ``run`` from a worker thread while diagnostic accessors
    run on the UI thread; like the previous implementation, thread safety
    relies on GIL-level guarantees only.
    """

    def __init__(
        self,
        folders: Sequence[str],
        *,
        median_filter_size: int = 11,
        loader: Loader | None = None,
        engine: NmfEngine | None = None,
        progress: Callable[[str], None] | None = None,
    ):
        if len(folders) == 0:
            raise ValueError("No source folders provided")
        self._folders = tuple(folders)
        self._labels = [os.path.basename(folder) for folder in self._folders]
        self._median_filter_size = int(median_filter_size)
        self._loader: Loader = loader if loader is not None else load_eml_folders
        self._engine: NmfEngine = engine if engine is not None else _default_engine()
        self._progress = progress

        self._pre: PreprocessedData | None = None
        self._masks: Dict[MaskParams, MaskResult] = {}
        self._decompositions: Dict[str, Tuple[DecompositionSpec, DecoBundle]] = {}

    # -- parameters and lifecycle ------------------------------------------

    @property
    def folders(self) -> Tuple[str, ...]:
        return self._folders

    @property
    def labels(self) -> List[str]:
        return list(self._labels)

    @property
    def median_filter_size(self) -> int:
        return self._median_filter_size

    def set_folders(self, folders: Sequence[str]) -> None:
        if len(folders) == 0:
            raise ValueError("No source folders provided")
        self._folders = tuple(folders)
        self._labels = [os.path.basename(folder) for folder in self._folders]
        self._invalidate_all()

    def set_median_filter_size(self, size: int) -> None:
        size = int(size)
        if size == self._median_filter_size:
            return
        self._median_filter_size = size
        self._invalidate_all()

    def _invalidate_all(self) -> None:
        self._pre = None
        self._masks.clear()
        self._decompositions.clear()

    def _log(self, message: str) -> None:
        logger.info(message)
        if self._progress is not None:
            self._progress(message)

    # -- stage 1: preprocessing ---------------------------------------------

    def preprocessed(self) -> PreprocessedData:
        """Load and preprocess all SIs (cached until parameters change)."""
        if self._pre is None:
            self._log(
                f"Preprocessing {len(self._folders)} folder(s) with median "
                f"filter size {self._median_filter_size}..."
            )
            sis = self._loader(
                self._folders,
                self._labels,
                median_filter_size=self._median_filter_size,
            )
            folder_index = []
            for si in sis:
                folder_index.append(
                    self._labels.index(si.label)
                    if si.label in self._labels
                    else 0
                )
            self._pre = PreprocessedData(
                sis=sis,
                folder_index=folder_index,
                labels=list(self._labels),
                energy_axis=self._common_axis(sis),
            )
            self._log("Preprocessing done.")
        else:
            self._log("Cache hit: preprocessing")
        return self._pre

    @staticmethod
    def _common_axis(sis: Sequence[PreprocessedSI]) -> np.ndarray:
        return common_energy_axis(
            [si.energy_offset for si in sis],
            [si.energy_scale for si in sis],
            [si.n_energy for si in sis],
        )

    # -- masks ----------------------------------------------------------------

    def mask_result(self, mask: MaskParams) -> MaskResult | None:
        """Auto-built masks over all SIs (None when masking is disabled)."""
        if not mask.enabled:
            return None
        cached = self._masks.get(mask)
        if cached is None:
            self._log(
                f"Building masks ({mask.dilation_iters} dilation iters, "
                f"mode={mask.mode})..."
            )
            cached = build_mask_result(self.preprocessed().sis, mask)
            self._masks[mask] = cached
        else:
            self._log("Cache hit: masks")
        return cached

    def mask_preview(
        self, mask: MaskParams
    ) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """(label, aggregated image, mask) per SI for a quick preview."""
        previews: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for si in self.preprocessed().sis:
            aggregated = aggregate_energy(si.data, mask.mode)
            preview_mask = threshold_mask(
                aggregated, dilation_iters=mask.dilation_iters
            )
            display_label = os.path.basename(si.key) or si.label
            previews.append((display_label, aggregated, preview_mask))
        return previews

    # -- stage 2: diagnostic model objects ------------------------------------

    def _folder_sis(self, pre: PreprocessedData, folder_idx: int):
        sis = [
            si
            for si, fi in zip(pre.sis, pre.folder_index)
            if fi == folder_idx
        ]
        indices = [
            j
            for j, fi in enumerate(pre.folder_index)
            if fi == folder_idx
        ]
        return sis, indices

    def list_of_si(
        self,
        *,
        individual: bool = True,
        joint: bool = True,
    ):
        """(label, ListOfSI) pairs for scree plots (freshly built)."""
        if not (individual or joint):
            raise ValueError("Select at least one processing mode")
        pre = self.preprocessed()
        items = []
        if joint:
            items.append(
                ("Joint", self._engine.list_of_si(pre.sis, pre.energy_axis))
            )
        if individual:
            for folder_idx, label in enumerate(pre.labels):
                sis, _ = self._folder_sis(pre, folder_idx)
                items.append((label, self._engine.list_of_si(sis, self._common_axis(sis))))
        return items

    def decos_with_g(
        self,
        beam: BeamParams,
        fine_structure_ranges: Dict[str, Tuple[float, float]],
        mask: MaskParams,
        *,
        n_power_laws: int = 50,
    ) -> List[object]:
        """Per-folder EELSNMF objects with G built (for energy-range plots)."""
        if len(fine_structure_ranges) == 0:
            raise ValueError("Select at least one edge")
        pre = self.preprocessed()
        mask_result = self.mask_result(mask)
        decos = []
        for folder_idx in range(len(pre.labels)):
            sis, indices = self._folder_sis(pre, folder_idx)
            mask_flat = self._mask_flat_for(mask_result, indices)
            bundle = self._engine.build_deco(
                sis,
                self._common_axis(sis),
                E0_V=beam.E0_V,
                alpha_rad=beam.alpha_rad,
                beta_rad=beam.beta_rad,
                mask_flat=mask_flat,
            )
            self._engine.build_g(
                bundle, fine_structure_ranges, n_power_laws=n_power_laws
            )
            decos.append(bundle.deco)
        return decos

    # -- stage 3: decomposition -------------------------------------------------

    def run(
        self,
        *,
        beam: BeamParams,
        fine_structure_ranges: Dict[str, Tuple[float, float]],
        n_components: int,
        max_iters: int,
        mask: MaskParams | None = None,
        colors: Sequence[np.ndarray] = (),
        process_individual: bool = True,
        process_joint: bool = True,
        transmit_w_init: bool = True,
        user_masks: Sequence[np.ndarray] | None = None,
        use_cupy: bool | None = None,
        n_power_laws: int = 50,
        tol: float = 1e-8,
        error_skip_step: int = 100,
        decomposition_method: str = "default_decomposition",
        regularization_tv: float | None = None,
        regularization_sr: float | None = None,
        fixed_weights: Sequence[Tuple[str, int, str]] | None = None,
        force: bool = False,
    ) -> ProcessFoldersResult:
        """Run joint and/or individual NMF decompositions.

        *user_masks* (optional) must contain one 2D boolean mask per folder;
        the mask is applied to every SI of that folder and requires
        ``mask.enabled``. Decompositions are cached per
        :class:`DecompositionSpec` unless *force* is set. *use_cupy*
        selects the backend (None auto-detects a CUDA device).
        *n_power_laws* sizes the power-law background grid in G; *tol*,
        *error_skip_step* and *decomposition_method* are forwarded to the
        EELSNMF decomposition. *regularization_tv* is the Total-Variation
        lambda (EdgeTV, combinedTVSR) and *regularization_sr* the SumRule
        lambda (SumRule, combinedTVSR); each method receives exactly the
        lambdas it accepts. *fixed_weights* lists (edge, component, mode)
        constraints: mode "fixed" freezes the edge's cross-section weight
        in that component at its initial value, mode "excluded" also
        zero-fixes its fine-structure channels so the element cannot
        appear in that component.
        """
        mask = MaskParams() if mask is None else mask
        if not (process_individual or process_joint):
            raise ValueError("Select at least one processing mode")
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        if len(colors) < n_components:
            raise ValueError(
                f"Need at least {n_components} colors; got {len(colors)}"
            )
        if len(fine_structure_ranges) == 0:
            raise ValueError("Select at least one edge")

        pre = self.preprocessed()
        mask_result, using_user_masks = self._resolve_masks(
            mask, user_masks, pre
        )

        joint_result: ProcessingResult | None = None
        w_init: np.ndarray | None = None
        if process_joint:
            joint_result = self._run_scope(
                scope="joint",
                sis=pre.sis,
                energy_axis=pre.energy_axis,
                mask_result=mask_result,
                using_user_masks=using_user_masks,
                beam=beam,
                fine_structure_ranges=fine_structure_ranges,
                n_components=n_components,
                max_iters=max_iters,
                w_init=None,
                colors=colors,
                use_cupy=use_cupy,
                n_power_laws=n_power_laws,
                tol=tol,
                error_skip_step=error_skip_step,
                decomposition_method=decomposition_method,
                regularization_tv=regularization_tv,
                regularization_sr=regularization_sr,
                fixed_weights=fixed_weights,
                force=force,
            )
            w_init = joint_result.W

        individual_results: List[ProcessingResult] = []
        decos_individual: List[object] = []
        processed_labels: List[str] = []
        if process_individual:
            shared_w_init = w_init if transmit_w_init else None
            self._log(
                f"transmit_w_init = {transmit_w_init}, "
                f"W_init = {shared_w_init is not None}"
            )
            for folder_idx, label in enumerate(pre.labels):
                sis, indices = self._folder_sis(pre, folder_idx)
                folder_mask = self._slice_mask_result(mask_result, indices)
                result = self._run_scope(
                    scope=f"individual:{folder_idx}",
                    sis=sis,
                    energy_axis=self._common_axis(sis),
                    mask_result=folder_mask,
                    using_user_masks=using_user_masks,
                    beam=beam,
                    fine_structure_ranges=fine_structure_ranges,
                    n_components=n_components,
                    max_iters=max_iters,
                    w_init=shared_w_init,
                    colors=colors,
                    use_cupy=use_cupy,
                    n_power_laws=n_power_laws,
                    tol=tol,
                    error_skip_step=error_skip_step,
                    decomposition_method=decomposition_method,
                    regularization_tv=regularization_tv,
                    regularization_sr=regularization_sr,
                    fixed_weights=fixed_weights,
                    force=force,
                )
                individual_results.append(result)
                decos_individual.append(result.deco)
                processed_labels.append(label)

        if joint_result is not None:
            joint_result = self._map_joint_result(joint_result, pre)

        return ProcessFoldersResult(
            individual=individual_results,
            joint=joint_result,
            labels=processed_labels,
            labels_joint=list(self._labels),
            deco_joint=joint_result.deco if joint_result is not None else None,
            decos_individual=decos_individual,
        )

    # -- internals ------------------------------------------------------------

    def _resolve_masks(
        self,
        mask: MaskParams,
        user_masks: Sequence[np.ndarray] | None,
        pre: PreprocessedData,
    ) -> Tuple[MaskResult | None, bool]:
        """Resolve the mask for a run: user masks take precedence."""
        if user_masks is not None:
            if not mask.enabled:
                raise ValueError(
                    "user_masks requires mask.enabled=True"
                )
            per_si_masks = self._broadcast_user_masks(user_masks, pre)
            flat = np.concatenate([m.ravel() for m in per_si_masks])
            if flat.size == 0 or not np.any(flat):
                raise ValueError("Mask removed all pixels")
            mask_result = MaskResult(
                params=mask,
                per_si=per_si_masks,
                flat=flat,
                counts=[int(m.sum()) for m in per_si_masks],
                shapes=[m.shape[:2] for m in per_si_masks],
            )
            return mask_result, True

        mask_result = self.mask_result(mask)
        if mask_result is not None and (
            mask_result.flat.size == 0 or not np.any(mask_result.flat)
        ):
            raise ValueError("Mask removed all pixels")
        return mask_result, False

    @staticmethod
    def _broadcast_user_masks(
        user_masks: Sequence[np.ndarray], pre: PreprocessedData
    ) -> List[np.ndarray]:
        """Expand one mask per folder into one mask per SI."""
        if len(user_masks) != len(pre.labels):
            raise ValueError(
                f"Number of masks ({len(user_masks)}) must match number "
                f"of folders ({len(pre.labels)})"
            )
        per_si_masks: List[np.ndarray] = []
        for si, folder_idx in zip(pre.sis, pre.folder_index):
            mask = np.asarray(user_masks[folder_idx], dtype=bool)
            if mask.shape != si.spatial_shape:
                raise ValueError(
                    f"Mask shape {mask.shape} for folder "
                    f"'{pre.labels[folder_idx]}' does not match spectrum "
                    f"image shape {si.spatial_shape}"
                )
            per_si_masks.append(mask)
        return per_si_masks

    @staticmethod
    def _mask_flat_for(
        mask_result: MaskResult | None, indices: Sequence[int]
    ) -> np.ndarray | None:
        if mask_result is None:
            return None
        return np.concatenate(
            [mask_result.per_si[j].ravel() for j in indices]
        )

    @staticmethod
    def _slice_mask_result(
        mask_result: MaskResult | None, indices: Sequence[int]
    ) -> MaskResult | None:
        if mask_result is None:
            return None
        per_si = [mask_result.per_si[j] for j in indices]
        return MaskResult(
            params=mask_result.params,
            per_si=per_si,
            flat=np.concatenate([m.ravel() for m in per_si]),
            counts=[int(m.sum()) for m in per_si],
            shapes=[mask_result.shapes[j] for j in indices],
        )

    def _run_scope(
        self,
        *,
        scope: str,
        sis: Sequence[PreprocessedSI],
        energy_axis: np.ndarray,
        mask_result: MaskResult | None,
        using_user_masks: bool,
        beam: BeamParams,
        fine_structure_ranges: Dict[str, Tuple[float, float]],
        n_components: int,
        max_iters: int,
        w_init: np.ndarray | None,
        colors: Sequence[np.ndarray],
        use_cupy: bool | None,
        n_power_laws: int,
        tol: float,
        error_skip_step: int,
        decomposition_method: str,
        regularization_tv: float | None,
        regularization_sr: float | None,
        fixed_weights: Sequence[Tuple[str, int, str]] | None,
        force: bool,
    ) -> ProcessingResult:
        spec = DecompositionSpec(
            beam=beam,
            ranges=_ranges_key(fine_structure_ranges),
            mask=None if using_user_masks else self._spec_mask(mask_result),
            mask_signature=(
                array_signature(mask_result.flat)
                if using_user_masks and mask_result is not None
                else None
            ),
            n_components=n_components,
            max_iters=max_iters,
            w_init_signature=array_signature(w_init),
            use_cupy=use_cupy,
            n_power_laws=n_power_laws,
            tol=tol,
            error_skip_step=error_skip_step,
            decomposition_method=decomposition_method,
            regularization_tv=regularization_tv,
            regularization_sr=regularization_sr,
            fixed_weights=_fixed_weights_key(fixed_weights),
        )

        cached = self._decompositions.get(scope)
        if cached is not None and not force and cached[0] == spec:
            bundle = cached[1]
            self._log(f"Cache hit: decomposition ({scope})")
        else:
            self._log(f"Running NMF decomposition ({scope})...")
            mask_flat = (
                mask_result.flat if mask_result is not None else None
            )
            bundle = self._engine.build_deco(
                sis,
                energy_axis,
                E0_V=beam.E0_V,
                alpha_rad=beam.alpha_rad,
                beta_rad=beam.beta_rad,
                mask_flat=mask_flat,
            )
            self._engine.build_g(
                bundle, fine_structure_ranges, n_power_laws=n_power_laws
            )
            self._engine.decompose(
                bundle,
                n_components=n_components,
                max_iters=max_iters,
                w_init=w_init,
                use_cupy=use_cupy,
                tol=tol,
                error_skip_step=error_skip_step,
                decomposition_method=decomposition_method,
                regularization_tv=regularization_tv,
                regularization_sr=regularization_sr,
                fixed_weights=_fixed_weights_key(fixed_weights),
            )
            self._decompositions[scope] = (spec, bundle)

        return self._post_process(
            bundle,
            sis,
            mask_result,
            fine_structure_ranges=fine_structure_ranges,
            n_components=n_components,
            colors=colors,
        )

    @staticmethod
    def _spec_mask(
        mask_result: MaskResult | None,
    ) -> MaskParams | None:
        # Auto masks are keyed by MaskParams (their content is deterministic
        # for given preprocess + params); user masks use mask_signature.
        if mask_result is None:
            return None
        return mask_result.params

    def _post_process(
        self,
        bundle: DecoBundle,
        sis: Sequence[PreprocessedSI],
        mask_result: MaskResult | None,
        *,
        fine_structure_ranges: Dict[str, Tuple[float, float]],
        n_components: int,
        colors: Sequence[np.ndarray],
    ) -> ProcessingResult:
        deco = bundle.deco

        totals = edge_totals(deco, fine_structure_ranges)
        order = order_by_power(totals, n_components=n_components)
        summary = summary_rows(totals, order) if totals is not None else None

        edges = extract_edge_collection(
            deco,
            fine_structure_ranges,
            n_components,
            order=order,
        )

        maps = component_maps(
            deco,
            bundle.list_of_si,
            fine_structure_ranges,
            n_components,
            order=order,
            mask_result=mask_result,
        )
        mixed, colored = apply_colors(maps, colors)
        original = [si.original_image for si in sis]
        mask_images: List[np.ndarray] | None = None
        if mask_result is not None:
            zero_outside_masks(mixed, original, colored, mask_result.per_si)
            mask_images = list(mask_result.per_si)

        return ProcessingResult(
            images=ImageCollection(
                mixed=mixed,
                original=original,
                components=colored,
                mask=mask_images,
            ),
            edges=edges,
            pixel_scale=sis[0].pixel_scale if sis else None,
            energy_axis=deco.energy_axis,
            list_keys=[si.key for si in sis],
            W=np.asarray(deco.W).copy(),
            deco=deco,
            edge_contributions_summary=summary,
        )

    @staticmethod
    def _map_joint_result(
        result: ProcessingResult, pre: PreprocessedData
    ) -> ProcessingResult:
        """Reorder joint per-SI images from key order to label order."""
        keys = [si.key for si in pre.sis]
        labels = pre.labels

        mixed = (
            map_images_to_labels(result.images.mixed, keys, labels)
            if result.images.mixed is not None
            else []
        )
        original = (
            map_images_to_labels(result.images.original, keys, labels)
            if result.images.original is not None
            else []
        )
        components = None
        if result.images.components is not None:
            components = [
                map_images_to_labels(component, keys, labels)
                for component in result.images.components
            ]
        mask = None
        if result.images.mask is not None:
            mask = map_images_to_labels(result.images.mask, keys, labels)

        return ProcessingResult(
            images=ImageCollection(
                mixed=mixed,
                original=original,
                components=components,
                mask=mask,
            ),
            edges=result.edges,
            pixel_scale=result.pixel_scale,
            energy_axis=result.energy_axis,
            list_keys=result.list_keys,
            W=result.W,
            deco=result.deco,
            edge_contributions_summary=result.edge_contributions_summary,
        )
