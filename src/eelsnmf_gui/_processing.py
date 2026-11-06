"""Deprecated single-function API kept for backward compatibility.

New code should use :class:`eelsnmf_gui.session.Session` directly; these
wrappers create a throwaway session for each call, so no result is cached
between calls.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Dict, Tuple

import numpy as np

from .session import BeamParams, MaskParams, Session


def _deprecation(name: str) -> None:
    warnings.warn(
        f"eelsnmf_gui._processing.{name} is deprecated; use "
        "eelsnmf_gui.session.Session instead",
        DeprecationWarning,
        stacklevel=3,
    )


def _reject_preprocess_cache(preprocess_cache: dict | None) -> None:
    if preprocess_cache is not None:
        warnings.warn(
            "preprocess_cache is ignored; Session caches results internally",
            DeprecationWarning,
            stacklevel=3,
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
    masks: Sequence[np.ndarray] | None = None,
    preprocess_cache: Dict[Tuple, object] | None = None,
):
    """Process one or more folders (deprecated: use Session.run)."""
    _deprecation("process_folders")
    _reject_preprocess_cache(preprocess_cache)
    session = Session(source_folders, median_filter_size=median_filter_size)
    return session.run(
        beam=BeamParams(E0_V=E0_V, alpha_rad=alpha_rad, beta_rad=beta_rad),
        fine_structure_ranges=fine_structure_ranges,
        n_components=n_components,
        max_iters=max_iters,
        mask=MaskParams(
            enabled=mask_enabled,
            dilation_iters=mask_dilation_iters,
            mode=mask_mode,
        ),
        colors=colors,
        process_individual=process_individual,
        process_joint=process_joint,
        transmit_w_init=transmit_W_init,
        user_masks=masks,
    )


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
    preprocess_cache: Dict[Tuple, object] | None = None,
):
    """Prepare ListOfSI objects for scree plots (deprecated: Session.list_of_si)."""
    _deprecation("build_scree_plot_inputs")
    _reject_preprocess_cache(preprocess_cache)
    if len(source_folders) == 0:
        raise ValueError("No source folders provided")
    session = Session(source_folders, median_filter_size=median_filter_size)
    return session.list_of_si(
        individual=process_individual, joint=process_joint
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
    preprocess_cache: Dict[Tuple, object] | None = None,
):
    """Build EELSNMF objects with G (deprecated: Session.decos_with_g)."""
    _deprecation("build_deco_for_folders_to_plot_er")
    _reject_preprocess_cache(preprocess_cache)
    if len(source_folders) == 0:
        raise ValueError("No source folders provided")
    session = Session(source_folders, median_filter_size=median_filter_size)
    return session.decos_with_g(
        BeamParams(E0_V=E0_V, alpha_rad=alpha_rad, beta_rad=beta_rad),
        fine_structure_ranges,
        MaskParams(
            enabled=mask_enabled,
            dilation_iters=mask_dilation_iters,
            mode=mask_mode,
        ),
    )


def build_mask_preview_for_folders(
    source_folders: Sequence[str],
    *,
    median_filter_size: int,
    dilation_iters: int,
    mode: str = "mean",
    preprocess_cache: Dict[Tuple, object] | None = None,
):
    """Generate mask previews (deprecated: Session.mask_preview)."""
    _deprecation("build_mask_preview_for_folders")
    _reject_preprocess_cache(preprocess_cache)
    if len(source_folders) == 0:
        raise ValueError("No source folders provided")
    session = Session(source_folders, median_filter_size=median_filter_size)
    return session.mask_preview(
        MaskParams(enabled=True, dilation_iters=dilation_iters, mode=mode)
    )
