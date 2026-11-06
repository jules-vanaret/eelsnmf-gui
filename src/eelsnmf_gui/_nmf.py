from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import interp1d

from ._results import DecoBundle, PreprocessedSI
from ._util import HiddenPrints

logger = logging.getLogger(__name__)

_CUPY_AVAILABLE: bool | None = None


def cupy_gpu_available() -> bool:
    """Return True if a usable CUDA device is available via CuPy (cached)."""
    global _CUPY_AVAILABLE
    if _CUPY_AVAILABLE is None:
        try:
            import cupy as cp  # optional dependency

            _CUPY_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            _CUPY_AVAILABLE = False
    return _CUPY_AVAILABLE


class EmlEngine:
    """NMF engine backed by the EELSNMF package (imported lazily).

    All methods take plain :class:`PreprocessedSI` objects and numpy arrays;
    only the *raw* signals carried by the SIs are EELSNMF-specific.
    """

    def list_of_si(
        self,
        sis: Sequence[PreprocessedSI],
        energy_axis: np.ndarray,
    ):
        """Build a ListOfSI wrapper (e.g. for scree plots)."""
        import EELSNMF.utils as enmfu  # lazy heavy import

        sl = enmfu.ListOfSI([si.raw for si in sis], energy_axis)
        sl.unfolded_data = np.clip(sl.unfolded_data, 0, None)
        return sl

    def build_deco(
        self,
        sis: Sequence[PreprocessedSI],
        energy_axis: np.ndarray,
        *,
        E0_V: float,
        alpha_rad: float,
        beta_rad: float,
        mask_flat: np.ndarray | None = None,
    ) -> DecoBundle:
        """Build an EELSNMF object and its ListOfSI wrapper."""
        import EELSNMF as enmf  # lazy heavy import
        import EELSNMF.utils as enmfu  # lazy heavy import

        sl = enmfu.ListOfSI([si.raw for si in sis], energy_axis)
        sl.unfolded_data = np.clip(sl.unfolded_data, 0, None)

        unfolded = sl.unfolded_si
        # Apply the mask by reducing the unfolded data to masked pixels only.
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
        # EELSNMF derives spatial_shape from the unfolded signal (1-D), which
        # is meaningless; restore the real (ny, nx) shape when unambiguous.
        if mask_flat is None and len(sis) == 1:
            deco.spatial_shape = sis[0].spatial_shape
        return DecoBundle(deco=deco, list_of_si=sl)

    def build_g(
        self,
        bundle: DecoBundle,
        fine_structure_ranges: Dict[str, Tuple[float, float]],
        *,
        n_power_laws: int = 50,
    ) -> None:
        """Build the spectral model (G) while suppressing library output.

        *n_power_laws* controls how many power-law background exponents
        (spanning 0.5-8) are included in G.
        """
        with HiddenPrints():
            bundle.deco.build_G(
                fine_structure_ranges=fine_structure_ranges,
                backgrounds=np.linspace(0.5, 8, int(n_power_laws)),
            )

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
        regularization: float | None = None,
        fixed_weights: Sequence[Tuple[str, int, str]] | None = None,
    ) -> None:
        """Run the decomposition, optionally interpolating a W init.

        *use_cupy* selects the backend (None auto-detects a CUDA device).
        *decomposition_method* is any method of the EELSNMF object (e.g.
        "default_decomposition", "default_kl_decomposition",
        "alternate_decomposition", "SumRule_decomposition",
        "EdgeTV_decomposition", "combinedTVSR_decomposition"). *regularization*
        is the lambda passed to the methods that use one (ignored otherwise).
        *fixed_weights* is a sequence of (edge, component, mode) triples
        constraining the decomposition: mode "fixed" freezes the edge's
        cross-section weight in the given component at its initial value;
        mode "excluded" additionally zero-fixes the edge's fine-structure
        channels, so the element cannot appear in that component at all
        (``W_fixed_bool``/``W_fixed_values`` in EELSNMF).
        """
        deco = bundle.deco
        kwargs: dict = {}
        w_init_interp: np.ndarray | None = None
        if w_init is not None:
            if w_init.shape[0] == n_components:
                # Interpolate W_init onto the number of model columns.
                s = deco.G.shape[1]
                range_old = np.linspace(0, 1, w_init.shape[0])
                range_new = np.linspace(0, 1, s)
                interpolator = interp1d(range_old, w_init, axis=0)
                w_init_interp = interpolator(range_new)
            else:
                logger.info(
                    "W_init component count mismatch; ignoring W_init "
                    "for this run"
                )

        if regularization is not None:
            if decomposition_method == "SumRule_decomposition" or decomposition_method == "EdgeTV_decomposition":
                kwargs["lmbda"] = regularization
            elif decomposition_method == "combinedTVSR_decomposition":
                kwargs["TV_lmbda"] = regularization
                kwargs["SR_lmbda"] = regularization

        if fixed_weights:
            # Freeze the pinned W entries at their initial value: compute the
            # initial W exactly as the decomposition will, then pass it as
            # W_init so the init and the constraints agree.
            deco.n_components = n_components
            if w_init_interp is not None:
                initial_w = w_init_interp
            else:
                deco.W_init = None
                deco.H_init = None
                deco.init_nmf = None
                deco.random_state_nmf = None
                # Set what decomposition() would set before its own init
                # (_default_init_WH relies on self.eps).
                deco.eps = 1e-10
                deco._default_init_WH()
                initial_w = np.asarray(deco.W, dtype=float).copy()
            bool_arr, values_arr = self._build_W_fixed(
                deco, fixed_weights, initial_w
            )
            kwargs["W_init"] = initial_w
            kwargs["W_fixed_bool"] = bool_arr
            kwargs["W_fixed_values"] = values_arr
        elif w_init_interp is not None:
            kwargs["W_init"] = w_init_interp

        if use_cupy is None:
            use_cupy = cupy_gpu_available()

        deco.decomposition(
            n_components=n_components,
            max_iters=max_iters,
            tol=tol,
            error_skip_step=error_skip_step,
            use_cupy=use_cupy,
            rescale_G=False,
            decomposition_method=decomposition_method,
            **kwargs,
        )

    @staticmethod
    def _build_W_fixed(
        deco,
        fixed_weights: Sequence[Tuple[str, int, str]],
        initial_w: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build the W_fixed_bool/W_fixed_values arrays from *fixed_weights*.

        Each (edge, component, mode) triple pins rows of W in the given
        component column:

        - mode "fixed": the edge's cross-section weight
          (``model.xsection_idx[edge]``) stays at its initial value;
        - mode "excluded": the cross-section weight and all the edge's
          fine-structure channels (``model._edge_slices[edge]``) are set to
          zero, so the element cannot contribute to that component.
        """
        n_basis = deco.G.shape[1]
        n_components = initial_w.shape[1]
        bool_arr = np.zeros((n_basis, n_components), dtype=bool)
        values_arr = initial_w.copy()
        for edge, component, mode in fixed_weights:
            if edge not in deco.model.xsection_idx:
                raise ValueError(f"Unknown edge for fixed weight: {edge!r}")
            if not 0 <= component < n_components:
                raise ValueError(
                    f"Fixed-weight component {component} out of range for "
                    f"edge {edge!r}"
                )
            if mode not in ("fixed", "excluded"):
                raise ValueError(f"Unknown fixed-weight mode: {mode!r}")

            rows = [deco.model.xsection_idx[edge]]
            if mode == "excluded":
                slice_ = deco.model._edge_slices[edge]
                rows.extend(range(slice_.start, slice_.stop))
            for row in rows:
                bool_arr[row, component] = True
                if mode == "excluded":
                    values_arr[row, component] = 0.0
        return bool_arr, values_arr
