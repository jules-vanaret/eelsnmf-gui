"""End-to-end tests against the real EELSNMF backend.

These tests are skipped automatically when EELSNMF (or hyperspy) is not
installed; everything else in the suite runs without them.
"""

import numpy as np
import pytest

pytest.importorskip("EELSNMF")

pytestmark = pytest.mark.integration

from eelsnmf_gui._nmf import EmlEngine
from eelsnmf_gui._results import MaskParams, PreprocessedSI
from eelsnmf_gui.session import BeamParams, Session

N_ENERGY = 48
ENERGY_OFFSET = 500.0
ENERGY_SCALE = 2.0

RANGES = {"O_K": (525.0, 545.0), "Fe_L": (550.0, 570.0)}


def _make_raw_si(ny=4, nx=5, seed=0):
    import hyperspy.api as hs

    rng = np.random.default_rng(seed)
    axis = ENERGY_OFFSET + ENERGY_SCALE * np.arange(N_ENERGY)
    yy, xx = np.mgrid[0:ny, 0:nx]
    weight_a = (xx + 1) / (nx + 1)
    weight_b = 1 - weight_a
    spectrum_a = np.exp(-0.5 * ((axis - 530.0) / 6.0) ** 2)
    spectrum_b = np.exp(-0.5 * ((axis - 555.0) / 6.0) ** 2)
    data = (
        weight_a[..., None] * spectrum_a + weight_b[..., None] * spectrum_b
    )
    data = data + 0.05 * rng.normal(size=data.shape)
    data = np.clip(data, 0, None).astype(np.float32)

    signal = hs.signals.Signal1D(data)
    signal.axes_manager.signal_axes[0].offset = ENERGY_OFFSET
    signal.axes_manager.signal_axes[0].scale = ENERGY_SCALE
    for nav_axis in signal.axes_manager.navigation_axes:
        nav_axis.scale = 0.2
    return signal


def _loader(folders, labels, median_filter_size):
    sis = []
    for label in labels:
        signal = _make_raw_si(seed=len(label))
        sis.append(
            PreprocessedSI(
                key=label,
                label=label,
                data=np.asarray(signal.data, dtype=np.float32),
                raw=signal,
                original_image=np.asarray(signal.data).mean(-1),
                pixel_scale=0.2,
                energy_offset=ENERGY_OFFSET,
                energy_scale=ENERGY_SCALE,
            )
        )
    return sis


def _make_session(tmp_path):
    folders = []
    for name in ("runA", "runB"):
        folder = tmp_path / name
        folder.mkdir(exist_ok=True)
        folders.append(str(folder))
    return Session(folders, loader=_loader, engine=EmlEngine())


def test_end_to_end_run(tmp_path):
    """A joint+individual run on real EELSNMF yields W, images, summaries."""
    session = _make_session(tmp_path)
    result = session.run(
        beam=BeamParams(E0_V=300_000.0, alpha_rad=0.0154, beta_rad=0.01846),
        fine_structure_ranges=RANGES,
        n_components=2,
        max_iters=30,
        colors=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])],
    )

    assert len(result.individual) == 2
    assert result.joint is not None
    for processing_result in [result.joint, *result.individual]:
        assert processing_result.W.shape[1] == 2
        assert processing_result.W.min() >= 0
        assert len(processing_result.images.mixed) >= 1
        assert processing_result.images.mixed[0].shape[-1] == 3
        assert len(processing_result.edges.by_label) == 2
        assert len(processing_result.edge_contributions_summary) == 2
        assert processing_result.pixel_scale == 0.2


def test_end_to_end_run_with_user_masks(tmp_path):
    """User masks flow through the real engine and zero masked-out pixels."""
    session = _make_session(tmp_path)
    mask_a = np.ones((4, 5), dtype=bool)
    mask_a[0, 0] = False
    mask_b = np.ones((4, 5), dtype=bool)
    result = session.run(
        beam=BeamParams(E0_V=300_000.0, alpha_rad=0.0154, beta_rad=0.01846),
        fine_structure_ranges=RANGES,
        n_components=2,
        max_iters=30,
        colors=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])],
        mask=MaskParams(True),
        user_masks=[mask_a, mask_b],
    )

    assert result.joint.images.mask is not None
    assert result.joint.images.mixed[0][0, 0].sum() == 0
    assert len(result.individual) == 2
