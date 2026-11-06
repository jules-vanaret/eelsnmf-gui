"""Loader contract tests using the real EML preprocessing chain.

EML.load_results is patched to return synthetic signals (as if loaded from
dm4 files); everything else — folder copying, EML_Processing, ZLP alignment,
despiking, PreprocessedSI construction — runs for real. These tests pin the
call contract between Session and the default loader, and are skipped when
EELSNMF is not installed.
"""

import numpy as np
import pytest

pytest.importorskip("EELSNMF")

pytestmark = pytest.mark.integration

import hyperspy.api as hs

from eelsnmf_gui.session import Session

N_ENERGY = 48
NY, NX = 4, 5


def _make_pair(seed=0):
    """Synthetic (low-loss, core-loss) signals as a dual-EELS dm4 would load."""
    rng = np.random.default_rng(seed)
    ll_axis = -50.0 + 2.0 * np.arange(50)  # contains the ZLP at 0 eV
    cl_axis = 500.0 + 2.0 * np.arange(N_ENERGY)

    zlp = 10.0 * np.exp(-0.5 * (ll_axis / 5.0) ** 2)
    ll = np.broadcast_to(zlp, (NY, NX, 50)).copy()
    ll = ll + 0.05 * rng.normal(size=ll.shape)

    yy, xx = np.mgrid[0:NY, 0:NX]
    weight = (xx + 1) / (NX + 1)
    edge = np.exp(-0.5 * ((cl_axis - 530.0) / 6.0) ** 2)
    cl = weight[..., None] * edge + 0.05 * rng.normal(size=(NY, NX, N_ENERGY))
    cl = np.clip(cl, 0, None)

    def make_signal(data, offset):
        # EELS-typed signals expose align_zero_loss_peak, as real dm4s do.
        signal = hs.signals.Signal1D(data.astype(np.float32))
        signal.axes_manager.signal_axes[0].offset = offset
        signal.axes_manager.signal_axes[0].scale = 2.0
        signal.axes_manager.signal_axes[0].units = "eV"
        for nav_axis in signal.axes_manager.navigation_axes:
            nav_axis.scale = 0.2
        signal.set_signal_type("EELS")
        return signal

    return [make_signal(ll, -50.0), make_signal(cl, 500.0)]


@pytest.fixture
def patched_eml_load(monkeypatch):
    """Patch EML.load_results to serve synthetic signals per subfolder."""

    def fake_load(folder, target, **kwargs):
        import os

        results = {}
        for entry in sorted(os.listdir(folder)):
            if os.path.isdir(os.path.join(folder, entry)):
                seed = sum(ord(c) for c in entry)
                results[entry] = [_make_pair(seed=seed)]
        return results

    import EELSNMF.experimental.EML as eml_module

    monkeypatch.setattr(eml_module, "load_results", fake_load)
    return eml_module


def test_session_uses_default_loader_contract(patched_eml_load, tmp_path):
    """Session.preprocessed() calls the default loader with the right signature."""
    from functools import partial

    from eelsnmf_gui._preprocess import load_eml_folders

    folders = []
    for name in ("runA", "runB"):
        folder = tmp_path / name
        folder.mkdir()
        folders.append(str(folder))

    session = Session(
        folders,
        # The real EML despike has an edge-case bug with tiny synthetic
        # signals (unguarded ef+1 in kill_channel_range); the contract under
        # test is the Session/loader call signature, not the despike method.
        loader=partial(load_eml_folders, despike_method="hampel"),
    )
    pre = session.preprocessed()

    assert [si.label for si in pre.sis] == ["runA", "runB"]
    assert pre.folder_index == [0, 1]
    assert pre.sis[0].data.shape == (NY, NX, N_ENERGY - 4)  # isig[2:-2] crop
    assert pre.sis[0].original_image.shape == (NY, NX)
    assert pre.sis[0].pixel_scale == pytest.approx(0.2)
    assert len(pre.energy_axis) > 2


def test_loader_without_preprocess_skips_eml_processing(
    patched_eml_load, tmp_path
):
    """With preprocess=False, SIs are used as loaded (no crop, no despiking)."""
    import numpy as np

    from eelsnmf_gui._preprocess import load_eml_folders

    folder = tmp_path / "runA"
    folder.mkdir()
    labels = ["runA"]
    seed = sum(ord(c) for c in "runA")
    _, expected_cl = _make_pair(seed=seed)

    # Record any use of the EML preprocessing chain.
    processing_calls = []

    class RecordingProcessing:
        def __init__(self, core_loss, low_loss):
            processing_calls.append((core_loss, low_loss))
            self.core_loss = core_loss
            self.low_loss = low_loss
            self.despiked = False
            self.despike_calls = []

        def zlp_align(self):
            pass

        def despike(self, core_loss):
            self.despike_calls.append("despike")

    original = patched_eml_load.EML_Processing
    patched_eml_load.EML_Processing = RecordingProcessing
    try:
        raw_sis = load_eml_folders(
            [str(folder)], labels, median_filter_size=5, preprocess=False
        )
        assert processing_calls == []
        pre_sis = load_eml_folders(
            [str(folder)], labels, median_filter_size=5, preprocess=True
        )
        assert len(processing_calls) == 1  # only the preprocessed load
    finally:
        patched_eml_load.EML_Processing = original

    # Preprocessed: cropped by isig[2:-2]; raw: full length as loaded.
    assert pre_sis[0].data.shape == (NY, NX, N_ENERGY - 4)
    assert raw_sis[0].data.shape == (NY, NX, N_ENERGY)
    np.testing.assert_allclose(
        raw_sis[0].data, np.asarray(expected_cl.data, dtype=np.float32)
    )
    assert raw_sis[0].original_image.shape == (NY, NX)
    assert raw_sis[0].pixel_scale == pytest.approx(0.2)


def test_default_despike_is_eml_and_hampel_is_optional(
    patched_eml_load, tmp_path
):
    """The EML despike runs by default; the Hampel filter is opt-in."""
    import numpy as np

    from eelsnmf_gui._preprocess import load_eml_folders

    folder = tmp_path / "runA"
    folder.mkdir()
    labels = ["runA"]
    seed = sum(ord(c) for c in "runA")
    _, expected_cl = _make_pair(seed=seed)

    processing_calls = []

    class RecordingProcessing:
        def __init__(self, core_loss, low_loss):
            processing_calls.append(self)
            self.core_loss = core_loss
            self.low_loss = low_loss
            self.despiked = False
            self.long_despike = None
            self.despike_calls = 0
            self.hampel_calls = 0

        def zlp_align(self):
            pass

        def despike(self, core_loss):
            self.despike_calls += 1

    original = patched_eml_load.EML_Processing
    patched_eml_load.EML_Processing = RecordingProcessing
    try:
        default_sis = load_eml_folders(
            [str(folder)], labels, median_filter_size=5
        )
        hampel_sis = load_eml_folders(
            [str(folder)],
            labels,
            median_filter_size=5,
            despike_method="hampel",
        )
    finally:
        patched_eml_load.EML_Processing = original

    # Default: EML despike (10 iterations), data untouched by Hampel.
    assert default_sis[0].data.shape == (NY, NX, N_ENERGY - 4)
    assert processing_calls[0].despike_calls == 1
    assert processing_calls[0].hampel_calls == 0
    # Hampel: data actually replaced by the filtered array.
    assert hampel_sis[0].data.shape == (NY, NX, N_ENERGY - 4)
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        hampel_sis[0].data,
        np.asarray(expected_cl.data, dtype=np.float32)[:, :, 2:-2],
    )


def test_session_run_with_default_loader_and_engine(patched_eml_load, tmp_path):
    """A full run through the real loader, EML preprocessing, and EELSNMF engine."""
    from functools import partial

    from eelsnmf_gui._preprocess import load_eml_folders
    from eelsnmf_gui._results import BeamParams

    folders = [str(tmp_path / "runA"), str(tmp_path / "runB")]
    for folder in folders:
        (tmp_path / folder.split("/")[-1]).mkdir(exist_ok=True)

    session = Session(
        folders,
        # The synthetic low-loss signals are too small for the real EML
        # despiking algorithm; use the Hampel filter for this contract test
        # (the real EML despike is exercised on actual data).
        loader=partial(load_eml_folders, despike_method="hampel"),
    )
    result = session.run(
        beam=BeamParams(E0_V=300_000.0, alpha_rad=0.0154, beta_rad=0.01846),
        fine_structure_ranges={"O_K": (525.0, 545.0)},
        n_components=2,
        max_iters=10,
        colors=[np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])],
    )

    assert result.joint is not None
    assert len(result.individual) == 2
    assert result.joint.W.shape[1] == 2
