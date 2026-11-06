import numpy as np
import pytest

from eelsnmf_gui._preprocess import (
    aggregate_energy,
    common_energy_axis,
    despike_array,
)


def test_despike_removes_isolated_spike():
    """An isolated outlier is replaced by the local median."""
    data = np.ones((2, 2, 16), dtype=np.float32)
    data[0, 0, 5] = 100.0

    filtered = despike_array(data, median_filter_size=3)

    assert filtered.shape == data.shape
    assert filtered[0, 0, 5] == pytest.approx(1.0, abs=1e-6)
    # Untouched samples stay identical.
    assert filtered[1, 1] == pytest.approx(data[1, 1])


def test_despike_keeps_smooth_data():
    """A smooth signal passes through despiking unchanged."""
    axis = np.arange(24, dtype=np.float32)
    data = np.repeat((axis**2)[None, None, :], 3, axis=1).reshape(1, 3, 24)

    filtered = despike_array(data, median_filter_size=3)

    np.testing.assert_allclose(filtered, data, rtol=1e-5)


def test_common_energy_axis_identical_inputs():
    """Identical SI axes give an axis at the same offset, floor-rounded."""
    axis = common_energy_axis([500.0, 500.0], [2.0, 2.0], [32, 32])
    assert axis[0] == 500.0
    # The overlapping window is floored to whole steps of the mean scale.
    assert len(axis) == 31
    np.testing.assert_allclose(np.diff(axis), 2.0)


def test_common_energy_axis_uses_mean_scale_and_overlap():
    """The axis uses the mean dispersion and the overlapping energy window."""
    axis = common_energy_axis([0.0, 10.0], [1.0, 2.0], [100, 100])
    assert axis[0] == pytest.approx(10.0)
    # min end = min(99, 10 + 99*2) = 99; mean scale 1.5
    assert axis[-1] < 99.0 + 1.5
    np.testing.assert_allclose(np.diff(axis), 1.5)


def test_common_energy_axis_too_small_raises():
    """An overlap smaller than two steps raises ValueError."""
    with pytest.raises(ValueError):
        common_energy_axis([0.0, 90.0], [100.0, 100.0], [2, 2])


def test_aggregate_energy_modes():
    """'sum' and 'mean' aggregate as expected; other modes use the mean."""
    data = np.arange(2 * 2 * 4, dtype=float).reshape(2, 2, 4)
    np.testing.assert_allclose(
        aggregate_energy(data, "sum"), data.sum(axis=-1)
    )
    np.testing.assert_allclose(
        aggregate_energy(data, "mean"), data.mean(axis=-1)
    )
    # Anything else falls back to the mean.
    np.testing.assert_allclose(
        aggregate_energy(data, "other"), data.mean(axis=-1)
    )
