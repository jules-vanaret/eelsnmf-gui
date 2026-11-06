import numpy as np
import pytest

from eelsnmf_gui._analysis import (
    apply_colors,
    component_maps,
    edge_totals,
    extract_edge_collection,
    map_images_to_labels,
    order_by_power,
    summary_rows,
    zero_outside_masks,
)
from eelsnmf_gui._results import MaskParams, MaskResult
from tests.fakes import (
    ENERGY_OFFSET,
    ENERGY_SCALE,
    N_ENERGY,
    RUN_KWARGS,
    StubEngine,
    make_si,
)

IDENTITY_SMOOTHER = lambda arr: arr  # noqa: E731

RANGES = RUN_KWARGS["fine_structure_ranges"]


@pytest.fixture
def fitted_stub():
    """A stub deco fitted with two components, the first one dominant."""
    engine = StubEngine()
    sis = [make_si("runA", seed=0), make_si("runB", seed=1)]
    axis = ENERGY_OFFSET + ENERGY_SCALE * np.arange(N_ENERGY)
    bundle = engine.build_deco(sis, axis, E0_V=1.0, alpha_rad=1.0, beta_rad=1.0)
    engine.build_g(bundle, RANGES)
    engine.decompose(bundle, n_components=2, max_iters=10)
    return engine, bundle


def test_edge_totals_component_ranking(fitted_stub):
    """The dominant component gets strictly larger power."""
    _, bundle = fitted_stub
    totals = edge_totals(bundle.deco, RANGES)
    assert totals is not None
    assert totals.power.shape == (2,)
    assert totals.power[0] > totals.power[1] > 0
    assert totals.explained_variance.shape == (2,)
    assert totals.leave_out_loss.shape == (2,)


def test_edge_totals_requires_fitted_deco():
    """A deco without fitted components (n_components=0) yields no totals."""
    from tests.fakes import StubDeco

    deco = StubDeco([(4, 5)], np.arange(10.0))
    deco.n_components = 0  # not fitted
    assert edge_totals(deco, RANGES) is None


def test_order_by_power(fitted_stub):
    """Components are ordered by decreasing power."""
    _, bundle = fitted_stub
    totals = edge_totals(bundle.deco, RANGES)
    assert order_by_power(totals, n_components=2) == [0, 1]


def test_order_by_power_reversed_when_second_is_stronger():
    """With the second component stronger, the ordering flips to [1, 0]."""
    engine = StubEngine(powers=(0.25, 1.0))
    sis = [make_si("runA", seed=0)]
    axis = ENERGY_OFFSET + ENERGY_SCALE * np.arange(N_ENERGY)
    bundle = engine.build_deco(sis, axis, E0_V=1.0, alpha_rad=1.0, beta_rad=1.0)
    engine.build_g(bundle, RANGES)
    engine.decompose(bundle, n_components=2, max_iters=10)
    totals = edge_totals(bundle.deco, RANGES)
    assert order_by_power(totals, n_components=2) == [1, 0]


def test_order_by_power_falls_back_to_identity():
    """Without totals, ordering falls back to the identity permutation."""
    assert order_by_power(None, n_components=3) == [0, 1, 2]


def test_summary_rows_structure_and_ordering(fitted_stub):
    """Rows carry all metric columns, follow the order, and sum to 100%."""
    _, bundle = fitted_stub
    totals = edge_totals(bundle.deco, RANGES)
    order = [1, 0]
    rows = summary_rows(totals, order)
    assert len(rows) == 2
    expected_keys = {
        "Component",
        "Leave-Out Loss",
        "Relative Leave-Out Loss (%)",
        "Power",
        "Relative Power (%)",
        "Explained Variance",
        "Relative Explained Variance (%)",
    }
    assert set(rows[0]) == expected_keys
    assert [row["Component"] for row in rows] == [0, 1]
    # Rows follow the given order: first row shows component 1's values.
    assert rows[0]["Leave-Out Loss"] == pytest.approx(
        totals.leave_out_loss[1]
    )
    for key in (
        "Relative Leave-Out Loss (%)",
        "Relative Power (%)",
        "Relative Explained Variance (%)",
    ):
        assert sum(row[key] for row in rows) == pytest.approx(100.0)


def test_extract_edge_collection(fitted_stub):
    """Edges combine sorted by onset; normalized chunks in [0, 1]."""
    _, bundle = fitted_stub
    collection = extract_edge_collection(
        bundle.deco, RANGES, 2, smoother=IDENTITY_SMOOTHER
    )

    assert set(collection.by_label) == set(RANGES)
    combined = collection.combined
    # Edges are concatenated sorted by onset energy: O_K (530) then Fe_L
    # (554) on the synthetic axis.
    ok_x = np.linspace(ENERGY_OFFSET, 562.0, N_ENERGY)[15:25]
    fe_x = np.linspace(ENERGY_OFFSET, 562.0, N_ENERGY)[27:31]
    np.testing.assert_allclose(combined.x, np.concatenate([ok_x, fe_x]))
    assert len(combined.raw) == 2
    assert len(combined.raw[0]) == len(combined.x)

    # Raw edge of component j is a ramp scaled by (j + 1).
    expected_chunk = np.linspace(0.1, 1.0, N_ENERGY)[15:25]
    np.testing.assert_allclose(
        collection.by_label["O_K"].raw[1], expected_chunk * 2
    )
    # Smoothed (identity) and normalized chunks live in [0, 1].
    norm = collection.by_label["O_K"].norm[1]
    assert norm.min() == pytest.approx(0.0)
    assert norm.max() == pytest.approx(1.0)


def test_extract_edge_collection_applies_order(fitted_stub):
    """The order argument selects which component's spectra come first."""
    _, bundle = fitted_stub
    collection = extract_edge_collection(
        bundle.deco, RANGES, 2, order=[1, 0], smoother=IDENTITY_SMOOTHER
    )
    expected_chunk = np.linspace(0.1, 1.0, N_ENERGY)[15:25]
    # With order [1, 0], the first entry is component 1 (scaled by 2).
    np.testing.assert_allclose(
        collection.by_label["O_K"].raw[0], expected_chunk * 2
    )


def test_component_maps_with_mask():
    """Masked maps are 2D images in [0, 1] with zeros at masked-out pixels."""
    engine = StubEngine()
    sis = [make_si("runA", seed=0), make_si("runB", seed=1)]
    axis = ENERGY_OFFSET + ENERGY_SCALE * np.arange(N_ENERGY)
    mask_a = np.ones((4, 5), dtype=bool)
    mask_a[0, 0] = False
    mask_b = np.ones((4, 5), dtype=bool)
    mask_flat = np.concatenate([mask_a.ravel(), mask_b.ravel()])
    mask_result = MaskResult(
        params=MaskParams(True),
        per_si=[mask_a, mask_b],
        flat=mask_flat,
        counts=[int(mask_a.sum()), int(mask_b.sum())],
        shapes=[(4, 5), (4, 5)],
    )
    # A masked decomposition produces H over the masked pixels only.
    bundle = engine.build_deco(
        sis, axis, E0_V=1.0, alpha_rad=1.0, beta_rad=1.0, mask_flat=mask_flat
    )
    engine.build_g(bundle, RANGES)
    engine.decompose(bundle, n_components=2, max_iters=10)

    maps = component_maps(
        bundle.deco,
        bundle.list_of_si,
        RANGES,
        2,
        order=[0, 1],
        mask_result=mask_result,
    )
    assert len(maps) == 2  # one entry per component
    assert [len(m) for m in maps] == [2, 2]  # one image per SI
    for comp_maps in maps:
        for img in comp_maps:
            assert img.shape == (4, 5)
            assert img.max() <= 1.0
            assert img.min() >= 0.0
    # Masked-out pixels are zero in every component map.
    assert maps[0][0][0, 0] == 0.0
    assert maps[1][0][0, 0] == 0.0


def test_component_maps_without_mask_uses_fold(fitted_stub):
    """Without a mask, maps come from folding H rows via ListOfSI."""
    _, bundle = fitted_stub
    maps = component_maps(
        bundle.deco,
        bundle.list_of_si,
        RANGES,
        2,
        order=[0, 1],
        mask_result=None,
    )
    assert [len(m) for m in maps] == [2, 2]
    for comp_maps in maps:
        for img in comp_maps:
            assert img.shape == (4, 5)


def test_apply_colors():
    """Maps mix into RGB images; each component map is colored individually."""
    maps = [
        [np.full((2, 2), 0.5), np.full((2, 2), 1.0)],
        [np.full((2, 2), 0.5), np.zeros((2, 2))],
    ]
    colors = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    mixed, colored = apply_colors(maps, colors)

    assert len(mixed) == 2
    np.testing.assert_allclose(mixed[0][0, 0], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(mixed[1][0, 0], [1.0, 0.0, 0.0])
    assert colored[0][0].shape == (2, 2, 3)
    np.testing.assert_allclose(colored[0][0][0, 0], [0.5, 0.0, 0.0])


def test_zero_outside_masks():
    """Masked-out pixels are zeroed in mixed, original, and colored images."""
    mask = np.ones((2, 2), dtype=bool)
    mask[0, 0] = False
    mixed = [np.ones((2, 2, 3))]
    original = [np.ones((2, 2))]
    colored = [[np.ones((2, 2, 3))]]
    zero_outside_masks(mixed, original, colored, [mask])
    assert (mixed[0][0, 0] == 0).all()
    assert original[0][0, 0] == 0
    assert (colored[0][0][0, 0] == 0).all()
    assert mixed[0][1, 1].min() == 1.0


def test_map_images_to_labels():
    """Images reorder to label order; unknown labels map to None."""
    images = ["imgA", "imgB"]
    keys = ["/data/runB", "/data/runA"]
    assert map_images_to_labels(images, keys, ["runA", "runB"]) == [
        "imgB",
        "imgA",
    ]
    assert map_images_to_labels(images, keys, ["runA", "other"]) == [
        "imgB",
        None,
    ]
    assert map_images_to_labels(None, keys, ["runA"]) == []
    assert map_images_to_labels(images, None, ["runA"]) == []
