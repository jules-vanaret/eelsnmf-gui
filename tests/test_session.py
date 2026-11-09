import numpy as np
import pytest

from eelsnmf_gui._results import MaskParams
from eelsnmf_gui.session import Session
from tests.fakes import RUN_KWARGS, FakeLoader, StubEngine


@pytest.fixture
def two_folders(tmp_path):
    """Two synthetic folders named runA and runB."""
    folder_a = tmp_path / "runA"
    folder_b = tmp_path / "runB"
    folder_a.mkdir()
    folder_b.mkdir()
    return [str(folder_a), str(folder_b)]


@pytest.fixture
def make_session(two_folders):
    """Build a Session with the fake loader and stub engine."""
    def _make(loader=None, engine=None, **kwargs):
        return Session(
            two_folders,
            loader=loader or FakeLoader(),
            engine=engine or StubEngine(),
            **kwargs,
        )

    return _make


# ---------------------------------------------------------------------------
# preprocessing cache
# ---------------------------------------------------------------------------


def test_preprocessed_is_cached(make_session):
    """preprocessed() returns the same object and calls the loader once."""
    session = make_session()
    loader = session._loader
    pre1 = session.preprocessed()
    pre2 = session.preprocessed()
    assert pre1 is pre2
    assert loader.calls == 1


def test_preprocessed_content(make_session):
    """Preprocessed data carries labels, folder indices, axis, and metadata."""
    session = make_session()
    pre = session.preprocessed()
    assert [si.label for si in pre.sis] == ["runA", "runB"]
    assert pre.folder_index == [0, 1]
    assert pre.labels == ["runA", "runB"]
    assert len(pre.energy_axis) == 31
    assert pre.sis[0].pixel_scale == 0.2


def test_set_median_filter_size_invalidates(make_session):
    """Changing the median filter size forces a reload with the new value."""
    session = make_session()
    loader = session._loader
    session.preprocessed()
    session.set_median_filter_size(5)
    session.preprocessed()
    assert loader.calls == 2
    assert loader.last_median_filter_size == 5


def test_set_median_filter_size_noop_keeps_cache(make_session):
    """Setting the same median filter size keeps the cached preprocessing."""
    session = make_session()
    loader = session._loader
    session.preprocessed()
    session.set_median_filter_size(session.median_filter_size)
    session.preprocessed()
    assert loader.calls == 1


def test_set_folders_invalidates(make_session, tmp_path):
    """Changing folders clears all caches and reloads from the new folders."""
    session = make_session()
    loader = session._loader
    session.preprocessed()

    folder_c = tmp_path / "runC"
    folder_c.mkdir()
    session.set_folders([str(folder_c)])

    pre = session.preprocessed()
    assert loader.calls == 2
    assert loader.last_folders == [str(folder_c)]
    assert pre.labels == ["runC"]


def test_session_rejects_empty_folders():
    """Constructing a Session without folders raises ValueError."""
    with pytest.raises(ValueError):
        Session([])


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------


def test_mask_result_cached_per_params(make_session):
    """Auto masks are cached per MaskParams; None when masking is disabled."""
    session = make_session()
    params = MaskParams(enabled=True, dilation_iters=1, mode="mean")
    result1 = session.mask_result(params)
    result2 = session.mask_result(params)
    assert result1 is result2
    other = MaskParams(enabled=True, dilation_iters=3, mode="mean")
    assert session.mask_result(other) is not result1
    assert session.mask_result(MaskParams()) is None


def test_mask_preview_matches_auto_masks(make_session):
    """Preview masks are identical to the auto masks used in runs."""
    session = make_session()
    params = MaskParams(enabled=True, dilation_iters=1, mode="mean")
    previews = session.mask_preview(params)
    assert len(previews) == 2
    mask_result = session.mask_result(params)
    for (label, aggregated, preview_mask), auto_mask in zip(
        previews, mask_result.per_si
    ):
        assert label in ("runA", "runB")
        assert aggregated.shape == (4, 5)
        np.testing.assert_array_equal(preview_mask, auto_mask)


# ---------------------------------------------------------------------------
# diagnostic stages
# ---------------------------------------------------------------------------


def test_list_of_si_items(make_session):
    """list_of_si returns Joint plus per-folder items in label order."""
    session = make_session()
    items = session.list_of_si(individual=True, joint=True)
    assert [label for label, _ in items] == ["Joint", "runA", "runB"]
    assert [label for label, _ in session.list_of_si(joint=True, individual=False)] == [
        "Joint"
    ]
    with pytest.raises(ValueError):
        session.list_of_si(individual=False, joint=False)


def test_decos_with_g(make_session):
    """decos_with_g builds one G-fitted deco per folder, each masked with its own SIs only."""
    session = make_session()
    engine = session._engine
    mask = MaskParams(enabled=True, dilation_iters=1, mode="mean")
    decos = session.decos_with_g(
        RUN_KWARGS["beam"],
        RUN_KWARGS["fine_structure_ranges"],
        mask,
        n_power_laws=7,
    )
    assert len(decos) == 2
    assert len(engine.build_g_calls) == 2
    assert engine.build_g_calls[0]["fine_structure_ranges"] == (
        RUN_KWARGS["fine_structure_ranges"]
    )
    assert engine.build_g_calls[0]["n_power_laws"] == 7
    # Each folder deco received the mask of its own SIs.
    mask_result = session.mask_result(mask)
    assert (
        engine.build_deco_calls[0]["mask_pixels"]
        == int(mask_result.per_si[0].sum())
    )
    assert (
        engine.build_deco_calls[1]["mask_pixels"]
        == int(mask_result.per_si[1].sum())
    )


def test_run_forwards_decomposition_params(make_session):
    """tol, error_skip_step, method, and lambdas reach the engine."""
    session = make_session()
    engine = session._engine
    session.run(
        **RUN_KWARGS,
        tol=1e-6,
        error_skip_step=5,
        decomposition_method="SumRule_decomposition",
        regularization_sr=0.3,
        n_power_laws=11,
    )
    for call in engine.decompose_calls:
        assert call["tol"] == 1e-6
        assert call["error_skip_step"] == 5
        assert call["decomposition_method"] == "SumRule_decomposition"
        assert call["regularization_sr"] == 0.3
        assert call["regularization_tv"] is None
    assert all(c["n_power_laws"] == 11 for c in engine.build_g_calls)


def test_run_redecomposes_on_decomposition_param_changes(make_session):
    """Changing tol, method, lambdas, or power laws forces a new run."""
    session = make_session()
    engine = session._engine
    session.run(**RUN_KWARGS)
    assert len(engine.decompose_calls) == 3

    session.run(**{**RUN_KWARGS, "tol": 1e-6})
    assert len(engine.decompose_calls) == 6
    session.run(
        **{**RUN_KWARGS, "decomposition_method": "default_kl_decomposition"}
    )
    assert len(engine.decompose_calls) == 9
    session.run(
        **{**RUN_KWARGS, "decomposition_method": "SumRule_decomposition"}
    )  # no lambdas -> same as method change only
    assert len(engine.decompose_calls) == 12
    session.run(
        **{
            **RUN_KWARGS,
            "decomposition_method": "SumRule_decomposition",
            "regularization_sr": 0.5,
        }
    )
    assert len(engine.decompose_calls) == 15
    session.run(
        **{
            **RUN_KWARGS,
            "decomposition_method": "combinedTVSR_decomposition",
            "regularization_tv": 0.2,
        }
    )
    assert len(engine.decompose_calls) == 18
    session.run(**{**RUN_KWARGS, "n_power_laws": 80})
    assert len(engine.decompose_calls) == 21


def test_run_regularization_ignored_for_plain_methods(make_session):
    """lambdas are forwarded to the engine; filtering is the GUI's job."""
    session = make_session()
    engine = session._engine
    session.run(**RUN_KWARGS, regularization_tv=0.4, regularization_sr=0.7)
    assert all(
        c["regularization_tv"] == 0.4 and c["regularization_sr"] == 0.7
        for c in engine.decompose_calls
    )


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


def test_run_returns_results_for_all_scopes(make_session):
    """A full run returns joint and individual results with all outputs."""
    session = make_session()
    result = session.run(**RUN_KWARGS)

    assert [label for label in result.labels] == ["runA", "runB"]
    assert result.labels_joint == ["runA", "runB"]
    assert len(result.individual) == 2
    assert result.joint is not None

    joint = result.joint
    assert len(joint.images.mixed) == 2
    assert joint.images.mixed[0].shape == (4, 5, 3)
    assert len(joint.images.original) == 2
    assert len(joint.images.components) == 2
    assert len(joint.images.components[0]) == 2
    assert joint.images.mask is None
    assert len(joint.edges.by_label) == 2
    assert len(joint.edges.combined.x) == 13  # 10 (O_K) + 3 (Fe_L) channels
    assert len(joint.edge_contributions_summary) == 2
    assert joint.W.shape[1] == 2
    assert joint.deco is result.deco_joint
    assert joint.pixel_scale == 0.2

    individual = result.individual[0]
    assert len(individual.images.mixed) == 1
    assert len(individual.images.components) == 2
    assert len(individual.images.components[0]) == 1
    assert individual.list_keys == [session.folders[0]]


def test_run_reuses_decompositions(make_session):
    """Repeated runs and color changes reuse the cached decompositions."""
    session = make_session()
    engine = session._engine
    session.run(**RUN_KWARGS)
    assert len(engine.decompose_calls) == 3  # joint + 2 individual

    session.run(**RUN_KWARGS)
    assert len(engine.decompose_calls) == 3  # fully cached

    new_colors = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0, 0.0])]
    result = session.run(**{**RUN_KWARGS, "colors": new_colors})
    assert len(engine.decompose_calls) == 3  # colors do not re-decompose
    # ...but the recolored output reflects the new colors.
    assert result.joint.images.mixed[0].dtype == result.joint.images.mixed[
        0
    ].dtype


def test_run_force_redecomposes(make_session):
    """force=True bypasses the decomposition cache in every scope."""
    session = make_session()
    engine = session._engine
    session.run(**RUN_KWARGS)
    session.run(**RUN_KWARGS, force=True)
    assert len(engine.decompose_calls) == 6


def test_run_redecomposes_on_param_changes(make_session):
    """Range, max_iters, or n_components changes force a re-decomposition."""
    session = make_session()
    engine = session._engine
    session.run(**RUN_KWARGS)
    session.run(
        **{**RUN_KWARGS, "fine_structure_ranges": {"O_K": (530.0, 548.0)}}
    )
    assert len(engine.decompose_calls) == 6
    session.run(**{**RUN_KWARGS, "max_iters": 7})
    assert len(engine.decompose_calls) == 9
    session.run(**{**RUN_KWARGS, "n_components": 1})
    assert len(engine.decompose_calls) == 12


def test_run_transmits_w_init_to_individuals(make_session):
    """Individual runs receive the joint W as init; the joint starts bare."""
    session = make_session()
    engine = session._engine
    session.run(**RUN_KWARGS)
    joint_call, ind_calls = engine.decompose_calls[0], engine.decompose_calls[1:]
    assert joint_call["w_init_shape"] is None
    joint_w_shape = session._decompositions["joint"][1].deco.W.shape
    for call in ind_calls:
        assert call["w_init_shape"] == joint_w_shape


def test_run_without_w_init_transmission(make_session):
    """With transmit_w_init=False, no run receives a W init."""
    session = make_session()
    engine = session._engine
    session.run(**RUN_KWARGS, transmit_w_init=False)
    for call in engine.decompose_calls:
        assert call["w_init_shape"] is None


def test_run_use_cupy_forwarded_and_part_of_spec(make_session):
    """The use_cupy flag reaches the engine and switching it re-decomposes."""
    session = make_session()
    engine = session._engine
    session.run(**RUN_KWARGS, use_cupy=True)
    assert all(call["use_cupy"] is True for call in engine.decompose_calls)

    session.run(**RUN_KWARGS, use_cupy=True)
    assert len(engine.decompose_calls) == 3  # cached

    session.run(**RUN_KWARGS, use_cupy=False)
    assert len(engine.decompose_calls) == 6  # backend change re-runs


def test_run_fixed_weights_forwarded_and_part_of_spec(make_session):
    """Fixed-weight constraints reach the engine; changing them re-runs."""
    session = make_session()
    engine = session._engine
    session.run(**RUN_KWARGS, fixed_weights=(("Fe_L", 0, "fixed"),))
    for call in engine.decompose_calls:
        assert call["fixed_weights"] == (("Fe_L", 0, "fixed"),)

    session.run(**RUN_KWARGS, fixed_weights=(("Fe_L", 0, "fixed"),))
    assert len(engine.decompose_calls) == 3  # cached

    session.run(
        **RUN_KWARGS,
        fixed_weights=(("Fe_L", 0, "fixed"), ("O_K", 1, "excluded")),
    )
    assert len(engine.decompose_calls) == 6  # constraints changed
    # Order of the constraints does not matter for caching.
    session.run(
        **RUN_KWARGS,
        fixed_weights=(("O_K", 1, "excluded"), ("Fe_L", 0, "fixed")),
    )
    assert len(engine.decompose_calls) == 6  # cached: same sorted set


def test_run_fixed_weights_empty_is_none_like(make_session):
    """An empty constraint tuple behaves like no constraints."""
    session = make_session()
    engine = session._engine
    session.run(**RUN_KWARGS)
    first = engine.decompose_calls[0]["fixed_weights"]
    assert first == ()
    session.run(**RUN_KWARGS, fixed_weights=())
    assert len(engine.decompose_calls) == 3  # cached: () == no constraints


def test_run_joint_only(make_session):
    """With only the joint mode, no individual results are produced."""
    session = make_session()
    result = session.run(**RUN_KWARGS, process_individual=False)
    assert result.joint is not None
    assert result.individual == []
    assert result.labels == []
    assert result.deco_joint is not None


def test_run_individual_only(make_session):
    """With only the individual mode, no joint result is produced."""
    session = make_session()
    result = session.run(**RUN_KWARGS, process_joint=False)
    assert result.joint is None
    assert len(result.individual) == 2
    assert result.deco_joint is None


def test_run_validation_errors(make_session):
    """run() rejects empty modes, bad n_components, few colors, no ranges."""
    session = make_session()
    with pytest.raises(ValueError):
        session.run(
            **{**RUN_KWARGS, "process_individual": False, "process_joint": False}
        )
    with pytest.raises(ValueError):
        session.run(**{**RUN_KWARGS, "n_components": 0})
    with pytest.raises(ValueError):
        session.run(
            **{**RUN_KWARGS, "colors": [np.array([1.0, 0.0, 0.0])]}
        )
    with pytest.raises(ValueError):
        session.run(**{**RUN_KWARGS, "fine_structure_ranges": {}})


# ---------------------------------------------------------------------------
# user-provided masks
# ---------------------------------------------------------------------------


def test_user_masks_are_applied(make_session):
    """User masks are broadcast per SI and zero masked-out output pixels."""
    session = make_session()
    engine = session._engine
    mask_a = np.ones((4, 5), dtype=bool)
    mask_a[0, 0] = False
    mask_b = np.ones((4, 5), dtype=bool)
    result = session.run(
        **RUN_KWARGS,
        mask=MaskParams(enabled=True),
        user_masks=[mask_a, mask_b],
    )

    assert engine.decompose_calls[0]["mask_pixels"] == 19 + 20
    assert engine.decompose_calls[1]["mask_pixels"] == 19
    individual = result.individual[0]
    assert individual.images.mask is not None
    assert individual.images.mixed[0][0, 0].sum() == 0
    assert individual.images.components[0][0][0, 0].sum() == 0
    assert individual.images.original[0][0, 0] == 0
    joint = result.joint
    assert joint.images.mask is not None


def test_user_masks_validation(make_session):
    """User masks are rejected on count, shape, disabled flag, or all-False."""
    session = make_session()
    good = np.ones((4, 5), dtype=bool)
    with pytest.raises(ValueError):
        session.run(**RUN_KWARGS, mask=MaskParams(True), user_masks=[good])
    with pytest.raises(ValueError):
        session.run(
            **RUN_KWARGS,
            mask=MaskParams(True),
            user_masks=[good, np.ones((3, 5), dtype=bool)],
        )
    with pytest.raises(ValueError):
        session.run(**RUN_KWARGS, user_masks=[good, good])
    with pytest.raises(ValueError):
        session.run(
            **RUN_KWARGS,
            mask=MaskParams(True),
            user_masks=[np.zeros((4, 5), dtype=bool)] * 2,
        )


# ---------------------------------------------------------------------------
# deprecated single-function API
# ---------------------------------------------------------------------------


def test_process_folders_deprecated_shim(monkeypatch, two_folders):
    """The deprecated process_folders maps args onto Session.run and warns."""
    import eelsnmf_gui._processing as shim

    calls = {}

    class FakeSession:
        def __init__(self, folders, *, median_filter_size):
            calls["folders"] = list(folders)
            calls["median_filter_size"] = median_filter_size

        def run(self, **kwargs):
            calls["run_kwargs"] = kwargs
            return "RESULT"

    monkeypatch.setattr(shim, "Session", FakeSession)

    colors = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    with pytest.deprecated_call():
        result = shim.process_folders(
            two_folders,
            E0_V=1.0,
            alpha_rad=2.0,
            beta_rad=3.0,
            max_iters=10,
            median_filter_size=5,
            fine_structure_ranges={"O_K": (1.0, 2.0)},
            colors=colors,
            n_components=2,
            mask_enabled=True,
            mask_dilation_iters=2,
        )

    assert result == "RESULT"
    assert calls["folders"] == two_folders
    assert calls["median_filter_size"] == 5
    run_kwargs = calls["run_kwargs"]
    assert run_kwargs["beam"].E0_V == 1.0
    assert run_kwargs["mask"].enabled is True
    assert run_kwargs["mask"].dilation_iters == 2
    assert run_kwargs["colors"] == colors


def test_preprocess_cache_argument_is_ignored_with_warning(two_folders):
    """Passing preprocess_cache warns it is ignored (empty folders fail)."""
    import eelsnmf_gui._processing as shim

    with pytest.raises(ValueError), pytest.warns(DeprecationWarning):
        shim.build_mask_preview_for_folders(
            [], median_filter_size=5, dilation_iters=1, preprocess_cache={}
        )


# ---------------------------------------------------------------------------
# joint result mapping
# ---------------------------------------------------------------------------


def test_joint_images_are_mapped_to_label_order(tmp_path):
    """Joint images loaded out of key order are remapped to label order."""
    folder_a = tmp_path / "runA"
    folder_b = tmp_path / "runB"
    folder_a.mkdir()
    folder_b.mkdir()
    loader = FakeLoader(reverse=True)  # loads runB's SI first
    session = Session(
        [str(folder_a), str(folder_b)], loader=loader, engine=StubEngine()
    )
    result = session.run(**RUN_KWARGS, process_individual=False)

    pre = session.preprocessed()
    si_a = next(si for si in pre.sis if si.label == "runA")
    assert np.array_equal(result.joint.images.original[0], si_a.original_image)
    assert result.joint.list_keys == [si.key for si in pre.sis]
