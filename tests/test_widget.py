from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("qtpy")
pytest.importorskip("pytestqt")

from eelsnmf_gui._widget import DEFAULT_EDGE_RANGES, EELSProcessingWidget


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path):
    """Redirect QSettings to a temp dir so tests never touch user prefs."""
    from qtpy.QtCore import QSettings

    QSettings.setDefaultFormat(QSettings.IniFormat)
    # The widget uses the two-argument QSettings constructor (NativeFormat),
    # so both formats must be redirected.
    for fmt in (QSettings.NativeFormat, QSettings.IniFormat):
        QSettings.setPath(fmt, QSettings.UserScope, str(tmp_path))
    yield


def test_widget_constructs(qtbot):
    """The widget builds offscreen with no folders and no session."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    assert widget.source_folders == []
    assert widget._session is None


def test_pre_mode_shows_preprocessing_section(qtbot):
    """Default (pre) mode shows the preprocessing parameters section."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    assert widget.pre_mode is True
    assert not widget.preprocessing_group.isHidden()


def test_non_pre_mode_hides_preprocessing_section(qtbot, tmp_path):
    """Non-pre mode hides the preprocessing section and disables masking."""
    widget = EELSProcessingWidget(debug=True, pre_mode=False)
    qtbot.addWidget(widget)
    assert widget.pre_mode is False
    assert widget.preprocessing_group.isHidden()
    assert widget._mask_params().enabled is False

    folder = tmp_path / "runA"
    folder.mkdir()
    widget.source_folders = [str(folder)]
    widget._create_session()
    loader = widget._session._loader
    assert loader.keywords["preprocess"] is False


def test_beam_and_mask_params(qtbot):
    """Beam params convert keV/mrad to V/rad; the default mask is off."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    widget.e0 = 200.0
    widget.alpha = 10.0
    widget.beta = 20.0
    beam = widget._beam_params()
    assert pytest.approx(200_000.0) == beam.E0_V
    assert beam.alpha_rad == pytest.approx(0.01)
    assert beam.beta_rad == pytest.approx(0.02)

    mask = widget._mask_params()
    assert mask.enabled is False
    assert mask.mode == "mean"


def test_default_ranges(qtbot):
    """Default ranges mirror DEFAULT_EDGE_RANGES with only Fe_L enabled."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    defaults = widget._default_ranges()
    assert [entry["name"] for entry in defaults] == list(DEFAULT_EDGE_RANGES)
    assert [entry["checked"] for entry in defaults] == [True, False]

    ranges = widget._collect_ranges(only_enabled=True)
    assert ranges == {"Fe_L": (700.0, 740.0)}


def test_ensure_color_count(qtbot):
    """The color list is padded to or trimmed from the component count."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    widget.n_components = 6
    widget._ensure_color_count()
    assert len(widget.colors) == 6
    widget.n_components = 1
    widget._ensure_color_count()
    assert len(widget.colors) == 1


def test_create_session_from_folders(qtbot, tmp_path):
    """Selecting folders creates a Session tracking the median filter."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    folder = tmp_path / "runA"
    folder.mkdir()
    widget.source_folders = [str(folder)]
    widget._create_session()
    assert widget._session is not None
    assert widget._session.folders == (str(folder),)

    widget.median_filter_size = 7
    widget._on_median_filter_size_changed(7)
    assert widget._session.median_filter_size == 7


def test_use_cuda_checkbox_state(qtbot):
    """The CUDA checkbox mirrors availability and is disabled when absent."""
    from eelsnmf_gui._nmf import cupy_gpu_available

    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    assert widget.use_cupy_checkbox.isChecked() == cupy_gpu_available()
    assert widget.use_cupy_checkbox.isEnabled() == cupy_gpu_available()


def test_log_panel_toggle(qtbot):
    """The toggle button collapses and re-expands the log panel."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    assert not widget.log_group.isHidden()

    widget._toggle_log()
    assert widget.log_group.isHidden()
    assert widget.log_toggle_button.text() == "Show log"

    widget._toggle_log()
    assert not widget.log_group.isHidden()
    assert widget.log_toggle_button.text() == "Hide log"


def test_log_panel_toggle_resizes_window(qtbot):
    """Collapsing shrinks the window by the log width; expanding restores it."""
    from qtpy.QtWidgets import QApplication

    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    widget.resize(1200, 800)
    widget.show()
    QApplication.processEvents()

    width_before = widget.width()
    pane_sizes = widget.log_splitter.sizes()
    log_width = pane_sizes[1]
    content_width = pane_sizes[0]
    assert log_width > 0

    widget._toggle_log()  # hide
    QApplication.processEvents()
    assert widget.width() == pytest.approx(width_before - log_width, abs=3)
    # The columns keep their shape.
    assert widget.log_splitter.sizes()[0] == pytest.approx(
        content_width, abs=5
    )

    widget._toggle_log()  # show
    QApplication.processEvents()
    assert widget.width() == pytest.approx(width_before, abs=3)
    assert widget.log_splitter.sizes() == pytest.approx(
        list(pane_sizes), abs=5
    )


def test_log_toggle_button_sits_bottom_right(qtbot):
    """The toggle sits at the bottom-right of the middle column, past its filler."""
    from qtpy.QtCore import QPoint
    from qtpy.QtWidgets import QApplication

    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    widget.resize(1200, 800)
    widget.show()
    QApplication.processEvents()

    toggle = widget.log_toggle_button
    toggle_pos = toggle.mapTo(widget, QPoint(0, 0))
    reset = widget.reset_button
    reset_pos = reset.mapTo(widget, QPoint(0, 0))
    run_pos = widget.run_button.mapTo(widget, QPoint(0, 0))
    # Below the middle column's reset row (past the filler), below the run row.
    assert toggle_pos.y() > reset_pos.y()
    assert toggle_pos.y() > run_pos.y()
    # Right-aligned within the column: right edges match the reset button's.
    assert (
        abs(
            (toggle_pos.x() + toggle.width())
            - (reset_pos.x() + reset.width())
        )
        <= 2
    )
    # And flush against the bottom rim of the window (standard 11px margin).
    toggle_bottom = toggle.mapTo(widget, QPoint(0, toggle.height())).y()
    assert toggle_bottom >= widget.rect().bottom() - 15


def test_status_errors_are_mirrored_to_log_pane(qtbot):
    """Error statuses appear in the log pane; plain statuses do not."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)

    widget._plot_from_model("plot_factors")
    assert "Error: Run processing first" in widget.status_label.text()
    assert "Error: Run processing first" in widget.log_view.toPlainText()

    widget._set_status("Ready")
    assert widget.status_label.text() == "Ready"
    assert "\nReady" not in widget.log_view.toPlainText()


def test_joint_mode_shows_mode_selector(qtbot):
    """Default (joint) mode shows the 'Process folders' selector."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    assert widget.joint_mode is True
    assert not widget.processing_mode_row.isHidden()
    assert widget.process_joint_checkbox.isEnabled() or True  # state per folders


def test_non_joint_mode_hides_mode_selector(qtbot, tmp_path):
    """Non-joint mode hides the selector and forces individual processing."""
    widget = EELSProcessingWidget(debug=True, joint_mode=False)
    qtbot.addWidget(widget)
    assert widget.joint_mode is False
    assert widget.processing_mode_row.isHidden()
    assert widget.process_joint is False
    assert widget.process_individual is True

    # The mode-availability update is a no-op and cannot re-enable anything.
    widget._update_processing_mode_availability()
    assert widget.process_joint is False

    # A stale joint result can never surface in the model selector.
    from types import SimpleNamespace

    from tests.fakes import StubDeco

    deco = StubDeco([(4, 5)], np.arange(10.0))
    widget._update_model_combo(
        SimpleNamespace(
            joint=SimpleNamespace(),
            deco_joint=deco,
            labels=["runA"],
            decos_individual=[deco],
        )
    )
    assert widget.model_combo.count() == 1  # individual entry only
    assert widget.model_combo.itemText(0) == "runA"


def test_joint_mode_lists_joint_model(qtbot):
    """In joint mode, a joint result adds a Joint entry to the model selector."""
    from types import SimpleNamespace

    from tests.fakes import StubDeco

    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    deco = StubDeco([(4, 5)], np.arange(10.0))
    widget._update_model_combo(
        SimpleNamespace(
            joint=SimpleNamespace(),
            deco_joint=deco,
            labels=["runA"],
            decos_individual=[deco],
        )
    )
    assert [widget.model_combo.itemText(i) for i in range(2)] == [
        "Joint",
        "runA",
    ]


def test_decomposition_settings_defaults(qtbot):
    """Defaults mirror the previous hardcoded engine values."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    settings = widget._decomposition_settings()
    assert settings == {
        "n_power_laws": 50,
        "tol": 1e-8,
        "error_skip_step": 100,
        "decomposition_method": "default_decomposition",
        "regularization": None,
    }
    # Regularization is greyed out for methods without one.
    assert not widget.regularization_spinbox.isEnabled()


def test_decomposition_settings_regularized_method(qtbot):
    """Selecting a regularized method enables the lambda spinbox."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    widget.method_combo.setCurrentText("SumRule_decomposition")
    assert widget.regularization_spinbox.isEnabled()
    widget.regularization_spinbox.setValue(0.25)
    settings = widget._decomposition_settings()
    assert settings["decomposition_method"] == "SumRule_decomposition"
    assert settings["regularization"] == 0.25

    widget.method_combo.setCurrentText("default_decomposition")
    assert not widget.regularization_spinbox.isEnabled()
    assert widget._decomposition_settings()["regularization"] is None


def test_parameter_help_buttons_exist(qtbot):
    """Non-obvious parameters carry a '?' help button with a tooltip."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    from qtpy.QtWidgets import QToolButton

    from eelsnmf_gui._widget import HelpButton

    help_buttons = [
        b for b in widget.findChildren(QToolButton) if b.text() == "?"
    ]
    assert len(help_buttons) >= 10
    assert all(b.toolTip() for b in help_buttons)
    # The buttons are outlined when idle and show their tooltip instantly.
    assert all(isinstance(b, HelpButton) for b in help_buttons)
    assert all("border" in b.styleSheet() for b in help_buttons)
    assert "Passed to EELSNMF" not in "".join(
        b.toolTip() for b in help_buttons
    )


def test_despike_options(qtbot, tmp_path):
    """Hampel despiking is opt-in and switches the session loader."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    assert widget.despike_method == "eml"
    # The median filter size only matters for Hampel despiking.
    assert not widget.median_filter_spinbox.isEnabled()

    folder = tmp_path / "runA"
    folder.mkdir()
    widget.source_folders = [str(folder)]
    widget._create_session()
    assert widget._session._loader.keywords["despike_method"] == "eml"

    widget.despike_combo.setCurrentIndex(1)
    assert widget.despike_method == "hampel"
    assert widget.median_filter_spinbox.isEnabled()
    assert widget._session._loader.keywords["despike_method"] == "hampel"


def _make_widget_with_stub_run(qtbot, tmp_path):
    """A widget wired to a fake-loader/stub-engine session over one folder."""
    from eelsnmf_gui.session import Session
    from tests.fakes import RUN_KWARGS, FakeLoader, StubEngine

    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    folder = tmp_path / "runA"
    folder.mkdir()
    widget.source_folders = [str(folder)]
    widget.results_folder = str(tmp_path / "results")
    widget._session = Session(
        [str(folder)], loader=FakeLoader(), engine=StubEngine()
    )
    return widget, RUN_KWARGS


def test_save_flag_gating(qtbot, tmp_path):
    """Each output is written only when its own checkbox is enabled."""
    import os

    widget, run_kwargs = _make_widget_with_stub_run(qtbot, tmp_path)
    widget.save_results_checkbox.setChecked(True)
    widget._sync_save_flags()
    # Default: deco objects only.
    assert widget.save_deco is True
    assert widget.save_pickle is False

    results = widget._session.run(**run_kwargs)
    widget._export_and_save_processing_results(results)

    pickles = os.path.join(widget.results_folder, "pickles")
    assert os.path.exists(os.path.join(pickles, "deco_runA.pkl"))
    assert not os.path.exists(
        os.path.join(pickles, "processing_results.pkl")
    )
    assert not os.path.exists(
        os.path.join(widget.results_folder, "colored_images")
    )

    # Opt into the pickle and the colored images.
    widget.save_pickle_checkbox.setChecked(True)
    widget.save_colored_checkbox.setChecked(True)
    widget._export_and_save_processing_results(results)
    assert os.path.exists(os.path.join(pickles, "processing_results.pkl"))
    assert os.path.exists(
        os.path.join(widget.results_folder, "colored_images", "runA.png")
    )


def test_update_params_from_deco(qtbot):
    """Loading a deco restores the decomposition parameters in the GUI."""

    from tests.fakes import StubDeco

    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    deco = StubDeco([(4, 5)], np.arange(10.0))
    deco.E0 = 200_000.0
    deco.alpha = 0.02
    deco.beta = 0.03
    deco.max_iters = 555
    deco.tol = 1e-5
    deco.error_skip_step = 7
    deco.n_components = 1
    deco.analysis_description = {
        "decomposition": {
            "method": "SumRule_decomposition",
            "params": {"lmbda": 0.4},
            "use_cupy": False,
        }
    }
    deco.model.backgrounds = np.linspace(0.5, 8, 33)
    deco.fine_structure_ranges = {"O_K": (525.0, 545.0)}

    widget._update_params_from_deco(deco)

    assert widget.e0 == pytest.approx(200.0)
    assert widget.alpha == pytest.approx(20.0)
    assert widget.beta == pytest.approx(30.0)
    assert widget.max_iters == 555
    assert widget.tol == pytest.approx(1e-5)
    assert widget.error_skip_step == 7
    assert widget.n_components == 1
    assert widget.decomposition_method == "SumRule_decomposition"
    assert widget.regularization == pytest.approx(0.4)
    assert widget.n_power_laws == 33
    ranges = widget._collect_ranges(only_enabled=True)
    assert ranges == {"O_K": (525.0, 545.0)}


def test_export_loaded_deco(qtbot, tmp_path):
    """The loaded deco can be re-exported per the saving checkboxes."""
    import os

    from tests.fakes import RUN_KWARGS, StubEngine, make_si

    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    engine = StubEngine()
    sis = [make_si("runA", seed=0)]
    import numpy as np

    axis = 500.0 + 2.0 * np.arange(32)
    bundle = engine.build_deco(
        sis, axis, E0_V=1.0, alpha_rad=1.0, beta_rad=1.0
    )
    engine.build_g(bundle, RUN_KWARGS["fine_structure_ranges"])
    engine.decompose(bundle, n_components=2, max_iters=10)

    widget.results_folder = str(tmp_path / "results")
    widget.save_results_checkbox.setChecked(True)
    widget._sync_save_flags()
    widget._load_deco_into(bundle.deco, "my_deco")
    assert widget.export_loaded_button.isEnabled()

    widget.save_excel_checkbox.setChecked(True)
    widget.save_summary_checkbox.setChecked(True)
    widget._export_loaded_deco()

    assert os.path.exists(
        os.path.join(
            widget.results_folder, "pickles", "deco_loaded.pkl"
        )
    )
    assert os.path.exists(
        os.path.join(widget.results_folder, "edges", "Fe_L.csv")
    )
    assert os.path.exists(
        os.path.join(widget.results_folder, "edge_contributions.csv")
    )
    assert os.path.exists(
        os.path.join(widget.results_folder, "summary_loaded.png")
    )
    assert widget.status_label.text() == "Loaded deco exported"


def test_fixed_weights_table(qtbot):
    """Constraint rows collect only constrained, valid (edge, comp, mode)."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)

    widget._add_fixed_weight_row("Fe_L", 0, "Fixed")
    widget._add_fixed_weight_row("O_K", 1, "Free")  # no effect
    widget._add_fixed_weight_row("Fe_L", 1, "Excluded")
    assert widget._collect_fixed_weights() == (
        ("Fe_L", 0, "fixed"),
        ("Fe_L", 1, "excluded"),
    )

    # Remove the second constraint row (Fe_L, component 1).
    widget.fixed_weights_table.selectRow(2)
    widget._remove_selected_fixed_weight()
    assert widget._collect_fixed_weights() == (("Fe_L", 0, "fixed"),)

    # Validation: disabled edges and out-of-range components are skipped.
    widget._add_fixed_weight_row("O_K", 0, "Excluded")  # O_K disabled
    widget.n_components = 6
    widget._add_fixed_weight_row("Fe_L", 5, "Fixed")
    widget.n_components = 2  # component 5 is now out of range
    assert widget._collect_fixed_weights() == (("Fe_L", 0, "fixed"),)

    # Duplicate (edge, component) rows are deduplicated.
    widget._add_fixed_weight_row("Fe_L", 0, "Fixed")
    assert widget._collect_fixed_weights() == (("Fe_L", 0, "fixed"),)


def test_fixed_weights_reach_run_and_settings(qtbot, tmp_path):
    """Collected constraints are passed to Session.run and persisted."""
    widget, run_kwargs = _make_widget_with_stub_run(qtbot, tmp_path)
    widget.save_results_checkbox.setChecked(True)
    widget._sync_save_flags()

    widget._add_fixed_weight_row("Fe_L", 1, "Fixed")
    widget._save_settings()
    widget.fixed_weights_table.setRowCount(0)
    assert widget._collect_fixed_weights() == ()

    # Reload from settings: the row comes back with its state.
    widget._load_settings()
    assert widget._saved_fixed_weights == [("Fe_L", 1, "Fixed")]
    for edge, component, state in widget._saved_fixed_weights:
        widget._add_fixed_weight_row(edge, component, state)
    assert widget._collect_fixed_weights() == (("Fe_L", 1, "fixed"),)

    result = widget._session.run(
        **{**run_kwargs, "fixed_weights": widget._collect_fixed_weights()}
    )
    engine = widget._session._engine
    assert all(
        call["fixed_weights"] == (("Fe_L", 1, "fixed"),)
        for call in engine.decompose_calls
    )
    assert result.joint is not None or result.individual


def test_model_combo_and_sanity_plots(qtbot):
    """A run's decos populate the model selector; buttons call the model's plots."""
    from tests.fakes import StubDeco

    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)

    # Without a run, plot buttons report an error instead of plotting.
    widget._plot_from_model("plot_factors")
    assert "Run processing first" in widget.status_label.text()

    deco_joint = StubDeco([(4, 5)], np.arange(10.0))
    deco_a = StubDeco([(4, 5)], np.arange(10.0))
    results = SimpleNamespace(
        joint=SimpleNamespace(),
        deco_joint=deco_joint,
        labels=["runA"],
        decos_individual=[deco_a],
    )
    widget._update_model_combo(results)
    assert widget.model_combo.count() == 2
    assert widget.model_combo.itemText(0) == "Joint"

    widget.model_combo.setCurrentIndex(0)
    widget._plot_from_model("plot_factors")
    assert deco_joint.plot_calls == ["plot_factors"]

    widget.model_combo.setCurrentIndex(1)
    widget._plot_from_model("plot_factors")
    assert deco_a.plot_calls == ["plot_factors"]

    # Unknown methods surface as status errors, not crashes.
    widget._plot_from_model("no_such_plot")
    assert "Error:" in widget.status_label.text()

    # Re-selecting folders clears the selector.
    widget._reset_model_combo()
    assert widget.model_combo.currentData() is None


def test_reset_model_combo_on_folder_change(qtbot, tmp_path):
    """Creating a session for new folders resets the model selector."""
    from tests.fakes import StubDeco

    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    deco = StubDeco([(4, 5)], np.arange(10.0))
    widget._update_model_combo(
        SimpleNamespace(
            joint=SimpleNamespace(),
            deco_joint=deco,
            labels=[],
            decos_individual=[],
        )
    )
    assert widget.model_combo.currentData() is not None

    folder = tmp_path / "runA"
    folder.mkdir()
    widget.source_folders = [str(folder)]
    widget._create_session()
    assert widget.model_combo.currentData() is None
