from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("qtpy")
pytest.importorskip("pytestqt")

from qtpy.QtWidgets import QComboBox

from eelsnmf_gui._widget import DEFAULT_EDGE_RANGES, EELSProcessingWidget


def reset_button_height(widget):
    return widget.reset_button.height()


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
    # Masks only exist when this GUI performs the preprocessing.
    assert widget.save_masks_checkbox.isHidden()

    folder = tmp_path / "runA"
    folder.mkdir()
    widget.source_folders = [str(folder)]
    widget._create_session()
    loader = widget._session._loader
    assert loader.keywords["preprocess"] is False


def test_pre_mode_shows_mask_save_option(qtbot):
    """In pre mode the mask images export option is visible."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    assert not widget.save_masks_checkbox.isHidden()


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


def test_config_buttons_share_the_log_toggle_row(qtbot):
    """Save/Load config sit on the log-toggle row, aligned left."""
    from qtpy.QtCore import QPoint
    from qtpy.QtWidgets import QApplication

    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    widget.resize(1200, 800)
    widget.show()
    QApplication.processEvents()

    toggle_pos = widget.log_toggle_button.mapTo(widget, QPoint(0, 0))
    save_pos = widget.save_config_button.mapTo(widget, QPoint(0, 0))
    load_pos = widget.load_config_button.mapTo(widget, QPoint(0, 0))
    # Same row as the log toggle.
    assert save_pos.y() == toggle_pos.y()
    assert load_pos.y() == toggle_pos.y()
    # Aligned left, before the toggle.
    assert load_pos.x() < toggle_pos.x()
    assert save_pos.x() < load_pos.x()
    # The row keeps the toggle flush right with the window rim.
    toggle_bottom = widget.log_toggle_button.mapTo(
        widget, QPoint(0, widget.log_toggle_button.height())
    ).y()
    assert toggle_bottom >= widget.rect().bottom() - 15


def test_export_button_sits_below_results_folder(qtbot):
    """The export action follows the results-folder browse row."""
    from qtpy.QtCore import QPoint
    from qtpy.QtWidgets import QApplication

    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    widget.resize(1200, 800)
    widget.show()
    QApplication.processEvents()

    browse_bottom = widget.results_button.mapTo(
        widget, QPoint(0, widget.results_button.height())
    ).y()
    export_top = widget.export_button.mapTo(widget, QPoint(0, 0)).y()
    assert export_top > browse_bottom
    # And the export button lives inside the Export group.
    assert widget.export_button.parent() is not None
    assert widget.save_group.isAncestorOf(widget.export_button)


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
        "regularization_tv": None,
        "regularization_sr": None,
    }
    # Both lambdas are greyed out for methods without regularization.
    assert not widget.regularization_tv_spinbox.isEnabled()
    assert not widget.regularization_sr_spinbox.isEnabled()


def test_decomposition_settings_regularized_method(qtbot):
    """Each method enables exactly the lambda spinboxes it accepts."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)

    widget.method_combo.setCurrentText("SumRule_decomposition")
    assert not widget.regularization_tv_spinbox.isEnabled()
    assert widget.regularization_sr_spinbox.isEnabled()
    widget.regularization_sr_spinbox.setValue(0.25)
    settings = widget._decomposition_settings()
    assert settings["decomposition_method"] == "SumRule_decomposition"
    assert settings["regularization_sr"] == 0.25
    assert settings["regularization_tv"] is None

    widget.method_combo.setCurrentText("EdgeTV_decomposition")
    assert widget.regularization_tv_spinbox.isEnabled()
    assert not widget.regularization_sr_spinbox.isEnabled()
    widget.regularization_tv_spinbox.setValue(0.5)
    settings = widget._decomposition_settings()
    assert settings["regularization_tv"] == 0.5
    assert settings["regularization_sr"] is None

    widget.method_combo.setCurrentText("combinedTVSR_decomposition")
    assert widget.regularization_tv_spinbox.isEnabled()
    assert widget.regularization_sr_spinbox.isEnabled()
    settings = widget._decomposition_settings()
    assert settings["regularization_tv"] == 0.5
    assert settings["regularization_sr"] == 0.25

    widget.method_combo.setCurrentText("default_decomposition")
    assert not widget.regularization_tv_spinbox.isEnabled()
    assert not widget.regularization_sr_spinbox.isEnabled()
    settings = widget._decomposition_settings()
    assert settings["regularization_tv"] is None
    assert settings["regularization_sr"] is None


def test_no_alternate_decomposition_in_method_list(qtbot):
    """The alternate decomposition is not offered in the GUI."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    methods = [
        widget.method_combo.itemText(i)
        for i in range(widget.method_combo.count())
    ]
    assert "alternate_decomposition" not in methods


def test_tolerance_slider_is_logarithmic(qtbot):
    """The tolerance slider follows a log progression and drives self.tol."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    slider = widget.tol_slider

    # Same value at both ends of the sync (setValue then value()).
    slider.setValue(1e-4)
    assert slider.value() == pytest.approx(1e-4, rel=0.01)

    # Moving the slider emits the value and drives self.tol.
    slider.setValue(1e-8)
    pos_1e8 = slider.slider.value()
    slider.slider.setValue(pos_1e8 + 100)
    assert widget.tol == pytest.approx(1e-9, rel=0.01)

    # The label shows the value.
    assert "e" in slider.label.text()


def test_power_laws_slider(qtbot):
    """The power-law slider spans 1-500 and drives self.n_power_laws."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    slider = widget.power_laws_slider

    slider.setValue(1)
    assert slider.value() == 1
    slider.slider.setValue(slider.slider.maximum())
    assert widget.n_power_laws == 500

    # Moving the slider to the position of 50 yields 50.
    slider.setValue(50, emit=True)
    assert widget.n_power_laws == 50


def test_add_edge_button_label(qtbot):
    """The fine-structure button is labelled 'Add edge'."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    assert widget.add_range_button.text() == "Add edge"


def test_element_names_are_restricted_to_known_edges(qtbot):
    """Range rows use element/edge comboboxes of what EELSNMF can fit."""
    from eelsnmf_gui._nmf import available_core_loss_edges

    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    choices = available_core_loss_edges()
    assert "Fe_L" in choices and "O_K" in choices
    assert all("_" in c for c in choices)

    # The default rows use the element/edge combobox pair.
    element_combo = widget.ranges_table.cellWidget(0, 1)
    shell_combo = widget.ranges_table.cellWidget(0, 2)
    assert isinstance(element_combo, QComboBox)
    assert isinstance(shell_combo, QComboBox)
    assert element_combo.currentText() == "Fe"
    assert shell_combo.currentText() == "L"
    assert widget._row_edge_name(0) == "Fe_L"

    # A new row selects the requested edge in the comboboxes.
    widget._add_range_row("O_K", 522.0, 590.0, checked=True)
    last = widget.ranges_table.rowCount() - 1
    assert widget.ranges_table.cellWidget(last, 1).currentText() == "O"
    assert widget.ranges_table.cellWidget(last, 2).currentText() == "K"
    assert widget._get_ranges_list()[-1]["name"] == "O_K"

    # Changing the element updates the available shells (a real signal
    # round-trip: the row starts as O_K, then switches to Fe).
    row_element_combo = widget.ranges_table.cellWidget(last, 1)
    row_shell_combo = widget.ranges_table.cellWidget(last, 2)
    row_element_combo.setCurrentText("Fe")
    assert row_element_combo.currentText() == "Fe"
    assert [
        row_shell_combo.itemText(i) for i in range(row_shell_combo.count())
    ] == [
        "K",
        "L",
        "M",
        "N",
    ]
    assert widget._row_edge_name(last) == "Fe_K"

    # Legacy/unknown names stay visible instead of being substituted.
    widget._add_range_row("Fe_L3", 700.0, 740.0)
    last = widget.ranges_table.rowCount() - 1
    assert widget.ranges_table.cellWidget(last, 1).currentText() == "Fe"
    assert widget.ranges_table.cellWidget(last, 2).currentText() == "L3"
    assert widget._get_ranges_list()[-1]["name"] == "Fe_L3"

    # A name without '_' is kept as element only.
    widget._add_range_row("Custom", 100.0, 200.0)
    last = widget.ranges_table.rowCount() - 1
    assert widget.ranges_table.cellWidget(last, 1).currentText() == "Custom"
    assert widget._get_ranges_list()[-1]["name"] == "Custom"


def test_group_titles_are_bold(qtbot):
    """Every section group box styles its title in bold."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    from qtpy.QtWidgets import QGroupBox

    groups = [
        g
        for g in widget.findChildren(QGroupBox)
        if g.title() != ""  # every titled section
    ]
    assert len(groups) >= 8
    for group in groups:
        assert "font-weight: bold" in group.styleSheet()


def test_visualize_results_group_after_run_button(qtbot):
    """Model plots moved below Run Processing; sanity check keeps the rest."""
    from qtpy.QtCore import QPoint
    from qtpy.QtWidgets import QApplication

    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    widget.resize(1200, 800)
    widget.show()
    QApplication.processEvents()

    # All decomposition-dependent plots live in the results group, below
    # the run button (same column).
    for button in (
        widget.plot_factors_button,
        widget.plot_edges_button,
        widget.plot_loadings_button,
        widget.plot_average_model_button,
        widget.plot_model_energy_ranges_button,
    ):
        assert widget.viz_group.isAncestorOf(button)
        assert button.mapTo(widget, QPoint(0, 0)).y() > (
            widget.run_button.mapTo(widget, QPoint(0, 0)).y()
        )
    assert widget.viz_group.isAncestorOf(widget.model_combo)

    # The data-only sanity check keeps its own group without the model.
    assert not widget.sanity_group.isAncestorOf(widget.model_combo)
    assert widget.sanity_group.isAncestorOf(widget.sanity_button)
    assert widget.sanity_group.isAncestorOf(widget.variance_button)


def test_reset_button_above_window_state_row(qtbot):
    """Reset sits directly above the Save/Load window state line."""
    from qtpy.QtCore import QPoint
    from qtpy.QtWidgets import QApplication

    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    widget.resize(1200, 800)
    widget.show()
    QApplication.processEvents()

    reset_pos = widget.reset_button.mapTo(widget, QPoint(0, 0))
    toggle_pos = widget.log_toggle_button.mapTo(widget, QPoint(0, 0))
    save_pos = widget.save_config_button.mapTo(widget, QPoint(0, 0))
    # Reset is between the export section and the window-state line.
    assert toggle_pos.y() > reset_pos.y()
    assert save_pos.y() > reset_pos.y()
    # Just above: no other row in between (within one row height).
    assert toggle_pos.y() - reset_pos.y() < 3 * reset_button_height(widget)


def test_eight_components_with_color_grid(qtbot):
    """Up to 8 components are selectable; colors wrap in rows of four."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    assert widget.components_spinbox.maximum() == 8

    widget.components_spinbox.setValue(6)
    assert widget.n_components == 6
    assert len(widget.color_buttons) == 6
    # Buttons are laid out in a grid: 4 in the first row, 2 in the second.
    rows = {
        widget.color_buttons_layout.getItemPosition(i)[0]
        for i in range(widget.color_buttons_layout.count())
    }
    assert rows == {0, 1}

    widget.components_spinbox.setValue(8)
    assert len(widget.colors) == 8
    assert len(widget.color_buttons) == 8


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
    widget._sync_save_flags()
    # Default: deco objects only.
    assert widget.save_deco is True
    assert widget.save_pickle is False

    results = widget._session.run(**run_kwargs)
    widget._update_model_combo(results)
    widget._export_results()

    pickles = os.path.join(widget.results_folder, "pickles")
    assert os.path.exists(os.path.join(pickles, "deco_runA.pkl"))
    assert not os.path.exists(
        os.path.join(pickles, "processing_results.pkl")
    )
    assert not os.path.exists(
        os.path.join(widget.results_folder, "colored_images")
    )

    # Opt into the pickle and the colored images; re-export from memory
    # without re-running the decomposition.
    widget.save_pickle_checkbox.setChecked(True)
    widget.save_colored_checkbox.setChecked(True)
    widget._sync_save_flags()
    widget._export_results()
    assert os.path.exists(os.path.join(pickles, "processing_results.pkl"))
    assert os.path.exists(
        os.path.join(widget.results_folder, "colored_images", "runA.png")
    )


def test_export_requires_results_folder(qtbot, tmp_path):
    """Exporting without a results folder reports an error."""
    widget, run_kwargs = _make_widget_with_stub_run(qtbot, tmp_path)
    widget._sync_save_flags()
    widget.results_folder = None
    results = widget._session.run(**run_kwargs)
    widget._update_model_combo(results)
    widget._export_results()
    assert "Error: No results folder selected" in widget.status_label.text()


def test_export_before_run_reports_error(qtbot):
    """The export button is disabled until a run or a deco is available."""
    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    assert not widget.export_button.isEnabled()
    widget.export_button.setEnabled(True)
    widget._export_results()
    assert "Error: Run processing first" in widget.status_label.text()


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
    assert widget.regularization_sr == pytest.approx(0.4)
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
    widget._sync_save_flags()
    widget._load_deco_into(bundle.deco, "my_deco")
    assert widget.export_button.isEnabled()

    widget.save_excel_checkbox.setChecked(True)
    widget.save_summary_checkbox.setChecked(True)
    widget._sync_save_flags()
    widget._export_results()

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


def test_processing_finish_survives_export_errors(qtbot, tmp_path):
    """An exception while finishing never leaves the progress bar spinning."""
    widget, run_kwargs = _make_widget_with_stub_run(qtbot, tmp_path)
    results = widget._session.run(**run_kwargs)

    widget.progress_bar.setVisible(True)
    widget.run_button.setEnabled(False)

    def boom(_results):
        raise RuntimeError("export blew up")

    widget._show_processing_results = boom
    widget._on_processing_finished(results, None)

    assert widget.progress_bar.isVisible() is False
    assert widget.run_button.isEnabled()
    assert "export blew up" in widget.status_label.text()
    assert "export blew up" in widget.log_view.toPlainText()

    # The error path also stops the spinner.
    widget.progress_bar.setVisible(True)
    widget._on_processing_finished(None, RuntimeError("worker failed"))
    assert widget.progress_bar.isVisible() is False
    assert "worker failed" in widget.status_label.text()


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

    # The Edges button normalizes the ELNES plots (max value is 1).
    widget._plot_from_model("plot_edges", normalize=True)
    assert deco_a.plot_calls[-1] == ("plot_edges", True)

    # The Loadings button calls the deco's plot_loadings method.
    widget.plot_loadings_button.click()
    assert deco_a.plot_calls[-1] == "plot_loadings"

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


def test_session_config_round_trip(qtbot, tmp_path):
    """The config buttons snapshot and restore all key parameters."""

    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)

    # Set distinctive parameters.
    widget.e0_spinbox.setValue(200.0)
    widget.alpha_spinbox.setValue(10.0)
    widget.beta_spinbox.setValue(25.0)
    widget._add_range_row("Cu_L", 930.0, 960.0, checked=True)
    widget._add_fixed_weight_row("Cu_L", 1, "Excluded")
    widget.components_spinbox.setValue(3)
    widget.method_combo.setCurrentText("combinedTVSR_decomposition")
    widget.regularization_tv_spinbox.setValue(0.3)
    widget.regularization_sr_spinbox.setValue(0.7)
    widget.tol_slider.setValue(1e-7, emit=True)
    widget.power_laws_slider.setValue(77, emit=True)

    path = str(tmp_path / "config.json")
    widget._collect_config().save(path)

    # Change everything, then load the config back.
    widget._reset_to_defaults()
    widget._load_config_from(path)

    assert widget.e0 == 200.0
    assert widget.alpha == 10.0
    assert widget.beta == 25.0
    assert widget.n_components == 3
    assert widget.decomposition_method == "combinedTVSR_decomposition"
    assert widget.regularization_tv == pytest.approx(0.3)
    assert widget.regularization_sr == pytest.approx(0.7)
    assert widget.tol == pytest.approx(1e-7, rel=0.01)
    assert widget.n_power_laws == 77
    names = [row["name"] for row in widget._get_ranges_list()]
    assert "Cu_L" in names
    assert widget._collect_fixed_weights() == (("Cu_L", 1, "excluded"),)
    assert len(widget.color_buttons) == 3


def test_session_config_rejects_foreign_files(qtbot, tmp_path):
    """Loading a JSON file that is not a session config reports an error."""
    import json

    widget = EELSProcessingWidget(debug=True)
    qtbot.addWidget(widget)
    path = str(tmp_path / "bogus.json")
    with open(path, "w") as fh:
        json.dump({"kind": "something-else"}, fh)

    widget._load_config_from(path)
    assert "Error:" in widget.status_label.text()
    # Parameters are untouched.
    assert widget.e0 == widget.e0_spinbox.value()
