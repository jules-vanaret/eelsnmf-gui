"""
This module contains Qt widgets for EELS spectroscopy processing and analysis.
"""

import logging
import os
import pickle
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from qtpy.QtCore import QObject, QPoint, QSettings, Qt, QThread, Signal
from qtpy.QtGui import QColor, QTextCursor, QToolTip
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListView,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from ._analysis import (  # noqa: F401  (re-exported for tests)
    apply_colors,
    component_maps,
    component_maps_from_deco,
    edge_totals,
    extract_edge_collection,
    order_by_power,
    summary_rows,
)
from ._export import (
    export_edge_contributions_summary_to_csv,
    export_edges_by_label_to_csv,
    save_colored_images,
    save_component_images,
    save_mask_images,
    save_original_images,
    save_summary_plot,
)
from ._file_dialog import (
    MultiFolderDialog,
)
from ._labels import (
    split_common_label_parts,
)
from ._nmf import cupy_gpu_available
from ._plotting import plot_energy_ranges
from ._preprocess import load_eml_folders
from .session import BeamParams, MaskParams, Session

# Default edge energy ranges (in eV)
DEFAULT_EDGE_RANGES = {
    "Fe_L": (700.0, 740.0),
    "O_K": (522.0, 590.0),
}

# Decomposition methods exposed by the EELSNMF package.
DECOMPOSITION_METHODS = (
    "default_decomposition",
    "default_kl_decomposition",
    "alternate_decomposition",
    "SumRule_decomposition",
    "EdgeTV_decomposition",
    "combinedTVSR_decomposition",
)
# Methods accepting a regularization lambda.
REGULARIZED_METHODS = {
    "SumRule_decomposition",
    "EdgeTV_decomposition",
    "combinedTVSR_decomposition",
}


class _ProcessingWorker(QObject):
    finished = Signal(object, object)

    def __init__(self, session: Session, run_kwargs: dict):
        super().__init__()
        self._session = session
        self._run_kwargs = run_kwargs

    def run(self):
        try:
            results = self._session.run(**self._run_kwargs)
            self.finished.emit(results, None)
        except Exception as exc:  # pragma: no cover - UI error path
            self.finished.emit(None, exc)


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class HelpButton(QToolButton):
    """'?' button that shows its tooltip immediately on hover."""

    def enterEvent(self, event):
        super().enterEvent(event)
        pos = self.mapToGlobal(QPoint(0, self.height()))
        QToolTip.showText(pos, self.toolTip(), self)

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)


class _QtLogHandler(logging.Handler, QObject):
    message_emitted = Signal(str)

    def __init__(self, target: QPlainTextEdit):
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self._target = target
        self.message_emitted.connect(self._append)

    def emit(self, record):
        raw_message = record.getMessage()
        if "\r" in raw_message:
            self.message_emitted.emit("\r" + raw_message.split("\r")[-1])
            return
        message = self.format(record)
        self.message_emitted.emit(message)

    def _append(self, message: str):
        if message.startswith("\r"):
            text = message[1:]
            cursor = self._target.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.movePosition(
                QTextCursor.StartOfBlock, QTextCursor.KeepAnchor
            )
            cursor.removeSelectedText()
            cursor.insertText(text)
            self._target.setTextCursor(cursor)
            return
        self._target.appendPlainText(message)


class _StdoutStderrRedirect:
    def __init__(self, logger: logging.Logger, level: int):
        self._logger = logger
        self._level = level
        self._line = ""
        self._saw_carriage = False

    def write(self, message: str):
        if not message:
            return
        for ch in message:
            if ch == "\r":
                self._line = ""
                self._saw_carriage = True
                continue
            if ch == "\n":
                line = self._line.strip()
                if line:
                    if self._saw_carriage:
                        self._logger.log(self._level, "\r" + line)
                    else:
                        self._logger.log(self._level, line)
                self._line = ""
                self._saw_carriage = False
                continue
            self._line += ch

    def flush(self):
        line = self._line.strip()
        if line:
            if self._saw_carriage:
                self._logger.log(self._level, "\r" + line)
            else:
                self._logger.log(self._level, line)
        self._line = ""
        self._saw_carriage = False


class EELSProcessingWidget(QWidget):
    """Widget for EELS NMF processing with configurable parameters."""

    def __init__(
        self,
        debug: bool = False,
        pre_mode: bool = True,
        joint_mode: bool = True,
    ):
        super().__init__()
        self.debug = debug
        # In pre_mode the GUI shows the preprocessing controls and the loader
        # runs ZLP alignment + despiking. Otherwise the loaded SIs are
        # assumed already preprocessed and the section is hidden.
        self.pre_mode = pre_mode
        # In joint_mode the GUI offers joint processing of several folders.
        # Otherwise the "Process folders" controls are hidden and individual
        # processing is forced.
        self.joint_mode = joint_mode

        # Initialize parameter values
        self.e0 = 300.0  # keV
        self.alpha = 15.4  # mrad
        self.beta = 18.46  # mrad
        self.max_iters = 2000
        self.median_filter_size = 11
        self._saved_ranges = None
        self.n_components = 2
        self.tol = 1e-8
        self.error_skip_step = 100
        self.decomposition_method = "default_decomposition"
        self.regularization = 0.1
        self.n_power_laws = 50
        self.mask_enabled = False
        self.mask_dilation_iters = 1
        self.colors = [
            np.array([1.0, 0.0, 0.0]),  # Red
            np.array([0.0, 1.0, 0.0]),  # Green
            np.array([0.0, 0.0, 1.0]),  # Blue
            np.array([1.0, 0.5, 0.0]),  # Orange
        ]
        self.source_folder = None
        self.source_folders = []
        self.results_folder = None
        self.save_results = False
        self.save_deco = True  # deco objects are the default saved output
        self.save_pickle = False
        self.save_colored_images = False
        self.save_masks = False
        self.save_original_images = False
        self.save_summary_plot = False
        self.show_summary_plot = True
        self.save_excel = False
        self.process_individual = True
        self.despike_method = "eml"
        self._loaded_deco = None
        self.process_joint = False
        self._worker_thread = None
        self._worker = None
        self._logger = None
        self._log_handler = None
        self._original_stdout = None
        self._original_stderr = None
        self._session: Session | None = None
        self._last_results = None
        self._log_splitter_sizes = None
        self._app = QApplication.instance()
        if self._app is not None:
            self._app.aboutToQuit.connect(
                lambda: self._cleanup_thread("app quit")
            )

        self._load_settings()

        if not self.joint_mode:
            # Joint processing is not accessible: force individual-only.
            self.process_joint = False
            self.process_individual = True

        # Build UI
        self._build_ui()
        self._setup_logging()

    def _build_ui(self):
        """Construct the widget layout."""
        left_column_layout = QVBoxLayout()
        middle_column_layout = QVBoxLayout()
        # No bottom margin so the log toggle can sit flush with the window rim.
        middle_column_layout.setContentsMargins(9, 9, 9, 0)

        # Data Input Section
        input_group = QGroupBox("Data Input")
        input_layout = QVBoxLayout()

        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("No folder(s) selected")
        self.folder_button = QPushButton("Browse...")
        self.folder_button.clicked.connect(self._select_folder)
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(self.folder_button)
        input_layout.addLayout(folder_layout)

        input_group.setLayout(input_layout)
        left_column_layout.addWidget(input_group)

        # Preprocessing Parameters Section
        preprocessing_group = QGroupBox("Preprocessing Parameters")
        preprocessing_layout = QVBoxLayout()

        median_filter_layout = QHBoxLayout()
        median_filter_layout.addWidget(QLabel("Median filter size (Hampel):"))
        self.median_filter_spinbox = NoWheelSpinBox()
        self.median_filter_spinbox.setRange(1, 101)
        self.median_filter_spinbox.setValue(self.median_filter_size)
        self.median_filter_spinbox.setSingleStep(1)
        self.median_filter_spinbox.valueChanged.connect(
            self._on_median_filter_size_changed
        )
        median_filter_layout.addWidget(self.median_filter_spinbox)
        median_filter_layout.addWidget(
            self._make_help_button(
                "Size of the Hampel median filter along the energy axis.\n"
                "Only used when Hampel despiking is enabled."
            )
        )
        preprocessing_layout.addLayout(median_filter_layout)

        despike_layout = QHBoxLayout()
        despike_layout.addWidget(QLabel("Despiking:"))
        self.despike_combo = QComboBox()
        self.despike_combo.addItem("EML despike (default)", "eml")
        self.despike_combo.addItem("Hampel filter (median-based)", "hampel")
        index = self.despike_combo.findData(self.despike_method)
        self.despike_combo.setCurrentIndex(max(index, 0))
        self.despike_combo.currentIndexChanged.connect(
            self._on_despike_changed
        )
        despike_layout.addWidget(self.despike_combo)
        despike_layout.addWidget(
            self._make_help_button(
                "Despiking algorithm applied after ZLP alignment:\n"
                "- EML despike (default): the EELSNMF despike algorithm,\n"
                "  run 10 times on each image\n"
                "- Hampel filter: a median-based filter on each spectrum,\n"
                "  using the median filter size above."
            )
        )
        despike_layout.addStretch()
        preprocessing_layout.addLayout(despike_layout)
        self._update_median_availability()

        mask_layout = QHBoxLayout()
        self.mask_checkbox = QCheckBox("Mask dark pixels")
        self.mask_checkbox.setChecked(self.mask_enabled)
        self.mask_checkbox.stateChanged.connect(self._on_mask_enabled_changed)
        mask_layout.addWidget(self.mask_checkbox)
        mask_layout.addWidget(
            self._make_help_button(
                "Exclude dark pixels (vacuum / substrate) from the "
                "decomposition.\nA threshold mask is computed from the "
                "energy-aggregated image."
            )
        )

        mask_layout.addStretch()
        mask_layout.addWidget(QLabel("Dilation iters"))
        self.mask_dilation_spinbox = NoWheelSpinBox()
        self.mask_dilation_spinbox.setRange(0, 20)
        self.mask_dilation_spinbox.setSingleStep(1)
        self.mask_dilation_spinbox.setValue(self.mask_dilation_iters)
        self.mask_dilation_spinbox.setEnabled(self.mask_enabled)
        self.mask_dilation_spinbox.valueChanged.connect(
            self._on_mask_dilation_iters_changed
        )
        mask_layout.addWidget(self.mask_dilation_spinbox)
        preprocessing_layout.addLayout(mask_layout)

        self.mask_preview_button = QPushButton("Preview mask")
        self.mask_preview_button.clicked.connect(self._preview_mask)
        preprocessing_layout.addWidget(self.mask_preview_button)

        preprocessing_group.setLayout(preprocessing_layout)
        left_column_layout.addWidget(preprocessing_group)
        self.preprocessing_group = preprocessing_group
        if not self.pre_mode:
            # SIs are assumed already preprocessed: hide the whole section
            # (median filter size and mask controls are not applicable).
            preprocessing_group.setVisible(False)

        # Microscope Parameters Section
        microscope_group = QGroupBox("Microscope Parameters")
        microscope_layout = QVBoxLayout()

        # Beam energy (E0)
        e0_layout = QHBoxLayout()
        e0_layout.addWidget(QLabel("Beam Energy (keV):"))
        self.e0_spinbox = NoWheelDoubleSpinBox()
        self.e0_spinbox.setRange(100.0, 500.0)
        self.e0_spinbox.setValue(self.e0)
        self.e0_spinbox.setSingleStep(10.0)
        self.e0_spinbox.valueChanged.connect(self._on_e0_changed)
        e0_layout.addWidget(self.e0_spinbox)
        e0_layout.addWidget(
            self._make_help_button(
                "Acceleration voltage of the microscope, in keV."
            )
        )
        microscope_layout.addLayout(e0_layout)

        # Convergence and collection angles in one line
        angles_layout = QHBoxLayout()
        angles_layout.addWidget(QLabel("Angles:"))

        angles_layout.addWidget(QLabel("α"))
        self.alpha_spinbox = NoWheelDoubleSpinBox()
        self.alpha_spinbox.setRange(0.0, 1000.0)
        self.alpha_spinbox.setValue(self.alpha)
        self.alpha_spinbox.setSingleStep(0.1)
        self.alpha_spinbox.valueChanged.connect(self._on_alpha_changed)
        angles_layout.addWidget(self.alpha_spinbox)
        angles_layout.addWidget(
            self._make_help_button(
                "Convergence semi-angle of the beam, in mrad."
            )
        )

        angles_layout.addWidget(QLabel("β"))
        self.beta_spinbox = NoWheelDoubleSpinBox()
        self.beta_spinbox.setRange(0.0, 1000.0)
        self.beta_spinbox.setValue(self.beta)
        self.beta_spinbox.setSingleStep(0.1)
        self.beta_spinbox.valueChanged.connect(self._on_beta_changed)
        angles_layout.addWidget(self.beta_spinbox)
        angles_layout.addWidget(
            self._make_help_button(
                "Collection semi-angle of the detector, in mrad."
            )
        )
        microscope_layout.addLayout(angles_layout)

        microscope_group.setLayout(microscope_layout)
        left_column_layout.addWidget(microscope_group)

        # Fine Structure Definition Section
        structure_group = QGroupBox("Fine Structure Definition")
        structure_layout = QVBoxLayout()

        self.ranges_table = QTableWidget(0, 4)
        self.ranges_table.setHorizontalHeaderLabels(
            ["Use", "Name", "Min (eV)", "Max (eV)"]
        )
        self.ranges_table.verticalHeader().setVisible(False)
        self.ranges_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ranges_table.setSelectionMode(QAbstractItemView.SingleSelection)
        header = self.ranges_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._populate_ranges_table(self._saved_ranges)
        structure_layout.addWidget(self.ranges_table)

        ranges_buttons_layout = QHBoxLayout()
        self.add_range_button = QPushButton("Add range")
        self.add_range_button.clicked.connect(self._add_range_dialog)
        ranges_buttons_layout.addWidget(self.add_range_button)
        self.remove_range_button = QPushButton("Remove selected")
        self.remove_range_button.clicked.connect(self._remove_selected_range)
        ranges_buttons_layout.addWidget(self.remove_range_button)
        ranges_buttons_layout.addWidget(
            self._make_help_button(
                "Fine-structure (ELNES) energy windows, in eV.\n"
                "Each enabled entry defines one edge to fit: its name "
                "and its (min, max) energy window."
            )
        )
        ranges_buttons_layout.addStretch()
        structure_layout.addLayout(ranges_buttons_layout)

        # Number of power-law backgrounds in the spectral model (G)
        power_laws_layout = QHBoxLayout()
        power_laws_layout.addWidget(QLabel("Number of power laws:"))
        self.power_laws_spinbox = NoWheelSpinBox()
        self.power_laws_spinbox.setRange(1, 500)
        self.power_laws_spinbox.setValue(self.n_power_laws)
        self.power_laws_spinbox.setSingleStep(5)
        self.power_laws_spinbox.valueChanged.connect(
            self._on_n_power_laws_changed
        )
        power_laws_layout.addWidget(self.power_laws_spinbox)
        power_laws_layout.addWidget(
            self._make_help_button(
                "Number of power-law backgrounds used to fit the "
                "background signal (exponents spanning 0.5-8)."
            )
        )
        power_laws_layout.addStretch()
        structure_layout.addLayout(power_laws_layout)

        structure_group.setLayout(structure_layout)
        left_column_layout.addWidget(structure_group)

        # Visual Sanity Check Section
        sanity_group = QGroupBox("Visual Sanity Check")
        sanity_layout = QVBoxLayout()
        self.sanity_button = QPushButton("Plot energy ranges")
        self.sanity_button.clicked.connect(self._plot_energy_ranges)
        sanity_layout.addWidget(self.sanity_button)
        self.variance_button = QPushButton("Plot explained variance ratio")
        self.variance_button.clicked.connect(
            self._plot_explained_variance_ratio
        )
        sanity_layout.addWidget(self.variance_button)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("None (run processing first)", None)
        model_layout.addWidget(self.model_combo, stretch=1)
        sanity_layout.addLayout(model_layout)

        model_buttons = QGridLayout()
        self.plot_factors_button = QPushButton("Factors")
        self.plot_factors_button.clicked.connect(
            lambda: self._plot_from_model("plot_factors")
        )
        self.plot_edges_button = QPushButton("Edges")
        self.plot_edges_button.clicked.connect(
            lambda: self._plot_from_model("plot_edges")
        )
        self.plot_average_model_button = QPushButton("Average model")
        self.plot_average_model_button.clicked.connect(
            lambda: self._plot_from_model("plot_average_model")
        )
        self.plot_model_energy_ranges_button = QPushButton(
            "Model energy ranges"
        )
        self.plot_model_energy_ranges_button.clicked.connect(
            lambda: self._plot_from_model("plot_energy_ranges")
        )
        model_buttons.addWidget(self.plot_factors_button, 0, 0)
        model_buttons.addWidget(self.plot_edges_button, 0, 1)
        model_buttons.addWidget(self.plot_average_model_button, 1, 0)
        model_buttons.addWidget(self.plot_model_energy_ranges_button, 1, 1)
        sanity_layout.addLayout(model_buttons)

        sanity_group.setLayout(sanity_layout)
        left_column_layout.addWidget(sanity_group)

        # Processing Parameters Section
        processing_group = QGroupBox("Processing Parameters")
        processing_layout = QVBoxLayout()

        # Max iterations
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Max Iterations:"))
        self.iter_spinbox = NoWheelSpinBox()
        self.iter_spinbox.setRange(100, 10000)
        self.iter_spinbox.setValue(self.max_iters)
        self.iter_spinbox.setSingleStep(100)
        self.iter_spinbox.valueChanged.connect(self._on_max_iters_changed)
        iter_layout.addWidget(self.iter_spinbox)
        iter_layout.addWidget(
            self._make_help_button(
                "Maximum number of NMF iterations.\n"
                "The decomposition stops earlier if the convergence "
                "tolerance is reached."
            )
        )
        processing_layout.addLayout(iter_layout)

        # Convergence tolerance
        tol_layout = QHBoxLayout()
        tol_layout.addWidget(QLabel("Convergence tolerance:"))
        self.tol_spinbox = NoWheelDoubleSpinBox()
        self.tol_spinbox.setRange(1e-12, 1.0)
        self.tol_spinbox.setDecimals(12)
        self.tol_spinbox.setSingleStep(1e-8)
        self.tol_spinbox.setValue(self.tol)
        self.tol_spinbox.valueChanged.connect(self._on_tol_changed)
        tol_layout.addWidget(self.tol_spinbox)
        tol_layout.addWidget(
            self._make_help_button(
                "Convergence tolerance of the NMF: the decomposition "
                "stops when the relative change of the total error "
                "between two error evaluations falls below this "
                "value."
            )
        )
        processing_layout.addLayout(tol_layout)

        # Error skip step
        skip_layout = QHBoxLayout()
        skip_layout.addWidget(QLabel("Error skip step:"))
        self.error_skip_spinbox = NoWheelSpinBox()
        self.error_skip_spinbox.setRange(1, 100000)
        self.error_skip_spinbox.setValue(self.error_skip_step)
        self.error_skip_spinbox.setSingleStep(10)
        self.error_skip_spinbox.valueChanged.connect(
            self._on_error_skip_step_changed
        )
        skip_layout.addWidget(self.error_skip_spinbox)
        skip_layout.addWidget(
            self._make_help_button(
                "How often (in iterations) the full reconstruction "
                "error is evaluated for the convergence test.\n"
                "Increase for speed on large datasets."
            )
        )
        processing_layout.addLayout(skip_layout)

        # Decomposition method and regularization lambda
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        for method_name in DECOMPOSITION_METHODS:
            self.method_combo.addItem(method_name)
        self.method_combo.setCurrentText(self.decomposition_method)
        self.method_combo.currentTextChanged.connect(
            self._on_decomposition_method_changed
        )
        method_layout.addWidget(self.method_combo, stretch=1)
        method_layout.addWidget(
            self._make_help_button(
                "Algorithm used for the decomposition.\n"
                "default: standard NMF; default_kl: KL divergence; "
                "alternate: alternating background/ELNES updates; "
                "SumRule / EdgeTV / combinedTVSR: regularized variants "
                "(these use the lambda parameter)."
            )
        )
        processing_layout.addLayout(method_layout)

        reg_layout = QHBoxLayout()
        reg_layout.addWidget(QLabel("Regularization \u03bb:"))
        self.regularization_spinbox = NoWheelDoubleSpinBox()
        self.regularization_spinbox.setRange(0.0, 1000.0)
        self.regularization_spinbox.setDecimals(4)
        self.regularization_spinbox.setSingleStep(0.01)
        self.regularization_spinbox.setValue(self.regularization)
        self.regularization_spinbox.valueChanged.connect(
            self._on_regularization_changed
        )
        reg_layout.addWidget(self.regularization_spinbox)
        reg_layout.addWidget(
            self._make_help_button(
                "Regularization lambda of the selected method "
                "(SumRule, EdgeTV or combinedTVSR only).\n"
                "Larger values enforce the constraint more strongly."
            )
        )
        reg_layout.addStretch()
        processing_layout.addLayout(reg_layout)
        self._update_regularization_availability()

        components_layout = QHBoxLayout()
        components_layout.addWidget(QLabel("Number of components:"))
        self.components_spinbox = NoWheelSpinBox()
        self.components_spinbox.setRange(1, 4)
        self.components_spinbox.setValue(self.n_components)
        self.components_spinbox.valueChanged.connect(
            self._on_n_components_changed
        )
        components_layout.addWidget(self.components_spinbox)
        components_layout.addWidget(
            self._make_help_button(
                "Number of NMF components to decompose the data into."
            )
        )
        processing_layout.addLayout(components_layout)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Process folders:"))
        mode_layout.addStretch()
        mode_layout.addWidget(QLabel("individually"))
        self.process_individual_checkbox = QCheckBox()
        self.process_individual_checkbox.setChecked(self.process_individual)
        self.process_individual_checkbox.setStyleSheet(
            "margin-left: 2px; margin-right: 2px;"
        )
        mode_layout.addWidget(self.process_individual_checkbox)
        mode_layout.addWidget(QLabel("jointly"))
        self.process_joint_checkbox = QCheckBox()
        self.process_joint_checkbox.setChecked(self.process_joint)
        self.process_joint_checkbox.setStyleSheet(
            "margin-left: 2px; margin-right: 2px;"
        )
        mode_layout.addWidget(self.process_joint_checkbox)
        self.process_individual_checkbox.stateChanged.connect(
            self._on_processing_mode_changed
        )
        self.process_joint_checkbox.stateChanged.connect(
            self._on_processing_mode_changed
        )
        # Wrapped in a widget row so the whole mode selector can be hidden.
        mode_layout.setContentsMargins(0, 0, 0, 0)
        self.processing_mode_row = QWidget()
        self.processing_mode_row.setLayout(mode_layout)
        processing_layout.addWidget(self.processing_mode_row)
        if not self.joint_mode:
            self.processing_mode_row.setVisible(False)
        self._on_processing_mode_changed(None)
        self._update_processing_mode_availability()

        # Component colors
        color_layout = QVBoxLayout()
        color_header_layout = QHBoxLayout()
        color_header_layout.addWidget(QLabel("Component Colors:"))
        color_layout.addLayout(color_header_layout)

        self.color_buttons_layout = QHBoxLayout()
        color_layout.addLayout(self.color_buttons_layout)
        self._rebuild_color_buttons()

        processing_layout.addLayout(color_layout)

        # Backend selection (CUDA when available)
        cuda_layout = QHBoxLayout()
        cuda_layout.addWidget(QLabel("Backend:"))
        self.use_cupy_checkbox = QCheckBox("Use CUDA (GPU)")
        cuda_available = cupy_gpu_available()
        self.use_cupy_checkbox.setChecked(cuda_available)
        self.use_cupy_checkbox.setEnabled(cuda_available)
        if cuda_available:
            self.use_cupy_checkbox.setToolTip(
                "Run the decomposition on the GPU (faster when available)."
            )
        else:
            self.use_cupy_checkbox.setToolTip(
                "CuPy or a CUDA device is not available"
            )
        cuda_layout.addWidget(self.use_cupy_checkbox)
        cuda_layout.addStretch()
        processing_layout.addLayout(cuda_layout)

        processing_group.setLayout(processing_layout)
        middle_column_layout.addWidget(processing_group)

        # Save Options Section
        save_group = QGroupBox("Saving parameters")
        save_layout = QVBoxLayout()

        self.save_results_checkbox = QCheckBox("Save results")
        self.save_results_checkbox.setChecked(self.save_results)
        self.save_results_checkbox.stateChanged.connect(
            self._sync_save_flags
        )
        save_layout.addWidget(self.save_results_checkbox)

        self.save_deco_checkbox = QCheckBox("Deco objects (EELSNMF)")
        self.save_deco_checkbox.setToolTip(
            "Save each EELSNMF deco object (its own .pkl format,\n"
            "plus the preprocessed signal as .hspy). Default output."
        )
        self.save_deco_checkbox.setChecked(self.save_deco)
        self.save_deco_checkbox.stateChanged.connect(self._sync_save_flags)
        save_layout.addWidget(self.save_deco_checkbox)

        self.save_pickle_checkbox = QCheckBox("Full results pickle")
        self.save_pickle_checkbox.setToolTip(
            "Pickle the full processing results (images, edges,\n"
            "summaries and decos). Required to re-export every output\n"
            "exactly later, especially for joint or masked runs."
        )
        self.save_pickle_checkbox.setChecked(self.save_pickle)
        self.save_pickle_checkbox.stateChanged.connect(self._sync_save_flags)
        save_layout.addWidget(self.save_pickle_checkbox)

        self.save_colored_checkbox = QCheckBox("Colored component images")
        self.save_colored_checkbox.setChecked(self.save_colored_images)
        self.save_colored_checkbox.stateChanged.connect(self._sync_save_flags)
        save_layout.addWidget(self.save_colored_checkbox)

        self.save_masks_checkbox = QCheckBox("Mask images")
        self.save_masks_checkbox.setChecked(self.save_masks)
        self.save_masks_checkbox.stateChanged.connect(self._sync_save_flags)
        save_layout.addWidget(self.save_masks_checkbox)

        self.save_original_checkbox = QCheckBox("Original images")
        self.save_original_checkbox.setChecked(self.save_original_images)
        self.save_original_checkbox.stateChanged.connect(
            self._sync_save_flags
        )
        save_layout.addWidget(self.save_original_checkbox)

        self.save_summary_checkbox = QCheckBox("Summary plot")
        self.save_summary_checkbox.setChecked(self.save_summary_plot)
        self.save_summary_checkbox.stateChanged.connect(self._sync_save_flags)
        save_layout.addWidget(self.save_summary_checkbox)

        self.save_excel_checkbox = QCheckBox("Edges & summaries (CSV)")
        self.save_excel_checkbox.setChecked(self.save_excel)
        self.save_excel_checkbox.stateChanged.connect(self._sync_save_flags)
        save_layout.addWidget(self.save_excel_checkbox)

        self.export_loaded_button = QPushButton("Export loaded deco results")
        self.export_loaded_button.clicked.connect(self._export_loaded_deco)
        self.export_loaded_button.setEnabled(False)
        save_layout.addWidget(self.export_loaded_button)

        results_folder_layout = QHBoxLayout()
        self.results_label = QLabel("No folder selected")
        self.results_button = QPushButton("Browse...")
        self.results_button.clicked.connect(self._select_results_folder)
        results_folder_layout.addWidget(self.results_label)
        results_folder_layout.addWidget(self.results_button)
        save_layout.addLayout(results_folder_layout)
        save_group.setLayout(save_layout)
        middle_column_layout.addWidget(save_group)
        self._update_save_options_availability()

        # Run button and status
        run_layout = QHBoxLayout()
        self.run_button = QPushButton("Run Processing")
        self.run_button.clicked.connect(self._run_processing)
        run_layout.addWidget(self.run_button)

        self.status_label = QLabel("Ready")
        run_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        run_layout.addWidget(self.progress_bar)

        middle_column_layout.addLayout(run_layout)

        reset_layout = QHBoxLayout()
        reset_layout.addStretch()
        self.reset_button = QPushButton("Reset all params to Defaults")
        self.reset_button.clicked.connect(self._reset_to_defaults)
        reset_layout.addWidget(self.reset_button)
        middle_column_layout.addLayout(reset_layout)

        left_column_layout.addStretch()
        middle_column_layout.addStretch()

        # Log toggle at the bottom of the middle column: the stretch above
        # pushes it past the filler, flush with the bottom rim of the window.
        self.log_toggle_button = QPushButton("Hide log")
        self.log_toggle_button.clicked.connect(self._toggle_log)
        toggle_row = QHBoxLayout()
        toggle_row.addStretch()
        toggle_row.addWidget(self.log_toggle_button)
        middle_column_layout.addLayout(toggle_row)

        left_widget = QWidget()
        left_widget.setLayout(left_column_layout)

        middle_widget = QWidget()
        middle_widget.setLayout(middle_column_layout)

        content_layout = QHBoxLayout()
        # Columns span the full splitter height so the log toggle can reach
        # the bottom rim of the window.
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(left_widget)
        content_layout.addWidget(middle_widget)
        content_layout.setStretch(0, 1)
        content_layout.setStretch(1, 1)

        content_panel = QWidget()
        content_panel.setLayout(content_layout)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        log_layout.addWidget(self.log_view)
        log_group.setLayout(log_layout)
        log_group.setMinimumWidth(360)
        self.log_group = log_group

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(content_panel)
        splitter.addWidget(log_group)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.log_splitter = splitter

        outer_layout = QHBoxLayout()
        outer_layout.addWidget(splitter)
        self.setLayout(outer_layout)

    def _setup_logging(self):
        if not hasattr(self, "log_view"):
            return

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        for handler in list(root_logger.handlers):
            if isinstance(
                handler, logging.StreamHandler
            ) and handler.stream in (
                sys.stdout,
                sys.stderr,
            ):
                root_logger.removeHandler(handler)
            if isinstance(handler, _QtLogHandler):
                root_logger.removeHandler(handler)

        self._log_handler = _QtLogHandler(self.log_view)
        self._log_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root_logger.addHandler(self._log_handler)

        self._logger = logging.getLogger("eelsnmf_gui")
        if self._original_stdout is None:
            self._original_stdout = sys.stdout
        if self._original_stderr is None:
            self._original_stderr = sys.stderr
        sys.stdout = _StdoutStderrRedirect(self._logger, logging.INFO)
        sys.stderr = _StdoutStderrRedirect(self._logger, logging.INFO)

        self._logger.info("Logging initialized")

        # Log unhandled exceptions instead of letting PyQt abort the process.
        self._original_excepthook = sys.excepthook
        sys.excepthook = self._excepthook

    def _excepthook(self, exc_type, exc_value, exc_tb):
        self._logger.error(
            "Unhandled exception", exc_info=(exc_type, exc_value, exc_tb)
        )

    def _restore_logging(self):
        if getattr(self, "_original_excepthook", None) is not None:
            sys.excepthook = self._original_excepthook
            self._original_excepthook = None

        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
            self._original_stdout = None
        if self._original_stderr is not None:
            sys.stderr = self._original_stderr
            self._original_stderr = None

        if self._log_handler is not None:
            root_logger = logging.getLogger()
            if self._log_handler in root_logger.handlers:
                root_logger.removeHandler(self._log_handler)
            self._log_handler = None

    def _select_folder(self):
        start_dir = (
            None
            if not self.source_folders
            else (
                os.path.commonpath(self.source_folders)
                if len(self.source_folders) > 1
                else os.path.dirname(self.source_folders[0])
            )
        )
        if sys.platform.startswith("linux"):
            dlg = MultiFolderDialog(
                self,
                "Select Source Folder(s) or a deco file",
                start_dir=start_dir,
            )
            paths = dlg.selected_paths() if dlg.exec_() else []
        else:
            dlg = QFileDialog(self, "Select Source Folder(s)")
            if start_dir:
                dlg.setDirectory(start_dir)
            dlg.setFileMode(QFileDialog.Directory)
            dlg.setOption(QFileDialog.ShowDirsOnly, True)
            dlg.setOption(QFileDialog.DontUseNativeDialog, True)

            list_view = dlg.findChild(QListView)
            if list_view is not None:
                list_view.setSelectionMode(QAbstractItemView.MultiSelection)
            tree_view = dlg.findChild(QTreeView)
            if tree_view is not None:
                tree_view.setSelectionMode(QAbstractItemView.MultiSelection)

            paths = dlg.selectedFiles() if dlg.exec_() else []

        if not paths:
            return

        if len(paths) == 1 and os.path.isfile(paths[0]):
            self._load_deco(paths[0])
            return

        if not all(os.path.isdir(p) for p in paths):
            self._set_status(
                "Error: Select either one deco (.pkl) file or several folders"
            )
            return

        folders = paths
        self._loaded_deco = None
        self.source_folders = folders
        self.source_folder = folders[0] if len(folders) == 1 else None
        self._create_session()
        self.folder_label.setText(
            f"{len(folders)} folders selected"
            if len(folders) > 1
            else (
                f"...{folders[0][-40:]}"
                if len(folders[0]) > 40
                else folders[0]
            )
        )
        self._set_status("Folder selected")
        self._update_processing_mode_availability()

    def _load_deco(self, path: str):
        """Load one EELSNMF deco object and restore its parameters."""
        try:
            from EELSNMF import load as load_eelsnmf

            deco = load_eelsnmf(path)
        except Exception as exc:
            self._set_status(f"Error: {exc}", exc_info=True)
            return
        name = os.path.splitext(os.path.basename(path))[0]
        self._load_deco_into(deco, name)
        self._set_status(f"Deco loaded: {name}")

    def _load_deco_into(self, deco, name: str):
        """Register an already-loaded deco object (see ``_load_deco``)."""
        # A loaded deco is standalone: it detaches from the current folders.
        self._loaded_deco = deco
        self.source_folders = []
        self.source_folder = None
        self._session = None
        self.folder_label.setText(f"Deco: {name}")
        self._reset_model_combo()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItem(name, deco)
        self.model_combo.blockSignals(False)
        self._update_save_options_availability()

        self._update_params_from_deco(deco)

    def _update_params_from_deco(self, deco):
        """Push the parameters stored in *deco* back into the GUI.

        Preprocessing parameters (median filter, mask) and colors are not
        stored in deco objects and are left untouched.
        """
        restored, missing = [], []

        def set_value(name, getter, apply, description):
            try:
                value = getter()
            except AttributeError:
                missing.append(name)
                return
            apply(value)
            restored.append(f"{name} = {description}")

        set_value(
            "E0",
            lambda: deco.E0 / 1e3,
            lambda v: self.e0_spinbox.setValue(v),
            "keV",
        )
        set_value(
            "alpha",
            lambda: deco.alpha * 1e3,
            lambda v: self.alpha_spinbox.setValue(v),
            "mrad",
        )
        set_value(
            "beta",
            lambda: deco.beta * 1e3,
            lambda v: self.beta_spinbox.setValue(v),
            "mrad",
        )
        set_value(
            "max_iters",
            lambda: deco.max_iters,
            lambda v: self.iter_spinbox.setValue(int(v)),
            "",
        )
        set_value(
            "tol",
            lambda: deco.tol,
            lambda v: self.tol_spinbox.setValue(float(v)),
            "",
        )
        set_value(
            "error_skip_step",
            lambda: deco.error_skip_step,
            lambda v: self.error_skip_spinbox.setValue(int(v)),
            "",
        )
        set_value(
            "n_components",
            lambda: deco.n_components,
            lambda v: self.components_spinbox.setValue(int(v)),
            "",
        )

        method = None
        try:
            method = deco.analysis_description["decomposition"]["method"]
        except (AttributeError, KeyError, TypeError):
            missing.append("decomposition_method")
        if method in DECOMPOSITION_METHODS:
            self.method_combo.setCurrentText(method)
            restored.append(f"decomposition_method = {method}")
        elif method is not None:
            missing.append(f"decomposition_method ({method})")

        try:
            params = deco.analysis_description["decomposition"]["params"]
            lam = params.get("lmbda", params.get("TV_lmbda"))
            if lam is not None:
                self.regularization_spinbox.setValue(float(lam))
                restored.append(f"regularization = {lam}")
        except (AttributeError, KeyError, TypeError):
            missing.append("regularization")

        try:
            backgrounds = deco.model.backgrounds
            self.power_laws_spinbox.setValue(int(len(backgrounds)))
            restored.append("n_power_laws")
        except (AttributeError, TypeError):
            missing.append("n_power_laws")

        try:
            ranges = deco.fine_structure_ranges
            entries = [
                {
                    "name": name,
                    "min": float(lo),
                    "max": float(hi),
                    "checked": True,
                }
                for name, (lo, hi) in ranges.items()
            ]
            self._populate_ranges_table(entries)
            restored.append(f"edge ranges ({len(entries)})")
        except (AttributeError, TypeError):
            missing.append("edge ranges")

        if cupy_gpu_available():
            try:
                use_cupy = deco.analysis_description["decomposition"][
                    "use_cupy"
                ]
                self.use_cupy_checkbox.setChecked(bool(use_cupy))
                restored.append("use_cupy")
            except (AttributeError, KeyError, TypeError):
                missing.append("use_cupy")

        self._logger.info(
            "Restored from deco: %s. Not stored in deco (left unchanged): %s.",
            ", ".join(restored) if restored else "nothing",
            ", ".join(missing) if missing else "none",
        )

    def _create_session(self):
        """(Re)create the processing session for the selected folders."""
        if not self.source_folders:
            self._session = None
            return
        loader = partial(
            load_eml_folders,
            preprocess=self.pre_mode,
            despike_method=self.despike_method,
        )
        self._session = Session(
            self.source_folders,
            median_filter_size=self.median_filter_size,
            loader=loader,
        )
        self._reset_model_combo()

    def _reset_model_combo(self):
        """Clear the model selector (no fitted model available anymore)."""
        self._last_results = None
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItem("None (run processing first)", None)
        self.model_combo.blockSignals(False)

    def _set_status(self, message: str, *, exc_info=False):
        """Update the status label; error messages are mirrored to the log."""
        self.status_label.setText(message)
        if message.startswith("Error") and self._logger is not None:
            self._logger.error(message, exc_info=exc_info)

    def _toggle_log(self):
        """Collapse or expand the log panel, keeping the columns' shape.

        Collapsing shrinks the top-level window by the log pane's width;
        expanding restores it, so the left/middle columns never stretch.
        """
        show = self.log_group.isHidden()
        if not show:
            self._log_splitter_sizes = self.log_splitter.sizes()
        self.log_group.setVisible(show)
        self.log_toggle_button.setText("Hide log" if show else "Show log")
        if show:
            sizes = self._log_splitter_sizes
            if sizes is None or sum(sizes) <= 0:
                total = max(1, self.log_splitter.width())
                sizes = [total * 3 // 5, total * 2 // 5]
            self.log_splitter.setSizes(list(sizes))

        # Resize the window so the columns keep their horizontal shape. The
        # layout pins an explicit minimum size that includes the log pane, so
        # relax it before shrinking.
        if self.isWindow():
            window = self.window()
            sizes = self._log_splitter_sizes
            log_width = sizes[1] if sizes else 0
            if log_width > 0:
                if not show:
                    window.setMinimumSize(0, 0)
                    window.resize(window.width() - log_width, window.height())
                else:
                    window.resize(window.width() + log_width, window.height())
        if show:
            sizes = self._log_splitter_sizes
            if sizes is None or sum(sizes) <= 0:
                total = max(1, self.log_splitter.width())
                sizes = [total * 3 // 5, total * 2 // 5]
            self.log_splitter.setSizes(list(sizes))

    def _update_model_combo(self, results):
        """Populate the model selector from the last run's decos."""
        self._last_results = results
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        entries = []
        if (
            self.joint_mode
            and results.joint is not None
            and results.deco_joint is not None
        ):
            entries.append(("Joint", results.deco_joint))
        for label, deco in zip(results.labels, results.decos_individual):
            if deco is not None:
                entries.append((label, deco))
        if not entries:
            self.model_combo.addItem("None (run processing first)", None)
        for name, deco in entries:
            self.model_combo.addItem(name, deco)
        self.model_combo.blockSignals(False)

    def _plot_from_model(self, method_name: str):
        """Run an EELSNMF plot method on the model selected in the combo."""
        deco = self.model_combo.currentData()
        if deco is None:
            self._set_status("Error: Run processing first")
            return
        try:
            getattr(deco, method_name)()
            plt.show(block=False)
        except Exception as exc:  # pragma: no cover - UI error path
            self._set_status(f"Error: {exc}", exc_info=True)

    def _make_help_button(self, tooltip: str):
        """A small outlined '?' button showing *tooltip* on hover (no delay)."""
        button = HelpButton(self)
        button.setText("?")
        button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        button.setFixedSize(18, 18)
        button.setStyleSheet(
            "QToolButton { border: 1px solid #8a8a8a; border-radius: 4px;"
            " background: transparent; font-weight: bold; }"
            " QToolButton:hover { background: #dddddd; }"
        )
        button.setToolTip(tooltip)
        button.setFocusPolicy(Qt.NoFocus)
        return button

    def _update_median_availability(self):
        """The Hampel median filter size only matters for Hampel despiking."""
        self.median_filter_spinbox.setEnabled(
            self.despike_method == "hampel"
        )

    def _on_despike_changed(self, _index):
        """Switch despiking algorithm and rebuild the session (data reload)."""
        self.despike_method = self.despike_combo.currentData()
        self._update_median_availability()
        self._create_session()

    def _on_e0_changed(self, value):
        """Update beam energy parameter."""
        self.e0 = value

    def _on_alpha_changed(self, value):
        """Update convergence angle parameter."""
        self.alpha = float(value)

    def _on_beta_changed(self, value):
        """Update collection angle parameter."""
        self.beta = float(value)

    def _on_max_iters_changed(self, value):
        """Update maximum iterations parameter."""
        self.max_iters = value

    def _on_tol_changed(self, value):
        """Update convergence tolerance parameter."""
        self.tol = float(value)

    def _on_error_skip_step_changed(self, value):
        """Update error skip step parameter."""
        self.error_skip_step = int(value)

    def _on_decomposition_method_changed(self, method_name):
        """Update decomposition method; regularization only applies to some."""
        self.decomposition_method = method_name
        self._update_regularization_availability()

    def _on_regularization_changed(self, value):
        """Update regularization lambda parameter."""
        self.regularization = float(value)

    def _update_regularization_availability(self):
        """Grey out the lambda spinbox for methods without regularization."""
        enabled = self.decomposition_method in REGULARIZED_METHODS
        self.regularization_spinbox.setEnabled(enabled)
        if enabled:
            self.regularization_spinbox.setToolTip(
                "Regularization lambda of the selected method"
            )
        else:
            self.regularization_spinbox.setToolTip(
                "The selected decomposition method has no regularization"
            )

    def _on_n_power_laws_changed(self, value):
        """Update the number of power-law backgrounds in G."""
        self.n_power_laws = int(value)

    def _decomposition_settings(self) -> dict:
        """Current decomposition parameters, as passed to Session.run."""
        regularization = self.regularization
        if self.decomposition_method not in REGULARIZED_METHODS:
            regularization = None
        return {
            "n_power_laws": self.n_power_laws,
            "tol": self.tol,
            "error_skip_step": self.error_skip_step,
            "decomposition_method": self.decomposition_method,
            "regularization": regularization,
        }

    def _on_median_filter_size_changed(self, value):
        """Update median filter size parameter."""
        self.median_filter_size = value
        if self._session is not None:
            self._session.set_median_filter_size(value)

    def _on_mask_enabled_changed(self, value):
        self.mask_enabled = self.mask_checkbox.isChecked()
        self.mask_dilation_spinbox.setEnabled(self.mask_enabled)

    def _on_mask_dilation_iters_changed(self, value):
        self.mask_dilation_iters = int(value)

    def _preview_mask(self):
        if not self.source_folders or self._session is None:
            self._set_status("Error: No folder selected")
            return

        try:
            previews = self._session.mask_preview(
                MaskParams(
                    enabled=True,
                    dilation_iters=self.mask_dilation_iters,
                    mode="mean",
                )
            )
        except Exception as exc:  # pragma: no cover - UI error path
            self._set_status(f"Error: {exc}", exc_info=True)
            return

        if not previews:
            self._set_status("Error: No previews available")
            return

        n = len(previews)
        fig, axes = plt.subplots(n, 3, figsize=(10, max(3, 2.6 * n)))
        if n == 1:
            axes = np.array([axes])

        for idx, (label, img, mask) in enumerate(previews):
            ax_img = axes[idx, 0]
            ax_mask = axes[idx, 1]
            ax_overlay = axes[idx, 2]

            ax_img.imshow(img, cmap="gray")
            ax_img.set_title("Raw")
            ax_img.axis("off")

            ax_mask.imshow(mask, cmap="gray")
            ax_mask.set_title("Mask")
            ax_mask.axis("off")

            ax_overlay.imshow(img, cmap="gray")
            ax_overlay.imshow(mask, cmap="Reds", alpha=0.5)
            ax_overlay.set_title("Overlay")
            ax_overlay.axis("off")

            if label:
                ax_img.set_ylabel(label, rotation=90)

        fig.suptitle(
            f"Mask preview (dilation iters {self.mask_dilation_iters})",
            fontsize=12,
        )
        fig.tight_layout()
        plt.show(block=False)

    def _select_results_folder(self):
        """Open dialog to select results folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Results Folder", ""
        )
        if folder:
            self.results_folder = folder
            self.results_label.setText(
                f"...{folder[-40:]}" if len(folder) > 40 else folder
            )
            self._update_save_options_availability()

    def _update_save_options_availability(self):
        enabled = self.save_results
        self.results_label.setEnabled(enabled)
        self.results_button.setEnabled(enabled)
        for checkbox in (
            self.save_deco_checkbox,
            self.save_pickle_checkbox,
            self.save_colored_checkbox,
            self.save_masks_checkbox,
            self.save_original_checkbox,
            self.save_summary_checkbox,
            self.save_excel_checkbox,
        ):
            checkbox.setEnabled(enabled)
        self.export_loaded_button.setEnabled(
            enabled and self._loaded_deco is not None
        )

    def _sync_save_flags(self, _value=None):
        """Read all saving checkboxes into the corresponding flags."""
        self.save_results = self.save_results_checkbox.isChecked()
        self.save_deco = self.save_deco_checkbox.isChecked()
        self.save_pickle = self.save_pickle_checkbox.isChecked()
        self.save_colored_images = self.save_colored_checkbox.isChecked()
        self.save_masks = self.save_masks_checkbox.isChecked()
        self.save_original_images = self.save_original_checkbox.isChecked()
        self.save_summary_plot = self.save_summary_checkbox.isChecked()
        self.save_excel = self.save_excel_checkbox.isChecked()
        self._update_save_options_availability()

    def _on_processing_mode_changed(self, value):
        self.process_individual = self.process_individual_checkbox.isChecked()
        self.process_joint = self.process_joint_checkbox.isChecked()

    def _update_processing_mode_availability(self):
        if not self.joint_mode:
            # Individual processing is forced; the selector is hidden.
            return

        folder_count = len(self.source_folders)

        if folder_count == 0:
            self.process_individual_checkbox.setEnabled(False)
            self.process_joint_checkbox.setEnabled(False)
            return

        self.process_individual_checkbox.setEnabled(True)

        if folder_count == 1:
            self.process_joint_checkbox.setChecked(False)
            self.process_joint_checkbox.setEnabled(False)
        else:
            self.process_joint_checkbox.setEnabled(True)

    def _ensure_color_count(self):
        if len(self.colors) < self.n_components:
            default_palette = [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([1.0, 0.5, 0.0]),
            ]
            while len(self.colors) < self.n_components:
                self.colors.append(
                    default_palette[len(self.colors) % len(default_palette)]
                )
        elif len(self.colors) > self.n_components:
            self.colors = self.colors[: self.n_components]

    def _rebuild_color_buttons(self):
        while self.color_buttons_layout.count():
            item = self.color_buttons_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._ensure_color_count()
        self.color_buttons = []
        for idx in range(self.n_components):
            button = QPushButton(f"{idx + 1}")
            self._update_color_button(button, self.colors[idx])
            button.clicked.connect(lambda _, i=idx: self._pick_color(i))
            self.color_buttons_layout.addWidget(button)
            self.color_buttons.append(button)

    def _on_n_components_changed(self, value):
        self.n_components = int(value)
        self._rebuild_color_buttons()

    def _default_ranges(self):
        return [
            {
                "name": name,
                "min": float(bounds[0]),
                "max": float(bounds[1]),
                "checked": name == "Fe_L",
            }
            for name, bounds in DEFAULT_EDGE_RANGES.items()
        ]

    def _populate_ranges_table(self, ranges):
        if ranges is None:
            ranges = self._default_ranges()
        self.ranges_table.setRowCount(0)
        for entry in ranges:
            self._add_range_row(
                entry.get("name", ""),
                float(entry.get("min", 0.0)),
                float(entry.get("max", 0.0)),
                bool(entry.get("checked", False)),
            )

    def _add_range_row(self, name, min_val, max_val, checked=True):
        row = self.ranges_table.rowCount()
        self.ranges_table.insertRow(row)

        use_item = QTableWidgetItem("")
        use_item.setFlags(
            Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable
        )
        use_item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.ranges_table.setItem(row, 0, use_item)

        name_item = QTableWidgetItem(str(name))
        name_item.setFlags(
            Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable
        )
        self.ranges_table.setItem(row, 1, name_item)

        min_spin = NoWheelDoubleSpinBox()
        min_spin.setRange(0.0, 3000.0)
        min_spin.setSingleStep(1.0)
        min_spin.setValue(float(min_val))
        self.ranges_table.setCellWidget(row, 2, min_spin)

        max_spin = NoWheelDoubleSpinBox()
        max_spin.setRange(0.0, 3000.0)
        max_spin.setSingleStep(1.0)
        max_spin.setValue(float(max_val))
        self.ranges_table.setCellWidget(row, 3, max_spin)

    def _get_ranges_list(self):
        ranges = []
        for row in range(self.ranges_table.rowCount()):
            use_item = self.ranges_table.item(row, 0)
            name_item = self.ranges_table.item(row, 1)
            min_spin = self.ranges_table.cellWidget(row, 2)
            max_spin = self.ranges_table.cellWidget(row, 3)
            ranges.append(
                {
                    "name": name_item.text().strip() if name_item else "",
                    "min": (
                        float(min_spin.value())
                        if min_spin is not None
                        else 0.0
                    ),
                    "max": (
                        float(max_spin.value())
                        if max_spin is not None
                        else 0.0
                    ),
                    "checked": (
                        use_item.checkState() == Qt.Checked
                        if use_item is not None
                        else False
                    ),
                }
            )
        return ranges

    def _collect_ranges(self, *, only_enabled: bool):
        ranges = {}
        for entry in self._get_ranges_list():
            if only_enabled and not entry["checked"]:
                continue
            name = entry["name"].strip()
            if not name:
                self._set_status("Error: Range name cannot be empty")
                return None
            if name in ranges:
                self._set_status(
                    f"Error: Duplicate range name '{name}'"
                )
                return None
            if entry["min"] >= entry["max"]:
                self._set_status(
                    f"Error: Range '{name}' min must be < max"
                )
                return None
            ranges[name] = (entry["min"], entry["max"])
        return ranges

    def _add_range_dialog(self):
        name, ok = QInputDialog.getText(self, "Add range", "Range name:")
        if not ok:
            return
        name = name.strip()
        if not name:
            QMessageBox.warning(
                self, "Add range", "Range name cannot be empty."
            )
            return
        existing_names = {r["name"] for r in self._get_ranges_list()}
        if name in existing_names:
            QMessageBox.warning(
                self,
                "Add range",
                f"Range name '{name}' already exists.",
            )
            return

        min_val, ok = QInputDialog.getDouble(
            self,
            "Add range",
            "Min (eV):",
            0.0,
            0.0,
            3000.0,
            1,
        )
        if not ok:
            return
        max_val, ok = QInputDialog.getDouble(
            self,
            "Add range",
            "Max (eV):",
            min_val + 1.0,
            0.0,
            3000.0,
            1,
        )
        if not ok:
            return
        if min_val >= max_val:
            QMessageBox.warning(
                self,
                "Add range",
                "Min must be less than Max.",
            )
            return

        self._add_range_row(name, min_val, max_val, checked=True)

    def _remove_selected_range(self):
        row = self.ranges_table.currentRow()
        if row < 0:
            self._set_status("Error: No range selected")
            return
        self.ranges_table.removeRow(row)

    def _load_ranges_from_settings(self, settings):
        saved = settings.value("edges/ranges")
        if isinstance(saved, (list, tuple)) and len(saved) > 0:
            ranges = []
            for entry in saved:
                if isinstance(entry, dict):
                    ranges.append(entry)
                elif isinstance(entry, (list, tuple)) and len(entry) >= 4:
                    ranges.append(
                        {
                            "name": str(entry[0]),
                            "min": float(entry[1]),
                            "max": float(entry[2]),
                            "checked": bool(entry[3]),
                        }
                    )
            if ranges:
                return ranges
        return self._default_ranges()

    def _reset_to_defaults(self):
        self.e0 = 300.0
        self.alpha = 15.4
        self.beta = 18.46
        self.max_iters = 2000
        self.median_filter_size = 11
        self.n_components = 2
        self.tol = 1e-8
        self.error_skip_step = 100
        self.decomposition_method = "default_decomposition"
        self.regularization = 0.1
        self.n_power_laws = 50
        self.mask_enabled = False
        self.mask_dilation_iters = 1
        self.colors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.5, 0.0]),
        ]
        self.save_results = False
        self.save_deco = True
        self.save_pickle = False
        self.save_colored_images = False
        self.save_masks = False
        self.save_original_images = False
        self.save_summary_plot = False
        self.show_summary_plot = True
        self.save_excel = False
        self.process_individual = True
        self.process_joint = False
        if self._session is not None:
            self._session.set_median_filter_size(self.median_filter_size)

        self.e0_spinbox.setValue(self.e0)
        self.alpha_spinbox.setValue(self.alpha)
        self.beta_spinbox.setValue(self.beta)
        self.iter_spinbox.setValue(self.max_iters)
        self.tol_spinbox.setValue(self.tol)
        self.error_skip_spinbox.setValue(self.error_skip_step)
        self.method_combo.setCurrentText(self.decomposition_method)
        self.regularization_spinbox.setValue(self.regularization)
        self.power_laws_spinbox.setValue(self.n_power_laws)
        self.median_filter_spinbox.setValue(self.median_filter_size)
        index = self.despike_combo.findData(self.despike_method)
        self.despike_combo.setCurrentIndex(max(index, 0))
        self.save_deco_checkbox.setChecked(self.save_deco)
        self.save_pickle_checkbox.setChecked(self.save_pickle)
        self.save_colored_checkbox.setChecked(self.save_colored_images)
        self.save_masks_checkbox.setChecked(self.save_masks)
        self.save_original_checkbox.setChecked(self.save_original_images)
        self.save_summary_checkbox.setChecked(self.save_summary_plot)
        self.save_excel_checkbox.setChecked(self.save_excel)
        self.mask_checkbox.setChecked(self.mask_enabled)
        self.mask_dilation_spinbox.setValue(self.mask_dilation_iters)
        self.mask_dilation_spinbox.setEnabled(self.mask_enabled)
        self.components_spinbox.setValue(self.n_components)

        self.process_individual_checkbox.setChecked(self.process_individual)
        self.process_joint_checkbox.setChecked(self.process_joint)
        self._update_processing_mode_availability()

        self.save_results_checkbox.setChecked(self.save_results)
        self._update_save_options_availability()

        self._populate_ranges_table(self._default_ranges())

        self._rebuild_color_buttons()
        self._save_settings()
        self._set_status("Defaults restored")

    def _update_color_button(self, button, rgb_array):
        """Update button background to reflect chosen color."""
        r, g, b = (rgb_array * 255).astype(int)
        luminance = (
            0.2126 * rgb_array[0]
            + 0.7152 * rgb_array[1]
            + 0.0722 * rgb_array[2]
        )
        text_color = "white" if luminance < 0.5 else "black"
        button.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); color: {text_color};"
        )

    def _plot_energy_ranges(self):
        if not self.source_folders or self._session is None:
            self._set_status("Error: No folder selected")
            return

        fine_structure_ranges = self._collect_ranges(only_enabled=True)
        if fine_structure_ranges is None:
            return

        if not fine_structure_ranges:
            self._set_status("Error: Select at least one edge")
            return

        try:
            list_decos = self._session.decos_with_g(
                self._beam_params(),
                fine_structure_ranges,
                self._mask_params(),
                n_power_laws=self.n_power_laws,
            )
            plot_energy_ranges(list_decos, fine_structure_ranges)
        except Exception as exc:  # pragma: no cover - UI error path
            self._set_status(f"Error: {exc}", exc_info=True)

    def _plot_explained_variance_ratio(self):
        if not self.source_folders or self._session is None:
            self._set_status("Error: No folder selected")
            return

        if not (self.process_individual or self.process_joint):
            self._set_status("Error: Select a processing mode")
            return

        only_joint = self.process_joint
        try:
            items = self._session.list_of_si(
                individual=(self.process_individual and not only_joint),
                joint=only_joint,
            )

            if not items:
                self._set_status("Error: No data available")
                return

            fig, axes = plt.subplots(
                nrows=len(items),
                ncols=1,
                figsize=(7, max(3, 2.8 * len(items))),
                constrained_layout=True,
            )
            axes_list = (
                [axes] if not isinstance(axes, (list, np.ndarray)) else axes
            )

            for ax, (label, sl) in zip(axes_list, items):
                sl.unfolded_si.decomposition()
                sl.unfolded_si.plot_explained_variance_ratio(
                    fig=fig,
                    ax=ax,
                    n=10,
                )
                tick_count = 10
                tick_positions = list(range(tick_count))
                tick_labels = [
                    self._ordinal_label(i + 1) for i in tick_positions
                ]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels)
                ax.set_title(label)

            plt.show(block=False)
        except Exception as exc:  # pragma: no cover - UI error path
            self._set_status(f"Error: {exc}", exc_info=True)

    def _pick_color(self, component_idx):
        """Open color picker dialog for component."""
        current_color = self.colors[component_idx]
        r, g, b = (current_color * 255).astype(int)

        color = QColorDialog.getColor(
            QColor(r, g, b),
            self,
            f"Select Color for Component {component_idx + 1}",
        )

        if color.isValid():
            rgb_array = (
                np.array([color.red(), color.green(), color.blue()]) / 255.0
            )
            self.colors[component_idx] = rgb_array
            self._update_color_button(
                self.color_buttons[component_idx], rgb_array
            )

    @staticmethod
    def _ordinal_label(value: int) -> str:
        if 10 <= value % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
        return f"{value}{suffix}"

    def _beam_params(self) -> BeamParams:
        return BeamParams(
            E0_V=self.e0 * 1e3,
            alpha_rad=self.alpha * 1e-3,
            beta_rad=self.beta * 1e-3,
        )

    def _mask_params(self) -> MaskParams:
        return MaskParams(
            enabled=self.mask_enabled,
            dilation_iters=self.mask_dilation_iters,
            mode="mean",
        )

    def _run_processing(self):
        """Execute EELS processing with current parameters."""
        if self._worker_thread is not None and self._worker_thread.isRunning():
            self._set_status("Processing already running")
            return

        if not self.source_folders or self._session is None:
            self._set_status("Error: No folder selected")
            return

        fine_structure_ranges = self._collect_ranges(only_enabled=True)
        if fine_structure_ranges is None:
            return

        if not fine_structure_ranges:
            self._set_status("Error: Select at least one edge")
            return

        if not (self.process_individual or self.process_joint):
            self._set_status("Error: Select a processing mode")
            return

        if self.save_results and self.results_folder is None:
            self._set_status("Error: No results folder selected")
            return

        self._set_status("Processing...")

        self._save_settings()

        run_kwargs = {
            "beam": self._beam_params(),
            "fine_structure_ranges": fine_structure_ranges,
            "n_components": self.n_components,
            "max_iters": self.max_iters,
            "mask": self._mask_params(),
            "colors": self.colors,
            "process_individual": self.process_individual,
            "process_joint": self.process_joint,
            "use_cupy": self.use_cupy_checkbox.isChecked(),
            **self._decomposition_settings(),
        }

        print("Processing parameters:")
        print(run_kwargs)

        if self.debug:
            self.progress_bar.setVisible(True)
            results = self._session.run(**run_kwargs)
            self._update_model_combo(results)
            self._export_and_save_processing_results(results)
            self.progress_bar.setVisible(False)
        else:
            self._start_processing_on_thread(run_kwargs)

    def _load_settings(self):
        settings = QSettings("eelsnmf-gui", "eelsnmf-gui")

        self.e0 = settings.value("params/e0", self.e0, type=float)
        self.alpha = settings.value("params/alpha", self.alpha, type=float)
        self.beta = settings.value("params/beta", self.beta, type=float)
        self.max_iters = settings.value(
            "params/max_iters", self.max_iters, type=int
        )
        if self.pre_mode:
            # Preprocessing-only settings are not touched in non-pre mode so
            # that both modes can share the same QSettings file.
            self.median_filter_size = settings.value(
                "params/median_filter_size", self.median_filter_size, type=int
            )
            self.mask_enabled = settings.value(
                "params/mask_enabled", self.mask_enabled, type=bool
            )
            self.mask_dilation_iters = settings.value(
                "params/mask_dilation_iters",
                self.mask_dilation_iters,
                type=int,
            )
        if self.joint_mode:
            self.process_individual = settings.value(
                "params/process_individual", self.process_individual, type=bool
            )
            self.process_joint = settings.value(
                "params/process_joint", self.process_joint, type=bool
            )
        self.n_components = settings.value(
            "params/n_components", self.n_components, type=int
        )
        self.tol = settings.value("params/tol", self.tol, type=float)
        self.error_skip_step = settings.value(
            "params/error_skip_step", self.error_skip_step, type=int
        )
        saved_method = settings.value(
            "params/decomposition_method", self.decomposition_method
        )
        if saved_method in DECOMPOSITION_METHODS:
            self.decomposition_method = saved_method
        self.regularization = settings.value(
            "params/regularization", self.regularization, type=float
        )
        self.n_power_laws = settings.value(
            "params/n_power_laws", self.n_power_laws, type=int
        )
        if self.pre_mode:
            saved_despike = settings.value(
                "params/despike_method", self.despike_method
            )
            if saved_despike in ("eml", "hampel"):
                self.despike_method = saved_despike
        self.save_results = False
        self.save_deco = settings.value(
            "saving/deco_objects", self.save_deco, type=bool
        )
        self.save_pickle = settings.value(
            "saving/full_pickle", self.save_pickle, type=bool
        )
        self.save_colored_images = settings.value(
            "saving/colored_images", self.save_colored_images, type=bool
        )
        self.save_masks = settings.value(
            "saving/mask_images", self.save_masks, type=bool
        )
        self.save_original_images = settings.value(
            "saving/original_images", self.save_original_images, type=bool
        )
        self.save_summary_plot = settings.value(
            "saving/summary_plot", self.save_summary_plot, type=bool
        )
        self.save_excel = settings.value(
            "saving/edges_csv", self.save_excel, type=bool
        )
        self.show_summary_plot = True

        self.colors = self._settings_colors(
            settings.value("params/colors"),
            self.colors,
            settings.value("params/color_0"),
            settings.value("params/color_1"),
        )
        self._ensure_color_count()

        self._saved_ranges = self._load_ranges_from_settings(settings)

    def _save_settings(self):
        settings = QSettings("eelsnmf-gui", "eelsnmf-gui")
        settings.setValue("params/e0", self.e0)
        settings.setValue("params/alpha", self.alpha)
        settings.setValue("params/beta", self.beta)
        settings.setValue("params/max_iters", self.max_iters)
        if self.pre_mode:
            settings.setValue(
                "params/median_filter_size", self.median_filter_size
            )
            settings.setValue("params/mask_enabled", self.mask_enabled)
            settings.setValue(
                "params/mask_dilation_iters", self.mask_dilation_iters
            )
            settings.setValue("params/despike_method", self.despike_method)
        if self.joint_mode:
            settings.setValue(
                "params/process_individual", self.process_individual
            )
            settings.setValue("params/process_joint", self.process_joint)
        settings.setValue("params/n_components", self.n_components)
        settings.setValue("params/tol", self.tol)
        settings.setValue("params/error_skip_step", self.error_skip_step)
        settings.setValue(
            "params/decomposition_method", self.decomposition_method
        )
        settings.setValue("params/regularization", self.regularization)
        settings.setValue("params/n_power_laws", self.n_power_laws)
        settings.setValue(
            "params/colors", [color.tolist() for color in self.colors]
        )
        settings.setValue("saving/deco_objects", self.save_deco)
        settings.setValue("saving/full_pickle", self.save_pickle)
        settings.setValue("saving/colored_images", self.save_colored_images)
        settings.setValue("saving/mask_images", self.save_masks)
        settings.setValue("saving/original_images", self.save_original_images)
        settings.setValue("saving/summary_plot", self.save_summary_plot)
        settings.setValue("saving/edges_csv", self.save_excel)

        settings.setValue("edges/ranges", self._get_ranges_list())

    @staticmethod
    def _settings_color(value, fallback: np.ndarray) -> np.ndarray:
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return np.array([float(v) for v in value])
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",")]
            if len(parts) == 3:
                try:
                    return np.array([float(p) for p in parts])
                except ValueError:
                    return fallback
        return fallback

    @staticmethod
    def _settings_colors(value, fallback, legacy_color0, legacy_color1):
        if isinstance(value, (list, tuple)) and len(value) > 0:
            colors = []
            for entry in value:
                if isinstance(entry, (list, tuple)) and len(entry) == 3:
                    colors.append(np.array([float(v) for v in entry]))
                elif isinstance(entry, str):
                    parts = [p.strip() for p in entry.split(",")]
                    if len(parts) == 3:
                        try:
                            colors.append(np.array([float(p) for p in parts]))
                        except ValueError:
                            continue
            if colors:
                return colors

        legacy_colors = []
        if legacy_color0 is not None:
            legacy_colors.append(
                EELSProcessingWidget._settings_color(
                    legacy_color0, fallback[0]
                )
            )
        if legacy_color1 is not None:
            legacy_colors.append(
                EELSProcessingWidget._settings_color(
                    legacy_color1, fallback[1]
                )
            )
        return legacy_colors if legacy_colors else fallback

    def _start_processing_on_thread(self, run_kwargs: dict):
        print("[EELS] Starting processing thread")
        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)

        self._worker_thread = QThread()
        self._worker = _ProcessingWorker(self._session, run_kwargs)
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_processing_finished)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.finished.connect(self._on_thread_finished)

        self._worker_thread.start()

    def _on_processing_finished(self, results, error):
        print("[EELS] Processing thread finished")
        if error is not None:
            self._set_status(f"Error: {error}", exc_info=error)
            self._finish_processing_ui()
            return

        self._update_model_combo(results)
        self._export_and_save_processing_results(results)
        self._finish_processing_ui()

    def _finish_processing_ui(self):
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self._worker = None

    def _on_thread_finished(self):
        print("[EELS] Processing thread cleanup")
        self._worker_thread = None

    def _cleanup_thread(self, reason: str):
        if self._worker_thread is None:
            return
        if self._worker_thread.isRunning():
            print(f"[EELS] Stopping thread due to {reason}")
            self._worker_thread.quit()
            self._worker_thread.wait(2000)
        self._worker_thread = None

    def closeEvent(self, event):
        self._restore_logging()
        self._cleanup_thread("widget close")
        super().closeEvent(event)

    def _ensure_pickle_dir(self) -> "str | None":
        pickle_dir = os.path.join(self.results_folder, "pickles")
        try:
            os.makedirs(pickle_dir, exist_ok=True)
            return pickle_dir
        except Exception as exc:
            print(f"Failed to create the pickle folder: {exc}")
            return None

    def _save_deco_objects(self, results) -> None:
        """Save every deco object of *results* with EELSNMF's own format.

        Each deco is pickled without the hyperspy signals (see
        ``EELSNMF.__getstate__``) and, with ``save_hspy_objects=True``,
        stores the preprocessed signal (``cl``) alongside as a ``.hspy``
        file so that no data is lost.
        """
        pickle_dir = self._ensure_pickle_dir()
        if pickle_dir is None:
            return

        decos = []
        if results.joint is not None and results.joint.deco is not None:
            decos.append(("deco_joint", results.joint.deco))
        for label, res in zip(results.labels, results.individual):
            if res.deco is not None:
                decos.append((f"deco_{label}", res.deco))

        for fname, deco in decos:
            try:
                deco.save(
                    fname,
                    path=pickle_dir,
                    save_hspy_objects=True,
                    overwrite=True,
                )
                print(
                    f"Saved deco object to "
                    f"{os.path.join(pickle_dir, fname + '.pkl')}"
                )
            except Exception as exc:
                print(f"Failed to save the deco object '{fname}': {exc}")

    def _save_full_pickle(self, results) -> None:
        """Pickle the full ``ProcessFoldersResult`` (opt-in)."""
        pickle_dir = self._ensure_pickle_dir()
        if pickle_dir is None:
            return

        results_path = os.path.join(pickle_dir, "processing_results.pkl")
        try:
            with open(results_path, "wb") as fh:
                pickle.dump(results, fh)
            print(f"Pickled processing results to {results_path}")
        except Exception as exc:
            print(f"Failed to pickle the processing results: {exc}")

    def _save_pickled_results(self, results) -> None:
        """Backward-compatible wrapper: save decos and the full pickle."""
        self._save_deco_objects(results)
        self._save_full_pickle(results)

    def _export_loaded_deco(self):
        """Re-export the outputs of the loaded deco per the saving options."""
        deco = self._loaded_deco
        if deco is None:
            self._set_status("Error: No deco loaded")
            return
        if not self.save_results or self.results_folder is None:
            self._set_status("Error: No results folder selected")
            return

        try:
            self._export_loaded_deco_impl(deco)
        except Exception as exc:
            self._set_status(f"Error: {exc}", exc_info=True)
            return
        self._set_status("Loaded deco exported")

    def _export_loaded_deco_impl(self, deco):
        ranges = getattr(deco, "fine_structure_ranges", None)
        if not ranges:
            raise ValueError("The loaded deco has no fine structure ranges")
        n_components = int(getattr(deco, "n_components", 0) or 0)
        if n_components < 1:
            raise ValueError("The loaded deco has no fitted components")
        self._ensure_color_count()
        colors = self.colors[:n_components]
        pickle_dir = os.path.join(self.results_folder, "pickles")

        if self.save_deco:
            os.makedirs(pickle_dir, exist_ok=True)
            deco.save(
                "deco_loaded",
                path=pickle_dir,
                save_hspy_objects=True,
                overwrite=True,
            )

        edges = extract_edge_collection(deco, ranges, n_components)
        totals = edge_totals(deco, ranges)
        order = order_by_power(totals, n_components=n_components)
        rows = summary_rows(totals, order) if totals is not None else None

        if self.save_excel:
            export_edges_by_label_to_csv(
                edges.by_label, os.path.join(self.results_folder, "edges")
            )
            export_edge_contributions_summary_to_csv(
                rows or [],
                os.path.join(self.results_folder, "edge_contributions.csv"),
            )

        if self.save_colored_images:
            from ._analysis import component_maps_from_deco

            maps = component_maps_from_deco(deco, ranges)
            if maps is None:
                self._logger.info(
                    "Cannot reconstruct images from this deco "
                    "(joint or masked run); skipping image export."
                )
            else:
                mixed, colored = apply_colors(maps, colors)
                save_colored_images(
                    [[img] for img in mixed],
                    ["deco_loaded"],
                    os.path.join(self.results_folder, "colored_images"),
                )
                for comp_idx, comp_maps in enumerate(colored):
                    save_component_images(
                        comp_maps,
                        ["deco_loaded"],
                        os.path.join(self.results_folder, "colored_images"),
                        f"component_{comp_idx + 1}",
                    )
            if self.save_masks:
                self._logger.info(
                    "Mask images are not stored in deco objects; "
                    "skipping mask export."
                )
            if self.save_original_images:
                self._logger.info(
                    "Original images are not stored in deco objects; "
                    "skipping original image export."
                )

        if self.save_summary_plot and rows is not None:
            from ._plotting import plot_joint_edges

            fig, _ = plot_joint_edges(
                edges.combined.norm,
                edges.combined.x,
                title="EELS Processing (loaded deco)",
                colors=colors,
                summary_table=rows,
            )
            save_summary_plot(
                fig, os.path.join(self.results_folder, "summary_loaded.png")
            )
            if self.show_summary_plot:
                plt.show(block=False)
            else:
                plt.close(fig)

    def _export_and_save_processing_results(self, results):
        if self.save_results and self.results_folder is not None:
            if self.save_deco:
                self._save_deco_objects(results)
            if self.save_pickle:
                self._save_full_pickle(results)

        labels_individual = results.labels
        labels_joint = results.labels_joint

        list_imgs = [r.images.mixed or [] for r in results.individual]
        list_imgs_original = [
            r.images.original or [] for r in results.individual
        ]
        list_masks = [r.images.mask or [] for r in results.individual]
        if not any(bool(m) for m in list_masks):
            list_masks = None
        list_imgs_components = None
        if results.individual and results.individual[0].images.components:
            list_imgs_components = []
            for comp_idx in range(
                len(results.individual[0].images.components)
            ):
                list_imgs_components.append(
                    [
                        (
                            r.images.components[comp_idx]
                            if r.images.components is not None
                            else None
                        )
                        for r in results.individual
                    ]
                )
        list_edges = [r.edges.combined.norm for r in results.individual]
        list_x_axis = [r.edges.combined.x for r in results.individual]
        list_pixel_scales = [r.pixel_scale for r in results.individual]
        list_edge_summaries = [
            r.edge_contributions_summary or [] for r in results.individual
        ]

        edges_combined = None
        x_axis_combined = None
        list_imgs_joint = []
        list_imgs_original_joint = []
        list_imgs_components_joint = None
        list_masks_joint = None
        summary_joint = None
        if results.joint is not None:
            edges_combined = results.joint.edges.combined.norm
            x_axis_combined = results.joint.edges.combined.x
            summary_joint = results.joint.edge_contributions_summary
            list_imgs_joint = [
                [img] if img is not None else []
                for img in (results.joint.images.mixed or [])
            ]
            list_imgs_original_joint = [
                [img] if img is not None else []
                for img in (results.joint.images.original or [])
            ]
            if results.joint.images.components is not None:
                list_imgs_components_joint = []
                for comp_imgs in results.joint.images.components:
                    list_imgs_components_joint.append(
                        [[img] if img is not None else [] for img in comp_imgs]
                    )
            if results.joint.images.mask is not None:
                list_masks_joint = [
                    [img] if img is not None else []
                    for img in results.joint.images.mask
                ]
                if not any(bool(m) for m in list_masks_joint):
                    list_masks_joint = None

        labels_base = labels_individual if labels_individual else labels_joint
        display_labels = (
            labels_individual if labels_individual else labels_joint
        )
        suptitle = None
        if len(self.source_folders) == 1 and self.source_folders:
            suptitle = os.path.basename(self.source_folders[0])
            if display_labels:
                display_labels = [""] * len(display_labels)
        elif labels_base:
            common, individuals = split_common_label_parts(labels_base)
            suptitle = common if common else None
            if len(individuals) == len(labels_base):
                display_labels = individuals

        if self.save_colored_images and self.process_individual:
            save_colored_images(
                list_imgs,
                labels_individual,
                os.path.join(self.results_folder, "colored_images"),
            )
            if list_imgs_components is not None:
                for comp_idx, comp_imgs in enumerate(list_imgs_components):
                    save_component_images(
                        comp_imgs,
                        labels_individual,
                        os.path.join(self.results_folder, "colored_images"),
                        f"component_{comp_idx + 1}",
                    )

        if self.save_original_images and self.process_individual:
            save_original_images(
                list_imgs_original,
                labels_individual,
                os.path.join(self.results_folder, "original_images"),
            )

        if self.save_colored_images and self.process_joint:
            save_colored_images(
                list_imgs_joint,
                labels_joint,
                os.path.join(self.results_folder, "colored_images_joint"),
            )
            if list_imgs_components_joint is not None:
                for comp_idx, comp_imgs in enumerate(
                    list_imgs_components_joint
                ):
                    save_component_images(
                        comp_imgs,
                        labels_joint,
                        os.path.join(
                            self.results_folder, "colored_images_joint"
                        ),
                        f"component_{comp_idx + 1}",
                    )

        if self.save_masks and self.process_individual:
            if list_masks is not None:
                save_mask_images(
                    list_masks,
                    labels_individual,
                    os.path.join(self.results_folder, "mask_images"),
                )

        if self.save_masks and self.process_joint:
            if list_masks_joint is not None:
                save_mask_images(
                    list_masks_joint,
                    labels_joint,
                    os.path.join(self.results_folder, "mask_images_joint"),
                )

        if self.save_original_images and self.process_joint:
            save_original_images(
                list_imgs_original_joint,
                labels_joint,
                os.path.join(self.results_folder, "original_images_joint"),
            )

        if (
            self.save_excel
            and self.process_joint
            and results.joint is not None
        ):
            export_edges_by_label_to_csv(
                results.joint.edges.by_label,
                os.path.join(self.results_folder, "edges_joint"),
            )
            export_edge_contributions_summary_to_csv(
                summary_joint or [],
                os.path.join(
                    self.results_folder, "edge_contributions_joint.csv"
                ),
            )

        if self.save_excel and self.process_individual:
            edges_dir = os.path.join(self.results_folder, "edges")
            for label, result in zip(labels_individual, results.individual):
                export_edges_by_label_to_csv(
                    result.edges.by_label,
                    edges_dir,
                    filename_prefix=f"{label}_",
                )
            summary_dir = os.path.join(
                self.results_folder, "edge_contributions"
            )
            for label, summary_rows in zip(
                labels_individual, list_edge_summaries
            ):
                export_edge_contributions_summary_to_csv(
                    summary_rows,
                    os.path.join(summary_dir, f"{label}.csv"),
                )

        if self.save_summary_plot or self.show_summary_plot:
            use_joint_images = self.process_joint and bool(list_imgs_joint)
            plot_imgs = list_imgs_joint if use_joint_images else list_imgs
            plot_imgs_original = (
                list_imgs_original_joint
                if use_joint_images
                else list_imgs_original
            )
            plot_masks = list_masks_joint if use_joint_images else list_masks
            plot_imgs_components = (
                list_imgs_components_joint
                if use_joint_images
                else list_imgs_components
            )
            plot_pixel_scales = list_pixel_scales
            if (
                use_joint_images
                and not plot_pixel_scales
                and results.joint is not None
                and display_labels
            ):
                plot_pixel_scales = [results.joint.pixel_scale] * len(
                    display_labels
                )

            if self.process_individual and self.process_joint:
                fig, _ = self._create_summary_plot(
                    plot_imgs,
                    plot_imgs_original,
                    plot_masks,
                    plot_imgs_components,
                    list_edges,
                    display_labels,
                    list_x_axis,
                    edges_combined,
                    x_axis_combined,
                    plot_pixel_scales,
                    summary_tables=list_edge_summaries,
                    summary_joint=summary_joint,
                    suptitle=suptitle,
                )
            elif self.process_individual:
                fig, _ = self._create_individual_summary_plot(
                    plot_imgs,
                    plot_imgs_original,
                    plot_masks,
                    plot_imgs_components,
                    list_edges,
                    display_labels,
                    list_x_axis,
                    plot_pixel_scales,
                    summary_tables=list_edge_summaries,
                    suptitle=suptitle,
                )
            elif self.process_joint:
                if edges_combined is None or x_axis_combined is None:
                    self._set_status("Error: Joint edges unavailable")
                    fig = None
                elif display_labels and plot_imgs:
                    fig, _ = self._create_joint_summary_with_images(
                        plot_imgs,
                        plot_imgs_original,
                        plot_masks,
                        plot_imgs_components,
                        display_labels,
                        edges_combined,
                        x_axis_combined,
                        plot_pixel_scales,
                        suptitle=suptitle,
                    )
                else:
                    fig, _ = self._create_joint_summary_plot(
                        edges_combined,
                        x_axis_combined,
                        summary_table=summary_joint,
                        suptitle=suptitle,
                    )
            else:
                fig = None

            if fig is not None:
                if self.save_summary_plot:
                    save_summary_plot(
                        fig,
                        os.path.join(self.results_folder, "summary.png"),
                        dpi=300,
                    )

                import matplotlib.pyplot as plt

                if self.show_summary_plot:
                    plt.show(block=False)
                else:
                    plt.close(fig)

        self._set_status("Ready")

    def _create_summary_plot(
        self,
        list_imgs,
        list_imgs_original,
        list_masks,
        list_imgs_components,
        list_edges,
        labels,
        list_x_axis,
        edges_combined,
        x_axis_combined,
        list_pixel_scales,
        *,
        suptitle: str | None = None,
        summary_tables=None,
        summary_joint=None,
    ):
        from ._plotting import plot_image_grid_and_shifted_pairs

        return plot_image_grid_and_shifted_pairs(
            L=list_imgs,
            L_original=list_imgs_original,
            L_mask=list_masks,
            L_components=list_imgs_components,
            E=list_edges,
            S=labels,
            list_x_axis=list_x_axis,
            title="EELS Processing",
            E_combined=edges_combined,
            x_axis_combined=x_axis_combined,
            list_pixel_scales=list_pixel_scales,
            colors=self.colors,
            suptitle=suptitle,
            summary_tables=summary_tables,
            summary_joint=summary_joint,
        )

    def _create_individual_summary_plot(
        self,
        list_imgs,
        list_imgs_original,
        list_masks,
        list_imgs_components,
        list_edges,
        labels,
        list_x_axis,
        list_pixel_scales,
        *,
        suptitle: str | None = None,
        summary_tables=None,
    ):
        from ._plotting import plot_image_grid_with_edges

        return plot_image_grid_with_edges(
            L=list_imgs,
            L_original=list_imgs_original,
            L_mask=list_masks,
            L_components=list_imgs_components,
            E=list_edges,
            S=labels,
            list_x_axis=list_x_axis,
            title="EELS Processing",
            list_pixel_scales=list_pixel_scales,
            colors=self.colors,
            suptitle=suptitle,
            summary_tables=summary_tables,
        )

    def _create_joint_summary_plot(
        self,
        edges_combined,
        x_axis_combined,
        *,
        suptitle: str | None = None,
        summary_table=None,
    ):
        from ._plotting import plot_joint_edges

        return plot_joint_edges(
            edges_combined=edges_combined,
            x_axis_combined=x_axis_combined,
            title="EELS Processing",
            colors=self.colors,
            suptitle=suptitle,
            summary_table=summary_table,
        )

    def _create_joint_summary_with_images(
        self,
        list_imgs,
        list_imgs_original,
        list_masks,
        list_imgs_components,
        labels,
        edges_combined,
        x_axis_combined,
        list_pixel_scales,
        *,
        suptitle: str | None = None,
    ):
        from ._plotting import plot_joint_summary_with_images

        return plot_joint_summary_with_images(
            L=list_imgs,
            L_original=list_imgs_original,
            L_mask=list_masks,
            L_components=list_imgs_components,
            S=labels,
            edges_combined=edges_combined,
            x_axis_combined=x_axis_combined,
            list_pixel_scales=list_pixel_scales,
            colors=self.colors,
            suptitle=suptitle,
            summary_table=None,
        )
