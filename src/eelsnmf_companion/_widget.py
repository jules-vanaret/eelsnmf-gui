"""
This module contains Qt widgets for EELS spectroscopy processing and analysis.
"""

from typing import Dict, List, Optional
import logging
import os
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt

from qtpy.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QLabel,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QFileDialog,
    QAbstractItemView,
    QListView,
    QTreeView,
    QColorDialog,
    QProgressBar,
    QApplication,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QInputDialog,
)
from qtpy.QtCore import Qt, QThread, Signal, QObject, QSettings
from qtpy.QtGui import QColor, QTextCursor

from ._processing import (
    process_folders,
    build_deco_for_folders_to_plot_er,
    build_scree_plot_inputs,
    build_mask_preview_for_folders,
)
from ._export import (
    export_edges_by_label_to_csv,
    export_edge_contributions_summary_to_csv,
    save_component_images,
    save_colored_images,
    save_mask_images,
    save_original_images,
    save_summary_plot,
)
from ._labels import (
    split_common_label_parts,
)
from ._file_dialog import (
    MultiFolderDialog,
)
from ._plotting import plot_energy_ranges

# Default edge energy ranges (in eV)
DEFAULT_EDGE_RANGES = {
    "Fe_L": (700.0, 740.0),
    "O_K": (522.0, 590.0),
}


class _ProcessingWorker(QObject):
    finished = Signal(object, object)

    def __init__(self, params: dict):
        super().__init__()
        self._params = params

    def run(self):
        try:
            results = process_folders(**self._params)
            self.finished.emit(results, None)
        except Exception as exc:  # pragma: no cover - UI error path
            self.finished.emit(None, exc)


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


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

    def __init__(self, debug: bool = False):
        super().__init__()
        self.debug = debug

        # Initialize parameter values
        self.e0 = 300.0  # keV
        self.alpha = 15.4  # mrad
        self.beta = 18.46  # mrad
        self.max_iters = 2000
        self.median_filter_size = 11
        self._saved_ranges = None
        self.n_components = 2
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
        self.save_colored_images = False
        self.save_original_images = False
        self.save_summary_plot = False
        self.show_summary_plot = True
        self.save_excel = False
        self.process_individual = True
        self.process_joint = False
        self._worker_thread = None
        self._worker = None
        self._logger = None
        self._log_handler = None
        self._original_stdout = None
        self._original_stderr = None
        self._preprocess_cache: Dict[tuple, object] = {}
        self._app = QApplication.instance()
        if self._app is not None:
            self._app.aboutToQuit.connect(
                lambda: self._cleanup_thread("app quit")
            )

        self._load_settings()

        # Build UI
        self._build_ui()
        self._setup_logging()

    def _build_ui(self):
        """Construct the widget layout."""
        left_column_layout = QVBoxLayout()
        middle_column_layout = QVBoxLayout()

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
        median_filter_layout.addWidget(QLabel("Median filter size:"))
        self.median_filter_spinbox = NoWheelSpinBox()
        self.median_filter_spinbox.setRange(1, 101)
        self.median_filter_spinbox.setValue(self.median_filter_size)
        self.median_filter_spinbox.setSingleStep(1)
        self.median_filter_spinbox.valueChanged.connect(
            self._on_median_filter_size_changed
        )
        median_filter_layout.addWidget(self.median_filter_spinbox)
        preprocessing_layout.addLayout(median_filter_layout)

        mask_layout = QHBoxLayout()
        self.mask_checkbox = QCheckBox("Mask dark pixels")
        self.mask_checkbox.setChecked(self.mask_enabled)
        self.mask_checkbox.stateChanged.connect(self._on_mask_enabled_changed)
        mask_layout.addWidget(self.mask_checkbox)

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

        angles_layout.addWidget(QLabel("β"))
        self.beta_spinbox = NoWheelDoubleSpinBox()
        self.beta_spinbox.setRange(0.0, 1000.0)
        self.beta_spinbox.setValue(self.beta)
        self.beta_spinbox.setSingleStep(0.1)
        self.beta_spinbox.valueChanged.connect(self._on_beta_changed)
        angles_layout.addWidget(self.beta_spinbox)
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
        ranges_buttons_layout.addStretch()
        structure_layout.addLayout(ranges_buttons_layout)

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
        processing_layout.addLayout(iter_layout)

        components_layout = QHBoxLayout()
        components_layout.addWidget(QLabel("Number of components:"))
        self.components_spinbox = NoWheelSpinBox()
        self.components_spinbox.setRange(1, 4)
        self.components_spinbox.setValue(self.n_components)
        self.components_spinbox.valueChanged.connect(
            self._on_n_components_changed
        )
        components_layout.addWidget(self.components_spinbox)
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
        processing_layout.addLayout(mode_layout)
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

        processing_group.setLayout(processing_layout)
        middle_column_layout.addWidget(processing_group)

        # Save Options Section
        save_group = QGroupBox("Save Results")
        save_layout = QVBoxLayout()

        self.save_results_checkbox = QCheckBox("Save results")
        self.save_results_checkbox.setChecked(self.save_results)
        self.save_results_checkbox.stateChanged.connect(
            self._on_save_results_changed
        )
        save_layout.addWidget(self.save_results_checkbox)

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

        left_widget = QWidget()
        left_widget.setLayout(left_column_layout)

        middle_widget = QWidget()
        middle_widget.setLayout(middle_column_layout)

        content_layout = QHBoxLayout()
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

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(content_panel)
        splitter.addWidget(log_group)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

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

    def _restore_logging(self):
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
                self, "Select Source Folder(s)", start_dir=start_dir
            )
            folders = dlg.selected_folders() if dlg.exec_() else []
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

            folders = dlg.selectedFiles() if dlg.exec_() else []

        if folders:
            self.source_folders = folders
            self.source_folder = folders[0] if len(folders) == 1 else None
            self._preprocess_cache.clear()
            self.folder_label.setText(
                f"{len(folders)} folders selected"
                if len(folders) > 1
                else (
                    f"...{folders[0][-40:]}"
                    if len(folders[0]) > 40
                    else folders[0]
                )
            )
            self.status_label.setText("Folder selected")
            self._update_processing_mode_availability()

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

    def _on_median_filter_size_changed(self, value):
        """Update median filter size parameter."""
        self.median_filter_size = value

    def _on_mask_enabled_changed(self, value):
        self.mask_enabled = self.mask_checkbox.isChecked()
        self.mask_dilation_spinbox.setEnabled(self.mask_enabled)

    def _on_mask_dilation_iters_changed(self, value):
        self.mask_dilation_iters = int(value)

    def _preview_mask(self):
        if not self.source_folders:
            self.status_label.setText("Error: No folder selected")
            return

        try:
            previews = build_mask_preview_for_folders(
                self.source_folders,
                median_filter_size=self.median_filter_size,
                dilation_iters=self.mask_dilation_iters,
                mode="mean",
                preprocess_cache=self._preprocess_cache,
            )
        except Exception as exc:  # pragma: no cover - UI error path
            self.status_label.setText(f"Error: {exc}")
            return

        if not previews:
            self.status_label.setText("Error: No previews available")
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

    def _on_save_results_changed(self, value):
        self.save_results = self.save_results_checkbox.isChecked()
        self.save_colored_images = self.save_results
        self.save_original_images = self.save_results
        self.save_summary_plot = self.save_results
        self.save_excel = self.save_results
        self.show_summary_plot = True
        self._update_save_options_availability()

    def _on_processing_mode_changed(self, value):
        self.process_individual = self.process_individual_checkbox.isChecked()
        self.process_joint = self.process_joint_checkbox.isChecked()

    def _update_processing_mode_availability(self):
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
                self.status_label.setText("Error: Range name cannot be empty")
                return None
            if name in ranges:
                self.status_label.setText(
                    f"Error: Duplicate range name '{name}'"
                )
                return None
            if entry["min"] >= entry["max"]:
                self.status_label.setText(
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
            self.status_label.setText("Error: No range selected")
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
        self.mask_enabled = False
        self.mask_dilation_iters = 1
        self.colors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.5, 0.0]),
        ]
        self.save_results = False
        self.save_colored_images = False
        self.save_original_images = False
        self.save_summary_plot = False
        self.show_summary_plot = True
        self.save_excel = False
        self.process_individual = True
        self.process_joint = False
        self._preprocess_cache.clear()

        self.e0_spinbox.setValue(self.e0)
        self.alpha_spinbox.setValue(self.alpha)
        self.beta_spinbox.setValue(self.beta)
        self.iter_spinbox.setValue(self.max_iters)
        self.median_filter_spinbox.setValue(self.median_filter_size)
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
        self.status_label.setText("Defaults restored")

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
            f"background-color: rgb({r}, {g}, {b}); " f"color: {text_color};"
        )

    def _plot_energy_ranges(self):
        if not self.source_folders:
            self.status_label.setText("Error: No folder selected")
            return

        fine_structure_ranges = self._collect_ranges(only_enabled=True)
        if fine_structure_ranges is None:
            return

        if not fine_structure_ranges:
            self.status_label.setText("Error: Select at least one edge")
            return

        list_decos = build_deco_for_folders_to_plot_er(
            self.source_folders,
            E0_V=self.e0 * 1e3,
            alpha_rad=self.alpha * 1e-3,
            beta_rad=self.beta * 1e-3,
            median_filter_size=self.median_filter_size,
            fine_structure_ranges=fine_structure_ranges,
            mask_enabled=self.mask_enabled,
            mask_dilation_iters=self.mask_dilation_iters,
            mask_mode="mean",
            preprocess_cache=self._preprocess_cache,
        )

        plot_energy_ranges(list_decos, fine_structure_ranges)

    def _plot_explained_variance_ratio(self):
        if not self.source_folders:
            self.status_label.setText("Error: No folder selected")
            return

        if not (self.process_individual or self.process_joint):
            self.status_label.setText("Error: Select a processing mode")
            return

        only_joint = self.process_joint
        items = build_scree_plot_inputs(
            self.source_folders,
            E0_V=self.e0 * 1e3,
            alpha_rad=self.alpha * 1e-3,
            beta_rad=self.beta * 1e-3,
            median_filter_size=self.median_filter_size,
            process_individual=(self.process_individual and not only_joint),
            process_joint=only_joint,
            mask_enabled=self.mask_enabled,
            mask_dilation_iters=self.mask_dilation_iters,
            mask_mode="mean",
            preprocess_cache=self._preprocess_cache,
        )

        if not items:
            self.status_label.setText("Error: No data available")
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
            tick_labels = [self._ordinal_label(i + 1) for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.set_title(label)

        plt.show(block=False)

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

    def _run_processing(self):
        """Execute EELS processing with current parameters."""
        if self._worker_thread is not None and self._worker_thread.isRunning():
            self.status_label.setText("Processing already running")
            return

        if not self.source_folders:
            self.status_label.setText("Error: No folder selected")
            return

        fine_structure_ranges = self._collect_ranges(only_enabled=True)
        if fine_structure_ranges is None:
            return

        if not fine_structure_ranges:
            self.status_label.setText("Error: Select at least one edge")
            return

        if not (self.process_individual or self.process_joint):
            self.status_label.setText("Error: Select a processing mode")
            return

        if self.save_results and self.results_folder is None:
            self.status_label.setText("Error: No results folder selected")
            return

        self.status_label.setText("Processing...")

        self._save_settings()

        params = {
            "source_folders": self.source_folders,
            "E0_V": self.e0 * 1e3,
            "alpha_rad": self.alpha * 1e-3,
            "beta_rad": self.beta * 1e-3,
            "max_iters": self.max_iters,
            "median_filter_size": self.median_filter_size,
            "fine_structure_ranges": fine_structure_ranges,
            "colors": self.colors,
            "n_components": self.n_components,
            "process_individual": self.process_individual,
            "process_joint": self.process_joint,
            "mask_enabled": self.mask_enabled,
            "mask_dilation_iters": self.mask_dilation_iters,
            "mask_mode": "mean",
            # "preprocess_cache": self._preprocess_cache,
        }

        print(f"Processing parameters:")
        # for key, value in params.items():
        #     if key in ["source_folders"]:
        #         print(f"  {key}: {[os.path.basename(f) for f in value]}")
        #     else:
        #         print(f"  {key}: {value}")
        print(params)

        if self.debug:
            self.progress_bar.setVisible(True)
            results = process_folders(**params)
            self._export_and_save_processing_results(results)
            self.progress_bar.setVisible(False)
        else:
            self._start_processing_on_thread(params)

    def _load_settings(self):
        settings = QSettings("eelsnmf-gui", "eelsnmf-gui")

        self.e0 = settings.value("params/e0", self.e0, type=float)
        self.alpha = settings.value("params/alpha", self.alpha, type=float)
        self.beta = settings.value("params/beta", self.beta, type=float)
        self.max_iters = settings.value(
            "params/max_iters", self.max_iters, type=int
        )
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
        self.n_components = settings.value(
            "params/n_components", self.n_components, type=int
        )
        self.process_individual = settings.value(
            "params/process_individual", self.process_individual, type=bool
        )
        self.process_joint = settings.value(
            "params/process_joint", self.process_joint, type=bool
        )
        self.save_results = False
        self.save_colored_images = False
        self.save_original_images = False
        self.save_summary_plot = False
        self.save_excel = False
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
        settings.setValue("params/median_filter_size", self.median_filter_size)
        settings.setValue("params/mask_enabled", self.mask_enabled)
        settings.setValue(
            "params/mask_dilation_iters", self.mask_dilation_iters
        )
        settings.setValue("params/n_components", self.n_components)
        settings.setValue("params/process_individual", self.process_individual)
        settings.setValue("params/process_joint", self.process_joint)
        settings.setValue(
            "params/colors", [color.tolist() for color in self.colors]
        )

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

    def _start_processing_on_thread(self, params: dict):
        print("[EELS] Starting processing thread")
        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)

        self._worker_thread = QThread()
        self._worker = _ProcessingWorker(params)
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
            self.status_label.setText(f"Error: {error}")
            self._finish_processing_ui()
            return

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

    def _save_pickled_results(self, results) -> None:
        """Pickle the processing results and the EELSNMF deco objects.

        Everything is written into a dedicated ``pickles`` subfolder of the
        results folder. The full ``ProcessFoldersResult`` (images, edges, W,
        decos, ...) is written to ``processing_results.pkl``. In addition,
        every deco object is saved with its own ``save`` method, which
        pickles it without the hyperspy signals (see
        ``EELSNMF.__getstate__``) and, with ``save_hspy_objects=True``,
        stores the preprocessed signal (``cl``) alongside as a ``.hspy``
        file so that no data is lost.
        """
        pickle_dir = os.path.join(self.results_folder, "pickles")
        try:
            os.makedirs(pickle_dir, exist_ok=True)
        except Exception as exc:
            print(f"Failed to create the pickle folder: {exc}")
            return

        results_path = os.path.join(pickle_dir, "processing_results.pkl")
        try:
            with open(results_path, "wb") as fh:
                pickle.dump(results, fh)
            print(f"Pickled processing results to {results_path}")
        except Exception as exc:
            print(f"Failed to pickle the processing results: {exc}")

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

    def _export_and_save_processing_results(self, results):
        if self.save_results and self.results_folder is not None:
            self._save_pickled_results(results)

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
            if list_masks is not None:
                save_mask_images(
                    list_masks,
                    labels_individual,
                    os.path.join(self.results_folder, "mask_images"),
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
            if list_masks_joint is not None:
                save_mask_images(
                    list_masks_joint,
                    labels_joint,
                    os.path.join(self.results_folder, "mask_images_joint"),
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
                    self.status_label.setText("Error: Joint edges unavailable")
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

        self.status_label.setText("Ready")

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
        suptitle: Optional[str] = None,
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
        suptitle: Optional[str] = None,
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
        suptitle: Optional[str] = None,
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
        suptitle: Optional[str] = None,
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
