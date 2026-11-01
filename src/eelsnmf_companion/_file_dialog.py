import os

from qtpy.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTreeView,
    QAbstractItemView,
    QLineEdit,
)
from qtpy.QtCore import QDir, QEvent, QItemSelectionModel, Qt
from qtpy.QtGui import QFileSystemModel


class MultiFolderDialog(QDialog):
    """Dialog for selecting multiple folders with a path jump box."""

    def __init__(
        self, parent=None, title="Select Source Folder(s)", start_dir=None
    ):
        super().__init__(parent)
        self.setWindowTitle(title)

        layout = QVBoxLayout(self)

        self.model = QFileSystemModel(self)
        self.model.setFilter(QDir.Dirs | QDir.NoDotAndDotDot)
        self.model.setRootPath("")  # required for model to populate

        self.path_edit = QLineEdit(self)
        self.path_edit.setPlaceholderText("Paste path to jump...")
        self.path_edit.textChanged.connect(self._on_path_changed)
        self.path_edit.returnPressed.connect(self._accept_on_enter)
        layout.addWidget(self.path_edit)

        self.view = QTreeView(self)
        self.view.setModel(self.model)
        self.view.setHeaderHidden(True)
        self.view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.installEventFilter(self)

        # Single click selects; do NOT treat it as "activate"
        self.view.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # Double click should expand/collapse, not “enter directory”
        self.view.setExpandsOnDoubleClick(True)

        # Hide non-name columns (Size/Type/Date) if they appear
        for col in range(1, self.model.columnCount()):
            self.view.hideColumn(col)

        if start_dir:
            root_index = self.model.index(start_dir)
            if root_index.isValid():
                self.view.setRootIndex(root_index)
                self.path_edit.setText(start_dir)

        self.view.selectionModel().selectionChanged.connect(
            self._on_selection_changed
        )

        layout.addWidget(self.view)

        btns = QHBoxLayout()
        btns.addStretch(1)
        ok = QPushButton("OK", self)
        cancel = QPushButton("Cancel", self)
        ok.setDefault(False)
        ok.setAutoDefault(False)
        cancel.setDefault(False)
        cancel.setAutoDefault(False)
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        layout.addLayout(btns)

    def _on_selection_changed(self, *_):
        pass
        # indexes = self.view.selectionModel().selectedRows(0)
        # if not indexes:
        #     return
        # path = self.model.filePath(indexes[0])
        # if path:
        #     self.path_edit.setText(path)

    def _set_root_from_path(self, path: str) -> None:
        index = self.model.index(path)
        if not index.isValid():
            return
        self.view.setRootIndex(index)
        self.view.setCurrentIndex(index)
        self.view.scrollTo(index)

    def _on_path_changed(self, text: str) -> None:
        path = text.strip()
        if not path or not os.path.isdir(path):
            return
        self._set_root_from_path(path)

    def _accept_on_enter(self) -> None:
        selection = self.view.selectionModel()
        if not selection.selectedRows(0):
            return
        self.accept()

    def eventFilter(self, source, event):
        if source is self.view and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._accept_on_enter()
                return True
        return super().eventFilter(source, event)

    def selected_folders(self):
        """Return selected folder paths, deduplicated in selection order."""
        # Use column 0 so we get one index per row
        indexes = self.view.selectionModel().selectedRows(0)
        paths = []
        for ix in indexes:
            path = self.model.filePath(ix)
            if path:
                paths.append(path)
        # De-duplicate while preserving order
        seen = set()
        out = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out
