import os

from qtpy.QtCore import QDir, QEvent, QSortFilterProxyModel, Qt
from qtpy.QtGui import QFileSystemModel
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTreeView,
    QVBoxLayout,
)


def drop_ancestor_folders(paths):
    """Drop paths that contain another selected path.

    Selecting both a folder and its parent would load the same data
    twice (the loader scans folders recursively), so only the most
    specific selections are kept, in their original order.
    """
    kept = []
    for path in paths:
        if any(
            path != other
            and other.startswith(path + os.sep)
            for other in paths
        ):
            continue
        kept.append(path)
    return kept


class FolderOrDecoProxyModel(QSortFilterProxyModel):
    """Show directories always; hide files except deco ``*.pkl`` pickles.

    A plain ``QFileSystemModel`` name filter cannot do this: with
    ``setNameFilterDisables(False)`` the ``*.pkl`` filter hides the
    directories themselves.
    """

    def filterAcceptsRow(self, source_row, source_parent):
        model = self.sourceModel()
        index = model.index(source_row, 0, source_parent)
        if model.isDir(index):
            return True
        return model.fileName(index).lower().endswith(".pkl")


class MultiFolderDialog(QDialog):
    """Dialog for selecting multiple folders or one deco .pkl file.

    Folders and ``*.pkl`` files are listed; other files are hidden.
    """

    def __init__(
        self,
        parent=None,
        title="Select Source Folder(s) or a deco file",
        start_dir=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)

        layout = QVBoxLayout(self)

        self.model = QFileSystemModel(self)
        self.model.setFilter(QDir.Dirs | QDir.Files | QDir.NoDotAndDotDot)
        self.model.setRootPath("")  # required for model to populate
        self.proxy = FolderOrDecoProxyModel(self)
        self.proxy.setSourceModel(self.model)

        self.path_edit = QLineEdit(self)
        self.path_edit.setPlaceholderText("Paste path to jump...")
        self.path_edit.textChanged.connect(self._on_path_changed)
        self.path_edit.returnPressed.connect(self._accept_on_enter)
        layout.addWidget(self.path_edit)

        self.view = QTreeView(self)
        self.view.setModel(self.proxy)
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
                self.view.setRootIndex(self.proxy.mapFromSource(root_index))
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
        self.view.setRootIndex(self.proxy.mapFromSource(index))
        self.view.setCurrentIndex(self.proxy.mapFromSource(index))
        self.view.scrollTo(self.proxy.mapFromSource(index))

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

    def selected_paths(self):
        """Return selected paths (folders or one deco file), in order.

        Paths that contain another selected path are dropped: a dialog
        quirk (observed on macOS with the stock QFileDialog) can report
        both the clicked folder and its parent, which would load the
        same data twice.
        """
        root_source = self.proxy.mapToSource(self.view.rootIndex())
        root_path = (
            self.model.filePath(root_source) if root_source.isValid() else None
        )
        paths = []
        seen = set()
        for ix in self.view.selectionModel().selectedIndexes():
            if ix.column() != 0:
                continue
            source_ix = self.proxy.mapToSource(ix)
            path = self.model.filePath(source_ix)
            # Skip the view's root directory itself (it can appear in the
            # selection as a side effect of programmatic row selection).
            if not path or path == root_path:
                continue
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)
        return drop_ancestor_folders(paths)

    def selected_folders(self):
        """Return selected folder paths, deduplicated in selection order."""
        return [
            p
            for p in self.selected_paths()
            if os.path.isdir(p)
        ]
