import pytest

pytest.importorskip("qtpy")

from qtpy.QtCore import QItemSelection, QItemSelectionModel
from qtpy.QtWidgets import QApplication


def test_multi_folder_dialog_shows_folders_and_deco_files(qtbot, tmp_path):
    """The browse dialog lists folders and .pkl files, hiding other files."""
    from eelsnmf_gui._file_dialog import MultiFolderDialog

    (tmp_path / "runA").mkdir()
    (tmp_path / "runB").mkdir()
    (tmp_path / "deco_x.pkl").touch()
    (tmp_path / "STEM SI.dm4").touch()
    dialog = MultiFolderDialog(title="test", start_dir=str(tmp_path))
    for _ in range(20):  # let the filesystem model populate
        QApplication.processEvents()

    root_index = dialog.proxy.mapFromSource(dialog.model.index(str(tmp_path)))
    names = {
        dialog.model.fileName(
            dialog.proxy.mapToSource(dialog.proxy.index(row, 0, root_index))
        )
        for row in range(dialog.proxy.rowCount(root_index))
    }
    assert names == {"runA", "runB", "deco_x.pkl"}


def test_multi_folder_dialog_selection(qtbot, tmp_path):
    """Multi-folder and single-deco selections are returned as paths."""
    from eelsnmf_gui._file_dialog import MultiFolderDialog

    for name in ("runA", "runB"):
        (tmp_path / name).mkdir()
    (tmp_path / "deco_x.pkl").touch()

    dialog = MultiFolderDialog(title="test", start_dir=str(tmp_path))
    for _ in range(20):
        QApplication.processEvents()

    root_index = dialog.proxy.mapFromSource(dialog.model.index(str(tmp_path)))

    def child_index(entry_name):
        for row in range(dialog.proxy.rowCount(root_index)):
            child = dialog.proxy.index(row, 0, root_index)
            source = dialog.proxy.mapToSource(child)
            if dialog.model.fileName(source) == entry_name:
                return child
        raise AssertionError(f"entry not found: {entry_name}")

    def select(*entry_names):
        selection = QItemSelection()
        for name in entry_names:
            child = child_index(name)
            selection.select(child, child)
        dialog.view.selectionModel().select(
            selection, QItemSelectionModel.Select
        )

    select("runA", "runB")
    assert sorted(dialog.selected_paths()) == [
        str(tmp_path / "runA"),
        str(tmp_path / "runB"),
    ]
    assert sorted(dialog.selected_folders()) == [
        str(tmp_path / "runA"),
        str(tmp_path / "runB"),
    ]

    dialog.view.selectionModel().clearSelection()
    select("deco_x.pkl")
    assert dialog.selected_paths() == [str(tmp_path / "deco_x.pkl")]
    assert dialog.selected_folders() == []


def test_multi_folder_dialog_drops_ancestor_folders(qtbot, tmp_path):
    """A parent folder selected alongside its child is dropped.

    Some dialogs report both the clicked folder and the surrounding
    directory (observed on macOS); loading both would process the same
    data twice.
    """
    from eelsnmf_gui._file_dialog import (
        MultiFolderDialog,
        drop_ancestor_folders,
    )

    parent = str(tmp_path)
    child = str(tmp_path / "runA")
    (tmp_path / "runA").mkdir()

    assert drop_ancestor_folders([parent, child]) == [child]
    assert drop_ancestor_folders([child, parent]) == [child]
    # Unrelated folders are all kept, in order.
    other = str(tmp_path / "runB")
    (tmp_path / "runB").mkdir()
    assert drop_ancestor_folders([child, other]) == [child, other]

    # And through the dialog itself: selecting tmp_path (the view root is
    # already tmp_path, so the extra parent selection is dropped).
    dialog = MultiFolderDialog(title="test", start_dir=str(tmp_path))
    for _ in range(20):
        QApplication.processEvents()
    root_index = dialog.proxy.mapFromSource(dialog.model.index(str(tmp_path)))
    for row in range(dialog.proxy.rowCount(root_index)):
        source = dialog.proxy.mapToSource(dialog.proxy.index(row, 0, root_index))
        if dialog.model.fileName(source) == "runA":
            dialog.view.selectionModel().select(
                dialog.proxy.index(row, 0, root_index),
                QItemSelectionModel.Select,
            )
    assert dialog.selected_paths() == [str(tmp_path / "runA")]
