import numpy as np
import pandas as pd

from eelsnmf_gui._export import (
    export_edge_contributions_summary_to_csv,
    export_edges_by_label_to_csv,
    export_edges_to_csv,
    save_colored_images,
    save_component_images,
    save_mask_images,
    save_original_images,
    save_summary_plot,
)
from eelsnmf_gui._results import EdgeSeries


def test_save_colored_images_rgb(tmp_path):
    """A single RGB image per label is saved as <label>.png."""
    img = np.zeros((4, 5, 3))
    img[..., 0] = 1.0
    save_colored_images([[img]], ["runA"], str(tmp_path))
    assert (tmp_path / "runA.png").exists()


def test_save_original_images_gray_and_indices(tmp_path):
    """Multiple images per label get zero-padded indexed filenames."""
    imgs = [np.ones((4, 5)), np.zeros((4, 5))]
    save_original_images([imgs], ["runA"], str(tmp_path))
    assert (tmp_path / "runA_image00.png").exists()
    assert (tmp_path / "runA_image01.png").exists()


def test_save_mask_images(tmp_path):
    """Mask images are saved through the same image-saving path."""
    mask = np.ones((4, 5))
    save_mask_images([[mask]], ["runA"], str(tmp_path))
    assert (tmp_path / "runA.png").exists()


def test_save_component_images_subdirectory(tmp_path):
    """Component images go into a per-component subdirectory."""
    img = np.ones((4, 5, 3))
    save_component_images([[img]], ["runA"], str(tmp_path), "component_1")
    assert (tmp_path / "component_1" / "runA.png").exists()


def test_export_edges_to_csv(tmp_path):
    """Edge CSVs carry the energy axis and per-component columns."""
    x = np.linspace(0, 1, 5)
    export_edges_to_csv(
        x,
        [x, 2 * x],
        [x, 2 * x],
        [x, x],
        str(tmp_path / "edges.csv"),
    )
    df = pd.read_csv(tmp_path / "edges.csv")
    assert list(df.columns) == [
        "Energy (eV)",
        "Component 1",
        "Component 2",
        "Component 1 Smoothed",
        "Component 2 Smoothed",
        "Component 1 Smoothed (Norm)",
        "Component 2 Smoothed (Norm)",
    ]
    assert len(df) == 5


def test_export_edges_by_label_to_csv(tmp_path):
    """Each edge label is exported to its own CSV file."""
    x = np.linspace(0, 1, 5)
    edges_by_label = {
        "O_K": EdgeSeries(x=x, raw=[x], smoothed=[x], norm=[x]),
        "Fe_L": EdgeSeries(x=x, raw=[2 * x], smoothed=[2 * x], norm=[x]),
    }
    export_edges_by_label_to_csv(edges_by_label, str(tmp_path))
    assert (tmp_path / "O_K.csv").exists()
    assert (tmp_path / "Fe_L.csv").exists()


def test_export_edges_by_label_prefix(tmp_path):
    """The filename prefix is prepended to per-label CSV names."""
    x = np.linspace(0, 1, 5)
    edges_by_label = {
        "O_K": EdgeSeries(x=x, raw=[x], smoothed=[x], norm=[x]),
    }
    export_edges_by_label_to_csv(
        edges_by_label, str(tmp_path), filename_prefix="runA_"
    )
    assert (tmp_path / "runA_O_K.csv").exists()


def test_export_summary_components_are_one_based(tmp_path):
    """Summary component indices are shifted to one-based numbering."""
    rows = [
        {"Component": 0, "Power": 0.75, "Relative Power (%)": 75.0},
        {"Component": 1, "Power": 0.25, "Relative Power (%)": 25.0},
    ]
    export_edge_contributions_summary_to_csv(
        rows, str(tmp_path / "summary.csv")
    )
    df = pd.read_csv(tmp_path / "summary.csv")
    assert df["Component"].tolist() == [1, 2]


def test_export_summary_empty_is_noop(tmp_path):
    """An empty summary list writes no file."""
    export_edge_contributions_summary_to_csv(
        [], str(tmp_path / "summary.csv")
    )
    assert not (tmp_path / "summary.csv").exists()


def test_save_summary_plot(tmp_path):
    """A figure is saved as PNG, creating missing parent directories."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    save_summary_plot(fig, str(tmp_path / "sub" / "summary.png"))
    assert (tmp_path / "sub" / "summary.png").exists()
    plt.close(fig)
