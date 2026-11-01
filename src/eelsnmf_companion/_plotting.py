from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import EELSNMF.utils as enmfu  # type: ignore[import-not-found]


def add_scale_bar(
    ax,
    pixel_scale,
    scale_bar_length_um=10,
    margin=0.05,
    bar_height_fraction=0.015,
    font_size=8,
):
    from matplotlib.patches import Rectangle

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    img_width = abs(xlim[1] - xlim[0])
    img_height = abs(ylim[1] - ylim[0])

    scale_bar_length_pixels = scale_bar_length_um / pixel_scale

    x_start = min(xlim) + margin * img_width
    y_start = max(ylim) - margin * img_height

    bar_height = bar_height_fraction * img_height

    rect = Rectangle(
        (x_start, y_start - bar_height),
        scale_bar_length_pixels,
        bar_height,
        linewidth=0,
        edgecolor="none",
        facecolor="white",
        transform=ax.transData,
        clip_on=False,
    )
    ax.add_patch(rect)

    text_y = y_start - bar_height - 0.02 * img_height
    ax.text(
        x_start + scale_bar_length_pixels / 2,
        text_y,
        f"{int(scale_bar_length_um)} nm",
        color="white",
        fontsize=font_size,
        ha="center",
        va="bottom",
        weight="bold",
        transform=ax.transData,
        clip_on=False,
    )


def _validate_plot_inputs(
    L: Sequence[Sequence[np.ndarray]],
    E: Sequence[Sequence[np.ndarray]],
    S: Sequence[str],
):
    if not (len(L) == len(E) == len(S)):
        raise ValueError(
            f"len(L), len(E), len(S) must match; got {len(L)}, {len(E)}, {len(S)}"
        )

    n = len(L)
    if n == 0:
        raise ValueError("Inputs are empty (len(L)==0).")

    n_components = None
    for i, comps in enumerate(E):
        if len(comps) == 0:
            raise ValueError(f"E[{i}] has no components")
        shapes = [np.asarray(c).ravel().shape for c in comps]
        if any(shape != shapes[0] for shape in shapes[1:]):
            raise ValueError(
                f"E[{i}] component arrays must have same shape; got {shapes}"
            )
        if n_components is None:
            n_components = len(comps)
        elif len(comps) != n_components:
            raise ValueError(
                f"All E entries must have same component count; got {n_components} and {len(comps)}"
            )

    return n, n_components or 0


def _validate_image_inputs(
    L: Sequence[Sequence[np.ndarray]],
    S: Sequence[str],
):
    if len(L) != len(S):
        raise ValueError(f"len(L), len(S) must match; got {len(L)}, {len(S)}")

    n = len(L)
    if n == 0:
        raise ValueError("Inputs are empty (len(L)==0).")

    return n


def _plot_image_row(
    fig,
    gs,
    row_index: int,
    label: str,
    image_columns,
    pixel_scale: Optional[float],
    *,
    image_interpolation: str,
    image_aspect: str,
    label_pad: float,
):
    for col_index, col in enumerate(image_columns):
        ax = fig.add_subplot(gs[row_index, col_index])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if col_index == 0 and label:
            ax.set_ylabel(
                label,
                rotation=90,
                ha="center",
                va="center",
                labelpad=label_pad,
            )

        imgs = col.get("imgs") or []
        if len(imgs) > 0:
            img = imgs[0]
            cmap = col.get("cmap")
            vmin = col.get("vmin")
            vmax = col.get("vmax")
            if np.asarray(img).ndim == 2:
                ax.imshow(
                    img,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation=image_interpolation,
                    aspect=image_aspect,
                )
            else:
                ax.imshow(
                    img,
                    interpolation=image_interpolation,
                    aspect=image_aspect,
                )

        if col_index == 0 and pixel_scale is not None:
            add_scale_bar(ax, pixel_scale, scale_bar_length_um=10)


def _summary_table_columns() -> List[str]:
    return [
        "Component",
        "Relative Leave-Out Loss (%)",
        "Relative Power (%)",
        "Relative Explained Variance (%)",
    ]


def _summary_table_labels() -> List[str]:
    return [
        "Comp",
        "Rel. Leave-Out\nLoss (%)",
        "Rel.\nPower (%)",
        "Rel. Explained\nVariance (%)",
    ]


def _format_summary_value(column: str, value) -> str:
    if column == "Component":
        try:
            return str(int(value) + 1)
        except (TypeError, ValueError):
            return str(value)
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def _has_summary_rows(
    summary_rows: Optional[Sequence[Dict[str, float]]],
) -> bool:
    return bool(summary_rows)


def _has_summary_tables(
    summary_tables: Optional[Sequence[Sequence[Dict[str, float]]]],
) -> bool:
    if not summary_tables:
        return False
    return any(bool(rows) for rows in summary_tables)


def _add_summary_table(
    ax,
    summary_rows: Sequence[Dict[str, float]],
    *,
    show_header: bool,
):
    ax.set_axis_off()
    if not summary_rows:
        return
    columns = _summary_table_columns()
    labels = _summary_table_labels()
    cell_text = [
        [_format_summary_value(col, row.get(col, "")) for col in columns]
        for row in summary_rows
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=labels if show_header else None,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1.0, 1.45)
    table.auto_set_column_width(list(range(len(columns))))
    for (row, col), cell in table.get_celld().items():
        if show_header and row == 0:
            cell.get_text().set_wrap(True)
            cell.set_height(cell.get_height() * 1.4)


def _plot_shifted_edges(
    axes,
    E: Sequence[Sequence[np.ndarray]],
    list_x_axis: Sequence[np.ndarray],
    labels: Sequence[str],
    colors: Sequence[np.ndarray],
    *,
    text_pad_points: float,
    split_indices: Optional[np.ndarray] = None,
):
    axes_list = list(axes) if isinstance(axes, (list, tuple)) else [axes]
    reference_chunks = None
    if split_indices is not None and len(list_x_axis) > 0:
        ref_x = np.asarray(list_x_axis[0]).ravel()
        reference_chunks = np.split(ref_x, split_indices)

    for i, comps in enumerate(E):
        shift = i + 0.1

        x = np.asarray(list_x_axis[i]).ravel()
        curve_split_indices = _find_gap_splits(x)
        x_chunks = (
            np.split(x, curve_split_indices)
            if curve_split_indices is not None
            else [x]
        )
        comp_chunks = [
            (
                np.split(np.asarray(comp).ravel(), curve_split_indices)
                if curve_split_indices is not None
                else [np.asarray(comp).ravel()]
            )
            for comp in comps
        ]

        for chunk_index, x_chunk in enumerate(x_chunks):
            if len(x_chunk) == 0:
                continue
            if reference_chunks is not None:
                chunk_mean = float(np.mean(x_chunk))
                axis_index = None
                for ref_index, ref_chunk in enumerate(reference_chunks):
                    if len(ref_chunk) == 0:
                        continue
                    if ref_chunk[0] <= chunk_mean <= ref_chunk[-1]:
                        axis_index = ref_index
                        break
                if axis_index is None:
                    ref_means = [
                        float(np.mean(rc)) if len(rc) > 0 else np.inf
                        for rc in reference_chunks
                    ]
                    axis_index = int(
                        np.argmin(np.abs(np.array(ref_means) - chunk_mean))
                    )
            else:
                axis_index = min(chunk_index, len(axes_list) - 1)

            ax = axes_list[axis_index]
            for comp_idx, comp_chunk_list in enumerate(comp_chunks):
                comp_chunk = comp_chunk_list[chunk_index]
                ax.plot(
                    x_chunk,
                    comp_chunk + shift,
                    color=colors[comp_idx % len(colors)],
                )

            first_chunk = comp_chunks[0][chunk_index]
            last_x = float(x_chunk[-1])
            last_y = float(first_chunk[-1]) + shift + 0.35

            if labels[i]:
                ax.annotate(
                    labels[i],
                    xy=(last_x, last_y),
                    xytext=(0, text_pad_points),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                )


def _find_gap_splits(x_axis: np.ndarray, *, gap_factor: float = 5.0):
    x_axis = np.asarray(x_axis).ravel()
    if x_axis.size < 2:
        return None

    diffs = np.diff(x_axis)
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return None

    median_step = float(np.median(positive_diffs))
    if median_step <= 0:
        return None

    gap_threshold = gap_factor * median_step
    split_indices = np.where(diffs > gap_threshold)[0] + 1
    return split_indices if split_indices.size else None


def plot_image_grid_and_shifted_pairs(
    L: Sequence[Sequence[np.ndarray]],
    L_original: Sequence[Sequence[np.ndarray]],
    L_mask: Optional[Sequence[Sequence[np.ndarray]]],
    E: Sequence[Sequence[np.ndarray]],
    S: Sequence[str],
    list_x_axis: Sequence[np.ndarray],
    title: str,
    E_combined: Sequence[np.ndarray],
    x_axis_combined: np.ndarray,
    list_pixel_scales: Sequence[float] = None,
    colors: Sequence[np.ndarray] = (
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
    ),
    *,
    L_components: Optional[Sequence[Sequence[Sequence[np.ndarray]]]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    left_width_ratios: Tuple[float, float] = (0.7, 0.7),
    image_cmap: Optional[str] = None,
    image_vmin: Optional[float] = None,
    image_vmax: Optional[float] = None,
    image_interpolation: str = "nearest",
    image_aspect: str = "equal",
    label_pad: float = 8.0,
    text_pad_points: float = 2.0,
    suptitle: Optional[str] = None,
    summary_tables: Optional[Sequence[Sequence[Dict[str, float]]]] = None,
    summary_joint: Optional[Sequence[Dict[str, float]]] = None,
):
    n, n_components = _validate_plot_inputs(L, E, S)
    if L_mask is not None and len(L_mask) != n:
        raise ValueError("len(L_mask) must match image count")
    if L_components is not None and len(L_components) != n_components:
        raise ValueError("len(L_components) must match component count")

    n_image_cols = 2 + (1 if L_mask is not None else 0)
    if L_components is not None:
        n_image_cols += len(L_components)

    left_widths = tuple(left_width_ratios)
    if len(left_widths) < n_image_cols:
        left_widths = left_widths + (left_widths[-1],) * (
            n_image_cols - len(left_widths)
        )
    elif len(left_widths) > n_image_cols:
        left_widths = left_widths[:n_image_cols]

    if figsize is None:
        figsize = (14, max(3, 2.2 + 2 * n))

    fig = plt.figure(figsize=figsize, constrained_layout=True)

    split_indices = _find_gap_splits(
        x_axis_combined if x_axis_combined is not None else list_x_axis[0]
    )
    x_split_reference = (
        x_axis_combined if x_axis_combined is not None else list_x_axis[0]
    )
    n_edge_axes = len(
        np.split(np.asarray(x_split_reference), split_indices)
        if split_indices is not None
        else [x_split_reference]
    )

    has_table = _has_summary_tables(summary_tables) or _has_summary_rows(
        summary_joint
    )
    table_width = (1.8,) if has_table else ()
    width_ratios = left_widths + (2.0,) * n_edge_axes + table_width
    gs_outer = fig.add_gridspec(
        nrows=2,
        ncols=n_image_cols + n_edge_axes + (1 if has_table else 0),
        height_ratios=(1, 1.8),
        width_ratios=width_ratios,
    )

    ax_combined_list = [
        fig.add_subplot(gs_outer[0, n_image_cols + i])
        for i in range(n_edge_axes)
    ]
    ax_combined_table = (
        fig.add_subplot(gs_outer[0, n_image_cols + n_edge_axes])
        if has_table
        else None
    )

    gs_left = gs_outer[1, 0:n_image_cols].subgridspec(
        nrows=n, ncols=n_image_cols, hspace=0.05, wspace=0.02
    )

    ax_r_list = [
        fig.add_subplot(gs_outer[1, n_image_cols + i])
        for i in range(n_edge_axes)
    ]

    gs_tables = (
        gs_outer[1, n_image_cols + n_edge_axes].subgridspec(
            nrows=n, ncols=1, hspace=0.05
        )
        if has_table
        else None
    )

    combined_components = [np.asarray(c).ravel() for c in E_combined]

    x_combined_chunks = (
        np.split(x_axis_combined, split_indices)
        if split_indices is not None
        else [x_axis_combined]
    )
    combined_chunks = [
        np.split(comp, split_indices) if split_indices is not None else [comp]
        for comp in combined_components
    ]

    for idx, (ax, x_chunk) in enumerate(
        zip(ax_combined_list, x_combined_chunks)
    ):
        for comp_idx, comp_chunk_list in enumerate(combined_chunks):
            ax.plot(
                x_chunk,
                comp_chunk_list[idx],
                color=colors[comp_idx % len(colors)],
                label=(f"Component {comp_idx + 1}" if idx == 0 else None),
            )
        ax.set_xlabel("energy loss (eV)")
        if idx == 0:
            ax.set_ylabel("intensities (a.u.)")
            ax.legend()
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        ax.margins(x=0.02, y=0.05)

    if ax_combined_table is not None and _has_summary_rows(summary_joint):
        _add_summary_table(
            ax_combined_table,
            summary_joint,
            show_header=True,
        )

    for i in range(n):
        row_imgs_original = list(L_original[i])
        row_imgs_processed = list(L[i])
        row_imgs_mask = list(L_mask[i]) if L_mask is not None else []
        component_rows = []
        if L_components is not None:
            for comp in L_components:
                component_rows.append(
                    list(comp[i]) if comp[i] is not None else []
                )
        pixel_scale = (
            list_pixel_scales[i]
            if list_pixel_scales is not None and i < len(list_pixel_scales)
            else None
        )
        image_columns = [
            {"imgs": row_imgs_original, "cmap": "gray"},
        ]
        if L_mask is not None:
            image_columns.append({"imgs": row_imgs_mask, "cmap": "gray"})
        image_columns.append(
            {
                "imgs": row_imgs_processed,
                "cmap": image_cmap,
                "vmin": image_vmin,
                "vmax": image_vmax,
            }
        )
        for row_imgs in component_rows:
            image_columns.append({"imgs": row_imgs, "cmap": None})
        _plot_image_row(
            fig,
            gs_left,
            n - 1 - i,
            S[i],
            image_columns,
            pixel_scale,
            image_interpolation=image_interpolation,
            image_aspect=image_aspect,
            label_pad=label_pad,
        )
        if gs_tables is not None and summary_tables is not None:
            row_index = n - 1 - i
            table_ax = fig.add_subplot(gs_tables[row_index, 0])
            show_header = True
            _add_summary_table(
                table_ax,
                summary_tables[i] if i < len(summary_tables) else [],
                show_header=show_header,
            )

    _plot_shifted_edges(
        ax_r_list,
        E,
        list_x_axis,
        S,
        colors,
        text_pad_points=text_pad_points,
        split_indices=split_indices,
    )

    for idx, ax in enumerate(ax_r_list):
        ax.set_xlabel("energy loss (eV)")
        if idx == 0:
            ax.set_ylabel("intensities (a.u.)")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        ax.margins(x=0.02, y=0.05)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16)
    fig.tight_layout()

    return fig, ax_r_list[0] if ax_r_list else None


def plot_image_grid_with_edges(
    L: Sequence[Sequence[np.ndarray]],
    L_original: Sequence[Sequence[np.ndarray]],
    L_mask: Optional[Sequence[Sequence[np.ndarray]]],
    E: Sequence[Sequence[np.ndarray]],
    S: Sequence[str],
    list_x_axis: Sequence[np.ndarray],
    title: str,
    list_pixel_scales: Sequence[float] = None,
    colors: Sequence[np.ndarray] = (
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
    ),
    *,
    L_components: Optional[Sequence[Sequence[Sequence[np.ndarray]]]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    image_cmap: Optional[str] = None,
    image_vmin: Optional[float] = None,
    image_vmax: Optional[float] = None,
    image_interpolation: str = "nearest",
    image_aspect: str = "equal",
    label_pad: float = 8.0,
    text_pad_points: float = 2.0,
    suptitle: Optional[str] = None,
    summary_tables: Optional[Sequence[Sequence[Dict[str, float]]]] = None,
):
    n, n_components = _validate_plot_inputs(L, E, S)
    if L_mask is not None and len(L_mask) != n:
        raise ValueError("len(L_mask) must match image count")
    if L_components is not None and len(L_components) != n_components:
        raise ValueError("len(L_components) must match component count")

    n_image_cols = 2 + (1 if L_mask is not None else 0)
    if L_components is not None:
        n_image_cols += len(L_components)

    if figsize is None:
        figsize = (12, max(3, 2.2 + 2 * n))

    fig = plt.figure(figsize=figsize, constrained_layout=True)

    split_indices = _find_gap_splits(list_x_axis[0])
    n_edge_axes = len(
        np.split(np.asarray(list_x_axis[0]), split_indices)
        if split_indices is not None
        else [list_x_axis[0]]
    )

    has_table = _has_summary_tables(summary_tables)
    spacer_width = 0.8
    width_ratios = (
        (0.7,) * n_image_cols
        + (spacer_width,)
        + (2.2,) * n_edge_axes
        + ((1.8,) if has_table else ())
    )
    gs = fig.add_gridspec(
        nrows=n,
        ncols=n_image_cols + 1 + n_edge_axes + (1 if has_table else 0),
        width_ratios=width_ratios,
        hspace=0.05,
        wspace=0.02,
    )
    ax_edges_list = [
        fig.add_subplot(gs[:, n_image_cols + 1 + i])
        for i in range(n_edge_axes)
    ]

    for i in range(n):
        row_imgs_original = list(L_original[i])
        row_imgs_processed = list(L[i])
        row_imgs_mask = list(L_mask[i]) if L_mask is not None else []
        component_rows = []
        if L_components is not None:
            for comp in L_components:
                component_rows.append(
                    list(comp[i]) if comp[i] is not None else []
                )
        pixel_scale = (
            list_pixel_scales[i]
            if list_pixel_scales is not None and i < len(list_pixel_scales)
            else None
        )
        image_columns = [
            {"imgs": row_imgs_original, "cmap": "gray"},
        ]
        if L_mask is not None:
            image_columns.append({"imgs": row_imgs_mask, "cmap": "gray"})
        image_columns.append(
            {
                "imgs": row_imgs_processed,
                "cmap": image_cmap,
                "vmin": image_vmin,
                "vmax": image_vmax,
            }
        )
        for row_imgs in component_rows:
            image_columns.append({"imgs": row_imgs, "cmap": None})
        _plot_image_row(
            fig,
            gs,
            n - 1 - i,
            S[i],
            image_columns,
            pixel_scale,
            image_interpolation=image_interpolation,
            image_aspect=image_aspect,
            label_pad=label_pad,
        )
        if has_table:
            row_index = n - 1 - i
            table_ax = fig.add_subplot(
                gs[row_index, n_image_cols + 1 + n_edge_axes]
            )
            show_header = True
            _add_summary_table(
                table_ax,
                summary_tables[i] if i < len(summary_tables) else [],
                show_header=show_header,
            )

    _plot_shifted_edges(
        ax_edges_list,
        E,
        list_x_axis,
        S,
        colors,
        text_pad_points=text_pad_points,
        split_indices=split_indices,
    )

    for idx, ax in enumerate(ax_edges_list):
        ax.set_xlabel("energy loss (eV)")
        if idx == 0:
            ax.set_ylabel("intensities (a.u.)")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        ax.margins(x=0.02, y=0.05)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16)
    fig.tight_layout()

    return fig, ax_edges_list[0] if ax_edges_list else None


def plot_joint_summary_with_images(
    L: Sequence[Sequence[np.ndarray]],
    L_original: Sequence[Sequence[np.ndarray]],
    L_mask: Optional[Sequence[Sequence[np.ndarray]]],
    S: Sequence[str],
    edges_combined: Sequence[np.ndarray],
    x_axis_combined: np.ndarray,
    list_pixel_scales: Sequence[float] = None,
    colors: Sequence[np.ndarray] = (
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
    ),
    *,
    L_components: Optional[Sequence[Sequence[Sequence[np.ndarray]]]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    image_cmap: Optional[str] = None,
    image_vmin: Optional[float] = None,
    image_vmax: Optional[float] = None,
    image_interpolation: str = "nearest",
    image_aspect: str = "equal",
    label_pad: float = 8.0,
    suptitle: Optional[str] = None,
    summary_table: Optional[Sequence[Dict[str, float]]] = None,
):
    n = _validate_image_inputs(L, S)
    if L_mask is not None and len(L_mask) != n:
        raise ValueError("len(L_mask) must match image count")

    n_image_cols = 2 + (1 if L_mask is not None else 0)
    if L_components is not None:
        n_image_cols += len(L_components)

    if figsize is None:
        figsize = (12, max(3, 2.2 + 2 * n))

    fig = plt.figure(figsize=figsize, constrained_layout=True)

    split_indices = _find_gap_splits(x_axis_combined)
    n_edge_axes = len(
        np.split(np.asarray(x_axis_combined), split_indices)
        if split_indices is not None
        else [x_axis_combined]
    )

    has_table = _has_summary_rows(summary_table)
    spacer_width = 0.8
    width_ratios = (
        (0.7,) * n_image_cols
        + (spacer_width,)
        + (2.2,) * n_edge_axes
        + ((1.8,) if has_table else ())
    )
    gs = fig.add_gridspec(
        nrows=n,
        ncols=n_image_cols + 1 + n_edge_axes + (1 if has_table else 0),
        width_ratios=width_ratios,
        hspace=0.05,
        wspace=0.02,
    )

    ax_edges_list = [
        fig.add_subplot(gs[:, n_image_cols + 1 + i])
        for i in range(n_edge_axes)
    ]
    for ax in ax_edges_list:
        # Keep the edge axes square so they do not stretch vertically when
        # many image rows are displayed.
        ax.set_box_aspect(1)
    ax_table = (
        fig.add_subplot(gs[:, n_image_cols + 1 + n_edge_axes])
        if has_table
        else None
    )

    for i in range(n):
        row_imgs_original = list(L_original[i])
        row_imgs_processed = list(L[i])
        row_imgs_mask = list(L_mask[i]) if L_mask is not None else []
        component_rows = []
        if L_components is not None:
            for comp in L_components:
                component_rows.append(
                    list(comp[i]) if comp[i] is not None else []
                )
        pixel_scale = (
            list_pixel_scales[i]
            if list_pixel_scales is not None and i < len(list_pixel_scales)
            else None
        )
        image_columns = [
            {"imgs": row_imgs_original, "cmap": "gray"},
        ]
        if L_mask is not None:
            image_columns.append({"imgs": row_imgs_mask, "cmap": "gray"})
        image_columns.append(
            {
                "imgs": row_imgs_processed,
                "cmap": image_cmap,
                "vmin": image_vmin,
                "vmax": image_vmax,
            }
        )
        for row_imgs in component_rows:
            image_columns.append({"imgs": row_imgs, "cmap": None})
        _plot_image_row(
            fig,
            gs,
            n - 1 - i,
            S[i],
            image_columns,
            pixel_scale,
            image_interpolation=image_interpolation,
            image_aspect=image_aspect,
            label_pad=label_pad,
        )

    combined_components = [np.asarray(c).ravel() for c in edges_combined]

    x_chunks = (
        np.split(x_axis_combined, split_indices)
        if split_indices is not None
        else [x_axis_combined]
    )
    combined_chunks = [
        np.split(comp, split_indices) if split_indices is not None else [comp]
        for comp in combined_components
    ]

    for idx, (ax, x_chunk) in enumerate(zip(ax_edges_list, x_chunks)):
        for comp_idx, comp_chunk_list in enumerate(combined_chunks):
            ax.plot(
                x_chunk,
                comp_chunk_list[idx],
                color=colors[comp_idx % len(colors)],
                label=(f"Component {comp_idx + 1}" if idx == 0 else None),
            )
        ax.set_xlabel("energy loss (eV)")
        if idx == 0:
            ax.set_ylabel("intensities (a.u.)")
            ax.legend()
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        ax.margins(x=0.02, y=0.05)

    if ax_table is not None and summary_table is not None:
        _add_summary_table(ax_table, summary_table, show_header=True)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16)
    fig.tight_layout()

    return fig, ax_edges_list[0] if ax_edges_list else None


def plot_joint_edges(
    edges_combined: Sequence[np.ndarray],
    x_axis_combined: np.ndarray,
    title: str,
    colors: Sequence[np.ndarray] = (
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
    ),
    *,
    figsize: Optional[Tuple[float, float]] = None,
    suptitle: Optional[str] = None,
    summary_table: Optional[Sequence[Dict[str, float]]] = None,
):
    if figsize is None:
        figsize = (8, 4)

    split_indices = _find_gap_splits(x_axis_combined)
    n_edge_axes = len(
        np.split(np.asarray(x_axis_combined), split_indices)
        if split_indices is not None
        else [x_axis_combined]
    )

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    has_table = summary_table is not None
    width_ratios = (1.0,) * n_edge_axes + ((1.5,) if has_table else ())
    gs = fig.add_gridspec(
        1,
        n_edge_axes + (1 if has_table else 0),
        width_ratios=width_ratios,
    )
    ax_list = [fig.add_subplot(gs[0, i]) for i in range(n_edge_axes)]
    ax_table = fig.add_subplot(gs[0, n_edge_axes]) if has_table else None

    combined_components = [np.asarray(c).ravel() for c in edges_combined]

    x_chunks = (
        np.split(x_axis_combined, split_indices)
        if split_indices is not None
        else [x_axis_combined]
    )
    combined_chunks = [
        np.split(comp, split_indices) if split_indices is not None else [comp]
        for comp in combined_components
    ]

    for idx, (ax, x_chunk) in enumerate(zip(ax_list, x_chunks)):
        for comp_idx, comp_chunk_list in enumerate(combined_chunks):
            ax.plot(
                x_chunk,
                comp_chunk_list[idx],
                color=colors[comp_idx % len(colors)],
                label=(f"Component {comp_idx + 1}" if idx == 0 else None),
            )
        ax.set_xlabel("energy loss (eV)")
        if idx == 0:
            ax.set_ylabel("intensities (a.u.)")
            ax.legend()
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        ax.margins(x=0.02, y=0.05)

    if ax_table is not None and summary_table is not None:
        _add_summary_table(ax_table, summary_table, show_header=True)

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()

    return fig, ax_list[0] if ax_list else None


def plot_energy_ranges(list_decos, fine_structure_ranges):
    plt.figure("Edges energy ranges")
    plt.clf()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for deco in list_decos:
        plt.plot(
            deco.energy_axis,
            enmfu.norm(deco.X.mean(1)),
            alpha=0.6,
            color=colors[0],
        )
    ax = plt.gca()
    for i, (k, v) in enumerate(fine_structure_ranges.items()):
        ii, ff = v
        rect = mpl.patches.Rectangle(
            (ii, 0.0),
            ff - ii,
            1,
            alpha=0.5,
            label=k,
            color=colors[i % len(colors)],
        )
        ax.add_patch(rect)

    x_lim_min = min(
        [
            min([d.energy_axis.min() for d in list_decos]),
            min([np.min(ii) for ii, ff in fine_structure_ranges.values()]),
        ]
    )
    x_lim_max = max(
        [
            max([d.energy_axis.max() for d in list_decos]),
            max([np.max(ff) for ii, ff in fine_structure_ranges.values()]),
        ]
    )
    ax.set_xlim(x_lim_min, x_lim_max)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)
