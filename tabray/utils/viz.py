"""Visualization, formatting, and filesystem helpers.

These helpers support the tutorial notebooks and other lightweight analysis
code that needs compact formatting, parquet sizing, or grid comparison plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def format_bytes(num_bytes: int | float) -> str:
    """Format a byte count using binary units."""
    units = ["B", "KiB", "MiB", "GiB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} GiB"


def format_table_value(value: object) -> str:
    """Format table cell values for compact display."""
    if isinstance(value, float):
        return "NaN" if np.isnan(value) else f"{value:.3f}"
    return str(value)


def parquet_disk_size(parquet_path: str) -> int:
    """Return the total on-disk size for a parquet file or directory."""
    path = Path(parquet_path)
    if not path.exists():
        path = path.parent
    if path.is_file():
        return path.stat().st_size
    return sum(
        file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file()
    )


# Tables in the storage schema show at most this many rows and columns; each
# row gets 1/(MAX_TABLE_ROWS + 1) of its area, so every table uses the same
# row height and stays inside its area.
MAX_TABLE_ROWS = 10
MAX_TABLE_COLS = 8


def draw_table(ax, cell_text, col_labels, row_labels=None, title=None):
    """Draw a styled Matplotlib table at the top of the provided axes."""
    ax.set_axis_off()
    height = min(1.0, (len(cell_text) + 1) / (MAX_TABLE_ROWS + 1))
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        rowLabels=row_labels,
        cellLoc="center",
        bbox=[0.0, 1.0 - height, 1.0, height],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("dimgray")
        if row == 0:
            cell.set_facecolor("#e6e6e6")
            cell.set_text_props(color="dimgray", weight="bold")
        elif col == -1:
            cell.set_facecolor("#f5f5f5")
            cell.set_text_props(color="dimgray")
    if title is not None:
        ax.set_title(title, color="dimgray", pad=4, fontsize=9)
    return table


def _truncate(cells, row_labels, col_labels):
    """Cut a table to MAX_TABLE_ROWS x MAX_TABLE_COLS, marking cuts with '...'.

    Returns:
        (cells, row_labels, col_labels, note): note says what was left out
    """
    n_rows, n_cols = len(cells), len(col_labels)
    note = []
    if n_cols > MAX_TABLE_COLS:
        keep = MAX_TABLE_COLS - 1
        cells = [row[:keep] + ["..."] for row in cells]
        col_labels = col_labels[:keep] + ["..."]
        note.append(f"first {keep} of {n_cols} columns")
    if n_rows > MAX_TABLE_ROWS:
        keep = MAX_TABLE_ROWS - 1
        cells = cells[:keep] + [["..."] * len(col_labels)]
        if row_labels is not None:
            row_labels = row_labels[:keep] + ["..."]
        note.append(f"first {keep} of {n_rows} rows")
    return cells, row_labels, col_labels, (" (" + ", ".join(note) + ")") if note else ""


def draw_storage_schema(ax, ds, df, var):
    """Draw the storage comparison schema for the tutorial figures.

    - tabular: every row of df; coordinate columns repeat per row, and a
      variable absent at a row's site is a NaN cell
    - array: every coordinate once; every variable stores its whole grid,
      vacant sites as NaN
    - counts are for the whole dataset; the tables preview ``var``
    """
    coord_cols = [col for col in df.columns if col.startswith("x")]
    data_cols = [col for col in df.columns if col not in coord_cols]
    rows = len(df)
    tabular_nan = int(df[data_cols].isna().to_numpy().sum())
    grid_points = int(np.prod([ds.sizes[dim] for dim in ds.dims]))
    array_coords = int(sum(ds.sizes[dim] for dim in ds.dims))
    array_values = int(sum(ds[name].size for name in ds.data_vars))
    array_nan = int(sum(ds[name].isnull().sum() for name in ds.data_vars))

    ax.set_axis_off()
    ax.text(
        0.5,
        1.0,
        "Storage schema",
        ha="center",
        va="top",
        color="dimgray",
        fontsize=12,
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.95,
        f"sites holding a value: {rows} | grid points: {grid_points}",
        ha="center",
        va="top",
        color="dimgray",
        fontsize=8,
        transform=ax.transAxes,
    )

    # each area leaves room above it for its two-line title
    tab_ax = ax.inset_axes([0.02, 0.48, 0.96, 0.33])
    arr_ax = ax.inset_axes([0.02, 0.0, 0.96, 0.33])

    preview = df[coord_cols + [var]]
    cells, _, labels, note = _truncate(
        [[format_table_value(value) for value in row] for row in preview.to_numpy()],
        None,
        list(preview.columns),
    )
    draw_table(
        tab_ax,
        cells,
        labels,
        title=(
            f"Tabular: {rows} rows x {len(df.columns)} columns, "
            f"{tabular_nan} NaN cells{note}\n"
            f"{rows * len(coord_cols)} coordinate values + "
            f"{rows * len(data_cols)} data values"
        ),
    )

    data_var = ds[var]
    record = data_var.values
    if record.ndim == 1:
        record = record[np.newaxis, :]
        row_labels = None
    else:
        row_labels = [format_table_value(v) for v in ds[data_var.dims[0]].values]
    cells, row_labels, labels, note = _truncate(
        [["NaN" if np.isnan(v) else f"{v:.3f}" for v in row] for row in record],
        row_labels,
        [format_table_value(v) for v in ds[data_var.dims[-1]].values],
    )
    draw_table(
        arr_ax,
        cells,
        labels,
        row_labels=row_labels,
        title=(
            f"Array: {array_coords} coordinate values + "
            f"{array_values} grid values{note}\n"
            f"vacant sites stored as NaN: {array_nan}"
        ),
    )

    return ax


def _plot_observation_masks(ax, x_values, y_values, values, color):
    """Plot observed and missing points using a shared mask-based routine."""
    mask_finite = np.isfinite(values)
    mask_nans = np.isnan(values)
    masks = [mask_finite, mask_nans]
    for k, mask in enumerate(masks):
        face = color if k == 0 else "none"
        ax.scatter(
            np.asarray(x_values).ravel()[mask.ravel()],
            np.asarray(y_values).ravel()[mask.ravel()],
            facecolors=face,
            edgecolors=color,
            s=50,
            linewidths=1.2,
            zorder=3,
        )


def _plot_full_grid_support(ax, x_values, y_values):
    """Draw the full 2D support grid for the main comparison plot."""
    support_x, support_y = np.meshgrid(x_values, y_values)
    ax.scatter(
        support_x.ravel(),
        support_y.ravel(),
        s=160,
        facecolors="none",
        edgecolors="dimgray",
        linewidths=1.2,
        zorder=2,
    )
    for xv in x_values:
        ax.axvline(xv, color="dimgray", linestyle=":", linewidth=1, zorder=0)
    for yv in y_values:
        ax.axhline(yv, color="dimgray", linestyle=":", linewidth=1, zorder=0)
    return support_x, support_y


def plot_grid_case(ds, df, var, var_id, title):
    """Plot a 1D or 2D grid case and show the storage-schema comparison."""
    var_dims = ds[var].dims
    if len(var_dims) not in (1, 2):
        raise ValueError(
            f"plot_grid_case supports only 1D or 2D variables, got {len(var_dims)}D"
        )

    fig = plt.figure(figsize=(13, 5))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.15)
    ax = fig.add_subplot(outer[0, 0])

    colors = ["green", "orange", "blue"]
    values = ds[var].values

    if len(var_dims) == 2:
        y_dim, x_dim = var_dims
        y_values = ds[y_dim].values
        x_values = ds[x_dim].values
        support_x, support_y = _plot_full_grid_support(ax, x_values, y_values)
        _plot_observation_masks(ax, support_x, support_y, values, colors[var_id])

        ax.set_xticks(x_values)
        ax.set_yticks(y_values)
        ax.set_xticklabels([f"{value:.3f}" for value in x_values], color="dimgray")
        ax.set_yticklabels([f"{value:.3f}" for value in y_values], color="dimgray")
        ax.set_xlabel(x_dim, color="dimgray")
        ax.set_ylabel(y_dim, color="dimgray")
        ax.set_aspect("equal", adjustable="box")
    else:
        if len(ds.dims) == 2:
            data_dims = list(ds.dims)
            y_dim, x_dim = data_dims[0], data_dims[1]
            y_values = ds[y_dim].values
            x_values = ds[x_dim].values
            support_x, support_y = _plot_full_grid_support(ax, x_values, y_values)

            observed = df[df[var].notna()]
            ax.scatter(
                observed[x_dim].to_numpy(),
                observed[y_dim].to_numpy(),
                facecolors=colors[var_id],
                edgecolors=colors[var_id],
                s=50,
                linewidths=1.2,
                zorder=3,
            )

            ax.set_xticks(x_values)
            ax.set_yticks(y_values)
            ax.set_xticklabels([f"{value:.3f}" for value in x_values], color="dimgray")
            ax.set_yticklabels([f"{value:.3f}" for value in y_values], color="dimgray")
            ax.set_xlabel(x_dim, color="dimgray")
            ax.set_ylabel(y_dim, color="dimgray")
            ax.set_aspect("equal", adjustable="box")
        else:
            x_dim = var_dims[0]
            x_values = ds[x_dim].values
            support_y = np.zeros_like(x_values, dtype=float)
            ax.axhline(0.0, color="dimgray", linestyle=":", linewidth=1, zorder=0)
            ax.scatter(
                x_values,
                support_y,
                s=160,
                facecolors="none",
                edgecolors="dimgray",
                linewidths=1.2,
                zorder=2,
            )

            _plot_observation_masks(ax, x_values, support_y, values, colors[var_id])

            ax.set_xticks(x_values)
            ax.set_xticklabels([f"{value:.3f}" for value in x_values], color="dimgray")
            ax.set_yticks([])
            ax.set_xlabel(x_dim, color="dimgray")
            ax.set_ylabel("present", color="dimgray")

    ax.set_title(title, color="dimgray")
    ax.tick_params(axis="both", colors="dimgray")
    for spine in ax.spines.values():
        spine.set_color("dimgray")
    ax.grid(False)

    schema_ax = fig.add_subplot(outer[0, 1])
    draw_storage_schema(schema_ax, ds, df, var)
    plt.tight_layout()
    plt.show()
