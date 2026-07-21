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
        return f"{value:.3f}"
    return str(value)


def parquet_disk_size(parquet_path: str) -> int:
    """Return the total on-disk size for a parquet file or directory."""
    path = Path(parquet_path)
    if not path.exists():
        path = path.parent
    if path.is_file():
        return path.stat().st_size
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())


def draw_table(ax, cell_text, col_labels, row_labels=None, title=None):
    """Draw a styled Matplotlib table on the provided axes."""
    ax.set_axis_off()
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        rowLabels=row_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.2)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("dimgray")
        if row == 0:
            cell.set_facecolor("#e6e6e6")
            cell.set_text_props(color="dimgray", weight="bold")
        elif col == -1:
            cell.set_facecolor("#f5f5f5")
            cell.set_text_props(color="dimgray")
    if title is not None:
        ax.set_title(title, color="dimgray", pad=6)
    return table


def draw_storage_schema(ax, ds, df, var):
    """Draw the storage comparison schema for the tutorial figures."""
    data_var = ds[var]
    occupied_sites = len(df)
    num_dims = len(ds.dims)
    num_vars = len(ds.data_vars)
    grid_shape = [int(ds.sizes[dim]) for dim in ds.dims]
    total_grid_points = int(np.prod(grid_shape))
    tabular_rows = occupied_sites
    tabular_values = occupied_sites * (num_dims + num_vars)
    array_coord_values = sum(grid_shape)
    array_values = total_grid_points * num_vars
    array_missing = total_grid_points - occupied_sites

    ax.set_axis_off()
    ax.text(
        0.5,
        0.98,
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
        0.93,
        f"occupied sites: {occupied_sites} | total grid points: {total_grid_points}",
        ha="center",
        va="top",
        color="dimgray",
        fontsize=8,
        transform=ax.transAxes,
    )

    tab_ax = ax.inset_axes([0.02, 0.53, 0.96, 0.34])
    arr_ax = ax.inset_axes([0.02, 0.07, 0.96, 0.34])

    tabular_preview = df[[col for col in df.columns if col.startswith("x") or col == var]].head(occupied_sites)
    tabular_cell_text = [[format_table_value(value) for value in row] for row in tabular_preview.to_numpy()]
    draw_table(
        tab_ax,
        tabular_cell_text,
        list(tabular_preview.columns),
        title=(
            f"Tabular: {tabular_rows} rows | {tabular_values} stored values\n"
            f"{occupied_sites * num_dims} repeated coordinate values + {occupied_sites * num_vars} data values"
        ),
    )

    record = data_var.values
    if record.ndim == 1:
        array_cell_text = [[
            "NaN" if np.isnan(value) else f"{value:.3f}"
            for value in record
        ]]
        draw_table(
            arr_ax,
            array_cell_text,
            [format_table_value(value) for value in ds[data_var.dims[0]].values],
            row_labels=None,
            title=(
                f"Array: {array_coord_values} coordinate values + {array_values} grid values\n"
                f"empty sites stored as NaN: {array_missing}"
            ),
        )
    else:
        array_cell_text = [
            ["NaN" if np.isnan(value) else f"{value:.3f}" for value in row]
            for row in record
        ]
        draw_table(
            arr_ax,
            array_cell_text,
            [format_table_value(value) for value in ds[data_var.dims[1]].values],
            row_labels=[format_table_value(value) for value in ds[data_var.dims[0]].values],
            title=(
                f"Array: {array_coord_values} coordinate values + {array_values} grid values\n"
                f"empty sites stored as NaN: {array_missing}"
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

    colors = ["green", "orange"]
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
