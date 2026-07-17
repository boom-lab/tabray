"""Shared notebook helpers for the tutorial notebooks.

The tutorial notebooks use these utilities to generate example datasets,
format size output, and render comparison figures. Keeping the logic in source
code avoids duplicated notebook cells and makes the tutorials easier to maintain.
"""

from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from data_sparsity.generate_data import GenerateData

__all__ = [
    "case_paths",
    "format_bytes",
    "parquet_disk_size",
    "format_table_value",
    "draw_table",
    "draw_storage_schema",
    "plot_grid_case",
    "run_case",
]


def case_paths(case_slug: str, base_dir: str = "./tutorial1") -> tuple[str, str, str]:
    """Return the NetCDF and Parquet output paths for a tutorial case."""
    case_root = Path(base_dir) / case_slug
    return (
        str(case_root / "netCDF" / "ds.nc"),
        str(case_root / "parquet" / "ddf"),
        str(case_root / "parquet" / "tmp"),
    )


def format_bytes(num_bytes: int | float) -> str:
    """Format a byte count using binary units."""
    units = ["B", "KiB", "MiB", "GiB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} GiB"


def parquet_disk_size(parquet_path: str) -> int:
    """Return the total on-disk size for a parquet file or directory."""
    path = Path(parquet_path)
    if not path.exists():
        path = path.parent
    if path.is_file():
        return path.stat().st_size
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())


def format_table_value(value: object) -> str:
    """Format table cell values for compact display."""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


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


def draw_storage_schema(ax, ds, df):
    """Draw the storage comparison schema for the tutorial figures."""
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

    tabular_preview = df[[col for col in df.columns if col.startswith("x") or col == "record"]].head(occupied_sites)
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

    record = ds["record"].values
    array_cell_text = [["NaN" if np.isnan(value) else f"{value:.3f}" for value in row] for row in record]
    draw_table(
        arr_ax,
        array_cell_text,
        [format_table_value(value) for value in ds["x1"].values],
        row_labels=[format_table_value(value) for value in ds["x0"].values],
        title=(
            f"Array: {array_coord_values} coordinate values + {array_values} grid values\n"
            f"empty sites stored as NaN: {array_missing}"
        ),
    )

    return ax


def plot_grid_case(ds, df, title):
    """Plot a 2D grid case and show the storage-schema comparison."""
    x0 = ds["x0"].values
    x1 = ds["x1"].values
    support_x1, support_x0 = np.meshgrid(x1, x0)
    present_mask = np.isfinite(ds["record"].values)

    fig = plt.figure(figsize=(13, 5))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.15)
    ax = fig.add_subplot(outer[0, 0])
    ax.scatter(
        support_x1.ravel(),
        support_x0.ravel(),
        s=160,
        facecolors="none",
        edgecolors="dimgray",
        linewidths=1.2,
        zorder=2,
    )

    for xv in x1:
        ax.axvline(xv, color="dimgray", linestyle=":", linewidth=1, zorder=0)
    for yv in x0:
        ax.axhline(yv, color="dimgray", linestyle=":", linewidth=1, zorder=0)

    ax.scatter(
        support_x1.ravel()[present_mask.ravel()],
        support_x0.ravel()[present_mask.ravel()],
        facecolors="green",
        edgecolors="green",
        s=50,
        linewidths=1.2,
        zorder=3,
    )

    ax.set_xticks(x1)
    ax.set_yticks(x0)
    ax.set_xticklabels([f"{value:.3f}" for value in x1], color="dimgray")
    ax.set_yticklabels([f"{value:.3f}" for value in x0], color="dimgray")
    ax.set_xlabel("x1", color="dimgray")
    ax.set_ylabel("x0", color="dimgray")
    ax.set_title(title, color="dimgray")
    ax.tick_params(axis="both", colors="dimgray")
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_color("dimgray")
    ax.grid(False)

    schema_ax = fig.add_subplot(outer[0, 1])
    draw_storage_schema(schema_ax, ds, df)
    plt.tight_layout()
    plt.show()


def run_case(case_slug, num_obs, density, seed, title, base_dir: str = "./tutorial1"):
    """Generate a tutorial case, print size statistics, and plot the result."""
    ncpath, pqpath, pqpathtmp = case_paths(case_slug, base_dir=base_dir)
    gen = GenerateData(
        num_obs=num_obs,
        num_dims=2,
        ratio_dims=1,
        density=density,
        seed=seed,
    )
    gen.generate(
        netcdf_filepath=ncpath,
        parquet_filepath=pqpath,
        parquet_tmp=pqpathtmp,
    )
    ds = xr.open_dataset(ncpath).load()
    df = pd.read_parquet(os.path.dirname(pqpath))

    nc_disk_bytes = Path(ncpath).stat().st_size
    pq_disk_bytes = parquet_disk_size(pqpath)
    ds_memory_bytes = ds.nbytes
    df_memory_bytes = df.memory_usage(index=True, deep=True).sum()

    print(f"Loaded netCDF into xarray: {format_bytes(ds_memory_bytes)} in memory")
    print(f"Loaded parquet into pandas: {format_bytes(df_memory_bytes)} in memory")
    print(f"On-disk netCDF size: {format_bytes(nc_disk_bytes)}")
    print(f"On-disk parquet size: {format_bytes(pq_disk_bytes)}")
    plot_grid_case(ds, df, title)
    return ds, df
