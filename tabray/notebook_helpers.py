"""Tutorial notebook orchestration helpers."""

from pathlib import Path
import os

import pandas as pd
import xarray as xr

from tabray.generate_data import GenerateData
from tabray.utils.viz import (
    draw_storage_schema,
    draw_table,
    format_bytes,
    format_table_value,
    parquet_disk_size,
    plot_grid_case,
)

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


def case_paths(case_name: str, base_dir: str = "./tutorial1") -> tuple[str, str, str]:
    """Return the NetCDF and Parquet output paths for a tutorial case."""
    case_root = Path(base_dir) / case_name
    return (
        str(case_root / "netCDF" / "ds.nc"),
        str(case_root / "parquet" / "ddf"),
        str(case_root / "parquet" / "tmp"),
    )


def run_case(
    case_name,
    num_obs,
    density,
    seed,
    title,
    base_dir: str = "./tutorial1",
    num_vars=None,
    var_dims=None,
    overlap=None,
    fixed_overlap=False,
    **gen_kwargs,
):
    """Generate a tutorial case, print size statistics, and plot the result.

    - gen_kwargs: further GenerateData arguments (num_dims, ratio_dims,
      layout, max_obs, ...); num_dims=2 and ratio_dims=1 unless given
    - parallel run (max_obs < num_obs): the chunk netCDF files are read as one
      dataset
    - 3D grid: one figure per coordinate of the first dimension
    """

    ncpath, pqpath, pqpathtmp = case_paths(case_name, base_dir=base_dir)
    params = {"num_dims": 2, "ratio_dims": 1, **gen_kwargs}
    if num_vars is not None and num_vars > 1:
        params.update(
            num_vars=num_vars,
            var_dims=var_dims,
            fixed_overlap=fixed_overlap,
        )
        if overlap is not None:  # else GenerateData's default, "random"
            params["overlap"] = overlap
    gen = GenerateData(num_obs=num_obs, density=density, seed=seed, **params)

    gen.generate(
        netcdf_filepath=ncpath,
        parquet_filepath=pqpath,
        parquet_tmp=pqpathtmp,
    )
    if gen.NTASKS > 1:
        # parallel run: one netCDF file per chunk, no file at ncpath
        # pylint: disable-next=protected-access
        chunks, chunk_files = gen._open_netcdf_chunks()
        ds = chunks.load()
        chunks.close()
        nc_disk_bytes = sum(Path(f).stat().st_size for f in chunk_files)
    else:
        ds = xr.open_dataset(ncpath).load()
        nc_disk_bytes = Path(ncpath).stat().st_size
    df = pd.read_parquet(os.path.dirname(pqpath))

    pq_disk_bytes = parquet_disk_size(pqpath)
    ds_memory_bytes = ds.nbytes
    df_memory_bytes = df.memory_usage(index=True, deep=True).sum()

    print(f"Loaded netCDF into xarray: {format_bytes(ds_memory_bytes)} in memory")
    print(f"Loaded parquet into pandas: {format_bytes(df_memory_bytes)} in memory")
    print(f"On-disk netCDF size: {format_bytes(nc_disk_bytes)}")
    print(f"On-disk parquet size: {format_bytes(pq_disk_bytes)}")
    for var_id, var in enumerate(ds.data_vars):
        if len(ds.dims) <= 2:
            plot_grid_case(ds, df, var, var_id, title)
            continue
        # plot_grid_case draws 1D and 2D only: slice along the first dim
        first = list(ds.dims)[0]
        for value in ds[first].values:
            plot_grid_case(
                ds.sel({first: value}),
                df[df[first] == value],
                var,
                var_id,
                f"{title}, {first} = {value:.3f}",
            )
    return ds, df
