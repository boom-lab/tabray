"""Tutorial notebook orchestration helpers."""

from pathlib import Path
import os

import pandas as pd
import xarray as xr

from data_sparsity.generate_data import GenerateData
from data_sparsity.utils.viz import (
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
        num_vars = None,
        var_dims = None,
        overlap = None,
):
    """Generate a tutorial case, print size statistics, and plot the result."""

    ncpath, pqpath, pqpathtmp = case_paths(case_name, base_dir=base_dir)
    if num_vars is None or num_vars==1:
        gen = GenerateData(
            num_obs=num_obs,
            num_dims=2,
            ratio_dims=1,
            density=density,
            seed=seed,
        )        
        
    else:
        gen = GenerateData(
            num_obs=num_obs,
            num_dims=2,
            ratio_dims=1,
            density=density,
            seed=seed,
            num_vars=num_vars,
            var_dims=var_dims,
            overlap=overlap
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
    for var_id, var in enumerate(ds.data_vars):
        plot_grid_case(ds, df, var, var_id, title)
    return ds, df
