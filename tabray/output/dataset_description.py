"""Statistics of a netCDF or parquet dataset, measured from the file.

Counts occupied cells rather than reading the generator's bookkeeping, so a
placement bug shows up here instead of being echoed.
"""

import os
from typing import List, Optional, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr


class DatasetDescription:
    """One row per variable: dims, sites, occupied sites, density, overlap."""

    COLUMNS = [
        "dims",
        "var_sites",
        "var_sites_occupied",
        "density",
        "overlap",
        "overlap_reverse",
        "unused_coords",
        "dtype",
    ]

    @staticmethod
    def describe_dataset(
        source: Union[str, xr.Dataset, xr.DataArray],
        coords: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Describe a dataset, read lazily.

        - var_sites = prod(axis lengths of the variable's dims)
        - density = var_sites_occupied / var_sites
        - overlap = |proj(S0) & Si| / |proj(S0)| on the variable's dims, S0 = the
          first variable (NaN for it)
        - overlap_reverse = |proj(S0) & Si| / |Si| (NaN for the first variable)
        - unused_coords = coordinates along the variable's dims holding no value

        Args:
            source: netCDF path, an open xarray object, or a parquet
                file, glob or directory
            coords: Coordinate columns; required for parquet, unused otherwise

        Returns:
            DataFrame indexed by variable; ``attrs["grid"]`` maps dim to length
        """
        is_parquet = isinstance(source, str) and (
            source.endswith(".parquet") or os.path.isdir(source)
        )
        if is_parquet:
            if not coords:
                raise ValueError("parquet needs its coordinate columns named")
            return DatasetDescription._from_parquet(source, coords)
        if isinstance(source, str):
            source = xr.open_dataset(source, chunks={})
        if isinstance(source, xr.DataArray):
            source = source.to_dataset(name=source.name or "record")
        return DatasetDescription._from_xarray(source)

    @staticmethod
    def _from_xarray(dataset: xr.Dataset) -> pd.DataFrame:
        masks = {name: dataset[name].notnull() for name in dataset.data_vars}
        ref = next(iter(masks.values()))
        rows = {}
        for name, mask in masks.items():
            var_sites_occupied = int(mask.sum())
            var_sites = int(np.prod(mask.shape))
            density = var_sites_occupied / var_sites if var_sites else 0.0
            drop = [d for d in ref.dims if d not in mask.dims]
            proj_ref = ref.any(dim=drop) if drop else ref
            if mask is ref:
                overlap = overlap_reverse = np.nan
            else:
                shared = int((proj_ref & mask).sum())
                overlap = shared / max(int(proj_ref.sum()), 1)
                overlap_reverse = shared / max(var_sites_occupied, 1)
            unused_coords = sum(
                int((~mask.any(dim=[o for o in mask.dims if o != d])).sum())
                for d in mask.dims
            )
            dtype = str(dataset[name].encoding.get("dtype", dataset[name].dtype))
            rows[name] = [
                tuple(str(d) for d in mask.dims),
                var_sites,
                var_sites_occupied,
                density,
                overlap,
                overlap_reverse,
                unused_coords,
                dtype,
            ]
        table = pd.DataFrame.from_dict(
            rows,
            orient="index",
            columns=DatasetDescription.COLUMNS,
        )
        table.attrs["grid"] = {str(d): int(n) for d, n in dataset.sizes.items()}
        return table

    @staticmethod
    def _from_parquet(path: str, coords: List[str]) -> pd.DataFrame:
        """Parquet has no axes, so two things are inferred from the rows.

        - grid: the coordinate values that appear in any row
        - a variable's dims: the coords along which it takes more than one
          value (a variable with one value per axis looks constant)
        """
        frame = dd.read_parquet(path)
        names = [c for c in frame.columns if c not in coords]
        grid = {c: int(frame[c].nunique().compute()) for c in coords}
        ref_rows = frame[frame[names[0]].notnull()][coords]
        rows = {}
        for name in names:
            held = frame[frame[name].notnull()][coords]
            seen = {c: int(held[c].nunique().compute()) for c in coords}
            dims = [c for c in coords if seen[c] > 1 or grid[c] == 1]
            var_sites_occupied = int(held.shape[0].compute())
            var_sites = int(np.prod([grid[c] for c in dims]))
            density = var_sites_occupied / var_sites if var_sites else 0.0
            if name == names[0]:
                overlap = overlap_reverse = np.nan
            else:
                proj_ref = ref_rows[dims].drop_duplicates()
                proj_var = held[dims].drop_duplicates()
                shared = int(proj_ref.merge(proj_var, on=dims).shape[0].compute())
                overlap = shared / max(int(proj_ref.shape[0].compute()), 1)
                overlap_reverse = shared / max(int(proj_var.shape[0].compute()), 1)
            unused_coords = sum(grid[c] - seen[c] for c in dims)
            rows[name] = [
                tuple(dims),
                var_sites,
                var_sites_occupied,
                density,
                overlap,
                overlap_reverse,
                unused_coords,
                str(frame[name].dtype),
            ]
        table = pd.DataFrame.from_dict(
            rows,
            orient="index",
            columns=DatasetDescription.COLUMNS,
        )
        table.attrs["grid"] = grid
        return table
