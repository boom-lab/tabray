"""Statistics of a netCDF or parquet dataset, measured from the file.

Counts occupied cells rather than reading the generator's bookkeeping, so a
placement bug shows up here instead of being echoed.
"""

import os
from typing import Dict, List, Optional, Union

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
        reference: Optional[str] = None,
    ) -> pd.DataFrame:
        """Describe a dataset, read lazily.

        - var_sites = prod(axis lengths of the variable's dims)
        - density = var_sites_occupied / var_sites
        - overlap = |proj(S0) & proj(Si)| / |proj(S0)|, S0 = the reference
          variable (NaN for it), proj = onto the dims S0 and Si share
          (= Si's dims when S0 holds every dim, as var0 does)
        - overlap_reverse = |proj(S0) & proj(Si)| / |proj(Si)|
        - no shared dim -> both NaN
        - reference = ``reference`` if given, else the variable with the most
          occupied sites (ties: first in file order); its row comes first
        - unused_coords = coordinates along the variable's dims holding no value

        Args:
            source: netCDF path, an open xarray object, or a parquet
                file, glob or directory
            coords: Coordinate columns; required for parquet, unused otherwise
            reference: Variable to measure overlap against (var0)

        Returns:
            DataFrame indexed by variable; ``attrs["grid"]`` maps dim to length
        """
        is_parquet = isinstance(source, str) and (
            source.endswith(".parquet") or os.path.isdir(source)
        )
        if is_parquet:
            if not coords:
                raise ValueError("parquet needs its coordinate columns named")
            return DatasetDescription._from_parquet(source, coords, reference)
        if isinstance(source, str):
            source = xr.open_dataset(source, chunks={})
        if isinstance(source, xr.DataArray):
            source = source.to_dataset(name=source.name or "record")
        return DatasetDescription._from_xarray(source, reference)

    @staticmethod
    def _pick_reference(occupied: Dict[str, int], reference: Optional[str]) -> str:
        """``reference`` if given, else argmax of occupied sites (first on ties)."""
        if reference is None:
            return max(occupied, key=occupied.get)
        if reference not in occupied:
            raise ValueError(
                f"reference {reference!r} is not a variable: {list(occupied)}"
            )
        return reference

    @staticmethod
    def _from_xarray(
        dataset: xr.Dataset, reference: Optional[str] = None
    ) -> pd.DataFrame:
        masks = {name: dataset[name].notnull() for name in dataset.data_vars}
        occupied = {name: int(mask.sum()) for name, mask in masks.items()}
        ref_name = DatasetDescription._pick_reference(occupied, reference)
        ref = masks[ref_name]
        order = [ref_name] + [n for n in masks if n != ref_name]
        rows = {}
        for name in order:
            mask = masks[name]
            var_sites_occupied = occupied[name]
            var_sites = int(np.prod(mask.shape))
            density = var_sites_occupied / var_sites if var_sites else 0.0
            common = [d for d in ref.dims if d in mask.dims]
            if name == ref_name or not common:
                overlap = overlap_reverse = np.nan
            else:
                proj_ref = ref.any(dim=[d for d in ref.dims if d not in common])
                proj_var = mask.any(dim=[d for d in mask.dims if d not in common])
                shared = int((proj_ref & proj_var).sum())
                overlap = shared / max(int(proj_ref.sum()), 1)
                overlap_reverse = shared / max(int(proj_var.sum()), 1)
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
    def _from_parquet(
        path: str, coords: List[str], reference: Optional[str] = None
    ) -> pd.DataFrame:
        """Parquet has no axes, so two things are inferred from the rows.

        - grid: the coordinate values that appear in any row
        - a variable's dims: the coords it does not drop (``_drops``)
        - occupied sites: distinct rows over the variable's dims
        """
        frame = dd.read_parquet(path)
        names = [c for c in frame.columns if c not in coords]
        grid = {c: int(frame[c].nunique().compute()) for c in coords}
        var_dims, seen, occupied = {}, {}, {}
        for name in names:
            held = frame[frame[name].notnull()][coords + [name]]
            seen[name] = {c: int(held[c].nunique().compute()) for c in coords}
            var_dims[name] = [
                c
                for c in coords
                if grid[c] == 1 or not DatasetDescription._drops(held, coords, c)
            ]
            occupied[name] = len(held[var_dims[name]].drop_duplicates())
        ref_name = DatasetDescription._pick_reference(occupied, reference)
        ref_rows = frame[frame[ref_name].notnull()][coords]
        rows = {}
        for name in [ref_name] + [n for n in names if n != ref_name]:
            held = frame[frame[name].notnull()][coords]
            dims = var_dims[name]
            var_sites_occupied = occupied[name]
            var_sites = int(np.prod([grid[c] for c in dims]))
            density = var_sites_occupied / var_sites if var_sites else 0.0
            if name == ref_name:
                ref_dims = dims  # the reference is the first row built
            common = [c for c in ref_dims if c in dims]
            if name == ref_name or not common:
                overlap = overlap_reverse = np.nan
            else:
                proj_ref = ref_rows[common].drop_duplicates()
                proj_var = held[common].drop_duplicates()
                shared = int(proj_ref.merge(proj_var, on=common).shape[0].compute())
                overlap = shared / max(int(proj_ref.shape[0].compute()), 1)
                overlap_reverse = shared / max(int(proj_var.shape[0].compute()), 1)
            unused_coords = sum(grid[c] - seen[name][c] for c in dims)
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

    @staticmethod
    def _drops(held: dd.DataFrame, coords: List[str], coord: str) -> bool:
        """Whether a variable drops ``coord``, from the rows it holds.

        - a row with ``coord`` NaN: the variable has no position along it
        - repeated: one value per key over the other coords (V == K) and some
          key on several rows (R > K); K, V = distinct keys, (key, value) pairs
        - a variable constant along a real dim reads as repeated
        """
        name = held.columns[-1]
        other = [c for c in coords if c != coord]
        if bool(held[coord].isnull().any().compute()):
            return True
        if not other:
            return False
        keys = len(held[other].drop_duplicates())
        values = len(held[other + [name]].drop_duplicates())
        return values == keys and len(held) > keys
