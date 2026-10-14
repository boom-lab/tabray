"""Statistics of a netCDF or parquet dataset, measured from the file.

Counts occupied cells rather than reading the generator's bookkeeping, so a
placement bug shows up here instead of being echoed.
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import dask
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
                var_cells = int(proj_var.sum())
                # empty variable: 0 / 0
                overlap_reverse = shared / var_cells if var_cells else np.nan
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
            var_dims[name], ambiguous_coords = DatasetDescription._infer_dims(
                frame, held, coords, grid
            )
            if ambiguous_coords:
                warnings.warn(
                    f"{name}: the rows cannot tell whether it varies along "
                    f"{ambiguous_coords} or drops them (one value per row, or a "
                    "single value, on every row of its keys); read as varying.",
                    UserWarning,
                    stacklevel=3,
                )
            # a variable that drops every coord: one site, like a scalar
            occupied[name] = (
                len(held[var_dims[name]].drop_duplicates()) if var_dims[name] else 1
            )
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
                var_cells = int(proj_var.shape[0].compute())
                # empty variable: 0 / 0
                overlap_reverse = shared / var_cells if var_cells else np.nan
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
    def _infer_dims(
        frame: dd.DataFrame, held: dd.DataFrame, coords: List[str], grid: Dict
    ) -> Tuple[List[str], List[str]]:
        """A variable's dims, from the rows it holds.

        A variable that drops coords is written on every row matching its own
        dims (its key), with one value per key: ParquetBuilder does this, and
        so do Argo-style tables (a profile's latitude on each of its levels).

        1. a held row with NaN on a coord: drops it (an unmatched value)
        2. a set of further coords the rows allow dropping (``_consistent``)
           needs evidence: a value seen at several positions along them,
           and more than one value overall. The largest such set is dropped.
        3. a coord the rows would also let it drop, without evidence (each
           value on one row, or a single value overall): both readings give
           these rows, so it is ambiguous and read as varying

        Consistency is monotone (a variable consistent on some dims is on
        any larger set), so a set is tried only if each of its subsets passed.

        Args:
            frame: Every row of the parquet
            held: The variable's rows, columns ``coords + [name]``
            coords: Coordinate columns
            grid: Number of distinct values per coord; a coord of length 1 is
                kept, as netCDF keeps it

        Returns:
            (dims, ambiguous_coords): ambiguous coords are among the dims
        """
        name = held.columns[-1]
        has_nan_coord = dask.compute(*[held[c].isnull().any() for c in coords])
        not_nan_coords = [c for c, has_nan in zip(coords, has_nan_coord) if not has_nan]
        num_held_rows, num_values, num_positions = dask.compute(
            held.shape[0],
            held[name].nunique(),
            held[not_nan_coords].drop_duplicates().shape[0],
        )
        if num_held_rows == 0:  # empty: nothing to infer, density is 0 anyway
            return not_nan_coords, []
        droppable = [c for c in not_nan_coords if grid[c] > 1]
        allowed_drops, evidenced_drops = [], []
        frontier = [()]
        while frontier:
            larger = []
            for subset in frontier:
                start = droppable.index(subset[-1]) + 1 if subset else 0
                for coord in droppable[start:]:
                    drop = subset + (coord,)
                    if any(
                        drop[:i] + drop[i + 1 :] not in allowed_drops
                        for i in range(len(drop))
                        if len(drop) > 1
                    ):
                        continue  # a subset already failed
                    key = [c for c in not_nan_coords if c not in drop]
                    if not DatasetDescription._consistent(
                        frame, held, key, num_held_rows
                    ):
                        continue
                    allowed_drops.append(drop)
                    larger.append(drop)
                    num_keys = (
                        int(held[key].drop_duplicates().shape[0].compute())
                        if key
                        else 1
                    )
                    # a value at several positions along the dropped coords
                    if num_positions > num_keys and num_values > 1:
                        evidenced_drops.append(drop)
            frontier = larger
        dropped = max(evidenced_drops, key=len, default=())
        dims = [c for c in not_nan_coords if c not in dropped]
        # a dim the rows would also let it drop, with no evidence either way
        ambiguous_coords = [
            c
            for c in dims
            if tuple(sorted(dropped + (c,), key=droppable.index)) in allowed_drops
        ]
        return dims, ambiguous_coords

    @staticmethod
    def _consistent(
        frame: dd.DataFrame, held: dd.DataFrame, key: List[str], num_held_rows: int
    ) -> bool:
        """Whether the rows fit a variable on ``key`` only.

        - one value per key
        - on every row of the parquet with a key it holds: a variable missing
          from a row of its key (held by another variable) varies along the
          coords that row differs in
        """
        name = held.columns[-1]
        if not key:  # a scalar: one value, on every row
            num_values, num_rows = dask.compute(held[name].nunique(), frame.shape[0])
            return num_values == 1 and num_rows == num_held_rows
        held_keys = held[key].drop_duplicates()
        num_keys, num_key_values, num_key_rows = dask.compute(
            held_keys.shape[0],
            held[key + [name]].drop_duplicates().shape[0],
            frame[key].merge(held_keys, on=key).shape[0],
        )
        return num_key_values == num_keys and num_key_rows == num_held_rows
