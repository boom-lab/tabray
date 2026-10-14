"""Statistics of a netCDF or parquet dataset, measured from the file.

Counts occupied cells rather than reading the generator's bookkeeping, so a
placement bug shows up here instead of being echoed.
"""

import os
import warnings
from itertools import combinations
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
        """Parquet stores no axes and no variable dims, so both come from the rows.

        - grid: the coordinate values that appear in any row
        - a variable's dims, the dims it is defined on (``_infer_dims``)
        - occupied sites: distinct rows over the variable's dims
        """
        parquet_rows = dd.read_parquet(path)
        names = [c for c in parquet_rows.columns if c not in coords]
        grid = {c: int(parquet_rows[c].nunique().compute()) for c in coords}
        var_dims, seen, occupied = {}, {}, {}
        for name in names:
            var_rows = parquet_rows[parquet_rows[name].notnull()][coords + [name]]
            seen[name] = {c: int(var_rows[c].nunique().compute()) for c in coords}
            var_dims[name], assumed_constant, ambiguous_dims = (
                DatasetDescription._infer_dims(parquet_rows, var_rows, coords, grid)
            )
            DatasetDescription._report_assumptions(
                name, assumed_constant, ambiguous_dims
            )
            # no dims: one site, like a scalar
            occupied[name] = (
                len(var_rows[var_dims[name]].drop_duplicates()) if var_dims[name] else 1
            )
        ref_name = DatasetDescription._pick_reference(occupied, reference)
        ref_rows = parquet_rows[parquet_rows[ref_name].notnull()][coords]
        rows = {}
        for name in [ref_name] + [n for n in names if n != ref_name]:
            var_rows = parquet_rows[parquet_rows[name].notnull()][coords]
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
                proj_var = var_rows[common].drop_duplicates()
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
                str(parquet_rows[name].dtype),
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
        parquet_rows: dd.DataFrame,
        var_rows: dd.DataFrame,
        coords: List[str],
        grid: Dict,
    ) -> Tuple[List[str], List[str], List[str]]:
        """A variable's dims, from the rows it holds.

        Each dim is a column of the table (one of ``coords``). A site of the
        variable is one combination of coordinates on its own dims. A variable
        with fewer dims than the table is written, with one value, on every
        row at each of its sites (ParquetBuilder writes it this way).

        - Certain: a row of the variable with NaN in a dim's column. The
          variable does not vary along that dim.
        - Assumed: the variable is taken as constant along a set of dims when
          both hold, assuming it is not defined on them: it has one value per
          site and is on every row at each of its sites (``_num_sites_if_fit``);
          and some site's value is on several rows with different coordinates
          on those dims, with more than one value overall. A variable that
          varies along them could give the same rows, so the assumption is
          reported (``assumed_constant``).
        - Ambiguous: the rows fit a variable constant along a dim, but no
          site's value repeats along it, or the variable holds a single value.
          The dim is kept and reported (``ambiguous_dims``).

        If the rows fit some dims they fit any larger set of dims, so a set of
        dims is tried only if each of its subsets passed.

        Args:
            parquet_rows: Every row of the parquet
            var_rows: The variable's rows, columns ``coords + [name]``
            coords: Coordinate columns
            grid: Number of distinct values per coord; a coord of length 1 is
                kept, as netCDF keeps it

        Returns:
            (dims, assumed_constant, ambiguous_dims): the dims left out by
            assumption, and the kept dims the rows would also fit without
        """
        name = var_rows.columns[-1]
        has_nan_coord = dask.compute(*[var_rows[c].isnull().any() for c in coords])
        dims_without_nan = [
            c for c, has_nan in zip(coords, has_nan_coord) if not has_nan
        ]
        # distinct rows on the dims no row leaves NaN; more of them than sites
        # on fewer dims means some site's value is on several rows
        num_var_rows, num_distinct_values, num_sites_on_dims_without_nan = dask.compute(
            var_rows.shape[0],
            var_rows[name].nunique(),
            var_rows[dims_without_nan].drop_duplicates().shape[0],
        )
        if num_var_rows == 0:  # empty: nothing to infer, density is 0 anyway
            return dims_without_nan, [], []
        dims_could_be_constant = [c for c in dims_without_nan if grid[c] > 1]
        # constant along no dim: every dim kept, the sets of size 1 are tried
        possible_constant_sets, constant_sets_with_repeats = {frozenset()}, []
        for num_constant_dims in range(1, len(dims_could_be_constant) + 1):
            for constant_dims in map(
                frozenset, combinations(dims_could_be_constant, num_constant_dims)
            ):
                if any(
                    constant_dims - {c} not in possible_constant_sets
                    for c in constant_dims
                ):
                    continue  # a subset did not fit
                num_sites = DatasetDescription._num_sites_if_fit(
                    parquet_rows,
                    var_rows,
                    [c for c in dims_without_nan if c not in constant_dims],
                    num_var_rows,
                )
                if num_sites is None:
                    continue
                possible_constant_sets.add(constant_dims)
                if (
                    num_sites_on_dims_without_nan > num_sites
                    and num_distinct_values > 1
                ):
                    constant_sets_with_repeats.append(constant_dims)
        assumed_constant = max(constant_sets_with_repeats, key=len, default=frozenset())
        dims = [c for c in dims_without_nan if c not in assumed_constant]
        ambiguous_dims = [
            c for c in dims if assumed_constant | {c} in possible_constant_sets
        ]
        return dims, [c for c in coords if c in assumed_constant], ambiguous_dims

    @staticmethod
    def _report_assumptions(
        name: str, assumed_constant: List[str], ambiguous_dims: List[str]
    ) -> None:
        """Print and warn when a variable's dims are not certain from the rows."""
        messages = []
        if assumed_constant:
            messages.append(
                f"{name} is ASSUMED constant along {assumed_constant}, so they are "
                f"not among its dims. Its rows fit this (one value at each site, "
                f"repeated on every row along {assumed_constant}) but do not prove "
                f"it: a variable varying along {assumed_constant} could give the "
                "same rows."
            )
        if ambiguous_dims:
            messages.append(
                f"{name} is ASSUMED to vary along {ambiguous_dims}. Its rows also "
                "fit a variable constant along one of them, and nothing in the "
                "rows tells the two apart."
            )
        for message in messages:
            print(f"WARNING: dims of {name} are not certain. {message}")
            warnings.warn(message, UserWarning, stacklevel=4)

    @staticmethod
    def _num_sites_if_fit(
        parquet_rows: dd.DataFrame,
        var_rows: dd.DataFrame,
        varying_dims: List[str],
        num_var_rows: int,
    ) -> Optional[int]:
        """The variable's number of sites on ``varying_dims``, if its rows fit.

        A site is one combination of coordinates on ``varying_dims``. The
        rows fit a variable on those dims only when it has

        - one value per site
        - a value on every row of the parquet at a site it holds: a variable
          missing from such a row (held by another variable) varies along the
          coords that row differs in

        Returns:
            The number of sites, or None if the rows do not fit
        """
        name = var_rows.columns[-1]
        if not varying_dims:  # a scalar: one value, on every row
            num_values, num_parquet_rows = dask.compute(
                var_rows[name].nunique(), parquet_rows.shape[0]
            )
            return 1 if num_values == 1 and num_parquet_rows == num_var_rows else None
        occupied_sites = var_rows[varying_dims].drop_duplicates()
        num_sites, num_site_value_pairs, num_parquet_rows_at_sites = dask.compute(
            occupied_sites.shape[0],
            var_rows[varying_dims + [name]].drop_duplicates().shape[0],
            parquet_rows[varying_dims].merge(occupied_sites, on=varying_dims).shape[0],
        )
        fits = (
            num_site_value_pairs == num_sites
            and num_parquet_rows_at_sites == num_var_rows
        )
        return num_sites if fits else None
