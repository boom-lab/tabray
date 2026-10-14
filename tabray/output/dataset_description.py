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
        """Parquet stores no axes and no variable dims, so both come from the rows.

        - grid: the coordinate values that appear in any row
        - a variable's dims, the dims it is defined on (``_infer_dims``)
        - occupied sites: distinct rows over the variable's dims
        """
        frame = dd.read_parquet(path)
        names = [c for c in frame.columns if c not in coords]
        grid = {c: int(frame[c].nunique().compute()) for c in coords}
        var_dims, seen, occupied = {}, {}, {}
        for name in names:
            held = frame[frame[name].notnull()][coords + [name]]
            seen[name] = {c: int(held[c].nunique().compute()) for c in coords}
            var_dims[name], assumed_constant, ambiguous_dims = (
                DatasetDescription._infer_dims(frame, held, coords, grid)
            )
            DatasetDescription._report_assumptions(
                name, assumed_constant, ambiguous_dims
            )
            # no dims: one site, like a scalar
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
          site and is on every row at each of its sites (``_consistent``);
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
            frame: Every row of the parquet
            held: The variable's rows, columns ``coords + [name]``
            coords: Coordinate columns
            grid: Number of distinct values per coord; a coord of length 1 is
                kept, as netCDF keeps it

        Returns:
            (dims, assumed_constant, ambiguous_dims): the dims left out by
            assumption, and the kept dims the rows would also fit without
        """
        name = held.columns[-1]
        has_nan_coord = dask.compute(*[held[c].isnull().any() for c in coords])
        not_nan_dims = [c for c, has_nan in zip(coords, has_nan_coord) if not has_nan]
        # distinct rows on the dims no row leaves NaN; more of them than sites
        # on fewer dims means some site's value is on several rows
        num_held_rows, num_values, num_sites_all_dims = dask.compute(
            held.shape[0],
            held[name].nunique(),
            held[not_nan_dims].drop_duplicates().shape[0],
        )
        if num_held_rows == 0:  # empty: nothing to infer, density is 0 anyway
            return not_nan_dims, [], []
        candidates = [c for c in not_nan_dims if grid[c] > 1]
        fitting_sets, repeated_sets = [], []
        frontier = [()]
        while frontier:
            larger = []
            for subset in frontier:
                start = candidates.index(subset[-1]) + 1 if subset else 0
                for dim in candidates[start:]:
                    constant_along = subset + (dim,)
                    if any(
                        constant_along[:i] + constant_along[i + 1 :] not in fitting_sets
                        for i in range(len(constant_along))
                        if len(constant_along) > 1
                    ):
                        continue  # a subset did not fit
                    candidate_dims = [
                        c for c in not_nan_dims if c not in constant_along
                    ]
                    if not DatasetDescription._consistent(
                        frame, held, candidate_dims, num_held_rows
                    ):
                        continue
                    fitting_sets.append(constant_along)
                    larger.append(constant_along)
                    num_sites = (
                        int(held[candidate_dims].drop_duplicates().shape[0].compute())
                        if candidate_dims
                        else 1
                    )
                    if num_sites_all_dims > num_sites and num_values > 1:
                        repeated_sets.append(constant_along)
            frontier = larger
        assumed_constant = max(repeated_sets, key=len, default=())
        dims = [c for c in not_nan_dims if c not in assumed_constant]
        ambiguous_dims = [
            c
            for c in dims
            if tuple(sorted(assumed_constant + (c,), key=candidates.index))
            in fitting_sets
        ]
        return dims, list(assumed_constant), ambiguous_dims

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
    def _consistent(
        frame: dd.DataFrame,
        held: dd.DataFrame,
        candidate_dims: List[str],
        num_held_rows: int,
    ) -> bool:
        """Whether the rows fit a variable on ``candidate_dims`` only.

        A site is one combination of coordinates on ``candidate_dims``.

        - one value per site
        - on every row of the parquet at a site it holds: a variable missing
          from such a row (held by another variable) varies along the coords
          that row differs in
        """
        name = held.columns[-1]
        if not candidate_dims:  # a scalar: one value, on every row
            num_values, num_rows = dask.compute(held[name].nunique(), frame.shape[0])
            return num_values == 1 and num_rows == num_held_rows
        sites_held = held[candidate_dims].drop_duplicates()
        num_sites, num_site_values, num_rows_at_sites = dask.compute(
            sites_held.shape[0],
            held[candidate_dims + [name]].drop_duplicates().shape[0],
            frame[candidate_dims].merge(sites_held, on=candidate_dims).shape[0],
        )
        return num_site_values == num_sites and num_rows_at_sites == num_held_rows
