"""Parquet/pandas output builders.

This module creates pandas DataFrame objects from generated data for
Parquet output format.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import dask.dataframe as dd


class ParquetBuilder:
    """Builder for Parquet/pandas output formats.

    This class creates pandas DataFrames from record arrays and coordinates
    in both single-variable and multi-variable configurations.
    """

    @staticmethod
    def _row_axes(num_dims: int, order_dim: int) -> List[int]:
        """Axis order that puts ``order_dim`` outermost.

        Rows are emitted with the split dimension slowest-varying so that a
        chunk's rows are exactly the slice a serial run would produce for the
        same strata. Concatenating the chunks in order then reproduces the
        serial row order, without a global sort.
        """
        return [order_dim] + [dim for dim in range(num_dims) if dim != order_dim]

    @staticmethod
    def extract_non_nan_points(
        record: np.ndarray,
        coordinates: Dict[str, np.ndarray],
        order_dim: int = 0,
    ) -> pd.DataFrame:
        """Extract non-NaN points from record as DataFrame.

        Args:
            record: Record array with observations
            coordinates: Dictionary mapping dimension names to coordinate arrays
            order_dim: Dimension to vary slowest in the row order

        Returns:
            DataFrame with coordinate columns and record column
        """
        names = list(coordinates)
        axes = ParquetBuilder._row_axes(len(names), order_dim)
        # A view with order_dim first. The indices and the values below both
        # come from it, so row k of every column describes the same site.
        reordered = np.moveaxis(record, axes, range(len(axes)))
        non_nan_indices = np.where(~np.isnan(reordered))

        data_dict = {}
        for position, dim in enumerate(axes):
            data_dict[names[dim]] = coordinates[names[dim]][non_nan_indices[position]]
        # restore x0..xN column order
        data_dict = {name: data_dict[name] for name in names}
        data_dict["record"] = reordered[non_nan_indices]

        return pd.DataFrame(data_dict)

    @staticmethod
    def build_single_var_dataframe(
        record: np.ndarray,
        coordinates: Dict[str, np.ndarray],
        order_dim: int = 0,
    ) -> pd.DataFrame:
        """Build DataFrame for single variable.

        Args:
            record: Record array with observations
            coordinates: Dictionary mapping dimension names to coordinate arrays
            order_dim: Dimension to vary slowest in the row order

        Returns:
            DataFrame with one row per observation
        """
        return ParquetBuilder.extract_non_nan_points(record, coordinates, order_dim)

    @staticmethod
    def build_multi_var_dataframe(
        records: Dict[str, np.ndarray],
        coordinates: Dict[str, np.ndarray],
        num_vars: int,
        num_dims: int,
        order_dim: int = 0,
        var_constant_dims: Optional[List[List[int]]] = None,
    ) -> pd.DataFrame:
        """Build DataFrame for multiple variables, a column per variable.

        Steps:
        1. project: each variable onto its own dims, fmax over the constant
           axes (fmax == the one non-NaN value along the axis)
        2. rows: every site occupied at least once by any full-dims variable;
           then, per variable that are constant along any dimensions, look for
           an existing row with the same coordinates on the variable's own dims:
           if there is one, the value goes on that row; if there isn't, add one
           row for that value
        3. order: lexsort, order_dim slowest, NaN coordinates last; depends
           on the data alone, so serial and chunked runs agree
        4. fill: a variable fills every row matching it on its own dims, so a
           fewer-dims variable repeats across the dims it drops; NaN elsewhere

        Args:
            records: Dictionary mapping variable names to record arrays
            coordinates: Dictionary mapping dimension names to coordinate arrays
            num_vars: Number of variables
            num_dims: Number of dimensions
            order_dim: Dimension to vary slowest in the row order
            var_constant_dims: Per variable, the dims it does not vary along
                (empty list: varies along all); None: every variable varies
                along all

        Returns:
            DataFrame with one row per unique coordinate location

        """
        if var_constant_dims is None:
            var_constant_dims = [[] for _ in range(num_vars)]
        dim_names = list(coordinates)
        grid_shape = records["var0"].shape
        varying_dims, own_values = [], []
        full_dims_sites = np.zeros(grid_shape, dtype=bool)
        # 1. project each variable onto its own dims; mark full-dims sites
        for var_idx in range(num_vars):
            record = records[f"var{var_idx}"]
            constant_dims = tuple(var_constant_dims[var_idx])
            varying_dims.append([d for d in range(num_dims) if d not in constant_dims])
            own_values.append(
                np.fmax.reduce(record, axis=constant_dims) if constant_dims else record
            )
            if not constant_dims:
                # update to track sites occupied by vars that spread on all dims
                full_dims_sites |= ~np.isnan(record)

        # 2. rows: full-dims sites, then one per unmatched fewer-dims value
        row_indices = np.argwhere(full_dims_sites)  # (n_rows, num_dims) grid indices
        vars_with_constant_dims = [v for v in range(num_vars) if var_constant_dims[v]]
        for var_idx in sorted(
            vars_with_constant_dims, key=lambda v: (len(var_constant_dims[v]), v)
        ):
            dims, values = varying_dims[var_idx], own_values[var_idx]
            # rows with a real coordinate on every dim the variable varies along
            has_coords = (row_indices[:, dims] >= 0).all(axis=1)
            # own-grid cells that already have a row, as flat positions in values
            cells_with_row = np.ravel_multi_index(
                row_indices[has_coords][:, dims].T, values.shape
            )
            # cells holding a value but no row
            cells_without_row = np.setdiff1d(
                np.flatnonzero(~np.isnan(values)), cells_with_row
            )
            # one new row each: own-dims coords, -1 (NaN) on the dropped dims
            new_rows = np.full((cells_without_row.size, num_dims), -1, dtype=np.int64)
            new_rows[:, dims] = np.column_stack(
                np.unravel_index(cells_without_row, values.shape)
            )
            row_indices = np.concatenate([row_indices, new_rows])

        # 3. sort: order_dim slowest; -1 replaced by the axis length sorts last
        sort_keys = np.where(row_indices < 0, np.array(grid_shape), row_indices)
        sort_order = ParquetBuilder._row_axes(num_dims, order_dim)
        row_indices = row_indices[
            np.lexsort([sort_keys[:, d] for d in reversed(sort_order)])
        ]

        # 4. coordinate columns: index -> coordinate value, -1 -> NaN
        columns = {}
        for dim, name in enumerate(dim_names):
            dim_indices = row_indices[:, dim]  # this dim's grid index, per row
            # index -> coordinate value; -1 -> NaN (max(.., 0) avoids wrapping)
            columns[name] = np.where(
                dim_indices >= 0,
                coordinates[name][np.maximum(dim_indices, 0)],
                np.nan,
            )
        # 4. value columns: each variable fills every row matching its own dims
        for var_idx in range(num_vars):
            dims, values = varying_dims[var_idx], own_values[var_idx]
            # rows with a real coordinate on every dim the variable varies along
            has_coords = (row_indices[:, dims] >= 0).all(axis=1)
            var_column = np.full(len(row_indices), np.nan)  # NaN on other rows
            # own-dims coords -> value; rows sharing them repeat the same value
            var_column[has_coords] = values.ravel()[
                np.ravel_multi_index(row_indices[has_coords][:, dims].T, values.shape)
            ]
            columns[f"var{var_idx}"] = var_column
        return pd.DataFrame(columns)

    @staticmethod
    def cast_values(dataframe, var_encodings: list = None):
        """Cast the value columns to the type netCDF stores.

        A variable packed to int16 on the netCDF side is stored decoded here,
        as float32: packing is a netCDF device for a format without per-column
        encodings, while parquet's idiom is the natural type. Casting keeps the
        two formats holding the same numbers, so a size comparison is about the
        formats rather than about their default precisions.

        Args:
            dataframe: Frame with columns ``record`` or ``var0..varN``
            var_encodings: One VariableEncoding per variable, or None

        Returns:
            The frame, with value columns cast
        """
        if not var_encodings:
            return dataframe
        for index, encoding in enumerate(var_encodings):
            for column in (f"var{index}", "record" if index == 0 else None):
                if column and column in dataframe.columns and encoding.text_width:
                    # codes -> pyarrow-backed strings
                    dataframe[column] = pd.Series(
                        encoding.to_text(dataframe[column].to_numpy(dtype=float)),
                        index=dataframe.index,
                        dtype=encoding.pandas_dtype(),
                    )
                elif column and column in dataframe.columns:
                    dataframe[column] = dataframe[column].astype(
                        encoding.pandas_dtype()
                    )
        return dataframe

    @staticmethod
    def save_to_file(
        dataframe: pd.DataFrame | dd.DataFrame,
        filepath: str,
        overwrite: bool = False,
        chunk_id: int = None,
        write_metadata: bool = None,
        var_encodings: list = None,
    ) -> None:
        """Save DataFrame to Parquet file.

        Args:
            dataframe: pandas or dask DataFrame
            filepath: Path to save file (or directory for dask)
            overwrite: Whether to overwrite existing file
            chunk_id: Optional chunk ID for parallel generation
            write_metadata: Whether to write the `_metadata` summary files.
                Defaults to "only when this is not a chunk". Parallel workers
                pass False explicitly: they write into a shared scratch
                directory, and several processes writing `_metadata` at once
                race for the same two filenames.
            var_encodings: One VariableEncoding per variable. The value
                columns are cast to match what netCDF stores, so the two
                formats hold the same numbers in the same precision.

        Raises:
            FileExistsError: If file exists and overwrite is False
        """
        import os

        dataframe = ParquetBuilder.cast_values(dataframe, var_encodings)

        # Convert pandas to dask if needed
        if isinstance(dataframe, pd.DataFrame):
            ddf = dd.from_pandas(dataframe, npartitions=1)
        else:
            ddf = dataframe

        # Parse filepath into directory and filename
        dirpath = os.path.dirname(filepath)
        filename = os.path.basename(filepath)

        # Handle case where filepath has no directory component
        if not dirpath:
            dirpath = "."

        # Remove .parquet extension if present for directory-based storage
        if filename.endswith(".parquet"):
            filename = filename[:-8]

        if not filename:
            filename = "data"

        # Add chunk_id to filename if provided
        if chunk_id is not None:
            filename += f"_{chunk_id}"

        # Create name function for partition files
        nb_digits = len(str(ddf.npartitions))

        def name_function(partition_idx: int) -> str:
            """Generate filename for a parquet partition."""
            return f"{filename}_{partition_idx:0{nb_digits}d}.parquet"

        # Check if directory exists when not overwriting
        if os.path.exists(dirpath) and not overwrite:
            # Check if any files matching the pattern exist
            existing_files = [
                f
                for f in os.listdir(dirpath)
                if f.startswith(filename) and f.endswith(".parquet")
            ]
            if existing_files:
                raise FileExistsError(
                    f"Files matching pattern {filename}*.parquet already exist "
                    f"in {dirpath}. Set overwrite=True to replace."
                )

        # For parallel generation with chunk_id, delete only files for this chunk
        if chunk_id is not None and overwrite and os.path.exists(dirpath):
            for f in os.listdir(dirpath):
                if f.startswith(filename) and f.endswith(".parquet"):
                    os.remove(os.path.join(dirpath, f))

        # For parallel generation, don't write metadata file
        # (will be written when chunks are merged)
        if write_metadata is None:
            write_metadata = chunk_id is None

        # Save to parquet (overwrite=False to avoid deleting other chunks)
        ddf.to_parquet(
            dirpath,
            engine="pyarrow",
            name_function=name_function,
            append=False,
            overwrite=False,  # Never overwrite at dask level for chunks
            write_metadata_file=write_metadata,
        )

        print(f"Saved to {dirpath}/{filename}_*.parquet")
