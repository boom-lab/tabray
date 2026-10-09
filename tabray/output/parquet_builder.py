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
        """Build DataFrame for multiple variables.

        One row per occupied coordinate, with a column per variable and NaN
        where a variable has no value there. A variable on fewer dims is
        repeated across the dims it drops (``_repeat_fewer_dims``).

        Rows are ordered by coordinate, ``order_dim`` slowest. The order is a
        function of the data alone, so serial and chunked runs produce the
        same rows in the same order.

        Args:
            records: Dictionary mapping variable names to record arrays
            coordinates: Dictionary mapping dimension names to coordinate arrays
            num_vars: Number of variables
            num_dims: Number of dimensions
            order_dim: Dimension to vary slowest in the row order
            var_constant_dims: Dims each variable is pinned on (None: none)

        Returns:
            DataFrame with one row per unique coordinate location
        """
        if var_constant_dims and any(var_constant_dims):
            return ParquetBuilder._repeat_fewer_dims(
                records, coordinates, num_vars, var_constant_dims, order_dim
            )
        names = list(coordinates)
        axes = ParquetBuilder._row_axes(num_dims, order_dim)
        reordered_shape = tuple(len(coordinates[names[dim]]) for dim in axes)

        flat_indices, values = [], []
        for var_idx in range(num_vars):
            record = np.moveaxis(records[f"var{var_idx}"], axes, range(num_dims))
            mask = ~np.isnan(record)
            indices = np.where(mask)
            flat_indices.append(
                np.ravel_multi_index(indices, reordered_shape)
                if indices[0].size
                else np.empty(0, dtype=np.int64)
            )
            values.append(record[mask])

        occupied = np.unique(np.concatenate(flat_indices))
        unravelled = np.unravel_index(occupied, reordered_shape)

        # Coordinate columns. unravelled[position] holds every row's index along
        # axis `position` of the reordered grid, which is original dimension
        # axes[position]; indexing that axis's labels turns indices into
        # coordinate values. The dict is filled in reordered order (split
        # dimension first), then rebuilt so the columns read x0..xN.
        columns = {}
        for position, dim in enumerate(axes):
            columns[names[dim]] = coordinates[names[dim]][unravelled[position]]
        columns = {name: columns[name] for name in names}

        for var_idx in range(num_vars):
            column = np.full(occupied.size, np.nan)
            if flat_indices[var_idx].size:
                rows = np.searchsorted(occupied, flat_indices[var_idx])
                column[rows] = values[var_idx]
            columns[f"var{var_idx}"] = column

        return pd.DataFrame(columns)

    @staticmethod
    def _repeat_fewer_dims(
        records: Dict[str, np.ndarray],
        coordinates: Dict[str, np.ndarray],
        num_vars: int,
        var_constant_dims: List[List[int]],
        order_dim: int,
    ) -> pd.DataFrame:
        """Rows when some variable is on fewer dims; its value holds along the rest.

        - rows: cells held by a full-dims variable, then, per fewer-dims
          variable (most dims first), one row per own-dims cell no row matches,
          coordinate NaN (index -1) on the dims it drops
        - a variable fills every row matching it on its own dims
        - own-dims value = fmax over the pinned axes: one non-NaN cell at most
        - order: lexsort with order_dim slowest, NaN coordinates last
        """
        names = list(coordinates)
        shape = records["var0"].shape
        own_dims, projected = [], []
        held = np.zeros(shape, dtype=bool)
        for var_idx in range(num_vars):
            record = records[f"var{var_idx}"]
            pinned = tuple(var_constant_dims[var_idx])
            own_dims.append([d for d in range(len(shape)) if d not in pinned])
            projected.append(np.fmax.reduce(record, axis=pinned) if pinned else record)
            if not pinned:
                held |= ~np.isnan(record)

        rows = np.argwhere(held)
        fewer = [v for v in range(num_vars) if var_constant_dims[v]]
        for var_idx in sorted(fewer, key=lambda v: (len(var_constant_dims[v]), v)):
            own, proj = own_dims[var_idx], projected[var_idx]
            valid = (rows[:, own] >= 0).all(axis=1)
            matched = np.ravel_multi_index(rows[valid][:, own].T, proj.shape)
            cells = np.setdiff1d(np.flatnonzero(~np.isnan(proj)), matched)
            new = np.full((cells.size, len(shape)), -1, dtype=np.int64)
            new[:, own] = np.column_stack(np.unravel_index(cells, proj.shape))
            rows = np.concatenate([rows, new])

        keys = np.where(rows < 0, np.array(shape), rows)
        axes = ParquetBuilder._row_axes(len(shape), order_dim)
        rows = rows[np.lexsort([keys[:, d] for d in reversed(axes)])]

        columns = {}
        for dim, name in enumerate(names):
            index = rows[:, dim]
            columns[name] = np.where(
                index >= 0, coordinates[name][np.maximum(index, 0)], np.nan
            )
        for var_idx in range(num_vars):
            own, proj = own_dims[var_idx], projected[var_idx]
            valid = (rows[:, own] >= 0).all(axis=1)
            column = np.full(len(rows), np.nan)
            column[valid] = proj.ravel()[
                np.ravel_multi_index(rows[valid][:, own].T, proj.shape)
            ]
            columns[f"var{var_idx}"] = column
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
                if column and column in dataframe.columns:
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
