"""Parquet/pandas output builders.

This module creates pandas DataFrame objects from generated data for
Parquet output format.
"""

from typing import Dict, List
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
        order_dim: int = 0
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
        non_nan_indices = np.where(~np.isnan(np.moveaxis(record, axes, range(len(axes)))))

        data_dict = {}
        for position, dim in enumerate(axes):
            data_dict[names[dim]] = coordinates[names[dim]][non_nan_indices[position]]
        # restore x0..xN column order
        data_dict = {name: data_dict[name] for name in names}
        data_dict["record"] = np.moveaxis(record, axes, range(len(axes)))[non_nan_indices]

        return pd.DataFrame(data_dict)

    @staticmethod
    def build_single_var_dataframe(
        record: np.ndarray,
        coordinates: Dict[str, np.ndarray],
        order_dim: int = 0
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
        order_dim: int = 0
    ) -> pd.DataFrame:
        """Build DataFrame for multiple variables.

        One row per occupied coordinate, with a column per variable and NaN
        where a variable has no value there.

        Rows are ordered by coordinate, ``order_dim`` slowest. The previous
        implementation emitted them in the order coordinates were discovered
        while looping over variables, which is not a function of the data, so
        serial and chunked runs produced different orderings of the same rows.
        It also re-masked the whole record array once per row, costing
        O(num_obs x grid_points); this is vectorised.

        Args:
            records: Dictionary mapping variable names to record arrays
            coordinates: Dictionary mapping dimension names to coordinate arrays
            num_vars: Number of variables
            num_dims: Number of dimensions
            order_dim: Dimension to vary slowest in the row order

        Returns:
            DataFrame with one row per unique coordinate location
        """
        names = list(coordinates)
        axes = ParquetBuilder._row_axes(num_dims, order_dim)
        reordered_shape = tuple(len(coordinates[names[dim]]) for dim in axes)

        flats, values = [], []
        for var_idx in range(num_vars):
            record = np.moveaxis(records[f"var{var_idx}"], axes, range(num_dims))
            mask = ~np.isnan(record)
            indices = np.where(mask)
            flats.append(np.ravel_multi_index(indices, reordered_shape)
                         if indices[0].size else np.empty(0, dtype=np.int64))
            values.append(record[mask])

        occupied = np.unique(np.concatenate(flats)) if flats else np.empty(0, dtype=np.int64)
        unravelled = np.unravel_index(occupied, reordered_shape)

        columns = {}
        for position, dim in enumerate(axes):
            columns[names[dim]] = coordinates[names[dim]][unravelled[position]]
        columns = {name: columns[name] for name in names}

        for var_idx in range(num_vars):
            column = np.full(occupied.size, np.nan)
            if flats[var_idx].size:
                column[np.searchsorted(occupied, flats[var_idx])] = values[var_idx]
            columns[f"var{var_idx}"] = column

        return pd.DataFrame(columns)

    @staticmethod
    def save_to_file(
        dataframe: pd.DataFrame | dd.DataFrame,
        filepath: str,
        overwrite: bool = False,
        chunk_id: int = None
    ) -> None:
        """Save DataFrame to Parquet file.
        
        Args:
            dataframe: pandas or dask DataFrame
            filepath: Path to save file (or directory for dask)
            overwrite: Whether to overwrite existing file
            chunk_id: Optional chunk ID for parallel generation
            
        Raises:
            FileExistsError: If file exists and overwrite is False
        """
        import os
        
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
            dirpath = '.'
        
        # Remove .parquet extension if present for directory-based storage
        if filename.endswith('.parquet'):
            filename = filename[:-8]
        
        if not filename:
            filename = 'data'
        
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
                f for f in os.listdir(dirpath) 
                if f.startswith(filename) and f.endswith('.parquet')
            ]
            if existing_files:
                raise FileExistsError(
                    f"Files matching pattern {filename}*.parquet already exist "
                    f"in {dirpath}. Set overwrite=True to replace."
                )
        
        # For parallel generation with chunk_id, delete only files for this chunk
        if chunk_id is not None and overwrite and os.path.exists(dirpath):
            for f in os.listdir(dirpath):
                if f.startswith(filename) and f.endswith('.parquet'):
                    os.remove(os.path.join(dirpath, f))
        
        # For parallel generation, don't write metadata file
        # (will be written when chunks are merged)
        write_metadata = (chunk_id is None)
        
        # Save to parquet (overwrite=False to avoid deleting other chunks)
        ddf.to_parquet(
            dirpath,
            engine="pyarrow",
            name_function=name_function,
            append=False,
            overwrite=False,  # Never overwrite at dask level for chunks
            write_metadata_file=write_metadata
        )
        
        print(f"Saved to {dirpath}/{filename}_*.parquet")
