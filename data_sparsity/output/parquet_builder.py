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
    def extract_non_nan_points(
        record: np.ndarray,
        coordinates: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """Extract non-NaN points from record as DataFrame.
        
        Args:
            record: Record array with observations
            coordinates: Dictionary mapping dimension names to coordinate arrays
            
        Returns:
            DataFrame with coordinate columns and record column
        """
        non_nan_mask = ~np.isnan(record)
        non_nan_indices = np.where(non_nan_mask)
        
        data_dict = {}
        
        coord_names = list(coordinates.keys())
        coord_arrays = list(coordinates.values())
        
        for i, (name, coords) in enumerate(zip(coord_names, coord_arrays)):
            data_dict[name] = coords[non_nan_indices[i]]
        
        data_dict["record"] = record[non_nan_mask]
        
        return pd.DataFrame(data_dict)

    @staticmethod
    def build_single_var_dataframe(
        record: np.ndarray,
        coordinates: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """Build DataFrame for single variable.
        
        Args:
            record: Record array with observations
            coordinates: Dictionary mapping dimension names to coordinate arrays
            
        Returns:
            DataFrame with one row per observation
        """
        return ParquetBuilder.extract_non_nan_points(record, coordinates)

    @staticmethod
    def build_multi_var_dataframe(
        records: Dict[str, np.ndarray],
        coordinates: Dict[str, np.ndarray],
        num_vars: int,
        num_dims: int
    ) -> pd.DataFrame:
        """Build DataFrame for multiple variables.
        
        Creates a DataFrame where each row is a unique coordinate with
        separate columns for each variable's observations.
        
        Args:
            records: Dictionary mapping variable names to record arrays
            coordinates: Dictionary mapping dimension names to coordinate arrays
            num_vars: Number of variables
            num_dims: Number of dimensions
            
        Returns:
            DataFrame with one row per unique coordinate location
        """
        coord_to_obs = {}
        
        for var_idx in range(num_vars):
            var_name = f"var{var_idx}"
            record = records[var_name]
            
            non_nan_mask = ~np.isnan(record)
            non_nan_indices = np.where(non_nan_mask)
            
            for obs_idx in range(len(non_nan_indices[0])):
                full_coords = []
                for dim_idx in range(num_dims):
                    coord_name = f"x{dim_idx}"
                    coord_val = coordinates[coord_name][
                        non_nan_indices[dim_idx][obs_idx]
                    ]
                    full_coords.append(coord_val)
                
                coord_tuple = tuple(full_coords)
                
                if coord_tuple not in coord_to_obs:
                    coord_to_obs[coord_tuple] = {}
                
                coord_to_obs[coord_tuple][var_name] = record[non_nan_mask][obs_idx]
        
        rows = []
        for coord_tuple, var_obs in coord_to_obs.items():
            row = {}
            for dim_idx in range(num_dims):
                coord_name = f"x{dim_idx}"
                row[coord_name] = coord_tuple[dim_idx]
            
            for var_idx in range(num_vars):
                var_name = f"var{var_idx}"
                row[var_name] = var_obs.get(var_name, pd.NA)
            
            rows.append(row)
        
        return pd.DataFrame(rows)

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
