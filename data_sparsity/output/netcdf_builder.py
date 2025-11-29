"""NetCDF/xarray output builders.

This module creates xarray DataArray and Dataset objects from generated data.
"""

from typing import Dict, List, Optional
import numpy as np
import xarray as xr


class NetCDFBuilder:
    """Builder for NetCDF/xarray output formats.
    
    This class creates xarray DataArray (single variable) or Dataset
    (multiple variables) objects from record arrays and coordinates.
    """

    @staticmethod
    def create_default_attrs(
        num_obs: int,
        num_dims: int,
        ratio_dims: np.ndarray,
        sparsity: float,
        seed: int
    ) -> Dict:
        """Create default attributes dictionary.
        
        Args:
            num_obs: Number of observations
            num_dims: Number of dimensions
            ratio_dims: Dimension ratios
            sparsity: Sparsity value
            seed: Random seed
            
        Returns:
            Dictionary of attributes
        """
        return {
            "description": "Sparse observation data",
            "num_obs": num_obs,
            "num_dims": num_dims,
            "ratio_dims": ratio_dims.tolist() if isinstance(ratio_dims, np.ndarray) else ratio_dims,
            "sparsity": float(sparsity),
            "seed": seed
        }

    @staticmethod
    def build_dataarray(
        record: np.ndarray,
        coordinates: Dict[str, np.ndarray],
        var_name: str = "record",
        attrs: Optional[Dict] = None
    ) -> xr.DataArray:
        """Build xarray DataArray from record and coordinates.
        
        Args:
            record: Record array with observations
            coordinates: Dictionary mapping dimension names to coordinate arrays
            var_name: Name for the DataArray variable
            attrs: Optional attributes dictionary
            
        Returns:
            xarray DataArray
        """
        dataarray = xr.DataArray(
            record,
            coords=coordinates,
            dims=list(coordinates.keys()),
            name=var_name,
            attrs=attrs or {}
        )
        return dataarray

    @staticmethod
    def build_dataset(
        records: Dict[str, np.ndarray],
        coordinates: Dict[str, np.ndarray],
        attrs: Optional[Dict] = None
    ) -> xr.Dataset:
        """Build xarray Dataset from multiple records.
        
        Args:
            records: Dictionary mapping variable names to record arrays
            coordinates: Dictionary mapping dimension names to coordinate arrays
            attrs: Optional attributes dictionary
            
        Returns:
            xarray Dataset
        """
        data_vars = {}
        for var_name, record in records.items():
            data_vars[var_name] = xr.DataArray(
                record,
                coords=coordinates,
                dims=list(coordinates.keys())
            )
        
        dataset = xr.Dataset(data_vars, attrs=attrs or {})
        return dataset

    @staticmethod
    def save_to_file(
        data: xr.DataArray | xr.Dataset,
        filepath: str,
        overwrite: bool = False
    ) -> None:
        """Save DataArray or Dataset to NetCDF file.
        
        Args:
            data: xarray DataArray or Dataset
            filepath: Path to save file
            overwrite: Whether to overwrite existing file
            
        Raises:
            FileExistsError: If file exists and overwrite is False
        """
        import os
        
        if os.path.exists(filepath) and not overwrite:
            raise FileExistsError(
                f"File {filepath} already exists. Set overwrite=True to replace."
            )
        
        data.to_netcdf(filepath)
        print(f"Saved to {filepath}")
