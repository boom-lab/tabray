"""NetCDF/xarray output builders.

This module creates xarray DataArray and Dataset objects from generated data.
"""

from typing import Dict, List, Optional
import numpy as np
import xarray as xr

from data_sparsity.output.compression_settings import CompressionSettings


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
        density: float,
        seed: int
    ) -> Dict:
        """Create default attributes dictionary.
        
        Args:
            num_obs: Number of observations
            num_dims: Number of dimensions
            ratio_dims: Dimension ratios
            density: Density value
            seed: Random seed
            
        Returns:
            Dictionary of attributes
        """
        return {
            "description": "Sparse observation data",
            "num_obs": num_obs,
            "num_dims": num_dims,
            "ratio_dims": ratio_dims.tolist() if isinstance(ratio_dims, np.ndarray) else ratio_dims,
            "density": float(density),
            "seed": seed
        }

    # Which convention the `overlap_target` attribute uses. Written into every
    # multi-variable file so a dataset states what its own numbers mean --
    # F1 and F2 are reciprocally related and either can be assumed by a reader.
    OVERLAP_CONVENTION = (
        "F1 = |proj(S0) & proj(Si)| / |proj(S0)|: share of var0's sites that "
        "also carry the variable, measured on the dimensions the two share"
    )

    @staticmethod
    def create_multivar_attrs(
        num_vars: int,
        var_densities,
        var_num_obs,
        overlap_target,
        fixed_overlap,
        overlap_actual_f1=None,
        overlap_actual_f2=None,
    ) -> Dict:
        """Build the multi-variable attributes, identically for both paths.

        A multi-variable dataset has no single density or observation count,
        so the per-variable arrays are what actually describe it. Both paths
        must write them, or the two disagree on what the file says.

        Args:
            num_vars: Number of variables
            var_densities: Density per variable
            var_num_obs: Observation count per variable
            overlap_target: Requested overlap for var1..varN-1
            fixed_overlap: Per-variable fixed-overlap flags
            overlap_actual_f1: Achieved overlap, if measured
            overlap_actual_f2: The reverse ratio, if measured

        Returns:
            Dictionary of attributes to merge into the base set
        """
        def as_list(value):
            if value is None:
                return []
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (list, tuple)):
                return list(value)
            return [value]

        attrs = {
            "num_vars": num_vars,
            "var_densities": as_list(var_densities),
            "var_num_obs": as_list(var_num_obs),
            "overlap_target": as_list(overlap_target)
            if not isinstance(overlap_target, str) else overlap_target,
            "overlap_convention": NetCDFBuilder.OVERLAP_CONVENTION,
            "fixed_overlap": [int(bool(flag)) for flag in as_list(fixed_overlap)],
        }
        if overlap_actual_f1 is not None:
            attrs["overlap_actual_f1"] = as_list(overlap_actual_f1)
        if overlap_actual_f2 is not None:
            attrs["overlap_actual_f2"] = as_list(overlap_actual_f2)
        return attrs

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
        attrs: Optional[Dict] = None,
        var_constant_dims: Optional[List[List[int]]] = None,
        squeeze_constant_dims: bool = True,
    ) -> xr.Dataset:
        """Build xarray Dataset from multiple records.
        
        Args:
            records: Dictionary mapping variable names to record arrays
            coordinates: Dictionary mapping dimension names to coordinate arrays
            attrs: Optional attributes dictionary
            var_constant_dims: Constant dimension indices per variable
            squeeze_constant_dims: Remove constant dimensions when possible.
                Disable this for chunk files that must be concatenated with
                ``xarray.open_mfdataset``.

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
        if squeeze_constant_dims and var_constant_dims:
            dataset = NetCDFBuilder.squeeze_constant_dims(
                dataset, var_constant_dims
            )
        return dataset

    @staticmethod
    def squeeze_constant_dims(
        dataset: xr.Dataset,
        var_constant_dims: List[List[int]],
    ) -> xr.Dataset:
        """Drop the dimensions a variable is constant along.

        A variable pinned to one coordinate of a dimension carries no
        information along it, so storing it as a mostly-NaN slab inflates the
        array representation -- which is the thing being measured. Chunk files
        cannot do this (they must concatenate), so the merge step applies it
        instead.

        Args:
            dataset: Dataset whose variables may carry constant dimensions
            var_constant_dims: Constant dimension indices per variable, in the
                same order as the dataset's variables

        Returns:
            Dataset with those dimensions squeezed out where possible
        """
        dim_names = list(dataset.sizes)
        data_vars = {}
        for var_id, var_name in enumerate(dataset.data_vars):
            data_var = dataset[var_name]
            for dim_id in var_constant_dims[var_id]:
                dim_name = dim_names[dim_id]
                if dim_name not in data_var.dims:
                    continue
                data_var = data_var.dropna(dim=dim_name, how="all")
                if data_var.sizes.get(dim_name, 0) == 1:
                    data_var = data_var.squeeze(dim_name, drop=True)
            data_vars[var_name] = data_var
        return xr.Dataset(data_vars, attrs=dataset.attrs)

    @staticmethod
    def save_to_file(
        data: xr.DataArray | xr.Dataset,
        filepath: str,
        overwrite: bool = False,
        compression: CompressionSettings = None
    ) -> None:
        """Save DataArray or Dataset to NetCDF file.
        
        Args:
            data: xarray DataArray or Dataset
            filepath: Path to save file
            overwrite: Whether to overwrite existing file
            compression: Codec to apply, shared with the parquet output.
                None writes uncompressed.
            
        Raises:
            FileExistsError: If file exists and overwrite is False
        """
        import os
        
        if os.path.exists(filepath) and not overwrite:
            raise FileExistsError(
                f"File {filepath} already exists. Set overwrite=True to replace."
            )
        
        encoding = compression.netcdf_encoding(data) if compression else {}
        data.to_netcdf(filepath, encoding=encoding)
        print(f"Saved to {filepath}")
