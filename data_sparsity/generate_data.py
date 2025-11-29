"""Generate synthetic data for array and tabular format comparison.

This module provides the GenerateData class for creating dummy observations
and storing them in both array (netCDF) and tabular (Parquet) formats.

This is a refactored version that uses modular validators, configurators,
generators, and output builders for improved testability and maintainability.
"""

import gc
import logging
import os
from typing import Dict, List, Tuple, Union
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster, as_completed
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import xarray as xr

import data_sparsity.utils as ds_utils
from data_sparsity.validators import (
    ParameterValidator,
    DimensionValidator,
    SparsityValidator
)
from data_sparsity.config import (
    MultiVarSparsityConfig,
    MultiVarDimensionsConfig,
    MultiVarOverlapConfig
)
from data_sparsity.generators import (
    CoordinateGenerator,
    SingleVarRecordGenerator,
    MultiVarRecordGenerator
)
from data_sparsity.output import (
    NetCDFBuilder,
    ParquetBuilder
)


class GenerateData:
    """Generate synthetic observation data in array and tabular formats.

    This class creates dummy observations with specified sparsity levels
    and stores them in both netCDF (array) and Parquet (tabular) formats
    for performance comparison studies.

    Attributes:
        num_obs: Number of observations to generate
        num_dims: Number of dimensions in the coordinate space
        ratio_dims: Tuple of relative sizes for each dimension
        sparsity: Sparsity of observation (between smin>0. and 1.)
        seed: Random seed for reproducibility
        num_vars: Number of variables in the dataset
        var_dims: Dimensions for each variable
        overlap: Overlap between variables (0-1 or 'random')
    """

    def __init__(
        self,
        num_obs: int,
        num_dims: int,
        ratio_dims: Union[int, ArrayLike],
        sparsity: Union[int, float, List, Tuple],
        seed: int,
        max_obs: int = None,
        num_vars: int = 1,
        var_dims: Union[int, List, Tuple] = None,
        overlap: Union[float, str] = 'random',
    ) -> None:
        """Initialize the data generator with validation.

        Args:
            num_obs: Number of observations to generate
            num_dims: Number of dimensions in the coordinate space
            ratio_dims: Tuple of relative sizes for each dimension
            sparsity: Sparsity of observations (scalar, 2-element, or num_vars-element)
            seed: Random seed for reproducibility
            max_obs: Maximum observations per chunk for parallel generation
            num_vars: Number of variables in the dataset (default=1)
            var_dims: Dimensions for each variable (default=num_dims for all)
            overlap: Overlap between variables (0-1 or 'random', default='random')

        Raises:
            TypeError: If arguments are not of expected types
            ValueError: If arguments fail validation checks
        """
        # Store parameters as instance variables
        self.num_obs = num_obs
        self.num_dims = num_dims
        self.ratio_dims = ratio_dims
        self.sparsity = sparsity
        self.seed = seed
        self.num_vars = num_vars
        self.var_dims = var_dims if var_dims is not None else num_dims
        self.overlap = overlap

        self._print_input_config()

        self.ratio_dims_prod = np.prod(self.ratio_dims)
        self._rng = np.random.default_rng(seed)

        # Initialize attributes set during validation
        self.var_sparsities = None
        self.var_num_obs = None
        self.var_dims_indices = None
        self.var_constant_dims = None
        self.var_constant_coord_indices = None
        self.overlap_target = None
        self.overlap_actual = None

        # Validate and configure
        self._validate_parameters()

        self._print_updated_config()

        self._multiprocessing_setup(max_obs=max_obs)

        # Storage for generated data
        self._coordinates = None
        self._observations = None
        self._record = None
        self._dataarray = None
        self._dataframe = None
        self._records = None
        self._dataset = None

    def _print_input_config(self) -> None:
        """Print input configuration."""
        print("Input configuration:")
        print(f"  Number of observations: {self.num_obs}")
        print(f"  Number of dimensions: {self.num_dims}")
        print(f"  Ratio of dimensions: {self.ratio_dims}")
        print(f"  Sparsity: {self.sparsity}")
        print(f"  Random seed: {self.seed}")
        print(f"  Number of variables: {self.num_vars}")
        print(f"  Variable dimensions: {self.var_dims}")
        print(f"  Overlap: {self.overlap}")

    def _print_updated_config(self) -> None:
        """Print updated configuration after validation."""
        print("Updated configuration after validation:")
        print(f"  Number of observations: {self.num_obs}")
        print(f"  Number of dimensions: {self.num_dims}")
        print(f"  Ratio of dimensions: {self.ratio_dims}")
        print(f"  Dimensions shape: {self.shape}")
        print(f"  Maximum available grid sites: {self.total_grid_points}")
        print(f"  Sparsity: {self.sparsity}")
        print(f"  Random seed: {self.seed}")
        print(f"  Number of variables: {self.num_vars}")
        if self.num_vars > 1:
            print(f"  Variable varying dimensions: {self.var_dims_indices}")
            print(f"  Variable constant dimensions: {self.var_constant_dims}")
        print(f"  Variable sparsities: {self.var_sparsities}")
        print(f"  Variable observations: {self.var_num_obs}")
        print(f"  Overlap: {self.overlap}")

    def _validate_parameters(self) -> None:
        """Validate initialization parameters using validator classes.

        This method delegates to specialized validator and configurator classes
        to validate and configure all parameters.
        """
        # Validate basic parameters
        ParameterValidator.validate_num_obs(self.num_obs)
        ParameterValidator.validate_num_dims(self.num_dims)
        ParameterValidator.validate_seed(self.seed)
        ParameterValidator.validate_num_vars(self.num_vars)

        # Validate and convert ratio_dims
        self.ratio_dims = ParameterValidator.validate_ratio_dims(
            self.ratio_dims, self.num_dims
        )

        # Validate sparsity type and get representative value for grid calculation
        sparsity_for_grid = ParameterValidator.validate_sparsity_type(self.sparsity)

        # Compute and validate dimensions
        nb_coords_dim1 = DimensionValidator.compute_nb_coords_dim1(
            self.num_obs, sparsity_for_grid, self.ratio_dims_prod, self.num_dims
        )
        self.nb_coords_dim1 = DimensionValidator.round_to_integer(nb_coords_dim1)

        self.nb_coords_per_dim = DimensionValidator.compute_nb_coords_per_dim(
            self.ratio_dims, self.nb_coords_dim1
        )
        DimensionValidator.validate_min_elements_per_dim(self.nb_coords_per_dim)
        self.nb_coords_per_dim = DimensionValidator.validate_integer_elements(
            self.nb_coords_per_dim
        )
        self.shape, self.total_grid_points = DimensionValidator.compute_shape_and_grid_points(
            self.nb_coords_per_dim
        )

        # Validate and adjust sparsity
        self.sparsity_zero = SparsityValidator.compute_min_sparsity(self.nb_coords_per_dim)
        sparsity_for_grid = SparsityValidator.validate_sparsity_bounds(
            sparsity_for_grid, self.sparsity_zero
        )
        self.num_obs, sparsity_for_grid = SparsityValidator.validate_num_obs_consistency(
            self.num_obs, sparsity_for_grid, self.nb_coords_per_dim
        )

        # Update sparsity if it was a scalar
        if isinstance(self.sparsity, (float, int)):
            self.sparsity = sparsity_for_grid

        # Configure multi-variable settings
        if self.num_vars > 1:
            self._configure_multi_var(sparsity_for_grid)
        else:
            self._configure_single_var(sparsity_for_grid)

    def _configure_single_var(self, sparsity: float) -> None:
        """Configure for single variable case.

        Args:
            sparsity: Validated sparsity value
        """
        self.var_sparsities = np.array([sparsity])
        self.var_num_obs = np.array([self.num_obs])
        self.var_dims_indices = [list(range(self.num_dims))]
        self.var_constant_dims = [[]]
        self.var_constant_coord_indices = {0: {}}
        self.overlap_target = None
        self.overlap_actual = None

    def _configure_multi_var(self, sparsity_for_grid: float) -> None:
        """Configure for multiple variables case.

        Args:
            sparsity_for_grid: Representative sparsity value
        """
        # Setup sparsity configuration
        self.var_sparsities, self.var_num_obs = MultiVarSparsityConfig.setup_from_parameter(
            self.sparsity, self.num_vars, self.num_obs, self.sparsity_zero, self._rng
        )

        # Setup dimension configuration
        (self.var_dims_indices,
         self.var_constant_dims,
         self.var_constant_coord_indices) = MultiVarDimensionsConfig.setup_from_parameter(
            self.var_dims, self.num_vars, self.num_dims, self.shape, self.seed
        )

        # Setup overlap configuration
        self.overlap_target = MultiVarOverlapConfig.setup_from_parameter(
            self.overlap, self.num_vars, self.var_dims_indices
        )

    def _multiprocessing_setup(self, max_obs: int = None) -> None:
        """Setup multiprocessing configuration.

        Args:
            max_obs: Maximum observations per chunk
        """
        if max_obs is None:
            self.NTASKS = 1
            return

        if max_obs <= 0:
            raise ValueError(f"max_obs must be positive, got {max_obs}")

        if self.num_obs <= max_obs:
            self.NTASKS = 1
        else:
            self.NTASKS = int(np.ceil(self.num_obs / max_obs))
            self.max_obs = max_obs
            print(f"Parallel generation: {self.NTASKS} tasks, {max_obs} obs per task")

    def _format_var_name(self, var_idx: int) -> str:
        """Format variable name.

        Args:
            var_idx: Variable index

        Returns:
            Formatted variable name
        """
        return f"var{var_idx}"

    def _generate_coordinates(self) -> None:
        """Generate coordinates for all dimensions."""
        coord_dict = {}
        coords_list = CoordinateGenerator.generate_all_coords(self.shape, self._rng)
        for i, coords in enumerate(coords_list):
            coord_dict[f"x{i}"] = coords
        self._coordinates = coord_dict

    def _generate_observations(self) -> None:
        """Generate observation values."""
        from data_sparsity.generators import ObservationGenerator
        self._observations = ObservationGenerator.generate_observations(
            self.num_obs, self._rng
        )

    def _generate_record(
        self,
        shape: list = None,
        num_obs: int = None,
        observations: np.ndarray = None,
        rng: np.random.Generator = None
    ) -> np.ndarray:
        """Generate sparse record array with observations.

        Args:
            shape: Shape of the record array. If None, uses self.shape
            num_obs: Number of observations to place. If None, uses self.num_obs
            observations: Pre-generated observation values. If None, generates them
            rng: Random number generator. If None, uses self._rng

        Returns:
            Multi-dimensional array with sparse observations
        """
        if shape is None:
            shape = self.shape
        if num_obs is None:
            num_obs = self.num_obs
        if rng is None:
            rng = self._rng

        expected_sparsity = self.var_sparsities[0] if self.num_vars == 1 else None

        record = SingleVarRecordGenerator.generate(
            shape, num_obs, rng, observations, expected_sparsity
        )

        if self.NTASKS == 1:
            self._record = record

        return record

    def _generate_multi_var_records(
        self,
        shape: list = None,
        rng: np.random.Generator = None
    ) -> dict:
        """Generate sparse record arrays for multiple variables.

        Args:
            shape: Shape of the record arrays. If None, uses self.shape
            rng: Base random number generator. If None, uses self._rng

        Returns:
            Dictionary mapping variable names to record arrays
        """
        if shape is None:
            shape = self.shape
        if rng is None:
            rng = self._rng

        records, overlap_actual = MultiVarRecordGenerator.generate(
            shape, self.overlap_target, self.num_vars, self.var_num_obs,
            self.var_dims_indices, self.var_constant_dims,
            self.var_constant_coord_indices, self.num_dims, self.seed
        )

        if self.NTASKS == 1:
            self._records = records
            self.overlap_actual = overlap_actual

        return records

    def _create_dataarray(
        self,
        record: np.ndarray = None,
        coordinates: dict = None,
        attrs: dict = None
    ) -> xr.DataArray:
        """Create xarray DataArray from generated data.

        Args:
            record: The record array to use. If None, uses self._record
            coordinates: Dictionary of coordinates. If None, uses self._coordinates
            attrs: Dictionary of attributes. If None, creates defaults

        Returns:
            xarray DataArray containing the sparse observation data
        """
        if coordinates is None:
            coordinates = self._coordinates
        if record is None:
            if hasattr(self, '_records') and self._records is not None:
                record = self._records.get('var0')
            else:
                record = self._record

        if attrs is None:
            attrs = NetCDFBuilder.create_default_attrs(
                self.num_obs, self.num_dims, self.ratio_dims,
                float(self.var_sparsities[0]), self.seed
            )

        dataarray = NetCDFBuilder.build_dataarray(record, coordinates, "record", attrs)

        if self.NTASKS == 1:
            self._dataarray = dataarray

        return dataarray

    def _create_dataset(
        self,
        records: dict = None,
        coordinates: dict = None,
        attrs: dict = None
    ) -> xr.Dataset:
        """Create xarray Dataset from multiple variable records.

        Args:
            records: Dictionary of variable records. If None, uses self._records
            coordinates: Dictionary of coordinates. If None, uses self._coordinates
            attrs: Dictionary of attributes. If None, creates defaults

        Returns:
            xarray Dataset with multiple variables
        """
        if coordinates is None:
            coordinates = self._coordinates
        if records is None:
            records = self._records

        if attrs is None:
            attrs = NetCDFBuilder.create_default_attrs(
                self.num_obs, self.num_dims, self.ratio_dims,
                float(self.var_sparsities[0]), self.seed
            )

        dataset = NetCDFBuilder.build_dataset(records, coordinates, attrs)

        if self.NTASKS == 1:
            self._dataset = dataset

        return dataset

    def _create_dataframe(
        self,
        record: np.ndarray = None,
        coordinates: dict = None,
    ) -> pd.DataFrame:
        """Create pandas DataFrame from generated data.

        Args:
            record: The record array to use. If None, uses self._record
            coordinates: Dictionary of coordinates. If None, uses self._coordinates

        Returns:
            pandas DataFrame with observation coordinates and values
        """
        if coordinates is None:
            coordinates = self._coordinates
        if record is None:
            if hasattr(self, '_records') and self._records is not None:
                record = self._records.get('var0')
            else:
                record = self._record

        dataframe = ParquetBuilder.build_single_var_dataframe(record, coordinates)

        if self.NTASKS == 1:
            self._dataframe = dataframe

        return dataframe

    def _create_multi_var_dataframe(
        self,
        records: dict = None,
        coordinates: dict = None
    ) -> pd.DataFrame:
        """Create pandas DataFrame from multiple variable records.

        Args:
            records: Dictionary of variable records. If None, uses self._records
            coordinates: Dictionary of coordinates. If None, uses self._coordinates

        Returns:
            pandas DataFrame with one column per variable
        """
        if coordinates is None:
            coordinates = self._coordinates
        if records is None:
            records = self._records

        dataframe = ParquetBuilder.build_multi_var_dataframe(
            records, coordinates, self.num_vars, self.num_dims
        )

        if self.NTASKS == 1:
            self._dataframe = dataframe

        return dataframe

    def save_to_netcdf(
        self,
        filepath: str,
        dataarray: xr.DataArray = None,
        overwrite: bool = False
    ) -> None:
        """Save DataArray/Dataset to NetCDF file.

        Args:
            filepath: Path to save file
            dataarray: DataArray or Dataset to save. If None, uses self._dataarray or self._dataset
            overwrite: Whether to overwrite existing file
        """
        if dataarray is None:
            dataarray = self._dataset if self.num_vars > 1 else self._dataarray

        NetCDFBuilder.save_to_file(dataarray, filepath, overwrite)

    def save_to_parquet(
        self,
        filepath: str,
        dataframe: Union[pd.DataFrame, dd.DataFrame] = None,
        overwrite: bool = False,
        chunk_id: int = None
    ) -> None:
        """Save DataFrame to Parquet file.

        Args:
            filepath: Path to save file
            dataframe: DataFrame to save. If None, uses self._dataframe
            overwrite: Whether to overwrite existing file
            chunk_id: Optional chunk ID for parallel generation
        """
        if dataframe is None:
            dataframe = self._dataframe

        ParquetBuilder.save_to_file(dataframe, filepath, overwrite, chunk_id)

    def generate(
        self,
        netcdf_filepath: str = None,
        parquet_filepath: str = None,
        parquet_tmp: str = None,
    ) -> Tuple[Union[xr.DataArray, xr.Dataset], pd.DataFrame]:
        """Generate all data and optionally save to files.

        Main orchestration method that executes the complete data generation
        workflow.

        Args:
            netcdf_filepath: Optional path to save NetCDF file
            parquet_filepath: Optional path to save Parquet file
            parquet_tmp: Optional temporary directory for parallel generation

        Returns:
            Tuple of (DataArray/Dataset, DataFrame) containing the generated data
        """
        if netcdf_filepath is None:
            netcdf_filepath = "./nc/test.nc"
        if parquet_filepath is None:
            parquet_filepath = "./parquet/test.parquet"
        if parquet_tmp is None:
            parquet_tmp = "./parquet_tmp/test_tmp.parquet"

        self.netcdf_filepath = netcdf_filepath
        self.parquet_filepath = parquet_filepath
        self.parquet_tmp = parquet_tmp

        ds_utils.set_up_paths(
            netcdf_filepath=self.netcdf_filepath,
            parquet_filepath=self.parquet_filepath,
            parquet_tmp=self.parquet_tmp
        )

        # Execute generation pipeline for single process
        if self.NTASKS == 1:
            self._generate_coordinates()
            self._generate_multi_var_records()

            if self.num_vars == 1:
                dataarray = self._create_dataarray()
                dataframe = self._create_dataframe()
            else:
                dataset = self._create_dataset()
                dataframe = self._create_multi_var_dataframe()
                dataarray = dataset

            if netcdf_filepath is not None:
                self.save_to_netcdf(self.netcdf_filepath, dataarray=dataarray)

            if parquet_filepath is not None:
                self.save_to_parquet(self.parquet_filepath, dataframe=dataframe)

            return dataarray, dataframe

        if self.NTASKS > 1:
            self._generate_par()
            return None, None

        raise ValueError(f"NTASKS must be positive, got {self.NTASKS}")

    def _generate_par(self) -> None:
        """Generate data in parallel using Dask.

        This method is kept from the original implementation for parallel generation.
        It will be refactored in Phase 5.
        """
        # Import the original method for now - will be refactored in Phase 5
        raise NotImplementedError(
            "Parallel generation not yet refactored. "
            "This will be implemented in Phase 5 of the refactoring."
        )
