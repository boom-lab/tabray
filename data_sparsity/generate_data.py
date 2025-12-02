"""Generate synthetic data for array and tabular format comparison.

This module provides the GenerateData class for creating dummy observations
and storing them in both array (netCDF) and tabular (Parquet) formats.

This is a refactored version that uses modular validators, configurators,
generators, and output builders for improved testability and maintainability.
"""

import gc
import logging
import os
from typing import Dict, List, Tuple, Union, Optional
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster, as_completed
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import xarray as xr

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
    MultiVarRecordGenerator
)
from data_sparsity.output import (
    NetCDFBuilder,
    ParquetBuilder,
    PathManager
)
from data_sparsity.utils import (
    ChunkUtils,
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
        max_obs: Maximum observations per chunk for parallel generation
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
            var_dims: Number of dimensions for each variable (default=num_dims for all)
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
        print(f"  Random seed: {self.seed}")
        print(f"  Number of variables: {self.num_vars}")
        if self.num_vars > 1:
            print(f"  Variable varying dimensions: {self.var_dims_indices}")
            print(f"  Variable constant dimensions: {self.var_constant_dims}")
        print(f"  Variable sparsities: {self.var_sparsities}")
        print(f"  Variable observations: {self.var_num_obs}")
        print(f"  Overlap: {self.overlap}")

    def _validate_parameters(self) -> None:
        """Validate initialization parameters.

        Checks that:
        * input parameters values are admissible
        * all dimensions have at least two elements (one element does not make
          sense, as we can drop that dimension and reduce the system's size)
        * all dimensions have a natural number of elements (no floats)
        * sparsity is larger than the minimum theoretical value and smaller
          than 1
        * input number of observations is consistent with input number of
          dimensions and sparsity

        Some checks are hard checks (i.e. an error is raised if the check fails),
        others are soft (i.e. the expected values is enforced instead of raising
        an error). The latter is done when the check fails likely due to
        rounding. Input and updated configuration are printed to screen to make
        user aware of changes.

        This method delegates to specialized validator and configurator classes
        to validate and configure all parameters.

        Raises:
            TypeError: If parameters are not of expected types
            ValueError: If parameters fail validation checks
        """
        # Validate basic parameters
        ParameterValidator.validate_num_obs(self.num_obs)
        ParameterValidator.validate_num_dims(self.num_dims)
        ParameterValidator.validate_seed(self.seed)
        ParameterValidator.validate_num_vars(self.num_vars)

        # Validate that reference variable occupies all dimensions
        self.var_dims = ParameterValidator.validate_var_dims(
            self.var_dims, self.num_vars, self.num_dims
        )

        # Validate and convert ratio_dims
        self.ratio_dims = ParameterValidator.validate_ratio_dims(
            self.ratio_dims, self.num_dims
        )

        # Validate sparsity type and get representative value for grid calculation
        self.sparsity = ParameterValidator.validate_sparsity_refvar(self.sparsity)
        sparsity_for_grid = ParameterValidator.validate_sparsity_type(self.sparsity)

        # Compute and validate dimensions
        nb_coords_dim1 = DimensionValidator.compute_nb_coords_dim1(
            self.num_obs, sparsity_for_grid, self.ratio_dims_prod, self.num_dims
        )
        self.nb_coords_dim1 = DimensionValidator.round_to_integer(nb_coords_dim1)

        # Check that all dimensions have at least one element
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

        # Check that sparsity is larger than minimum allowed for this set of parameters
        self.sparsity_zero = SparsityValidator.compute_min_sparsity(self.nb_coords_per_dim)
        sparsity_for_grid = SparsityValidator.validate_sparsity_bounds(
            sparsity_for_grid, self.sparsity_zero
        )

        # Check that num_obs is consistent with sparsity and dimensions size
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
        # Setup sparsity values for each variable
        self.var_sparsities, self.var_num_obs = MultiVarSparsityConfig.setup_from_parameter(
            self.sparsity, self.num_vars, self.num_obs, self.sparsity_zero, self._rng
        )

        # Setup dimensions: the grid is defined over num_dims dimensions, but
        # each variable is measured at var_dims <= num_dims, the other
        # dimensions are set to a constant value
        (self.var_dims_indices,
         self.var_constant_dims,
         self.var_constant_coord_indices) = MultiVarDimensionsConfig.setup_from_parameter(
            self.var_dims, self.num_vars, self.num_dims, self.shape, self.seed
        )

        # Setup overlap configuration and adjust observations if needed
        self.overlap_target, adjusted_obs = MultiVarOverlapConfig.setup_from_parameter(
            self.overlap, self.num_vars, self.shape, self.var_num_obs, self.var_dims_indices
        )
        
        # Update observation counts if they were adjusted
        if not np.array_equal(self.var_num_obs, adjusted_obs):
            self.var_num_obs = adjusted_obs

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
            
            # Set up dimension splitting for parallel processing
            # Split along the largest dimension
            max_dim = np.argmax(self.nb_coords_per_dim)
            max_dim_size = self.nb_coords_per_dim[max_dim]
            
            if max_dim_size < self.NTASKS:
                raise ValueError(
                    f"Dimension has size {max_dim_size} but {self.NTASKS} "
                    f"blocks should be generated?"
                )
            
            # Divide the largest dimension into chunks
            Neach_section, extras = divmod(max_dim_size, self.NTASKS)
            Neach_section = int(Neach_section)
            section_sizes = ([0] + extras * [Neach_section + 1] + 
                           (self.NTASKS - extras) * [Neach_section])
            div_points = np.array(section_sizes, dtype=int).cumsum()
            
            self.dim_split = max_dim
            self.max_dim_size = max_dim_size
            self.section_sizes = section_sizes[1:]
            self.div_points = div_points
            
            print(f"Parallel generation: {self.NTASKS} tasks, {max_obs} obs per task")
            print(f"  Dataset split along dimension {self.dim_split}")
            print(f"  Block sizes along it: {self.section_sizes}")

    def _format_var_name(self, var_idx: int) -> str:
        """Format variable name.

        Args:
            var_idx: Variable index

        Returns:
            Formatted variable name
        """
        return f"var{var_idx}"

    def _generate_coordinates(
        self,
        shape: List[int],
        rng: np.random.Generator,
        dim_ranges: Optional[Dict[int, Tuple[float, float]]] = None,
        dim_rngs: Optional[Dict[int, np.random.Generator]] = None
    ) -> dict:
        """Generate coordinates for all dimensions.

        Creates coordinate arrays for each dimension with values sorted
        in ascending order within specified ranges.

        Args:
            shape: shape tuple/list
            rng: random number generator
            dim_ranges: Optional dict mapping dimension indices to (low, high) tuples
                       for custom coordinate ranges. If None, uses [0, 1) for all dims.
            dim_rngs: Optional dict mapping dimension indices to specific RNGs to use.
                     If provided, these override the default rng for those dimensions.

        Returns:
            Dictionary mapping dimension names to coordinate arrays
        """

        coordinates = CoordinateGenerator.generate_all_coords(
            shape,
            rng,
            dim_ranges,
            dim_rngs
        )
        return coordinates

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
        
        This method now uses MultiVarRecordGenerator with num_vars=1 to
        maintain consistency with multi-variable generation and eliminate
        code duplication.

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

        # Use MultiVarRecordGenerator with num_vars=1 for consistency
        # For single-var, all dimensions vary (no constant dims)
        records, overlap_actual = MultiVarRecordGenerator.generate(
            shape=shape,
            overlap='random',  # Irrelevant for single variable
            num_vars=1,
            var_num_obs=np.array([num_obs]),
            var_dims_indices=[list(range(len(shape)))],  # All dims vary
            var_constant_dims=[[]],  # No constant dims
            var_constant_coord_indices={},  # No constant coords
            num_dims=len(shape),
            seed=self.seed
        )

        # Extract the single record from the dictionary
        record = records['var0']

        if self.NTASKS == 1:
            self._record = record

        return record

    def _generate_multi_var_records(
        self,
        shape: list = None,
        rng: np.random.Generator = None
    ) -> dict:
        """Generate sparse record arrays for multiple variables.

        Creates a multi-dimensional array for each variable with the specified shape,
        then randomly selects positions and assigns observation values to those points
        for each variable. Overlap is controlled by using shared and separate RNGs.

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
        workflow:
        1. Generate coordinates for each dimension
        2. Generate random observation values
        3. Create sparse record array
        4. Build xarray DataArray/Dataset
        5. Build pandas DataFrame
        6. Optionally save to NetCDF and/or Parquet files

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

        self.netcdf_filepath, self.parquet_filepath, self.parquet_tmp = (
            PathManager.setup_output_paths(
                netcdf_filepath=netcdf_filepath,
                parquet_filepath=parquet_filepath,
                parquet_tmp=parquet_tmp,
                overwrite=True
            )
        )

        # Execute generation pipeline for single process
        if self.NTASKS == 1:
            #self._coordinates = self._generate_coordinates(self.shape, self._rng)
            self._coordinates = CoordinateGenerator.generate_all_coords(
                self.shape,
                self._rng,
            )

            # Use unified multi-variable workflow for both single and multiple variables
            self._generate_multi_var_records()

            if self.num_vars == 1:
                # For single variable, return DataArray and use standard DataFrame
                dataarray = self._create_dataarray()
                dataframe = self._create_dataframe()
            else:
                # For multiple variables, return Dataset with multi-var DataFrame
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
        """Generate sparse record array with observations using parallel processing.

        Submit as many dataset generation tasks as number of blocks needed.
        Supports both single-variable and multi-variable datasets.
        """
        # Clean up any existing temporary files from previous runs
        import os
        import shutil
        
        # Ensure output directories exist
        nc_dir = os.path.dirname(self.netcdf_filepath)
        if nc_dir and not os.path.exists(nc_dir):
            os.makedirs(nc_dir, exist_ok=True)
        
        parquet_dir = os.path.dirname(self.parquet_filepath)
        if parquet_dir and not os.path.exists(parquet_dir):
            os.makedirs(parquet_dir, exist_ok=True)
            
        tmp_dir = os.path.dirname(self.parquet_tmp)
        
        # Remove the entire temporary directory if it exists
        if os.path.exists(tmp_dir):
            try:
                shutil.rmtree(tmp_dir)
                print(f"Deleted temporary directory: {tmp_dir}")
            except Exception as e:
                print(f"Warning: Could not remove {tmp_dir}: {e}")
        
        # Recreate the temporary directory
        if tmp_dir:
            os.makedirs(tmp_dir, exist_ok=True)
        
        cluster = LocalCluster(n_workers=4, threads_per_worker=1, processes=True)
        client = Client(cluster)
        print("Dask dashboard:", client.dashboard_link)

        mp_obs, sparsity_new, per_chunk_obs = ChunkUtils.get_observations_per_chunk(
            self.num_obs,
            self.shape,
            self.max_dim_size,
            self.section_sizes,
            self.sparsity
        )
        self.sparsity = sparsity_new
        self.num_obs = mp_obs

        # Submit one task per seed
        futures = [
            client.submit(self._generate_record_par, chunk_id, chunk_obs)
            for chunk_id, chunk_obs in zip(range(self.NTASKS), per_chunk_obs)
        ]
        tot_completed = 0
        tot_obs = 0
        for f in as_completed(futures):
            chunk_id, obs_num = f.result()
            tot_completed += 1
            tot_obs += obs_num
            print(
                f"Completed {tot_completed} of {self.NTASKS} chunks "
                f"(completed chunk #{chunk_id})"
            )

        print(f"Total obs stored to disk: {tot_obs}.")
        client.close()
        cluster.close()

        # Read all chunks from temporary directory
        import os
        tmp_dir = os.path.dirname(self.parquet_tmp)
        ddf = dd.read_parquet(tmp_dir)
        ddf = ddf.repartition(partition_size="300MB")
        self.save_to_parquet(
            self.parquet_filepath,
            ddf,
            overwrite=True
        )

    def _generate_record_par(self, chunk_id: int, obs_in_chunk: int) -> Tuple[int, int]:
        """Generate sparse record array for a single chunk in parallel processing.

        This method generates a chunk of the full array by:
        1. Setting up chunk-specific and global random generators
        2. Generating coordinates (using global RNG for shared dims, local for split dim)
        3. Generating the sparse record array(s) for this chunk (single or multi-variable)
        4. Creating and saving DataArray/Dataset and DataFrame representations

        Supports both single-variable and multi-variable datasets. For multi-variable
        datasets with overlap control, the overlap is computed per-chunk based on the
        global overlap target.

        Args:
            chunk_id: Identifier for this chunk
            obs_in_chunk: Number of observations to generate in this chunk (for single-var)
                         or observations for the reference variable (for multi-var)

        Returns:
            Tuple of (chunk_id, total number of observations stored across all variables)
        """
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(process)d %(levelname)s %(message)s',
            filename=f'dask_worker_{chunk_id}.log',
        )
        logging.debug("")
        logging.debug("######------ NEW CHUNK ------######")

        # Generate two distinct generators:
        # global_rng: identical across tasks, used for coordinates along non-split dimensions
        # task_rng: unique per task, used for split dimension coordinates and observations
        global_rng, task_rng = ChunkUtils.generate_rngs(self.seed, chunk_id)

        # Determine chunk dimensions and range along split dimension
        task_range = (self.div_points[chunk_id], self.div_points[chunk_id + 1])
        task_size = self.section_sizes[chunk_id]
        task_shape = ChunkUtils.update_chunk_shape(self.shape, self.dim_split, task_size)
        
        logging.debug("task_range: %s", task_range)
        logging.debug("task_size: %s", task_size)
        logging.debug("task_shape: %s", task_shape)

        total_chunk_points = ChunkUtils.validate_chunk_points(task_shape)
        logging.debug("total_chunk_points: %s", total_chunk_points)

        # Build dimension ranges and RNGs for coordinate generation
        # Non-split dimensions use [0, 1) with global_rng
        # Split dimension uses normalized chunk range with task_rng
        dim_ranges = ChunkUtils.generate_split_dimension_range(
            self.dim_split,
            task_range,
            self.max_dim_size
        )
        
        dim_rngs = ChunkUtils.assign_rngs_to_dimensions(
            self.dim_split,
            task_shape,
            global_rng,
            task_rng
        )

        # Generate coordinates with optional dimension-specific ranges and RNGs
        coordinates = CoordinateGenerator.generate_all_coords(
            task_shape,
            global_rng,
            dim_ranges,
            dim_rngs
        )
        
        logging.debug("obs in chunk: %s", obs_in_chunk)
        logging.debug("total chunk points: %s", total_chunk_points)

        # Generate records for single or multiple variables
        if self.num_vars == 1:
            # Single variable mode: use the original single-record logic
            record = self._generate_record(
                shape=task_shape,
                num_obs=obs_in_chunk,
                observations=None,  # Will be generated inside _generate_record
                rng=task_rng
            )

            logging.debug("chunk id: %s", chunk_id)
            logging.debug("record.shape: %s", record.shape)
            logging.debug("num obs in chunk: %s", obs_in_chunk)
            logging.debug("non-nans in chunk: %s", np.sum(~np.isnan(record)))
            logging.debug("dims: %s", list(coordinates.keys()))
            logging.debug("coords: %s", coordinates)

            # Create DataArray with chunk-specific attributes
            chunk_attrs = NetCDFBuilder.create_default_attrs(
                self.num_obs, self.num_dims, self.ratio_dims, self.sparsity, self.seed
            )
            chunk_attrs["chunk_id"] = chunk_id
            chunk_attrs["description"] = "Sparse observation data (chunk)"
            
            dataarray = NetCDFBuilder.build_dataarray(
                record, coordinates, attrs=chunk_attrs
            )

            # Save to NetCDF
            nb_digits = len(str(self.NTASKS))
            fpath = f"{self.netcdf_filepath[:-3]}_{chunk_id:0{nb_digits}d}.nc"
            self.save_to_netcdf(fpath, dataarray=dataarray, overwrite=False)
            del dataarray
            gc.collect()

            # Create and save DataFrame (keep as pandas, save_to_parquet will convert)
            dataframe = ParquetBuilder.build_single_var_dataframe(record, coordinates)

            self.save_to_parquet(
                self.parquet_tmp,
                dataframe,
                overwrite=False,
                chunk_id=chunk_id
            )

            total_obs = np.sum(~np.isnan(record))

        else:
            # Multi-variable mode: generate all variables for this chunk
            records, overlap_actual = MultiVarRecordGenerator.generate(
                shape, self.overlap_target, self.num_vars, self.var_num_obs,
                self.var_dims_indices, self.var_constant_dims,
                self.var_constant_coord_indices, self.num_dims, self.seed,
                chunk_id, self.max_dim_size, self.dim_split
            )

            logging.debug("chunk id: %s", chunk_id)
            total_obs = 0
            for var_idx in range(self.num_vars):
                var_name = self._format_var_name(var_idx)
                var_obs = np.sum(~np.isnan(records[var_name]))
                total_obs += var_obs
                logging.debug("%s obs in chunk: %s", var_name, var_obs)

            # Create Dataset with chunk-specific attributes
            chunk_attrs = NetCDFBuilder.create_default_attrs(
                self.num_obs, self.num_dims, self.ratio_dims,
                float(self.var_sparsities[0]), self.seed
            )
            chunk_attrs.update({
                "chunk_id": chunk_id,
                "description": "Multi-variable sparse observation data (chunk)",
                "num_vars": self.num_vars,
                "var_sparsities": self.var_sparsities.tolist(),
                "var_num_obs": self.var_num_obs.tolist(),
                "overlap_target": self.overlap_target if isinstance(
                    self.overlap_target, str
                ) else float(self.overlap_target)
            })
            
            dataset = NetCDFBuilder.build_dataset(
                records, coordinates, attrs=chunk_attrs
            )

            # Save to NetCDF
            nb_digits = len(str(self.NTASKS))
            fpath = f"{self.netcdf_filepath[:-3]}_{chunk_id:0{nb_digits}d}.nc"
            self.save_to_netcdf(fpath, dataarray=dataset, overwrite=False)
            del dataset
            gc.collect()

            # Create and save DataFrame (keep as pandas, save_to_parquet will convert)
            dataframe = ParquetBuilder.build_multi_var_dataframe(
                records, coordinates, self.num_vars, self.num_dims
            )

            self.save_to_parquet(
                self.parquet_tmp,
                dataframe,
                overwrite=False,
                chunk_id=chunk_id
            )

        return chunk_id, total_obs
