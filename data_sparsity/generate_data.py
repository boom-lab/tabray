"""Generate synthetic data for array and tabular format comparison.

This module provides the GenerateData class for creating dummy observations
and storing them in both array (netCDF) and tabular (Parquet) formats.

This is a refactored version that uses modular validators, configurators,
generators, and output builders for improved testability and maintainability.
"""

import os
import warnings
from typing import List, Tuple, Union, Optional
import dask.dataframe as dd
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

    This class creates dummy observations with specified density levels
    and stores them in both netCDF (array) and Parquet (tabular) formats
    for performance comparison studies.

    Attributes:
        num_obs: Number of observations to generate
        num_dims: Number of dimensions in the coordinate space
        ratio_dims: Tuple of relative sizes for each dimension
        density: Density of observation (between minimum allowed density and 1.)
        seed: Random seed for reproducibility
        max_obs: Maximum observations per chunk for parallel generation
        max_workers: Maximum worker processes for parallel generation
        num_vars: Number of variables in the dataset
        var_dims: Dimensions for each variable
        overlap: Overlap between variables (0-1 or 'random')
        fixed_overlap: Whether overlap draws are shared across variables
    """

    def __init__(
        self,
        num_obs: int,
        num_dims: int,
        ratio_dims: Union[int, ArrayLike],
        sparsity: Union[int, float, List, Tuple, None] = None,
        seed: int = None,
        max_obs: int = None,
        max_workers: int = None,
        num_vars: int = 1,
        var_dims: Union[int, List, Tuple] = None,
        overlap: Union[float, str] = 'random',
        fixed_overlap: Union[bool, List[bool]] = False,
        density: Union[int, float, List, Tuple, None] = None,
    ) -> None:
        """Initialize the data generator with validation.

        Args:
            num_obs: Number of observations to generate
            num_dims: Number of dimensions in the coordinate space
            ratio_dims: Tuple of relative sizes for each dimension
            sparsity: Sparsity of observations, i.e. fraction of vacant grid
                points (scalar, 2-element, or num_vars-element).
                sparsity = 1 - density. Provide either density or sparsity,
                not both.
            density: Density of observations, i.e. fraction of occupied grid
                points (scalar, 2-element, or num_vars-element). Provide
                either density or sparsity, not both.
            seed: Random seed for reproducibility
            max_obs: Maximum observations per chunk for parallel generation
            max_workers: Maximum worker processes for parallel generation. If
                omitted, use the available CPU count.
            num_vars: Number of variables in the dataset (default=1)
            var_dims: Number of dimensions for each variable (default=num_dims for all)
            overlap: Overlap between variables (0-1 or 'random', default='random')
            fixed_overlap: Whether overlapping sites should be shared across
                variables (bool or list of bools, default=False)

        Raises:
            TypeError: If arguments are not of expected types
            ValueError: If arguments fail validation checks
        """
        # Store parameters as instance variables
        self.num_obs = num_obs
        self.num_dims = num_dims
        self.ratio_dims = ratio_dims
        self._input_sparsity = sparsity
        self._input_density = density
        self.seed = seed
        if max_workers is not None and (
            isinstance(max_workers, bool)
            or not isinstance(max_workers, int)
            or max_workers <= 0
        ):
            raise ValueError("max_workers must be a positive integer or None")
        self.max_workers = max_workers
        self.num_vars = num_vars
        self.var_dims = var_dims if var_dims is not None else num_dims
        self.overlap = overlap
        self.fixed_overlap = fixed_overlap
        self._resolve_density_input()

        self._print_input_config()

        self.ratio_dims_prod = np.prod(self.ratio_dims)
        self._rng = np.random.default_rng(seed)

        # Initialize attributes set during validation
        self.var_densities = None
        self.var_num_obs = None
        self.var_dims_indices = None
        self.var_constant_dims = None
        self.var_constant_coord_indices = None
        self.overlap_target = None
        self.overlap_actual = None
        self.overlap_actual_f2 = None

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

    def _resolve_density_input(self) -> None:
        """Resolve input density and sparsity values into internal density."""
        if self.seed is None:
            raise TypeError("seed must be provided")

        if self._input_density is not None and self._input_sparsity is not None:
            warnings.warn(
                "Both density and sparsity were provided; using density and "
                "ignoring sparsity.",
                UserWarning,
                stacklevel=2,
            )
            density = self._input_density
        elif self._input_density is not None:
            density = self._input_density
        elif self._input_sparsity is not None:
            density = self._sparsity_to_density(self._input_sparsity)
        else:
            raise TypeError("Either density or sparsity must be provided")

        self.density = density

    @staticmethod
    def _sparsity_to_density(
        sparsity: Union[int, float, List, Tuple]
    ) -> Union[float, List[float]]:
        """Convert sparsity input to density."""
        if isinstance(sparsity, (list, tuple)):
            return [1.0 - float(value) for value in sparsity]
        return 1.0 - float(sparsity)

    def _print_input_config(self) -> None:
        """Print input configuration."""
        print("Input configuration:")
        print(f"  Number of observations: {self.num_obs}")
        print(f"  Number of dimensions: {self.num_dims}")
        print(f"  Ratio of dimensions: {self.ratio_dims}")
        print(f"  Density: {self.density}")
        if self._input_sparsity is not None:
            print(f"  Sparsity input: {self._input_sparsity}")
        print(f"  Random seed: {self.seed}")
        print(f"  Number of variables: {self.num_vars}")
        print(f"  Variable dimensions: {self.var_dims}")
        print(f"  Overlap: {self.overlap}")
        print(f"  Fixed overlap: {self.fixed_overlap}")

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
        print(f"  Variable densities: {self.var_densities}")
        print(f"  Variable observations: {self.var_num_obs}")
        print(f"  Overlap: {self.overlap}")
        print(f"  Fixed overlap: {self.fixed_overlap}")

    def _validate_parameters(self) -> None:
        """Validate initialization parameters.

        Checks that:
        * input parameters values are admissible
        * all dimensions have at least two elements (one element does not make
          sense, as we can drop that dimension and reduce the system's size)
        * all dimensions have a natural number of elements (no floats)
        * density is larger than the minimum theoretical value and at most 1
        * input number of observations is consistent with input number of
          dimensions and density

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

        # Validate density type and get representative value for grid calculation
        self.density = ParameterValidator.validate_density_refvar(self.density)
        density_for_grid = ParameterValidator.validate_density_type(self.density)

        # Compute and validate dimensions
        nb_coords_dim1 = DimensionValidator.compute_nb_coords_dim1(
            self.num_obs, density_for_grid, self.ratio_dims_prod, self.num_dims
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

        # Provisional stratum dimension; narrowed below once the per-variable
        # dimensions are known (see _choose_split_dim).
        self.dim_split = int(np.argmax(self.nb_coords_per_dim))

        # Check that density is larger than minimum allowed for this set of parameters
        self.density_zero = SparsityValidator.compute_min_density(self.nb_coords_per_dim)
        density_for_grid = SparsityValidator.validate_density_bounds(
            density_for_grid, self.density_zero
        )

        # Check that num_obs is consistent with density and dimensions size
        self.num_obs, density_for_grid = SparsityValidator.validate_num_obs_consistency(
            self.num_obs, density_for_grid, self.nb_coords_per_dim
        )

        # Update density if it was a scalar
        if isinstance(self.density, (float, int)):
            self.density = density_for_grid

        # Configure multi-variable settings
        if self.num_vars > 1:
            self._configure_multi_var(density_for_grid)
        else:
            self._configure_single_var(density_for_grid)

        self.dim_split = self._choose_split_dim()

    def _choose_split_dim(self) -> int:
        """Pick the dimension the grid is stratified along.

        The largest dimension, restricted to those **every** variable varies
        along. Overlap is measured on the dimensions two variables share, so if
        a variable were constant along the split dimension its projection would
        collapse across strata: the reference footprint seen from one stratum
        would differ from the global one, and a per-stratum overlap target
        would not add up to the requested global overlap.

        Excluding such dimensions can mean splitting a shorter axis, which
        lowers the maximum number of chunks (see S6).

        Returns:
            Index of the dimension to stratify along
        """
        shared = set(range(self.num_dims))
        for varying in self.var_dims_indices:
            shared &= set(varying)
        if not shared:
            raise ValueError(
                "No dimension is shared by every variable, so the grid cannot "
                "be stratified consistently. Give at least one dimension to all "
                "variables via var_dims."
            )
        candidates = sorted(shared)
        sizes = [self.nb_coords_per_dim[d] for d in candidates]
        chosen = candidates[int(np.argmax(sizes))]
        largest = int(np.argmax(self.nb_coords_per_dim))
        if chosen != largest:
            print(
                f"  Stratifying along dimension {chosen} (size "
                f"{self.nb_coords_per_dim[chosen]}) rather than the largest "
                f"dimension {largest} (size {self.nb_coords_per_dim[largest]}), "
                f"which is not shared by every variable."
            )
        return int(chosen)

    def _configure_single_var(self, density: float) -> None:
        """Configure for single variable case.

        Args:
            density: Validated density value
        """
        self.var_densities = np.array([density])
        self.var_num_obs = np.array([self.num_obs])
        self.var_dims_indices = [list(range(self.num_dims))]
        self.var_constant_dims = [[]]
        self.var_constant_coord_indices = {0: {}}
        self.overlap_target = None
        self.overlap_actual = None
        self.overlap_actual_f2 = None

    def _configure_multi_var(self, density_for_grid: float) -> None:
        """Configure for multiple variables case.

        Args:
            density_for_grid: Representative density value
        """
        # Setup density values for each variable
        self.var_densities, self.var_num_obs = MultiVarSparsityConfig.setup_from_parameter(
            self.density, self.num_vars, self.num_obs, self.density_zero, self._rng
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
        (
            self.overlap_target,
            adjusted_obs,
            self.fixed_overlap,
        ) = MultiVarOverlapConfig.setup_from_parameter(
            self.overlap,
            self.fixed_overlap,
            self.num_vars,
            self.shape,
            self.var_num_obs,
            self.var_dims_indices
        )
        
        # Update observation counts if they were adjusted
        if not np.array_equal(self.var_num_obs, adjusted_obs):
            self.var_num_obs = adjusted_obs

    def _multiprocessing_setup(self, max_obs: int = None) -> None:
        """Setup multiprocessing configuration.

        Args:
            max_obs: Maximum observations per chunk
            
        Raises:
            ValueError: If max_obs is incompatible with LHS requirements
        """
        if max_obs is None:
            self.NTASKS = 1
            return

        if max_obs <= 0:
            raise ValueError(f"max_obs must be positive, got {max_obs}")

        if self.num_obs <= max_obs:
            self.NTASKS = 1
        else:
            # Note: LHS compatibility is handled via RNG state advancement
            # Each chunk generates its observations independently with properly
            # advanced RNG state to maintain serial/parallel equivalence
            
            self.NTASKS = int(np.ceil(self.num_obs / max_obs))
            self.max_obs = max_obs
            
            # Group the grid's strata into chunks. The split dimension was
            # already fixed during validation (see self.dim_split).
            max_dim = self.dim_split
            max_dim_size = self.nb_coords_per_dim[max_dim]
            
            if max_dim_size < self.NTASKS:
                raise ValueError(
                    f"Dimension has size {max_dim_size} but {self.NTASKS} "
                    f"chunks should be generated?"
                )
            
            # Divide the largest dimension into chunks
            Neach_section, extras = divmod(max_dim_size, self.NTASKS)
            Neach_section = int(Neach_section)
            section_sizes = ([0] + extras * [Neach_section + 1] + 
                           (self.NTASKS - extras) * [Neach_section])
            div_points = np.array(section_sizes, dtype=int).cumsum()
            
            self.max_dim_size = max_dim_size
            self.section_sizes = section_sizes[1:]
            self.div_points = div_points
            
            print(f"Parallel generation: {self.NTASKS} tasks, {max_obs} obs per task")
            print(f"  Dataset split along dimension {self.dim_split}")
            print(f"  Block sizes along it: {self.section_sizes}")

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
            self.var_constant_coord_indices, self.num_dims, self.seed,
            dim_split=self.dim_split,
            fixed_overlap=self.fixed_overlap
        )

        if self.NTASKS == 1:
            self._records = records
            self.overlap_actual = overlap_actual
            if self.num_vars > 1:
                from data_sparsity.generators import OverlapCalculator
                report = OverlapCalculator.compute_overlap_report(
                    records, self.num_vars, self.num_dims, self.var_dims_indices
                )
                self.overlap_actual = report["f1"]
                self.overlap_actual_f2 = report["f2"]

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
                float(self.var_densities[0]), self.seed
            )
        attrs.update({
            "overlap_target": self.overlap_target if self.num_vars > 1 else "random",
            "fixed_overlap": (
                [int(value) for value in self.fixed_overlap]
                if self.num_vars > 1 else []
            ),
        })

        dataarray = NetCDFBuilder.build_dataarray(record, coordinates, "record", attrs)

        if self.NTASKS == 1:
            self._dataarray = dataarray

        return dataarray

    def _create_dataset(
        self,
        records: dict = None,
        coordinates: dict = None,
        attrs: dict = None,
        var_constant_dims: Optional[List[List[int]]] = None,
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
        if var_constant_dims is None:
            var_constant_dims = self.var_constant_dims
            
        if attrs is None:
            # num_obs and density describe the reference variable, the same
            # quantities the chunk files record; the per-variable arrays below
            # carry everything else.
            attrs = NetCDFBuilder.create_default_attrs(
                int(self.var_num_obs[0]), self.num_dims, self.ratio_dims,
                float(self.var_num_obs[0]) / float(self.total_grid_points),
                self.seed
            )
        attrs.update(NetCDFBuilder.create_multivar_attrs(
            num_vars=self.num_vars,
            var_densities=self.var_densities,
            var_num_obs=self.var_num_obs,
            overlap_target=self.overlap_target,
            fixed_overlap=self.fixed_overlap,
            overlap_actual_f1=self.overlap_actual,
            overlap_actual_f2=getattr(self, "overlap_actual_f2", None),
        ))

        dataset = NetCDFBuilder.build_dataset(records, coordinates, attrs, var_constant_dims)

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

        dataframe = ParquetBuilder.build_single_var_dataframe(
            record, coordinates, order_dim=self.dim_split
        )

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
            records, coordinates, self.num_vars, self.num_dims,
            order_dim=self.dim_split
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
        merge_nc: bool = False,
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
            merge_nc: Parallel mode only. If True, concatenate the chunk files
                into one netCDF and delete them. Off by default: large datasets
                are routinely served as many files (daily observation files,
                for instance), and merging doubles the I/O and the peak disk.

        Returns:
            Tuple of (DataArray/Dataset, DataFrame) containing the generated data
        """

        # Execute generation pipeline for single process
        if self.NTASKS == 1:
            # Generate per-dimension RNGs using shared utility
            # Serial mode: chunk_id=None (no advancement needed)
            dim_rngs = ChunkUtils.generate_rngs(
                seed=self.seed,
                num_dims=self.num_dims
            )
            
            self._coordinates = CoordinateGenerator.generate_all_coords(
                self.shape,
                rng=None,  # Not used when dim_rngs provided
                dim_ranges=None,
                dim_rngs=dim_rngs
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

            nc_path, pq_path, pq_path_tmp = (
                PathManager.setup_output_paths(
                    netcdf_filepath=netcdf_filepath,
                    parquet_filepath=parquet_filepath,
                    parquet_tmp=parquet_tmp,
                    overwrite=True
                )
            )
            self.save_to_netcdf(nc_path, dataarray=dataarray)
            self.save_to_parquet(pq_path, dataframe=dataframe)

            return dataarray, dataframe

        if self.NTASKS > 1:
            # Setup paths for parallel generation
            self.netcdf_filepath, self.parquet_filepath, self.parquet_tmp = (
                PathManager.setup_output_paths(
                    netcdf_filepath=netcdf_filepath,
                    parquet_filepath=parquet_filepath,
                    parquet_tmp=parquet_tmp,
                    overwrite=True,
                )
            )
            self._generate_par()
            if merge_nc:
                self._merge_netcdf_files()
            return None, None

        raise ValueError(f"NTASKS must be positive, got {self.NTASKS}")

    def _generate_par(self) -> None:
        """Generate sparse record array with observations using parallel processing.

        Submit as many dataset generation tasks as number of blocks needed.
        Supports both single-variable and multi-variable datasets.
        
        Uses ProcessPoolExecutor for simpler, more robust parallelization without
        external dependencies.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        from data_sparsity.workers import generate_chunk
        
        # Ensure output directories exist
        nc_dir = os.path.dirname(self.netcdf_filepath)
        if nc_dir and not os.path.exists(nc_dir):
            os.makedirs(nc_dir, exist_ok=True)
        
        parquet_dir = os.path.dirname(self.parquet_filepath)
        if parquet_dir and not os.path.exists(parquet_dir):
            os.makedirs(parquet_dir, exist_ok=True)
            
        # Scratch chunks go in their own directory, not alongside the real
        # output. Clear any leftovers from an interrupted run, but only if the
        # directory holds nothing this code did not write.
        PathManager.remove_scratch_dir(self.parquet_tmp)
        os.makedirs(self.parquet_tmp, exist_ok=True)

        if self.num_vars == 1:
            density_for_parallel = (
                max(self.density)
                if isinstance(self.density, (list, tuple, np.ndarray))
                else self.density
            )

            mp_obs, density_new, per_chunk_obs = ChunkUtils.get_observations_per_chunk(
                self.num_obs,
                self.shape,
                self.max_dim_size,
                self.section_sizes,
                density_for_parallel
            )
            self.density = density_new
            self.num_obs = mp_obs

            # Prepare arguments for all chunks
            chunk_args = []
            for chunk_id, chunk_obs in zip(range(self.NTASKS), per_chunk_obs):
                args = {
                    'chunk_id': chunk_id,
                    'obs_in_chunk': chunk_obs,
                    'seed': self.seed,
                    'shape': self.shape,
                    'density': self.density,
                    'num_vars': self.num_vars,
                    'num_dims': self.num_dims,
                    'ratio_dims': self.ratio_dims,
                    'num_obs': self.num_obs,
                    'var_densities': self.var_densities if self.num_vars > 1 else None,
                    'var_num_obs': self.var_num_obs if self.num_vars > 1 else None,
                    'var_dims_indices': self.var_dims_indices if self.num_vars > 1 else None,
                    'var_constant_dims': self.var_constant_dims if self.num_vars > 1 else None,
                    'var_constant_coord_indices': (
                        self.var_constant_coord_indices if self.num_vars > 1 else None
                    ),
                    'overlap_target': self.overlap_target if self.num_vars > 1 else 0.0,
                    'fixed_overlap': self.fixed_overlap if self.num_vars > 1 else False,
                    'dim_split': self.dim_split,
                    'max_dim_size': self.max_dim_size,
                    'div_points': self.div_points,
                    'section_sizes': self.section_sizes,
                    'netcdf_filepath': self.netcdf_filepath,
                    'parquet_tmp': self.parquet_tmp,
                    'ntasks': self.NTASKS,
                    'num_obs_global': self.num_obs,
                }
                chunk_args.append(args)
        else:
            chunk_var_num_obs = ChunkUtils.get_multi_var_observations_per_chunk(
                self.var_num_obs,
                self.max_dim_size,
                self.section_sizes,
            )

            chunk_args = []
            for chunk_id, (_chunk_size, var_obs_chunk) in enumerate(
                zip(self.section_sizes, chunk_var_num_obs)
            ):
                args = {
                    'chunk_id': chunk_id,
                    'obs_in_chunk': int(np.sum(var_obs_chunk)),
                    'seed': self.seed,
                    'shape': self.shape,
                    'density': self.density,
                    'num_vars': self.num_vars,
                    'num_dims': self.num_dims,
                    'ratio_dims': self.ratio_dims,
                    'num_obs': int(np.sum(var_obs_chunk)),
                    'var_densities': self.var_densities,
                    'var_num_obs': self.var_num_obs,   # GLOBAL: strata apportion
                    'var_dims_indices': self.var_dims_indices,
                    'var_constant_dims': self.var_constant_dims,
                    'var_constant_coord_indices': self.var_constant_coord_indices,
                    'overlap_target': self.overlap_target,
                    'fixed_overlap': self.fixed_overlap,
                    'dim_split': self.dim_split,
                    'max_dim_size': self.max_dim_size,
                    'div_points': self.div_points,
                    'section_sizes': self.section_sizes,
                    'netcdf_filepath': self.netcdf_filepath,
                    'parquet_tmp': self.parquet_tmp,
                    'ntasks': self.NTASKS,
                    'num_obs_global': self.num_obs,
                }
                chunk_args.append(args)

        # Default to available CPUs while allowing callers to cap memory use.
        available_cpus = os.cpu_count() or 1
        max_workers = min(
            self.NTASKS,
            self.max_workers if self.max_workers is not None else available_cpus,
        )
        print(f"Starting parallel generation with {max_workers} workers for {self.NTASKS} chunks")

        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = [executor.submit(generate_chunk, **args) for args in chunk_args]
            
            tot_completed = 0
            tot_obs = 0
            for future in as_completed(futures):
                chunk_id, obs_num, chunk_path = future.result()
                tot_completed += 1
                tot_obs += obs_num
                print(
                    f"Completed {tot_completed} of {self.NTASKS} chunks "
                    f"(completed chunk #{chunk_id})"
                )

        print(f"Total obs stored to disk: {tot_obs}.")

        # Consolidate parquet files
        self._consolidate_parquet_files()

    def _merge_netcdf_files(self) -> None:
        """Concatenate the chunk netCDF files into one, then delete them.

        Opened lazily with dask so a larger-than-memory dataset streams rather
        than being materialised. Chunk files keep every dimension because they
        have to concatenate; the constant dimensions are squeezed out here, so
        the merged file matches what a serial run writes.
        """
        import glob

        base = self.netcdf_filepath[:-3]
        chunk_files = sorted(glob.glob(f"{base}_*.nc"))
        if not chunk_files:
            raise RuntimeError(f"No netCDF chunk files found matching {base}_*.nc")

        print(f"Merging {len(chunk_files)} netCDF chunk files...")
        split_dim = f"x{self.dim_split}"
        merged = xr.open_mfdataset(
            chunk_files, combine="nested", concat_dim=split_dim,
            data_vars="minimal", coords="minimal", compat="override",
        )
        if self.num_vars > 1:
            merged = NetCDFBuilder.squeeze_constant_dims(
                merged, self.var_constant_dims
            )

        # Chunk-local bookkeeping does not describe the merged dataset.
        merged.attrs.pop("chunk_id", None)
        merged.attrs["description"] = merged.attrs.get(
            "description", ""
        ).replace(" (chunk)", "")
        merged.attrs["num_obs"] = int(self.var_num_obs[0])
        merged.attrs["density"] = float(
            self.var_num_obs[0] / self.total_grid_points
        )
        if self.num_vars > 1:
            merged.attrs["var_num_obs"] = [int(n) for n in self.var_num_obs]

        merged.to_netcdf(self.netcdf_filepath)
        merged.close()
        for chunk_file in chunk_files:
            os.remove(chunk_file)
        print(
            f"Merged netCDF written to {self.netcdf_filepath}; "
            f"removed {len(chunk_files)} chunk files"
        )

    def _consolidate_parquet_files(self) -> None:
        """Consolidate temporary parquet files into single output.
        
        Uses Dask for memory-efficient consolidation of potentially larger-than-memory
        datasets. This is critical for the parallel workflow's primary use case:
        generating datasets that exceed available memory.
        
        The consolidation reads all temporary parquet chunks lazily using Dask,
        repartitions for optimal I/O, and writes the consolidated output.
        """
        import glob
        
        tmp_dir = self.parquet_tmp
        tmp_pattern = os.path.join(tmp_dir, "chunk_*.parquet")
        tmp_files = sorted(glob.glob(tmp_pattern))
        
        if not tmp_files:
            raise RuntimeError(f"No temporary parquet files found in {tmp_dir}")
        
        print(f"Consolidating {len(tmp_files)} parquet chunk files...")
        
        # Use Dask to read all chunks lazily (memory-efficient for large datasets)
        ddf = dd.read_parquet(tmp_pattern)
        
        # Repartition for optimal write performance (300MB partitions is a good default)
        ddf = ddf.repartition(partition_size="300MB")
        
        print(f"Dask DataFrame has {ddf.npartitions} partitions")
        
        # Write consolidated file using ParquetBuilder (which handles dask DataFrames)
        ParquetBuilder.save_to_file(ddf, self.parquet_filepath, overwrite=True)
        
        # Cleanup: the scratch directory goes once the merge has succeeded
        print(f"Cleaning up {len(tmp_files)} temporary files...")
        for f in tmp_files:
            os.remove(f)
        PathManager.remove_scratch_dir(tmp_dir)
        print(f"Consolidated parquet file saved to {self.parquet_filepath}")
