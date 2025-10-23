"""Generate synthetic data for array and tabular format comparison.

This module provides the GenerateData class for creating dummy observations
and storing them in both array (netCDF) and tabular (Parquet) formats.
"""

from typing import Tuple, Union
import dask.array as da
import dask.dataframe as dd
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import xarray as xr

import data_sparsity.utils as ds_utils

class GenerateData:
    """Generate synthetic observation data in array and tabular formats.

    This class creates dummy observations with specified sparsity levels
    and stores them in both netCDF (array) and Parquet (tabular) formats
    for performance comparison studies.

    Attributes:
        num_obs: Number of observations to generate
        num_dims: Number of dimensions in the coordinate space
        ratio_dims: Tuple of relative sizes for each dimension; the first element
                    is always 1, the following elements express the relative size
                    of the other dimensions to the first one
        sparsity: Sparsity of observation (between smin>0. and 1.)
        seed: Random seed for reproducibility

    """

    def __init__(
        self,
        num_obs: int,
        num_dims: int,
        ratio_dims: Union[int,ArrayLike],
        sparsity: Union[int,float],
        seed: int
    ) -> None:
        """Initialize the data generator with validation.

        Args:
            see attributes above

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

        print("Input configuration:")
        print(f"  Number of observations: {self.num_obs}")
        print(f"  Number of dimensions: {self.num_dims}")
        print(f"  Ratio of dimensions: {self.ratio_dims}")
        print(f"  Sparsity: {self.sparsity}")
        print(f"  Random seed: {self.seed}")

        self.ratio_dims_prod = np.prod(self.ratio_dims)

        # Validate all parameters
        self._validate_parameters()
        print("Updated configuration after validation:")
        print(f"  Number of observations: {self.num_obs}")
        print(f"  Number of dimensions: {self.num_dims}")
        print(f"  Ratio of dimensions: {self.ratio_dims}")
        print(f"  Dimensions sizes: {self.nb_coords_per_dim}")
        print(f"  Sparsity: {self.sparsity}")
        print(f"  Random seed: {self.seed}")


        # Initialize random number generator with seed
        self._rng = np.random.default_rng(seed)

        # Storage for generated data
        self._coordinates = None
        self._observations = None
        self._record = None
        self._dataarray = None
        self._dataframe = None

    def _validate_parameters(self) -> None:
        """Validate initialization parameters.

        Checks that:
        * input parameters values are admissible
        * all dimensions have at least two elements (one element does not make
          sense, as we can drop that dimension and reduce the system's size)
        * all dimensions have a natural number of elements (no floats)
        * sparsity is larger than the minimum theoritcal value and smaller
          than 1
        * input number of observations is consistent with input number of
          dimensions and sparsity

        Some checks are hard checks (i.e. an error is raised if the check fails),
        others are soft (i.e. the expected values is enforced instead of raising
        an error). The latter is done when the check fail is likely due to
        rounding. Input and updated configuration are printed to screen to make
        user aware of changes.

        Raises:
            TypeError: If parameters are not of expected types
            ValueError: If parameters fail validation checks

        """

        # Check that num_obs is int larger than 0
        if not isinstance(self.num_obs, int):
            raise TypeError(f"num_obs must be an integer, got {type(self.num_obs)}")
        if self.num_obs <= 0:
            raise ValueError(f"num_obs must be positive, got {self.num_obs}")

        # Check that sparsity is positive and <= 1
        if not isinstance(self.sparsity, (float, int)):
            raise TypeError(
                f"sparsity must be a number, got {type(self.sparsity)}"
            )
        if not 0.0 <= self.sparsity <= 1.0:
            raise ValueError(
                f"sparsity must be between positive and less than or equal to 1.0, got {self.sparsity}"
            )

        # Check num_dims and ratio_dims consistency
        if not isinstance(self.num_dims, int):
            raise TypeError(
                f"num_dims must be an int, got {type(self.num_dims)}"
            )
        if self.ratio_dims!=1:
            if self.num_dims != len(self.ratio_dims):
                raise ValueError(
                    "num_dims must be equivalent to the number of elements in ratio_dims "
                    f"(if this is not 1), got {self.num_dims} and {len(self.ratio_dims)}, respectively."
                )

        # Check ratio_dims type and convert to np array
        if isinstance(self.ratio_dims, int):
            if self.ratio_dims == 1:
                # impose same size along all dimensions
                self.ratio_dims = self.num_dims*[1]
        elif not isinstance(self.ratio_dims, (list, tuple, np.ndarray)):
            raise TypeError(
                f"ratio_dims must be a tuple, list, or numpy array, got {type(self.ratio_dims)}"
            )
        self.ratio_dims = np.asarray(self.ratio_dims)

        # Check that first dimension in coordinate space has at least one element
        if not isinstance(self.num_dims, int):
            raise TypeError(f"num_dims must be an integer, got {type(self.num_dims)}")
        if self.num_dims <= 0:
            raise ValueError(f"num_dims must be positive, got {self.num_dims}")
        base = self.num_obs / (self.sparsity*self.ratio_dims_prod)
        self.nb_coords_dim1 = np.power( base, 1/self.num_dims )
        if self.nb_coords_dim1 < 1:
            raise ValueError(f"number of elements for dimension 1 must be larger than 1, got {self.nb_coords_dim1}")

        # Enforce nb_coords_dim1 to be an integer
        print(
            f"Number of elements in the first dimension is {self.nb_coords_dim1}, "
            f"rounding to closest integer: {np.rint(self.nb_coords_dim1).astype(int)}"
        )
        self.nb_coords_dim1 = np.rint(self.nb_coords_dim1).astype(int)

        # Check that all other dimensions have at least one element
        self.nb_coords_per_dim = self.ratio_dims*self.nb_coords_dim1
        fewer_than_one = self.nb_coords_per_dim<1
        if np.any(fewer_than_one):
            bad_idxs = np.flatnonzero(fewer_than_one)
            msgs = [
                f"Error: dimension {idx} must have at least one element, got "
                f"{self.nb_coords_per_dim[idx]}."
                for idx in bad_idxs
            ]
            for m in msgs:
                print(m)
            raise ValueError("One or more dimensions do not contain at least one element.")

        # Check that all dimensions contain an integer number of elements
        not_integers = np.logical_not(
            np.isclose(
                self.nb_coords_per_dim,
                np.rint(self.nb_coords_per_dim)
                )
            )
        if np.any(not_integers):
            bad_idxs = np.flatnonzero(not_integers)
            msgs = [
                f"Error: dimension {idx} does not have an integer number of elements, got {self.nb_coords_per_dim[idx]}"
                for idx in bad_idxs
            ]
            # print or include messages in the exception
            for m in msgs:
                print(m)
            raise ValueError("One or more dimensions contain a decimal number elements.")
        else:
            print("All dimensions contain approximately a natural number of elements, casting and/or rounding them:")
            print(f"  Old number of elements: {self.nb_coords_per_dim}")
            self.nb_coords_per_dim = np.rint(self.nb_coords_per_dim).astype(int)
            print(f"  New number of elements: {self.nb_coords_per_dim}")

        # Check that sparsity is larger than minimum allowed for this set of parameters
        self.sparsity_zero = 1/np.min(self.nb_coords_per_dim)
        print(f"Minimum sparsity value for the current set of dimensions: {self.sparsity_zero}")
        if self.sparsity == 0.:
            self.sparsity = self.sparsity_zero
            print(f"Input sparsity is zero, imposing minimum value: {self.sparsity_zero}")
        elif self.sparsity < self.sparsity_zero:
            raise ValueError(
                f"Provided sparsity value of {self.sparsity} is lower than minimum value"
                f" of {self.sparsity_zero}. If you want to impose the minimum value possible"
                ", set sparsity to 0. as input."
            )

        # Check that num_obs is consistent with sparsity and dimensions size, else update it
        num_obs_exp = self.sparsity*np.prod(self.nb_coords_per_dim)
        if num_obs_exp != self.num_obs:
            print(
                f"Input number of observations num_obs ({self.num_obs}) does not match the "
                f"number of observations num_obs_exp {num_obs_exp} expected from values of "
                "sparsity and the number of elements per dimension. This can happen due to"
                " rounding operations and is not necessarily an issue, so we are enforcing "
                "num_obs to match num_obs_exp and rounding it."
            )
            self.num_obs = np.rint(num_obs_exp).astype(int)
            print(f"New number of observations is {self.num_obs}")
            self.sparsity = self.num_obs/np.prod(self.nb_coords_per_dim)
            print(f"Actual sparsity is now {self.sparsity}")
            if self.sparsity < self.sparsity_zero or self.sparsity > 1:
                raise ValueError(
                    f"Sparsity value {self.sparsity} out of bounds [{self.sparsity_zero},1]"
                )

        # Check that seed is a non-negative integer
        if not isinstance(self.seed, int):
            raise TypeError(f"seed must be an integer, got {type(self.seed)}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

    def _generate_coordinates(self) -> dict:
        """Generate random coordinates for each dimension.

        Creates coordinate arrays for each dimension with values in [0, 1).
        The number of points in each dimension is determined by ratio_dims
        and constrained by the total number of grid points needed for the
        specified sparsity level.

        Returns:
            Dictionary mapping dimension names to dask coordinate arrays
        """

        # Generate random coordinate arrays for each dimension as dask arrays
        coordinates = {}
        for idx, n_coords in enumerate(self.nb_coords_per_dim):
            dim_name = f"x{idx}"
            # Generate numpy array first, then convert to dask array
            numpy_coords = np.sort(
                self._rng.uniform(0, 1, size=n_coords)
            )
            # Convert to dask array with single chunk (1D arrays are small)
            coordinates[dim_name] = da.from_array(numpy_coords, chunks=-1)

        self._coordinates = coordinates
        return coordinates

    def _generate_observations(self) -> da.Array:
        """Generate random observation values.

        Creates num_obs random values in the range [0, 1] as a dask array.

        Returns:
            Dask array of random observation values
        """
        # Generate numpy array first, then convert to dask array
        numpy_observations = self._rng.uniform(0, 1, size=self.num_obs)
        # Convert to dask array with single chunk (1D observation array)
        observations = da.from_array(numpy_observations, chunks=-1)

        self._observations = observations
        return observations

    def _generate_record(self) -> da.Array:
        """Generate sparse record array with observations.

        Creates a multi-dimensional dask array with the shape defined by
        coordinates, then randomly selects num_obs points and assigns
        the observation values to those points. All other points are NaN.

        The dask array is chunked to optimize for access patterns where
        x0 is accessed first, then x1, then x2, etc.

        Returns:
            Multi-dimensional dask array with sparse observations

        Raises:
            RuntimeError: If coordinates have not been generated yet
        """
        if self._coordinates is None:
            raise RuntimeError(
                "Coordinates must be generated before record. "
                "Call _generate_coordinates() first."
            )

        if self._observations is None:
            raise RuntimeError(
                "Observations must be generated before record. "
                "Call _generate_observations() first."
            )

        # Get shape of the full grid
        shape = tuple(len(coords) for coords in self._coordinates.values())
        total_points_in_grid = np.prod(shape)
        s_estim = self.num_obs/total_points_in_grid
        if s_estim != self.sparsity:
            raise ValueError(
                f"Sparcity {s_estim} determined from number "
                f"of coordinates differs from sparsity {self.sparsity} "
                "determined during paramaters validation step."
            )

        # Initialize record with NaN as a numpy array first
        record_np = np.full(shape, np.nan)

        # Generate random indices for observation placement
        # Flatten the multi-dimensional index space
        flat_indices = self._rng.choice(
            total_points_in_grid,
            size=self.num_obs,
            replace=False
        )

        # Convert flat indices to multi-dimensional indices:
        # it reconstructs where the index idx of the flattened 1D array with
        # elements np.prod(shape) is in the multidimensional array of dimensions
        # shape[0], shape[1], ... shape[n]
        # multi_indices contains a total of len(flat_indices) 1D arrays with each
        # containing len(shape) elements, and corresponds to the
        # num_obs=len(flat_indices) number of points where observations are known
        multi_indices = np.unravel_index(flat_indices, shape)

        # Assign observation values to selected points
        # Need to compute observations to assign them
        record_np[multi_indices] = self._observations.compute()

        # Define chunks: prioritize x0, then x1, then x2, etc.
        # Use automatic chunking but ensure chunks are reasonable
        # For optimal access pattern, make x0 chunks smallest, x1 larger, etc.
        chunks = []
        for i, dim_size in enumerate(shape):
            # Make chunks progressively larger for later dimensions
            # This optimizes for x0-first access patterns
            chunk_size = max(1, min(dim_size, 100 * (2 ** i)))
            chunks.append(chunk_size)
        chunks = tuple(chunks)

        # Convert to dask array with specified chunking
        record = da.from_array(record_np, chunks=chunks)

        self._record = record
        return record

    def _create_dataarray(self) -> xr.DataArray:
        """Create xarray DataArray from generated data.

        Constructs an xarray DataArray with the generated coordinates
        and record data.

        Returns:
            xarray DataArray containing the sparse observation data

        Raises:
            RuntimeError: If required data has not been generated yet
        """
        if self._coordinates is None:
            raise RuntimeError("Coordinates must be generated first")

        if self._record is None:
            raise RuntimeError("Record must be generated first")

        # Create DataArray with coordinates
        dataarray = xr.DataArray(
            self._record,
            coords=self._coordinates,
            dims=list(self._coordinates.keys()),
            name="record",
            attrs={
                "description": "Sparse observation data",
                "num_obs": self.num_obs,
                "num_dims": self.num_dims,
                "ratio_dims": self.ratio_dims,
                "sparsity": self.sparsity,
                "seed": self.seed
            }
        )

        self._dataarray = dataarray
        return dataarray

    def _create_dataframe(self) -> dd.DataFrame:
        """Create dask DataFrame from generated data.

        Constructs a dask DataFrame with num_obs rows and num_dims + 1 columns.
        Each row contains the coordinates of an observation point and the
        corresponding record value.

        Returns:
            dask DataFrame with observation coordinates and values

        Raises:
            RuntimeError: If required data has not been generated yet
        """
        if self._coordinates is None:
            raise RuntimeError("Coordinates must be generated first")

        if self._observations is None:
            raise RuntimeError("Observations must be generated first")

        if self._record is None:
            raise RuntimeError("Record must be generated first")

        # Find non-NaN points in record
        # Need to compute the record to find non-NaN locations
        record_computed = self._record.compute()

        # non_nan_mask has the same shape of _record, and contains False where
        # the corresponding value in _record is nan, True otherwise
        #
        # non_nan_indices is a tuple containing num_dims arrays, each containing
        # num_obs elements, where each element is the index of the coordinate
        # along that dimension for the corresponding observation value.
        #
        # Example:
        # _record =
        # array([[ 0.1, 0.2, nan, 0.4],
        #        [ nan, nan, 0.7, 0.8],
        #        [ 0.9, 0.2, 0.4, 0.5]])
        # non_nan_indices =
        # (array([0, 0, 0, 1, 1, 2, 2, 2, 2]), array([0, 1, 3, 2, 3, 0, 1, 2, 3]))
        #
        # so num_obs=9, and the location of the record values along dim0 is at positions
        # non_nan_indices[0]=array([0, 0, 0, 1, 1, 2, 2, 2, 2]
        # and along dim1 at positions
        # non_nan_indices[1]=array([0, 1, 3, 2, 3, 0, 1, 2, 3])
        # e.g: _record[0,0] = 0.1, _record[1,3] = 0.7, etc.
        non_nan_mask = ~np.isnan(record_computed)
        non_nan_indices = np.where(non_nan_mask)

        # Build DataFrame columns
        data_dict = {}

        # Add coordinate columns
        coord_names = list(self._coordinates.keys())
        coord_arrays = list(self._coordinates.values())

        for i, (name, coords) in enumerate(zip(coord_names, coord_arrays)):
            # Map indices to coordinate values
            # Need to compute coordinates to index them
            coords_computed = coords.compute()
            data_dict[name] = coords_computed[non_nan_indices[i]]

        # Add record values
        data_dict["record"] = record_computed[non_nan_mask]

        # Create pandas DataFrame first
        dataframe_pd = pd.DataFrame(data_dict)

        # Convert to dask DataFrame
        # Target 300MB per partition as specified in requirements
        # Estimate row size: num_dims * 8 bytes (float64) + 8 bytes (record float64)
        row_size_bytes = (self.num_dims + 1) * 8
        target_partition_size = 300 * 1024 * 1024  # 300MB in bytes
        rows_per_partition = max(1, int(target_partition_size / row_size_bytes))

        # Create dask dataframe with calculated partition size
        npartitions = max(1, len(dataframe_pd) // rows_per_partition)
        dataframe = dd.from_pandas(dataframe_pd, npartitions=npartitions)

        self._dataframe = dataframe
        return dataframe

    def save_to_netcdf(self, filepath: str, overwrite: str = False) -> None:
        """Save data to NetCDF file format.

        The dask array is saved with chunking optimized for x0-first access,
        then x1, then x2, etc.

        Args:
            filepath: Path where the NetCDF file should be saved
            overwrite: overwrites existing file

        Raises:
            RuntimeError: If DataArray has not been created yet
        """
        if self._dataarray is None:
            raise RuntimeError(
                "DataArray must be created before saving. "
                "Call generate() first."
            )

        ds_utils.check_nc(filepath, overwrite)

        # Save with compute=True to write the actual data to disk
        # The chunking is already set in the dask array
        self._dataarray.to_netcdf(filepath, compute=True)

    def save_to_parquet(self, dirname: str, filename: str = None, overwrite: bool = False) -> None:
        """Save data to Parquet file format.

        The dask dataframe is repartitioned to target 300MB per partition
        before saving.

        Args:
            dirname: path to directory to store parquet dataset to
            filename: basename for all parquet files in the dataset
            overwrite: overwrites existing datasets

        Raises:
            RuntimeError: If DataFrame has not been created yet
        """
        if self._dataframe is None:
            raise RuntimeError(
                "DataFrame must be created before saving. "
                "Call generate() first."
            )

        # The dataframe is already partitioned in _create_dataframe()
        # with target size of 300MB per partition
        nb_digits = len(str(self._dataframe.npartitions))
        if filename is None:
            filename = 'test'
        name_function = lambda x: f"{filename}_{x:0{nb_digits}d}.parquet"
        ds_utils.check_parquet(
            dirname,
            overwrite=overwrite
        )
        self._dataframe.to_parquet(
            dirname,
            engine="pyarrow",
            name_function=name_function,
            append=False,
            overwrite=overwrite,
            write_metadata_file = True,
        )


    def generate(
        self,
        netcdf_filepath: str = None,
        parquet_filepath: str = None
    ) -> Tuple[xr.DataArray, dd.DataFrame]:
        """Generate all data and optionally save to files.

        Main orchestration method that executes the complete data generation
        workflow:
        1. Generate coordinates for each dimension as dask arrays
        2. Generate random observation values as dask array
        3. Create sparse record dask array
        4. Build xarray DataArray backed by dask
        5. Build dask DataFrame
        6. Optionally save to NetCDF and/or Parquet files

        The returned objects are lazy - they do not compute values until needed.
        This allows for larger-than-memory data generation.

        Args:
            netcdf_filepath: Optional path to save NetCDF file
            parquet_filepath: Optional path to save Parquet file

        Returns:
            Tuple of (DataArray, dask DataFrame) containing the generated data
        """
        # Execute generation pipeline
        self._generate_coordinates()
        self._generate_observations()
        self._generate_record()
        dataarray = self._create_dataarray()
        dataframe = self._create_dataframe()

        # Save files if paths provided
        if netcdf_filepath is not None:
            self.save_to_netcdf(netcdf_filepath)

        if parquet_filepath is not None:
            self.save_to_parquet(parquet_filepath)

        return dataarray, dataframe
