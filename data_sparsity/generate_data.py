"""Generate synthetic data for array and tabular format comparison.

This module provides the GenerateData class for creating dummy observations
and storing them in both array (netCDF) and tabular (Parquet) formats.
"""

from typing import Tuple, Optional
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import xarray as xr


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
        ratio_dims: ArrayLike,
        sparsity: Optional[int,float],
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

        self.ratio_dims_prod = 1
        for r in self.ratio_dims:
            self.ratio_dims_prod *= r

        # Validate all parameters
        self._validate_parameters()
        print("Updated configuration after validation:")
        print(f"  Number of observations: {self.num_obs}")
        print(f"  Number of dimensions: {self.num_dims}")
        print(f"  Ratio of dimensions: {self.ratio_dims}")
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

        Some checks are hard check (i.e. an error is raised if the check fails),
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

        # Check ratio_dims type and convert to np array
        if not isinstance(self.ratio_dims, (list, tuple, np.ndarray)):
            raise TypeError(
                f"ratio_dims must be a tuple, list, or numpy array, got {type(self.ratio_dims)}"
            )
        self.ratio_dims = np.asarray(self.ratio_dims)

        # Check that first dimension in coordinate space has at least one element
        if not isinstance(self.num_dims, int):
            raise TypeError(f"num_dims must be an integer, got {type(self.num_dims)}")
        if self.num_dims <= 0:
            raise ValueError(f"num_dims must be positive, got {self.num_dims}")
        self.nb_elements_dim1 = self.num_obs / (self.sparsity*self.ratio_dims_prod)
        if self.nb_elements_dim1 < 1:
            raise ValueError(f"number of elements for dimension 1 must be larger than 1, got {self.nb_elements_dim1}")

        # Enforce nb_elements_dim1 to be an integer
        print(f"Number of elements in the first dimension is {self.nb_elements_dim1}, rounding to closest integer: {np.rint(self.nb_elements_dim1)}")
        self.nb_elements_dim1 = np.rint(self.nb_elements_dim1)

        # Check that all other dimensions have at least one element
        self.nb_elements_all_dims = self.ratio_dims*self.nb_elements_dim1
        fewer_than_one = self.nb_elements_all_dims<1
        if np.any(fewer_than_one):
            bad_idxs = np.flatnonzero(fewer_than_one)
            msgs = [
                f'Error: dimension {idx+1} must have at least one element, got {self.nb_elements_all_dims[idx]}.'
                for idx in bad_idxs
            ]
            for m in msgs:
                print(m)
            raise ValueError("One or more dimensions do not contain at least one element.")

        # Check that all dimensions contain an integer number of elements
        not_integers = np.logical_not(
            np.isclose(
                self.nb_elements_all_dims,
                np.rint(self.nb_elements_all_dims)
                )
            )
        if np.any(not_integers):
            bad_idxs = np.flatnonzero(not_integers)
            msgs = [
                f"Error: dimension {i+1} does not have an integer number of elements, got {self.nb_elements_all_dims[idx]}"
                for idx in bad_idxs
            ]
            # print or include messages in the exception
            for m in msgs:
                print(m)
            raise ValueError("One or more dimensions contain a decimal number elements.")
        else:
            print("All dimensions contain approximately a natural number of elements, rounding them.")
            print(f"Old number of elements: {self.nb_elements_all_dims}")
            self.nb_elements_all_dims = np.rint(self.nb_elements_all_dims).astype(int)

        # Check that sparsity is larger than minimum allowed for this set of parameters
        self.sparsity_zero = 1/np.min(self.nb_elements_all_dims)
        print(f"Minimum sparsity value for the current set of dimensions: {self.sparsity_zero}")
        if self.sparsity == 0.:
            self.sparsity = self.sparsity_zero
        elif self.sparsity < self.sparsity_zero:
            raise ValueError(f"Provided sparsity value of {self.sparsity} is lower than minimum value of {self.sparsity_zero}. If you want to impose the minimum value possible, set sparsity to 0. as input.")

        # Check that num_obs is consistent with sparsity and dimensions size, else update it
        num_obs_exp = self.sparsity*np.prod(self.nb_elements_all_dims)
        if num_obs_exp != self.num_obs:
            print(f"Input number of observations num_obs ({self.num_obs}) does not match the number of observations num_obs_exp {num_obs_exp} expected from values of sparsity and the number of elements per dimension. This can happen due to rounding operations and is not necessarily an issue, so we are enforcing num_obs to match num_obs_exp.")
            self.num_obs = num_obs_exp

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
            Dictionary mapping dimension names to coordinate arrays
        """
        # Calculate total grid points needed based on sparsity
        total_points = int(np.ceil(self.num_obs / self.sparsity))

        # Normalize ratios to sum to 1
        ratio_sum = sum(self.ratio_dims)
        normalized_ratios = [r / ratio_sum for r in self.ratio_dims]

        # Calculate points per dimension based on normalized ratios
        # We distribute total_points across dimensions maintaining the ratios
        # This is an approximation - we calculate the geometric mean approach
        points_per_dim = []
        dim_product = 1
        for ratio in normalized_ratios:
            # Each dimension gets a size such that product equals total_points
            dim_size = int(np.ceil((total_points * ratio) ** (1 / self.num_dims)))
            points_per_dim.append(max(dim_size, 2))  # Minimum 2 points per dimension
            dim_product *= dim_size

        # Adjust if needed to ensure we have enough total points
        while dim_product < total_points:
            # Find dimension with smallest size relative to its ratio
            ratios_to_size = [r / s for r, s in zip(normalized_ratios, points_per_dim)]
            dim_to_increase = ratios_to_size.index(max(ratios_to_size))
            points_per_dim[dim_to_increase] += 1
            dim_product = np.prod(points_per_dim)

        # Generate coordinate arrays for each dimension
        coordinates = {}
        for i, n_points in enumerate(points_per_dim):
            dim_name = f"dim_{i}"
            # Generate evenly spaced coordinates in [0, 1)
            coordinates[dim_name] = np.linspace(0, 1, n_points, endpoint=False)

        self._coordinates = coordinates
        return coordinates

    def _generate_observations(self) -> np.ndarray:
        """Generate random observation values.

        Creates num_obs random values in the range [0, 1].

        Returns:
            Array of random observation values
        """
        observations = self._rng.uniform(0, 1, size=self.num_obs)
        self._observations = observations
        return observations

    def _generate_record(self) -> np.ndarray:
        """Generate sparse record array with observations.

        Creates a multi-dimensional array with the shape defined by
        coordinates, then randomly selects num_obs points and assigns
        the observation values to those points. All other points are NaN.

        Returns:
            Multi-dimensional array with sparse observations

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
        total_points = np.prod(shape)

        # Initialize record with NaN
        record = np.full(shape, np.nan)

        # Generate random indices for observation placement
        # Flatten the multi-dimensional index space
        flat_indices = self._rng.choice(
            total_points,
            size=self.num_obs,
            replace=False
        )

        # Convert flat indices to multi-dimensional indices
        multi_indices = np.unravel_index(flat_indices, shape)

        # Assign observation values to selected points
        record[multi_indices] = self._observations

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
                "sparsity": self.sparsity,
                "seed": self.seed
            }
        )

        self._dataarray = dataarray
        return dataarray

    def _create_dataframe(self) -> pd.DataFrame:
        """Create pandas DataFrame from generated data.

        Constructs a pandas DataFrame with num_obs rows and num_dims + 1 columns.
        Each row contains the coordinates of an observation point and the
        corresponding record value.

        Returns:
            pandas DataFrame with observation coordinates and values

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
        non_nan_mask = ~np.isnan(self._record)
        non_nan_indices = np.where(non_nan_mask)

        # Build DataFrame columns
        data_dict = {}

        # Add coordinate columns
        coord_names = list(self._coordinates.keys())
        coord_arrays = list(self._coordinates.values())

        for i, (name, coords) in enumerate(zip(coord_names, coord_arrays)):
            # Map indices to coordinate values
            data_dict[name] = coords[non_nan_indices[i]]

        # Add record values
        data_dict["record"] = self._record[non_nan_mask]

        # Create DataFrame
        dataframe = pd.DataFrame(data_dict)

        self._dataframe = dataframe
        return dataframe

    def save_to_netcdf(self, filepath: str) -> None:
        """Save data to NetCDF file format.

        Args:
            filepath: Path where the NetCDF file should be saved

        Raises:
            RuntimeError: If DataArray has not been created yet
        """
        if self._dataarray is None:
            raise RuntimeError(
                "DataArray must be created before saving. "
                "Call generate() first."
            )

        self._dataarray.to_netcdf(filepath)

    def save_to_parquet(self, filepath: str) -> None:
        """Save data to Parquet file format.

        Args:
            filepath: Path where the Parquet file should be saved

        Raises:
            RuntimeError: If DataFrame has not been created yet
        """
        if self._dataframe is None:
            raise RuntimeError(
                "DataFrame must be created before saving. "
                "Call generate() first."
            )

        self._dataframe.to_parquet(filepath, index=False)

    def generate(
        self,
        netcdf_filepath: str = None,
        parquet_filepath: str = None
    ) -> Tuple[xr.DataArray, pd.DataFrame]:
        """Generate all data and optionally save to files.

        Main orchestration method that executes the complete data generation
        workflow:
        1. Generate coordinates for each dimension
        2. Generate random observation values
        3. Create sparse record array
        4. Build xarray DataArray
        5. Build pandas DataFrame
        6. Optionally save to NetCDF and/or Parquet files

        Args:
            netcdf_filepath: Optional path to save NetCDF file
            parquet_filepath: Optional path to save Parquet file

        Returns:
            Tuple of (DataArray, DataFrame) containing the generated data
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
