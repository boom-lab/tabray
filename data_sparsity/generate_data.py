"""Generate synthetic data for array and tabular format comparison.

This module provides the GenerateData class for creating dummy observations
and storing them in both array (netCDF) and tabular (Parquet) formats.
"""

import logging
from typing import Tuple, Union
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster, as_completed
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
        seed: int,
        max_obs: int = None,
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
        # Store global variables

        print("Updated configuration after validation:")
        print(f"  Number of observations: {self.num_obs}")
        print(f"  Number of dimensions: {self.num_dims}")
        print(f"  Ratio of dimensions: {self.ratio_dims}")
        print(f"  Dimensions shape: {self.shape}")
        print(f"  Total grid poitns: {self.total_grid_points}")
        print(f"  Sparsity: {self.sparsity}")
        print(f"  Random seed: {self.seed}")

        self._multiprocessing_setup(max_obs=max_obs)

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

        print("All dimensions contain approximately a natural number of elements, casting and/or rounding them:")
        print(f"  Old number of elements: {self.nb_coords_per_dim}")
        self.nb_coords_per_dim = np.rint(self.nb_coords_per_dim).astype(int)
        print(f"  New number of elements: {self.nb_coords_per_dim}")
        self.shape = [int(dim_size) for dim_size in self.nb_coords_per_dim]
        self.total_grid_points = np.prod(self.shape)

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

    def _multiprocessing_setup(self, max_obs : int = None) -> None:
        """Set up multiprocessing environment with dask
        """

        ntasks = 1 # this is equivalent to the number of chunks that will be generated
        if max_obs is None:
            max_obs = 10000000 #1e7 obs, very empirical

        num_obs = self.num_obs
        if max_obs < num_obs:
            ntasks = int(np.ceil(num_obs/max_obs))
            print(
                "Dataset will be generated in parallel: total observations are more "
                f"than threshold ({self.num_obs}>{max_obs})."
            )

        self.NTASKS = ntasks
        if ntasks == 1:
            return

        max_dim = np.argmax(self.nb_coords_per_dim)
        max_dim_size = self.nb_coords_per_dim[max_dim]
        if max_dim_size < ntasks:
            raise ValueError(
                f"Dimension has size {max_dim_size} but {ntasks} blocks should be generated?"
            )
        Neach_section, extras = divmod(max_dim_size, ntasks)
        Neach_section = int(Neach_section)
        section_sizes = [0] + extras * [Neach_section + 1] + (ntasks - extras) * [Neach_section]
        div_points = np.array(section_sizes, dtype=int).cumsum()

        self.dim_split = max_dim
        self.max_dim_size = max_dim_size
        self.section_sizes = section_sizes[1:]
        self.div_points = div_points

        print(f"  Number of blocks: {self.NTASKS}")
        print(f"  Dataset split along {self.dim_split}-th dimension")
        print(f"  Block dimensions along it: {self.section_sizes}")


    def _generate_coordinates(
        self,
        shape: list = None,
        rng: np.random.Generator = None,
        dim_ranges: dict = None,
        dim_rngs: dict = None
    ) -> dict:
        """Generate random coordinates for each dimension.

        Creates coordinate arrays for each dimension with values sorted
        in ascending order within specified ranges.

        Args:
            shape: Optional shape tuple/list. If None, uses self.nb_coords_per_dim
            rng: Optional random number generator. If None, uses self._rng
            dim_ranges: Optional dict mapping dimension indices to (low, high) tuples
                       for custom coordinate ranges. If None, uses [0, 1) for all dims.
            dim_rngs: Optional dict mapping dimension indices to specific RNGs to use.
                     If provided, these override the default rng for those dimensions.

        Returns:
            Dictionary mapping dimension names to coordinate arrays
        """
        if shape is None:
            shape = self.nb_coords_per_dim
        if rng is None:
            rng = self._rng
        if dim_ranges is None:
            dim_ranges = {}
        if dim_rngs is None:
            dim_rngs = {}

        # Generate random coordinate arrays for each dimension
        coordinates = {}
        for idx, n_coords in enumerate(shape):
            dim_name = f"x{idx}"
            low, high = dim_ranges.get(idx, (0.0, 1.0))
            dim_rng = dim_rngs.get(idx, rng)
            coordinates[dim_name] = np.sort(dim_rng.uniform(low, high, size=n_coords))

        self._coordinates = coordinates
        return coordinates

    def _generate_observations(
        self,
        num_obs: int = None,
        rng: np.random.Generator = None
    ) -> np.ndarray:
        """Generate random observation values.

        Creates random values in the range [0, 1].

        Args:
            num_obs: Number of observations to generate. If None, uses self.num_obs
            rng: Random number generator to use. If None, uses self._rng

        Returns:
            Array of random observation values
        """
        if num_obs is None:
            num_obs = self.num_obs
        if rng is None:
            rng = self._rng

        observations = rng.uniform(0, 1, size=num_obs)

        self._observations = observations
        return observations

    def _generate_record(
        self,
        shape: list = None,
        num_obs: int = None,
        observations: np.ndarray = None,
        rng: np.random.Generator = None
    ) -> np.ndarray:
        """Generate sparse record array with observations.

        Creates a multi-dimensional array with the specified shape, then randomly
        selects positions and assigns the observation values to those points.
        All other points are NaN.

        Args:
            shape: Shape of the record array. If None, uses self.shape
            num_obs: Number of observations to place. If None, uses self.num_obs
            observations: Pre-generated observation values. If None, generates them
            rng: Random number generator. If None, uses self._rng

        Returns:
            Multi-dimensional array with sparse observations

        Raises:
            RuntimeError: If coordinates or observations have not been generated
                         when using default parameters
        """
        # Validate that required data exists when using defaults (serial workflow)
        if shape is None:
            if self._coordinates is None:
                raise RuntimeError(
                    "Coordinates must be generated before record. "
                    "Call _generate_coordinates() first."
                )
            shape = self.shape

        if num_obs is None:
            num_obs = self.num_obs

        if observations is None and rng is None:
            if self._observations is None:
                raise RuntimeError(
                    "Observations must be generated before record. "
                    "Call _generate_observations() first."
                )
            observations = self._observations

        if rng is None:
            rng = self._rng

        # Get total grid points and check sparsity for serial workflow
        total_grid_points = np.prod(shape)
        if shape == self.shape:
            s_estim = num_obs / total_grid_points
            if s_estim != self.sparsity:
                raise ValueError(
                    f"Sparsity {s_estim} determined from number "
                    f"of coordinates differs from sparsity {self.sparsity} "
                    "determined during parameters validation step."
                )

        # Initialize record with NaN
        record = np.full(shape, np.nan)

        # Generate random indices for observation placement
        # Flatten the multi-dimensional index space
        flat_indices = rng.choice(
            total_grid_points,
            size=num_obs,
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

        # Generate or use provided observations
        if observations is None:
            observations = rng.uniform(0, 1, size=num_obs)

        # Assign observation values to selected points
        record[multi_indices] = observations

        # Store as instance variable for serial workflow
        if shape == self.shape:
            self._record = record

        return record

    def _generate_par(self) -> None:
        """Generate sparse record array with observations using parallel
        processing.

        Submit as many dataset generation tasks as number of blocks needed.
        """

        cluster = LocalCluster(n_workers=4, threads_per_worker=1, processes=True)
        client = Client(cluster)
        print("Dask dashboard:", client.dashboard_link)

        total_obs = self.num_obs
        total_points = np.prod(self.shape)
        total_points_slice = total_points/self.max_dim_size
        chunk_points = np.array(
            [total_points_slice*chunk_size for chunk_size in self.section_sizes]
        ).astype(int)
        print("type chunk_points", type(chunk_points))
        print("chunk_points", chunk_points)
        # as randomness is uniform
        per_chunk_obs = np.rint(self.sparsity*chunk_points).astype(int)
        for idx, c in enumerate(per_chunk_obs):
            if c > chunk_points[idx]:
                per_chunk_obs[idx]=chunk_points[idx]

        mp_obs = per_chunk_obs.sum()
        if not mp_obs.is_integer():
            raise ValueError(f"Got non integer value of observations {mp_obs}.")
        mp_obs = int(mp_obs)
        sparsity = mp_obs/total_points
        print(f"Multiprocessing approximations lead to {mp_obs} total observation (goal: {total_obs}).")
        print(f"Updated sparsity is {sparsity} (was: {self.sparsity}).")
        self.sparsity = sparsity
        self.num_obs = mp_obs

        # Submit one task per seed
        futures = [
            client.submit(self._generate_record_par, chunk_id, chunk_obs)
            for chunk_id, chunk_obs in zip(range(self.NTASKS),per_chunk_obs)
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

        ddf = dd.read_parquet('./parquet_tmp/')
        ddf = ddf.repartition(partition_size="300MB")
        self.save_to_parquet(
            './pqt/',
            ddf,
            overwrite=True
        )


    def _generate_record_par(self, chunk_id : int, obs_in_chunk : int) -> Tuple[int, int]:
        """Generate sparse record array for a single chunk in parallel processing.

        This method generates a chunk of the full array by:
        1. Setting up chunk-specific and global random generators
        2. Generating coordinates (using global RNG for shared dims, local for split dim)
        3. Generating the sparse record array for this chunk
        4. Creating and saving DataArray and DataFrame representations

        Args:
            chunk_id: Identifier for this chunk
            obs_in_chunk: Number of observations to generate in this chunk

        Returns:
            Tuple of (chunk_id, number of observations stored)
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
        global_rng = np.random.default_rng(self.seed)
        task_rng = np.random.default_rng(self.seed + chunk_id)

        # Determine chunk dimensions and range along split dimension
        task_range = (self.div_points[chunk_id], self.div_points[chunk_id+1])
        task_size  = self.section_sizes[chunk_id]
        task_shape = self.shape.copy()
        task_shape[self.dim_split] = task_size
        logging.debug("task_range: %s", task_range)
        logging.debug("task_size: %s", task_size)
        logging.debug("task_shape: %s", task_shape)

        total_chunk_points = np.prod(task_shape)
        if not total_chunk_points.is_integer():
            raise ValueError("total_chunk_points must be an int")
        total_chunk_points = int(total_chunk_points)
        logging.debug("total_chunk_points: %s", total_chunk_points)

        # Build dimension ranges and RNGs for coordinate generation
        # Non-split dimensions use [0, 1) with global_rng
        # Split dimension uses normalized chunk range with task_rng
        dim_ranges = {
            self.dim_split: (
                task_range[0]/self.max_dim_size,
                task_range[1]/self.max_dim_size
            )
        }
        dim_rngs = {self.dim_split: task_rng}

        # Generate coordinates using generalized method
        coordinates = self._generate_coordinates(
            shape=task_shape,
            rng=global_rng,
            dim_ranges=dim_ranges,
            dim_rngs=dim_rngs
        )

        logging.debug("obs in chunk: %s", obs_in_chunk)
        logging.debug("total chunk points: %s", total_chunk_points)

        # Generate record using generalized method
        record = self._generate_record(
            shape=task_shape,
            num_obs=obs_in_chunk,
            observations=None,  # Will be generated inside _generate_record
            rng=task_rng
        )

        logging.debug("chunk id: %s", chunk_id)
        logging.debug("record.shape: %s", record.shape)
        logging.debug("num obs in chunk: %s", obs_in_chunk)
        logging.debug("non-nans in chunk: %s", np.sum( ~np.isnan(record) ))
        logging.debug("dims: %s", list(coordinates.keys()))
        logging.debug("coords: %s", coordinates)

        # Create DataArray with chunk-specific attributes using generalized method
        chunk_attrs = {
            "description": "Sparse observation data",
            "chunk_id": chunk_id,
            "global_num_obs": self.num_obs,
            "global_num_dims": self.num_dims,
            "global_ratio_dims": self.ratio_dims,
            "global_sparsity": self.sparsity,
            "global_seed": self.seed
        }
        dataarray = self._create_dataarray(
            record=record,
            coordinates=coordinates,
            attrs=chunk_attrs
        )

        # Save to NetCDF
        fpath = f'./nc/test_{chunk_id}.nc'
        self.save_to_netcdf(fpath, dataarray=dataarray, overwrite=True)

        # Create and save DataFrame using generalized method
        dataframe = dd.from_pandas(
            self._create_dataframe(record=record, coordinates=coordinates)
        )

        dirpath = './parquet_tmp/'
        self.save_to_parquet(
            dirpath,
            dataframe,
            filename=f'test_{chunk_id}',
            overwrite=False
        )

        return chunk_id, np.sum( ~np.isnan(record) )


    def _create_dataarray(
        self,
        record: np.ndarray = None,
        coordinates: dict = None,
        attrs: dict = None
    ) -> xr.DataArray:
        """Create xarray DataArray from generated data.

        Constructs an xarray DataArray with coordinates and record data.

        Args:
            record: The record array to use. If None, uses self._record
            coordinates: Dictionary of coordinates. If None, uses self._coordinates
            attrs: Dictionary of attributes for the DataArray. If None, creates
                   default attributes from instance parameters.

        Returns:
            xarray DataArray containing the sparse observation data

        Raises:
            RuntimeError: If required data has not been generated yet
        """
        if coordinates is None:
            if self._coordinates is None:
                raise RuntimeError("Coordinates must be generated first")
            coordinates = self._coordinates

        if record is None:
            if self._record is None:
                raise RuntimeError("Record must be generated first")
            record = self._record

        # Create default attributes if not provided
        if attrs is None:
            attrs = {
                "description": "Sparse observation data",
                "num_obs": self.num_obs,
                "num_dims": self.num_dims,
                "ratio_dims": self.ratio_dims,
                "sparsity": self.sparsity,
                "seed": self.seed
            }

        # Create DataArray with coordinates
        dataarray = xr.DataArray(
            record,
            coords=coordinates,
            dims=list(coordinates.keys()),
            name="record",
            attrs=attrs
        )

        # Store as instance variable for serial workflow
        if record is self._record:
            self._dataarray = dataarray

        return dataarray

    def _create_dataframe(
            self,
            record : np.ndarray = None,
            coordinates : dict = None,
    ) -> pd.DataFrame:
        """Create pandas DataFrame from generated data.

        Constructs a pandas DataFrame where each row contains the coordinates
        of an observation point and the corresponding record value. Only non-NaN
        points from the record are included.

        Args:
            record: The record array to use. If None, uses self._record
            coordinates: Dictionary of coordinates. If None, uses self._coordinates

        Returns:
            pandas DataFrame with observation coordinates and values

        Raises:
            RuntimeError: If required data has not been generated yet
        """
        if coordinates is None:
            if self._coordinates is None:
                raise RuntimeError("Coordinates must be generated first")
            coordinates = self._coordinates

        if record is None:
            if self._record is None:
                raise RuntimeError("Record must be generated first")
            record = self._record

        # Find non-NaN points in record
        #
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
        non_nan_mask = ~np.isnan(record)
        non_nan_indices = np.where(non_nan_mask)

        # Build DataFrame columns
        data_dict = {}

        # Add coordinate columns
        coord_names = list(coordinates.keys())
        coord_arrays = list(coordinates.values())

        for i, (name, coords) in enumerate(zip(coord_names, coord_arrays)):
            # Map indices to coordinate values
            data_dict[name] = coords[non_nan_indices[i]]

        # Add record values
        data_dict["record"] = record[non_nan_mask]

        # Create DataFrame
        dataframe = pd.DataFrame(data_dict)

        if self.NTASKS==1:
            self._dataframe = dataframe

        return dataframe

    def save_to_netcdf(self, filepath: str, dataarray: np.array = None, overwrite: str = False) -> None:
        """Save data to NetCDF file format.

        Args:
            filepath: Path where the NetCDF file should be saved
            overwrite: overwrites existing file

        Raises:
            RuntimeError: If DataArray has not been created yet
        """
        if dataarray is None:
            if self._dataarray is None:
                raise RuntimeError(
                    "DataArray must be created before saving. "
                    "Call generate() first."
                )
            dataarray = self._dataarray

        ds_utils.check_nc(filepath, overwrite)
        dataarray.to_netcdf(filepath)

    def save_to_parquet(self, dirname: str, dataframe: Union[pd.DataFrame, dd.DataFrame] = None, filename: str = None, overwrite: bool = False) -> None:
        """Save data to Parquet file format.

        Args:
            dirname: path to directory to store parquet dataset to
            filename: basename for all parquet files in the dataset
            overwrite: overwrites existing datasets

        Raises:
            RuntimeError: If DataFrame has not been created yet
        """
        if dataframe is None:
            if self._dataframe is None:
                raise RuntimeError("DataFrame must be created before saving.")
            dataframe=self._dataframe

        if isinstance(dataframe, pd.DataFrame):
            ddf = dd.from_pandas(dataframe)
        else:
            ddf = dataframe

        nb_digits = len(str(ddf.npartitions))
        if filename is None:
            filename = 'test'

        def name_function(partition_idx):
            """Generate filename for a parquet partition."""
            return f"{filename}_{partition_idx:0{nb_digits}d}.parquet"

        ds_utils.check_parquet(
            dirname,
            overwrite=overwrite
        )

        write_metadata_file = True
        if self.NTASKS > 1:
            write_metadata_file = False
        ddf.to_parquet(
            dirname,
            engine="pyarrow",
            name_function=name_function,
            append=False,
            overwrite=overwrite,
            write_metadata_file = write_metadata_file
        )


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

        # Execute generation pipeline for single process
        if self.NTASKS == 1:
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

        if self.NTASKS > 1:
            self._generate_par()
            return None, None

        raise ValueError(f"NTASKS must positive, got {self.NTASKS} instead.")
