"""Generate synthetic data for array and tabular format comparison.

This module provides the GenerateData class for creating dummy observations
and storing them in both array (netCDF) and tabular (Parquet) formats.
"""

import gc
import logging
import os
from typing import List, Tuple, Union
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

        print("Input configuration:")
        print(f"  Number of observations: {self.num_obs}")
        print(f"  Number of dimensions: {self.num_dims}")
        print(f"  Ratio of dimensions: {self.ratio_dims}")
        print(f"  Sparsity: {self.sparsity}")
        print(f"  Random seed: {self.seed}")
        print(f"  Number of variables: {self.num_vars}")
        print(f"  Variable dimensions: {self.var_dims}")
        print(f"  Overlap: {self.overlap}")

        self.ratio_dims_prod = np.prod(self.ratio_dims)

        # Initialize random number generator early (needed for validation)
        self._rng = np.random.default_rng(seed)

        # Initialize attributes that will be set during validation
        self.var_sparsities = None
        self.var_num_obs = None
        self.var_dims_indices = None
        self.overlap_target = None
        self.overlap_actual = None

        # Validate all parameters
        self._validate_parameters()
        # Store global variables

        print("Updated configuration after validation:")
        print(f"  Number of observations: {self.num_obs}")
        print(f"  Number of dimensions: {self.num_dims}")
        print(f"  Ratio of dimensions: {self.ratio_dims}")
        print(f"  Dimensions shape: {self.shape}")
        print(f"  Total grid points: {self.total_grid_points}")
        print(f"  Sparsity: {self.sparsity}")
        print(f"  Random seed: {self.seed}")
        print(f"  Number of variables: {self.num_vars}")
        print(f"  Variable dimensions: {self.var_dims}")
        print(f"  Variable sparsities: {self.var_sparsities}")
        print(f"  Variable observations: {self.var_num_obs}")
        print(f"  Overlap: {self.overlap}")

        self._multiprocessing_setup(max_obs=max_obs)

        # Storage for generated data
        self._coordinates = None
        self._observations = None
        self._record = None
        self._dataarray = None
        self._dataframe = None
        self._records = None  # For multi-variable
        self._dataset = None  # For multi-variable

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

        # Check that sparsity is valid type and store representative value
        # for grid calculations (we'll process it fully later)
        if isinstance(self.sparsity, (float, int)):
            sparsity_for_grid = float(self.sparsity)
        elif isinstance(self.sparsity, (list, tuple)):
            # Use max sparsity for grid calculation (most observations)
            sparsity_for_grid = float(max(self.sparsity))
        else:
            raise TypeError(
                f"sparsity must be a number, list, or tuple, got {type(self.sparsity)}"
            )
        if not 0.0 <= sparsity_for_grid <= 1.0:
            raise ValueError(
                f"sparsity values must be between 0 and 1.0, "
                f"got max={sparsity_for_grid}"
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
                    f"(if this is not 1), got {self.num_dims} and "
                    f"{len(self.ratio_dims)}, respectively."
                )

        # Check ratio_dims type and convert to np array
        if isinstance(self.ratio_dims, int):
            if self.ratio_dims == 1:
                # impose same size along all dimensions
                self.ratio_dims = self.num_dims*[1]
        elif not isinstance(self.ratio_dims, (list, tuple, np.ndarray)):
            raise TypeError(
                f"ratio_dims must be a tuple, list, or numpy array, "
                f"got {type(self.ratio_dims)}"
            )
        self.ratio_dims = np.asarray(self.ratio_dims)

        # Check that first dimension in coordinate space has at least one element
        if not isinstance(self.num_dims, int):
            raise TypeError(f"num_dims must be an integer, got {type(self.num_dims)}")
        if self.num_dims <= 0:
            raise ValueError(f"num_dims must be positive, got {self.num_dims}")
        base = self.num_obs / (sparsity_for_grid*self.ratio_dims_prod)
        self.nb_coords_dim1 = np.power( base, 1/self.num_dims )
        if self.nb_coords_dim1 < 1:
            raise ValueError(
                f"number of elements for dimension 1 must be larger than 1, "
                f"got {self.nb_coords_dim1}"
            )

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
        print(
            f"Minimum sparsity value for the current set of dimensions: "
            f"{self.sparsity_zero}"
        )
        if sparsity_for_grid == 0.:
            sparsity_for_grid = self.sparsity_zero
            print(
                f"Input sparsity is zero, imposing minimum value: "
                f"{self.sparsity_zero}"
            )
            # Update the original sparsity too
            if isinstance(self.sparsity, (float, int)):
                self.sparsity = self.sparsity_zero
        elif sparsity_for_grid < self.sparsity_zero:
            raise ValueError(
                f"Provided sparsity value of {sparsity_for_grid} is lower than "
                f"minimum value of {self.sparsity_zero}. If you want to impose "
                "the minimum value possible, set sparsity to 0. as input."
            )

        # Check that num_obs is consistent with sparsity and dimensions size
        num_obs_exp = sparsity_for_grid*np.prod(self.nb_coords_per_dim)
        if num_obs_exp != self.num_obs:
            print(
                f"Input number of observations num_obs ({self.num_obs}) does not "
                f"match the number of observations num_obs_exp {num_obs_exp} "
                "expected from values of sparsity and the number of elements per "
                "dimension. This can happen due to rounding operations and is not "
                "necessarily an issue, so we are enforcing num_obs to match "
                "num_obs_exp and rounding it."
            )
            self.num_obs = np.rint(num_obs_exp).astype(int)
            print(f"New number of observations is {self.num_obs}")
            sparsity_for_grid = self.num_obs/np.prod(self.nb_coords_per_dim)
            print(f"Actual sparsity for grid is now {sparsity_for_grid}")
            if sparsity_for_grid < self.sparsity_zero or sparsity_for_grid > 1:
                raise ValueError(
                    f"Sparsity value {sparsity_for_grid} out of bounds "
                    f"[{self.sparsity_zero},1]"
                )
            # Update sparsity for use in multi-variable validation
            # For single-value sparsity, update it
            if isinstance(self.sparsity, (float, int)):
                self.sparsity = sparsity_for_grid

        # Check that seed is a non-negative integer
        if not isinstance(self.seed, int):
            raise TypeError(f"seed must be an integer, got {type(self.seed)}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

        # Validate num_vars
        if not isinstance(self.num_vars, int):
            raise TypeError(f"num_vars must be an integer, got {type(self.num_vars)}")
        if self.num_vars <= 0:
            raise ValueError(f"num_vars must be positive, got {self.num_vars}")

        # Validate and process parameters for multiple variables
        # NOTE: This must come after sparsity adjustment above
        if self.num_vars > 1:
            self._validate_and_setup_sparsity()
            self._validate_and_setup_var_dims()
            self._validate_and_setup_overlap()
        else:
            # For single variable, set up simple values
            self.var_sparsities = np.array([sparsity_for_grid])
            self.var_num_obs = np.array([self.num_obs])
            self.var_dims_indices = [list(range(self.num_dims))]  # All dimensions vary
            self.var_constant_dims = [[]]  # No constant dimensions
            self.overlap_target = None
            self.overlap_actual = None

    def _validate_and_setup_sparsity(self) -> None:
        """Validate and setup sparsity for multiple variables.

        This method processes the sparsity parameter which can be:
        - A scalar: all variables get the same sparsity
        - A 2-element list/tuple: one var gets min, one gets max, rest are random
        - A num_vars-element list/tuple: each var gets its corresponding sparsity

        Sets self.var_sparsities and self.var_num_obs
        """
        # Store the original sparsity value for the main variable
        # (used for grid calculations which have already been done)
        original_sparsity = self.sparsity

        if isinstance(self.sparsity, (float, int)):
            # Scalar: all variables get the same sparsity
            self.var_sparsities = np.array([original_sparsity] * self.num_vars)
        elif isinstance(self.sparsity, (list, tuple)):
            sparsity_list = list(self.sparsity)
            if len(sparsity_list) == 2:
                # Two elements: assign min and max, randomize the rest
                min_spar = min(sparsity_list)
                max_spar = max(sparsity_list)
                if self.num_vars == 2:
                    # Randomly assign which gets min and which gets max
                    if self._rng.random() < 0.5:
                        self.var_sparsities = np.array([min_spar, max_spar])
                    else:
                        self.var_sparsities = np.array([max_spar, min_spar])
                else:
                    # Generate random values for all except 2
                    random_sparsities = self._rng.uniform(
                        min_spar, max_spar, size=self.num_vars - 2
                    )
                    # Combine and shuffle
                    all_sparsities = np.concatenate(
                        [[min_spar, max_spar], random_sparsities]
                    )
                    self._rng.shuffle(all_sparsities)
                    self.var_sparsities = all_sparsities
            elif len(sparsity_list) == self.num_vars:
                # One sparsity per variable
                self.var_sparsities = np.array(sparsity_list)
            else:
                raise ValueError(
                    f"sparsity list must have 2 or {self.num_vars} elements, "
                    f"got {len(sparsity_list)}"
                )
        else:
            raise TypeError(
                f"sparsity must be a scalar, list, or tuple, got {type(self.sparsity)}"
            )

        # Validate all sparsity values and clip to minimum
        for j, spar in enumerate(self.var_sparsities):
            if not 0.0 <= spar <= 1.0:
                raise ValueError(
                    f"Variable {j} sparsity {spar} must be between 0 and 1"
                )
            if spar < self.sparsity_zero:
                print(
                    f"WARNING: Variable {j} sparsity {spar} is below minimum "
                    f"{self.sparsity_zero}, clipping to minimum"
                )
                self.var_sparsities[j] = self.sparsity_zero

        # Compute number of observations for each variable
        # The variable with the highest sparsity has num_obs observations
        max_sparsity = np.max(self.var_sparsities)
        self.var_num_obs = np.rint(
            (self.var_sparsities / max_sparsity) * self.num_obs
        ).astype(int)

        # Ensure at least 1 observation per variable
        self.var_num_obs = np.maximum(self.var_num_obs, 1)

    def _validate_and_setup_var_dims(self) -> None:
        """Validate and setup dimensions for each variable.

        This method processes the var_dims parameter which can be:
        - An int: all variables have this many VARYING dimensions (randomly selected)
        - A list/tuple of ints: each var has that many VARYING dims (randomly selected)
        - A list/tuple of lists/tuples: each var has explicitly specified VARYING dims

        All variables are measured in the full num_dims space. Dimensions not in
        var_dims_indices are held constant for that variable.

        Sets self.var_dims_indices (varying dimensions) and self.var_constant_dims
        """
        if isinstance(self.var_dims, int):
            # Same number of dimensions for all variables
            if self.var_dims > self.num_dims:
                raise ValueError(
                    f"var_dims {self.var_dims} cannot exceed num_dims {self.num_dims}"
                )
            if self.var_dims == self.num_dims:
                # All variables use all dimensions
                self.var_dims_indices = [list(range(self.num_dims))] * self.num_vars
            else:
                # Ensure all dimensions are covered across variables by cycling through
                all_dims = list(range(self.num_dims))
                self.var_dims_indices = []
                
                # First, ensure each dimension appears at least once
                for var_idx in range(self.num_vars):
                    if var_idx < self.num_dims:
                        # For the first num_dims variables, ensure dimension coverage
                        # Start with the dimension that matches the variable index
                        start_dim = var_idx
                        available_dims = [start_dim] + [
                            d for d in all_dims if d != start_dim
                        ]
                        # Randomly select remaining dimensions
                        selected = [start_dim]
                        remaining_needed = self.var_dims - 1
                        if remaining_needed > 0:
                            other_dims = self._rng.choice(
                                [d for d in available_dims if d != start_dim],
                                size=min(remaining_needed, len(available_dims) - 1),
                                replace=False
                            ).tolist()
                            selected.extend(other_dims)
                        self.var_dims_indices.append(sorted(selected))
                    else:
                        # For additional variables, just random selection
                        dims = sorted(
                            self._rng.choice(
                                self.num_dims, size=self.var_dims, replace=False
                            ).tolist()
                        )
                        self.var_dims_indices.append(dims)
        elif isinstance(self.var_dims, (list, tuple)):
            if len(self.var_dims) != self.num_vars:
                raise ValueError(
                    f"var_dims list must have {self.num_vars} elements, "
                    f"got {len(self.var_dims)}"
                )
            self.var_dims_indices = []
            for j, vd in enumerate(self.var_dims):
                if isinstance(vd, int):
                    if vd > self.num_dims:
                        raise ValueError(
                            f"Variable {j} var_dims {vd} cannot exceed "
                            f"num_dims {self.num_dims}"
                        )
                    if vd == self.num_dims:
                        self.var_dims_indices.append(list(range(self.num_dims)))
                    else:
                        dims = sorted(
                            self._rng.choice(
                                self.num_dims, size=vd, replace=False
                            ).tolist()
                        )
                        self.var_dims_indices.append(dims)
                elif isinstance(vd, (list, tuple)):
                    dims = list(vd)
                    if len(dims) == 0 or len(dims) > self.num_dims:
                        raise ValueError(
                            f"Variable {j} dim indices must have 1 to "
                            f"{self.num_dims} elements, got {len(dims)}"
                        )
                    if not all(0 <= d < self.num_dims for d in dims):
                        raise ValueError(
                            f"Variable {j} dim indices {dims} must be in "
                            f"range [0, {self.num_dims})"
                        )
                    if len(set(dims)) != len(dims):
                        raise ValueError(
                            f"Variable {j} dim indices {dims} contain duplicates"
                        )
                    self.var_dims_indices.append(sorted(dims))
                else:
                    raise TypeError(
                        f"Variable {j} var_dims must be int or list/tuple, "
                        f"got {type(vd)}"
                    )
        else:
            raise TypeError(
                f"var_dims must be int, list, or tuple, got {type(self.var_dims)}"
            )

        # Set up constant dimensions for each variable
        # Constant dimensions will have shape=1, and we'll randomly select
        # a coordinate value from the available coordinates
        self.var_constant_dims = []
        self.var_shapes = []  # Full shape where constant dims have size 1
        for var_idx in range(self.num_vars):
            varying_dims = self.var_dims_indices[var_idx]
            constant_dims = [d for d in range(self.num_dims) if d not in varying_dims]
            self.var_constant_dims.append(constant_dims)

    def _validate_and_setup_overlap(self) -> None:
        """Validate and setup overlap parameter.

        This method validates the overlap parameter and computes constraints.
        Sets self.overlap_target and self.overlap_actual
        """
        if isinstance(self.overlap, str):
            if self.overlap != 'random':
                raise ValueError(
                    f"overlap string must be 'random', got '{self.overlap}'"
                )
            self.overlap_target = 'random'
        elif isinstance(self.overlap, (float, int)):
            if not 0.0 <= self.overlap <= 1.0:
                raise ValueError(
                    f"overlap must be between 0 and 1, got {self.overlap}"
                )
            self.overlap_target = float(self.overlap)

            # Check for sparsity=1 case
            if np.all(self.var_sparsities == 1.0):
                if self.overlap_target != 1.0:
                    print(
                        "WARNING: All variables have sparsity=1, "
                        "so overlap will be 1 regardless of target"
                    )
                    self.overlap_target = 1.0

            # Compute minimum possible overlap
            min_overlap = self._compute_min_overlap()
            if self.overlap_target < min_overlap:
                print(
                    f"WARNING: Requested overlap {self.overlap_target} is below "
                    f"minimum possible {min_overlap:.4f} for this configuration. "
                    f"Using minimum overlap instead."
                )
                self.overlap_target = min_overlap
        else:
            raise TypeError(
                f"overlap must be float or 'random', got {type(self.overlap)}"
            )

        self.overlap_actual = None  # Will be computed after generation

    def _compute_min_overlap(self) -> float:
        """Compute the minimum possible overlap given the configuration.

        The minimum overlap is determined by the available grid points and the
        number of observations. If there are fewer grid points than the sum of
        all observations, some observations must overlap.

        Returns:
            Minimum overlap value (0.0 to 1.0)
        """
        # Get total sites available
        total_sites = self.total_grid_points

        # Get observation counts for all variables
        sorted_obs = np.sort(self.var_num_obs)[::-1]  # Descending order

        # The variable with most observations sets the baseline
        max_obs = sorted_obs[0]

        # Sum of all other observations
        other_obs = np.sum(sorted_obs[1:])

        # If there are no other observations, this is an error
        # (shouldn't happen with num_vars>1 after validation)
        if other_obs == 0:
            raise ValueError(
                "Cannot compute minimum overlap: no observations in non-reference variables"
            )

        # If total_sites >= max_obs + other_obs, min overlap is 0
        if total_sites >= max_obs + other_obs:
            return 0.0

        # Otherwise, compute how many must overlap
        must_overlap = max_obs + other_obs - total_sites
        min_overlap = must_overlap / other_obs

        return min_overlap

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

        # Store as instance variable only for serial workflow
        if self.NTASKS == 1:
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

        # Store as instance variable only for serial workflow
        if self.NTASKS == 1:
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
            # For single variable, check against the stored sparsity value
            if self.num_vars == 1:
                expected_sparsity = self.var_sparsities[0]
            else:
                # For multi-var, this check is not applicable
                expected_sparsity = s_estim
            if not np.isclose(s_estim, expected_sparsity):
                raise ValueError(
                    f"Sparsity {s_estim} determined from number "
                    f"of coordinates differs from sparsity {expected_sparsity} "
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

        # Store as instance variable only for serial workflow
        if self.NTASKS == 1:
            self._record = record

        return record

    def _generate_multi_var_records(
        self,
        shape: list = None,
        rng: np.random.Generator = None
    ) -> dict:
        """Generate sparse record arrays for multiple variables with overlap control.

        Creates a multi-dimensional array for each variable with the specified shape,
        then randomly selects positions and assigns observation values to those points
        for each variable. Overlap is controlled by using shared and separate RNGs.

        Args:
            shape: Shape of the record arrays. If None, uses self.shape
            rng: Base random number generator. If None, uses self._rng

        Returns:
            Dictionary mapping variable names to record arrays

        Raises:
            RuntimeError: If coordinates have not been generated
        """
        if shape is None:
            if self._coordinates is None:
                raise RuntimeError(
                    "Coordinates must be generated before records. "
                    "Call _generate_coordinates() first."
                )
            shape = self.shape

        if rng is None:
            rng = self._rng

        # Initialize record arrays for all variables
        # ALL variables are now in the full num_dims space
        records = {}
        for var_idx in range(self.num_vars):
            var_name = f"record{var_idx}"
            # All records have the same full shape
            records[var_name] = np.full(shape, np.nan)

        # For overlap control, we need to carefully place observations
        if self.overlap_target == 'random' or self.num_vars == 1:
            # No overlap constraints: generate each variable independently
            records = self._generate_without_overlap(shape, records)
        else:
            # Overlap is constrained: use shared and separate RNGs for coordinates
            records = self._generate_with_overlap(shape, rng, records)

        # Store as instance variable only for serial workflow
        if self.NTASKS == 1:
            self._records = records

        # Compute actual overlap
        self._compute_actual_overlap(records, shape)

        return records

    def _generate_without_overlap(
        self,
        shape: list,
        records: dict
    ) -> dict:
        """Generate multi-variable records without overlap constraints.

        Each variable's observations are placed independently using separate RNGs.
        All variables exist in full num_dims space, with constant dimensions held fixed.

        Args:
            shape: Shape of the full coordinate space
            records: Pre-initialized dictionary of empty record arrays (all full shape)

        Returns:
            Dictionary mapping variable names to record arrays with observations
        """
        for var_idx in range(self.num_vars):
            var_name = f"record{var_idx}"
            
            # Build variable shape: varying dims use full size, constant dims use size 1
            var_shape = list(shape)
            for const_dim in self.var_constant_dims[var_idx]:
                var_shape[const_dim] = 1
            
            var_num_obs = self.var_num_obs[var_idx]
            var_total_points = np.prod(var_shape)

            # Ensure we don't exceed available points
            if var_num_obs > var_total_points:
                var_num_obs = var_total_points
                print(
                    f"WARNING: Variable {var_idx} limited to {var_total_points} "
                    f"observations (requested {self.var_num_obs[var_idx]})"
                )

            # Create a separate RNG for this variable
            var_rng = np.random.default_rng(self.seed + var_idx + 1000)

            # Generate random indices in the full space (with size 1 for constant dims)
            flat_indices = var_rng.choice(
                var_total_points,
                size=var_num_obs,
                replace=False
            )
            multi_indices = np.unravel_index(flat_indices, var_shape)

            # For constant dimensions (size=1), randomly select actual coordinate values
            full_coords = list(multi_indices)
            for const_dim in self.var_constant_dims[var_idx]:
                # Randomly select a coordinate value from available coordinates
                const_val = var_rng.integers(0, shape[const_dim])
                full_coords[const_dim] = np.full(var_num_obs, const_val)

            full_multi_indices = tuple(full_coords)

            # Generate observations
            observations = var_rng.uniform(0, 1, size=var_num_obs)

            # Assign to record (now in full dimensional space)
            records[var_name][full_multi_indices] = observations

        return records

    def _generate_with_overlap(
        self,
        shape: list,
        rng: np.random.Generator,
        records: dict
    ) -> dict:
        """Generate multi-variable records with overlap control.

        Uses a shared RNG for coordinates of overlapping observations and separate
        RNGs for non-overlapping observations to achieve the target overlap level.
        All variables exist in full num_dims space with constant dimensions held at
        randomly selected coordinate values.

        Args:
            shape: Shape of the full coordinate space
            rng: Base random number generator (unused, kept for signature compatibility)
            records: Pre-initialized dictionary of empty record arrays (all full shape)

        Returns:
            Dictionary mapping variable names to record arrays with observations
        """
        # Create shared RNG for selecting coordinates of overlapping sites/points
        # This ensures overlapping observations are at the same spatial locations
        shared_rng = np.random.default_rng(self.seed + 9999)

        # Create separate RNGs for each variable's non-overlapping observations
        # and for selecting constant dimension coordinate values
        var_rngs = [
            np.random.default_rng(self.seed + var_idx + 2000)
            for var_idx in range(self.num_vars)
        ]

        # Pre-compute variable shapes: varying dims use full size, constant dims use size 1
        var_shapes = {}
        var_total_points = {}
        var_constant_coords = {}  # Store selected constant coordinate values
        
        for var_idx in range(self.num_vars):
            var_shape = list(shape)
            constant_dims = self.var_constant_dims[var_idx]
            
            # Set constant dimensions to size 1
            for const_dim in constant_dims:
                var_shape[const_dim] = 1
            
            var_shapes[var_idx] = var_shape
            var_total_points[var_idx] = np.prod(var_shape)
            
            # Randomly select constant coordinate values for this variable
            # These will be the actual coordinate values in the full space
            const_coords = {}
            for const_dim in constant_dims:
                const_coords[const_dim] = var_rngs[var_idx].integers(0, shape[const_dim])
            var_constant_coords[var_idx] = const_coords

        # Strategy: First place observations for the variable with most observations,
        # then for each subsequent variable, place a portion at the same locations
        # (shared) and the rest at different locations (non-shared)

        # Sort variables by number of observations (descending)
        sorted_var_indices = np.argsort(self.var_num_obs)[::-1]

        # Generate observations for the first (largest) variable
        first_var_idx = sorted_var_indices[0]
        first_var_name = f"record{first_var_idx}"
        first_var_shape = var_shapes[first_var_idx]
        first_var_num_obs = self.var_num_obs[first_var_idx]
        first_var_total_points = var_total_points[first_var_idx]

        # Ensure we don't try to place more observations than points available
        if first_var_num_obs > first_var_total_points:
            first_var_num_obs = first_var_total_points
            print(
                f"WARNING: Variable {first_var_idx} has more observations "
                f"({self.var_num_obs[first_var_idx]}) than available points "
                f"({first_var_total_points}), limiting to {first_var_total_points}"
            )

        # Use shared RNG for the first variable
        first_flat_indices = shared_rng.choice(
            first_var_total_points,
            size=first_var_num_obs,
            replace=False
        )
        first_multi = np.unravel_index(first_flat_indices, first_var_shape)
        
        # Convert to full coordinates (expanding size-1 dims to actual values)
        first_full_coords = list(first_multi)
        for const_dim, const_val in var_constant_coords[first_var_idx].items():
            first_full_coords[const_dim] = np.full(first_var_num_obs, const_val)
        first_full_multi = tuple(first_full_coords)
        
        # Generate and assign observations
        first_observations = var_rngs[first_var_idx].uniform(0, 1, size=first_var_num_obs)
        records[first_var_name][first_full_multi] = first_observations

        # For remaining variables, determine overlap
        for var_idx in sorted_var_indices[1:]:
            var_name = f"record{var_idx}"
            var_shape = var_shapes[var_idx]
            var_num_obs = self.var_num_obs[var_idx]
            var_total = var_total_points[var_idx]

            # Ensure we don't try to place more observations than points available
            if var_num_obs > var_total:
                var_num_obs = var_total
                print(
                    f"WARNING: Variable {var_idx} has more observations "
                    f"({self.var_num_obs[var_idx]}) than available points "
                    f"({var_total}), limiting to {var_total}"
                )

            # Determine how many observations should overlap
            num_overlap = int(np.round(self.overlap_target * var_num_obs))
            num_non_overlap = var_num_obs - num_overlap

            # Check if overlap is possible: need at least one varying dimension in common
            first_varying_dims = set(d for d in range(self.num_dims) 
                                     if d not in self.var_constant_dims[first_var_idx])
            var_varying_dims = set(d for d in range(self.num_dims) 
                                   if d not in self.var_constant_dims[var_idx])
            shared_varying_dims = first_varying_dims.intersection(var_varying_dims)

            if len(shared_varying_dims) == 0 or num_overlap == 0:
                # No shared varying dimensions or no overlap requested
                if len(shared_varying_dims) == 0 and num_overlap > 0:
                    print(
                        f"WARNING: Variable {var_idx} shares no varying dimensions with "
                        f"variable {first_var_idx}, generating independently"
                    )
                # Generate all observations independently
                flat_indices = var_rngs[var_idx].choice(
                    var_total,
                    size=var_num_obs,
                    replace=False
                )
            else:
                # Map first variable's coordinates to this variable's space for overlap
                overlap_indices = self._map_indices_for_overlap(
                    first_flat_indices,
                    first_var_shape,
                    var_shape,
                    num_overlap,
                    shared_rng
                )

                # Generate non-overlapping indices
                all_indices = set(range(var_total))
                available_indices = list(all_indices - set(overlap_indices))
                
                if len(available_indices) >= num_non_overlap:
                    non_overlap_indices = var_rngs[var_idx].choice(
                        available_indices,
                        size=num_non_overlap,
                        replace=False
                    )
                else:
                    raise RuntimeError(
                        f"Variable {var_idx}: Cannot place {num_non_overlap} "
                        f"non-overlapping observations, only {len(available_indices)} "
                        f"positions available. This indicates a validation error."
                    )

                flat_indices = np.concatenate([overlap_indices, non_overlap_indices])

            # Convert to multi-dimensional coordinates
            multi = np.unravel_index(flat_indices, var_shape)
            
            # Convert to full coordinates (expanding size-1 dims to actual values)
            full_coords = list(multi)
            for const_dim, const_val in var_constant_coords[var_idx].items():
                full_coords[const_dim] = np.full(len(flat_indices), const_val)
            full_multi = tuple(full_coords)

            # Generate and assign observations
            actual_num_obs = len(flat_indices)
            observations = var_rngs[var_idx].uniform(0, 1, size=actual_num_obs)
            records[var_name][full_multi] = observations

        return records

    def _map_indices_for_overlap(
        self,
        source_flat_indices: np.ndarray,
        source_shape: list,
        target_shape: list,
        num_needed: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Map indices from source to target variable space for overlap.
        
        Simplified version that works with full shapes where constant dimensions have size 1.
        
        Args:
            source_flat_indices: Flat indices in source variable's space
            source_shape: Shape of source variable (with size 1 for constant dims)
            target_shape: Shape of target variable (with size 1 for constant dims)
            num_needed: Number of overlapping observations needed
            rng: Random number generator for selection
            
        Returns:
            Array of flat indices in target variable's space that correspond to
            overlapping locations in source variable's space
        """
        # Convert source flat indices to multi-dimensional
        source_multi = np.unravel_index(source_flat_indices, source_shape)
        
        # Find which source observations can be mapped to target space
        # An observation can be mapped if all its non-size-1 coordinates fit in target
        valid_source_indices = []
        corresponding_target_flat = []
        
        for i in range(len(source_flat_indices)):
            # Get coordinates for this observation
            coords = [source_multi[d][i] for d in range(len(source_multi))]
            
            # Check if this coordinate is valid in target space
            valid = True
            target_coords = []
            for d in range(len(coords)):
                if target_shape[d] == 1:
                    # Target has constant dim here, use 0
                    target_coords.append(0)
                elif source_shape[d] == 1:
                    # Source has constant dim, target varies
                    # Use a random coordinate value for target
                    target_coords.append(rng.integers(0, target_shape[d]))
                else:
                    # Both vary - use source coordinate if it fits
                    if coords[d] < target_shape[d]:
                        target_coords.append(coords[d])
                    else:
                        valid = False
                        break
            
            if valid:
                # Convert target coords to flat index
                target_flat = np.ravel_multi_index(target_coords, target_shape)
                valid_source_indices.append(i)
                corresponding_target_flat.append(target_flat)
        
        # Select from valid mappings
        if len(corresponding_target_flat) >= num_needed:
            selected_indices = rng.choice(
                len(corresponding_target_flat),
                size=num_needed,
                replace=False
            )
            return np.array([corresponding_target_flat[i] for i in selected_indices], dtype=np.int64)
        else:
            # Not enough valid mappings - return what we have
            print(
                f"WARNING: Only {len(corresponding_target_flat)} valid overlap mappings "
                f"found, needed {num_needed}. Using all available."
            )
            return np.array(corresponding_target_flat, dtype=np.int64)

    def _compute_actual_overlap(
        self,
        records: dict,
        shape: list
    ) -> None:
        """Compute and store the actual overlap achieved.

        The overlap is computed as the fraction of observations (excluding the variable
        with the most observations) that are co-located with observations from other
        variables. Since all variables are now in full dimensional space, comparison
        is straightforward using full coordinates.

        For example, with 3 variables having 100, 100, and 20 observations:
        - If all of variable 2's points match variable 1's and variable 3's points don't
          overlap with variable 1 or 2: overlap = 100/120 = 0.83
        - If only 30 points total match: overlap = 30/120 = 0.25

        Args:
            records: Dictionary of variable records (all in full dimensional space)
            shape: Shape of the full coordinate space
        """
        if self.num_vars == 1:
            self.overlap_actual = None
            return

        # For each variable, find which full coordinates have observations
        var_coords_sets = []
        for var_idx in range(self.num_vars):
            var_name = f"record{var_idx}"
            record = records[var_name]

            # Find non-NaN indices (where observations exist) in full space
            non_nan_indices = np.where(~np.isnan(record))

            # Build set of full coordinate tuples
            coords_set = set()
            for obs_idx in range(len(non_nan_indices[0])):
                # All dimensions are present now
                coord_tuple = tuple(
                    non_nan_indices[dim_idx][obs_idx]
                    for dim_idx in range(self.num_dims)
                )
                coords_set.add(coord_tuple)

            var_coords_sets.append(coords_set)

        # Compute overlap: count how many observations from other variables
        # are co-located with observations from the reference (largest) variable
        sorted_indices = np.argsort(self.var_num_obs)[::-1]
        reference_set = var_coords_sets[sorted_indices[0]]

        # Count overlaps with reference across all other variables
        overlap_count = 0
        total_other_obs = 0

        for idx in sorted_indices[1:]:
            var_set = var_coords_sets[idx]
            total_other_obs += len(var_set)

            # Since all variables are in the same full space, we need to consider
            # only the VARYING dimensions for overlap comparison
            ref_varying_dims = self.var_dims_indices[sorted_indices[0]]
            var_varying_dims = self.var_dims_indices[idx]
            shared_varying_dims = sorted(
                set(ref_varying_dims).intersection(set(var_varying_dims))
            )

            if len(shared_varying_dims) == 0:
                # No shared varying dimensions, no overlap possible
                continue

            # Project coordinates onto shared varying dimensions only
            ref_projected = set()
            for coord in reference_set:
                proj_coord = tuple(coord[d] for d in shared_varying_dims)
                ref_projected.add(proj_coord)

            var_projected = set()
            for coord in var_set:
                proj_coord = tuple(coord[d] for d in shared_varying_dims)
                var_projected.add(proj_coord)

            # Count matches
            overlap_count += len(ref_projected.intersection(var_projected))

        # Compute overlap ratio
        if total_other_obs > 0:
            self.overlap_actual = overlap_count / total_other_obs
        else:
            self.overlap_actual = 0.0

        print(f"Actual overlap achieved: {self.overlap_actual:.4f}")

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
            self.parquet_filepath,
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
        dim_rngs = {}
        for idx in range(len(task_shape)):
            if idx == self.dim_split:
                dim_rngs[idx] = task_rng
            else:
                dim_rngs[idx] = global_rng

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
        nb_digits = len(str(self.NTASKS))
        fpath = f"{self.netcdf_filepath[:-3]}_{chunk_id:0{nb_digits}d}.nc"
        self.save_to_netcdf(fpath, dataarray=dataarray, overwrite=False)
        del dataarray
        gc.collect()

        # Create and save DataFrame using generalized method
        dataframe = dd.from_pandas(
            self._create_dataframe(record=record, coordinates=coordinates)
        )

        self.save_to_parquet(
            self.parquet_tmp,
            dataframe,
            overwrite=False,
            chunk_id=chunk_id
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
            # For single variable, extract from multi-var records structure
            if hasattr(self, '_records') and self._records is not None:
                record = self._records.get('record0')
            elif self._record is not None:
                record = self._record
            else:
                raise RuntimeError("Record must be generated first")

        # Create default attributes if not provided
        if attrs is None:
            attrs = {
                "description": "Sparse observation data",
                "num_obs": self.num_obs,
                "num_dims": self.num_dims,
                "ratio_dims": self.ratio_dims.tolist() if isinstance(self.ratio_dims, np.ndarray) else self.ratio_dims,
                "sparsity": float(self.var_sparsities[0]) if hasattr(self, 'var_sparsities') else self.sparsity,
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

        # Store as instance variable only for serial workflow
        if self.NTASKS == 1:
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
            # For single variable, extract from multi-var records structure
            if hasattr(self, '_records') and self._records is not None:
                record = self._records.get('record0')
            elif self._record is not None:
                record = self._record
            else:
                raise RuntimeError("Record must be generated first")

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

        if self.NTASKS == 1:
            self._dataframe = dataframe

        return dataframe

    def _create_dataset(
        self,
        records: dict = None,
        coordinates: dict = None,
        attrs: dict = None
    ) -> xr.Dataset:
        """Create xarray Dataset from multiple variable records.

        Constructs an xarray Dataset with multiple data variables.

        Args:
            records: Dictionary mapping variable names to record arrays.
                    If None, uses self._records
            coordinates: Dictionary of coordinates. If None, uses self._coordinates
            attrs: Dictionary of attributes. If None, creates defaults.

        Returns:
            xarray Dataset containing multiple sparse observation variables

        Raises:
            RuntimeError: If required data has not been generated yet
        """
        if coordinates is None:
            if self._coordinates is None:
                raise RuntimeError("Coordinates must be generated first")
            coordinates = self._coordinates

        if records is None:
            if not hasattr(self, '_records') or self._records is None:
                raise RuntimeError("Records must be generated first")
            records = self._records

        # Create default attributes if not provided
        if attrs is None:
            attrs = {
                "description": "Multi-variable sparse observation data",
                "num_obs": self.num_obs,
                "num_dims": self.num_dims,
                "num_vars": self.num_vars,
                "ratio_dims": self.ratio_dims.tolist(),
                "var_sparsities": self.var_sparsities.tolist(),
                "var_num_obs": self.var_num_obs.tolist(),
                "seed": self.seed,
                "overlap_target": self.overlap_target if isinstance(
                    self.overlap_target, str
                ) else float(self.overlap_target),
                "overlap_actual": float(self.overlap_actual) if self.overlap_actual is not None else None
            }

        # Create data variables dict
        # All variables now have the same full dimensional shape
        data_vars = {}
        all_dim_names = [f"x{d}" for d in range(self.num_dims)]
        for var_idx in range(self.num_vars):
            var_name = f"record{var_idx}"
            varying_dims = self.var_dims_indices[var_idx]
            constant_dims = self.var_constant_dims[var_idx]

            data_vars[var_name] = xr.DataArray(
                records[var_name],
                coords=coordinates,
                dims=all_dim_names,
                attrs={
                    "variable_index": var_idx,
                    "sparsity": float(self.var_sparsities[var_idx]),
                    "num_obs": int(self.var_num_obs[var_idx]),
                    "varying_dimensions": varying_dims,
                    "constant_dimensions": constant_dims
                }
            )

        # Create Dataset
        dataset = xr.Dataset(
            data_vars=data_vars,
            coords=coordinates,
            attrs=attrs
        )

        # Store as instance variable
        if self.NTASKS == 1:
            self._dataset = dataset

        return dataset

    def _create_multi_var_dataframe(
        self,
        records: dict = None,
        coordinates: dict = None
    ) -> pd.DataFrame:
        """Create pandas DataFrame from multiple variable records.

        Constructs a DataFrame where each row represents a unique coordinate point,
        with separate columns for each variable's observations. Variables that don't
        have observations at a coordinate get pd.NA. All variables are now in full
        dimensional space.

        Args:
            records: Dictionary of variable records (all in full space). If None, uses self._records
            coordinates: Dictionary of coordinates. If None, uses self._coordinates

        Returns:
            pandas DataFrame with coordinate columns and one column per variable

        Raises:
            RuntimeError: If required data has not been generated yet
        """
        if coordinates is None:
            if self._coordinates is None:
                raise RuntimeError("Coordinates must be generated first")
            coordinates = self._coordinates

        if records is None:
            if not hasattr(self, '_records') or self._records is None:
                raise RuntimeError("Records must be generated first")
            records = self._records

        # Build a dictionary mapping coordinate tuples to variable observations
        # Key: tuple of coordinates, Value: dict of {var_name: observation_value}
        coord_to_obs = {}

        for var_idx in range(self.num_vars):
            var_name = f"record{var_idx}"
            record = records[var_name]

            # Find non-NaN points (all records are now in full dimensional space)
            non_nan_mask = ~np.isnan(record)
            non_nan_indices = np.where(non_nan_mask)

            # For each observation, build the full coordinate tuple
            for obs_idx in range(len(non_nan_indices[0])):
                # Build full coordinate tuple - all dimensions are present
                full_coords = []
                for dim_idx in range(self.num_dims):
                    coord_name = f"x{dim_idx}"
                    coord_val = coordinates[coord_name][
                        non_nan_indices[dim_idx][obs_idx]
                    ]
                    full_coords.append(coord_val)

                coord_tuple = tuple(full_coords)
                
                # Initialize dict for this coordinate if needed
                if coord_tuple not in coord_to_obs:
                    coord_to_obs[coord_tuple] = {}
                
                # Store the observation value
                coord_to_obs[coord_tuple][var_name] = record[non_nan_mask][obs_idx]

        # Convert to DataFrame
        rows = []
        for coord_tuple, var_obs in coord_to_obs.items():
            row = {}
            # Add coordinate columns - all present now
            for dim_idx in range(self.num_dims):
                coord_name = f"x{dim_idx}"
                row[coord_name] = coord_tuple[dim_idx]
            
            # Add variable columns
            for var_idx in range(self.num_vars):
                var_name = f"record{var_idx}"
                row[var_name] = var_obs.get(var_name, pd.NA)
            
            rows.append(row)

        dataframe = pd.DataFrame(rows)

        if self.NTASKS == 1:
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

        dataarray.to_netcdf(
            filepath,
            engine="h5netcdf",
            mode='w'
        )

    def save_to_parquet(self, filepath: str, dataframe: Union[pd.DataFrame, dd.DataFrame] = None, overwrite: bool = False, chunk_id: int = None) -> None:
        """Save data to Parquet file format.

        Args:
            filepath: path with filename to store parquet dataset to
            dataframe: dask or pandas dataframe to store
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
        dirpath = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        if not filename:
            filename = 'test'
        if chunk_id is not None:
            filename += f"_{chunk_id}"
        
        # Handle case where filepath has no directory component
        if not dirpath:
            dirpath = '.'

        def name_function(partition_idx: int = None):
            """Generate filename for a parquet partition."""
            return f"{filename}_{partition_idx:0{nb_digits}d}.parquet"

        write_metadata_file = True
        if self.NTASKS > 1:
            write_metadata_file = False
        ddf.to_parquet(
            dirpath,
            engine="pyarrow",
            name_function=name_function,
            append=False,
            overwrite=overwrite,
            write_metadata_file = write_metadata_file
        )


    def generate(
        self,
        netcdf_filepath: str = None,
        parquet_filepath: str = None,
        parquet_tmp: str = None,
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

            # Use unified multi-variable workflow for both single and multiple variables
            # This simplifies the codebase and ensures consistency
            self._generate_multi_var_records()
            
            if self.num_vars == 1:
                # For single variable, return DataArray and use standard DataFrame
                dataarray = self._create_dataarray()
                dataframe = self._create_dataframe()
            else:
                # For multiple variables, return Dataset with multi-var DataFrame
                dataset = self._create_dataset()
                dataframe = self._create_multi_var_dataframe()
                dataarray = dataset  # Return dataset for consistency

            # Save files if paths provided
            if netcdf_filepath is not None:
                self.save_to_netcdf(self.netcdf_filepath, dataarray=dataarray)

            if parquet_filepath is not None:
                self.save_to_parquet(self.parquet_filepath, dataframe=dataframe)

            return dataarray, dataframe

        if self.NTASKS > 1:
            self._generate_par()
            return None, None

        raise ValueError(f"NTASKS must positive, got {self.NTASKS} instead.")
