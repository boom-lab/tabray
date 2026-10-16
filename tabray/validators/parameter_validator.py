"""Parameter validation for data generation.

This module provides validation for basic input parameters such as num_obs,
sparsity, num_dims, ratio_dims, seed, and num_vars.
"""

from typing import List, Tuple, Union
import numpy as np
from numpy.typing import ArrayLike


class ParameterValidator:
    """Validator for basic data generation parameters.

    This class provides static methods to validate individual parameters
    before they are used in data generation.

    Methods are ordered by their typical call sequence in the workflow.
    """

    @staticmethod
    def validate_num_obs(num_obs: int) -> None:
        """Validate number of observations.

        Check that num_obs to generate is int and larger than 0

        Args:
            num_obs: Number of observations to generate

        Raises:
            TypeError: If num_obs is not an integer
            ValueError: If num_obs is not positive
        """
        if not isinstance(num_obs, int):
            raise TypeError(f"num_obs must be an integer, got {type(num_obs)}")
        if num_obs <= 0:
            raise ValueError(f"num_obs must be positive, got {num_obs}")

    @staticmethod
    def validate_num_dims(num_dims: int) -> None:
        """Validate number of dimensions.

        Args:
            num_dims: Number of dimensions in coordinate space

        Raises:
            TypeError: If num_dims is not an integer
            ValueError: If num_dims is not positive
        """
        if not isinstance(num_dims, int):
            raise TypeError(f"num_dims must be an integer, got {type(num_dims)}")
        if num_dims <= 0:
            raise ValueError(f"num_dims must be positive, got {num_dims}")

    @staticmethod
    def validate_seed(seed: int) -> None:
        """Validate random seed.

        Args:
            seed: Random seed for reproducibility

        Raises:
            TypeError: If seed is not an integer
            ValueError: If seed is negative
        """
        if not isinstance(seed, int):
            raise TypeError(f"seed must be an integer, got {type(seed)}")
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")

    @staticmethod
    def validate_num_vars(num_vars: int) -> None:
        """Validate number of variables.

        Args:
            num_vars: Number of variables in dataset

        Raises:
            TypeError: If num_vars is not an integer
            ValueError: If num_vars is not positive
        """
        if not isinstance(num_vars, int):
            raise TypeError(f"num_vars must be an integer, got {type(num_vars)}")
        if num_vars <= 0:
            raise ValueError(f"num_vars must be positive, got {num_vars}")

    @staticmethod
    def validate_var_dims(
        var_dims: Union[int, List, Tuple],
        num_vars: int,
        num_dims: int,
    ) -> float:
        """Check that var_dims is valid:

        - First variable (reference variable) occupies all dimensions.
        - var_dims is an int or has as many elements as number of variables

        Args:
            var_dims: Number of dimensions for each variable
            num_vars: Number of variables in the dataset
            num_dims: Number of dimensions in the coordinate space

        Returns:
            var_dims_update: Updated value to enforce first variable to occupy all
            dimensions

        Raises:
            TypeError: If var_dims type is not admitted

        """

        if isinstance(var_dims, int):
            var_dims_update = [var_dims] * num_vars
            if num_dims > var_dims:
                var_dims_update[0] = num_dims
                print(
                    "Reference variable must occupy all dimensions. Received "
                    f"var_dims={var_dims} but num_dims={num_dims}. Assigning "
                    f"{num_dims} to reference variable and {var_dims} to all other "
                    "variables"
                )

        elif isinstance(var_dims, (list, tuple)):
            # Copy: element 0 is written below, and the caller still holds
            # the argument.
            var_dims_update = list(var_dims)
            if not var_dims_update:
                raise ValueError(
                    f"var_dims must have one entry per variable ({num_vars}), got an "
                    "empty sequence"
                )

            reference = var_dims_update[0]
            if isinstance(reference, (list, tuple)):
                # Explicit dimension indices, e.g. [[0,1,2], [1,2]]. This form
                # is what MultiVarDimensionsConfig.from_list_element accepts and
                # what the README's examples use. The reference must list every
                # dimension, so compare its sorted contents to range(num_dims).
                if sorted(reference) != list(range(num_dims)):
                    var_dims_update[0] = list(range(num_dims))
                    print(
                        "Reference variable must occupy all dimensions. Received "
                        f"var_dims[0]={list(reference)} but num_dims={num_dims}. "
                        f"Assigning {list(range(num_dims))} to reference variable."
                    )
            elif isinstance(reference, (int, np.integer)) and not isinstance(
                reference, bool
            ):
                if num_dims > reference:
                    var_dims_update[0] = num_dims
                    print(
                        "Reference variable must occupy all dimensions. Received "
                        f"var_dims[0]={reference} but num_dims={num_dims}. Assigning "
                        f"{num_dims} to reference variable."
                    )
            else:
                raise TypeError(
                    "var_dims entries must be an int (a number of dimensions) or a "
                    f"list/tuple of dimension indices, got {type(reference)}"
                )

        else:
            raise TypeError(
                f"var_dims must be an int, list or tuple, got {type(var_dims)}"
            )

        return var_dims_update

    @staticmethod
    def validate_ratio_dims(
        ratio_dims: Union[int, ArrayLike],
        num_dims: int,
    ) -> np.ndarray:
        """Validate and convert ratio_dims to numpy array.

        If ratio_dims is 1, all dimensions have the same size; if ratio_dims is
        a list, it contains the factors of the number of elements per dimension,
        and it must contain as many elements as num_dims.
        Examples:
        - num_dims = 4, ratio_dims=1: all four dimensions have the same number
          of elements
        - num_dims = 3, ratio_dims=[1,2,5]: dimension #2 has twice the points of
          dimension #1, and dimension #3 has 5-times the points of dimension #1
          (and 2.5-times of dimension #2)
        - num_dims = 5, ratio_dims=[1,3]: raises error because ratio_dims have
          fewer elements than num_dims

        Args:
            ratio_dims: Relative sizes for each dimension
            num_dims: Number of dimensions (for consistency check)

        Returns:
            Validated ratio_dims as numpy array

        Raises:
            TypeError: If ratio_dims is not a valid type
            ValueError: If ratio_dims length doesn't match num_dims

        """
        if isinstance(ratio_dims, int):
            if ratio_dims == 1:
                ratio_dims = num_dims * [1]
            else:
                raise ValueError(
                    f"If ratio_dims is an int, it must be 1, got {ratio_dims}"
                )
        elif not isinstance(ratio_dims, (list, tuple, np.ndarray)):
            raise TypeError(
                f"ratio_dims must be int, tuple, list, or numpy array, "
                f"got {type(ratio_dims)}"
            )

        ratio_dims = np.asarray(ratio_dims)

        if len(ratio_dims) != num_dims:
            raise ValueError(
                f"num_dims must match length of ratio_dims, "
                f"got {num_dims} and {len(ratio_dims)}"
            )

        return ratio_dims

    @staticmethod
    def validate_density_refvar(density: Union[List, Tuple]) -> List:
        """Validate density value when a list is provided.

        A two-element list is a ``[max, min]`` range, and var0 takes the
        maximum: its first value must be the larger one. Raises rather than
        reordering: moving the maximum into position 0 would collapse the range
        to a point and give every variable the same density.

        A full list (one density per variable) is not compared here. Each
        density is a share of the variable's own grid, so densities of
        variables on different dimensions are shares of different totals:
        a per-profile variable at 1.0 can hold far fewer values than var0 at
        0.2. That var0 holds the most observations is checked on the counts,
        by validate_var_num_obs_refvar.

        Args:
            density: Density input, scalar or sequence. Scalars pass through.

        Returns:
            A new list. The argument is never modified.

        Raises:
            ValueError: If a [max, min] range does not list the maximum first
        """

        if not isinstance(density, (tuple, list)):
            return density

        # Copy unconditionally -- a list argument must not be written through.
        density = list(density)

        if len(density) == 2 and density[0] < density[1]:
            raise ValueError(
                f"The reference variable takes density[0]={density[0]}, but the "
                f"largest density given is {density[1]}. A two-element density is "
                "a [max, min] range: var0, the overlap reference, takes the maximum "
                "and must have the largest density, so list the largest value first."
            )

        return density

    @staticmethod
    def validate_var_num_obs_refvar(var_num_obs: np.ndarray) -> None:
        """Check that var0 holds the most observations.

        var0 is the overlap reference: every overlap is a share of its sites,
        and the overlap setup assumes no variable has more.

        Args:
            var_num_obs: Observations per variable, var0 first

        Raises:
            ValueError: If a variable holds more observations than var0
        """
        var_num_obs = np.asarray(var_num_obs)
        larger = np.flatnonzero(var_num_obs > var_num_obs[0])
        if larger.size:
            raise ValueError(
                f"var0 holds {var_num_obs[0]} observations, but "
                + ", ".join(f"var{i} holds {var_num_obs[i]}" for i in larger)
                + ". var0 is the overlap reference and must hold the most "
                "observations: raise its density or lower theirs."
            )

    @staticmethod
    def validate_density_type(density: Union[int, float, List, Tuple]) -> float:
        """Validate density type and return representative value for grid calculations.

            density_i = n_i / prod(shape),   sparsity_i = 1 - density_i

        where n_i is the number of observations of variable i and prod(shape)
        the number of sites in the full grid.

        The grid is sized from var0's density: num_obs is var0's count and var0
        varies along every dimension, so prod(shape) = num_obs / density[0]. A
        scalar is every variable's density; a [max, min] range gives var0 the
        maximum, density[0].

        Args:
            density: Density value(s) - scalar, 2-element, or num_vars-element

        Returns:
            var0's density, for the grid calculation

        Raises:
            TypeError: If density is not a valid type
            ValueError: If var0's density is not in [0, 1], or is 0

        """
        if isinstance(density, (float, int)):
            density_for_grid = float(density)
        elif isinstance(density, (list, tuple)):
            density_for_grid = float(density[0])
        else:
            raise TypeError(
                f"density must be a number, list, or tuple, got {type(density)}"
            )

        values = density if isinstance(density, (list, tuple)) else [density]
        outside = [value for value in values if not 0.0 <= value <= 1.0]
        if outside:
            raise ValueError(f"density values must be between 0 and 1.0, got {outside}")

        # Grid size is num_obs / density: zero gives an infinite grid.
        if density_for_grid == 0.0:
            raise ValueError(
                "density must be greater than 0 (sparsity less than 1): the grid "
                "size is num_obs / density. The smallest density a grid allows "
                "is max(shape) / prod(shape)."
            )

        return density_for_grid
