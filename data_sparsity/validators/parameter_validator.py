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
    def validate_ratio_dims(
        ratio_dims: Union[int, ArrayLike],
        num_dims: int
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
    def validate_sparsity_type(
        sparsity: Union[int, float, List, Tuple]
    ) -> float:
        """Validate sparsity type and return representative value for grid calculations.

        The sparsity at the whole grid level is determined as the maximum value
        of sparsity available across variables (if sparsity is an int, it is the
        same for all variables)
        
        Args:
            sparsity: Sparsity value(s) - scalar, 2-element, or num_vars-element
            
        Returns:
            Representative sparsity value (max if list/tuple) for grid calculation
            
        Raises:
            TypeError: If sparsity is not a valid type
            ValueError: If sparsity values are not in [0, 1]

        """

        if sparsity == 0.0:
            raise ValueError(
                f"Input sparsity cannot be zero."
            )
        if isinstance(sparsity, (float, int)):
            sparsity_for_grid = float(sparsity)
        elif isinstance(sparsity, (list, tuple)):
            sparsity_for_grid = float(max(sparsity))
        else:
            raise TypeError(
                f"sparsity must be a number, list, or tuple, got {type(sparsity)}"
            )
        
        if not 0.0 <= sparsity_for_grid <= 1.0:
            raise ValueError(
                f"sparsity values must be between 0 and 1.0, "
                f"got max={sparsity_for_grid}"
            )
        
        return sparsity_for_grid
