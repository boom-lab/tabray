"""Sparsity validation and adjustment for data generation.

This module handles validation of sparsity values, bounds checking,
and consistency validation between observations and grid parameters.
"""

from typing import Union
import numpy as np


class SparsityValidator:
    """Validator for sparsity-related parameters and consistency checks.
    
    This class provides static methods to validate sparsity bounds and
    ensure consistency between number of observations, sparsity, and grid size.
    """

    @staticmethod
    def compute_min_sparsity(nb_coords_per_dim: np.ndarray) -> float:
        """Compute minimum allowable sparsity for given dimensions.
        
        The minimum sparsity is 1 / (smallest dimension size), which ensures
        at least one observation can be placed.
        
        Args:
            nb_coords_per_dim: Number of coordinates per dimension
            
        Returns:
            Minimum sparsity value
        """
        return 1.0 / np.min(nb_coords_per_dim)

    @staticmethod
    def validate_sparsity_bounds(
        sparsity: float,
        sparsity_min: float
    ) -> float:
        """Validate sparsity is within allowable bounds.
        
        Args:
            sparsity: Input sparsity value
            sparsity_min: Minimum allowable sparsity
            
        Returns:
            Validated sparsity (adjusted to min if input was 0)
            
        Raises:
            ValueError: If sparsity is below minimum and not 0
        """
        print(
            f"Minimum sparsity value for the current set of dimensions: "
            f"{sparsity_min}"
        )
        
        if sparsity < sparsity_min:
            raise ValueError(
                f"Provided sparsity value of {sparsity} is lower than "
                f"minimum value of {sparsity_min}. If you want to impose "
                "the minimum value possible, set sparsity to 0. as input."
            )
        
        return sparsity

    @staticmethod
    def validate_num_obs_consistency(
        num_obs: int,
        sparsity: float,
        nb_coords_per_dim: np.ndarray
    ) -> tuple[int, float]:
        """Validate and adjust num_obs to be consistent with sparsity and dimensions.
        
        Due to rounding, the input num_obs may not exactly match the expected
        value from sparsity * grid_size. This method adjusts num_obs and
        recomputes sparsity to maintain consistency.
        
        Args:
            num_obs: Input number of observations
            sparsity: Input sparsity value
            nb_coords_per_dim: Number of coordinates per dimension
            
        Returns:
            Tuple of (adjusted num_obs, adjusted sparsity)
            
        Raises:
            ValueError: If adjusted sparsity falls outside valid bounds
        """
        num_obs_exp = sparsity * np.prod(nb_coords_per_dim)
        
        if num_obs_exp != num_obs:
            print(
                f"Input number of observations num_obs ({num_obs}) does not "
                f"match the number of observations num_obs_exp {num_obs_exp} "
                "expected from values of sparsity and the number of elements per "
                "dimension. This can happen due to rounding operations and is not "
                "necessarily an issue, so we are enforcing num_obs to match "
                "num_obs_exp and rounding it."
            )
            num_obs = int(np.rint(num_obs_exp))
            print(f"New number of observations is {num_obs}")
            
            sparsity = num_obs / np.prod(nb_coords_per_dim)
            print(f"Actual sparsity for grid is now {sparsity}")
            
            sparsity_min = SparsityValidator.compute_min_sparsity(nb_coords_per_dim)
            if sparsity < sparsity_min or sparsity > 1:
                raise ValueError(
                    f"Sparsity value {sparsity} out of bounds "
                    f"[{sparsity_min}, 1]"
                )
        
        return num_obs, sparsity
