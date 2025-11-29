"""Multi-variable sparsity configuration.

This module handles sparsity setup for multiple variables, including
scalar, 2-element, and full list specifications.
"""

from typing import List, Tuple, Union
import numpy as np


class MultiVarSparsityConfig:
    """Configuration manager for multi-variable sparsity.
    
    This class processes different sparsity specifications and generates
    appropriate sparsity values and observation counts for each variable.
    """

    @staticmethod
    def from_scalar(sparsity: float, num_vars: int) -> np.ndarray:
        """Create sparsity array from scalar value.
        
        All variables get the same sparsity.
        
        Args:
            sparsity: Single sparsity value
            num_vars: Number of variables
            
        Returns:
            Array of sparsity values, one per variable
        """
        return np.array([sparsity] * num_vars)

    @staticmethod
    def from_two_element_list(
        sparsity_list: List[float],
        num_vars: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Create sparsity array from 2-element list.
        
        If num_vars == 2: randomly assign which gets min and which gets max
        If num_vars > 2: assign min and max, fill rest with random values
        
        Args:
            sparsity_list: List with exactly 2 elements [min, max]
            num_vars: Number of variables
            rng: Random number generator
            
        Returns:
            Array of sparsity values, one per variable
        """
        min_spar = min(sparsity_list)
        max_spar = max(sparsity_list)
        
        if num_vars == 2:
            if rng.random() < 0.5:
                return np.array([min_spar, max_spar])
            else:
                return np.array([max_spar, min_spar])
        else:
            random_sparsities = rng.uniform(min_spar, max_spar, size=num_vars - 2)
            all_sparsities = np.concatenate([[min_spar, max_spar], random_sparsities])
            rng.shuffle(all_sparsities)
            return all_sparsities

    @staticmethod
    def from_full_list(sparsity_list: List[float], num_vars: int) -> np.ndarray:
        """Create sparsity array from full list.
        
        Args:
            sparsity_list: List with one sparsity per variable
            num_vars: Number of variables
            
        Returns:
            Array of sparsity values, one per variable
            
        Raises:
            ValueError: If list length doesn't match num_vars
        """
        if len(sparsity_list) != num_vars:
            raise ValueError(
                f"sparsity list must have {num_vars} elements, "
                f"got {len(sparsity_list)}"
            )
        return np.array(sparsity_list)

    @staticmethod
    def validate_and_clip(
        var_sparsities: np.ndarray,
        sparsity_min: float
    ) -> np.ndarray:
        """Validate sparsity values and clip to minimum if needed.
        
        Args:
            var_sparsities: Array of sparsity values
            sparsity_min: Minimum allowable sparsity
            
        Returns:
            Validated and clipped array
            
        Raises:
            ValueError: If any sparsity is outside [0, 1]
        """
        for j, spar in enumerate(var_sparsities):
            if not 0.0 <= spar <= 1.0:
                raise ValueError(
                    f"Variable {j} sparsity {spar} must be between 0 and 1"
                )
            if spar < sparsity_min:
                print(
                    f"WARNING: Variable {j} sparsity {spar} is below minimum "
                    f"{sparsity_min}, clipping to minimum"
                )
                var_sparsities[j] = sparsity_min
        
        return var_sparsities

    @staticmethod
    def compute_var_num_obs(
        var_sparsities: np.ndarray,
        num_obs: int
    ) -> np.ndarray:
        """Compute number of observations for each variable.
        
        The variable with highest sparsity gets num_obs observations,
        others are scaled proportionally.
        
        Args:
            var_sparsities: Array of sparsity values
            num_obs: Total number of observations for max sparsity variable
            
        Returns:
            Array of observation counts, one per variable
        """
        max_sparsity = np.max(var_sparsities)
        var_num_obs = np.rint((var_sparsities / max_sparsity) * num_obs).astype(int)
        var_num_obs = np.maximum(var_num_obs, 1)
        return var_num_obs

    @staticmethod
    def setup_from_parameter(
        sparsity: Union[float, int, List, Tuple],
        num_vars: int,
        num_obs: int,
        sparsity_min: float,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Setup multi-variable sparsity configuration from parameter.
        
        Main entry point that processes sparsity parameter and returns
        complete configuration.
        
        Args:
            sparsity: Sparsity specification (scalar, 2-element, or full list)
            num_vars: Number of variables
            num_obs: Total observations for max sparsity variable
            sparsity_min: Minimum allowable sparsity
            rng: Random number generator
            
        Returns:
            Tuple of (var_sparsities array, var_num_obs array)
            
        Raises:
            TypeError: If sparsity is not a valid type
            ValueError: If sparsity list has wrong length
        """
        if isinstance(sparsity, (float, int)):
            var_sparsities = MultiVarSparsityConfig.from_scalar(
                float(sparsity), num_vars
            )
        elif isinstance(sparsity, (list, tuple)):
            sparsity_list = list(sparsity)
            if len(sparsity_list) == 2:
                var_sparsities = MultiVarSparsityConfig.from_two_element_list(
                    sparsity_list, num_vars, rng
                )
            elif len(sparsity_list) == num_vars:
                var_sparsities = MultiVarSparsityConfig.from_full_list(
                    sparsity_list, num_vars
                )
            else:
                raise ValueError(
                    f"sparsity list must have 2 or {num_vars} elements, "
                    f"got {len(sparsity_list)}"
                )
        else:
            raise TypeError(
                f"sparsity must be a scalar, list, or tuple, got {type(sparsity)}"
            )
        
        var_sparsities = MultiVarSparsityConfig.validate_and_clip(
            var_sparsities, sparsity_min
        )
        var_num_obs = MultiVarSparsityConfig.compute_var_num_obs(
            var_sparsities, num_obs
        )
        
        return var_sparsities, var_num_obs
