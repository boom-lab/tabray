"""Multi-variable overlap configuration.

This module handles overlap validation and configuration for multiple
variables, including computing minimum feasible overlap.
"""

from typing import List, Union
import numpy as np


class MultiVarOverlapConfig:
    """Configuration manager for multi-variable overlap.
    
    This class validates overlap specifications and computes feasibility
    constraints based on dimensional configuration.
    """

    @staticmethod
    def validate_overlap_value(overlap: Union[float, str]) -> Union[float, str]:
        """Validate overlap parameter value.
        
        Args:
            overlap: Overlap specification (0-1 or 'random')
            
        Returns:
            Validated overlap value
            
        Raises:
            TypeError: If overlap is not float or string
            ValueError: If float overlap is outside [0, 1] or string is not 'random'
        """
        if isinstance(overlap, (float, int)):
            overlap = float(overlap)
            if not 0.0 <= overlap <= 1.0:
                raise ValueError(
                    f"overlap must be between 0 and 1, got {overlap}"
                )
            return overlap
        elif isinstance(overlap, str):
            if overlap != 'random':
                raise ValueError(
                    f"overlap string must be 'random', got '{overlap}'"
                )
            return overlap
        else:
            raise TypeError(
                f"overlap must be float or 'random', got {type(overlap)}"
            )

    @staticmethod
    def compute_min_overlap(total_grid_points: int, var_num_obs: np.ndarray) -> float:
        """Compute the minimum possible overlap given the configuration.

        The minimum overlap is determined by the available grid points and the
        number of observations. If there are fewer grid points than the sum of
        all observations, some observations must overlap.

        Args:
            total_grid_points: Total number of grid points available
            var_num_obs: Array of observation counts for each variable

        Returns:
            Minimum overlap value (0.0 to 1.0)
        """
        # Get total sites available
        total_sites = total_grid_points

        # Get observation counts for all variables
        sorted_obs = np.sort(var_num_obs)[::-1]  # Descending order

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

    @staticmethod
    def validate_overlap_feasibility(
        overlap: Union[float, str],
        min_overlap: float,
        num_vars: int
    ) -> None:
        """Validate that requested overlap is feasible.
        
        Args:
            overlap: Requested overlap value
            min_overlap: Minimum feasible overlap
            num_vars: Number of variables
            
        Raises:
            ValueError: If requested overlap is below minimum feasible
        """
        if num_vars < 2:
            return
        
        if isinstance(overlap, float):
            if overlap < min_overlap:
                raise ValueError(
                    f"Requested overlap {overlap} is below minimum feasible "
                    f"overlap {min_overlap} given the dimension configuration. "
                    f"Either increase overlap or adjust var_dims."
                )

    @staticmethod
    def setup_from_parameter(
        overlap: Union[float, str],
        num_vars: int,
        total_grid_points: int,
        var_num_obs: np.ndarray
    ) -> Union[float, str]:
        """Setup overlap configuration from parameter.
        
        Main entry point that validates overlap parameter and checks feasibility.
        
        Args:
            overlap: Overlap specification (0-1 or 'random')
            num_vars: Number of variables
            total_grid_points: Total number of grid points available
            var_num_obs: Array of observation counts for each variable
            
        Returns:
            Validated overlap value
            
        Raises:
            TypeError: If overlap is not a valid type
            ValueError: If overlap is infeasible
        """
        overlap = MultiVarOverlapConfig.validate_overlap_value(overlap)
        
        if num_vars > 1:
            min_overlap = MultiVarOverlapConfig.compute_min_overlap(
                total_grid_points, var_num_obs
            )
            print(
                f"Minimum feasible overlap given grid points and observations: "
                f"{min_overlap}"
            )
            MultiVarOverlapConfig.validate_overlap_feasibility(
                overlap, min_overlap, num_vars
            )
        
        return overlap
