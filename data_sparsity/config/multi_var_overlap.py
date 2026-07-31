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
    def validate_overlap_value(
        overlap: Union[float, str, List[float]]
    ) -> Union[float, str, List[float]]:
        """Validate overlap parameter value.
        
        Args:
            overlap: Overlap specification (0-1, list of overlaps, or 'random')
            
        Returns:
            Validated overlap value
            
        Raises:
            TypeError: If overlap is not numeric, sequence-like, or string
            ValueError: If overlap values are outside [0, 1] or string is not 'random'
        """
        if isinstance(overlap, (float, int)):
            overlap = float(overlap)
            if not 0.0 <= overlap <= 1.0:
                raise ValueError(
                    f"overlap must be between 0 and 1, got {overlap}"
                )
            return overlap
        elif isinstance(overlap, (list, tuple, np.ndarray)):
            validated_overlap = []
            for value in overlap:
                if not isinstance(value, (float, int)):
                    raise TypeError(
                        "overlap list values must be numeric, "
                        f"got {type(value)}"
                    )
                value = float(value)
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"overlap list values must be between 0 and 1, got {value}"
                    )
                validated_overlap.append(value)
            return validated_overlap
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
    def validate_fixed_overlap_value(
        fixed_overlap: Union[bool, List[bool]],
        num_vars: int
    ) -> List[bool]:
        """Validate and normalize fixed-overlap control.

        Args:
            fixed_overlap: Boolean flag or list of flags for var1..varN-1
            num_vars: Number of variables

        Returns:
            List of booleans with length num_vars - 1

        Raises:
            TypeError: If the value is not boolean or sequence of booleans
            ValueError: If the list length is invalid
        """
        if num_vars <= 1:
            return []

        if isinstance(fixed_overlap, (bool, np.bool_)):
            return [fixed_overlap] * (num_vars - 1)

        if isinstance(fixed_overlap, (list, tuple, np.ndarray)):
            fixed_list = []
            for value in fixed_overlap:
                if not isinstance(value, (bool, np.bool_)):
                    raise TypeError(
                        "fixed_overlap list values must be booleans, "
                        f"got {type(value)}"
                    )
                fixed_list.append(value)

            expected_len = num_vars - 1
            if len(fixed_list) != expected_len:
                raise ValueError(
                    f"fixed_overlap list must contain {expected_len} values "
                    f"for var1..var{num_vars - 1}, got {len(fixed_list)}"
                )
            return fixed_list

        raise TypeError(
            f"fixed_overlap must be a bool or list of bools, got {type(fixed_overlap)}"
        )

    @staticmethod
    def validate_reference_is_largest(var_num_obs: np.ndarray) -> None:
        """Ensure the reference variable has the largest observation count.

        The generator treats var0 as the fixed reference variable, so it must
        be at least as large as every other variable. If any later variable has
        more observations, the overlap mapping becomes ambiguous.

        Args:
            var_num_obs: Array of observation counts for each variable

        Raises:
            ValueError: If var0 is smaller than any other variable
        """
        if len(var_num_obs) <= 1:
            return

        ref_obs = var_num_obs[0]
        if np.any(var_num_obs[1:] > ref_obs):
            raise ValueError(
                "var0 must be the largest variable (or tied for largest) "
                "because it is the fixed overlap reference."
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
    def adjust_observations_to_grid_space(
        shape: List[int],
        var_num_obs: np.ndarray,
        var_dims_indices: List[List[int]],
        overlap: Union[float, str, List[float]]
    ) -> np.ndarray:
        """Adjust observation counts to fit within available grid space.
        
        Each variable needs grid_points >= observations. If a variable's observation
        count exceeds its available grid space, adjust it down to the maximum feasible
        value while accounting for overlap requirements with the reference variable.
        
        For non-reference variables (index > 0):
        - Target observations = overlap * refvar_num_obs
        - If target > available grid points, reduce to grid points
        - Print warning with actual overlap achieved
        
        Args:
            shape: Full grid shape
            var_num_obs: Array of observation counts for each variable
            var_dims_indices: List of varying dimension indices per variable
            overlap: Overlap specification (0-1 or 'random')
            
        Returns:
            Adjusted observation counts (may be modified from input)
        """
        adjusted_obs = var_num_obs.copy()
        refvar_num_obs = var_num_obs[0]  # Reference variable is always first
        
        for var_idx in range(len(var_num_obs)):
            var_varying_dims = var_dims_indices[var_idx]
            
            # Compute effective grid points for this variable
            if len(var_varying_dims) == 0 or len(var_varying_dims) == len(shape):
                # All dimensions vary
                var_grid_points = int(np.prod(shape))
            else:
                # Only count grid points in varying dimensions
                var_grid_points = int(np.prod([shape[d] for d in var_varying_dims]))
            
            var_obs = var_num_obs[var_idx]
            
            if var_grid_points < var_obs:
                # For non-reference variables, consider overlap
                if var_idx > 0 and isinstance(overlap, (float, int, list)):
                    if isinstance(overlap, list):
                        overlap_target = overlap[var_idx - 1]
                    else:
                        overlap_target = float(overlap)

                    # Target observations based on overlap with reference variable
                    target_obs = int(np.round(overlap_target * refvar_num_obs))
                    
                    if target_obs > var_grid_points:
                        # Even overlap target exceeds grid space
                        max_feasible_obs = var_grid_points
                        actual_overlap = max_feasible_obs / refvar_num_obs
                        overlap_display = overlap_target if isinstance(
                            overlap_target, (int, float)
                        ) else "list"
                        
                        print(
                            f"WARNING: Variable {var_idx} requested {var_obs} observations "
                            f"with target overlap {overlap_display} ({target_obs} obs), "
                            f"but only has {var_grid_points} grid points "
                            f"(varying dims: {var_varying_dims}). "
                            f"Reducing to {max_feasible_obs} observations. "
                            f"Actual overlap for this variable: {actual_overlap:.4f}"
                        )
                    else:
                        # Use overlap-based target
                        max_feasible_obs = min(target_obs, var_grid_points)
                        actual_overlap = max_feasible_obs / refvar_num_obs
                        
                        print(
                            f"WARNING: Variable {var_idx} requested {var_obs} observations "
                            f"but only has {var_grid_points} grid points "
                            f"(varying dims: {var_varying_dims}). "
                            f"Reducing to {max_feasible_obs} observations based on "
                            f"overlap {overlap_target:.4f}. "
                            f"Actual overlap for this variable: {actual_overlap:.4f}"
                        )
                    
                    adjusted_obs[var_idx] = max_feasible_obs
                else:
                    # Reference variable or random overlap - just use grid limit
                    max_feasible_obs = var_grid_points
                    
                    print(
                        f"WARNING: Variable {var_idx} requested {var_obs} observations "
                        f"but only has {var_grid_points} grid points "
                        f"(varying dims: {var_varying_dims}). "
                        f"Reducing to {max_feasible_obs} observations."
                    )
                    
                    adjusted_obs[var_idx] = max_feasible_obs
        
        return adjusted_obs

    @staticmethod
    def setup_from_parameter(
        overlap: Union[float, str, List[float]],
        fixed_overlap: Union[bool, List[bool]],
        num_vars: int,
        shape: List[int],
        var_num_obs: np.ndarray,
        var_dims_indices: List[List[int]]
    ) -> tuple[Union[float, str, List[float]], np.ndarray, List[bool]]:
        """Setup overlap configuration from parameter.
        
        Main entry point that validates overlap parameter and adjusts observation
        counts if variables don't have sufficient grid space.
        
        Args:
            overlap: Overlap specification (0-1 or 'random')
            fixed_overlap: Fixed overlap control (bool or list of bools)
            num_vars: Number of variables
            shape: Full grid shape (size per dimension)
            var_num_obs: Array of observation counts for each variable
            var_dims_indices: List of varying dimension indices per variable
            
        Returns:
            Tuple of (validated overlap value, adjusted observation counts,
            normalized fixed-overlap flags)
            
        Raises:
            TypeError: If overlap is not a valid type
            ValueError: If overlap is infeasible
        """
        overlap = MultiVarOverlapConfig.validate_overlap_value(overlap)
        fixed_overlap = MultiVarOverlapConfig.validate_fixed_overlap_value(
            fixed_overlap, num_vars
        )

        if isinstance(overlap, list) and num_vars > 1:
            expected_len = num_vars - 1
            if len(overlap) != expected_len:
                raise ValueError(
                    f"overlap list must contain {expected_len} values "
                    f"for var1..var{num_vars - 1}, got {len(overlap)}"
                )
        elif isinstance(overlap, list) and num_vars == 1 and len(overlap) != 0:
            raise ValueError(
                "overlap list must be empty when num_vars == 1"
            )
        
        adjusted_obs = var_num_obs.copy()
        
        if num_vars > 1:
            MultiVarOverlapConfig.validate_reference_is_largest(var_num_obs)

            # Adjust observation counts to fit within grid space
            adjusted_obs = MultiVarOverlapConfig.adjust_observations_to_grid_space(
                shape, var_num_obs, var_dims_indices, overlap
            )

            if isinstance(overlap, float):
                # Compute and validate minimum overlap with adjusted observations
                total_grid_points = int(np.prod(shape))
                min_overlap = MultiVarOverlapConfig.compute_min_overlap(
                    total_grid_points, adjusted_obs
                )
                print(
                    f"Minimum feasible overlap given grid points and observations: "
                    f"{min_overlap}"
                )
                MultiVarOverlapConfig.validate_overlap_feasibility(
                    overlap, min_overlap, num_vars
                )
        
        return overlap, adjusted_obs, fixed_overlap
