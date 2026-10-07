"""Multi-variable overlap configuration.

This module handles overlap validation and configuration for multiple
variables, including computing minimum feasible overlap.
"""

from typing import List, Optional, Union
import numpy as np


class MultiVarOverlapConfig:
    """Configuration manager for multi-variable overlap.

    This class validates overlap specifications and computes feasibility
    constraints based on dimensional configuration.
    """

    @staticmethod
    def validate_overlap_value(
        overlap: Union[float, str, List[float]],
    ) -> Union[float, str, List[float]]:
        """Validate overlap parameter value.

            overlap_i = |proj(S_0) & proj(S_i)| / |proj(S_0)|

        where S_i is the set of sites of variable i and proj projects onto the
        dimensions var0 and variable i share.

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
                raise ValueError(f"overlap must be between 0 and 1, got {overlap}")
            return overlap
        elif isinstance(overlap, (list, tuple, np.ndarray)):
            validated_overlap = []
            for value in overlap:
                if not isinstance(value, (float, int)):
                    raise TypeError(
                        "overlap list values must be numeric, " f"got {type(value)}"
                    )
                value = float(value)
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"overlap list values must be between 0 and 1, got {value}"
                    )
                validated_overlap.append(value)
            return validated_overlap
        elif isinstance(overlap, str):
            if overlap != "random":
                raise ValueError(f"overlap string must be 'random', got '{overlap}'")
            return overlap
        else:
            raise TypeError(f"overlap must be float or 'random', got {type(overlap)}")

    @staticmethod
    def validate_fixed_overlap_value(
        fixed_overlap: Union[bool, List[bool]],
        num_vars: int,
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
    def compute_min_overlap(
        total_grid_points: int,
        var_num_obs: np.ndarray,
        var_dims_indices: Optional[List[List[int]]] = None,
    ) -> np.ndarray:
        """Compute the minimum overlap each non-reference variable can reach.

            min_overlap_i = max(0, n_0 + n_i - N) / n_0

        - n_i: observations of variable i; N: grid points
        - Positive when n_0 + n_i > N: the excess must land on var0's sites
        - Per variable: non-reference variables can share sites with each other
        - Variables on fewer dimensions get 0.0; the generator checks them per
          stratum and warns

        Args:
            total_grid_points: Total number of grid points available
            var_num_obs: Array of observation counts for each variable
            var_dims_indices: Dimensions each variable varies along. Defaults
                to every dimension for every variable.

        Returns:
            Array of minimum overlaps, one per non-reference variable
        """
        obs = np.asarray(var_num_obs)
        # var0 is the reference by definition, not whichever variable happens
        # to be largest -- validate_reference_is_largest guarantees the two
        # coincide for any accepted configuration.
        ref_obs = obs[0]
        other_obs = obs[1:]

        if np.any(other_obs == 0):
            raise ValueError(
                "Cannot compute minimum overlap: no observations in a "
                "non-reference variable"
            )

        forced = np.maximum(0, ref_obs + other_obs - total_grid_points)
        min_overlap = forced / ref_obs

        if var_dims_indices is not None:
            ref_dims = set(var_dims_indices[0])
            for var_idx in range(1, len(obs)):
                dims = var_dims_indices[var_idx]
                if len(dims) != 0 and set(dims) != ref_dims:
                    min_overlap[var_idx - 1] = 0.0

        return min_overlap

    @staticmethod
    def validate_overlap_feasibility(
        overlap: Union[float, str, List[float]],
        min_overlap: np.ndarray,
        num_vars: int,
    ) -> None:
        """Validate that each requested overlap is at least its minimum.

        Args:
            overlap: Requested overlap, one value for every variable or a list
                with one value per non-reference variable
            min_overlap: Minimum overlap per non-reference variable, from
                compute_min_overlap
            num_vars: Number of variables

        Raises:
            ValueError: If any requested overlap is below its minimum
        """
        if num_vars < 2 or isinstance(overlap, str):
            return

        targets = np.broadcast_to(
            np.asarray(overlap, dtype=float), (num_vars - 1,)
        )
        minimums = np.asarray(min_overlap, dtype=float)
        too_low = [
            f"var{idx + 1}: requested {targets[idx]}, minimum {minimums[idx]:.4f}"
            for idx in range(num_vars - 1)
            if targets[idx] < minimums[idx]
        ]
        if too_low:
            raise ValueError(
                "Requested overlap is below the minimum feasible overlap given "
                "the grid size and observation counts ("
                + "; ".join(too_low)
                + "). Either increase overlap or lower the density."
            )

    @staticmethod
    def adjust_observations_to_grid_space(
        shape: List[int],
        var_num_obs: np.ndarray,
        var_dims_indices: List[List[int]],
    ) -> np.ndarray:
        """Cap each variable's observation count at the grid points it can use.

        - Grid points of a variable: product of its varying dimension sizes
        - A larger count is reduced to that number, with a warning
        - The overlap target does not change the count

        Args:
            shape: Full grid shape
            var_num_obs: Array of observation counts for each variable
            var_dims_indices: List of varying dimension indices per variable

        Returns:
            Adjusted observation counts (may be modified from input)
        """
        adjusted_obs = var_num_obs.copy()

        for var_idx, var_varying_dims in enumerate(var_dims_indices):
            # An empty list means the variable varies along every dimension.
            if len(var_varying_dims) == 0:
                var_grid_points = int(np.prod(shape))
            else:
                var_grid_points = int(np.prod([shape[d] for d in var_varying_dims]))

            var_obs = var_num_obs[var_idx]
            if var_grid_points < var_obs:
                print(
                    f"WARNING: Variable {var_idx} requested {var_obs} observations "
                    f"but only has {var_grid_points} grid points "
                    f"(varying dims: {var_varying_dims}). "
                    f"Reducing to {var_grid_points} observations."
                )
                adjusted_obs[var_idx] = var_grid_points

        return adjusted_obs

    @staticmethod
    def setup_from_parameter(
        overlap: Union[float, str, List[float]],
        fixed_overlap: Union[bool, List[bool]],
        num_vars: int,
        shape: List[int],
        var_num_obs: np.ndarray,
        var_dims_indices: List[List[int]],
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
            fixed_overlap,
            num_vars,
        )

        if isinstance(overlap, list) and num_vars > 1:
            expected_len = num_vars - 1
            if len(overlap) != expected_len:
                raise ValueError(
                    f"overlap list must contain {expected_len} values "
                    f"for var1..var{num_vars - 1}, got {len(overlap)}"
                )
        elif isinstance(overlap, list) and num_vars == 1 and len(overlap) != 0:
            raise ValueError("overlap list must be empty when num_vars == 1")

        adjusted_obs = var_num_obs.copy()

        if num_vars > 1:
            MultiVarOverlapConfig.validate_reference_is_largest(var_num_obs)

            # Adjust observation counts to fit within grid space
            adjusted_obs = MultiVarOverlapConfig.adjust_observations_to_grid_space(
                shape,
                var_num_obs,
                var_dims_indices,
            )

            if not isinstance(overlap, str):
                # Compute and validate minimum overlap with adjusted observations
                total_grid_points = int(np.prod(shape))
                min_overlap = MultiVarOverlapConfig.compute_min_overlap(
                    total_grid_points,
                    adjusted_obs,
                    var_dims_indices,
                )
                print(
                    "Minimum feasible overlap per non-reference variable given "
                    f"grid points and observations: {np.round(min_overlap, 4).tolist()}"
                )
                MultiVarOverlapConfig.validate_overlap_feasibility(
                    overlap,
                    min_overlap,
                    num_vars,
                )

        return overlap, adjusted_obs, fixed_overlap
