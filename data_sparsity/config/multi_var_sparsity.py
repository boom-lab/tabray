"""Multi-variable density configuration.

This module handles density setup for multiple variables, including
scalar, 2-element, and full list specifications.
"""

from typing import List, Tuple, Union
import numpy as np


class MultiVarSparsityConfig:
    """Configuration manager for multi-variable density.

    This class processes the density parameter which can be:
    - A scalar: all variables get the same density
    - A 2-element list/tuple: one var gets min, one gets max, rest are random
    - A num_vars-element list/tuple: each var gets its corresponding density

    """

    @staticmethod
    def from_scalar(density: float, num_vars: int) -> np.ndarray:
        """Create density array from scalar value.

        All variables get the same density.

        Args:
            density: Single density value
            num_vars: Number of variables

        Returns:
            Array of density values, one per variable
        """
        return np.array([density] * num_vars)

    @staticmethod
    def from_two_element_list(
        density_list: List[float],
        num_vars: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Create density array from 2-element list.

        density_list is [max, min]. var0 takes the maximum and one other
        variable takes the minimum, so both prescribed values are used: with
        two variables the range is exactly the two densities. Any further
        variables are drawn uniformly from [min, max].

        Args:
            density_list: List with exactly 2 elements [min, max]
            num_vars: Number of variables
            rng: Random number generator

        Returns:
            Array of density values, one per variable
        """
        reference_density = density_list[0]
        min_density = min(density_list)
        if reference_density != max(density_list):
            raise ValueError(
                f"density[0]={reference_density} is not the largest of "
                f"{list(density_list)}. var0 is the overlap reference and must "
                "have the largest density, so list the largest value first."
            )

        # var0 takes the maximum and some other variable takes the minimum, so
        # both prescribed values appear; with two variables they are exactly
        # the two densities. Any further variables are drawn from the range.
        others = np.concatenate([
            [min_density],
            rng.uniform(min_density, reference_density, size=max(0, num_vars - 2)),
        ])
        rng.shuffle(others)
        return np.concatenate([[reference_density], others])

    @staticmethod
    def from_full_list(density_list: List[float], num_vars: int) -> np.ndarray:
        """Create density array from full list.

        Args:
            density_list: List with one density per variable
            num_vars: Number of variables

        Returns:
            Array of density values, one per variable

        Raises:
            ValueError: If list length doesn't match num_vars
        """
        if len(density_list) != num_vars:
            raise ValueError(
                f"density list must have {num_vars} elements, "
                f"got {len(density_list)}"
            )
        return np.array(density_list)

    @staticmethod
    def validate_and_clip(
        var_densities: np.ndarray,
        density_min: float
    ) -> np.ndarray:
        """Validate density values and clip to minimum if needed.

        Args:
            var_densities: Array of density values
            density_min: Minimum allowable density

        Returns:
            Validated and clipped array

        Raises:
            ValueError: If any density is outside [0, 1]
        """
        for j, density in enumerate(var_densities):
            if not 0.0 <= density <= 1.0:
                raise ValueError(
                    f"Variable {j} density {density} must be between 0 and 1"
                )
            if density < density_min:
                print(
                    f"WARNING: Variable {j} density {density} is below minimum "
                    f"{density_min}, clipping to minimum"
                )
                var_densities[j] = density_min

        return var_densities

    @staticmethod
    def compute_var_num_obs(
        var_densities: np.ndarray,
        num_obs: int
    ) -> np.ndarray:
        """Compute number of observations for each variable.

        The variable with highest density gets num_obs observations,
        others are scaled proportionally.

        Args:
            var_densities: Array of density values
            num_obs: Total number of observations for max density variable

        Returns:
            Array of observation counts, one per variable
        """
        max_density = np.max(var_densities)
        var_num_obs = np.rint((var_densities / max_density) * num_obs).astype(int)
        var_num_obs = np.maximum(var_num_obs, 1)
        return var_num_obs

    @staticmethod
    def setup_from_parameter(
        density: Union[float, int, List, Tuple],
        num_vars: int,
        num_obs: int,
        density_min: float,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Setup multi-variable density configuration from parameter.

        Main entry point that processes density parameter and returns
        complete configuration.

        Args:
            density: Density specification (scalar, 2-element, or full list)
            num_vars: Number of variables
            num_obs: Total observations for max density variable
            density_min: Minimum allowable density
            rng: Random number generator

        Returns:
            Tuple of (var_densities array, var_num_obs array)

        Raises:
            TypeError: If density is not a valid type
            ValueError: If density list has wrong length
        """
        if isinstance(density, (float, int)):
            var_densities = MultiVarSparsityConfig.from_scalar(
                float(density), num_vars
            )
        elif isinstance(density, (list, tuple)):
            density_list = list(density)
            if len(density_list) == 2:
                var_densities = MultiVarSparsityConfig.from_two_element_list(
                    density_list, num_vars, rng
                )
            elif len(density_list) == num_vars:
                var_densities = MultiVarSparsityConfig.from_full_list(
                    density_list, num_vars
                )
            else:
                raise ValueError(
                    f"density list must have 2 or {num_vars} elements, "
                    f"got {len(density_list)}"
                )
        else:
            raise TypeError(
                f"density must be a scalar, list, or tuple, got {type(density)}"
            )

        var_densities = MultiVarSparsityConfig.validate_and_clip(
            var_densities, density_min
        )
        var_num_obs = MultiVarSparsityConfig.compute_var_num_obs(
            var_densities, num_obs
        )

        return var_densities, var_num_obs
