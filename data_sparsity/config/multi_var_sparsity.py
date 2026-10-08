"""Multi-variable density configuration.

This module handles density setup for multiple variables, including
scalar, 2-element, and full list specifications.
"""

from typing import List, Optional, Tuple, Union
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
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Create density array from 2-element list.

        density_list is [max, min]. var0 takes the maximum and one other
        variable takes the minimum, so both prescribed values are used: with
        two variables the range is exactly the two densities. Any further
        variables are drawn uniformly from [min, max].

        Args:
            density_list: List with exactly 2 elements [max, min]; the order is
                checked by ParameterValidator.validate_density_refvar
            num_vars: Number of variables
            rng: Random number generator

        Returns:
            Array of density values, one per variable
        """
        reference_density, min_density = density_list

        # var0 takes the maximum and some other variable takes the minimum, so
        # both prescribed values appear; with two variables they are exactly
        # the two densities. Any further variables are drawn from the range.
        others = np.concatenate(
            [
                [min_density],
                rng.uniform(min_density, reference_density, size=max(0, num_vars - 2)),
            ]
        )
        rng.shuffle(others)
        return np.concatenate([[reference_density], others])

    @staticmethod
    def from_full_list(density_list: List[float]) -> np.ndarray:
        """Create density array from full list.

        Args:
            density_list: List with one density per variable (length checked
                by setup_from_parameter)

        Returns:
            Array of density values, one per variable
        """
        return np.array(density_list)

    @staticmethod
    def validate_and_clip(
        var_densities: np.ndarray,
        density_min: float,
    ) -> np.ndarray:
        """Validate density values; clip var0 to the minimum if needed.

        - every density must lie in [0, 1]
        - density_min is var0's coverage bound (every coordinate of every
          axis used), so only var0 is clipped to it; the other variables
          have no coverage requirement and keep the density they asked for

        Args:
            var_densities: Array of density values, var0 first
            density_min: Minimum allowable density for var0

        Returns:
            Validated array, var0 clipped

        Raises:
            ValueError: If any density is outside [0, 1]
        """
        for j, density in enumerate(var_densities):
            if not 0.0 <= density <= 1.0:
                raise ValueError(
                    f"Variable {j} density {density} must be between 0 and 1"
                )
            if j == 0 and density < density_min:
                print(
                    f"WARNING: Variable {j} density {density} is below minimum "
                    f"{density_min}, clipping to minimum"
                )
                var_densities[j] = density_min

        return var_densities

    @staticmethod
    def compute_var_num_obs(
        var_densities: np.ndarray,
        num_obs: int,
        grid_fractions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute number of observations for each variable.

            n_i = rint(d_i / d_max * num_obs * f_i)

        - d_i is a share of the variable's OWN grid: the product of the
          dimensions it varies along
        - f_i = own grid / full grid; 1 for a variable on every dimension
        - num_obs belongs to the max-density variable, which is var0 and
          varies along every dimension, so num_obs / d_max is the full grid

        Args:
            var_densities: Array of density values
            num_obs: Total number of observations for max density variable
            grid_fractions: Own grid / full grid per variable (default: all 1)

        Returns:
            Array of observation counts, one per variable
        """
        max_density = np.max(var_densities)
        if grid_fractions is None:
            grid_fractions = np.ones(len(var_densities))
        var_num_obs = np.rint(
            (var_densities / max_density) * num_obs * np.asarray(grid_fractions),
        ).astype(int)
        var_num_obs = np.maximum(var_num_obs, 1)
        return var_num_obs

    @staticmethod
    def setup_from_parameter(
        density: Union[float, int, List, Tuple],
        num_vars: int,
        num_obs: int,
        density_min: float,
        rng: np.random.Generator,
        grid_fractions: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Setup multi-variable density configuration from parameter.

        Main entry point that processes density parameter and returns
        complete configuration.

        Args:
            density: Density specification (scalar, 2-element, or full list),
                as returned by ParameterValidator.validate_density_refvar
            num_vars: Number of variables
            num_obs: Total observations for max density variable
            density_min: Minimum allowable density
            rng: Random number generator
            grid_fractions: Own grid / full grid per variable (default: all 1)

        Returns:
            Tuple of (var_densities array, var_num_obs array)

        Raises:
            ValueError: If density list has wrong length
        """
        if isinstance(density, (float, int)):
            var_densities = MultiVarSparsityConfig.from_scalar(
                float(density),
                num_vars,
            )
        else:
            density_list = list(density)
            if len(density_list) == 2:
                var_densities = MultiVarSparsityConfig.from_two_element_list(
                    density_list,
                    num_vars,
                    rng,
                )
            elif len(density_list) == num_vars:
                var_densities = MultiVarSparsityConfig.from_full_list(density_list)
            else:
                raise ValueError(
                    f"density list must have 2 or {num_vars} elements, "
                    f"got {len(density_list)}"
                )

        var_densities = MultiVarSparsityConfig.validate_and_clip(
            var_densities,
            density_min,
        )
        var_num_obs = MultiVarSparsityConfig.compute_var_num_obs(
            var_densities,
            num_obs,
            grid_fractions,
        )

        return var_densities, var_num_obs
