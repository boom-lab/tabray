"""Density validation and adjustment for data generation.

This module handles validation of density values, bounds checking,
and consistency validation between observations and grid parameters.
"""

import numpy as np


class SparsityValidator:
    """Validator for density-related parameters and consistency checks.

    This class provides static methods to validate density bounds and
    ensure consistency between number of observations, density, and grid size.
    """

    @staticmethod
    def compute_min_density(nb_coords_per_dim: np.ndarray) -> float:
        """Compute minimum allowable density for given dimensions.

        The minimum density is max(shape) / prod(shape): the sparsest grid in
        which every coordinate on every axis is still used at least once, an
        unused coordinate being stored without describing any data point.

        The LONGEST axis sets it, not the shortest: each observation supplies
        one coordinate per axis, so covering an axis of length L needs L
        observations. docs/explainer.md, "Minimum density, on any grid",
        derives it and shows the floor is reachable.

        Args:
            nb_coords_per_dim: Number of coordinates per dimension

        Returns:
            Minimum density value

        """
        shape = np.asarray(nb_coords_per_dim, dtype=np.int64)
        return float(shape.max()) / float(np.prod(shape, dtype=np.float64))

    @staticmethod
    def validate_density_bounds(
        density: float,
        density_min: float
    ) -> float:
        """Validate density is within allowable bounds.

        Special handling for density = 0.0: automatically adjusts to minimum.
        This allows users to request the minimum density by setting density=0.

        Args:
            density: Input density value
            density_min: Minimum allowable density

        Returns:
            Validated density (adjusted to min if input was 0)

        Raises:
            ValueError: If density is below minimum and not 0
        """
        print(
            f"Minimum density value for the current set of dimensions: "
            f"{density_min}"
        )

        # Special case: density=0 means "use minimum"
        if density == 0.0:
            print(
                f"Input density is zero, imposing minimum value: "
                f"{density_min}"
            )
            return density_min

        # Validate non-zero density is above minimum
        if density < density_min:
            raise ValueError(
                f"Provided density value of {density} is lower than "
                f"minimum value of {density_min}. If you want to impose "
                "the minimum value possible, set density to 0. as input."
            )

        return density

    @staticmethod
    def validate_num_obs_consistency(
        num_obs: int,
        density: float,
        nb_coords_per_dim: np.ndarray
    ) -> tuple[int, float]:
        """Validate and adjust num_obs to be consistent with density and dimensions.

        Due to rounding, the input num_obs may not exactly match the expected
        value from density * grid_size. This method adjusts num_obs and
        recomputes density to maintain consistency.

        Args:
            num_obs: Input number of observations
            density: Input density value
            nb_coords_per_dim: Number of coordinates per dimension

        Returns:
            Tuple of (adjusted num_obs, adjusted density)

        Raises:
            ValueError: If adjusted density falls outside valid bounds
        """
        num_obs_exp = density * np.prod(nb_coords_per_dim)

        if num_obs_exp != num_obs:
            print(
                f"Input number of observations num_obs ({num_obs}) does not "
                f"match the number of observations num_obs_exp {num_obs_exp} "
                "expected from values of density and the number of elements per "
                "dimension. This can happen due to rounding operations and is not "
                "necessarily an issue, so we are enforcing num_obs to match "
                "num_obs_exp and rounding it."
            )
            num_obs = int(np.rint(num_obs_exp))
            print(f"New number of observations is {num_obs}")

            density = num_obs / np.prod(nb_coords_per_dim)
            print(f"Actual density for grid is now {density}")

            density_min = SparsityValidator.compute_min_density(nb_coords_per_dim)
            if density < density_min or density > 1:
                raise ValueError(
                    f"Density value {density} out of bounds "
                    f"[{density_min}, 1]"
                )

        return num_obs, density
