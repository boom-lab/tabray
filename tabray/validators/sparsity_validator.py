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
    def compute_min_density(
        nb_coords_per_dim: np.ndarray,
        padded_dim: int = None,
    ) -> float:
        """Compute minimum allowable density for given dimensions.

        The minimum density is max(shape) / prod(shape): the sparsest grid in
        which every coordinate on every axis is still used at least once, an
        unused coordinate being stored without describing any data point.

        The LONGEST axis sets it: each observation supplies one coordinate per
        axis, so covering an axis of length L needs L observations. The floor
        is reachable: the LHS stage places max(shape) observations that use
        every coordinate of every axis.

        Under ``layout='padded'`` the bound is higher. Every line needs an
        observation and one line must run the full length of the padded axis,
        which is ``prod(shape)/n_padded + n_padded - 1`` observations. The
        number of lines is ``prod(shape)/n_padded`` whichever axis the strata
        run along, so the split dimension does not enter.

        Args:
            nb_coords_per_dim: Number of coordinates per dimension
            padded_dim: The padded axis, or None for a scattered layout

        Returns:
            Minimum density value

        """
        shape = np.asarray(nb_coords_per_dim, dtype=np.int64)
        total = float(np.prod(shape, dtype=np.float64))
        if padded_dim is None:
            return float(shape.max()) / total
        n_padded = float(shape[int(padded_dim)])
        return (total / n_padded + n_padded - 1.0) / total

    @staticmethod
    def validate_density_bounds(
        density: float,
        density_min: float,
    ) -> float:
        """Validate density is within allowable bounds.

        Args:
            density: Input density value
            density_min: Minimum allowable density

        Returns:
            Validated density

        Raises:
            ValueError: If density is below minimum
        """
        print(
            f"Minimum density value for the current set of dimensions: "
            f"{density_min}"
        )

        if density < density_min:
            raise ValueError(
                f"Provided density value of {density} is lower than "
                f"minimum value of {density_min}."
            )

        return density

    @staticmethod
    def validate_num_obs_consistency(
        num_obs: int,
        density: float,
        nb_coords_per_dim: np.ndarray,
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

        The adjusted density stays within [density_min, 1]: density_min * N
        is the integer max(shape), so rounding density * N cannot go below it.
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

        return num_obs, density
