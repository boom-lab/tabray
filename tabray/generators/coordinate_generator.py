"""Coordinate generation for multi-dimensional grids.

This module generates coordinate arrays for each dimension of the data grid.
"""

from typing import Dict, List
import numpy as np


class CoordinateGenerator:
    """Generator for coordinate arrays.

    This class creates coordinate values for each dimension of a
    multi-dimensional grid.
    """

    @staticmethod
    def generate_all_coords(
        shape: List[int],
        dim_rngs: Dict[int, np.random.Generator],
    ) -> Dict[str, np.ndarray]:
        """Generate coordinates for all dimensions.

        Args:
            shape: Number of coordinate points per dimension
            dim_rngs: RNG per dimension index (from ChunkUtils.generate_rngs)

        Returns:
            Dict mapping dimension name (x0, x1, ...) to sorted coordinates in [0, 1)
        """
        coordinates = {}
        for idx, n_coords in enumerate(shape):
            coordinates[f"x{idx}"] = np.sort(
                dim_rngs[idx].uniform(0.0, 1.0, size=n_coords)
            )

        return coordinates
