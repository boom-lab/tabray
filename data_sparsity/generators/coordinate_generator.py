"""Coordinate generation for multi-dimensional grids.

This module generates coordinate arrays for each dimension of the data grid.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class CoordinateGenerator:
    """Generator for coordinate arrays.

    This class creates coordinate values for each dimension of a
    multi-dimensional grid.
    """

    @staticmethod
    @staticmethod
    def generate_all_coords(
        shape: List[int],
        rng: np.random.Generator,
        dim_ranges: Optional[Dict[int, Tuple[float, float]]] = None,
        dim_rngs: Optional[Dict[int, np.random.Generator]] = None
    ) -> List[np.ndarray]:
        """Generate coordinates for all dimensions.

        Args:
            shape: Number of coordinate points per dimension
            rng: Random number generator (default for all dimensions)
            dim_ranges: Optional dict mapping dimension index to (min, max) range
            dim_rngs: Optional dict mapping dimension index to specific RNG

        Returns:
            List of coordinate arrays, one per dimension
        """
        if dim_ranges is None:
            dim_ranges = {}
        if dim_rngs is None:
            dim_rngs = {}

        # Generate random coordinate arrays for each dimension
        coordinates = {}
        for idx, n_coords in enumerate(shape):
            dim_name = f"x{idx}"
            low, high = dim_ranges.get(idx, (0.0, 1.0))
            dim_rng = dim_rngs.get(idx, rng)
            coordinates[dim_name] = np.sort(dim_rng.uniform(low, high, size=n_coords))

        return coordinates
