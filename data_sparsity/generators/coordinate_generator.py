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
    def generate_dimension_coords(
        dim_size: int,
        rng: np.random.Generator,
        coord_range: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """Generate random coordinates for a single dimension.
        
        Args:
            dim_size: Number of coordinate points in this dimension
            rng: Random number generator
            coord_range: Optional (min, max) range for coordinates (default: (0, 1))
            
        Returns:
            Sorted array of random coordinate values
        """
        if coord_range is None:
            coord_range = (0, 1)
        
        coords = rng.uniform(coord_range[0], coord_range[1], size=dim_size)
        coords.sort()
        return coords

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
        
        coordinates = []
        for dim_idx, dim_size in enumerate(shape):
            dim_rng = dim_rngs.get(dim_idx, rng)
            dim_range = dim_ranges.get(dim_idx, None)
            coords = CoordinateGenerator.generate_dimension_coords(
                dim_size, dim_rng, dim_range
            )
            coordinates.append(coords)
        return coordinates
