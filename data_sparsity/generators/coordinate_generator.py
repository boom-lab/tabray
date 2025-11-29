"""Coordinate generation for multi-dimensional grids.

This module generates coordinate arrays for each dimension of the data grid.
"""

from typing import List
import numpy as np


class CoordinateGenerator:
    """Generator for coordinate arrays.
    
    This class creates coordinate values for each dimension of a
    multi-dimensional grid.
    """

    @staticmethod
    def generate_dimension_coords(dim_size: int, rng: np.random.Generator) -> np.ndarray:
        """Generate random coordinates for a single dimension.
        
        Args:
            dim_size: Number of coordinate points in this dimension
            rng: Random number generator
            
        Returns:
            Sorted array of random coordinate values
        """
        coords = rng.uniform(0, 1, size=dim_size)
        coords.sort()
        return coords

    @staticmethod
    def generate_all_coords(
        shape: List[int],
        rng: np.random.Generator
    ) -> List[np.ndarray]:
        """Generate coordinates for all dimensions.
        
        Args:
            shape: Number of coordinate points per dimension
            rng: Random number generator
            
        Returns:
            List of coordinate arrays, one per dimension
        """
        coordinates = []
        for dim_size in shape:
            coords = CoordinateGenerator.generate_dimension_coords(dim_size, rng)
            coordinates.append(coords)
        return coordinates
