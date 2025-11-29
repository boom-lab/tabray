"""Base record generation functionality.

This module provides the base class for generating sparse record arrays
with common utilities for index generation and assignment.
"""

from typing import List, Tuple
import numpy as np


class RecordGenerator:
    """Base class for record generation.
    
    This class provides common functionality for creating sparse record
    arrays, including initialization, index generation, and assignment.
    """

    @staticmethod
    def initialize_record(shape: List[int]) -> np.ndarray:
        """Initialize an empty record array filled with NaN.
        
        Args:
            shape: Shape of the record array
            
        Returns:
            Array filled with NaN values
        """
        return np.full(shape, np.nan)

    @staticmethod
    def generate_flat_indices(
        total_points: int,
        num_obs: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Generate random flat indices for observation placement.
        
        Args:
            total_points: Total number of grid points
            num_obs: Number of observations to place
            rng: Random number generator
            
        Returns:
            Array of flat indices
        """
        return rng.choice(total_points, size=num_obs, replace=False)

    @staticmethod
    def convert_to_multi_indices(
        flat_indices: np.ndarray,
        shape: List[int]
    ) -> Tuple:
        """Convert flat indices to multi-dimensional indices.
        
        Args:
            flat_indices: Flat (1D) indices
            shape: Shape of the multi-dimensional array
            
        Returns:
            Tuple of index arrays for each dimension
        """
        return np.unravel_index(flat_indices, shape)

    @staticmethod
    def assign_observations(
        record: np.ndarray,
        multi_indices: Tuple,
        observations: np.ndarray
    ) -> None:
        """Assign observation values to record at specified indices.
        
        Modifies record in-place.
        
        Args:
            record: Record array to modify
            multi_indices: Tuple of index arrays
            observations: Observation values to assign
        """
        record[multi_indices] = observations

    @staticmethod
    def validate_sparsity(
        num_obs: int,
        total_grid_points: int,
        expected_sparsity: float
    ) -> None:
        """Validate that computed sparsity matches expected value.
        
        Args:
            num_obs: Number of observations
            total_grid_points: Total grid points
            expected_sparsity: Expected sparsity value
            
        Raises:
            ValueError: If computed sparsity doesn't match expected
        """
        computed_sparsity = num_obs / total_grid_points
        if not np.isclose(computed_sparsity, expected_sparsity):
            raise ValueError(
                f"Sparsity {computed_sparsity} determined from number "
                f"of coordinates differs from expected sparsity {expected_sparsity}"
            )
