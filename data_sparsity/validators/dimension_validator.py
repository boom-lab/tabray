"""Dimension validation and computation for data generation.

This module handles validation and computation of dimension-related parameters
such as coordinate counts, shapes, and grid points.
"""

from typing import List, Tuple
import numpy as np


class DimensionValidator:
    """Validator for dimension-related parameters and computations.
    
    This class provides static methods to compute and validate dimensions,
    coordinate counts, and grid properties.
    """

    @staticmethod
    def compute_nb_coords_dim1(
        num_obs: int,
        sparsity: float,
        ratio_dims_prod: float,
        num_dims: int
    ) -> float:
        """Compute number of coordinates in first dimension.
        
        Args:
            num_obs: Number of observations
            sparsity: Sparsity value
            ratio_dims_prod: Product of ratio_dims array
            num_dims: Number of dimensions
            
        Returns:
            Number of coordinates in first dimension (may be non-integer)
            
        Raises:
            ValueError: If computed value is less than 1
        """
        base = num_obs / (sparsity * ratio_dims_prod)
        nb_coords_dim1 = np.power(base, 1 / num_dims)
        
        if nb_coords_dim1 < 1:
            raise ValueError(
                f"Number of elements for dimension 1 must be at least 1, "
                f"got {nb_coords_dim1}"
            )
        
        return nb_coords_dim1

    @staticmethod
    def round_to_integer(nb_coords_dim1: float) -> int:
        """Round coordinates in first dimension to nearest integer.
        
        Args:
            nb_coords_dim1: Number of coordinates (may be non-integer)
            
        Returns:
            Rounded integer value
        """
        rounded = int(np.rint(nb_coords_dim1))
        print(
            f"Number of elements in the first dimension is {nb_coords_dim1}, "
            f"rounding to closest integer: {rounded}"
        )
        return rounded

    @staticmethod
    def compute_nb_coords_per_dim(
        ratio_dims: np.ndarray,
        nb_coords_dim1: int
    ) -> np.ndarray:
        """Compute number of coordinates per dimension.
        
        Args:
            ratio_dims: Relative sizes for each dimension
            nb_coords_dim1: Number of coordinates in first dimension
            
        Returns:
            Number of coordinates for each dimension
        """
        return ratio_dims * nb_coords_dim1

    @staticmethod
    def validate_min_elements_per_dim(nb_coords_per_dim: np.ndarray) -> None:
        """Validate that all dimensions have at least one element.
        
        Args:
            nb_coords_per_dim: Number of coordinates per dimension
            
        Raises:
            ValueError: If any dimension has less than one element
        """
        fewer_than_one = nb_coords_per_dim < 1
        if np.any(fewer_than_one):
            bad_idxs = np.flatnonzero(fewer_than_one)
            msgs = [
                f"Error: dimension {idx} must have at least one element, got "
                f"{nb_coords_per_dim[idx]}."
                for idx in bad_idxs
            ]
            for msg in msgs:
                print(msg)
            raise ValueError(
                "One or more dimensions do not contain at least one element."
            )

    @staticmethod
    def validate_integer_elements(nb_coords_per_dim: np.ndarray) -> np.ndarray:
        """Validate and round to ensure integer number of elements per dimension.
        
        Args:
            nb_coords_per_dim: Number of coordinates per dimension
            
        Returns:
            Rounded integer array
            
        Raises:
            ValueError: If any dimension has significantly non-integer elements
        """
        not_integers = np.logical_not(
            np.isclose(nb_coords_per_dim, np.rint(nb_coords_per_dim))
        )
        if np.any(not_integers):
            bad_idxs = np.flatnonzero(not_integers)
            msgs = [
                f"Error: dimension {idx} does not have an integer number of "
                f"elements, got {nb_coords_per_dim[idx]}"
                for idx in bad_idxs
            ]
            for msg in msgs:
                print(msg)
            raise ValueError(
                "One or more dimensions contain a decimal number of elements."
            )
        
        print(
            "All dimensions contain approximately a natural number of elements, "
            "casting and/or rounding them:"
        )
        print(f"  Old number of elements: {nb_coords_per_dim}")
        rounded = np.rint(nb_coords_per_dim).astype(int)
        print(f"  New number of elements: {rounded}")
        
        return rounded

    @staticmethod
    def compute_shape_and_grid_points(
        nb_coords_per_dim: np.ndarray
    ) -> Tuple[List[int], int]:
        """Compute shape and total grid points.
        
        Args:
            nb_coords_per_dim: Number of coordinates per dimension
            
        Returns:
            Tuple of (shape as list of ints, total grid points)
        """
        shape = [int(dim_size) for dim_size in nb_coords_per_dim]
        total_grid_points = int(np.prod(shape))
        return shape, total_grid_points
