"""Overlap calculation for multi-variable datasets.

This module computes the actual overlap achieved between variables
after record generation.
"""

from typing import Dict, List, Set, Tuple, Optional
import numpy as np


class OverlapCalculator:
    """Calculator for actual overlap between variables.
    
    This class computes how much observation overlap was achieved
    between multiple variables in a dataset.
    """

    @staticmethod
    def extract_coordinate_set(
        record: np.ndarray,
        num_dims: int
    ) -> Set[Tuple[int, ...]]:
        """Extract set of indices of coordinates where observations exist.
        
        Args:
            record: Record array for one variable
            num_dims: Total number of dimensions
            
        Returns:
            Set of indices tuples of coordinates where observations are non-NaN
        """
        non_nan_indices = np.where(~np.isnan(record))
        
        coords_set = set()
        for obs_idx in range(len(non_nan_indices[0])):
            coord_tuple = tuple(
                non_nan_indices[dim_idx][obs_idx]
                for dim_idx in range(num_dims)
            )
            coords_set.add(coord_tuple)
        
        return coords_set

    @staticmethod
    def project_coordinates(
        coords_set: Set[Tuple[int, ...]],
        dimensions: List[int]
    ) -> Set[Tuple[int, ...]]:
        """Project coordinates onto specific dimensions.
        
        Args:
            coords_set: Set of full coordinate tuples
            dimensions: Dimension indices to project onto
            
        Returns:
            Set of projected coordinate tuples
        """
        projected = set()
        for coord in coords_set:
            proj_coord = tuple(coord[d] for d in dimensions)
            projected.add(proj_coord)
        return projected

    @staticmethod
    def compute_pairwise_overlap_non_normalized(
        ref_coords: Set[Tuple[int, ...]],
        var_coords: Set[Tuple[int, ...]],
        ref_varying_dims: List[int],
        var_varying_dims: List[int]
    ) -> int:
        """Compute overlap count between two variables.
        
        Args:
            ref_coords: Coordinate set for reference variable
            var_coords: Coordinate set for comparison variable
            ref_varying_dims: Varying dimensions for reference variable
            var_varying_dims: Varying dimensions for comparison variable
            
        Returns:
            Number of overlapping observations
        """
        shared_varying_dims = sorted(
            set(ref_varying_dims).intersection(set(var_varying_dims))
        )
        
        if len(shared_varying_dims) == 0:
            return 0
        
        ref_projected = OverlapCalculator.project_coordinates(
            ref_coords, shared_varying_dims
        )
        var_projected = OverlapCalculator.project_coordinates(
            var_coords, shared_varying_dims
        )

        ref_count = len(ref_projected)
        var_count = len(var_projected)
        if ref_count < var_count:
            raise ValueError(
                f"Reference variable cannot have fewer elements"
                f"than the target variable, found {ref_count} "
                f"and {var_count}, respectively."
            )
        
        return len(ref_projected.intersection(var_projected))

    @staticmethod
    def compute_actual_overlap(
        records: Dict[str, np.ndarray],
        num_vars: int,
        num_dims: int,
        var_num_obs: np.ndarray,
        var_dims_indices: List[List[int]],
        ref_var: Optional[str] = None
    ) -> float:
        """Compute actual overlap achieved across all variables.
        
        Overlap is computed as the fraction of observations (excluding the
        variable with most observations) that are co-located with observations
        from other variables.
        
        Args:
            records: Dictionary mapping variable names to record arrays
            num_vars: Number of variables
            num_dims: Total number of dimensions
            var_num_obs: Array of observation counts per variable
            var_dims_indices: List of varying dimension indices per variable
            
        Returns:
            Overlap ratio (0 to 1)
        """
        if num_vars == 1:
            return None

        if set(var_num_obs) == 1 and ref_var is None:
            raise ValueError(
                "All variable have same number of observations, "
                "but no reference variable for the overlap has been defined."
            )
        
        var_coords_sets = []
        for var_idx in range(num_vars):
            var_name = f"var{var_idx}"
            record = records[var_name]
            coords_set = OverlapCalculator.extract_coordinate_set(record, num_dims)
            var_coords_sets.append(coords_set)
        
        sorted_indices = np.argsort(var_num_obs)[::-1]
        reference_set = var_coords_sets[sorted_indices[0]]
        
        overlap_count = 0
        total_other_obs = 0
        
        for idx in sorted_indices[1:]:
            var_set = var_coords_sets[idx]
            total_other_obs += len(var_set)
            
            ref_varying_dims = var_dims_indices[sorted_indices[0]]
            var_varying_dims = var_dims_indices[idx]
            
            overlap_count += OverlapCalculator.compute_pairwise_overlap_non_normalized(
                reference_set, var_set, ref_varying_dims, var_varying_dims
            )
            print(f"overlap count at idx={idx}: {overlap_count}")
        
        if total_other_obs > 0:
            overlap_actual = overlap_count / total_other_obs
        else:
            overlap_actual = 0.0
        
        print(f"Actual overlap achieved: {overlap_actual:.4f}")
        
        return overlap_actual
