"""Multi-variable record generation with overlap control.

This module generates sparse record arrays for multiple variables
with controlled overlap between them.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
from data_sparsity.generators.record_generator import RecordGenerator
from data_sparsity.generators.observation_generator import ObservationGenerator
from data_sparsity.generators.overlap_index_mapper import OverlapIndexMapper
from data_sparsity.generators.overlap_calculator import OverlapCalculator


class MultiVarRecordGenerator:
    """Generator for multi-variable record arrays with overlap control.
    
    This class creates sparse record arrays for multiple variables,
    with control over how much observation locations overlap between variables.
    """

    @staticmethod
    def _compute_var_shapes(
        shape: List[int],
        var_constant_dims: List[List[int]],
        num_vars: int
    ) -> Dict[int, List[int]]:
        """Compute reduced shapes for each variable.
        
        Constant dimensions get size 1, varying dimensions keep full size.
        
        Args:
            shape: Full grid shape
            var_constant_dims: Constant dimensions per variable
            num_vars: Number of variables
            
        Returns:
            Dictionary mapping var_idx to reduced shape
        """
        var_shapes = {}
        for var_idx in range(num_vars):
            var_shape = list(shape)
            for const_dim in var_constant_dims[var_idx]:
                var_shape[const_dim] = 1
            var_shapes[var_idx] = var_shape
        return var_shapes

    @staticmethod
    def _select_constant_coords(
        shape: List[int],
        var_constant_dims: List[List[int]],
        var_constant_coord_indices: Dict,
        num_vars: int
    ) -> Dict[int, Dict[int, int]]:
        """Select actual coordinate values for constant dimensions.
        
        Args:
            shape: Full grid shape
            var_constant_dims: Constant dimensions per variable
            var_constant_coord_indices: Pre-seeded RNGs per variable/dimension
            num_vars: Number of variables
            
        Returns:
            Dictionary mapping var_idx -> {const_dim -> coord_value}
        """
        var_constant_coords = {}
        for var_idx in range(num_vars):
            const_coords = {}
            for const_dim in var_constant_dims[var_idx]:
                const_rng = var_constant_coord_indices[var_idx][const_dim]
                const_coords[const_dim] = int(const_rng.integers(0, shape[const_dim]))
            var_constant_coords[var_idx] = const_coords
        return var_constant_coords

    @staticmethod
    def _expand_to_full_coords(
        multi_indices: Tuple,
        var_constant_coords: Dict[int, int],
        num_obs: int
    ) -> Tuple:
        """Expand reduced-space indices to full-space coordinates.
        
        Args:
            multi_indices: Indices in reduced shape (constant dims have 1 element)
            var_constant_coords: Constant coordinate values for this variable
            num_obs: Number of observations
            
        Returns:
            Multi-indices in full space
        """
        full_coords = list(multi_indices)
        for const_dim, const_val in var_constant_coords.items():
            full_coords[const_dim] = np.full(num_obs, const_val)
        return tuple(full_coords)

    @staticmethod
    def generate_without_overlap(
        shape: List[int],
        records: Dict[str, np.ndarray],
        num_vars: int,
        var_num_obs: np.ndarray,
        var_constant_dims: List[List[int]],
        var_constant_coord_indices: Dict,
        seed: int
    ) -> Dict[str, np.ndarray]:
        """Generate multi-variable records without overlap constraints.
        
        Each variable's observations are placed independently.
        
        Args:
            shape: Full grid shape
            records: Pre-initialized empty record arrays
            num_vars: Number of variables
            var_num_obs: Observation counts per variable
            var_constant_dims: Constant dimensions per variable
            var_constant_coord_indices: Pre-seeded RNGs for constant dims
            seed: Random seed
            
        Returns:
            Dictionary of filled record arrays
        """
        var_shapes = MultiVarRecordGenerator._compute_var_shapes(
            shape, var_constant_dims, num_vars
        )
        var_constant_coords = MultiVarRecordGenerator._select_constant_coords(
            shape, var_constant_dims, var_constant_coord_indices, num_vars
        )
        
        for var_idx in range(num_vars):
            var_name = f"var{var_idx}"
            var_shape = var_shapes[var_idx]
            var_total_points = int(np.prod(var_shape))
            num_obs = min(int(var_num_obs[var_idx]), var_total_points)
            
            if num_obs < var_num_obs[var_idx]:
                print(
                    f"WARNING: Variable {var_idx} limited to {num_obs} "
                    f"observations (requested {var_num_obs[var_idx]})"
                )
            
            var_rng = np.random.default_rng(seed + var_idx + 1000)
            
            flat_indices = RecordGenerator.generate_flat_indices(
                var_total_points, num_obs, var_rng
            )
            multi_indices = RecordGenerator.convert_to_multi_indices(
                flat_indices, var_shape
            )
            full_multi_indices = MultiVarRecordGenerator._expand_to_full_coords(
                multi_indices, var_constant_coords[var_idx], num_obs
            )
            
            observations = ObservationGenerator.generate_observations(num_obs, var_rng)
            RecordGenerator.assign_observations(
                records[var_name], full_multi_indices, observations
            )
        
        return records

    @staticmethod
    def generate_with_overlap(
        shape: List[int],
        records: Dict[str, np.ndarray],
        overlap_target: float,
        num_vars: int,
        var_num_obs: np.ndarray,
        var_constant_dims: List[List[int]],
        var_constant_coord_indices: Dict,
        seed: int
    ) -> Dict[str, np.ndarray]:
        """Generate multi-variable records with controlled overlap.
        
        Uses shared and separate RNGs to achieve target overlap level.
        
        Args:
            shape: Full grid shape
            records: Pre-initialized empty record arrays
            overlap_target: Target overlap fraction (0-1)
            num_vars: Number of variables
            var_num_obs: Observation counts per variable
            var_constant_dims: Constant dimensions per variable
            var_constant_coord_indices: Pre-seeded RNGs for constant dims
            seed: Random seed
            
        Returns:
            Dictionary of filled record arrays
        """
        var_shapes = MultiVarRecordGenerator._compute_var_shapes(
            shape, var_constant_dims, num_vars
        )
        var_constant_coords = MultiVarRecordGenerator._select_constant_coords(
            shape, var_constant_dims, var_constant_coord_indices, num_vars
        )
        
        shared_rng = np.random.default_rng(seed + 9999)
        var_rngs = [
            np.random.default_rng(seed + var_idx + 2000)
            for var_idx in range(num_vars)
        ]
        
        sorted_var_indices = np.argsort(var_num_obs)[::-1]
        
        # Generate reference (largest) variable first
        refvar_idx = int(sorted_var_indices[0])
        refvar_name = f"var{refvar_idx}"
        refvar_shape = var_shapes[refvar_idx]
        refvar_num_obs = min(
            int(var_num_obs[refvar_idx]),
            int(np.prod(refvar_shape))
        )
        
        refvar_flat_indices = shared_rng.choice(
            int(np.prod(refvar_shape)),
            size=refvar_num_obs,
            replace=False
        )
        refvar_multi = np.unravel_index(refvar_flat_indices, refvar_shape)
        refvar_full_multi = MultiVarRecordGenerator._expand_to_full_coords(
            refvar_multi, var_constant_coords[refvar_idx], refvar_num_obs
        )
        
        refvar_observations = var_rngs[refvar_idx].uniform(0, 1, size=refvar_num_obs)
        records[refvar_name][refvar_full_multi] = refvar_observations
        
        # Generate other variables with overlap
        for var_idx in sorted_var_indices[1:]:
            var_idx = int(var_idx)
            var_name = f"var{var_idx}"
            var_shape = var_shapes[var_idx]
            var_total_points = int(np.prod(var_shape))
            var_obs_count = min(int(var_num_obs[var_idx]), var_total_points)
            
            num_overlap = int(np.round(overlap_target * var_obs_count))
            num_separate = var_obs_count - num_overlap
            
            # Map overlapping indices from reference variable
            if num_overlap > 0:
                overlap_target_flat = OverlapIndexMapper.map_indices_for_overlap(
                    refvar_flat_indices, refvar_shape, var_shape,
                    num_overlap, shared_rng
                )
                overlap_multi = np.unravel_index(overlap_target_flat, var_shape)
                overlap_full_multi = MultiVarRecordGenerator._expand_to_full_coords(
                    overlap_multi, var_constant_coords[var_idx], len(overlap_target_flat)
                )
                overlap_obs = var_rngs[var_idx].uniform(0, 1, size=len(overlap_target_flat))
                records[var_name][overlap_full_multi] = overlap_obs
            
            # Generate separate (non-overlapping) observations
            if num_separate > 0:
                used_flat = set(overlap_target_flat) if num_overlap > 0 else set()
                available_flat = np.array([
                    i for i in range(var_total_points) if i not in used_flat
                ])
                
                if len(available_flat) >= num_separate:
                    separate_flat = var_rngs[var_idx].choice(
                        available_flat, size=num_separate, replace=False
                    )
                    separate_multi = np.unravel_index(separate_flat, var_shape)
                    separate_full_multi = MultiVarRecordGenerator._expand_to_full_coords(
                        separate_multi, var_constant_coords[var_idx], num_separate
                    )
                    separate_obs = var_rngs[var_idx].uniform(0, 1, size=num_separate)
                    records[var_name][separate_full_multi] = separate_obs
        
        return records

    @staticmethod
    def generate(
        shape: List[int],
        overlap: Union[float, str],
        num_vars: int,
        var_num_obs: np.ndarray,
        var_dims_indices: List[List[int]],
        var_constant_dims: List[List[int]],
        var_constant_coord_indices: Dict,
        num_dims: int,
        seed: int
    ) -> tuple[Dict[str, np.ndarray], float]:
        """Generate multi-variable records with overlap control.
        
        Main entry point for multi-variable record generation.
        
        Args:
            shape: Full grid shape
            overlap: Overlap specification ('random' or 0-1)
            num_vars: Number of variables
            var_num_obs: Observation counts per variable
            var_dims_indices: Varying dimensions per variable
            var_constant_dims: Constant dimensions per variable
            var_constant_coord_indices: Pre-seeded RNGs for constant dims
            num_dims: Total number of dimensions
            seed: Random seed
            
        Returns:
            Tuple of (records dict, actual overlap achieved)
        """
        records = {}
        for var_idx in range(num_vars):
            var_name = f"var{var_idx}"
            records[var_name] = RecordGenerator.initialize_record(shape)
        
        if overlap == 'random' or num_vars == 1:
            records = MultiVarRecordGenerator.generate_without_overlap(
                shape, records, num_vars, var_num_obs, var_constant_dims,
                var_constant_coord_indices, seed
            )
        else:
            records = MultiVarRecordGenerator.generate_with_overlap(
                shape, records, float(overlap), num_vars, var_num_obs,
                var_constant_dims, var_constant_coord_indices, seed
            )
        
        overlap_actual = OverlapCalculator.compute_actual_overlap(
            records, num_vars, num_dims, var_num_obs, var_dims_indices
        )
        
        return records, overlap_actual
