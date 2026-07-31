"""Multi-variable record generation with overlap control.

This module generates sparse record arrays for multiple variables
with controlled overlap between them.
"""

from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from data_sparsity.generators.record_generator import RecordGenerator
from data_sparsity.generators.observation_generator import ObservationGenerator
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
    def _reduced_indices_from_full_coords(
        full_indices: Tuple,
        var_constant_dims: List[int],
        num_obs: int
    ) -> Tuple:
        """Convert full-space coordinates to reduced-space coordinates."""
        reduced_coords = []
        for dim_idx, dim_values in enumerate(full_indices):
            if dim_idx in var_constant_dims:
                reduced_coords.append(np.zeros(num_obs, dtype=int))
            else:
                reduced_coords.append(np.asarray(dim_values))
        return tuple(reduced_coords)

    @staticmethod
    def generate_without_overlap(
        shape: List[int],
        records: Dict[str, np.ndarray],
        num_vars: int,
        var_num_obs: np.ndarray,
        var_constant_dims: List[List[int]],
        var_constant_coord_indices: Dict,
        seed: int,
        chunk_id: Optional[int] = None,
        max_dim_size: Optional[int] = None,
        dim_split: Optional[int] = None,
        lhs_rng: Optional[np.random.Generator] = None,
        lhs_shape: Optional[List[int]] = None,
        num_obs_global: Optional[int] = None,
        div_points: Optional[List[int]] = None
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
            chunk_id: Identifier for this chunk (if parallel workflow)
            max_dim_size: Size of largest dimension (if parallel workflow)
            dim_split: Dimension along which to split dataset (if parallel workflow)
            lhs_rng: Pre-advanced LHS RNG for parallel mode
            lhs_shape: Global shape for LHS generation in parallel mode
            num_obs_global: Global observation count for validation
            div_points: Division points for chunk filtering in parallel mode
            
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
            num_obs = int(np.rint(var_num_obs[var_idx]))
            
            # Ensure we don't exceed available points
            if num_obs < var_num_obs[var_idx]:
                print(
                    f"WARNING: Variable {var_idx} limited to {num_obs} "
                    f"observations (requested {var_num_obs[var_idx]})"
                )
            
            # Create observation RNG (separate from index RNG)
            obs_seed = seed + var_idx + 1000
            if chunk_id is not None:
                obs_seed += chunk_id*100
            obs_rng = np.random.default_rng(obs_seed)
            
            # Determine index generation approach
            if lhs_rng is not None and num_vars == 1 and chunk_id is not None and div_points is not None:
                # Parallel mode with single var: use global LHS with chunk filtering
                # This maintains strict LHS property across all chunks
                multi_indices = RecordGenerator.generate_global_lhs_indices_for_chunk(
                    global_shape=lhs_shape,
                    num_obs_global=num_obs_global,
                    rng=lhs_rng,
                    chunk_id=chunk_id,
                    div_points=div_points,
                    dim_split=dim_split
                )
                # Use actual number of filtered observations
                num_obs_actual = len(multi_indices[0])
            else:
                # Serial mode or multi-var: use chunk-local LHS
                # Note: Multi-var LHS coordination not yet implemented
                multi_indices = RecordGenerator.generate_hybrid_indices(
                    shape=var_shape,
                    num_obs=num_obs,
                    rng=obs_rng
                )
                num_obs_actual = num_obs

            # Expand to FULL space by filling constant dims with a single coordinate value
            full_multi_indices = MultiVarRecordGenerator._expand_to_full_coords(
                multi_indices, var_constant_coords[var_idx], num_obs_actual
            )
            # Generate random observation values
            observations = ObservationGenerator.generate_observations(num_obs_actual, obs_rng)

            # Assign observations to the full-space coordinates
            RecordGenerator.assign_observations(
                records[var_name], full_multi_indices, observations
            )
        
        return records

    @staticmethod
    def generate_with_overlap(
        shape: List[int],
        records: Dict[str, np.ndarray],
        overlap_target: Union[float, List[float]],
        num_vars: int,
        var_num_obs: np.ndarray,
        var_constant_dims: List[List[int]],
        var_constant_coord_indices: Dict,
        seed: int,
        chunk_id: Optional[int],
        max_dim_size: Optional[int],
        dim_split: Optional[int],
        lhs_rng: Optional[np.random.Generator] = None,
        lhs_shape: Optional[List[int]] = None,
        num_obs_global: Optional[int] = None,
        div_points: Optional[List[int]] = None,
        fixed_overlap: Union[bool, List[bool]] = False
    ) -> Dict[str, np.ndarray]:
        """Generate multi-variable records with controlled overlap.
        
        Uses a shared RNG for coordinates of overlapping observations and separate
        RNGs for non-overlapping observations to achieve the target overlap level.
        All variables exist in full num_dims space with constant dimensions held at
        randomly selected coordinate values.

        Args:
            shape: Full grid shape
            records: Pre-initialized empty record arrays
            overlap_target: Target overlap fraction (0-1) or list of targets
                for var1..varN-1
            fixed_overlap: Whether each non-reference variable should share the
                same overlap sequence as other fixed variables
            num_vars: Number of variables
            var_num_obs: Observation counts per variable
            var_constant_dims: Constant dimensions per variable
            var_constant_coord_indices: Pre-seeded RNGs for constant dims
            seed: Random seed
            
        Returns:
            Dictionary of filled record arrays
        """

        ### Phase 1: Setup
        # Pre-compute variable shapes: varying dims use full size, constant dims use size 1
        var_shapes = MultiVarRecordGenerator._compute_var_shapes(
            shape, var_constant_dims, num_vars
        )

        # Store selected constant coordinate values
        var_constant_coords = MultiVarRecordGenerator._select_constant_coords(
            shape, var_constant_dims, var_constant_coord_indices, num_vars
        )
        
        # Create shared RNG for selecting coordinates of overlapping sites/points
        # This ensures overlapping observations are at the same spatial locations
        shared_seed = seed + 9999
        if chunk_id is not None:
            shared_seed += chunk_id*10000
        shared_rng = np.random.default_rng(shared_seed)

        var_seeds = [seed + var_idx + 2000 for var_idx in range(num_vars)]
        if chunk_id is not None:
            var_seeds = [s + chunk_id*100 for s in var_seeds]
        var_rngs = [np.random.default_rng(s) for s in var_seeds]

        ### Phase 2: Reference Variable
        # Sort variables by observation count (largest is always the first, refvar)
        # Generate reference (largest) variable first using SHARED_RNG (for positions)
        refvar_idx = 0
        refvar_name = f"var{refvar_idx}"
        refvar_shape = var_shapes[refvar_idx]

        # Scale down number of points if we are in a chunk of the whole
        # variable
        # if chunk_id is not None:
        #     chunk_fraction = shape[dim_split] / max_dim_size
        # else:
        chunk_fraction = 1
        chunk_var_num_obs = [
            int(np.rint(var_num_obs[var_idx] * chunk_fraction))
            for var_idx in range(num_vars)
        ]
        refvar_num_obs = chunk_var_num_obs[refvar_idx]

        # Pick random positions
        refvar_flat_indices = shared_rng.choice(
            int(np.prod(refvar_shape)),
            size=refvar_num_obs,
            replace=False
        )
        refvar_multi = np.unravel_index(refvar_flat_indices, refvar_shape)

        # Fill constant dims
        refvar_full_multi = MultiVarRecordGenerator._expand_to_full_coords(
            refvar_multi, var_constant_coords[refvar_idx], refvar_num_obs
        )

        # Generate observation values using VAR_RNG
        refvar_observations = var_rngs[refvar_idx].uniform(0, 1, size=refvar_num_obs)
        records[refvar_name][refvar_full_multi] = refvar_observations
        
        ### Phase 3: Generate other variables with overlap
        if isinstance(overlap_target, (list, tuple, np.ndarray)):
            overlap_targets = list(overlap_target)
        else:
            overlap_targets = [float(overlap_target)] * (num_vars - 1)

        if isinstance(fixed_overlap, bool):
            fixed_overlap_flags = [fixed_overlap] * (num_vars - 1)
        else:
            fixed_overlap_flags = list(fixed_overlap)
            if len(fixed_overlap_flags) != num_vars - 1:
                raise ValueError(
                    f"fixed_overlap must contain {num_vars - 1} values"
                )

        shared_ref_order = shared_rng.permutation(refvar_num_obs)

        for var_idx in range(1, num_vars):
            var_name = f"var{var_idx}"
            var_shape = var_shapes[var_idx]
            var_total_points = int(np.prod(var_shape))
            var_obs_count = chunk_var_num_obs[var_idx]
            var_fixed_overlap = fixed_overlap_flags[var_idx - 1]

            num_overlap = int(np.round(overlap_targets[var_idx - 1] * var_obs_count))

            overlap_full_multi = None
            overlap_full_count = 0
            overlap_target_flat = np.array([], dtype=np.int64)
            if num_overlap > 0:
                compatible_mask = np.ones(refvar_num_obs, dtype=bool)
                for const_dim, const_val in var_constant_coords[var_idx].items():
                    compatible_mask &= refvar_full_multi[const_dim] == const_val

                compatible_ref_indices = np.flatnonzero(compatible_mask)
                if len(compatible_ref_indices) > 0:
                    if var_fixed_overlap:
                        compatible_ref_set = set(compatible_ref_indices)
                        selected_ref_indices = np.array([
                            ref_idx
                            for ref_idx in shared_ref_order
                            if ref_idx in compatible_ref_set
                        ], dtype=np.int64)[:num_overlap]
                    else:
                        selected_ref_indices = var_rngs[var_idx].choice(
                            compatible_ref_indices,
                            size=min(num_overlap, len(compatible_ref_indices)),
                            replace=False
                        )

                    if selected_ref_indices.size > 0:
                        overlap_full_multi = tuple(
                            dim_values[selected_ref_indices]
                            for dim_values in refvar_full_multi
                        )
                        overlap_obs = var_rngs[var_idx].uniform(
                            0, 1, size=selected_ref_indices.size
                        )
                        records[var_name][overlap_full_multi] = overlap_obs
                        overlap_full_count = selected_ref_indices.size

                        overlap_reduced_multi = MultiVarRecordGenerator._reduced_indices_from_full_coords(
                            overlap_full_multi, var_constant_dims[var_idx], overlap_full_count
                        )
                        overlap_target_flat = np.ravel_multi_index(
                            overlap_reduced_multi, var_shape
                        )

            num_separate = var_obs_count - overlap_full_count

            # Generate separate (non-overlapping) observations in reduced space
            if num_separate > 0:
                used_flat = set(overlap_target_flat) if overlap_full_count > 0 else set()
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
        overlap: Union[float, str, List[float]],
        num_vars: int,
        var_num_obs: np.ndarray,
        var_dims_indices: List[List[int]],
        var_constant_dims: List[List[int]],
        var_constant_coord_indices: Dict,
        num_dims: int,
        seed: int,
        chunk_id: Optional[int] = None,
        max_dim_size: Optional[int] = None,
        dim_split: Optional[int] = None,
        lhs_rng: Optional[np.random.Generator] = None,
        lhs_shape: Optional[List[int]] = None,
        num_obs_global: Optional[int] = None,
        div_points: Optional[List[int]] = None,
        fixed_overlap: Union[bool, List[bool]] = False
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
            chunk_id: Chunk ID for parallel mode
            max_dim_size: Maximum dimension size for parallel mode
            dim_split: Split dimension for parallel mode
            lhs_rng: Pre-advanced LHS RNG for parallel mode
            lhs_shape: Global shape for LHS generation in parallel mode
            num_obs_global: Global observation count for validation
            div_points: Division points for chunk filtering in parallel mode
            
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
                var_constant_coord_indices, seed, chunk_id, max_dim_size,
                dim_split, lhs_rng, lhs_shape, num_obs_global, div_points
            )
            overlap_actual = OverlapCalculator.compute_actual_overlap(
                records, num_vars, num_dims, var_num_obs, var_dims_indices
            )
        else:
            records = MultiVarRecordGenerator.generate_with_overlap(
                shape, records, overlap, num_vars, var_num_obs,
                var_constant_dims, var_constant_coord_indices, seed,
                chunk_id, max_dim_size, dim_split, lhs_rng, lhs_shape, 
                num_obs_global, div_points, fixed_overlap
            )
            if isinstance(overlap, list):
                overlap_actual = OverlapCalculator.compute_actual_overlaps_against_reference(
                    records, num_vars, num_dims
                )
            else:
                overlap_actual = OverlapCalculator.compute_actual_overlap(
                    records, num_vars, num_dims, var_num_obs, var_dims_indices
                )
        
        return records, overlap_actual
