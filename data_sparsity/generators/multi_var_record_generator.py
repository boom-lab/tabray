"""Multi-variable record generation with overlap control.

This module generates sparse record arrays for multiple variables
with controlled overlap between them.
"""

from typing import Dict, Iterable, List, Tuple, Union, Optional
import numpy as np
from data_sparsity.generators.record_generator import RecordGenerator
from data_sparsity.utils.chunk_utils import ChunkUtils
from data_sparsity.utils.streams import Stream, stream
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

        ``shape`` must be the GLOBAL grid shape. Passing a chunk's task shape
        draws the coordinate from ``[0, task_size)`` instead of the full extent,
        so each chunk picks a different local index and a variable that should
        sit at one coordinate ends up at one per chunk (D5).

        Args:
            shape: Full GLOBAL grid shape (never a chunk's task shape)
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
                const_source = var_constant_coord_indices[var_idx][const_dim]
                if isinstance(const_source, (int, np.integer)):
                    const_coords[const_dim] = int(const_source)
                else:
                    const_coords[const_dim] = int(
                        const_source.integers(0, shape[const_dim])
                    )
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
            
            # Determine index generation approach
            if num_vars == 1 and dim_split is not None:
                # Stratified placement. Serial asks for every stratum, a worker
                # asks for the ones in its chunk, and both get byte-identical
                # slices because a stratum depends only on (seed, stratum).
                # Peak memory is one hyperplane, not the whole grid.
                global_shape = list(lhs_shape) if lhs_shape is not None else list(shape)
                total_obs = num_obs_global if num_obs_global is not None else num_obs
                if chunk_id is not None and div_points is not None:
                    stratum_start = int(div_points[chunk_id])
                    wanted = range(stratum_start, int(div_points[chunk_id + 1]))
                else:
                    stratum_start = 0
                    wanted = None

                multi_indices, observations = (
                    RecordGenerator.generate_stratified_indices(
                        global_shape=global_shape,
                        num_obs=total_obs,
                        seed=seed,
                        split_dim=dim_split,
                        strata=wanted,
                    )
                )
                # The record array is chunk-shaped, so rebase the split axis.
                if stratum_start:
                    multi_indices = tuple(
                        values - stratum_start if dim == dim_split else values
                        for dim, values in enumerate(multi_indices)
                    )
                num_obs_actual = len(multi_indices[0])
            else:
                # Multi-variable: chunk-local hybrid LHS (see D4; stage C).
                obs_rng = (
                    stream(seed, Stream.VAR, var_idx)
                    if chunk_id is None
                    else stream(seed, Stream.VAR, var_idx, chunk_id)
                )
                multi_indices = RecordGenerator.generate_hybrid_indices(
                    shape=var_shape,
                    num_obs=num_obs,
                    rng=obs_rng
                )
                num_obs_actual = num_obs
                observations = ObservationGenerator.generate_observations(
                    num_obs_actual, obs_rng
                )

            # Expand to FULL space by filling constant dims with a single coordinate value
            full_multi_indices = MultiVarRecordGenerator._expand_to_full_coords(
                multi_indices, var_constant_coords[var_idx], num_obs_actual
            )

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
        # NOTE (D5): this is still passed the chunk's task shape, which is the
        # bug. Fixing it requires the stratum model -- a variable constant on
        # the split dimension lives in ONE stratum, so other chunks must place
        # nothing for it and the global coordinate must be mapped to a local
        # index. Handled in the D4 restructure, not as a standalone patch.
        var_constant_coords = MultiVarRecordGenerator._select_constant_coords(
            shape, var_constant_dims, var_constant_coord_indices, num_vars
        )
        
        # Create shared RNG for selecting coordinates of overlapping sites/points
        # This ensures overlapping observations are at the same spatial locations
        chunk_index = () if chunk_id is None else (chunk_id,)
        shared_rng = stream(seed, Stream.SHARED_OVERLAP, *chunk_index)
        var_rngs = [
            stream(seed, Stream.VAR, var_idx, *chunk_index)
            for var_idx in range(num_vars)
        ]

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
        
        if num_vars > 1 and dim_split is not None:
            # Stratified multi-variable placement: every variable is placed one
            # hyperplane at a time, so serial and parallel agree by construction
            # and peak memory is one hyperplane (D4, D5, A3).
            global_shape = list(lhs_shape) if lhs_shape is not None else list(shape)
            var_constant_coords = MultiVarRecordGenerator._select_constant_coords(
                global_shape, var_constant_dims, var_constant_coord_indices, num_vars
            )
            if overlap == 'random':
                targets = [None] * (num_vars - 1)
            elif isinstance(overlap, (list, tuple, np.ndarray)):
                targets = [float(value) for value in overlap]
            else:
                targets = [float(overlap)] * (num_vars - 1)
            if isinstance(fixed_overlap, bool):
                flags = [fixed_overlap] * (num_vars - 1)
            else:
                flags = list(fixed_overlap)

            if chunk_id is not None and div_points is not None:
                stratum_start = int(div_points[chunk_id])
                strata = range(stratum_start, int(div_points[chunk_id + 1]))
            else:
                stratum_start, strata = 0, None

            placed = MultiVarRecordGenerator.generate_multivar_stratified(
                global_shape=global_shape,
                num_vars=num_vars,
                var_num_obs=var_num_obs,
                var_dims_indices=var_dims_indices,
                var_constant_coords=var_constant_coords,
                overlap_targets=targets,
                fixed_overlap_flags=flags,
                seed=seed,
                split_dim=dim_split,
                strata=strata,
            )
            for var_idx in range(num_vars):
                var_name = f"var{var_idx}"
                indices, values = placed[var_name]
                if values.size == 0:
                    continue
                if stratum_start:
                    indices = tuple(
                        axis - stratum_start if dim == dim_split else axis
                        for dim, axis in enumerate(indices)
                    )
                records[var_name][indices] = values

            report = OverlapCalculator.compute_overlap_report(
                records, num_vars, num_dims, var_dims_indices
            )
            if chunk_id is None:
                OverlapCalculator.print_overlap_report(report, targets)
            return records, report["f1"]

        if overlap == 'random' or num_vars == 1:
            records = MultiVarRecordGenerator.generate_without_overlap(
                shape, records, num_vars, var_num_obs, var_constant_dims,
                var_constant_coord_indices, seed, chunk_id, max_dim_size,
                dim_split, lhs_rng, lhs_shape, num_obs_global, div_points
            )
            overlap_actual = OverlapCalculator.compute_overlap_report(
                records, num_vars, num_dims, var_dims_indices
            )["f1"]
        else:
            records = MultiVarRecordGenerator.generate_with_overlap(
                shape, records, overlap, num_vars, var_num_obs,
                var_constant_dims, var_constant_coord_indices, seed,
                chunk_id, max_dim_size, dim_split, lhs_rng, lhs_shape, 
                num_obs_global, div_points, fixed_overlap
            )
            # One metric regardless of how the target was expressed: a scalar
            # and a one-element list describe the same request.
            overlap_actual = OverlapCalculator.compute_overlap_report(
                records, num_vars, num_dims, var_dims_indices
            )["f1"]
        
        return records, overlap_actual

    @staticmethod
    def generate_multivar_stratified(
        global_shape: List[int],
        num_vars: int,
        var_num_obs: np.ndarray,
        var_dims_indices: List[List[int]],
        var_constant_coords: Dict[int, Dict[int, int]],
        overlap_targets: List[float],
        fixed_overlap_flags: List[bool],
        seed: int,
        split_dim: int,
        strata: Optional[Iterable[int]] = None,
    ) -> Dict[str, Tuple[Tuple[np.ndarray, ...], np.ndarray]]:
        """Place every variable, one stratum at a time.

        Two observations coincide only if all their indices match, including
        the split-dimension one, so overlap never spans strata. A stratum can
        therefore be generated from ``(seed, stratum)`` alone, and serial and
        parallel agree by construction.

        Overlap follows F1, the definition in ``docs/explainer_multivar.md``:

            O_i = |proj(S_0) & proj(S_i)| / |proj(S_0)|

        the fraction of the reference variable's (projected) sites that also
        carry variable i, where ``proj`` drops the dimensions variable i does
        not vary along. The per-stratum target is ``t_i * |proj_j(S_0)|``, which
        is local: the global denominator is never needed, so no worker has to
        see the whole grid. Independent per-stratum rounding costs a drift of
        order ``sqrt(num_strata)/2`` on the global numerator.

        The non-overlapping remainder is drawn from cells held by **neither**
        variable, so the achieved overlap equals the target instead of picking
        up accidental coincidences at var0's density (A3).

        Args:
            global_shape: Full grid shape
            num_vars: Number of variables
            var_num_obs: Observation count per variable
            var_dims_indices: Dimensions each variable varies along
            var_constant_coords: var_idx -> {constant dim -> GLOBAL coordinate}
            overlap_targets: Target F1 overlap for var1..varN-1
            fixed_overlap_flags: Whether each non-reference variable shares the
                reference ordering, so opted-in variables overlap each other
            seed: Base random seed
            split_dim: Dimension indexing the strata
            strata: Which strata to generate (default: all)

        Returns:
            var_name -> (multi-indices in GLOBAL space, values)
        """
        shape = [int(size) for size in global_shape]
        num_dims = len(shape)
        num_strata = shape[split_dim]

        # --- reference variable: the single-variable stratified placement ----
        ref_indices, ref_values = RecordGenerator.generate_stratified_indices(
            global_shape=shape,
            num_obs=int(var_num_obs[0]),
            seed=seed,
            split_dim=split_dim,
            strata=strata,
        )
        out = {"var0": (ref_indices, ref_values)}
        ref_split = ref_indices[split_dim]

        # --- per-variable geometry and per-stratum counts --------------------
        wanted_strata = list(range(num_strata)) if strata is None else [int(j) for j in strata]
        plane, shared, home, counts = {}, {}, {}, {}
        for var_idx in range(1, num_vars):
            dims = [d for d in var_dims_indices[var_idx] if d != split_dim]
            shared[var_idx] = dims
            plane[var_idx] = int(np.prod([shape[d] for d in dims])) if dims else 1
            if split_dim in var_dims_indices[var_idx]:
                home[var_idx] = None                      # lives in every stratum
                weights = np.full(num_strata, plane[var_idx], dtype=np.int64)
            else:
                # constant on the split dimension: the variable exists in ONE
                # stratum, so every other stratum places nothing for it (D5)
                home[var_idx] = int(var_constant_coords[var_idx][split_dim])
                weights = np.zeros(num_strata, dtype=np.int64)
                weights[home[var_idx]] = plane[var_idx]
            counts[var_idx] = ChunkUtils.apportion(
                int(var_num_obs[var_idx]), weights, weights
            )

        # --- per stratum ------------------------------------------------------
        per_var = {v: ([[] for _ in range(num_dims)], []) for v in range(1, num_vars)}
        clipped: Dict[int, list] = {}
        for stratum in wanted_strata:
            here = ref_split == stratum
            if not here.any() and all(counts[v][stratum] == 0 for v in range(1, num_vars)):
                continue
            ref_here = tuple(axis[here] for axis in ref_indices)
            shared_rng = stream(seed, Stream.SHARED_OVERLAP, stratum)
            shared_order_cache = {}

            for var_idx in range(1, num_vars):
                n_here = int(counts[var_idx][stratum])
                if n_here == 0:
                    continue
                dims = shared[var_idx]
                sizes = [shape[d] for d in dims]
                rng = stream(seed, Stream.VAR, var_idx, stratum)

                # proj_j(S_0): reference cells seen through this variable's dims
                if dims and ref_here[0].size:
                    proj = np.unique(np.ravel_multi_index(
                        tuple(ref_here[d] for d in dims), sizes
                    ))
                else:
                    proj = np.zeros(1, dtype=np.int64) if ref_here[0].size else \
                           np.empty(0, dtype=np.int64)

                # How many of this variable's cells must, may, and ideally do
                # land on the reference footprint.
                free_cells = plane[var_idx] - proj.size
                lo = max(0, n_here - free_cells)   # forced: nowhere else to go
                hi = min(n_here, proj.size)        # cannot exceed either set
                target = overlap_targets[var_idx - 1]
                if target is None:                 # overlap='random': no control
                    n_overlap = None
                else:
                    ideal = int(np.round(target * proj.size))
                    n_overlap = int(np.clip(ideal, lo, hi))
                    if n_overlap != ideal:
                        clipped.setdefault(var_idx, []).append(
                            (stratum, ideal, n_overlap, proj.size, free_cells)
                        )

                if n_overlap is None:
                    # draw from the whole cell space, ignoring the reference
                    cells = rng.choice(plane[var_idx], size=n_here, replace=False)
                    chosen = cells
                    outside = np.empty(0, dtype=np.int64)
                    n_overlap = 0
                elif n_overlap:
                    if fixed_overlap_flags[var_idx - 1]:
                        key = tuple(dims)
                        if key not in shared_order_cache:
                            shared_order_cache[key] = shared_rng.permutation(proj)
                        chosen = shared_order_cache[key][:n_overlap]
                    else:
                        chosen = rng.choice(proj, size=n_overlap, replace=False)
                else:
                    chosen = np.empty(0, dtype=np.int64)

                # remainder from cells held by NEITHER variable (A3)
                if target is not None:
                    n_free = n_here - n_overlap
                    if n_free:
                        ranks = rng.choice(free_cells, size=n_free, replace=False)
                        outside = RecordGenerator._ranks_to_local(ranks, proj)
                    else:
                        outside = np.empty(0, dtype=np.int64)
                    cells = np.concatenate([chosen, outside])
                cells = np.asarray(cells, dtype=np.int64)
                unravelled = np.unravel_index(cells, sizes) if dims else ()
                axes, values = per_var[var_idx]
                axis = 0
                for dim in range(num_dims):
                    if dim == split_dim:
                        axes[dim].append(np.full(cells.size, stratum, dtype=np.int64))
                    elif dim in dims:
                        axes[dim].append(unravelled[axis]); axis += 1
                    else:
                        axes[dim].append(np.full(
                            cells.size, var_constant_coords[var_idx][dim], dtype=np.int64
                        ))
                values.append(rng.uniform(0, 1, size=cells.size))

        for var_idx in range(1, num_vars):
            axes, values = per_var[var_idx]
            if values:
                out[f"var{var_idx}"] = (
                    tuple(np.concatenate(a) for a in axes), np.concatenate(values)
                )
            else:
                out[f"var{var_idx}"] = (
                    tuple(np.empty(0, dtype=np.int64) for _ in range(num_dims)),
                    np.empty(0, dtype=float),
                )

        # An overlap target that could not be met is worth saying out loud: it
        # means the (density, overlap) pair was over-determined for that
        # variable. A variable varying along a single dimension always lands
        # here, because the LHS guarantees the reference covers that whole axis,
        # so its density alone fixes the overlap (see docs/explainer_multivar).
        for var_idx, events in sorted(clipped.items()):
            _, first_ideal, first_got, proj_size, free = events[0]
            print(
                f"  WARNING var{var_idx}: overlap target not reachable in "
                f"{len(events)} of {len(wanted_strata)} strata (e.g. wanted "
                f"{first_ideal} of {proj_size} reference cells, used "
                f"{first_got}; {free} cells free of var0). Density takes "
                f"precedence; see the achieved overlap below."
            )
        return out
