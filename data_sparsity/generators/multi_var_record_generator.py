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
    def generate_without_overlap(
        shape: List[int],
        records: Dict[str, np.ndarray],
        num_vars: int,
        var_num_obs: np.ndarray,
        var_constant_dims: List[List[int]],
        var_constant_coord_indices: Dict,
        seed: int,
        chunk_id: Optional[int] = None,
        dim_split: Optional[int] = None,
        lhs_shape: Optional[List[int]] = None,
        num_obs_global: Optional[int] = None,
        div_points: Optional[List[int]] = None,
        layout: str = "scattered",
        padded_dim: Optional[int] = None
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
            dim_split: Dimension along which to split dataset (if parallel workflow)
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
                    layout=layout,
                    padded_dim=padded_dim,
                )
            )
            # The record array is chunk-shaped, so rebase the split axis.
            if stratum_start:
                multi_indices = tuple(
                    values - stratum_start if dim == dim_split else values
                    for dim, values in enumerate(multi_indices)
                )
            num_obs_actual = len(multi_indices[0])

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
        dim_split: Optional[int] = None,
        lhs_shape: Optional[List[int]] = None,
        num_obs_global: Optional[int] = None,
        div_points: Optional[List[int]] = None,
        fixed_overlap: Union[bool, List[bool]] = False,
        layout: str = "scattered",
        padded_dim: Optional[int] = None
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
            dim_split: Split dimension for parallel mode
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
                layout=layout,
                padded_dim=padded_dim,
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

        # Only a single variable reaches here: the branch above returns for
        # every multi-variable case, and dim_split is always set.
        records = MultiVarRecordGenerator.generate_without_overlap(
            shape, records, num_vars, var_num_obs, var_constant_dims,
            var_constant_coord_indices, seed, chunk_id,
            dim_split, lhs_shape, num_obs_global, div_points,
            layout, padded_dim
        )
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
        layout: str = "scattered",
        padded_dim: Optional[int] = None,
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
        not vary along.

        A variable that varies along every dimension has its overlap count
        decided once and apportioned across strata, so the achieved F1 is the
        requested one to within a cell. A variable that drops a dimension keeps
        a per-stratum ``round(t_i * |proj_j(S_0)|)``, which drifts by order
        ``sqrt(num_strata)/2`` on the global numerator: its ``p_j`` counts
        distinct PROJECTED cells, which depends on where the reference landed,
        and a worker holding one chunk cannot know it for strata it does not
        own. Both paths use the same rule for a given variable, so serial and
        parallel agree either way.

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
            layout=layout,
            padded_dim=padded_dim,
        )
        out = {"var0": (ref_indices, ref_values)}
        ref_split = ref_indices[split_dim]

        # --- per-variable geometry and per-stratum counts --------------------
        wanted_strata = list(range(num_strata)) if strata is None else [int(j) for j in strata]
        plane, shared, home, counts = {}, {}, {}, {}
        unreachable: Dict[int, tuple] = {}
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

        # --- how many overlapping cells each stratum gets --------------------
        # Rounding target * p_j in every stratum and adding the results up is
        # not the same as rounding the total: each stratum rounds to a whole
        # cell, and on a coarse grid one cell is a large share of the variable.
        # A 3x3 grid asking for 7 of 8 shared cells got 8, and one asking for 1
        # of 3 got 0. So the total is decided once and handed out across strata,
        # the way observation counts already are.
        #
        # Only for variables that vary along every dimension. Where a variable
        # drops one, p_j counts DISTINCT PROJECTED cells, which depends on
        # where the reference landed, and a worker holding one chunk cannot
        # know it for strata it does not own. Falling back for those keeps
        # serial and parallel identical, which matters more than the drift.
        overlap_counts = {}
        all_dims = set(range(num_dims))
        for var_idx in range(1, num_vars):
            target = overlap_targets[var_idx - 1]
            if target is None or set(var_dims_indices[var_idx]) != all_dims:
                overlap_counts[var_idx] = None
                continue
            p_all = RecordGenerator.stratum_counts(
                shape, int(var_num_obs[0]), seed, split_dim
            )
            n_all = np.asarray(counts[var_idx], dtype=np.int64)
            free_all = plane[var_idx] - p_all
            lo_all = np.maximum(0, n_all - free_all)
            hi_all = np.minimum(n_all, p_all)
            goal = int(np.round(target * p_all.sum()))
            room = hi_all - lo_all
            spare = goal - int(lo_all.sum())
            if spare <= 0:
                chosen = lo_all.copy()          # the forced minimum already exceeds it
            else:
                chosen = lo_all + ChunkUtils.apportion(
                    min(spare, int(room.sum())), room, room
                )
            overlap_counts[var_idx] = chosen
            if int(chosen.sum()) != goal:
                unreachable[var_idx] = (goal, int(chosen.sum()), int(p_all.sum()))

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
                elif overlap_counts[var_idx] is not None:
                    n_overlap = int(overlap_counts[var_idx][stratum])
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

        # An unreachable target means the (density, overlap) pair was
        # over-determined for that variable -- always so for a variable varying
        # along one dimension (docs/explainer_multivar.md).
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
