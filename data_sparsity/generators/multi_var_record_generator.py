"""Multi-variable record generation with overlap control.

This module generates sparse record arrays for multiple variables
with controlled overlap between them.
"""

from typing import Dict, Iterable, List, Tuple, Union, Optional
import numpy as np
from data_sparsity.generators.record_generator import RecordGenerator
from data_sparsity.utils.chunk_utils import ChunkUtils
from data_sparsity.utils.streams import Stream, stream
from data_sparsity.generators.overlap_calculator import OverlapCalculator


class MultiVarRecordGenerator:
    """Generator for multi-variable record arrays with overlap control.

    This class creates sparse record arrays for multiple variables,
    with control over how much observation locations overlap between variables.
    """

    @staticmethod
    def _select_constant_coords(
        shape: List[int],
        var_constant_dims: List[List[int]],
        var_constant_coord_indices: Dict,
        num_vars: int,
    ) -> Dict[int, Dict[int, int]]:
        """Select actual coordinate values for constant dimensions.

        ``shape`` must be the GLOBAL grid shape. A chunk's task shape would
        draw the coordinate from ``[0, task_size)`` instead of the full extent,
        so each chunk would pick a different local index and a variable that
        should sit at one coordinate would end up at one per chunk.

        Args:
            shape: Full GLOBAL grid shape (never a chunk's task shape)
            var_constant_dims: Constant dimensions per variable
            var_constant_coord_indices: Pre-seeded RNG per variable and constant dim
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
    def generate_without_overlap(
        records: Dict[str, np.ndarray],
        var_num_obs: np.ndarray,
        seed: int,
        dim_split: int,
        shape: List[int],
        chunk_id: Optional[int] = None,
        lhs_shape: Optional[List[int]] = None,
        num_obs_global: Optional[int] = None,
        div_points: Optional[List[int]] = None,
        layout: str = "scattered",
        padded_dim: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Place the single variable, one stratum at a time.

        ``generate`` sends every multi-variable case to
        ``generate_multivar_stratified``; only var0 reaches here, varying
        along every dimension.

        Args:
            records: Pre-initialized empty record array for var0
            var_num_obs: Observation count, one entry
            seed: Random seed
            dim_split: Dimension the strata are taken along
            shape: Record shape (a chunk's shape in parallel mode)
            chunk_id: Identifier for this chunk (if parallel workflow)
            lhs_shape: Global shape for LHS generation in parallel mode
            num_obs_global: Global observation count in parallel mode
            div_points: Division points for chunk filtering in parallel mode
            layout: ``scattered`` or ``padded``
            padded_dim: The axis whose first k cells each line fills under ``padded``

        Returns:
            Dictionary with the filled var0 record
        """
        # Stratified placement. Serial asks for every stratum, a worker asks
        # for the ones in its chunk, and both get byte-identical slices because
        # a stratum depends only on (seed, stratum). Peak memory is one
        # hyperplane, not the whole grid.
        global_shape = list(lhs_shape) if lhs_shape is not None else list(shape)
        num_obs = int(np.rint(var_num_obs[0]))
        total_obs = num_obs_global if num_obs_global is not None else num_obs
        if chunk_id is not None and div_points is not None:
            stratum_start = int(div_points[chunk_id])
            wanted = range(stratum_start, int(div_points[chunk_id + 1]))
        else:
            stratum_start = 0
            wanted = None

        multi_indices, observations = RecordGenerator.generate_stratified_indices(
            global_shape=global_shape,
            num_obs=total_obs,
            seed=seed,
            split_dim=dim_split,
            strata=wanted,
            layout=layout,
            padded_dim=padded_dim,
        )
        # The record array is chunk-shaped, so rebase the split axis.
        if stratum_start:
            multi_indices = tuple(
                values - stratum_start if dim == dim_split else values
                for dim, values in enumerate(multi_indices)
            )

        RecordGenerator.assign_observations(
            records["var0"],
            multi_indices,
            observations,
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
        dim_split: int,
        chunk_id: Optional[int] = None,
        lhs_shape: Optional[List[int]] = None,
        num_obs_global: Optional[int] = None,
        div_points: Optional[List[int]] = None,
        fixed_overlap: Optional[List[bool]] = None,
        layout: str = "scattered",
        padded_dim: Optional[int] = None,
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
            dim_split: Dimension the strata are taken along
                (GenerateData._choose_split_dim)
            chunk_id: Chunk ID for parallel mode
            lhs_shape: Global shape for LHS generation in parallel mode
            num_obs_global: Global observation count for validation
            div_points: Division points for chunk filtering in parallel mode
            fixed_overlap: Per non-reference variable, whether it shares the
                reference ordering (MultiVarOverlapConfig output). Multi-variable
                only.
            layout: ``scattered`` or ``padded``
            padded_dim: The axis whose first k cells each line fills under ``padded``

        Returns:
            Tuple of (records dict, actual overlap achieved)
        """
        records = {}
        for var_idx in range(num_vars):
            var_name = f"var{var_idx}"
            records[var_name] = RecordGenerator.initialize_record(shape)

        if num_vars > 1:
            # Stratified multi-variable placement: every variable is placed one
            # hyperplane at a time, so serial and parallel agree by construction
            # and peak memory is one hyperplane.
            global_shape = list(lhs_shape) if lhs_shape is not None else list(shape)
            var_constant_coords = MultiVarRecordGenerator._select_constant_coords(
                global_shape,
                var_constant_dims,
                var_constant_coord_indices,
                num_vars,
            )
            if overlap == "random":
                targets = [None] * (num_vars - 1)
            elif isinstance(overlap, (list, tuple, np.ndarray)):
                targets = [float(value) for value in overlap]
            else:
                targets = [float(overlap)] * (num_vars - 1)
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
                records,
                num_vars,
                num_dims,
                var_dims_indices,
            )
            if chunk_id is None:
                OverlapCalculator.print_overlap_report(report, targets)
            return records, report["overlap"]

        # One variable: the multi-variable branch above returns for every
        # other case.
        records = MultiVarRecordGenerator.generate_without_overlap(
            records,
            var_num_obs,
            seed,
            dim_split,
            shape,
            chunk_id,
            lhs_shape,
            num_obs_global,
            div_points,
            layout,
            padded_dim,
        )
        overlap_actual = OverlapCalculator.compute_overlap_report(
            records,
            num_vars,
            num_dims,
            var_dims_indices,
        )["overlap"]

        return records, overlap_actual

    @staticmethod
    def _padded_cells(
        count: int,
        sizes: List[int],
        padded_axis: int,
    ) -> np.ndarray:
        """A non-reference variable's cells in one stratum; each line fills its first k cells.

        - line: one combination of the variable's stratum dims other than
          the padded one; each holds positions 0..k-1 of the padded axis
        - k per line: count apportioned with equal weights, capped at the
          padded axis length; a line may stay empty (no coverage guarantee
          for non-reference variables)

        Args:
            count: Cells to place in this stratum
            sizes: Sizes of the variable's dims within the stratum
            padded_axis: Position of the padded dim within ``sizes``

        Returns:
            Cell indices, raveled over ``sizes``
        """
        n_padded = sizes[padded_axis]
        line_sizes = [n for axis, n in enumerate(sizes) if axis != padded_axis]
        lines = int(np.prod(line_sizes)) if line_sizes else 1
        lengths = ChunkUtils.apportion(
            count,
            np.ones(lines),
            np.full(lines, n_padded, dtype=np.int64),
        )
        line_index = np.repeat(np.arange(lines, dtype=np.int64), lengths)
        positions = np.concatenate(
            [np.arange(k, dtype=np.int64) for k in lengths]
            or [np.empty(0, dtype=np.int64)],
        )
        line_axes = list(np.unravel_index(line_index, line_sizes)) if line_sizes else []
        line_axes.insert(padded_axis, positions)
        return np.ravel_multi_index(tuple(line_axes), sizes)

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

        Overlap is defined as:

            O_i = |proj(S_0) & proj(S_i)| / |proj(S_0)|

        the fraction of the reference variable's (projected) sites that also
        carry variable i, where ``proj`` drops the dimensions variable i does
        not vary along.

        A variable that varies along every dimension has its overlap count
        decided once and apportioned across strata, so the achieved overlap is the
        requested one to within a cell. A variable that drops a dimension keeps
        a per-stratum ``round(t_i * |proj_j(S_0)|)``, which drifts by order
        ``sqrt(num_strata)/2`` on the global numerator: its ``p_j`` counts
        distinct PROJECTED cells, which depends on where the reference landed,
        and a worker holding one chunk cannot know it for strata it does not
        own. Both paths use the same rule for a given variable, so serial and
        parallel agree either way.

        The non-overlapping remainder is drawn from cells held by **neither**
        variable, so the achieved overlap equals the target instead of picking
        up accidental coincidences at var0's density.

        Args:
            global_shape: Full grid shape
            num_vars: Number of variables
            var_num_obs: Observation count per variable
            var_dims_indices: Dimensions each variable varies along
            var_constant_coords: var_idx -> {constant dim -> GLOBAL coordinate}
            overlap_targets: Target overlap O_i for var1..varN-1
            fixed_overlap_flags: Whether each non-reference variable shares the
                reference ordering, so opted-in variables overlap each other
            seed: Base random seed
            split_dim: Dimension indexing the strata
            strata: Which strata to generate (default: all)
            layout: ``scattered`` or ``padded``, passed to var0's placement
            padded_dim: The axis whose first k cells each line fills under ``padded``

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
        wanted_strata = (
            list(range(num_strata)) if strata is None else [int(j) for j in strata]
        )
        # Every variable varies along split_dim (_choose_split_dim), so every
        # stratum offers each variable the same number of cells.
        plane, shared, counts = {}, {}, {}
        unreachable: Dict[int, tuple] = {}
        for var_idx in range(1, num_vars):
            dims = [d for d in var_dims_indices[var_idx] if d != split_dim]
            shared[var_idx] = dims
            plane[var_idx] = int(np.prod([shape[d] for d in dims])) if dims else 1
            counts[var_idx] = ChunkUtils.apportion(
                int(var_num_obs[var_idx]),
                np.full(num_strata, plane[var_idx], dtype=np.int64),
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
                shape,
                int(var_num_obs[0]),
                seed,
                split_dim,
            )
            n_all = np.asarray(counts[var_idx], dtype=np.int64)
            free_all = plane[var_idx] - p_all
            lo_all = np.maximum(0, n_all - free_all)
            hi_all = np.minimum(n_all, p_all)
            goal = int(np.round(target * p_all.sum()))
            room = hi_all - lo_all
            spare = goal - int(lo_all.sum())
            if spare <= 0:
                chosen = lo_all.copy()  # the forced minimum already exceeds it
            else:
                chosen = lo_all + ChunkUtils.apportion(
                    min(spare, int(room.sum())),
                    room,
                )
            overlap_counts[var_idx] = chosen
            if int(chosen.sum()) != goal:
                unreachable[var_idx] = (goal, int(chosen.sum()), int(p_all.sum()))

        # --- per stratum ------------------------------------------------------
        per_var = {v: ([[] for _ in range(num_dims)], []) for v in range(1, num_vars)}
        clipped: Dict[int, list] = {}
        for stratum in wanted_strata:
            here = ref_split == stratum
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

                # proj_j(S_0): reference cells seen through this variable's dims.
                # Every stratum holds a reference site (LHS stage), so with no
                # dims the projection is the single cell 0.
                if dims:
                    proj = np.unique(
                        np.ravel_multi_index(
                            tuple(ref_here[d] for d in dims),
                            sizes,
                        )
                    )
                else:
                    proj = np.zeros(1, dtype=np.int64)

                # How many of this variable's cells must, may, and ideally do
                # land on the reference footprint.
                free_cells = plane[var_idx] - proj.size
                lo = max(0, n_here - free_cells)  # forced: nowhere else to go
                hi = min(n_here, proj.size)  # cannot exceed either set
                target = overlap_targets[var_idx - 1]
                if target is None:  # overlap='random': no control
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

                if n_overlap is None and layout == "padded" and padded_dim in dims:
                    # the dataset's layout applies to every variable that has
                    # the padded axis: each line fills its first k cells
                    cells = MultiVarRecordGenerator._padded_cells(
                        n_here,
                        sizes,
                        dims.index(padded_dim),
                    )
                    chosen = cells
                    outside = np.empty(0, dtype=np.int64)
                    n_overlap = 0
                elif n_overlap is None:
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

                # remainder from cells held by NEITHER variable, so the
                # overlap does not exceed the target
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
                        axes[dim].append(unravelled[axis])
                        axis += 1
                    else:
                        axes[dim].append(
                            np.full(
                                cells.size,
                                var_constant_coords[var_idx][dim],
                                dtype=np.int64,
                            )
                        )
                values.append(rng.uniform(0, 1, size=cells.size))

        for var_idx in range(1, num_vars):
            axes, values = per_var[var_idx]
            if values:
                out[f"var{var_idx}"] = (
                    tuple(np.concatenate(a) for a in axes),
                    np.concatenate(values),
                )
            else:
                out[f"var{var_idx}"] = (
                    tuple(np.empty(0, dtype=np.int64) for _ in range(num_dims)),
                    np.empty(0, dtype=float),
                )

        # An unreachable target means the (density, overlap) pair was
        # over-determined for that variable. This is always the case for a
        # variable varying along one dimension: the LHS stage uses every
        # coordinate of every axis, so var0's projection onto that axis covers
        # all of it and no cell is free of var0.
        for var_idx, events in sorted(clipped.items()):
            _, first_ideal, first_got, proj_size, free = events[0]
            print(
                f"  WARNING var{var_idx}: overlap target not reachable in "
                f"{len(events)} of {len(wanted_strata)} strata (e.g. wanted "
                f"{first_ideal} of {proj_size} reference cells, used "
                f"{first_got}; {free} cells free of var0). Density takes "
                f"precedence; see the achieved overlap below."
            )
        for var_idx, (goal, got, proj_total) in sorted(unreachable.items()):
            print(
                f"  WARNING var{var_idx}: overlap target not reachable (wanted "
                f"{goal} of {proj_total} reference cells, used {got}). Density "
                f"takes precedence; see the achieved overlap below."
            )
        return out
