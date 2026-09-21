"""Base record generation functionality.

This module provides the base class for generating sparse record arrays
with common utilities for index generation and assignment.
"""

from typing import Iterable, List, Optional, Tuple
import numpy as np

from data_sparsity.utils.chunk_utils import ChunkUtils
from data_sparsity.generators.observation_generator import (
    ObservationGenerator,
)
from data_sparsity.utils.streams import Stream, stream


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
    @staticmethod
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
    def generate_lhs_indices(
        shape: List[int],
        n_s: int,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, ...]:
        """Generate Latin Hypercube Sample indices for base coverage.
        
        Creates n_s observations ensuring each coordinate in each dimension
        is used at least once. This forms the LHS component of the hybrid
        sampling approach.
        
        For dimensions where size equals n_s, generates a full permutation
        ensuring each coordinate is used exactly once. For dimensions larger
        than n_s, randomly selects n_s unique coordinates.
        
        Args:
            shape: Grid shape [n0, n1, ..., nk]
            n_s: Number of LHS samples (typically min(shape))
            rng: Random number generator
            
        Returns:
            Tuple of index arrays, one per dimension, each of length n_s
            
        Raises:
            ValueError: If any dimension size < n_s
            
        Example:
            >>> rng = np.random.default_rng(42)
            >>> shape = [5, 7, 5]
            >>> n_s = 5
            >>> indices = RecordGenerator.generate_lhs_indices(shape, n_s, rng)
            >>> len(indices)  # 3 dimensions
            3
            >>> all(len(idx) == 5 for idx in indices)  # Each has n_s=5 samples
            True
            >>> set(indices[0])  # Dimension 0 (size 5): all coords used
            {0, 1, 2, 3, 4}
            >>> len(set(indices[1]))  # Dimension 1 (size 7): 5 unique selected
            5
        """
        max_dim_size = max(shape)
        if n_s < max_dim_size:
            # One coordinate per axis per point, so n_s below the longest axis
            # leaves some coordinate of it unused, which docs/explainer.md
            # excludes by definition.
            raise ValueError(
                f"n_s {n_s} < max(shape) {max_dim_size}: cannot cover every "
                f"coordinate of every axis. n_s must be max(shape) so that "
                f"each axis can be fully used at least once."
            )

        indices = []

        for dim_size in shape:
            if dim_size == n_s:
                # Full permutation: each coordinate used exactly once
                perm = rng.permutation(dim_size)
                indices.append(perm)
            else:
                # Tile permutations and truncate: the first block uses every
                # coordinate once, the rest spreads the surplus evenly. The
                # shuffle keeps the pairing with other axes random.
                blocks = [rng.permutation(dim_size)
                          for _ in range(-(-n_s // dim_size))]
                tiled = np.concatenate(blocks)[:n_s]
                rng.shuffle(tiled)
                indices.append(tiled)

        return tuple(indices)

    @staticmethod
    def _ranks_to_local(ranks: np.ndarray, excluded_sorted: np.ndarray) -> np.ndarray:
        """Map ranks within the free sites of a stratum to local site indices.

        ``ranks`` index the sites of a hyperplane once the ``excluded`` ones are
        removed; this returns the true local indices without ever materialising
        the complement.

        For sorted excluded values ``e``, ``e[i] - i`` counts the free sites
        below ``e[i]``, so the r-th free site is ``r`` plus however many excluded
        values sit at or below it. That is one searchsorted, O(k log |e|),
        rather than a pass over ``e`` per rank -- which matters once ``e`` is a
        projected footprint of hundreds of thousands of cells.
        """
        local = np.asarray(ranks, dtype=np.int64)
        excluded_sorted = np.asarray(excluded_sorted, dtype=np.int64)
        if excluded_sorted.size == 0:
            return local
        offset = excluded_sorted - np.arange(excluded_sorted.size, dtype=np.int64)
        return local + np.searchsorted(offset, local, side="right")

    @staticmethod
    def generate_padded_indices(
            global_shape: List[int],
            num_obs: int,
            seed: int,
            split_dim: int,
            padded_dim: int,
            strata: Optional[Iterable[int]] = None,
    ) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        """Place observations as a prefix along one axis, with values.

        Each combination of the dimensions other than the split and the padded
        one is a **line**, and a line holds positions ``0..k-1`` of the padded
        axis. This is the arrangement Argo and CrocoLake have: a profile fills
        its first so many levels and the rest of the row is fill.

        Coverage comes from the construction rather than from an LHS stage.
        Every line holds at least one observation, which uses every coordinate
        of the split axis and of every axis but the padded one; one line runs
        the full length, which uses every coordinate of the padded axis. So no
        cell sits outside the pattern, which an LHS would have put there.

        A stratum still depends only on ``(seed, stratum)``, so a caller asking
        for a subset of strata gets the same slices as one asking for all.

        Args:
            global_shape: Full grid shape
            num_obs: Total observations across the whole grid
            seed: Base random seed
            split_dim: Dimension indexing the strata
            padded_dim: Dimension the prefixes run along
            strata: Which strata to generate (default: all of them)

        Returns:
            Tuple of (multi-indices in GLOBAL space, observation values)

        Raises:
            ValueError: If the padded and split dimensions are the same
        """
        shape = [int(size) for size in global_shape]
        num_dims = len(shape)
        if padded_dim == split_dim:
            raise ValueError(
                f"padded_dim and split_dim are both {split_dim}: a stratum "
                "holds one index of the split dimension, so a prefix along it "
                "would be a single cell."
            )
        n_padded = shape[padded_dim]
        line_dims = [d for d in range(num_dims) if d not in (split_dim, padded_dim)]
        line_shape = [shape[d] for d in line_dims]
        lines = int(np.prod(line_shape)) if line_shape else 1

        counts, carrier = RecordGenerator.padded_stratum_counts(
            shape, num_obs, seed, split_dim, padded_dim
        )

        if strata is None:
            strata = range(shape[split_dim])
        per_dim = [[] for _ in range(num_dims)]
        values = []

        for stratum in strata:
            stratum = int(stratum)
            count = int(counts[stratum])
            if count == 0:
                continue
            rng = stream(seed, Stream.STRATUM, stratum)
            lengths = RecordGenerator._prefix_lengths(
                count, lines, n_padded, rng, full_line=(stratum == carrier)
            )

            # one line per column of the line space, each contributing a run
            line_index = np.repeat(np.arange(lines, dtype=np.int64), lengths)
            positions = np.concatenate(
                [np.arange(length, dtype=np.int64) for length in lengths]
            )
            unravelled = (np.unravel_index(line_index, line_shape)
                          if line_shape else ())

            per_dim[split_dim].append(
                np.full(line_index.size, stratum, dtype=np.int64))
            per_dim[padded_dim].append(positions)
            for axis, dim in enumerate(line_dims):
                per_dim[dim].append(unravelled[axis].astype(np.int64))

            # values after the sites, on the same stream, so site i and value i
            # stay paired however the strata are grouped
            values.append(
                ObservationGenerator.generate_observations(line_index.size, rng)
            )

        if not values:
            empty = tuple(np.empty(0, dtype=np.int64) for _ in range(num_dims))
            return empty, np.empty(0, dtype=float)
        return (tuple(np.concatenate(axis) for axis in per_dim),
                np.concatenate(values))

    @staticmethod
    def padded_stratum_counts(
            global_shape: List[int],
            num_obs: int,
            seed: int,
            split_dim: int,
            padded_dim: int,
            sigma: float = 1.5,
    ) -> Tuple[np.ndarray, int]:
        """Observations per stratum under the padded layout, and which stratum
        carries the full-length line.

        No LHS stage: coverage comes from the prefix construction, so the
        counts are a plain apportionment over stratum capacity with a floor of
        one observation per line. One line has to reach the last coordinate of
        the padded axis, or that coordinate goes unused; the stratum with the
        largest count is raised to afford it and the difference comes back from
        the others, bounded so none drops below its own floor.

        Every figure here follows from the arguments alone, so a worker holding
        one chunk derives the same vector as a serial run.

        Args:
            global_shape: Full grid shape
            num_obs: Total observations across the whole grid
            seed: Base random seed
            split_dim: Dimension indexing the strata
            padded_dim: Dimension the prefixes run along
            sigma: Spread of the lognormal weighting the strata. 1.5 puts the
                median near CrocoLake's, whose profiles run 1 / 70 / 155 / 1042
                for minimum, median, mean and maximum levels; the generated
                minimum comes out higher than the real one because the
                apportionment redistributes what will not fit under the cap,
                which lifts the low tail. Raising it further is self-defeating:
                at 2.0 the cap dominates and the minimum climbs.

        Returns:
            Tuple of (counts per stratum, index of the stratum holding the
            full-length line)

        Raises:
            ValueError: If num_obs cannot cover every line and reach the last
                padded coordinate
        """
        shape = [int(size) for size in global_shape]
        num_strata = shape[split_dim]
        n_padded = shape[padded_dim]
        lines = int(np.prod([size for dim, size in enumerate(shape)
                             if dim not in (split_dim, padded_dim)])) or 1

        floor = np.full(num_strata, lines, dtype=np.int64)
        room = np.full(num_strata, lines * (n_padded - 1), dtype=np.int64)
        capacity = int(np.prod(shape))
        if num_obs > capacity:
            raise ValueError(
                f"num_obs {num_obs} exceeds the {capacity} cells of shape "
                f"{shape}; the apportionment would cap it and the realised "
                f"count would be short with no error."
            )
        minimum = num_strata * lines + n_padded - 1
        if num_obs < minimum:
            raise ValueError(
                f"num_obs {num_obs} < {minimum} for a padded layout on shape "
                f"{shape}: every one of the {num_strata * lines} lines needs an "
                f"observation, and one line must reach coordinate "
                f"{n_padded - 1} of the padded axis."
            )
        # Lognormal weights rather than uniform, because on a two-dimensional
        # grid each stratum is one line and the profile-to-profile variation in
        # length comes from here, not from _prefix_lengths. Uniform weights
        # gave every profile the same length: median 155 of a maximum 1042,
        # where Argo's median is 70.
        weights = stream(seed, Stream.PADDED).lognormal(0.0, sigma, num_strata)
        counts = floor + ChunkUtils.apportion(
            int(num_obs) - int(floor.sum()), weights, room
        )

        # Raise the chosen stratum so one of its lines can run the full length,
        # and take the difference back from the slack the others hold.
        carrier = int(np.argmax(counts))
        needed = n_padded + lines - 1
        deficit = needed - int(counts[carrier])
        if deficit > 0:
            slack = counts - floor
            slack[carrier] = 0
            taken = ChunkUtils.apportion(deficit, slack, slack)
            counts = counts - taken
            counts[carrier] += int(taken.sum())
        return counts, carrier

    @staticmethod
    def _prefix_lengths(
            count: int,
            lines: int,
            n_padded: int,
            rng: np.random.Generator,
            full_line: bool,
            sigma: float = 0.8,
    ) -> np.ndarray:
        """How far along the padded axis each line reaches.

        Lengths are lognormal in shape, which is closer to real profile data
        than a uniform: Argo's levels per profile have a median of 70 against a
        maximum of 1042. The draw sets the proportions and an apportionment
        turns them into whole numbers summing to ``count``, so density stays
        exact whatever the distribution does.

        Args:
            count: Observations this stratum must place
            lines: How many lines it holds
            n_padded: Length of the padded axis
            rng: This stratum's generator
            full_line: Whether line 0 must reach the last coordinate
            sigma: Spread of the lognormal

        Returns:
            Array of ``lines`` prefix lengths, each between 1 and n_padded
        """
        weights = rng.lognormal(0.0, sigma, size=lines)
        lengths = np.ones(lines, dtype=np.int64)
        if full_line:
            lengths[0] = n_padded
            room = np.full(lines, n_padded - 1, dtype=np.int64)
            room[0] = 0
            weights = weights.copy()
            weights[0] = 0.0
        else:
            room = np.full(lines, n_padded - 1, dtype=np.int64)
        spare = int(count) - int(lengths.sum())
        if spare > 0:
            lengths = lengths + ChunkUtils.apportion(spare, weights, room)
        return lengths

    @staticmethod
    def stratum_counts(
            global_shape: List[int],
            num_obs: int,
            seed: int,
            split_dim: int,
    ) -> np.ndarray:
        """How many observations the reference variable puts in each stratum.

        The same figure ``generate_stratified_indices`` derives, without
        placing anything. Both stages behind it are global: the Latin
        hypercube draws from ``stream(seed, Stream.LHS)``, and the fill is
        apportioned across all strata at once. So a worker holding one chunk
        can still learn the counts of strata it does not own, at the cost of
        one LHS draw of ``max(shape)`` points.

        This is what lets the overlap target be apportioned globally rather
        than rounded in each stratum.

        Args:
            global_shape: Full grid shape
            num_obs: Reference variable's global observation count
            seed: Base random seed
            split_dim: Dimension the strata run along

        Returns:
            Array of length ``global_shape[split_dim]`` summing to ``num_obs``
        """
        shape = [int(size) for size in global_shape]
        num_strata = shape[split_dim]
        hyper_shape = [size for dim, size in enumerate(shape) if dim != split_dim]
        stratum_sites = int(np.prod(hyper_shape)) if hyper_shape else 1
        num_obs = int(num_obs)

        n_s = max(shape)
        lhs = RecordGenerator.generate_lhs_indices(
            shape, n_s, stream(seed, Stream.LHS)
        )
        taken = np.bincount(
            np.asarray(lhs[split_dim], dtype=np.int64), minlength=num_strata
        )
        available = stratum_sites - taken
        return taken + ChunkUtils.apportion(num_obs - n_s, available, available)

    @staticmethod
    def generate_stratified_indices(
        global_shape: List[int],
        num_obs: int,
        seed: int,
        split_dim: int,
        strata: Optional[Iterable[int]] = None,
        layout: str = "scattered",
        padded_dim: Optional[int] = None,
    ) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
        """Place observations one hyperplane at a time, with values.

        The grid is partitioned into ``global_shape[split_dim]`` strata, one per
        index along the split dimension. Two stages, mirroring
        ``generate_hybrid_indices`` but decomposed:

        * The LHS stage stays **global**. It is ``min(num_obs, max(shape))``
          points and its guarantee (every coordinate of every axis used)
          spans strata, so no stratum can enforce it alone. Every caller
          recomputes it identically for a few hundred bytes.
        * The fill stage is **per stratum**. Counts are apportioned globally,
          then each stratum draws its own sites and values from
          ``(seed, STRATUM_STREAM, j)`` and nothing else.

        Because a stratum depends only on that triple, any caller producing a
        subset of strata produces exactly the slices a caller producing all of
        them would. That is what makes serial and parallel agree by
        construction rather than by arranging for streams to line up, and it
        keeps peak memory at one hyperplane instead of the whole grid.

        Args:
            global_shape: Full grid shape
            num_obs: Total observations across the whole grid
            seed: Base random seed
            split_dim: Dimension indexing the strata
            strata: Which strata to generate (default: all of them)
            layout: ``scattered``, the Latin hypercube plus uniform fill, or
                ``padded``, a prefix along ``padded_dim`` in every line
            padded_dim: The axis the prefixes run along, required for
                ``padded``

        Returns:
            Tuple of (multi-indices in GLOBAL space, observation values). The
            caller maps the split dimension to chunk-local coordinates if it
            needs to.

        Raises:
            ValueError: If the layout is unknown, or padded is asked for
                without a padded_dim
        """
        if layout == "padded":
            if padded_dim is None:
                raise ValueError(
                    "layout='padded' needs padded_dim: the axis the prefixes "
                    "run along."
                )
            return RecordGenerator.generate_padded_indices(
                global_shape, num_obs, seed, split_dim, padded_dim, strata
            )
        if layout != "scattered":
            raise ValueError(
                f"Unknown layout {layout!r}. Use 'scattered' or 'padded'."
            )

        shape = [int(size) for size in global_shape]
        num_dims = len(shape)
        num_strata = shape[split_dim]
        hyper_shape = [size for dim, size in enumerate(shape) if dim != split_dim]
        stratum_sites = int(np.prod(hyper_shape)) if hyper_shape else 1
        num_obs = int(num_obs)

        # --- LHS stage: global, O(min(shape)) -----------------------------
        if num_obs < max(shape):
            raise ValueError(
                f"num_obs {num_obs} < max(shape) {max(shape)}: every "
                f"coordinate of every axis must be used at least once, which "
                f"needs at least max(shape) observations."
            )
        n_s = max(shape)
        lhs_rng = stream(seed, Stream.LHS)
        lhs = RecordGenerator.generate_lhs_indices(shape, n_s, lhs_rng)
        lhs_split = np.asarray(lhs[split_dim], dtype=np.int64)
        if hyper_shape:
            lhs_local = np.ravel_multi_index(
                tuple(np.asarray(lhs[dim], dtype=np.int64)
                      for dim in range(num_dims) if dim != split_dim),
                hyper_shape,
            )
        else:
            lhs_local = np.zeros(n_s, dtype=np.int64)

        # --- apportion the fill across strata: global, O(num_strata) ------
        taken = np.bincount(lhs_split, minlength=num_strata)
        available = stratum_sites - taken
        fill_counts = ChunkUtils.apportion(num_obs - n_s, available, available)
        # Same arithmetic as stratum_counts, which a worker calls to learn the
        # counts of strata it does not own. Kept in step by the test that
        # compares the two.

        # --- per-stratum draw ---------------------------------------------
        if strata is None:
            strata = range(num_strata)
        per_dim = [[] for _ in range(num_dims)]
        values = []

        for stratum in strata:
            stratum = int(stratum)
            here = lhs_split == stratum
            lhs_here = lhs_local[here]
            n_fill = int(fill_counts[stratum])
            rng = stream(seed, Stream.STRATUM, stratum)

            if n_fill > 0:
                ranks = rng.choice(
                    stratum_sites - lhs_here.size, size=n_fill, replace=False
                )
                fill_local = RecordGenerator._ranks_to_local(
                    ranks, np.sort(lhs_here)
                )
            else:
                fill_local = np.empty(0, dtype=np.int64)

            local = np.concatenate([lhs_here, fill_local]).astype(np.int64)
            if local.size == 0:
                continue

            # values share the stratum stream, drawn after the sites so that
            # site i and value i stay paired however the strata are grouped
            values.append(
                ObservationGenerator.generate_observations(local.size, rng)
            )

            hyper_idx = np.unravel_index(local, hyper_shape) if hyper_shape else ()
            axis = 0
            for dim in range(num_dims):
                if dim == split_dim:
                    per_dim[dim].append(np.full(local.size, stratum, dtype=np.int64))
                else:
                    per_dim[dim].append(hyper_idx[axis])
                    axis += 1

        if not values:
            empty = tuple(np.empty(0, dtype=np.int64) for _ in range(num_dims))
            return empty, np.empty(0, dtype=float)

        indices = tuple(np.concatenate(parts) for parts in per_dim)
        return indices, np.concatenate(values)
