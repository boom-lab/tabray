"""Base record generation functionality.

This module provides the base class for generating sparse record arrays
with common utilities for index generation and assignment.
"""

from typing import Iterable, List, Optional, Tuple
import numpy as np

from data_sparsity.utils.chunk_utils import ChunkUtils
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
        observations: np.ndarray,
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
        rng: np.random.Generator,
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
                blocks = [rng.permutation(dim_size) for _ in range(-(-n_s // dim_size))]
                tiled = np.concatenate(blocks)[:n_s]
                rng.shuffle(tiled)
                indices.append(tiled)

        return tuple(indices)

    @staticmethod
    def generate_hybrid_indices(
        shape: List[int],
        num_obs: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, ...]:
        """Generate indices using hybrid LHS + random sampling.

        This method implements a two-stage approach that guarantees all
        coordinates are used while maintaining randomness for additional
        observations:

        Stage 1 (LHS base): First n_s observations use Latin Hypercube
                            Sampling to ensure each coordinate in each
                            dimension is used at least once.

        Stage 2 (Random fill): Remaining observations (if any) use standard
                               random sampling without replacement, ensuring
                               no duplicates across LHS and random samples.

        This hybrid approach works at ALL density levels:
        - At minimum density (num_obs = min(shape)): Pure LHS
        - Above minimum: LHS base + random fill

        Args:
            shape: Grid shape [n0, n1, ..., nk]
            num_obs: Total number of observations to generate
            rng: Random number generator

        Returns:
            Tuple of index arrays, one per dimension, each of length num_obs

        Example:
            >>> rng = np.random.default_rng(42)
            >>> shape = [5, 5]
            >>> # Minimum density: pure LHS
            >>> indices_min = RecordGenerator.generate_hybrid_indices(shape, 5, rng)
            >>> len(indices_min[0])
            5
            >>> # Above minimum: LHS + random
            >>> rng = np.random.default_rng(42)
            >>> indices_high = RecordGenerator.generate_hybrid_indices(shape, 10, rng)
            >>> len(indices_high[0])
            10
            >>> # All coordinates guaranteed to be used
            >>> len(set(indices_high[0]))
            5
        """
        # One coordinate per axis per point, so the LHS is sized by the
        # LONGEST axis -- see docs/explainer.md on minimum density.
        max_dim_size = max(shape)
        if num_obs < max_dim_size:
            raise ValueError(
                f"num_obs {num_obs} < max(shape) {max_dim_size}: every "
                f"coordinate of every axis must be used at least once, which "
                f"needs at least max(shape) observations."
            )
        n_s = max_dim_size  # LHS base coverage: the whole longest axis
        n_random = num_obs - n_s  # Additional random points

        # Stage 1: LHS for base coverage (first n_s observations)
        lhs_indices = RecordGenerator.generate_lhs_indices(shape, n_s, rng)

        if n_random == 0:
            # At minimum density, pure LHS is sufficient
            return lhs_indices

        # Stage 2: Random sampling for additional observations
        # We need to avoid duplicating LHS positions
        total_points = int(np.prod(shape))

        # Convert LHS indices to flat indices to identify used positions
        lhs_flat = np.ravel_multi_index(lhs_indices, shape)
        lhs_set = set(lhs_flat)

        # Create list of available positions (excluding LHS positions)
        available_positions = np.array(
            [i for i in range(total_points) if i not in lhs_set]
        )

        # Sample from available positions
        if len(available_positions) >= n_random:
            # Enough positions available: sample without replacement
            random_flat = rng.choice(available_positions, size=n_random, replace=False)
        else:
            # Not enough positions: use all available, then sample remaining with replacement
            # This can happen when num_obs approaches or exceeds total_points
            random_flat = np.concatenate(
                [
                    available_positions,
                    rng.choice(
                        total_points,
                        size=n_random - len(available_positions),
                        replace=True,
                    ),
                ]
            )

        random_indices = np.unravel_index(random_flat, shape)

        # Combine LHS and random indices
        combined_indices = tuple(
            np.concatenate([lhs_indices[i], random_indices[i]])
            for i in range(len(shape))
        )

        return combined_indices

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
    def generate_stratified_indices(
        global_shape: List[int],
        num_obs: int,
        seed: int,
        split_dim: int,
        strata: Optional[Iterable[int]] = None,
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

        Returns:
            Tuple of (multi-indices in GLOBAL space, observation values). The
            caller maps the split dimension to chunk-local coordinates if it
            needs to.
        """
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
                tuple(
                    np.asarray(lhs[dim], dtype=np.int64)
                    for dim in range(num_dims)
                    if dim != split_dim
                ),
                hyper_shape,
            )
        else:
            lhs_local = np.zeros(n_s, dtype=np.int64)

        # --- apportion the fill across strata: global, O(num_strata) ------
        taken = np.bincount(lhs_split, minlength=num_strata)
        available = stratum_sites - taken
        fill_counts = ChunkUtils.apportion(num_obs - n_s, available, available)

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
                    stratum_sites - lhs_here.size,
                    size=n_fill,
                    replace=False,
                )
                fill_local = RecordGenerator._ranks_to_local(
                    ranks,
                    np.sort(lhs_here),
                )
            else:
                fill_local = np.empty(0, dtype=np.int64)

            local = np.concatenate([lhs_here, fill_local]).astype(np.int64)
            if local.size == 0:
                continue

            # values share the stratum stream, drawn after the sites so that
            # site i and value i stay paired however the strata are grouped
            values.append(rng.uniform(0, 1, size=local.size))

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
