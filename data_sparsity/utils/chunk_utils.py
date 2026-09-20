"""Utilities for parallel workflow.

This module handles chunk-level settings for splitting large datasets into
multiple smaller datasets.
"""

from typing import List
import numpy as np
from numpy.typing import ArrayLike

class ChunkUtils:
    """Utilities manager for chunked dataset generation"""

    @staticmethod
    def apportion(
            total: int,
            weights: ArrayLike,
            capacity: ArrayLike = None,
    ) -> np.ndarray:
        """Split an integer total across bins in proportion to weights.

        Uses largest-remainder apportionment: every bin gets the floor of its
        exact share, then the leftover units go to the bins with the largest
        fractional parts. The result sums to ``total`` exactly, which matters
        here because rounding each bin independently would change ``num_obs``,
        and ``num_obs`` seeds the index draw.

        Args:
            total: Integer amount to distribute
            weights: Relative weight of each bin
            capacity: Optional per-bin upper bound. Units that do not fit are
                redistributed to bins with room; if nothing has room the
                returned total is short of ``total``.

        Returns:
            Integer array summing to ``total`` (or to the total capacity, if
            that is smaller)
        """
        weights = np.asarray(weights, dtype=float)
        total = int(total)
        if total <= 0 or weights.size == 0 or weights.sum() <= 0:
            return np.zeros(weights.size, dtype=np.int64)

        raw = total * weights / weights.sum()
        counts = np.floor(raw).astype(np.int64)
        deficit = total - int(counts.sum())
        if deficit > 0:
            for idx in np.argsort(raw - counts)[::-1][:deficit]:
                counts[idx] += 1

        if capacity is None:
            return counts

        cap = np.asarray(capacity, dtype=np.int64)
        while True:
            overflow = int(np.maximum(counts - cap, 0).sum())
            counts = np.minimum(counts, cap)
            if overflow <= 0:
                return counts
            room = cap - counts
            if room.sum() <= 0:
                return counts
            counts = counts + ChunkUtils.apportion(
                min(overflow, int(room.sum())), room, room
            )

    @staticmethod
    def get_observations_per_chunk(
            total_obs: int,
            shape: tuple,
            max_dim_size: int,
            section_sizes: list,
            density: float,
    ) -> tuple[int, float, np.ndarray]:
        """Split the global observation count across chunks.

        The total is preserved exactly: observations are apportioned in
        proportion to each chunk's share of the grid, and the integer remainder
        goes to the chunks with the largest fractional parts (largest-remainder
        apportionment, as used by get_multi_var_observations_per_chunk).

        Rounding each chunk independently would move ``num_obs``, and with it
        ``density``, away from the serial value. That matters more than the one
        or two observations involved: ``num_obs`` seeds the global index draw,
        so a difference of one changes roughly half the occupied sites.

        Args:
            total_obs: Global number of observations to distribute
            shape: Full grid shape
            max_dim_size: Size of the split dimension
            section_sizes: Chunk sizes along the split dimension
            density: Global density, used only to report the outcome

        Returns:
            Tuple of (total observations, density, per-chunk counts)
        """
        total_points = int(np.prod(shape))
        total_points_slice = total_points / max_dim_size
        chunk_points = np.array(
            [total_points_slice * chunk_size for chunk_size in section_sizes]
        ).astype(int)

        # Apportion proportionally, then hand the leftover observations to the
        # chunks with the largest fractional parts so the total is preserved.
        raw = total_obs * chunk_points / chunk_points.sum()
        per_chunk_obs = np.floor(raw).astype(int)
        deficit = int(total_obs) - int(per_chunk_obs.sum())
        if deficit > 0:
            remainder = raw - per_chunk_obs
            for chunk_idx in np.argsort(remainder)[::-1][:deficit]:
                per_chunk_obs[chunk_idx] += 1

        # A chunk cannot hold more observations than it has grid points. This
        # only bites when num_obs exceeds the grid, which validation rejects.
        per_chunk_obs = np.minimum(per_chunk_obs, chunk_points)
        mp_obs = int(per_chunk_obs.sum())
        if mp_obs != int(total_obs):
            print(
                f"Chunk capacity limits observations to {mp_obs} "
                f"(requested {total_obs})."
            )

        density_new = mp_obs / total_points
        print(f"Observations per chunk: {per_chunk_obs.tolist()} (total {mp_obs}).")
        if not np.isclose(density_new, density):
            print(f"Updated density is {density_new} (was: {density}).")

        return mp_obs, density_new, per_chunk_obs

    @staticmethod
    def get_multi_var_observations_per_chunk(
            var_num_obs: np.ndarray,
            max_dim_size: int,
            section_sizes: list,
    ) -> List[np.ndarray]:
        """Split per-variable observation counts across chunks.

        The total per variable is preserved by distributing the integer
        remainder to the chunks with the largest fractional parts.

        Args:
            var_num_obs: Global observation counts per variable
            max_dim_size: Size of the split dimension
            section_sizes: Chunk sizes along the split dimension

        Returns:
            List with one integer array per chunk.
        """
        chunk_sizes = np.asarray(section_sizes, dtype=float)
        per_var_counts = []

        for total_obs in np.asarray(var_num_obs, dtype=float):
            raw_counts = total_obs * chunk_sizes / max_dim_size
            chunk_counts = np.floor(raw_counts).astype(int)
            remainder = raw_counts - chunk_counts
            deficit = int(round(total_obs)) - int(chunk_counts.sum())

            if deficit > 0:
                order = np.argsort(remainder)[::-1]
                for chunk_idx in order[:deficit]:
                    chunk_counts[chunk_idx] += 1
            elif deficit < 0:
                order = np.argsort(remainder)
                for chunk_idx in order:
                    if deficit == 0:
                        break
                    if chunk_counts[chunk_idx] > 0:
                        chunk_counts[chunk_idx] -= 1
                        deficit += 1

            per_var_counts.append(chunk_counts.astype(int))

        per_chunk_obs = [
            np.asarray([per_var_counts[var_idx][chunk_idx] for var_idx in range(len(per_var_counts))], dtype=int)
            for chunk_idx in range(len(section_sizes))
        ]

        return per_chunk_obs


    @staticmethod
    def generate_rngs(seed: int, num_dims: int) -> dict:
        """Generate one deterministic RNG per dimension.

        ``base_seed = seed + dim_idx * 1000``. Serial and every worker derive
        the same stream per dimension, which is what makes the non-split
        coordinate axes identical in every chunk file.

        This used to accept ``chunk_id``, ``dim_split`` and ``obs_in_chunk``
        and fast-forward the split dimension's stream by ``chunk_id *
        obs_in_chunk`` draws, so that a chunk would resume where the previous
        one left off. No caller ever passed them, and the idea cannot work:
        serial coordinates are the ORDER STATISTICS of the whole sample, so
        chunk k's coordinates are not draws k*n through (k+1)*n of the stream.
        Chunks slice the sorted global axis by index instead.

        Args:
            seed: Base random seed
            num_dims: Total number of dimensions

        Returns:
            Dict mapping dimension index to RNG
        """
        return {
            dim_idx: np.random.default_rng(seed + (dim_idx * 1000))
            for dim_idx in range(num_dims)
        }

    @staticmethod
    def update_chunk_shape(
            shape: tuple,
            dim_split: int,
            task_size: int,
    ) -> tuple:
        """Generate chunk shape by reducing its size along the split dimension"""

        task_shape = list(shape)
        task_shape[dim_split] = task_size

        return task_shape
        
    @staticmethod
    def validate_chunk_points(
            task_shape: tuple
    ) -> int:
        """Validate that total chunk points are int"""

        total_chunk_points = np.prod(task_shape)
        if not total_chunk_points.is_integer():
            raise ValueError("total_chunk_points must be an int")

        return int(total_chunk_points)

