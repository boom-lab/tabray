"""Utilities for parallel workflow.

This module handles chunk-level settings for splitting large datasets into
multiple smaller datasets.
"""

from typing import List
import numpy as np
from numpy.typing import ArrayLike

from data_sparsity.utils.streams import Stream, stream


class ChunkUtils:
    """Utilities manager for chunked dataset generation"""

    @staticmethod
    def apportion(
        total: int,
        weights: ArrayLike,
    ) -> np.ndarray:
        """Split an integer total across bins in proportion to weights.

        Uses largest-remainder apportionment: every bin gets the floor of its
        exact share, then the leftover units go to the bins with the largest
        fractional parts. The result sums to ``total`` exactly, which matters
        here because rounding each bin independently would change ``num_obs``,
        and a chunked run would then place a different number of observations
        than the serial run.

        With ``total <= sum(weights)`` no bin exceeds its weight, so weights
        double as capacities: bin i gets floor(total * w_i / W) <= w_i, plus
        one only when that floor is below the exact share. Callers pass the
        free sites per bin and never more observations than sites.

        Args:
            total: Integer amount to distribute
            weights: Relative weight of each bin

        Returns:
            Integer array summing to ``total``
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
        return counts

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
        ``density``, away from the serial value. Placement apportions
        ``num_obs`` across strata, so a different total changes the per-stratum
        counts and the chunked output no longer matches the serial output.

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

        mp_obs = int(per_chunk_obs.sum())
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

            # deficit >= 0: the floors never sum above the total
            if deficit > 0:
                order = np.argsort(remainder)[::-1]
                for chunk_idx in order[:deficit]:
                    chunk_counts[chunk_idx] += 1

            per_var_counts.append(chunk_counts.astype(int))

        per_chunk_obs = [
            np.asarray(
                [
                    per_var_counts[var_idx][chunk_idx]
                    for var_idx in range(len(per_var_counts))
                ],
                dtype=int,
            )
            for chunk_idx in range(len(section_sizes))
        ]

        return per_chunk_obs

    @staticmethod
    def generate_rngs(seed: int, num_dims: int) -> dict:
        """Generate one deterministic RNG per dimension.

        Serial and every worker derive the same stream per dimension, which is
        what makes the non-split coordinate axes identical in every chunk file.
        A chunk takes its coordinates by slicing the sorted global axis, never
        by advancing the stream: serial coordinates are the order statistics of
        the whole sample, so chunk k's are not draws k*n to (k+1)*n of it.

        Args:
            seed: Base random seed
            num_dims: Total number of dimensions

        Returns:
            Dict mapping dimension index to RNG
        """
        return {
            dim_idx: stream(seed, Stream.COORDINATE, dim_idx)
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
