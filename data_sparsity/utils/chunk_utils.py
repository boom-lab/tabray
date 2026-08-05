"""Utilities for parallel workflow.

This module handles chunk-level settings for splitting large datasets into
multiple smaller datasets.
"""

from typing import List, Union
import numpy as np
import logging

class ChunkUtils:
    """Utilities manager for chunked dataset generation"""

    @staticmethod
    def get_observations_per_chunk(
            total_obs: int,
            shape: tuple,
            max_dim_size: int,
            section_sizes: list,
            density: float,
    ) -> tuple[int, float, int]:

        total_points = np.prod(shape)
        total_points_slice = total_points / max_dim_size
        chunk_points = np.array(
            [total_points_slice * chunk_size for chunk_size in section_sizes]
        ).astype(int)
        print("type chunk_points", type(chunk_points))
        print("chunk_points", chunk_points)
        
        # As randomness is uniform
        per_chunk_obs = np.rint(density * chunk_points).astype(int)
        for idx, c in enumerate(per_chunk_obs):
            if c > chunk_points[idx]:
                per_chunk_obs[idx] = chunk_points[idx]

        mp_obs = per_chunk_obs.sum()
        if not mp_obs.is_integer():
            raise ValueError(f"Got non integer value of observations {mp_obs}.")
        mp_obs = int(mp_obs)
        print(
            f"Multiprocessing approximations lead to {mp_obs} "
            f"total observation (goal: {total_obs})."
        )
        if mp_obs > total_obs:
            extra_obs = mp_obs - total_obs
            if per_chunk_obs[idx] > extra_obs:
                per_chunk_obs[idx] = per_chunk_obs[idx] - extra_obs
                per_chunk_obs[idx] = per_chunk_obs[idx].astype(int)
                mp_obs = per_chunk_obs.sum()
                mp_obs = int(mp_obs)
            else:
                print(
                    f"Multiprocessing approximations lead to {mp_obs} "
                    f"total observation (goal: {total_obs})."
                )

        density_new = mp_obs / total_points
        print(f"Multiprocessing approximations lead to {mp_obs} total observation (goal: {total_obs}).")
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
    def generate_rngs(
            seed: int,
            num_dims: int,
            chunk_id: int = None,
            dim_split: int = None,
            obs_in_chunk: int = None
    ) -> dict:
        """
        Generate dimension-specific RNGs with optional state advancement for parallel mode.
        
        Implements deterministic RNG seeding: base_seed = seed + (dim_idx * 1000)
        
        For serial execution (chunk_id=None):
            All dimensions use fresh RNGs with their base seed.
        
        For parallel execution (chunk_id provided):
            Split dimension RNG is advanced by obs_in_chunk * chunk_id to maintain
            alignment with serial generation mode. Non-split dimensions use fresh RNGs.
        
        This ensures coordinates are identical between serial and parallel modes.
        
        Args:
            seed: Base random seed
            num_dims: Total number of dimensions
            chunk_id: Chunk index (0-based). None for serial mode.
            dim_split: Index of split dimension. None for serial mode.
            obs_in_chunk: Number of observations in this chunk. None for serial mode.
            
        Returns:
            Dict mapping dimension index to RNG (advanced if parallel split dimension)
            
        Examples:
            # Serial mode
            rngs = generate_rngs(seed=42, num_dims=3)
            
            # Parallel mode, chunk 2 of 5
            rngs = generate_rngs(seed=42, num_dims=3, chunk_id=2, 
                                dim_split=0, obs_in_chunk=100)
        """
        dim_rngs = {}
        
        # Determine if this is parallel mode
        is_parallel = chunk_id is not None and dim_split is not None
        
        for dim_idx in range(num_dims):
            # Base seed: same offset per dimension across all chunks and modes
            base_seed = seed + (dim_idx * 1000)
            dim_rng = np.random.default_rng(base_seed)
            
            # Advance RNG for split dimension in parallel mode
            if is_parallel and dim_idx == dim_split and obs_in_chunk is not None:
                draws_before_this_chunk = chunk_id * obs_in_chunk
                if draws_before_this_chunk > 0:
                    # Advance RNG state to align with serial generation
                    _ = dim_rng.uniform(0, 1, draws_before_this_chunk)
            
            dim_rngs[dim_idx] = dim_rng
        
        return dim_rngs

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

    @staticmethod
    def generate_split_dimension_range(
            dim_split: int,
            task_range: tuple,
            dim_size: int,
    ) -> dict:
        """Split dimension uses normalized chunk range with task_rng"""

        dim_ranges = {
            dim_split: (
                task_range[0] / dim_size,
                task_range[1] / dim_size
            )
        }

        return dim_ranges

    @staticmethod
    def assign_rngs_to_dimensions(
            dim_split: int,
            task_shape: tuple,
            dim_rngs_dict: dict
    ) -> dict:
        """
        Return dimension RNGs (already prepared by generate_rngs).
        
        This is now a simple pass-through since generate_rngs() already
        returns the properly advanced dimension RNGs.
        
        Args:
            dim_split: Index of split dimension (unused, for API compatibility)
            task_shape: Shape of current chunk (unused, for API compatibility)
            dim_rngs_dict: Dict from generate_rngs() with per-dimension RNGs
            
        Returns:
            Dict mapping dimension index to RNG
        """
        return dim_rngs_dict

    @staticmethod
    def calculate_lhs_draws_per_generation(
        shape: List[int],
        n_s: int
    ) -> int:
        """Calculate RNG draws for one complete LHS generation.
        
        Analyzes the dimensional structure to deterministically compute
        the exact number of RNG draws required for LHS generation across
        all dimensions. This is efficient and does not depend on NumPy
        version internals.
        
        Args:
            shape: Grid dimensions [n0, n1, ..., nk]
            n_s: LHS base size (typically min(shape))
            
        Returns:
            Total draws across all dimensions
            
        Raises:
            ValueError: If any dimension size < n_s
            
        Example:
            >>> ChunkUtils.calculate_lhs_draws_per_generation([5, 5, 5], 5)
            15
            >>> ChunkUtils.calculate_lhs_draws_per_generation([5, 7, 10], 5)
            25
        """
        total_draws = 0
        for dim_size in shape:
            if dim_size == n_s:
                # permutation(n_s) → n_s draws
                total_draws += n_s
            elif dim_size > n_s:
                # choice(dim_size, n_s, replace=False) → n_s draws
                # shuffle(n_s) → n_s draws
                total_draws += 2 * n_s
            else:
                raise ValueError(f"dim_size {dim_size} < n_s {n_s}")
        
        return total_draws

    @staticmethod
    def generate_lhs_rng(
        seed: int,
        shape: List[int],
        chunk_id: int = None,
        num_obs_global: int = None
    ) -> np.random.Generator:
        """Generate LHS RNG with state advancement for parallel mode.
        
        Creates deterministic RNG for Latin Hypercube Sampling. In parallel mode,
        advances RNG state by calculating and skipping the exact number of draws
        that previous chunks made, ensuring serial/parallel equivalence.
        
        With the global LHS approach, each chunk generates the full global LHS
        (num_obs_global observations) and filters to its coordinate range. Thus,
        all chunks make the same number of RNG draws, simplifying advancement.
        
        Args:
            seed: Base random seed
            shape: GLOBAL grid shape [n0, n1, ..., nk]
            chunk_id: Chunk index (None for serial)
            num_obs_global: Total observations globally (None for serial)
            
        Returns:
            RNG advanced to correct position for this chunk
            
        Examples:
            # Serial mode
            >>> lhs_rng = ChunkUtils.generate_lhs_rng(42, [5, 5])
            
            # Parallel mode, chunk 2 of 4, 10 global observations
            >>> lhs_rng = ChunkUtils.generate_lhs_rng(42, [5, 5], 2, 10)
        """
        # LHS uses seed offset 10000 to avoid collision with coordinates
        # Coordinate seeds use: seed + (dim_idx * 1000), where dim_idx < num_dims
        lhs_seed = seed + 1000
        lhs_rng = np.random.default_rng(lhs_seed)
        
        # Parallel mode: all chunks use the same RNG state
        # With global LHS filtering, all chunks generate the same global LHS
        # and filter to their coordinate range. No advancement needed!
        return lhs_rng
        
    
