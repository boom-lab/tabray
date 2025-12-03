"""Utilities for parallel workflow.

This module handles chunk-level settings for splitting large datasets into
multiple smaller datasets.
"""

from typing import List, Union
import numpy as np

class ChunkUtils:
    """Utilities manager for chunked dataset generation"""

    @staticmethod
    def get_observations_per_chunk(
            total_obs: int,
            shape: tuple,
            max_dim_size: int,
            section_sizes: list,
            sparsity: float,
    ) -> tuple[int, float, int]:

        total_points = np.prod(shape)
        total_points_slice = total_points / max_dim_size
        chunk_points = np.array(
            [total_points_slice * chunk_size for chunk_size in section_sizes]
        ).astype(int)
        print("type chunk_points", type(chunk_points))
        print("chunk_points", chunk_points)
        
        # As randomness is uniform
        per_chunk_obs = np.rint(sparsity * chunk_points).astype(int)
        for idx, c in enumerate(per_chunk_obs):
            if c > chunk_points[idx]:
                per_chunk_obs[idx] = chunk_points[idx]

        mp_obs = per_chunk_obs.sum()
        if not mp_obs.is_integer():
            raise ValueError(f"Got non integer value of observations {mp_obs}.")
        mp_obs = int(mp_obs)
        sparsity_new = mp_obs / total_points
        print(f"Multiprocessing approximations lead to {mp_obs} total observation (goal: {total_obs}).")
        print(f"Updated sparsity is {sparsity_new} (was: {sparsity}).")

        return mp_obs, sparsity_new, per_chunk_obs


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

        task_shape = shape.copy()
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
        
    
