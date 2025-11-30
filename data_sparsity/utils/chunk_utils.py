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
            chunk_id: int
    ) -> tuple[np.random.Generator,np.random.Generator]:
        """
        Generate two distinct generators:
        global_rng: identical across tasks, used for coordinates along non-split dimensions
        task_rng: unique per task, used for split dimension coordinates and observations
        """
        global_rng = np.random.default_rng(seed)
        task_rng = np.random.default_rng(seed + chunk_id)

        return global_rng, task_rng

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
            global_rng: np.random.Generator,
            task_rng: np.random.Generator
    ) -> dict:
        """
        Assign global or task-based RNG to different dimensions.
        Non-split dimensions use [0, 1) with global_rng.
        Split dimension uses normalized chunk range with task_rng.
        """

        dim_rngs = {}
        for idx in range(len(task_shape)):
            if idx == dim_split:
                dim_rngs[idx] = task_rng
            else:
                dim_rngs[idx] = global_rng

        return dim_rngs
        
    
