"""Base record generation functionality.

This module provides the base class for generating sparse record arrays
with common utilities for index generation and assignment.
"""

from typing import List, Tuple
import numpy as np


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
    def generate_flat_indices(
        total_points: int,
        num_obs: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Generate random flat indices for observation placement.
        
        Args:
            total_points: Total number of grid points
            num_obs: Number of observations to place
            rng: Random number generator
            
        Returns:
            Array of flat indices
        """
        return rng.choice(total_points, size=num_obs, replace=False)

    @staticmethod
    def convert_to_multi_indices(
        flat_indices: np.ndarray,
        shape: List[int]
    ) -> Tuple:
        """Convert flat indices to multi-dimensional indices.
        
        Args:
            flat_indices: Flat (1D) indices
            shape: Shape of the multi-dimensional array
            
        Returns:
            Tuple of index arrays for each dimension
        """
        return np.unravel_index(flat_indices, shape)

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
        indices = []
        
        for dim_size in shape:
            if dim_size == n_s:
                # Full permutation: each coordinate used exactly once
                perm = rng.permutation(dim_size)
                indices.append(perm)
            elif dim_size > n_s:
                # Dimension larger than n_s: randomly select n_s coordinates
                # Each coordinate has equal probability n_s/dim_size of selection
                selected = rng.choice(dim_size, size=n_s, replace=False)
                rng.shuffle(selected)  # Randomize order
                indices.append(selected)
            else:
                # dim_size < n_s: Invalid configuration
                raise ValueError(
                    f"Dimension size {dim_size} < n_s {n_s}. "
                    f"n_s should equal min(shape) = {min(shape)}"
                )
        
        return tuple(indices)

    @staticmethod
    def generate_hybrid_indices(
        shape: List[int],
        num_obs: int,
        rng: np.random.Generator
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
        min_dim_size = min(shape)
        # n_s is the number of LHS samples - cannot exceed num_obs
        n_s = min(num_obs, min_dim_size)  # LHS base coverage
        n_random = max(0, num_obs - n_s)  # Additional random points
        
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
        available_positions = np.array([i for i in range(total_points) if i not in lhs_set])
        
        # Sample from available positions
        if len(available_positions) >= n_random:
            # Enough positions available: sample without replacement
            random_flat = rng.choice(available_positions, size=n_random, replace=False)
        else:
            # Not enough positions: use all available, then sample remaining with replacement
            # This can happen when num_obs approaches or exceeds total_points
            random_flat = np.concatenate([
                available_positions,
                rng.choice(total_points, size=n_random - len(available_positions), replace=True)
            ])
        
        random_indices = np.unravel_index(random_flat, shape)
        
        # Combine LHS and random indices
        combined_indices = tuple(
            np.concatenate([lhs_indices[i], random_indices[i]])
            for i in range(len(shape))
        )
        
        return combined_indices

    @staticmethod
    def validate_density(
        num_obs: int,
        total_grid_points: int,
        expected_density: float
    ) -> None:
        """Validate that computed density matches expected value.
        
        Args:
            num_obs: Number of observations
            total_grid_points: Total grid points
            expected_density: Expected density value
            
        Raises:
            ValueError: If computed density doesn't match expected
        """
        computed_density = num_obs / total_grid_points
        if not np.isclose(computed_density, expected_density):
            raise ValueError(
                f"Density {computed_density} determined from number "
                f"of coordinates differs from expected density {expected_density}"
            )

    @staticmethod

    @staticmethod
    def generate_global_lhs_indices_for_chunk(
        global_shape: List[int],
        num_obs_global: int,
        rng: np.random.Generator,
        chunk_id: int,
        div_points: List[int],
        dim_split: int
    ) -> Tuple[np.ndarray, ...]:
        """Generate global LHS indices and filter to chunk range.
        
        This method maintains global LHS property in parallel execution by:
        1. Generating complete LHS index set for global shape
        2. Filtering to retain only indices within chunk's coordinate range
        3. Mapping global indices to chunk-local coordinates
        
        This ensures that each coordinate in the global space is used exactly
        once across all chunks combined, preserving the Latin Hypercube Sampling
        guarantee even in parallel execution.
        
        Args:
            global_shape: Full dataset shape [n0, n1, ..., nk]
            num_obs_global: Total observations across all chunks
            rng: Random number generator (pre-advanced for this chunk)
            chunk_id: Current chunk index (for coordinate range)
            div_points: Chunk boundaries along split dimension [0, b1, b2, ..., max]
            dim_split: Index of dimension being split across chunks
            
        Returns:
            Tuple of index arrays in chunk-local coordinates, one per dimension
            
        Example:
            >>> # Global: shape=[5,5], 5 obs, split into 3 chunks at dim 0
            >>> # Chunk 0 covers x0 ∈ [0, 2), Chunk 1: [2, 4), Chunk 2: [4, 5)
            >>> global_shape = [5, 5]
            >>> div_points = [0, 2, 4, 5]
            >>> rng = np.random.default_rng(42)
            >>> 
            >>> # Chunk 0
            >>> indices = RecordGenerator.generate_global_lhs_indices_for_chunk(
            ...     global_shape, 5, rng, chunk_id=0, div_points, dim_split=0
            ... )
            >>> # Returns observations with x0 ∈ [0,2), mapped to [0,2) local coords
        """
        # Generate full global LHS indices
        global_indices = RecordGenerator.generate_hybrid_indices(
            shape=global_shape,
            num_obs=num_obs_global,
            rng=rng
        )
        
        # Determine chunk's coordinate range for split dimension
        chunk_start = div_points[chunk_id]
        chunk_end = div_points[chunk_id + 1]
        
        # Filter to chunk range
        # Keep only observations where split dimension index falls in [chunk_start, chunk_end)
        split_dim_indices = global_indices[dim_split]
        mask = (split_dim_indices >= chunk_start) & (split_dim_indices < chunk_end)
        
        # Apply mask to all dimensions
        filtered_indices = tuple(
            dim_indices[mask] for dim_indices in global_indices
        )
        
        # Map split dimension to chunk-local coordinates
        chunk_local_indices = list(filtered_indices)
        chunk_local_indices[dim_split] = filtered_indices[dim_split] - chunk_start
        
        return tuple(chunk_local_indices)
