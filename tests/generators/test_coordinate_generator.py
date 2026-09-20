"""Tests for CoordinateGenerator class."""

import pytest
import numpy as np
from data_sparsity.generators.coordinate_generator import CoordinateGenerator
from data_sparsity.utils.chunk_utils import ChunkUtils


class TestGenerateAllCoords:
    """Tests for generate_all_coords method."""
    
    def test_1d_grid(self, fixed_rng):
        """Should generate 1D coordinate grid."""
        shape = (10,)
        coords = CoordinateGenerator.generate_all_coords(shape, fixed_rng)
        assert isinstance(coords,dict)
        assert len(coords.keys()) == 1
        assert len(coords["x0"]) == 10
    
    def test_2d_grid(self, fixed_rng):
        """Should generate 2D coordinate grid."""
        shape = (5, 8)
        coords = CoordinateGenerator.generate_all_coords(shape, fixed_rng)
        assert isinstance(coords,dict)
        assert len(coords.keys()) == 2
        assert len(coords["x0"]) == 5
        assert len(coords["x1"]) == 8
    
    def test_3d_grid(self, fixed_rng):
        """Should generate 3D coordinate grid."""
        shape = (4, 6, 3)
        coords = CoordinateGenerator.generate_all_coords(shape, fixed_rng)
        assert isinstance(coords,dict)
        assert len(coords.keys()) == 3
        assert len(coords["x0"]) == 4
        assert len(coords["x1"]) == 6
        assert len(coords["x2"]) == 3
        
    def test_list_length_matches_shape_length(self, fixed_rng):
        """Should return list with length matching number of dimensions."""
        shape = (2, 3, 4, 5)
        coords = CoordinateGenerator.generate_all_coords(shape, fixed_rng)
        assert len(coords.keys()) == len(shape)
    
    def test_each_array_correct_length(self, fixed_rng):
        """Should generate correct length for each dimension."""
        shape = (7, 11, 13)
        coords = CoordinateGenerator.generate_all_coords(shape, fixed_rng)
        for i, expected_len in enumerate(shape):
            assert len(coords[f"x{i}"]) == expected_len
    
    def test_all_arrays_sorted(self, fixed_rng):
        """Should return sorted arrays for all dimensions."""
        shape = (10, 15, 20)
        coords = CoordinateGenerator.generate_all_coords(shape, fixed_rng)
        for coord_array in coords.values():
            assert np.all(coord_array[:-1] <= coord_array[1:])
    
    def test_reproducible_with_seed(self):
        """Should produce same results with same seed."""
        shape = (5, 7)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        coords1 = CoordinateGenerator.generate_all_coords(shape, rng1)
        coords2 = CoordinateGenerator.generate_all_coords(shape, rng2)
        for c1, c2 in zip(coords1, coords2):
            np.testing.assert_array_equal(c1, c2)


class TestRNGAlignment:
    """Tests for serial/parallel RNG alignment via ChunkUtils.generate_rngs."""
    
    def test_serial_mode_generates_per_dimension_rngs(self):
        """Serial mode should generate per-dimension RNGs with base seed formula."""
        seed = 42
        num_dims = 3
        
        dim_rngs = ChunkUtils.generate_rngs(seed=seed, num_dims=num_dims)
        
        # Should return dict with one RNG per dimension
        assert isinstance(dim_rngs, dict)
        assert len(dim_rngs) == num_dims
        assert all(dim_idx in dim_rngs for dim_idx in range(num_dims))
    
    def test_serial_coordinates_match_across_calls(self):
        """Serial mode should produce identical coordinates with same seed."""
        seed = 42
        num_dims = 2
        shape = (5, 7)
        
        # First generation
        dim_rngs1 = ChunkUtils.generate_rngs(seed=seed, num_dims=num_dims)
        coords1 = CoordinateGenerator.generate_all_coords(
            shape, rng=None, dim_ranges=None, dim_rngs=dim_rngs1
        )
        
        # Second generation
        dim_rngs2 = ChunkUtils.generate_rngs(seed=seed, num_dims=num_dims)
        coords2 = CoordinateGenerator.generate_all_coords(
            shape, rng=None, dim_ranges=None, dim_rngs=dim_rngs2
        )
        
        # Should match exactly
        for dim_name in coords1:
            np.testing.assert_array_equal(coords1[dim_name], coords2[dim_name])
    
    def test_parallel_non_split_dims_match_serial(self):
        """Non-split dimensions in parallel should match serial mode."""
        seed = 42
        num_dims = 3
        dim_split = 0  # Split on x0
        shape = (10, 5, 7)
        
        # Generate serial coordinates
        serial_dim_rngs = ChunkUtils.generate_rngs(seed=seed, num_dims=num_dims)
        serial_coords = CoordinateGenerator.generate_all_coords(
            shape, rng=None, dim_ranges=None, dim_rngs=serial_dim_rngs
        )
        
        # Generate parallel coordinates (chunk 0 of 3)
        parallel_dim_rngs = ChunkUtils.generate_rngs(
            seed=seed,
            num_dims=num_dims,
            chunk_id=0,
            dim_split=dim_split,
            obs_in_chunk=100
        )
        # Use same shape for non-split dims (chunk splits x0, so x1 and x2 stay full)
        parallel_shape = (3, 5, 7)  # x0 is smaller in chunk
        parallel_coords = CoordinateGenerator.generate_all_coords(
            parallel_shape, rng=None, dim_ranges=None, dim_rngs=parallel_dim_rngs
        )
        
        # Non-split dimensions (x1, x2) should match serial
        np.testing.assert_array_equal(serial_coords['x1'], parallel_coords['x1'])
        np.testing.assert_array_equal(serial_coords['x2'], parallel_coords['x2'])
    
    def test_parallel_split_dim_advances_correctly(self):
        """Split dimension RNG should advance for later chunks."""
        seed = 42
        num_dims = 2
        dim_split = 0
        obs_in_chunk = 3
        
        # Generate RNGs for chunks 0, 1, 2
        chunk0_rngs = ChunkUtils.generate_rngs(
            seed=seed, num_dims=num_dims, chunk_id=0,
            dim_split=dim_split, obs_in_chunk=obs_in_chunk
        )
        chunk1_rngs = ChunkUtils.generate_rngs(
            seed=seed, num_dims=num_dims, chunk_id=1,
            dim_split=dim_split, obs_in_chunk=obs_in_chunk
        )
        chunk2_rngs = ChunkUtils.generate_rngs(
            seed=seed, num_dims=num_dims, chunk_id=2,
            dim_split=dim_split, obs_in_chunk=obs_in_chunk
        )
        
        # Generate single coordinate values from split dimension
        x0_chunk0 = chunk0_rngs[0].uniform(0, 1, 1)
        x0_chunk1 = chunk1_rngs[0].uniform(0, 1, 1)
        x0_chunk2 = chunk2_rngs[0].uniform(0, 1, 1)
        
        # All should be different (chunk1 advanced by 3, chunk2 by 6)
        assert not np.isclose(x0_chunk0[0], x0_chunk1[0])
        assert not np.isclose(x0_chunk0[0], x0_chunk2[0])
        assert not np.isclose(x0_chunk1[0], x0_chunk2[0])
        
        # Non-split dimension should be identical across chunks
        x1_chunk0 = chunk0_rngs[1].uniform(0, 1, 5)
        x1_chunk1 = chunk1_rngs[1].uniform(0, 1, 5)
        x1_chunk2 = chunk2_rngs[1].uniform(0, 1, 5)
        
        np.testing.assert_array_equal(x1_chunk0, x1_chunk1)
        np.testing.assert_array_equal(x1_chunk0, x1_chunk2)
    
    def test_parallel_chunks_reconstruct_serial_split_dim(self):
        """Parallel chunks should have deterministic values from advanced RNG."""
        seed = 42
        num_dims = 2
        dim_split = 0
        obs_in_chunk = 1  # 1 coordinate per chunk
        
        # Generate serial x0 RNG values (unsorted, to match parallel generation)
        serial_rngs = ChunkUtils.generate_rngs(seed=seed, num_dims=num_dims)
        serial_x0_values = serial_rngs[0].uniform(0, 1, 3)  # Unsorted
        
        # Generate parallel chunks with RNG advancement
        chunks_x0 = []
        for chunk_id in range(3):
            chunk_rngs = ChunkUtils.generate_rngs(
                seed=seed, num_dims=num_dims, chunk_id=chunk_id,
                dim_split=dim_split, obs_in_chunk=obs_in_chunk
            )
            # Generate one value per chunk (matches serial sequence position)
            chunk_x0 = chunk_rngs[0].uniform(0, 1, 1)
            chunks_x0.append(chunk_x0[0])
        
        # Parallel chunks should match serial RNG sequence
        for i, chunk_val in enumerate(chunks_x0):
            np.testing.assert_array_equal(
                chunk_val, serial_x0_values[i],
                err_msg=f"Chunk {i} x0 doesn't match serial RNG position {i}"
            )
    
    def test_empty_parallel_params_behaves_as_serial(self):
        """Calling generate_rngs with None parallel params should match serial."""
        seed = 42
        num_dims = 3
        
        # Serial explicit call
        serial_rngs = ChunkUtils.generate_rngs(seed=seed, num_dims=num_dims)
        
        # Parallel call with all None (should behave as serial)
        parallel_rngs = ChunkUtils.generate_rngs(
            seed=seed, num_dims=num_dims,
            chunk_id=None, dim_split=None, obs_in_chunk=None
        )
        
        # Generate some coordinates to compare RNG state
        for dim_idx in range(num_dims):
            serial_vals = serial_rngs[dim_idx].uniform(0, 1, 10)
            parallel_vals = parallel_rngs[dim_idx].uniform(0, 1, 10)
            np.testing.assert_array_equal(serial_vals, parallel_vals)
