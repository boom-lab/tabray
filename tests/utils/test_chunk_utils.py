"""Tests for ChunkUtils class.

This module tests the parallel workflow utilities used for chunked dataset
generation, including chunk size calculation, RNG generation, shape updates,
and dimension range handling.
"""

import pytest
import numpy as np
from data_sparsity.utils.chunk_utils import ChunkUtils


class TestGetObservationsPerChunk:
    """Tests for get_observations_per_chunk method."""

    def test_uniform_distribution_across_chunks(self):
        """Should distribute observations uniformly across equal-sized chunks."""
        total_obs = 1000
        shape = np.array([100, 100])
        max_dim_size = 100
        section_sizes = [25, 25, 25, 25]  # Four equal chunks
        sparsity = 0.1

        mp_obs, sparsity_new, per_chunk_obs = ChunkUtils.get_observations_per_chunk(
            total_obs, shape, max_dim_size, section_sizes, sparsity
        )

        # Each chunk should have roughly equal observations
        assert len(per_chunk_obs) == 4
        assert all(isinstance(obs, (int, np.integer)) for obs in per_chunk_obs)
        # Each chunk represents 25% of the data, so 25% of total_points
        expected_per_chunk = int(np.rint(sparsity * (np.prod(shape) / 4)))
        assert all(obs == expected_per_chunk for obs in per_chunk_obs)

    def test_non_uniform_chunk_sizes(self):
        """Should handle non-uniform chunk sizes correctly."""
        total_obs = 1000
        shape = np.array([100, 100])
        max_dim_size = 100
        section_sizes = [10, 20, 30, 40]  # Non-uniform chunks
        sparsity = 0.1

        mp_obs, sparsity_new, per_chunk_obs = ChunkUtils.get_observations_per_chunk(
            total_obs, shape, max_dim_size, section_sizes, sparsity
        )

        # Observations should be proportional to chunk size
        assert len(per_chunk_obs) == 4
        # Larger chunks should have more observations
        assert per_chunk_obs[0] < per_chunk_obs[1] < per_chunk_obs[2] < per_chunk_obs[3]

    def test_sparsity_adjustment(self):
        """Should return adjusted sparsity based on actual observations."""
        total_obs = 1000
        shape = np.array([100, 100])
        max_dim_size = 100
        section_sizes = [50, 50]
        sparsity = 0.1

        mp_obs, sparsity_new, per_chunk_obs = ChunkUtils.get_observations_per_chunk(
            total_obs, shape, max_dim_size, section_sizes, sparsity
        )

        # New sparsity should be based on actual total observations
        total_points = np.prod(shape)
        expected_sparsity = mp_obs / total_points
        assert np.isclose(sparsity_new, expected_sparsity)

    def test_observation_count_validation(self):
        """Should return integer observation counts."""
        total_obs = 1000
        shape = np.array([100, 100])
        max_dim_size = 100
        section_sizes = [25, 25, 25, 25]
        sparsity = 0.1

        mp_obs, sparsity_new, per_chunk_obs = ChunkUtils.get_observations_per_chunk(
            total_obs, shape, max_dim_size, section_sizes, sparsity
        )

        # All values should be integers
        assert isinstance(mp_obs, (int, np.integer))
        assert all(isinstance(obs, (int, np.integer)) for obs in per_chunk_obs)

    def test_edge_case_very_small_chunks(self):
        """Should handle very small chunks correctly."""
        total_obs = 100
        shape = np.array([100, 100])
        max_dim_size = 100
        section_sizes = [1, 1, 1, 97]  # Very small chunks
        sparsity = 0.01

        mp_obs, sparsity_new, per_chunk_obs = ChunkUtils.get_observations_per_chunk(
            total_obs, shape, max_dim_size, section_sizes, sparsity
        )

        # Should handle small chunks without errors
        assert len(per_chunk_obs) == 4
        assert mp_obs >= 0
        assert sparsity_new >= 0

    def test_edge_case_very_large_chunks(self):
        """Should handle very large chunks correctly."""
        total_obs = 10000
        shape = np.array([1000, 1000])
        max_dim_size = 1000
        section_sizes = [250, 250, 250, 250]
        sparsity = 0.01

        mp_obs, sparsity_new, per_chunk_obs = ChunkUtils.get_observations_per_chunk(
            total_obs, shape, max_dim_size, section_sizes, sparsity
        )

        # Should handle large values without overflow
        assert mp_obs > 0
        assert sum(per_chunk_obs) == mp_obs

    def test_capping_observations_at_chunk_size(self):
        """Should cap observations at chunk_points when sparsity is too high."""
        total_obs = 100000
        shape = np.array([100, 100])
        max_dim_size = 100
        section_sizes = [25, 25, 25, 25]
        sparsity = 0.9  # Very high sparsity

        mp_obs, sparsity_new, per_chunk_obs = ChunkUtils.get_observations_per_chunk(
            total_obs, shape, max_dim_size, section_sizes, sparsity
        )

        # No chunk should have more observations than available points
        total_points = np.prod(shape)
        chunk_points = total_points / len(section_sizes)
        for obs in per_chunk_obs:
            assert obs <= chunk_points

    def test_sum_equals_returned_total(self):
        """Should ensure sum of per_chunk_obs equals mp_obs."""
        total_obs = 1000
        shape = np.array([100, 100])
        max_dim_size = 100
        section_sizes = [20, 30, 50]
        sparsity = 0.15

        mp_obs, sparsity_new, per_chunk_obs = ChunkUtils.get_observations_per_chunk(
            total_obs, shape, max_dim_size, section_sizes, sparsity
        )

        # Sum should equal total
        assert sum(per_chunk_obs) == mp_obs


class TestUpdateChunkShape:
    """Tests for update_chunk_shape method."""

    def test_shape_reduction_along_split_dimension(self):
        """Should reduce shape along specified dimension."""
        shape = np.array([100, 200, 300])
        dim_split = 1
        task_size = 50

        task_shape = ChunkUtils.update_chunk_shape(shape, dim_split, task_size)

        # Should update only the split dimension
        assert task_shape[0] == 100
        assert task_shape[1] == 50
        assert task_shape[2] == 300

    def test_first_dimension_split(self):
        """Should handle split on first dimension."""
        shape = np.array([100, 200])
        dim_split = 0
        task_size = 25

        task_shape = ChunkUtils.update_chunk_shape(shape, dim_split, task_size)

        assert task_shape[0] == 25
        assert task_shape[1] == 200

    def test_last_dimension_split(self):
        """Should handle split on last dimension."""
        shape = np.array([100, 200, 300])
        dim_split = 2
        task_size = 75

        task_shape = ChunkUtils.update_chunk_shape(shape, dim_split, task_size)

        assert task_shape[0] == 100
        assert task_shape[1] == 200
        assert task_shape[2] == 75

    def test_2d_shape(self):
        """Should work with 2D shapes."""
        shape = np.array([50, 100])
        dim_split = 1
        task_size = 20

        task_shape = ChunkUtils.update_chunk_shape(shape, dim_split, task_size)

        assert len(task_shape) == 2
        assert task_shape[0] == 50
        assert task_shape[1] == 20

    def test_3d_shape(self):
        """Should work with 3D shapes."""
        shape = np.array([50, 100, 150])
        dim_split = 1
        task_size = 30

        task_shape = ChunkUtils.update_chunk_shape(shape, dim_split, task_size)

        assert len(task_shape) == 3
        assert task_shape[1] == 30

    def test_edge_case_single_element_dimension(self):
        """Should handle single-element task size."""
        shape = np.array([100, 200])
        dim_split = 0
        task_size = 1

        task_shape = ChunkUtils.update_chunk_shape(shape, dim_split, task_size)

        assert task_shape[0] == 1
        assert task_shape[1] == 200


class TestValidateChunkPoints:
    """Tests for validate_chunk_points method."""

    def test_valid_integer_shape(self):
        """Should return total points for valid shape."""
        task_shape = np.array([10, 20, 30])

        total_points = ChunkUtils.validate_chunk_points(task_shape)

        assert total_points == 6000
        assert isinstance(total_points, int)

    def test_2d_shape(self):
        """Should handle 2D shapes."""
        task_shape = np.array([50, 100])

        total_points = ChunkUtils.validate_chunk_points(task_shape)

        assert total_points == 5000

    def test_3d_shape(self):
        """Should handle 3D shapes."""
        task_shape = np.array([10, 20, 30])

        total_points = ChunkUtils.validate_chunk_points(task_shape)

        assert total_points == 6000

    def test_returns_integer(self):
        """Should return integer type."""
        task_shape = np.array([5, 5])

        total_points = ChunkUtils.validate_chunk_points(task_shape)

        assert isinstance(total_points, int)

    def test_single_dimension(self):
        """Should handle single dimension (though unlikely in practice)."""
        task_shape = np.array([100])

        total_points = ChunkUtils.validate_chunk_points(task_shape)

        assert total_points == 100


class TestGenerateSplitDimensionRange:
    """Tests for generate_split_dimension_range method."""

    def test_range_normalization(self):
        """Should normalize range by dimension size."""
        dim_split = 1
        task_range = (0, 50)
        dim_size = 100

        dim_ranges = ChunkUtils.generate_split_dimension_range(
            dim_split, task_range, dim_size
        )

        # Should contain only the split dimension
        assert len(dim_ranges) == 1
        assert dim_split in dim_ranges
        # Range should be normalized to [0, 0.5)
        assert dim_ranges[dim_split] == (0.0, 0.5)

    def test_different_dimension_indices(self):
        """Should work with any dimension index."""
        dim_split = 2
        task_range = (25, 75)
        dim_size = 100

        dim_ranges = ChunkUtils.generate_split_dimension_range(
            dim_split, task_range, dim_size
        )

        assert 2 in dim_ranges
        assert dim_ranges[2] == (0.25, 0.75)

    def test_full_range(self):
        """Should handle full dimension range."""
        dim_split = 0
        task_range = (0, 100)
        dim_size = 100

        dim_ranges = ChunkUtils.generate_split_dimension_range(
            dim_split, task_range, dim_size
        )

        assert dim_ranges[0] == (0.0, 1.0)

    def test_edge_case_boundaries(self):
        """Should handle edge cases at boundaries."""
        dim_split = 1
        task_range = (80, 100)
        dim_size = 100

        dim_ranges = ChunkUtils.generate_split_dimension_range(
            dim_split, task_range, dim_size
        )

        assert dim_ranges[1] == (0.8, 1.0)

    def test_returns_dict(self):
        """Should return dictionary with single entry."""
        dim_split = 0
        task_range = (10, 20)
        dim_size = 50

        dim_ranges = ChunkUtils.generate_split_dimension_range(
            dim_split, task_range, dim_size
        )

        assert isinstance(dim_ranges, dict)
        assert len(dim_ranges) == 1


class TestGenerateRngs:
    """Tests for generate_rngs method."""

    def test_serial_mode_basic(self):
        """Should generate RNGs for serial mode (no chunk_id)."""
        seed = 42
        num_dims = 3
        
        rngs = ChunkUtils.generate_rngs(seed, num_dims)
        
        # Should return dict with RNG for each dimension
        assert isinstance(rngs, dict)
        assert len(rngs) == num_dims
        assert all(dim_idx in rngs for dim_idx in range(num_dims))
        assert all(isinstance(rng, np.random.Generator) for rng in rngs.values())
    
    def test_serial_mode_deterministic_seeding(self):
        """Should use deterministic seed offsets per dimension."""
        seed = 42
        num_dims = 3
        
        rngs1 = ChunkUtils.generate_rngs(seed, num_dims)
        rngs2 = ChunkUtils.generate_rngs(seed, num_dims)
        
        # Same seed should produce RNGs that generate same values
        for dim_idx in range(num_dims):
            val1 = rngs1[dim_idx].uniform(0, 1, 5)
            val2 = rngs2[dim_idx].uniform(0, 1, 5)
            np.testing.assert_array_equal(val1, val2)
    
    def test_parallel_mode_chunk_0(self):
        """Should generate RNGs for first chunk (no advancement)."""
        seed = 42
        num_dims = 3
        chunk_id = 0
        dim_split = 1
        obs_in_chunk = 10
        
        rngs = ChunkUtils.generate_rngs(
            seed, num_dims, chunk_id, dim_split, obs_in_chunk
        )
        
        # Should return dict with RNG for each dimension
        assert isinstance(rngs, dict)
        assert len(rngs) == num_dims
        
        # Chunk 0 should match serial mode for first draws
        rngs_serial = ChunkUtils.generate_rngs(seed, num_dims)
        
        # Non-split dimensions should be identical
        for dim_idx in range(num_dims):
            if dim_idx != dim_split:
                val_par = rngs[dim_idx].uniform(0, 1, 5)
                val_ser = rngs_serial[dim_idx].uniform(0, 1, 5)
                np.testing.assert_array_equal(val_par, val_ser)
    
    def test_parallel_mode_chunk_advancement(self):
        """Should advance RNG state for split dimension in later chunks."""
        seed = 42
        num_dims = 2
        dim_split = 0
        obs_in_chunk = 10
        
        # Generate RNGs for chunk 0 and chunk 1
        rngs_chunk0 = ChunkUtils.generate_rngs(
            seed, num_dims, chunk_id=0, dim_split=dim_split, obs_in_chunk=obs_in_chunk
        )
        rngs_chunk1 = ChunkUtils.generate_rngs(
            seed, num_dims, chunk_id=1, dim_split=dim_split, obs_in_chunk=obs_in_chunk
        )
        
        # Split dimension RNG for chunk 1 should be advanced
        # Generate values from split dimension
        vals_chunk0 = rngs_chunk0[dim_split].uniform(0, 1, 5)
        vals_chunk1 = rngs_chunk1[dim_split].uniform(0, 1, 5)
        
        # These should differ (chunk1 is advanced)
        assert not np.array_equal(vals_chunk0, vals_chunk1)
        
        # Non-split dimension should be identical
        vals_chunk0_nonsplit = rngs_chunk0[1].uniform(0, 1, 5)
        vals_chunk1_nonsplit = rngs_chunk1[1].uniform(0, 1, 5)
        np.testing.assert_array_equal(vals_chunk0_nonsplit, vals_chunk1_nonsplit)
    
    def test_parallel_mode_alignment_with_serial(self):
        """Should generate values that align with serial mode when concatenated."""
        seed = 42
        num_dims = 2
        dim_split = 0
        obs_in_chunk = 10
        num_chunks = 3
        
        # Serial mode: generate all values
        rng_serial = ChunkUtils.generate_rngs(seed, num_dims)
        serial_vals = rng_serial[dim_split].uniform(0, 1, obs_in_chunk * num_chunks)
        
        # Parallel mode: generate from each chunk
        parallel_vals = []
        for chunk_id in range(num_chunks):
            rng_chunk = ChunkUtils.generate_rngs(
                seed, num_dims, chunk_id, dim_split, obs_in_chunk
            )
            chunk_vals = rng_chunk[dim_split].uniform(0, 1, obs_in_chunk)
            parallel_vals.extend(chunk_vals)
        
        # Concatenated parallel values should match serial
        np.testing.assert_array_almost_equal(serial_vals, parallel_vals)
    
    def test_dimension_seed_offsets(self):
        """Should use 1000-step seed offsets between dimensions."""
        seed = 42
        num_dims = 3
        
        rngs = ChunkUtils.generate_rngs(seed, num_dims)
        
        # Each dimension should produce different sequences
        vals = [rngs[i].uniform(0, 1, 5) for i in range(num_dims)]
        
        # All dimensions should produce different values
        for i in range(num_dims):
            for j in range(i+1, num_dims):
                assert not np.array_equal(vals[i], vals[j])
    
    def test_parallel_mode_none_obs_in_chunk(self):
        """Should handle None obs_in_chunk gracefully."""
        seed = 42
        num_dims = 2
        chunk_id = 1
        dim_split = 0
        
        # This should work without obs_in_chunk (no advancement)
        rngs = ChunkUtils.generate_rngs(
            seed, num_dims, chunk_id, dim_split, obs_in_chunk=None
        )
        
        assert isinstance(rngs, dict)
        assert len(rngs) == num_dims


class TestAssignRngsToDimensions:
    """Tests for assign_rngs_to_dimensions method."""

    def test_returns_input_dict(self):
        """Should return the input RNG dict (pass-through)."""
        seed = 42
        num_dims = 3
        dim_rngs = ChunkUtils.generate_rngs(seed, num_dims)
        
        result = ChunkUtils.assign_rngs_to_dimensions(
            dim_split=0,
            task_shape=np.array([10, 20, 30]),
            dim_rngs_dict=dim_rngs
        )
        
        # Should return same dict
        assert result is dim_rngs
        assert len(result) == num_dims
    
    def test_preserves_rng_state(self):
        """Should not modify RNG states."""
        seed = 42
        num_dims = 2
        dim_rngs = ChunkUtils.generate_rngs(seed, num_dims)
        
        # Generate some values
        vals_before = [dim_rngs[i].uniform(0, 1, 3) for i in range(num_dims)]
        
        # Call assign_rngs_to_dimensions
        result = ChunkUtils.assign_rngs_to_dimensions(
            dim_split=0,
            task_shape=np.array([10, 20]),
            dim_rngs_dict=dim_rngs
        )
        
        # RNGs should still be in the same state (generate next values)
        vals_after = [result[i].uniform(0, 1, 3) for i in range(num_dims)]
        
        # Should not match before values (state advanced)
        for i in range(num_dims):
            assert not np.array_equal(vals_before[i], vals_after[i])


class TestCalculateLhsDrawsPerGeneration:
    """Tests for calculate_lhs_draws_per_generation method."""

    def test_equal_dimensions(self):
        """Should calculate draws for equal-sized dimensions."""
        shape = [5, 5, 5]
        n_s = 5
        
        total_draws = ChunkUtils.calculate_lhs_draws_per_generation(shape, n_s)
        
        # Each dimension: permutation(5) = 5 draws
        # Total: 3 * 5 = 15
        assert total_draws == 15
    
    def test_larger_dimensions(self):
        """Should calculate draws for dimensions larger than n_s."""
        shape = [5, 7, 10]
        n_s = 5
        
        total_draws = ChunkUtils.calculate_lhs_draws_per_generation(shape, n_s)
        
        # Dim 0: permutation(5) = 5 draws
        # Dim 1: choice(7, 5) + shuffle(5) = 5 + 5 = 10 draws
        # Dim 2: choice(10, 5) + shuffle(5) = 5 + 5 = 10 draws
        # Total: 5 + 10 + 10 = 25
        assert total_draws == 25
    
    def test_mixed_dimensions(self):
        """Should handle mixed dimension sizes."""
        shape = [10, 10, 10]
        n_s = 10
        
        total_draws = ChunkUtils.calculate_lhs_draws_per_generation(shape, n_s)
        
        # Each dimension: permutation(10) = 10 draws
        # Total: 3 * 10 = 30
        assert total_draws == 30
    
    def test_2d_shape(self):
        """Should work with 2D shapes."""
        shape = [8, 5]
        n_s = 5
        
        total_draws = ChunkUtils.calculate_lhs_draws_per_generation(shape, n_s)
        
        # Dim 0: choice(8, 5) + shuffle(5) = 10 draws
        # Dim 1: permutation(5) = 5 draws
        # Total: 15
        assert total_draws == 15
    
    def test_invalid_dimension_smaller_than_n_s(self):
        """Should raise ValueError if any dimension < n_s."""
        shape = [5, 3, 7]  # Dim 1 is < 5
        n_s = 5
        
        with pytest.raises(ValueError, match="dim_size 3 < n_s 5"):
            ChunkUtils.calculate_lhs_draws_per_generation(shape, n_s)
    
    def test_single_dimension(self):
        """Should work with single dimension."""
        shape = [10]
        n_s = 5
        
        total_draws = ChunkUtils.calculate_lhs_draws_per_generation(shape, n_s)
        
        # choice(10, 5) + shuffle(5) = 10 draws
        assert total_draws == 10
    
    def test_deterministic(self):
        """Should return same value for same inputs."""
        shape = [7, 8, 9]
        n_s = 6
        
        draws1 = ChunkUtils.calculate_lhs_draws_per_generation(shape, n_s)
        draws2 = ChunkUtils.calculate_lhs_draws_per_generation(shape, n_s)
        
        assert draws1 == draws2


class TestGenerateLhsRng:
    """Tests for generate_lhs_rng method."""

    def test_serial_mode_basic(self):
        """Should generate LHS RNG for serial mode."""
        seed = 42
        shape = [10, 10]
        
        lhs_rng = ChunkUtils.generate_lhs_rng(seed, shape)
        
        assert isinstance(lhs_rng, np.random.Generator)
    
    def test_serial_mode_deterministic(self):
        """Should produce deterministic RNG with same seed."""
        seed = 42
        shape = [10, 10]
        
        lhs_rng1 = ChunkUtils.generate_lhs_rng(seed, shape)
        lhs_rng2 = ChunkUtils.generate_lhs_rng(seed, shape)
        
        # Should generate identical sequences
        vals1 = lhs_rng1.uniform(0, 1, 10)
        vals2 = lhs_rng2.uniform(0, 1, 10)
        
        np.testing.assert_array_equal(vals1, vals2)
    
    def test_parallel_mode_no_advancement(self):
        """Should not advance RNG in parallel mode (global LHS approach)."""
        seed = 42
        shape = [10, 10]
        chunk_id = 0
        num_obs_global = 50
        
        lhs_rng = ChunkUtils.generate_lhs_rng(seed, shape, chunk_id, num_obs_global)
        
        assert isinstance(lhs_rng, np.random.Generator)
        
        # Should match serial mode (no advancement with global LHS)
        lhs_rng_serial = ChunkUtils.generate_lhs_rng(seed, shape)
        
        vals_par = lhs_rng.uniform(0, 1, 10)
        vals_ser = lhs_rng_serial.uniform(0, 1, 10)
        
        np.testing.assert_array_equal(vals_par, vals_ser)
    
    def test_parallel_mode_all_chunks_identical(self):
        """Should generate identical RNG for all chunks (global LHS)."""
        seed = 42
        shape = [10, 10]
        num_obs_global = 50
        
        # Generate RNGs for multiple chunks
        rngs = []
        for chunk_id in range(3):
            rng = ChunkUtils.generate_lhs_rng(seed, shape, chunk_id, num_obs_global)
            rngs.append(rng)
        
        # All should generate identical sequences
        vals = [rng.uniform(0, 1, 10) for rng in rngs]
        
        for i in range(1, len(vals)):
            np.testing.assert_array_equal(vals[0], vals[i])
    
    def test_seed_offset_avoids_collision(self):
        """Should use 10000 offset to avoid collision with coordinate RNGs."""
        seed = 42
        shape = [10, 10]
        
        # LHS RNG
        lhs_rng = ChunkUtils.generate_lhs_rng(seed, shape)
        lhs_vals = lhs_rng.uniform(0, 1, 10)
        
        # Coordinate RNG for dimension 0 (seed + 0*1000 = 42)
        coord_rng = np.random.default_rng(seed)
        coord_vals = coord_rng.uniform(0, 1, 10)
        
        # Should produce different sequences
        assert not np.array_equal(lhs_vals, coord_vals)
    
    def test_different_seeds_produce_different_rngs(self):
        """Should produce different RNGs for different seeds."""
        shape = [10, 10]
        
        lhs_rng1 = ChunkUtils.generate_lhs_rng(42, shape)
        lhs_rng2 = ChunkUtils.generate_lhs_rng(99, shape)
        
        vals1 = lhs_rng1.uniform(0, 1, 10)
        vals2 = lhs_rng2.uniform(0, 1, 10)
        
        assert not np.array_equal(vals1, vals2)

