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

