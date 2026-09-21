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
        assert isinstance(coords, dict)
        assert len(coords.keys()) == 1
        assert len(coords["x0"]) == 10

    def test_2d_grid(self, fixed_rng):
        """Should generate 2D coordinate grid."""
        shape = (5, 8)
        coords = CoordinateGenerator.generate_all_coords(shape, fixed_rng)
        assert isinstance(coords, dict)
        assert len(coords.keys()) == 2
        assert len(coords["x0"]) == 5
        assert len(coords["x1"]) == 8

    def test_3d_grid(self, fixed_rng):
        """Should generate 3D coordinate grid."""
        shape = (4, 6, 3)
        coords = CoordinateGenerator.generate_all_coords(shape, fixed_rng)
        assert isinstance(coords, dict)
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
            shape,
            rng=None,
            dim_ranges=None,
            dim_rngs=dim_rngs1,
        )

        # Second generation
        dim_rngs2 = ChunkUtils.generate_rngs(seed=seed, num_dims=num_dims)
        coords2 = CoordinateGenerator.generate_all_coords(
            shape,
            rng=None,
            dim_ranges=None,
            dim_rngs=dim_rngs2,
        )

        # Should match exactly
        for dim_name in coords1:
            np.testing.assert_array_equal(coords1[dim_name], coords2[dim_name])
