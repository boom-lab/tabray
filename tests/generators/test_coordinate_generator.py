"""Tests for CoordinateGenerator class."""

import pytest
import numpy as np
from data_sparsity.generators.coordinate_generator import CoordinateGenerator


class TestGenerateDimensionCoords:
    """Tests for generate_dimension_coords method."""
    
    def test_correct_length(self, fixed_rng):
        """Should generate correct number of coordinates."""
        coords = CoordinateGenerator.generate_dimension_coords(10, fixed_rng)
        assert len(coords) == 10
    
    def test_values_in_range(self, fixed_rng):
        """Should generate values in [0,1] range."""
        coords = CoordinateGenerator.generate_dimension_coords(100, fixed_rng)
        assert np.all(coords >= 0)
        assert np.all(coords <= 1)
    
    def test_sorted_ascending(self, fixed_rng):
        """Should return sorted coordinates."""
        coords = CoordinateGenerator.generate_dimension_coords(50, fixed_rng)
        assert np.all(coords[:-1] <= coords[1:])
    
    def test_reproducible_with_seed(self):
        """Should produce same results with same seed."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        coords1 = CoordinateGenerator.generate_dimension_coords(20, rng1)
        coords2 = CoordinateGenerator.generate_dimension_coords(20, rng2)
        np.testing.assert_array_equal(coords1, coords2)
    
    def test_different_with_different_seed(self):
        """Should produce different results with different seeds."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(123)
        coords1 = CoordinateGenerator.generate_dimension_coords(20, rng1)
        coords2 = CoordinateGenerator.generate_dimension_coords(20, rng2)
        assert not np.array_equal(coords1, coords2)
    
    def test_single_coordinate(self, fixed_rng):
        """Should handle single coordinate generation."""
        coords = CoordinateGenerator.generate_dimension_coords(1, fixed_rng)
        assert len(coords) == 1
        assert 0 <= coords[0] <= 1
    
    def test_many_coordinates(self, fixed_rng):
        """Should handle large number of coordinates."""
        coords = CoordinateGenerator.generate_dimension_coords(1000, fixed_rng)
        assert len(coords) == 1000
        assert np.all(coords >= 0)
        assert np.all(coords <= 1)
    
    def test_returns_numpy_array(self, fixed_rng):
        """Should return numpy array."""
        coords = CoordinateGenerator.generate_dimension_coords(10, fixed_rng)
        assert isinstance(coords, np.ndarray)


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
