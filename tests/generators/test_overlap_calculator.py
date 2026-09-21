"""Tests for OverlapCalculator class."""

import pytest
import numpy as np
from data_sparsity.generators.overlap_calculator import OverlapCalculator


class TestExtractCoordinateSet:
    """Tests for extract_coordinate_set method."""
    
    def test_2d_record(self):
        """Should extract coordinate set from 2D record."""
        record = np.full((5, 4), np.nan)
        record[0, 1] = 0.5
        record[2, 3] = 0.7
        record[4, 0] = 0.3
        
        coords = OverlapCalculator.extract_coordinate_set(record, num_dims=2)
        expected = {(0, 1), (2, 3), (4, 0)}
        assert coords == expected
    
    def test_3d_record(self):
        """Should extract coordinate set from 3D record."""
        record = np.full((3, 4, 2), np.nan)
        record[0, 1, 0] = 0.5
        record[1, 2, 1] = 0.7
        record[2, 3, 0] = 0.3
        
        coords = OverlapCalculator.extract_coordinate_set(record, num_dims=3)
        expected = {(0, 1, 0), (1, 2, 1), (2, 3, 0)}
        assert coords == expected
    
    def test_sparse_record(self):
        """Should handle sparse records correctly."""
        record = np.full((10, 10), np.nan)
        record[0, 0] = 0.1
        record[9, 9] = 0.9
        
        coords = OverlapCalculator.extract_coordinate_set(record, num_dims=2)
        expected = {(0, 0), (9, 9)}
        assert coords == expected
    
    def test_dense_record(self):
        """Should handle dense records."""
        record = np.full((3, 3), np.nan)
        for i in range(3):
            for j in range(3):
                record[i, j] = i * 3 + j
        
        coords = OverlapCalculator.extract_coordinate_set(record, num_dims=2)
        assert len(coords) == 9
    
    def test_single_observation(self):
        """Should handle single observation."""
        record = np.full((5, 5), np.nan)
        record[2, 3] = 0.5
        
        coords = OverlapCalculator.extract_coordinate_set(record, num_dims=2)
        expected = {(2, 3)}
        assert coords == expected
    
    def test_empty_record(self):
        """Should return empty set for all-NaN record."""
        record = np.full((5, 5), np.nan)
        coords = OverlapCalculator.extract_coordinate_set(record, num_dims=2)
        assert coords == set()


class TestProjectCoordinates:
    """Tests for project_coordinates method."""
    
    def test_project_to_single_dimension(self):
        """Should project to single dimension."""
        coords = {(0, 1, 2), (1, 1, 3), (2, 1, 4)}
        varying_dims = [0]
        projected = OverlapCalculator.project_coordinates(coords, varying_dims)
        expected = {(0,), (1,), (2,)}
        assert projected == expected
    
    def test_project_to_multiple_dimensions(self):
        """Should project to multiple dimensions."""
        coords = {(0, 1, 2), (1, 1, 3), (0, 2, 3)}
        varying_dims = [0, 2]
        projected = OverlapCalculator.project_coordinates(coords, varying_dims)
        expected = {(0, 2), (1, 3), (0, 3)}
        assert projected == expected
    
    def test_project_to_all_dimensions(self):
        """Should return identity when projecting to all dimensions."""
        coords = {(0, 1, 2), (1, 2, 3), (2, 3, 4)}
        varying_dims = [0, 1, 2]
        projected = OverlapCalculator.project_coordinates(coords, varying_dims)
        assert projected == coords
    
    def test_dimension_order_matters(self):
        """Should preserve dimension order in projection."""
        coords = {(1, 2, 3)}
        varying_dims = [2, 0]  # Reversed order
        projected = OverlapCalculator.project_coordinates(coords, varying_dims)
        expected = {(3, 1)}  # Order from varying_dims
        assert projected == expected
    
    def test_duplicates_eliminated(self):
        """Should eliminate duplicate projected coordinates."""
        coords = {(0, 1, 2), (0, 2, 2), (0, 3, 2)}
        varying_dims = [0, 2]
        projected = OverlapCalculator.project_coordinates(coords, varying_dims)
        expected = {(0, 2)}
        assert projected == expected
    
    def test_empty_set_handling(self):
        """Should handle empty coordinate set."""
        coords = set()
        varying_dims = [0, 1]
        projected = OverlapCalculator.project_coordinates(coords, varying_dims)
        assert projected == set()


class TestComputeOverlapReport:
    """Tests for per-variable overlap computation against a fixed reference."""

    def test_returns_per_variable_ratios(self):
        """Should compute one overlap ratio per non-reference variable."""
        records = {
            'var0': np.full((5, 5), np.nan),
            'var1': np.full((5, 5), np.nan),
            'var2': np.full((5, 5), np.nan)
        }
        records['var0'][0:2, 0:2] = 0.5
        records['var1'][0:2, 0:2] = 0.7
        records['var2'][2:4, 2:4] = 0.3

        report = OverlapCalculator.compute_overlap_report(
            records, num_vars=3, num_dims=2, ref_var="var0"
        )

        np.testing.assert_array_equal(report["f1"], np.array([1.0, 0.0]))
        np.testing.assert_array_equal(report["f2"], np.array([1.0, 0.0]))
        np.testing.assert_array_equal(report["shared"], np.array([4, 0]))
