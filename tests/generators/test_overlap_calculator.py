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


class TestComputePairwiseOverlapNonNormalized:
    """Tests for compute_pairwise_overlap_non_normalized method."""
    
    def test_full_overlap(self):
        """Should return 1.0 for identical coordinate sets."""
        coords1 = {(0, 1), (1, 2), (2, 3)}
        coords2 = {(0, 1), (1, 2), (2, 3)}
        varying_dims1 = [0, 1]
        varying_dims2 = [0, 1]
        
        overlap = OverlapCalculator.compute_pairwise_overlap_non_normalized(
            coords1, coords2, varying_dims1, varying_dims2
        )
        assert overlap == len(coords1)
    
    def test_no_overlap(self):
        """Should return 0.0 for disjoint coordinate sets."""
        coords1 = {(0, 1), (1, 2)}
        coords2 = {(2, 3), (3, 4)}
        varying_dims1 = [0, 1]
        varying_dims2 = [0, 1]
        
        overlap = OverlapCalculator.compute_pairwise_overlap_non_normalized(
            coords1, coords2, varying_dims1, varying_dims2
        )
        assert overlap == 0.0
    
    def test_partial_overlap(self):
        """Should compute correct partial overlap."""
        coords1 = {(0, 1), (1, 2), (2, 3), (3, 4)}
        coords2 = {(2, 3), (3, 4), (4, 5), (5, 6)}
        varying_dims1 = [0, 1]
        varying_dims2 = [0, 1]
        
        overlap = OverlapCalculator.compute_pairwise_overlap_non_normalized(
            coords1, coords2, varying_dims1, varying_dims2
        )
        # 2 common out of max(4, 4)
        assert overlap == 2
    
    def test_no_shared_dimensions(self):
        """Should return 0.0 when no shared dimensions."""
        coords1 = {(0, 1, 2)}
        coords2 = {(3, 4, 5)}
        varying_dims1 = [0]
        varying_dims2 = [1]
        
        overlap = OverlapCalculator.compute_pairwise_overlap_non_normalized(
            coords1, coords2, varying_dims1, varying_dims2
        )
        assert overlap == 0.0
    
    def test_all_shared_dimensions(self):
        """Should handle case when all dimensions are shared."""
        coords1 = {(0, 1, 2), (1, 2, 3)}
        coords2 = {(0, 1, 2), (2, 3, 4)}
        varying_dims1 = [0, 1, 2]
        varying_dims2 = [0, 1, 2]
        
        overlap = OverlapCalculator.compute_pairwise_overlap_non_normalized(
            coords1, coords2, varying_dims1, varying_dims2
        )
        # 1 common out of max(2, 2)
        assert overlap == 1
    
    def test_different_dimension_combinations(self):
        """Should require exact full-coordinate matches."""
        coords1 = {(0, 1, 2), (1, 1, 3), (2, 1, 4)}
        coords2 = {(0, 2, 1), (0, 2, 3), (0, 2, 5)}
        varying_dims1 = [0, 2]
        varying_dims2 = [1, 2]
        
        overlap = OverlapCalculator.compute_pairwise_overlap_non_normalized(
            coords1, coords2, varying_dims1, varying_dims2
        )
        assert overlap == 0
    
    def test_one_observation_each(self):
        """Should handle single observation case."""
        coords1 = {(0, 1)}
        coords2 = {(0, 1)}
        varying_dims1 = [0, 1]
        varying_dims2 = [0, 1]
        
        overlap = OverlapCalculator.compute_pairwise_overlap_non_normalized(
            coords1, coords2, varying_dims1, varying_dims2
        )
        assert overlap == 1.0
    
    def test_many_observations(self):
        """Should handle many observations efficiently."""
        coords1 = {(i, j) for i in range(10) for j in range(10)}
        coords2 = {(i, j) for i in range(5, 15) for j in range(5, 15)}
        varying_dims1 = [0, 1]
        varying_dims2 = [0, 1]
        
        overlap = OverlapCalculator.compute_pairwise_overlap_non_normalized(
            coords1, coords2, varying_dims1, varying_dims2
        )
        # Overlap region: [5-9] x [5-9] = 25 points
        # Union: [0-14] x [0-14] but not all filled
        # coords1: 100 points, coords2: 100 points, common: 25 points
        # overlap = 25 / max(100, 100) = 0.25
        # non-normalized overlap = 25
        assert overlap == 25


class TestComputeActualOverlap:
    """Tests for compute_actual_overlap method."""
    
    def test_two_variables(self):
        """Should compute overlap for two variables."""
        records = {
            'var0': np.full((5, 5), np.nan),
            'var1': np.full((5, 5), np.nan)
        }
        records['var0'][0:2, 0:2] = 0.5  # 4 obs
        records['var1'][0:2, 0:2] = 0.7  # 4 obs (all overlapping)
        
        var_varying_dims = [
            [0, 1],
            [0, 1]
        ]

        num_vars = 2
        num_dims = 2
        var_num_obs = np.array(
            [4, 4]
        )
        
        overlap = OverlapCalculator.compute_actual_overlap(
            records, num_vars, num_dims, var_num_obs, var_varying_dims
        )
        assert overlap == 1.0
    
    def test_multiple_variables_different_observations(self):
        """Should compute average overlap for multiple variables where one
        variable is automatically identified as reference variable based on
        number of observations"""
        records = {
            'var0': np.full((5, 5), np.nan),
            'var1': np.full((5, 5), np.nan),
            'var2': np.full((5, 5), np.nan)
        }
        records['var0'][0:2, 0:3] = 0.5  # 5 obs
        records['var1'][0:2, 0:2] = 0.7  # 4 obs (all overlap with var0)
        records['var2'][2:4, 2:4] = 0.3  # 4 obs (no overlap)
        
        var_varying_dims = [
            [0, 1],
            [0, 1],
            [0, 1]
        ]

        num_vars = 3
        num_dims = 2
        var_num_obs = np.array(
            [5, 4, 4]
        )
        
        overlap = OverlapCalculator.compute_actual_overlap(
            records, num_vars, num_dims, var_num_obs, var_varying_dims
        )

        # var0-var1: 1.0, var0-var2: 0.0
        # Average: (1.0 + 0.0) / 2 = 0.5
        assert overlap == 0.5
    
    def test_multiple_variables_same_observations(self):
        """Should compute average overlap for multiple variables where reference
        variable is specified as all variables have same number of observations"""
        records = {
            'var0': np.full((5, 5), np.nan),
            'var1': np.full((5, 5), np.nan),
            'var2': np.full((5, 5), np.nan)
        }
        records['var0'][0:2, 0:2] = 0.5  # 4 obs
        records['var1'][0:2, 0:2] = 0.7  # 4 obs (all overlap with var0)
        records['var2'][2:4, 2:4] = 0.3  # 4 obs (no overlap)
        
        var_varying_dims = [
            [0, 1],
            [0, 1],
            [0, 1]
        ]

        num_vars = 3
        num_dims = 2
        var_num_obs = np.array(
            [5, 4, 4]
        )
        
        overlap = OverlapCalculator.compute_actual_overlap(
            records, num_vars, num_dims, var_num_obs, var_varying_dims, ref_var="var0"
        )

        # var0-var1: 1.0, var0-var2: 0.0
        # Average: (1.0 + 0.0) / 2 = 0.5
        assert overlap == 0.5
    
    def test_single_variable(self):
        """Should return None for single variable."""
        records = {'var0': np.full((5, 5), np.nan)}
        records['var0'][0:2, 0:2] = 0.5
        var_varying_dims = [ [0, 1] ]
        num_vars = 1
        num_dims = 2
        var_num_obs = np.array([4])
        
        overlap = OverlapCalculator.compute_actual_overlap(
            records, num_vars, num_dims, var_num_obs, var_varying_dims
        )

        assert overlap is None
    
    def test_no_overlap(self):
        """Should return 0.0 when no overlap exists."""
        records = {
            'var0': np.full((10, 10), np.nan),
            'var1': np.full((10, 10), np.nan)
        }
        records['var0'][0:2, 0:2] = 0.5
        records['var1'][8:10, 8:10] = 0.7
        
        var_varying_dims = [
            [0, 1],
            [0, 1]
        ]
        num_vars = 2
        num_dims = 2
        var_num_obs = np.array(
            [4, 4]
        )
        
        overlap = OverlapCalculator.compute_actual_overlap(
            records, num_vars, num_dims, var_num_obs, var_varying_dims, ref_var="var0"
        )

        assert overlap == 0.0
    
    def test_full_overlap(self):
        """Should return 1.0 when full overlap."""
        records = {
            'var0': np.full((5, 5), np.nan),
            'var1': np.full((5, 5), np.nan)
        }
        # Same observations
        indices = [(0, 1), (2, 3), (4, 4)]
        for i, j in indices:
            records['var0'][i, j] = 0.5
            records['var1'][i, j] = 0.7
        
        var_varying_dims = [
            [0, 1],
            [0, 1]
        ]
        num_vars = 2
        num_dims = 2
        var_num_obs = np.array(
            [3, 3]
        )
        
        overlap = OverlapCalculator.compute_actual_overlap(
            records, num_vars, num_dims, var_num_obs, var_varying_dims, ref_var="var0"
        )

        assert overlap == 1.0

    def test_mixed_dimensions_require_full_coordinate_match(self):
        """Shared coordinates alone should not count when full coordinates differ."""
        records = {
            'var0': np.full((3, 3), np.nan),
            'var1': np.full((3, 3), np.nan)
        }
        records['var0'][0, 1] = 0.5
        records['var1'][2, 1] = 0.7

        var_varying_dims = [
            [0, 1],
            [1]
        ]
        num_vars = 2
        num_dims = 2
        var_num_obs = np.array([1, 1])

        overlap = OverlapCalculator.compute_actual_overlap(
            records, num_vars, num_dims, var_num_obs, var_varying_dims, ref_var="var0"
        )

        assert overlap == 0.0


class TestComputeActualOverlapsAgainstReference:
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
