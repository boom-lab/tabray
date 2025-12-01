"""Tests for RecordGenerator class."""

import pytest
import numpy as np
from data_sparsity.generators.record_generator import RecordGenerator


class TestInitializeRecord:
    """Tests for initialize_record method."""
    
    def test_1d_shape(self):
        """Should initialize 1D array with NaN."""
        shape = (10,)
        record = RecordGenerator.initialize_record(shape)
        assert record.shape == shape
        assert np.all(np.isnan(record))
    
    def test_2d_shape(self):
        """Should initialize 2D array with NaN."""
        shape = (5, 8)
        record = RecordGenerator.initialize_record(shape)
        assert record.shape == shape
        assert np.all(np.isnan(record))
    
    def test_3d_shape(self):
        """Should initialize 3D array with NaN."""
        shape = (4, 6, 3)
        record = RecordGenerator.initialize_record(shape)
        assert record.shape == shape
        assert np.all(np.isnan(record))
    
    def test_all_values_nan(self):
        """Should fill entire array with NaN."""
        shape = (10, 10)
        record = RecordGenerator.initialize_record(shape)
        assert np.isnan(record).sum() == np.prod(shape)
    
    def test_float_dtype(self):
        """Should return float dtype."""
        shape = (5, 5)
        record = RecordGenerator.initialize_record(shape)
        assert np.issubdtype(record.dtype, np.floating)


class TestGenerateFlatIndices:
    """Tests for generate_flat_indices method."""
    
    def test_correct_count(self, fixed_rng):
        """Should generate correct number of indices."""
        indices = RecordGenerator.generate_flat_indices(1000, 50, fixed_rng)
        assert len(indices) == 50
    
    def test_all_unique(self, fixed_rng):
        """Should generate unique indices (no duplicates)."""
        indices = RecordGenerator.generate_flat_indices(1000, 100, fixed_rng)
        assert len(indices) == len(np.unique(indices))
    
    def test_all_in_valid_range(self, fixed_rng):
        """Should generate indices in valid range [0, total_points)."""
        total_points = 500
        indices = RecordGenerator.generate_flat_indices(total_points, 100, fixed_rng)
        assert np.all(indices >= 0)
        assert np.all(indices < total_points)
    
    def test_reproducible_with_seed(self):
        """Should produce same results with same seed."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        indices1 = RecordGenerator.generate_flat_indices(200, 30, rng1)
        indices2 = RecordGenerator.generate_flat_indices(200, 30, rng2)
        np.testing.assert_array_equal(indices1, indices2)
    
    def test_different_with_different_seed(self):
        """Should produce different results with different seeds."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(123)
        indices1 = RecordGenerator.generate_flat_indices(200, 30, rng1)
        indices2 = RecordGenerator.generate_flat_indices(200, 30, rng2)
        assert not np.array_equal(indices1, indices2)
    
    def test_num_obs_equals_total_points(self, fixed_rng):
        """Should handle case when selecting all points."""
        total_points = 50
        indices = RecordGenerator.generate_flat_indices(total_points, total_points, fixed_rng)
        assert len(indices) == total_points
        # Should contain all possible indices
        assert len(np.unique(indices)) == total_points
    
    def test_single_index(self, fixed_rng):
        """Should handle selecting single index."""
        indices = RecordGenerator.generate_flat_indices(100, 1, fixed_rng)
        assert len(indices) == 1
        assert 0 <= indices[0] < 100
    
    def test_large_grid(self, fixed_rng):
        """Should handle large grids efficiently."""
        indices = RecordGenerator.generate_flat_indices(100000, 1000, fixed_rng)
        assert len(indices) == 1000
        assert len(np.unique(indices)) == 1000


class TestConvertToMultiIndices:
    """Tests for convert_to_multi_indices method."""
    
    def test_1d_shape(self):
        """Should convert flat indices for 1D shape."""
        flat_indices = np.array([0, 3, 7, 9])
        shape = (10,)
        multi_indices = RecordGenerator.convert_to_multi_indices(flat_indices, shape)
        assert len(multi_indices) == 1
        np.testing.assert_array_equal(multi_indices[0], flat_indices)
    
    def test_2d_shape(self):
        """Should convert flat indices for 2D shape."""
        flat_indices = np.array([0, 5, 10, 15])
        shape = (4, 5)
        multi_indices = RecordGenerator.convert_to_multi_indices(flat_indices, shape)
        assert len(multi_indices) == 2
        # Verify conversion: flat = row * ncols + col
        expected_rows = flat_indices // shape[1]
        expected_cols = flat_indices % shape[1]
        np.testing.assert_array_equal(multi_indices[0], expected_rows)
        np.testing.assert_array_equal(multi_indices[1], expected_cols)
    
    def test_3d_shape(self):
        """Should convert flat indices for 3D shape."""
        flat_indices = np.array([0, 10, 20, 30])
        shape = (3, 4, 5)
        multi_indices = RecordGenerator.convert_to_multi_indices(flat_indices, shape)
        assert len(multi_indices) == 3
    
    def test_tuple_length_matches_shape_length(self):
        """Should return tuple with length matching number of dimensions."""
        flat_indices = np.array([5, 15, 25])
        shape = (2, 3, 4, 5)
        multi_indices = RecordGenerator.convert_to_multi_indices(flat_indices, shape)
        assert len(multi_indices) == len(shape)
    
    def test_each_array_length_matches_num_indices(self):
        """Should return arrays with length matching number of indices."""
        flat_indices = np.array([1, 5, 9, 13, 17])
        shape = (5, 5)
        multi_indices = RecordGenerator.convert_to_multi_indices(flat_indices, shape)
        for arr in multi_indices:
            assert len(arr) == len(flat_indices)
    
    def test_round_trip(self):
        """Should correctly convert between flat and multi indices."""
        shape = (4, 6, 3)
        flat_indices = np.array([0, 10, 20, 30, 40, 50])
        multi_indices = RecordGenerator.convert_to_multi_indices(flat_indices, shape)
        # Convert back to flat
        reconstructed = np.ravel_multi_index(multi_indices, shape)
        np.testing.assert_array_equal(flat_indices, reconstructed)


class TestAssignObservations:
    """Tests for assign_observations method."""
    
    def test_values_assigned_at_indices(self):
        """Should assign values at specified indices."""
        record = np.full((5, 5), np.nan)
        multi_indices = (np.array([0, 2, 4]), np.array([1, 3, 0]))
        observations = np.array([0.5, 0.7, 0.3])
        RecordGenerator.assign_observations(record, multi_indices, observations)
        assert record[0, 1] == 0.5
        assert record[2, 3] == 0.7
        assert record[4, 0] == 0.3
    
    def test_other_values_remain_nan(self):
        """Should leave other positions as NaN."""
        record = np.full((5, 5), np.nan)
        multi_indices = (np.array([0, 2]), np.array([1, 3]))
        observations = np.array([0.5, 0.7])
        RecordGenerator.assign_observations(record, multi_indices, observations)
        # Count NaN values
        nan_count = np.isnan(record).sum()
        assert nan_count == 25 - 2  # 25 total - 2 assigned
    
    def test_modifies_in_place(self):
        """Should modify record array in-place."""
        record = np.full((3, 3), np.nan)
        original_id = id(record)
        multi_indices = (np.array([1]), np.array([1]))
        observations = np.array([0.5])
        RecordGenerator.assign_observations(record, multi_indices, observations)
        assert id(record) == original_id
    
    def test_correct_observation_values(self):
        """Should assign correct observation values."""
        record = np.full((4, 4), np.nan)
        multi_indices = (np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))
        observations = np.array([0.1, 0.2, 0.3, 0.4])
        RecordGenerator.assign_observations(record, multi_indices, observations)
        np.testing.assert_array_equal(np.diag(record), observations)
    
    def test_multiple_assignments(self):
        """Should handle multiple observations correctly."""
        record = np.full((6, 6), np.nan)
        multi_indices = (np.array([0, 1, 2, 3, 4, 5]), np.array([5, 4, 3, 2, 1, 0]))
        observations = np.arange(6) * 0.1
        RecordGenerator.assign_observations(record, multi_indices, observations)
        for i, obs in enumerate(observations):
            assert record[i, 5-i] == obs


class TestValidateSparsity:
    """Tests for validate_sparsity method."""
    
    def test_matching_sparsity_passes(self):
        """Should pass when sparsity matches expected."""
        shape = (10,10)
        total_grid_points = np.prod(shape)
        num_obs = 10
        expected_sparsity = 0.1

        # Should not raise
        RecordGenerator.validate_sparsity(
            num_obs, total_grid_points, expected_sparsity
        )
    
    def test_close_sparsity_passes(self):
        """Should pass when sparsity is within tolerance."""
        shape = (10,10)
        total_grid_points = np.prod(shape)
        num_obs = 10
        expected_sparsity = 0.099999  # Within tolerance

        # Should not raise
        RecordGenerator.validate_sparsity(
            num_obs, total_grid_points, expected_sparsity
        )
    
    def test_different_sparsity_raises_value_error(self):
        """Should raise ValueError when sparsity differs significantly."""
        shape = (10,10)
        total_grid_points = np.prod(shape)
        num_obs = 10
        expected_sparsity = 0.2  # Within tolerance

        pattern = r"Sparsity \d+\.\d+ determined from number of coordinates differs from expected sparsity \d+\.\d+"
        with pytest.raises(ValueError, match=pattern):
            RecordGenerator.validate_sparsity(
                num_obs, total_grid_points, expected_sparsity
            )
    
    def test_error_message_contains_values(self):
        """Should include actual and expected values in error message."""
        shape = (10,10)
        total_grid_points = np.prod(shape)
        num_obs = 4
        expected_sparsity = 0.1
        with pytest.raises(ValueError) as exc_info:
            RecordGenerator.validate_sparsity(
                num_obs, total_grid_points, expected_sparsity
            )
        assert "0.04" in str(exc_info.value) or "4.0%" in str(exc_info.value)
        assert "0.1" in str(exc_info.value) or "10" in str(exc_info.value)
    
    def test_edge_case_full_sparsity(self):
        """Should handle sparsity = 1."""
        shape = (5, 5)
        total_grid_points = np.prod(shape)
        num_obs = 25
        expected_sparsity = 1.

        # Should not raise
        RecordGenerator.validate_sparsity(
            num_obs, total_grid_points, expected_sparsity
        )
    
    def test_edge_case_very_small_sparsity(self):
        """Should handle very small sparsity."""
        shape = (1e8, 1e8)
        total_grid_points = np.prod(shape)
        num_obs = 1
        expected_sparsity = 1./total_grid_points

        # Should not raise
        RecordGenerator.validate_sparsity(
            num_obs, total_grid_points, expected_sparsity
        )
