"""Tests for SingleVarRecordGenerator class."""

import pytest
import numpy as np
from data_sparsity.generators.single_var_record_generator import SingleVarRecordGenerator


class TestGenerate:
    """Tests for generate method."""
    
    def test_basic_2d_generation(self, fixed_rng):
        """Should generate 2D sparse record."""
        shape = (10, 10)
        num_obs = 20
        sparsity = 0.2
        record = SingleVarRecordGenerator.generate(shape, num_obs, sparsity, fixed_rng)
        assert record.shape == shape
    
    def test_3d_generation(self, fixed_rng):
        """Should generate 3D sparse record."""
        shape = (5, 6, 7)
        num_obs = 42
        sparsity = 0.2
        record = SingleVarRecordGenerator.generate(shape, num_obs, sparsity, fixed_rng)
        assert record.shape == shape
    
    def test_1d_generation(self, fixed_rng):
        """Should generate 1D sparse record."""
        shape = (50,)
        num_obs = 10
        sparsity = 0.2
        record = SingleVarRecordGenerator.generate(shape, num_obs, sparsity, fixed_rng)
        assert record.shape == shape
    
    def test_high_dimensional(self, fixed_rng):
        """Should handle high-dimensional arrays."""
        shape = (3, 4, 5, 3, 2)
        num_obs = 72
        sparsity = 0.2
        record = SingleVarRecordGenerator.generate(shape, num_obs, sparsity, fixed_rng)
        assert record.shape == shape
    
    def test_correct_shape(self, fixed_rng):
        """Should return array with correct shape."""
        shape = (8, 12, 6)
        num_obs = 115
        sparsity = 0.2
        record = SingleVarRecordGenerator.generate(shape, num_obs, sparsity, fixed_rng)
        assert record.shape == shape
    
    def test_correct_number_of_observations(self, fixed_rng):
        """Should have correct number of non-NaN observations."""
        shape = (10, 10)
        num_obs = 25
        sparsity = 0.25
        record = SingleVarRecordGenerator.generate(shape, num_obs, sparsity, fixed_rng)
        non_nan_count = np.count_nonzero(~np.isnan(record))
        assert non_nan_count == num_obs
    
    def test_values_in_range(self, fixed_rng):
        """Should have observation values in [0,1] range."""
        shape = (15, 15)
        num_obs = 45
        sparsity = 0.2
        record = SingleVarRecordGenerator.generate(shape, num_obs, sparsity, fixed_rng)
        observations = record[~np.isnan(record)]
        assert np.all(observations >= 0)
        assert np.all(observations <= 1)
    
    def test_reproducible_with_seed(self):
        """Should produce same results with same seed."""
        shape = (8, 8)
        num_obs = 16
        sparsity = 0.25
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        record1 = SingleVarRecordGenerator.generate(shape, num_obs, sparsity, rng1)
        record2 = SingleVarRecordGenerator.generate(shape, num_obs, sparsity, rng2)
        np.testing.assert_array_equal(record1, record2)
    
    def test_different_with_different_seed(self):
        """Should produce different results with different seeds."""
        shape = (8, 8)
        num_obs = 16
        sparsity = 0.25
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(123)
        record1 = SingleVarRecordGenerator.generate(shape, num_obs, sparsity, rng1)
        record2 = SingleVarRecordGenerator.generate(shape, num_obs, sparsity, rng2)
        assert not np.array_equal(record1, record2, equal_nan=True)
    
    def test_pre_provided_observations_used(self, fixed_rng):
        """Should use pre-provided observations when given."""
        shape = (5, 5)
        num_obs = 5
        sparsity = 0.2
        observations = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        record = SingleVarRecordGenerator.generate(
            shape, num_obs, sparsity, fixed_rng, observations=observations
        )
        non_nan_values = record[~np.isnan(record)]
        # Should contain all provided values
        assert set(non_nan_values).issubset(set(observations))
    
    def test_sparsity_validation_passes(self, fixed_rng):
        """Should pass sparsity validation when correct."""
        shape = (10, 10)
        num_obs = 20
        sparsity = 0.2
        # Should not raise
        record = SingleVarRecordGenerator.generate(
            shape, num_obs, sparsity, fixed_rng, validate_sparsity=True
        )
        assert record is not None
    
    def test_sparsity_validation_fails(self, fixed_rng):
        """Should raise when sparsity validation fails."""
        shape = (10, 10)
        num_obs = 20
        sparsity = 0.5  # Wrong sparsity for 20 obs out of 100
        with pytest.raises(ValueError, match="Expected sparsity"):
            SingleVarRecordGenerator.generate(
                shape, num_obs, sparsity, fixed_rng, validate_sparsity=True
            )
    
    def test_all_grid_points_used(self, fixed_rng):
        """Should use all grid points when sparsity=1."""
        shape = (5, 4)
        total_points = np.prod(shape)
        num_obs = total_points
        sparsity = 1.0
        record = SingleVarRecordGenerator.generate(shape, num_obs, sparsity, fixed_rng)
        non_nan_count = np.count_nonzero(~np.isnan(record))
        assert non_nan_count == total_points
    
    def test_single_observation(self, fixed_rng):
        """Should handle single observation case."""
        shape = (10, 10)
        num_obs = 1
        sparsity = 0.01
        record = SingleVarRecordGenerator.generate(shape, num_obs, sparsity, fixed_rng)
        non_nan_count = np.count_nonzero(~np.isnan(record))
        assert non_nan_count == 1
    
    def test_large_grid_low_sparsity(self, fixed_rng):
        """Should handle large grid with low sparsity."""
        shape = (50, 50)
        num_obs = 50
        sparsity = 0.02
        record = SingleVarRecordGenerator.generate(shape, num_obs, sparsity, fixed_rng)
        non_nan_count = np.count_nonzero(~np.isnan(record))
        assert non_nan_count == num_obs
    
    def test_small_grid_high_sparsity(self, fixed_rng):
        """Should handle small grid with high sparsity."""
        shape = (5, 5)
        num_obs = 20
        sparsity = 0.8
        record = SingleVarRecordGenerator.generate(shape, num_obs, sparsity, fixed_rng)
        non_nan_count = np.count_nonzero(~np.isnan(record))
        assert non_nan_count == num_obs
    
    def test_no_observation_overlap(self, fixed_rng):
        """Should place observations at unique locations."""
        shape = (10, 10)
        num_obs = 30
        sparsity = 0.3
        record = SingleVarRecordGenerator.generate(shape, num_obs, sparsity, fixed_rng)
        non_nan_count = np.count_nonzero(~np.isnan(record))
        # Count should equal num_obs (no overlap)
        assert non_nan_count == num_obs
    
    def test_returns_numpy_array(self, fixed_rng):
        """Should return numpy array."""
        shape = (5, 5)
        num_obs = 10
        sparsity = 0.4
        record = SingleVarRecordGenerator.generate(shape, num_obs, sparsity, fixed_rng)
        assert isinstance(record, np.ndarray)
    
    def test_nan_at_non_observation_points(self, fixed_rng):
        """Should have NaN at locations without observations."""
        shape = (10, 10)
        num_obs = 10
        sparsity = 0.1
        record = SingleVarRecordGenerator.generate(shape, num_obs, sparsity, fixed_rng)
        nan_count = np.isnan(record).sum()
        assert nan_count == 100 - num_obs
    
    def test_non_nan_at_observation_points(self, fixed_rng):
        """Should have valid values (not NaN) at observation points."""
        shape = (8, 8)
        num_obs = 16
        sparsity = 0.25
        record = SingleVarRecordGenerator.generate(shape, num_obs, sparsity, fixed_rng)
        observations = record[~np.isnan(record)]
        assert len(observations) == num_obs
        assert not np.any(np.isnan(observations))
