"""Tests for ObservationGenerator class."""

import pytest
import numpy as np
from data_sparsity.generators.observation_generator import ObservationGenerator


class TestGenerateObservations:
    """Tests for generate_observations method."""

    def test_correct_length(self, fixed_rng):
        """Should generate correct number of observations."""
        obs = ObservationGenerator.generate_observations(100, fixed_rng)
        assert len(obs) == 100

    def test_values_in_range(self, fixed_rng):
        """Should generate values in [0,1] range."""
        obs = ObservationGenerator.generate_observations(500, fixed_rng)
        assert np.all(obs >= 0)
        assert np.all(obs <= 1)

    def test_reproducible_with_seed(self):
        """Should produce same results with same seed."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        obs1 = ObservationGenerator.generate_observations(50, rng1)
        obs2 = ObservationGenerator.generate_observations(50, rng2)
        np.testing.assert_array_equal(obs1, obs2)

    def test_different_with_different_seed(self):
        """Should produce different results with different seeds."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(123)
        obs1 = ObservationGenerator.generate_observations(50, rng1)
        obs2 = ObservationGenerator.generate_observations(50, rng2)
        assert not np.array_equal(obs1, obs2)

    def test_single_observation(self, fixed_rng):
        """Should handle single observation."""
        obs = ObservationGenerator.generate_observations(1, fixed_rng)
        assert len(obs) == 1
        assert 0 <= obs[0] <= 1

    def test_many_observations(self, fixed_rng):
        """Should handle large number of observations."""
        obs = ObservationGenerator.generate_observations(10000, fixed_rng)
        assert len(obs) == 10000
        assert np.all(obs >= 0)
        assert np.all(obs <= 1)

    def test_returns_numpy_array(self, fixed_rng):
        """Should return numpy array."""
        obs = ObservationGenerator.generate_observations(100, fixed_rng)
        assert isinstance(obs, np.ndarray)

    def test_float_dtype(self, fixed_rng):
        """Should return float dtype."""
        obs = ObservationGenerator.generate_observations(100, fixed_rng)
        assert np.issubdtype(obs.dtype, np.floating)

    def test_no_nan_values(self, fixed_rng):
        """Should not contain NaN values."""
        obs = ObservationGenerator.generate_observations(100, fixed_rng)
        assert not np.any(np.isnan(obs))

    def test_statistical_distribution(self, fixed_rng):
        """Should have approximately uniform distribution."""
        obs = ObservationGenerator.generate_observations(10000, fixed_rng)
        # Check mean is approximately 0.5 (within 0.1)
        assert 0.4 <= np.mean(obs) <= 0.6
        # Check standard deviation is approximately 1/sqrt(12) ≈ 0.289
        assert 0.25 <= np.std(obs) <= 0.35
