"""Tests for SparsityValidator class."""

import pytest
import numpy as np
from data_sparsity.validators import SparsityValidator


class TestComputeMinDensity:
    """Tests for compute_min_density method."""

    def test_uniform_dimensions(self):
        """Uniform dimensions should return 1/(n^(d-1))."""
        nb_coords = np.array([10, 10, 10])
        result = SparsityValidator.compute_min_density(nb_coords)
        assert result == 1 / (10**(3 - 1))

    def test_non_uniform_dimensions(self):
        """Non-uniform dimensions should return 1/(n^(d-1))."""
        nb_coords = np.array([5, 10, 20])
        result = SparsityValidator.compute_min_density(nb_coords)
        assert result == 1 / (5**(3 - 1))

    def test_single_dimension(self):
        """Single dimension should work."""
        nb_coords = np.array([8])
        result = SparsityValidator.compute_min_density(nb_coords)
        assert result == 1

    def test_returns_one_over_total_grid(self):
        """Should return 1/(n^(d-1))."""
        nb_coords = np.array([3, 7, 5, 9])
        result = SparsityValidator.compute_min_density(nb_coords)
        assert result == 1 / (3**(4 - 1))

    def test_large_dimensions(self):
        """Large dimensions should give small density."""
        nb_coords = np.array([1000, 2000])
        result = SparsityValidator.compute_min_density(nb_coords)
        assert result == 1.0 / 1000


class TestValidateDensityBounds:
    """Tests for validate_density_bounds method."""

    def test_valid_density_in_range_passes(self):
        """Valid density in range should pass."""
        result = SparsityValidator.validate_density_bounds(0.5, 0.1)
        assert result == 0.5

    def test_density_zero_returns_minimum(self):
        """Density = 0 should return minimum."""
        result = SparsityValidator.validate_density_bounds(0.0, 0.1)
        assert result == 0.1

    def test_density_less_than_minimum_not_zero_raises_value_error(self):
        """Density < minimum (not 0) should raise ValueError."""
        with pytest.raises(ValueError, match="lower than minimum value"):
            SparsityValidator.validate_density_bounds(0.05, 0.1)

    def test_density_equals_minimum_passes(self):
        """Density = minimum should pass."""
        result = SparsityValidator.validate_density_bounds(0.1, 0.1)
        assert result == 0.1

    def test_density_equals_one_passes(self):
        """Density = 1 should pass."""
        result = SparsityValidator.validate_density_bounds(1.0, 0.1)
        assert result == 1.0

    def test_edge_case_minimum_plus_epsilon(self):
        """Density = minimum + epsilon should pass."""
        result = SparsityValidator.validate_density_bounds(0.10001, 0.1)
        assert result == pytest.approx(0.10001)

    def test_edge_case_minimum_minus_epsilon(self):
        """Density = minimum - epsilon should raise ValueError."""
        with pytest.raises(ValueError, match="lower than minimum value"):
            SparsityValidator.validate_density_bounds(0.09999, 0.1)


class TestValidateNumObsConsistency:
    """Tests for validate_num_obs_consistency method."""

    def test_matching_num_obs_passes_unchanged(self):
        """Matching num_obs should pass unchanged."""
        nb_coords = np.array([10, 10])
        num_obs, density = SparsityValidator.validate_num_obs_consistency(
            50, 0.5, nb_coords
        )
        assert num_obs == 50
        assert density == 0.5

    def test_non_matching_num_obs_adjusted(self):
        """Non-matching num_obs should be adjusted."""
        nb_coords = np.array([10, 10])
        num_obs, density = SparsityValidator.validate_num_obs_consistency(
            51, 0.5, nb_coords
        )
        assert num_obs == 50
        assert density == 0.5

    def test_returns_integer(self):
        """Should return integer num_obs."""
        nb_coords = np.array([10, 10])
        num_obs, _ = SparsityValidator.validate_num_obs_consistency(
            50.7, 0.5, nb_coords
        )
        assert isinstance(num_obs, int)

    def test_returns_adjusted_density(self):
        """Should return adjusted density."""
        nb_coords = np.array([10, 10])
        num_obs, density = SparsityValidator.validate_num_obs_consistency(
            55, 0.5, nb_coords
        )
        assert density == pytest.approx(0.5, rel=0.1)

    def test_adjusted_density_in_bounds(self):
        """Adjusted density should be in valid bounds."""
        nb_coords = np.array([10, 10])
        _, density = SparsityValidator.validate_num_obs_consistency(
            50, 0.5, nb_coords
        )
        assert 0.0 <= density <= 1.0

    def test_edge_case_large_grid(self):
        """Large grid should work."""
        nb_coords = np.array([100, 100, 100])
        num_obs, density = SparsityValidator.validate_num_obs_consistency(
            100000, 0.1, nb_coords
        )
        assert num_obs == 100000
        assert density == 0.1

    def test_edge_case_small_grid(self):
        """Small grid should work."""
        nb_coords = np.array([2, 2])
        num_obs, density = SparsityValidator.validate_num_obs_consistency(
            2, 0.5, nb_coords
        )
        assert num_obs == 2
        assert density == 0.5

    def test_out_of_bounds_after_adjustment_raises_value_error(self):
        """Out of bounds after adjustment should raise ValueError."""
        # Create a case where adjusted density would be > 1
        nb_coords = np.array([2, 2])
        with pytest.raises(ValueError, match="out of bounds"):
            SparsityValidator.validate_num_obs_consistency(
                5, 1.5, nb_coords
            )

    def test_density_recalculation_correct(self):
        """Density recalculation should be correct."""
        nb_coords = np.array([10, 10])
        num_obs, density = SparsityValidator.validate_num_obs_consistency(
            30, 0.3, nb_coords
        )
        expected_density = num_obs / np.prod(nb_coords)
        assert density == pytest.approx(expected_density)
