"""Tests for SparsityValidator class."""

import pytest
import numpy as np
from data_sparsity.validators import SparsityValidator


class TestComputeMinSparsity:
    """Tests for compute_min_sparsity method."""
    
    def test_uniform_dimensions(self):
        """Uniform dimensions should return 1/total_grid_points."""
        nb_coords = np.array([10, 10, 10])
        result = SparsityValidator.compute_min_sparsity(nb_coords)
        assert result == 1.0 / 1000  # 1 / (10*10*10)
    
    def test_non_uniform_dimensions(self):
        """Non-uniform dimensions should return 1/total_grid_points."""
        nb_coords = np.array([5, 10, 20])
        result = SparsityValidator.compute_min_sparsity(nb_coords)
        assert result == 1.0 / 1000  # 1 / (5*10*20)
    
    def test_single_dimension(self):
        """Single dimension should work."""
        nb_coords = np.array([8])
        result = SparsityValidator.compute_min_sparsity(nb_coords)
        assert result == 0.125  # 1 / 8
    
    def test_returns_one_over_total_grid(self):
        """Should return 1/total_grid_points."""
        nb_coords = np.array([3, 7, 5, 9])
        result = SparsityValidator.compute_min_sparsity(nb_coords)
        assert result == pytest.approx(1.0 / (3*7*5*9))  # 1 / 945
    
    def test_large_dimensions(self):
        """Large dimensions should give small sparsity."""
        nb_coords = np.array([1000, 2000])
        result = SparsityValidator.compute_min_sparsity(nb_coords)
        assert result == 1.0 / 2_000_000  # 1 / (1000*2000)


class TestValidateSparsityBounds:
    """Tests for validate_sparsity_bounds method."""
    
    def test_valid_sparsity_in_range_passes(self):
        """Valid sparsity in range should pass."""
        result = SparsityValidator.validate_sparsity_bounds(0.5, 0.1)
        assert result == 0.5
    
    def test_sparsity_zero_returns_minimum(self):
        """Sparsity = 0 should return minimum."""
        result = SparsityValidator.validate_sparsity_bounds(0.0, 0.1)
        assert result == 0.1
    
    def test_sparsity_less_than_minimum_not_zero_raises_value_error(self):
        """Sparsity < minimum (not 0) should raise ValueError."""
        with pytest.raises(ValueError, match="lower than minimum value"):
            SparsityValidator.validate_sparsity_bounds(0.05, 0.1)
    
    def test_sparsity_equals_minimum_passes(self):
        """Sparsity = minimum should pass."""
        result = SparsityValidator.validate_sparsity_bounds(0.1, 0.1)
        assert result == 0.1
    
    def test_sparsity_equals_one_passes(self):
        """Sparsity = 1 should pass."""
        result = SparsityValidator.validate_sparsity_bounds(1.0, 0.1)
        assert result == 1.0
    
    def test_edge_case_minimum_plus_epsilon(self):
        """Sparsity = minimum + epsilon should pass."""
        result = SparsityValidator.validate_sparsity_bounds(0.10001, 0.1)
        assert result == pytest.approx(0.10001)
    
    def test_edge_case_minimum_minus_epsilon(self):
        """Sparsity = minimum - epsilon should raise ValueError."""
        with pytest.raises(ValueError, match="lower than minimum value"):
            SparsityValidator.validate_sparsity_bounds(0.09999, 0.1)


class TestValidateNumObsConsistency:
    """Tests for validate_num_obs_consistency method."""
    
    def test_matching_num_obs_passes_unchanged(self):
        """Matching num_obs should pass unchanged."""
        nb_coords = np.array([10, 10])
        num_obs, sparsity = SparsityValidator.validate_num_obs_consistency(
            50, 0.5, nb_coords
        )
        assert num_obs == 50
        assert sparsity == 0.5
    
    def test_non_matching_num_obs_adjusted(self):
        """Non-matching num_obs should be adjusted."""
        nb_coords = np.array([10, 10])
        num_obs, sparsity = SparsityValidator.validate_num_obs_consistency(
            51, 0.5, nb_coords
        )
        assert num_obs == 50
        assert sparsity == 0.5
    
    def test_returns_integer(self):
        """Should return integer num_obs."""
        nb_coords = np.array([10, 10])
        num_obs, _ = SparsityValidator.validate_num_obs_consistency(
            50.7, 0.5, nb_coords
        )
        assert isinstance(num_obs, int)
    
    def test_returns_adjusted_sparsity(self):
        """Should return adjusted sparsity."""
        nb_coords = np.array([10, 10])
        num_obs, sparsity = SparsityValidator.validate_num_obs_consistency(
            55, 0.5, nb_coords
        )
        assert sparsity == pytest.approx(0.5, rel=0.1)
    
    def test_adjusted_sparsity_in_bounds(self):
        """Adjusted sparsity should be in valid bounds."""
        nb_coords = np.array([10, 10])
        _, sparsity = SparsityValidator.validate_num_obs_consistency(
            50, 0.5, nb_coords
        )
        assert 0.0 <= sparsity <= 1.0
    
    def test_edge_case_large_grid(self):
        """Large grid should work."""
        nb_coords = np.array([100, 100, 100])
        num_obs, sparsity = SparsityValidator.validate_num_obs_consistency(
            100000, 0.1, nb_coords
        )
        assert num_obs == 100000
        assert sparsity == 0.1
    
    def test_edge_case_small_grid(self):
        """Small grid should work."""
        nb_coords = np.array([2, 2])
        num_obs, sparsity = SparsityValidator.validate_num_obs_consistency(
            2, 0.5, nb_coords
        )
        assert num_obs == 2
        assert sparsity == 0.5
    
    def test_out_of_bounds_after_adjustment_raises_value_error(self):
        """Out of bounds after adjustment should raise ValueError."""
        # Create a case where adjusted sparsity would be > 1
        nb_coords = np.array([2, 2])
        # Grid size = 4, so 5 obs = sparsity > 1
        with pytest.raises(ValueError, match="out of bounds"):
            SparsityValidator.validate_num_obs_consistency(
                5, 1.5, nb_coords
            )
    
    def test_sparsity_recalculation_correct(self):
        """Sparsity recalculation should be correct."""
        nb_coords = np.array([10, 10])
        num_obs, sparsity = SparsityValidator.validate_num_obs_consistency(
            30, 0.3, nb_coords
        )
        expected_sparsity = num_obs / np.prod(nb_coords)
        assert sparsity == pytest.approx(expected_sparsity)
