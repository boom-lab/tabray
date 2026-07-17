"""Tests for MultiVarSparsityConfig class."""

import pytest
import numpy as np
from data_sparsity.config import MultiVarSparsityConfig


class TestFromScalar:
    """Tests for from_scalar method."""
    
    def test_single_variable(self):
        """Single variable should create array of length 1."""
        result = MultiVarSparsityConfig.from_scalar(0.5, 1)
        assert len(result) == 1
        assert result[0] == 0.5
    
    def test_two_variables(self):
        """Two variables should create array of length 2."""
        result = MultiVarSparsityConfig.from_scalar(0.3, 2)
        assert len(result) == 2
        np.testing.assert_array_equal(result, [0.3, 0.3])
    
    def test_many_variables(self):
        """Many variables should all have same value."""
        result = MultiVarSparsityConfig.from_scalar(0.7, 5)
        assert len(result) == 5
        assert all(v == 0.7 for v in result)
    
    def test_array_shape_correct(self):
        """Array shape should match num_vars."""
        result = MultiVarSparsityConfig.from_scalar(0.4, 10)
        assert result.shape == (10,)
    
    def test_all_elements_equal(self):
        """All elements should be equal to input."""
        value = 0.8
        result = MultiVarSparsityConfig.from_scalar(value, 7)
        np.testing.assert_array_equal(result, np.full(7, value))


class TestFromTwoElementList:
    """Tests for from_two_element_list method."""
    
    def test_two_variables_random_assignment(self, fixed_rng):
        """Two variables should get min and max."""
        result = MultiVarSparsityConfig.from_two_element_list([0.2, 0.8], 2, fixed_rng)
        assert len(result) == 2
        assert 0.2 in result
        assert 0.8 in result
    
    def test_three_variables_includes_min_max_random(self, fixed_rng):
        """Three variables should include min, max, and random value."""
        result = MultiVarSparsityConfig.from_two_element_list([0.1, 0.9], 3, fixed_rng)
        assert len(result) == 3
        assert min(result) >= 0.1
        assert max(result) <= 0.9
        assert 0.1 in result or 0.9 in result
    
    def test_many_variables_distribution_correct(self, fixed_rng):
        """Many variables should have values in [min, max]."""
        result = MultiVarSparsityConfig.from_two_element_list([0.2, 0.8], 10, fixed_rng)
        assert len(result) == 10
        assert all(0.2 <= v <= 0.8 for v in result)
    
    def test_all_elements_in_range(self, fixed_rng):
        """All elements should be in [min, max]."""
        min_val, max_val = 0.3, 0.7
        result = MultiVarSparsityConfig.from_two_element_list([min_val, max_val], 20, fixed_rng)
        assert np.all(result >= min_val)
        assert np.all(result <= max_val)
    
    def test_contains_exactly_one_min(self, fixed_rng):
        """Should contain exactly one minimum value."""
        result = MultiVarSparsityConfig.from_two_element_list([0.1, 0.9], 5, fixed_rng)
        assert np.sum(result == 0.1) >= 1
    
    def test_contains_exactly_one_max(self, fixed_rng):
        """Should contain exactly one maximum value."""
        result = MultiVarSparsityConfig.from_two_element_list([0.1, 0.9], 5, fixed_rng)
        assert np.sum(result == 0.9) >= 1
    
    def test_reproducible_with_same_seed(self):
        """Same seed should give same result."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        result1 = MultiVarSparsityConfig.from_two_element_list([0.2, 0.8], 5, rng1)
        result2 = MultiVarSparsityConfig.from_two_element_list([0.2, 0.8], 5, rng2)
        np.testing.assert_array_equal(result1, result2)
    
    def test_different_with_different_seed(self):
        """Different seed should give different result."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(456)
        result1 = MultiVarSparsityConfig.from_two_element_list([0.2, 0.8], 10, rng1)
        result2 = MultiVarSparsityConfig.from_two_element_list([0.2, 0.8], 10, rng2)
        assert not np.array_equal(result1, result2)
    
    def test_order_shuffled(self, fixed_rng):
        """Values should be shuffled, not sorted."""
        result = MultiVarSparsityConfig.from_two_element_list([0.1, 0.9], 10, fixed_rng)
        # Check that it's not simply sorted
        is_sorted = np.all(result[:-1] <= result[1:])
        is_reverse_sorted = np.all(result[:-1] >= result[1:])
        assert not (is_sorted or is_reverse_sorted)
    
    def test_array_length_matches_num_vars(self, fixed_rng):
        """Array length should match num_vars."""
        for num_vars in [2, 5, 10, 20]:
            result = MultiVarSparsityConfig.from_two_element_list([0.2, 0.8], num_vars, fixed_rng)
            assert len(result) == num_vars


class TestFromFullList:
    """Tests for from_full_list method."""
    
    def test_correct_length_matches(self):
        """Correct length should pass through."""
        sparsity_list = [0.2, 0.5, 0.8]
        result = MultiVarSparsityConfig.from_full_list(sparsity_list, 3)
        np.testing.assert_array_equal(result, sparsity_list)
    
    def test_values_preserved(self):
        """Values should be preserved exactly."""
        sparsity_list = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = MultiVarSparsityConfig.from_full_list(sparsity_list, 5)
        for i, val in enumerate(sparsity_list):
            assert result[i] == val
    
    def test_wrong_length_raises_value_error(self):
        """Wrong length should raise ValueError."""
        with pytest.raises(ValueError, match=r"density list must have \d+ elements, got \d+"):
            MultiVarSparsityConfig.from_full_list([0.2, 0.5], 3)
    
    def test_empty_list_raises_value_error(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError, match=r"density list must have \d+ elements, got \d+"):
            MultiVarSparsityConfig.from_full_list([], 1)
    
    def test_single_element_for_single_var(self):
        """Single element for single var should work."""
        result = MultiVarSparsityConfig.from_full_list([0.5], 1)
        assert result[0] == 0.5


class TestValidateAndClip:
    """Tests for validate_and_clip method."""
    
    def test_all_valid_values_pass(self):
        """All valid values should pass unchanged."""
        sparsities = np.array([0.2, 0.5, 0.8])
        result = MultiVarSparsityConfig.validate_and_clip(sparsities, 0.1)
        np.testing.assert_array_equal(result, sparsities)
    
    def test_value_below_minimum_clipped(self):
        """Value below minimum should be clipped."""
        sparsities = np.array([0.05, 0.5, 0.8])
        result = MultiVarSparsityConfig.validate_and_clip(sparsities, 0.1)
        assert result[0] == 0.1
        assert result[1] == 0.5
        assert result[2] == 0.8
    
    def test_multiple_values_clipped(self):
        """Multiple values below minimum should be clipped."""
        sparsities = np.array([0.05, 0.08, 0.5])
        result = MultiVarSparsityConfig.validate_and_clip(sparsities, 0.1)
        assert result[0] == 0.1
        assert result[1] == 0.1
        assert result[2] == 0.5
    
    def test_value_greater_than_one_raises_value_error(self):
        """Value > 1 should raise ValueError."""
        sparsities = np.array([0.5, 1.5])
        with pytest.raises(ValueError, match="must be between"):
            MultiVarSparsityConfig.validate_and_clip(sparsities, 0.1)
    
    def test_negative_value_raises_value_error(self):
        """Negative value should raise ValueError."""
        sparsities = np.array([-0.1, 0.5])
        with pytest.raises(ValueError, match="must be between"):
            MultiVarSparsityConfig.validate_and_clip(sparsities, 0.1)
    
    def test_zero_handled_correctly(self):
        """Zero should be clipped to minimum."""
        sparsities = np.array([0.0, 0.5])
        result = MultiVarSparsityConfig.validate_and_clip(sparsities, 0.1)
        assert result[0] == 0.1
    
    def test_returns_modified_array(self):
        """Should return modified array."""
        sparsities = np.array([0.05, 0.5])
        result = MultiVarSparsityConfig.validate_and_clip(sparsities, 0.1)
        assert isinstance(result, np.ndarray)
        assert result[0] == 0.1


class TestComputeVarNumObs:
    """Tests for compute_var_num_obs method."""
    
    def test_single_variable(self):
        """Single variable should get all observations."""
        sparsities = np.array([0.5])
        result = MultiVarSparsityConfig.compute_var_num_obs(sparsities, 100)
        assert result[0] == 100
    
    def test_equal_sparsities(self):
        """Equal sparsities should get equal observations."""
        sparsities = np.array([0.5, 0.5, 0.5])
        result = MultiVarSparsityConfig.compute_var_num_obs(sparsities, 100)
        np.testing.assert_array_equal(result, [100, 100, 100])
    
    def test_different_sparsities(self):
        """Different sparsities should scale proportionally."""
        sparsities = np.array([0.3, 0.6])
        result = MultiVarSparsityConfig.compute_var_num_obs(sparsities, 100)
        assert result[1] == 100  # Max gets full num_obs
        assert result[0] == 50   # Half of max
    
    def test_max_sparsity_gets_full_num_obs(self):
        """Variable with max sparsity should get full num_obs."""
        sparsities = np.array([0.2, 0.8, 0.5])
        result = MultiVarSparsityConfig.compute_var_num_obs(sparsities, 200)
        assert result[1] == 200
    
    def test_minimum_one_observation_per_variable(self):
        """Each variable should get at least 1 observation."""
        sparsities = np.array([0.001, 1.0])
        result = MultiVarSparsityConfig.compute_var_num_obs(sparsities, 10)
        assert all(obs >= 1 for obs in result)


class TestSetupFromParameter:
    """Tests for setup_from_parameter integration."""
    
    def test_scalar_input(self, fixed_rng):
        """Scalar input should work."""
        sparsities, num_obs = MultiVarSparsityConfig.setup_from_parameter(
            0.5, 3, 100, 0.1, fixed_rng
        )
        assert len(sparsities) == 3
        assert len(num_obs) == 3
        assert all(s >= 0.1 for s in sparsities)
    
    def test_list_input(self, fixed_rng):
        """List input should work."""
        sparsities, num_obs = MultiVarSparsityConfig.setup_from_parameter(
            [0.2, 0.8], 2, 100, 0.1, fixed_rng
        )
        assert len(sparsities) == 2
        assert len(num_obs) == 2
