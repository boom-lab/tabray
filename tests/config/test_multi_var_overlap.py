"""Tests for MultiVarOverlapConfig class."""

import pytest
import numpy as np
from data_sparsity.config import MultiVarOverlapConfig


class TestValidateOverlapValue:
    """Tests for validate_overlap_value method."""
    
    def test_valid_float_in_range_passes(self):
        """Valid float in [0,1] should pass."""
        MultiVarOverlapConfig.validate_overlap_value(0.5)
        MultiVarOverlapConfig.validate_overlap_value(0.0)
        MultiVarOverlapConfig.validate_overlap_value(1.0)
    
    def test_float_zero_passes(self):
        """Float 0 should pass."""
        MultiVarOverlapConfig.validate_overlap_value(0.0)
    
    def test_float_one_passes(self):
        """Float 1 should pass."""
        MultiVarOverlapConfig.validate_overlap_value(1.0)
    
    def test_float_less_than_zero_raises_value_error(self):
        """Float < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            MultiVarOverlapConfig.validate_overlap_value(-0.1)
    
    def test_float_greater_than_one_raises_value_error(self):
        """Float > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            MultiVarOverlapConfig.validate_overlap_value(1.5)
    
    def test_string_random_passes(self):
        """String 'random' should pass."""
        MultiVarOverlapConfig.validate_overlap_value('random')
    
    def test_invalid_string_raises_value_error(self):
        """Invalid string should raise ValueError."""
        with pytest.raises(ValueError, match="must be 'random'"):
            MultiVarOverlapConfig.validate_overlap_value('invalid')
    
    def test_invalid_type_raises_type_error(self):
        """Invalid type should raise TypeError."""
        with pytest.raises(TypeError, match="must be"):
            MultiVarOverlapConfig.validate_overlap_value([0.5])


class TestComputeMinOverlap:
    """Tests for compute_min_overlap method."""
    
    def test_no_shared_dimensions_returns_zero(self):
        """No shared dimensions should return 0."""
        var_dims_indices = [[0], [1]]
        result = MultiVarOverlapConfig.compute_min_overlap(var_dims_indices)
        assert result == 0.0
    
    def test_all_shared_dimensions_returns_one(self):
        """All shared dimensions should return 1."""
        var_dims_indices = [[0, 1, 2], [0, 1, 2]]
        result = MultiVarOverlapConfig.compute_min_overlap(var_dims_indices)
        assert result == 1.0
    
    def test_partial_overlap_correct_ratio(self):
        """Partial overlap should return correct ratio."""
        var_dims_indices = [[0, 1], [0, 1, 2]]
        result = MultiVarOverlapConfig.compute_min_overlap(var_dims_indices)
        assert result == pytest.approx(2.0/3.0)
    
    def test_two_variables(self):
        """Two variables should work."""
        var_dims_indices = [[0], [0, 1]]
        result = MultiVarOverlapConfig.compute_min_overlap(var_dims_indices)
        assert result == 0.5
    
    def test_many_variables(self):
        """Many variables should compute correctly."""
        var_dims_indices = [[0, 1], [0, 2], [1, 2]]
        result = MultiVarOverlapConfig.compute_min_overlap(var_dims_indices)
        assert 0.0 <= result <= 1.0
    
    def test_different_dimension_combinations(self):
        """Different combinations should give different results."""
        var_dims_1 = [[0], [1]]
        var_dims_2 = [[0], [0, 1]]
        result1 = MultiVarOverlapConfig.compute_min_overlap(var_dims_1)
        result2 = MultiVarOverlapConfig.compute_min_overlap(var_dims_2)
        assert result1 != result2
    
    def test_single_variable_returns_zero(self):
        """Single variable should return 0."""
        var_dims_indices = [[0, 1, 2]]
        result = MultiVarOverlapConfig.compute_min_overlap(var_dims_indices)
        assert result == 0.0
    
    def test_empty_list_returns_zero(self):
        """Empty list should return 0."""
        var_dims_indices = []
        result = MultiVarOverlapConfig.compute_min_overlap(var_dims_indices)
        assert result == 0.0
    
    def test_complex_scenario(self):
        """Complex scenario should compute correctly."""
        var_dims_indices = [[0, 2], [1, 2], [2]]
        result = MultiVarOverlapConfig.compute_min_overlap(var_dims_indices)
        # Should be based on smallest shared dimension set
        assert 0.0 <= result <= 1.0
    
    def test_edge_case_one_var_one_dim_other_all_dims(self):
        """One var with 1 dim, other with all dims."""
        var_dims_indices = [[0], [0, 1, 2]]
        result = MultiVarOverlapConfig.compute_min_overlap(var_dims_indices)
        assert result == pytest.approx(1.0/3.0)


class TestValidateOverlapFeasibility:
    """Tests for validate_overlap_feasibility method."""
    
    def test_feasible_overlap_passes(self):
        """Feasible overlap should pass."""
        var_dims_indices = [[0, 1], [0, 2]]
        min_overlap = MultiVarOverlapConfig.compute_min_overlap(var_dims_indices)
        MultiVarOverlapConfig.validate_overlap_feasibility(
            min_overlap + 0.1, min_overlap, 2
        )
    
    def test_infeasible_raises_value_error(self):
        """Infeasible overlap should raise ValueError."""
        with pytest.raises(ValueError, match="below minimum feasible"):
            MultiVarOverlapConfig.validate_overlap_feasibility(0.3, 0.5, 2)
    
    def test_single_variable_passes(self):
        """Single variable should pass without check."""
        MultiVarOverlapConfig.validate_overlap_feasibility(0.5, 1.0, 1)
    
    def test_random_overlap_passes(self):
        """Random overlap should pass without check."""
        MultiVarOverlapConfig.validate_overlap_feasibility('random', 0.5, 2)
    
    def test_exact_minimum_passes(self):
        """Exact minimum should pass."""
        MultiVarOverlapConfig.validate_overlap_feasibility(0.5, 0.5, 2)


class TestSetupFromParameter:
    """Tests for setup_from_parameter integration."""
    
    def test_float_overlap(self):
        """Float overlap should work."""
        var_dims_indices = [[0, 1], [0, 2]]
        result = MultiVarOverlapConfig.setup_from_parameter(
            0.5, 2, var_dims_indices
        )
        assert result == 0.5
    
    def test_random_overlap(self):
        """Random overlap should return 'random'."""
        var_dims_indices = [[0, 1], [0, 2]]
        result = MultiVarOverlapConfig.setup_from_parameter(
            'random', 2, var_dims_indices
        )
        assert result == 'random'
