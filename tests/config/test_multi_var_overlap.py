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
            MultiVarOverlapConfig.validate_overlap_value({"value": 0.5})

    def test_list_overlap_values_pass(self):
        """A list of numeric overlap values should pass validation."""
        result = MultiVarOverlapConfig.validate_overlap_value([0.25, 0.5])
        assert result == [0.25, 0.5]


class TestValidateFixedOverlapValue:
    """Tests for validate_fixed_overlap_value method."""

    def test_scalar_false_expands_to_all_false(self):
        """False should expand to one flag per non-reference variable."""
        result = MultiVarOverlapConfig.validate_fixed_overlap_value(False, 4)
        assert result == [False, False, False]

    def test_scalar_true_expands_to_all_true(self):
        """True should expand to one flag per non-reference variable."""
        result = MultiVarOverlapConfig.validate_fixed_overlap_value(True, 3)
        assert result == [True, True]

    def test_list_is_preserved(self):
        """A valid list should be preserved."""
        result = MultiVarOverlapConfig.validate_fixed_overlap_value(
            [True, False], 3
        )
        assert result == [True, False]

    def test_wrong_length_raises_value_error(self):
        """A wrong-length list should raise ValueError."""
        with pytest.raises(ValueError, match="must contain 2 values"):
            MultiVarOverlapConfig.validate_fixed_overlap_value([True], 3)

    def test_non_boolean_value_raises_type_error(self):
        """Non-boolean list values should raise TypeError."""
        with pytest.raises(TypeError, match="must be booleans"):
            MultiVarOverlapConfig.validate_fixed_overlap_value([True, 1], 3)


class TestComputeMinOverlap:
    """Tests for compute_min_overlap method."""
    
    def test_sufficient_grid_points_returns_zero(self):
        """Sufficient grid points for all observations should return 0."""
        total_grid_points = 1000
        var_num_obs = np.array([300, 200, 100])
        result = MultiVarOverlapConfig.compute_min_overlap(total_grid_points, var_num_obs)
        assert result == 0.0
    
    def test_insufficient_grid_points_computes_overlap(self):
        """Insufficient grid points should compute required overlap."""
        total_grid_points = 100
        var_num_obs = np.array([80, 50])  # ref=80, others=50, total=130 > 100
        # must_overlap = 130 - 100 = 30; F1 divides by the REFERENCE count,
        # so min_overlap = 30/80 = 0.375
        result = MultiVarOverlapConfig.compute_min_overlap(total_grid_points, var_num_obs)
        assert result == pytest.approx(0.375)
    
    def test_exact_fit_returns_zero(self):
        """Exact fit (total_sites == max + others) should return 0."""
        total_grid_points = 150
        var_num_obs = np.array([100, 50])
        result = MultiVarOverlapConfig.compute_min_overlap(total_grid_points, var_num_obs)
        assert result == 0.0
    
    def test_two_variables(self):
        """Two variables should work correctly."""
        total_grid_points = 80
        var_num_obs = np.array([60, 40])  # ref=60, others=40, total=100 > 80
        # must_overlap = 100 - 80 = 20, min_overlap = 20/60 = 1/3
        result = MultiVarOverlapConfig.compute_min_overlap(total_grid_points, var_num_obs)
        assert result == pytest.approx(1.0 / 3.0)
    
    def test_many_variables(self):
        """Many variables should compute correctly."""
        total_grid_points = 100
        var_num_obs = np.array([50, 30, 25, 20])  # ref=50, others=75, total=125 > 100
        # must_overlap = 125 - 100 = 25, min_overlap = 25/50 = 0.5
        result = MultiVarOverlapConfig.compute_min_overlap(total_grid_points, var_num_obs)
        assert result == pytest.approx(0.5)
    
    def test_very_tight_space(self):
        """Very limited grid points should give high overlap."""
        total_grid_points = 10
        var_num_obs = np.array([20, 10, 5])  # max=20, others=15, total=35 > 10
        # must_overlap = 35 - 10 = 25, min_overlap = 25/20 = 1.25 > 1.0
        result = MultiVarOverlapConfig.compute_min_overlap(total_grid_points, var_num_obs)
        assert result > 1.0
    
    def test_single_other_observation(self):
        """Single observation in non-max variable."""
        total_grid_points = 50
        var_num_obs = np.array([100, 1])  # ref=100, other=1, total=101 > 50
        # must_overlap = 101 - 50 = 51, min_overlap = 51/100 = 0.51
        result = MultiVarOverlapConfig.compute_min_overlap(total_grid_points, var_num_obs)
        assert result == pytest.approx(0.51)
    
    def test_zero_other_observations_raises(self):
        """Zero observations in non-max variables should raise ValueError."""
        total_grid_points = 100
        var_num_obs = np.array([100, 0])
        with pytest.raises(ValueError, match="no observations in non-reference"):
            MultiVarOverlapConfig.compute_min_overlap(total_grid_points, var_num_obs)
    
    def test_reference_is_var0_not_the_largest(self):
        """var0 is the reference by definition, whatever the counts say.

        validate_reference_is_largest rejects configurations where var0 is not
        the largest, so this only matters for direct calls; the point is that
        the denominator is var0's count and not a sorted maximum.
        """
        total_grid_points = 100
        var_num_obs = np.array([30, 80, 20])  # ref=30 (var0), others=100
        # must_overlap = 130 - 100 = 30, min_overlap = 30/30 = 1.0
        result = MultiVarOverlapConfig.compute_min_overlap(total_grid_points, var_num_obs)
        assert result == pytest.approx(1.0)


class TestValidateOverlapFeasibility:
    """Tests for validate_overlap_feasibility method."""
    
    def test_feasible_overlap_passes(self):
        """Feasible overlap should pass."""
        total_grid_points = 100
        var_num_obs = np.array([80, 50])
        min_overlap = MultiVarOverlapConfig.compute_min_overlap(
            total_grid_points, var_num_obs
        )
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
        shape = [10, 10, 10]  # 1000 total grid points
        var_num_obs = np.array([300, 200])
        var_dims_indices = [[0, 1, 2], [0, 1, 2]]  # Both use all dimensions
        overlap, adjusted_obs, fixed_overlap = MultiVarOverlapConfig.setup_from_parameter(
            0.5, False, 2, shape, var_num_obs, var_dims_indices
        )
        assert overlap == 0.5
        np.testing.assert_array_equal(adjusted_obs, var_num_obs)  # No adjustment needed
        assert fixed_overlap == [False]
    
    def test_random_overlap(self):
        """Random overlap should return 'random'."""
        shape = [10, 10, 10]  # 1000 total grid points
        var_num_obs = np.array([300, 200])
        var_dims_indices = [[0, 1, 2], [0, 1, 2]]  # Both use all dimensions
        overlap, adjusted_obs, fixed_overlap = MultiVarOverlapConfig.setup_from_parameter(
            'random', False, 2, shape, var_num_obs, var_dims_indices
        )
        assert overlap == 'random'
        np.testing.assert_array_equal(adjusted_obs, var_num_obs)  # No adjustment needed
        assert fixed_overlap == [False]

    def test_per_variable_overlap(self):
        """A per-variable overlap list should be preserved and applied."""
        shape = [5, 5]
        var_num_obs = np.array([12, 10, 8])
        var_dims_indices = [[0, 1], [0, 1], [0, 1]]

        overlap, adjusted_obs, fixed_overlap = MultiVarOverlapConfig.setup_from_parameter(
            [0.5, 0.25], [True, False], 3, shape, var_num_obs, var_dims_indices
        )

        assert overlap == [0.5, 0.25]
        np.testing.assert_array_equal(adjusted_obs, var_num_obs)
        assert fixed_overlap == [True, False]

    def test_per_variable_overlap_length_mismatch(self):
        """Overlap lists with the wrong length should fail validation."""
        shape = [5, 5]
        var_num_obs = np.array([12, 10, 8])
        var_dims_indices = [[0, 1], [0, 1], [0, 1]]

        with pytest.raises(ValueError, match="must contain 2 values"):
            MultiVarOverlapConfig.setup_from_parameter(
                [0.5], False, 3, shape, var_num_obs, var_dims_indices
            )

    def test_reference_must_be_largest(self):
        """var0 smaller than another variable should fail validation."""
        shape = [5, 5]
        var_num_obs = np.array([8, 10, 6])
        var_dims_indices = [[0, 1], [0, 1], [0, 1]]

        with pytest.raises(ValueError, match="var0 must be the largest variable"):
            MultiVarOverlapConfig.setup_from_parameter(
                [0.5, 0.25], False, 3, shape, var_num_obs, var_dims_indices
            )
    
    def test_observation_adjustment(self):
        """Should adjust observations when they exceed grid space."""
        shape = [5, 5, 5]  # 125 total grid points
        var_num_obs = np.array([100, 50])  # var1 with 2 dims has only 25 points
        var_dims_indices = [[0, 1, 2], [1, 2]]  # var1 has 5*5=25 grid points
        
        overlap, adjusted_obs, fixed_overlap = MultiVarOverlapConfig.setup_from_parameter(
            0.5, False, 2, shape, var_num_obs, var_dims_indices
        )
        
        # var0 should remain unchanged
        assert adjusted_obs[0] == 100
        # var1 should be reduced to grid capacity (25 points)
        assert adjusted_obs[1] <= 25
        # With overlap 0.5, target would be 0.5*100=50, but max is 25
        assert adjusted_obs[1] == 25
        assert fixed_overlap == [False]
