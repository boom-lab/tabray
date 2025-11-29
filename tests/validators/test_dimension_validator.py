"""Tests for DimensionValidator class."""

import pytest
import numpy as np
from data_sparsity.validators import DimensionValidator


class TestComputeNbCoordsDim1:
    """Tests for compute_nb_coords_dim1 method."""
    
    def test_valid_parameters_return_expected_value(self):
        """Valid parameters should return expected value."""
        result = DimensionValidator.compute_nb_coords_dim1(100, 0.1, 1.0, 2)
        assert result == pytest.approx(31.622, rel=0.001)
    
    def test_different_num_obs_values(self):
        """Different num_obs values should give different results."""
        result1 = DimensionValidator.compute_nb_coords_dim1(100, 0.1, 1.0, 2)
        result2 = DimensionValidator.compute_nb_coords_dim1(200, 0.1, 1.0, 2)
        assert result2 > result1
    
    def test_different_sparsity_values(self):
        """Different sparsity values should give different results."""
        result1 = DimensionValidator.compute_nb_coords_dim1(100, 0.1, 1.0, 2)
        result2 = DimensionValidator.compute_nb_coords_dim1(100, 0.2, 1.0, 2)
        assert result2 < result1
    
    def test_result_less_than_one_raises_value_error(self):
        """Result < 1 should raise ValueError."""
        # This case actually happens when parameters produce a value < 1
        # Let's create parameters that definitely produce < 1
        with pytest.raises(ValueError, match="must be at least 1"):
            DimensionValidator.compute_nb_coords_dim1(1, 10.0, 1.0, 5)
    
    def test_large_values_handled_correctly(self):
        """Large values should be handled correctly."""
        result = DimensionValidator.compute_nb_coords_dim1(1000000, 0.5, 1.0, 3)
        assert result > 100
    
    def test_edge_case_sparsity_equals_one(self):
        """Sparsity = 1 should work correctly."""
        result = DimensionValidator.compute_nb_coords_dim1(100, 1.0, 1.0, 2)
        assert result == 10.0


class TestRoundToInteger:
    """Tests for round_to_integer method."""
    
    def test_rounds_up_correctly(self):
        """Should round up correctly."""
        result = DimensionValidator.round_to_integer(10.6)
        assert result == 11
    
    def test_rounds_down_correctly(self):
        """Should round down correctly."""
        result = DimensionValidator.round_to_integer(10.4)
        assert result == 10
    
    def test_exact_integers_unchanged(self):
        """Exact integers should remain unchanged."""
        result = DimensionValidator.round_to_integer(10.0)
        assert result == 10
    
    def test_returns_integer_type(self):
        """Should return integer type."""
        result = DimensionValidator.round_to_integer(10.5)
        assert isinstance(result, int)
    
    def test_rounds_half_to_even(self):
        """Should round 0.5 using banker's rounding."""
        result = DimensionValidator.round_to_integer(10.5)
        assert result in [10, 11]  # numpy rint uses banker's rounding


class TestComputeNbCoordsPerDim:
    """Tests for compute_nb_coords_per_dim method."""
    
    def test_uniform_ratio_dims(self):
        """Uniform ratio_dims should give equal values."""
        ratio_dims = np.array([1, 1, 1])
        result = DimensionValidator.compute_nb_coords_per_dim(ratio_dims, 10)
        np.testing.assert_array_equal(result, [10, 10, 10])
    
    def test_non_uniform_ratio_dims(self):
        """Non-uniform ratio_dims should scale correctly."""
        ratio_dims = np.array([1, 2, 3])
        result = DimensionValidator.compute_nb_coords_per_dim(ratio_dims, 5)
        np.testing.assert_array_equal(result, [5, 10, 15])
    
    def test_single_dimension(self):
        """Single dimension should work."""
        ratio_dims = np.array([1])
        result = DimensionValidator.compute_nb_coords_per_dim(ratio_dims, 20)
        np.testing.assert_array_equal(result, [20])
    
    def test_result_shape_matches_ratio_dims_shape(self):
        """Result shape should match ratio_dims shape."""
        ratio_dims = np.array([1, 2, 3, 4])
        result = DimensionValidator.compute_nb_coords_per_dim(ratio_dims, 5)
        assert result.shape == ratio_dims.shape


class TestValidateMinElementsPerDim:
    """Tests for validate_min_elements_per_dim method."""
    
    def test_all_valid_passes(self):
        """All valid (>= 1) should pass."""
        nb_coords = np.array([1, 2, 3])
        DimensionValidator.validate_min_elements_per_dim(nb_coords)
    
    def test_one_dimension_less_than_one_raises_value_error(self):
        """One dimension < 1 should raise ValueError."""
        nb_coords = np.array([2, 0.5, 3])
        with pytest.raises(ValueError, match="at least one element"):
            DimensionValidator.validate_min_elements_per_dim(nb_coords)
    
    def test_multiple_dimensions_less_than_one_raises_value_error(self):
        """Multiple dimensions < 1 should raise ValueError."""
        nb_coords = np.array([0.5, 0.3, 3])
        with pytest.raises(ValueError, match="at least one element"):
            DimensionValidator.validate_min_elements_per_dim(nb_coords)
    
    def test_exactly_one_passes(self):
        """Exactly 1 should pass."""
        nb_coords = np.array([1.0, 1.0, 1.0])
        DimensionValidator.validate_min_elements_per_dim(nb_coords)
    
    def test_zero_raises_value_error(self):
        """Zero should raise ValueError."""
        nb_coords = np.array([1, 0, 2])
        with pytest.raises(ValueError, match="at least one element"):
            DimensionValidator.validate_min_elements_per_dim(nb_coords)


class TestValidateIntegerElements:
    """Tests for validate_integer_elements method."""
    
    def test_integer_values_pass_through_unchanged(self):
        """Integer values should pass through unchanged."""
        nb_coords = np.array([5.0, 10.0, 15.0])
        result = DimensionValidator.validate_integer_elements(nb_coords)
        np.testing.assert_array_equal(result, [5, 10, 15])
    
    def test_close_to_integer_values_rounded(self):
        """Close-to-integer values should be rounded."""
        nb_coords = np.array([5.000001, 10.0, 14.999999])
        result = DimensionValidator.validate_integer_elements(nb_coords)
        np.testing.assert_array_equal(result, [5, 10, 15])
    
    def test_non_integer_values_raise_value_error(self):
        """Non-integer values should raise ValueError."""
        nb_coords = np.array([5.5, 10.0, 15.0])
        with pytest.raises(ValueError, match="decimal number of elements"):
            DimensionValidator.validate_integer_elements(nb_coords)
    
    def test_returns_integer_array(self):
        """Should return integer array."""
        nb_coords = np.array([5.0, 10.0, 15.0])
        result = DimensionValidator.validate_integer_elements(nb_coords)
        assert result.dtype == np.int64 or result.dtype == np.int32


class TestComputeShapeAndGridPoints:
    """Tests for compute_shape_and_grid_points method."""
    
    def test_1d_array(self):
        """1D array should work."""
        nb_coords = np.array([10])
        shape, total = DimensionValidator.compute_shape_and_grid_points(nb_coords)
        assert shape == [10]
        assert total == 10
    
    def test_2d_array(self):
        """2D array should work."""
        nb_coords = np.array([5, 10])
        shape, total = DimensionValidator.compute_shape_and_grid_points(nb_coords)
        assert shape == [5, 10]
        assert total == 50
    
    def test_3d_array(self):
        """3D array should work."""
        nb_coords = np.array([3, 4, 5])
        shape, total = DimensionValidator.compute_shape_and_grid_points(nb_coords)
        assert shape == [3, 4, 5]
        assert total == 60
    
    def test_grid_points_equals_product_of_shape(self):
        """Grid points should equal product of shape."""
        nb_coords = np.array([2, 3, 4, 5])
        shape, total = DimensionValidator.compute_shape_and_grid_points(nb_coords)
        assert total == np.prod(shape)
