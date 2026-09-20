"""Tests for ParameterValidator class."""

import pytest
import numpy as np
from data_sparsity.validators import ParameterValidator


class TestValidateNumObs:
    """Tests for validate_num_obs method."""
    
    def test_valid_positive_integer(self):
        """Valid positive integer should pass."""
        ParameterValidator.validate_num_obs(100)
        ParameterValidator.validate_num_obs(1)
        ParameterValidator.validate_num_obs(1000000)
    
    def test_zero_raises_value_error(self):
        """Zero should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            ParameterValidator.validate_num_obs(0)
    
    def test_negative_raises_value_error(self):
        """Negative should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            ParameterValidator.validate_num_obs(-10)
    
    def test_float_raises_type_error(self):
        """Float should raise TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            ParameterValidator.validate_num_obs(100.5)
    
    def test_string_raises_type_error(self):
        """String should raise TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            ParameterValidator.validate_num_obs("100")
    
    def test_none_raises_type_error(self):
        """None should raise TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            ParameterValidator.validate_num_obs(None)


class TestValidateDensityType:
    """Tests for validate_density_type method."""
    
    def test_valid_float_returns_float(self):
        """Valid float should return float."""
        result = ParameterValidator.validate_density_type(0.5)
        assert result == 0.5
        assert isinstance(result, float)
    
    def test_valid_int_returns_float(self):
        """Valid int should return float."""
        result = ParameterValidator.validate_density_type(1)
        assert result == 1.0
        assert isinstance(result, float)
    
    def test_valid_list_returns_max(self):
        """Valid list should return max value."""
        result = ParameterValidator.validate_density_type([0.2, 0.5, 0.3])
        assert result == 0.5
    
    def test_valid_tuple_returns_max(self):
        """Valid tuple should return max value."""
        result = ParameterValidator.validate_density_type((0.1, 0.9))
        assert result == 0.9
    
    def test_zero_is_allowed(self):
        """Sparsity = 0 is now allowed (will be converted to minimum later)."""
        result = ParameterValidator.validate_density_type(0.0)
        assert result == 0.0
    
    def test_negative_raises_value_error(self):
        """Negative value should raise ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            ParameterValidator.validate_density_type(-0.1)
    
    def test_greater_than_one_raises_value_error(self):
        """Value > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            ParameterValidator.validate_density_type(1.5)
    
    def test_string_raises_type_error(self):
        """String should raise TypeError."""
        with pytest.raises(TypeError, match="must be a number, list, or tuple"):
            ParameterValidator.validate_density_type("0.5")
    
    def test_list_with_invalid_values_raises_value_error(self):
        """List with values outside [0,1] should raise ValueError."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            ParameterValidator.validate_density_type([0.5, 1.5])


class TestValidateNumDims:
    """Tests for validate_num_dims method."""
    
    def test_valid_positive_integer(self):
        """Valid positive integer should pass."""
        ParameterValidator.validate_num_dims(3)
        ParameterValidator.validate_num_dims(1)
        ParameterValidator.validate_num_dims(10)
    
    def test_zero_raises_value_error(self):
        """Zero should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            ParameterValidator.validate_num_dims(0)
    
    def test_negative_raises_value_error(self):
        """Negative should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            ParameterValidator.validate_num_dims(-1)
    
    def test_float_raises_type_error(self):
        """Float should raise TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            ParameterValidator.validate_num_dims(3.5)
    
    def test_string_raises_type_error(self):
        """String should raise TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            ParameterValidator.validate_num_dims("3")
    
    def test_none_raises_type_error(self):
        """None should raise TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            ParameterValidator.validate_num_dims(None)


class TestValidateVarDimsExplicitIndices:
    """var_dims given as explicit dimension indices, one list per variable."""

    def test_list_of_lists_is_accepted(self):
        """The form MultiVarDimensionsConfig supports, and the README uses.

        This used to raise TypeError: the reference check compared num_dims to
        var_dims[0] as a number, and var_dims[0] was a list.
        """
        result = ParameterValidator.validate_var_dims(
            [[0, 1, 2], [1, 2, 3], [0, 2, 3]], num_vars=3, num_dims=4
        )
        # var0 must occupy every dimension, so its entry is completed
        assert result[0] == [0, 1, 2, 3]
        assert result[1] == [1, 2, 3]
        assert result[2] == [0, 2, 3]

    def test_complete_reference_is_left_alone(self):
        """A reference already covering every dimension is untouched."""
        result = ParameterValidator.validate_var_dims(
            [[0, 1], [1]], num_vars=2, num_dims=2
        )
        assert result == [[0, 1], [1]]

    def test_caller_list_is_not_mutated(self):
        """Completing the reference must not write through to the argument.

        var_dims_update was bound to the caller's list and then assigned at
        index 0, so the caller's object changed underneath them: passing
        [2, 2] left the caller holding [3, 2].

        Uses the integer form deliberately. The list-of-lists form raised
        TypeError before reaching that assignment, so it cannot distinguish
        the mutation from the crash.
        """
        var_dims = [2, 2]

        result = ParameterValidator.validate_var_dims(var_dims, num_vars=2, num_dims=3)

        assert result == [3, 2], "reference variable should be given every dimension"
        assert var_dims == [2, 2], "the caller's list must be left alone"

    def test_caller_list_of_lists_is_not_mutated(self):
        """Same guarantee for the explicit-indices form."""
        var_dims = [[0, 1, 2], [1, 2]]

        result = ParameterValidator.validate_var_dims(var_dims, num_vars=2, num_dims=4)

        assert result[0] == [0, 1, 2, 3]
        assert var_dims == [[0, 1, 2], [1, 2]]

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError, match="one entry per variable"):
            ParameterValidator.validate_var_dims([], num_vars=2, num_dims=3)

    def test_bad_entry_type_raises(self):
        with pytest.raises(TypeError, match="int .* or a list/tuple"):
            ParameterValidator.validate_var_dims(["x", 2], num_vars=2, num_dims=3)

    def test_integer_entries_still_work(self):
        """The int-per-variable form is unchanged."""
        assert ParameterValidator.validate_var_dims([2, 2], num_vars=2, num_dims=3) == [3, 2]


class TestValidateRatioDims:
    """Tests for validate_ratio_dims method."""
    
    def test_integer_one_converts_to_list(self):
        """Integer 1 should convert to list of ones."""
        result = ParameterValidator.validate_ratio_dims(1, 3)
        np.testing.assert_array_equal(result, [1, 1, 1])
    
    def test_list_passes_through_as_array(self):
        """List should convert to numpy array."""
        result = ParameterValidator.validate_ratio_dims([1, 2, 3], 3)
        np.testing.assert_array_equal(result, [1, 2, 3])
        assert isinstance(result, np.ndarray)
    
    def test_tuple_converts_to_array(self):
        """Tuple should convert to numpy array."""
        result = ParameterValidator.validate_ratio_dims((1, 2, 3), 3)
        np.testing.assert_array_equal(result, [1, 2, 3])
        assert isinstance(result, np.ndarray)
    
    def test_length_mismatch_raises_value_error(self):
        """Length mismatch should raise ValueError."""
        with pytest.raises(ValueError, match="must match length of ratio_dims"):
            ParameterValidator.validate_ratio_dims([1, 2], 3)
    
    def test_invalid_type_raises_type_error(self):
        """Invalid type should raise TypeError."""
        with pytest.raises(TypeError, match="must be int, tuple, list, or numpy array"):
            ParameterValidator.validate_ratio_dims("123", 3)
    
    def test_integer_not_one_raises_value_error(self):
        """Integer != 1 should raise ValueError."""
        with pytest.raises(ValueError, match="must be 1"):
            ParameterValidator.validate_ratio_dims(2, 3)


class TestValidateSeed:
    """Tests for validate_seed method."""
    
    def test_valid_non_negative_integer(self):
        """Valid non-negative integer should pass."""
        ParameterValidator.validate_seed(0)
        ParameterValidator.validate_seed(42)
        ParameterValidator.validate_seed(999999)
    
    def test_negative_raises_value_error(self):
        """Negative should raise ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            ParameterValidator.validate_seed(-1)


class TestValidateNumVars:
    """Tests for validate_num_vars method."""
    
    def test_valid_positive_integer(self):
        """Valid positive integer should pass."""
        ParameterValidator.validate_num_vars(1)
        ParameterValidator.validate_num_vars(5)
        ParameterValidator.validate_num_vars(100)
    
    def test_zero_raises_value_error(self):
        """Zero should raise ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            ParameterValidator.validate_num_vars(0)
