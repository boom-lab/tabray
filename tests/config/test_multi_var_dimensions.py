"""Tests for MultiVarDimensionsConfig class."""

import pytest
import numpy as np
from data_sparsity.config import MultiVarDimensionsConfig


class TestSelectRandomDims:
    """Tests for select_random_dims method."""
    
    def test_correct_count_returned(self):
        """Should return correct number of dimensions."""
        result = MultiVarDimensionsConfig.select_random_dims(5, 3, 42)
        assert len(result) == 3
    
    def test_all_elements_in_valid_range(self):
        """All elements should be in [0, num_dims-1]."""
        result = MultiVarDimensionsConfig.select_random_dims(5, 3, 42)
        assert all(0 <= d < 5 for d in result)
    
    def test_no_duplicates(self):
        """Should have no duplicate dimensions."""
        result = MultiVarDimensionsConfig.select_random_dims(100, 100, 42)
        assert len(result) == len(set(result))
    
    def test_sorted_output(self):
        """Output should be sorted."""
        result = MultiVarDimensionsConfig.select_random_dims(10, 9, 42)
        assert result == sorted(result)
    
    def test_reproducible_with_same_seed(self):
        """Same seed should give same result."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        result1 = MultiVarDimensionsConfig.select_random_dims(10, 5, rng1)
        result2 = MultiVarDimensionsConfig.select_random_dims(10, 5, rng2)
        assert result1 == result2
    
    def test_different_with_different_seed(self):
        """Different seed should give different result."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(456)
        result1 = MultiVarDimensionsConfig.select_random_dims(8, 5, rng1)
        result2 = MultiVarDimensionsConfig.select_random_dims(8, 5, rng2)
        assert result1 != result2
    
    def test_edge_case_select_one_dim(self):
        """Should work when selecting 1 dimension."""
        result = MultiVarDimensionsConfig.select_random_dims(1, 1, 42)
        assert len(result) == 1
        assert 0 <= result[0] < 5
    
    def test_edge_case_select_all_dims(self):
        """Should work when selecting all dimensions."""
        result = MultiVarDimensionsConfig.select_random_dims(5, 5, 42)
        assert result == [0, 1, 2, 3, 4]


class TestFromInt:
    """Tests for from_int method."""
    
    def test_all_dims_all_variables_identical(self):
        """All dims should give all variables identical dimension lists."""
        result = MultiVarDimensionsConfig.from_int(3, 3, 3, 42)
        assert len(result) == 3
        for dims in result:
            assert dims == [0, 1, 2]
    
    def test_subset_dims_random_selection(self):
        """Subset of dims should give random selection."""
        result = MultiVarDimensionsConfig.from_int(2, 3, 5, 42)
        assert len(result) == 3
        for dims in result:
            assert len(dims) == 2
            assert all(0 <= d < 5 for d in dims)
    
    def test_exceeds_num_dims_raises_value_error(self):
        """var_dims > num_dims should raise ValueError."""
        with pytest.raises(ValueError, match="cannot exceed"):
            MultiVarDimensionsConfig.from_int(5, 2, 3, 42)
    
    def test_single_dimension_per_variable(self):
        """Single dimension per variable should work."""
        result = MultiVarDimensionsConfig.from_int(1, 3, 5, 42)
        assert len(result) == 3
        for dims in result:
            assert len(dims) == 1
    
    def test_num_vars_copies_for_all_dims(self):
        """All dims case should create num_vars copies."""
        result = MultiVarDimensionsConfig.from_int(4, 3, 4, 42)
        assert len(result) == 3
    
    def test_length_matches_num_vars(self):
        """Result length should match num_vars."""
        for num_vars in [1, 3, 5, 10]:
            result = MultiVarDimensionsConfig.from_int(2, num_vars, 5, 42)
            assert len(result) == num_vars
    
    def test_each_element_sorted(self):
        """Each element should be sorted."""
        result = MultiVarDimensionsConfig.from_int(3, 5, 5, 42)
        for dims in result:
            assert dims == sorted(dims)


class TestFromListElement:
    """Tests for from_list_element method."""
    
    def test_integer_element_random_selection(self):
        """Integer element should trigger random selection."""
        result = MultiVarDimensionsConfig.from_list_element(2, 0, 5, 42)
        assert len(result) == 2
        assert all(0 <= d < 5 for d in result)
    
    def test_integer_element_all_dims(self):
        """Integer = num_dims should return all dims."""
        result = MultiVarDimensionsConfig.from_list_element(5, 0, 5, 42)
        assert result == [0, 1, 2, 3, 4]
    
    def test_list_element_preserved(self):
        """List element should be preserved."""
        dims_list = [0, 2, 4]
        result = MultiVarDimensionsConfig.from_list_element(dims_list, 0, 5, 42)
        assert result == dims_list
    
    def test_tuple_element_converted_to_list(self):
        """Tuple element should be converted to list."""
        dims_tuple = (1, 3)
        result = MultiVarDimensionsConfig.from_list_element(dims_tuple, 0, 5, 42)
        assert result == [1, 3]
        assert isinstance(result, list)
    
    def test_empty_list_raises_value_error(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError, match=r"Variable \d+ dim indices must have \d+ to \d+ elements, got \d+"):
            MultiVarDimensionsConfig.from_list_element([], 0, 5, 42)
    
    def test_out_of_range_raises_value_error(self):
        """Out of range dimension should raise ValueError."""
        with pytest.raises(ValueError, match=r"Variable \d+ dim indices .+ must be in range \[0, \d+\)"):
            MultiVarDimensionsConfig.from_list_element([0, 5], 0, 5, 42)
    
    def test_duplicates_raise_value_error(self):
        """Duplicate dimensions should raise ValueError."""
        with pytest.raises(ValueError, match=r"Variable \d+ dim indices .+ contain duplicates"):
            MultiVarDimensionsConfig.from_list_element([0, 1, 0], 0, 5, 42)
    
    def test_negative_index_raises_value_error(self):
        """Negative index should raise ValueError."""
        with pytest.raises(ValueError, match=r"Variable \d+ dim indices .+ must be in range \[0, \d+\)"):
            MultiVarDimensionsConfig.from_list_element([0, -1], 0, 5, 42)
    
    def test_invalid_type_raises_type_error(self):
        """Invalid type should raise TypeError."""
        with pytest.raises(TypeError, match=r"Variable \d+ var_dims must be int or list/tuple, got .+"):
            MultiVarDimensionsConfig.from_list_element("012", 0, 5, 42)
    
    def test_result_sorted(self):
        """Result should be sorted."""
        result = MultiVarDimensionsConfig.from_list_element([2, 0, 1], 0, 5, 42)
        assert result == [0, 1, 2]


class TestFromList:
    """Tests for from_list method."""
    
    def test_all_integers(self):
        """All integers should work."""
        result = MultiVarDimensionsConfig.from_list([2, 3, 2], 3, 5, 42)
        assert len(result) == 3
        for dims in result:
            assert isinstance(dims, list)
    
    def test_all_lists(self):
        """All lists should work."""
        result = MultiVarDimensionsConfig.from_list([[0, 1], [2, 3], [1, 4]], 3, 5, 42)
        assert result == [[0, 1], [2, 3], [1, 4]]
    
    def test_mixed_integers_and_lists(self):
        """Mixed integers and lists should work."""
        result = MultiVarDimensionsConfig.from_list([2, [0, 1], 3], 3, 5, 42)
        assert len(result) == 3
        assert result[1] == [0, 1]
    
    def test_wrong_length_raises_value_error(self):
        """Wrong length should raise ValueError."""
        with pytest.raises(ValueError, match=r"var_dims list must have \d+ elements, got \d+"):
            MultiVarDimensionsConfig.from_list([2, 3], 3, 3, 42)
    
    def test_length_matches_num_vars(self):
        """Result length should match num_vars."""
        result = MultiVarDimensionsConfig.from_list([2, 3, 2], 3, 5, 42)
        assert len(result) == 3


class TestComputeConstantDims:
    """Tests for compute_constant_dims method."""
    
    def test_no_varying_dims_all_constant(self):
        """No varying dims should mean all constant."""
        var_dims_indices = [[]]
        result = MultiVarDimensionsConfig.compute_constant_dims(var_dims_indices, 3)
        assert result == [[0, 1, 2]]
    
    def test_all_varying_dims_no_constant(self):
        """All varying dims should mean no constant."""
        var_dims_indices = [[0, 1, 2]]
        result = MultiVarDimensionsConfig.compute_constant_dims(var_dims_indices, 3)
        assert result == [[]]
    
    def test_mixed_correct_complement(self):
        """Mixed should compute correct complement."""
        var_dims_indices = [[0, 2], [1]]
        result = MultiVarDimensionsConfig.compute_constant_dims(var_dims_indices, 3)
        assert result == [[1], [0, 2]]
    
    def test_single_dimension_varying(self):
        """Single dimension varying should work."""
        var_dims_indices = [[0]]
        result = MultiVarDimensionsConfig.compute_constant_dims(var_dims_indices, 3)
        assert result == [[1, 2]]
    
    def test_length_matches_num_vars(self):
        """Result length should match input length."""
        var_dims_indices = [[0], [1], [2]]
        result = MultiVarDimensionsConfig.compute_constant_dims(var_dims_indices, 3)
        assert len(result) == 3


class TestPreselectConstantCoordIndices:
    """Tests for preselect_constant_coord_indices method."""
    
    def test_creates_rng_for_each_constant_dim(self):
        """Should create RNG for each constant dim."""
        var_constant_dims = [[1, 2], [0]]
        shape = (10, 10, 30)
        result = MultiVarDimensionsConfig.preselect_constant_coord_indices(
            var_constant_dims, shape, 42
        )
        assert 0 in result
        assert 1 in result[0]
        assert 2 in result[0]
        assert 0 in result[1]
    
    def test_empty_dict_for_no_constant_dims(self):
        """No constant dims should give empty inner dicts."""
        var_constant_dims = [[], []]
        shape = (10, 10, 30)
        result = MultiVarDimensionsConfig.preselect_constant_coord_indices(
            var_constant_dims, shape, 42
        )
        assert result[0] == {}
        assert result[1] == {}
    
    def test_correct_structure_returned(self):
        """Should return correct nested dict structure."""
        var_constant_dims = [[1]]
        shape = (10, 10, 30)
        result = MultiVarDimensionsConfig.preselect_constant_coord_indices(
            var_constant_dims, shape, 42
        )
        assert isinstance(result, dict)
        assert isinstance(result[0], dict)
        assert isinstance(result[0][1], np.random.Generator)
