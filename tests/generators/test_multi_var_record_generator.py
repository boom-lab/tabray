"""Tests for MultiVarRecordGenerator class."""

import pytest
import numpy as np
from data_sparsity.generators.multi_var_record_generator import MultiVarRecordGenerator
from data_sparsity.generators.overlap_calculator import OverlapCalculator


class TestComputeVarShapes:
    """Tests for _compute_var_shapes method."""
    
    def test_no_constant_dims_same_as_full_shape(self):
        """Should return full shape when no constant dims."""
        shape = [10, 20, 30]
        var_constant_dims = [[], []]
        result = MultiVarRecordGenerator._compute_var_shapes(shape, var_constant_dims, 2)
        
        assert result[0] == [10, 20, 30]
        assert result[1] == [10, 20, 30]
    
    def test_with_constant_dims_size_1_for_constant(self):
        """Should set size 1 for constant dimensions."""
        shape = [10, 20, 30]
        var_constant_dims = [[0, 2], [1]]
        result = MultiVarRecordGenerator._compute_var_shapes(shape, var_constant_dims, 2)
        
        assert result[0] == [1, 20, 1]  # Dims 0 and 2 are constant
        assert result[1] == [10, 1, 30]  # Dim 1 is constant
    
    def test_multiple_variables(self):
        """Should handle multiple variables."""
        shape = [5, 5, 5]
        var_constant_dims = [[0], [1], [2]]
        result = MultiVarRecordGenerator._compute_var_shapes(shape, var_constant_dims, 3)
        
        assert len(result) == 3
        assert result[0] == [1, 5, 5]
        assert result[1] == [5, 1, 5]
        assert result[2] == [5, 5, 1]
    
    def test_all_constant_dims(self):
        """Should handle all dimensions constant."""
        shape = [10, 20]
        var_constant_dims = [[0, 1]]
        result = MultiVarRecordGenerator._compute_var_shapes(shape, var_constant_dims, 1)
        
        assert result[0] == [1, 1]
    
    def test_dictionary_keys_correct(self):
        """Should return dictionary with correct keys."""
        shape = [10, 20]
        var_constant_dims = [[], [], []]
        result = MultiVarRecordGenerator._compute_var_shapes(shape, var_constant_dims, 3)
        
        assert 0 in result
        assert 1 in result
        assert 2 in result
        assert len(result) == 3


class TestSelectConstantCoords:
    """Tests for _select_constant_coords method."""
    
    def test_selects_valid_coordinate_indices(self):
        """Should select indices within valid range."""
        shape = [10, 20, 30]
        var_constant_dims = [[0], [1]]
        
        # Create RNGs
        var_constant_coord_indices = {
            0: {0: np.random.default_rng(42)},
            1: {1: np.random.default_rng(43)}
        }
        
        result = MultiVarRecordGenerator._select_constant_coords(
            shape, var_constant_dims, var_constant_coord_indices, 2
        )
        
        assert 0 <= result[0][0] < 10
        assert 0 <= result[1][1] < 20
    
    def test_uses_preseeded_rngs(self):
        """Should use provided RNGs for selection."""
        shape = [100, 100]
        var_constant_dims = [[0]]
        
        var_constant_coord_indices = {
            0: {0: np.random.default_rng(999)}
        }
        
        result1 = MultiVarRecordGenerator._select_constant_coords(
            shape, var_constant_dims, var_constant_coord_indices, 1
        )
        
        # Reset same seed
        var_constant_coord_indices = {
            0: {0: np.random.default_rng(999)}
        }
        
        result2 = MultiVarRecordGenerator._select_constant_coords(
            shape, var_constant_dims, var_constant_coord_indices, 1
        )
        
        assert result1[0][0] == result2[0][0]
    
    def test_no_constant_dims_empty_dict(self):
        """Should return empty dict for no constant dims."""
        shape = [10, 20]
        var_constant_dims = [[]]
        var_constant_coord_indices = {0: {}}
        
        result = MultiVarRecordGenerator._select_constant_coords(
            shape, var_constant_dims, var_constant_coord_indices, 1
        )
        
        assert result[0] == {}
    
    def test_multiple_constant_dims_per_var(self):
        """Should handle multiple constant dims per variable."""
        shape = [10, 20, 30]
        var_constant_dims = [[0, 2]]
        
        var_constant_coord_indices = {
            0: {
                0: np.random.default_rng(42),
                2: np.random.default_rng(43)
            }
        }
        
        result = MultiVarRecordGenerator._select_constant_coords(
            shape, var_constant_dims, var_constant_coord_indices, 1
        )
        
        assert 0 in result[0]
        assert 2 in result[0]
        assert 0 <= result[0][0] < 10
        assert 0 <= result[0][2] < 30
    
    def test_different_values_per_variable(self):
        """Should select potentially different values for each variable."""
        shape = [50, 50]
        var_constant_dims = [[0], [0]]
        
        var_constant_coord_indices = {
            0: {0: np.random.default_rng(42)},
            1: {0: np.random.default_rng(999)}
        }
        
        result = MultiVarRecordGenerator._select_constant_coords(
            shape, var_constant_dims, var_constant_coord_indices, 2
        )
        
        # Values might be different (different seeds)
        assert 0 in result[0]
        assert 0 in result[1]


class TestExpandToFullCoords:
    """Tests for _expand_to_full_coords method."""
    
    def test_constant_dims_filled_with_constant_value(self):
        """Should fill constant dimensions with fixed values."""
        multi_indices = (np.array([0, 1, 2]), np.array([3, 4, 5]))
        var_constant_coords = {0: 7}
        num_obs = 3
        
        result = MultiVarRecordGenerator._expand_to_full_coords(
            multi_indices, var_constant_coords, num_obs
        )
        
        np.testing.assert_array_equal(result[0], [7, 7, 7])
        np.testing.assert_array_equal(result[1], [3, 4, 5])
    
    def test_varying_dims_unchanged(self):
        """Should keep varying dimensions unchanged."""
        multi_indices = (np.array([1, 2, 3]), np.array([4, 5, 6]))
        var_constant_coords = {}
        num_obs = 3
        
        result = MultiVarRecordGenerator._expand_to_full_coords(
            multi_indices, var_constant_coords, num_obs
        )
        
        np.testing.assert_array_equal(result[0], [1, 2, 3])
        np.testing.assert_array_equal(result[1], [4, 5, 6])
    
    def test_correct_tuple_length(self):
        """Should return tuple with correct length."""
        multi_indices = (np.array([0]), np.array([1]), np.array([2]))
        var_constant_coords = {1: 5}
        num_obs = 1
        
        result = MultiVarRecordGenerator._expand_to_full_coords(
            multi_indices, var_constant_coords, num_obs
        )
        
        assert len(result) == 3
    
    def test_correct_array_lengths(self):
        """Should return arrays with correct lengths."""
        num_obs = 5
        multi_indices = (
            np.array([0, 1, 2, 3, 4]),
            np.array([5, 6, 7, 8, 9])
        )
        var_constant_coords = {0: 10}
        
        result = MultiVarRecordGenerator._expand_to_full_coords(
            multi_indices, var_constant_coords, num_obs
        )
        
        assert len(result[0]) == num_obs
        assert len(result[1]) == num_obs
    
    def test_multiple_constant_dims(self):
        """Should handle multiple constant dimensions."""
        multi_indices = (
            np.array([0, 1]),
            np.array([2, 3]),
            np.array([4, 5])
        )
        var_constant_coords = {0: 10, 2: 20}
        num_obs = 2
        
        result = MultiVarRecordGenerator._expand_to_full_coords(
            multi_indices, var_constant_coords, num_obs
        )
        
        np.testing.assert_array_equal(result[0], [10, 10])
        np.testing.assert_array_equal(result[1], [2, 3])
        np.testing.assert_array_equal(result[2], [20, 20])


class TestGenerateWithoutOverlap:
    """Tests for generate_without_overlap method."""
    
    def test_two_variables(self, fixed_rng):
        """Should generate records for two variables."""
        shape = [10, 10]
        records = {
            'var0': np.full(shape, np.nan),
            'var1': np.full(shape, np.nan)
        }
        var_num_obs = np.array([10, 10])
        var_constant_dims = [[], []]
        var_constant_coord_indices = {0: {}, 1: {}}
        
        result = MultiVarRecordGenerator.generate_without_overlap(
            shape, records, 2, var_num_obs, var_constant_dims,
            var_constant_coord_indices, 42
        )
        
        assert 'var0' in result
        assert 'var1' in result
    
    def test_different_observation_counts(self, fixed_rng):
        """Should respect different observation counts."""
        shape = [20, 20]
        records = {
            'var0': np.full(shape, np.nan),
            'var1': np.full(shape, np.nan)
        }
        var_num_obs = np.array([15, 25])
        var_constant_dims = [[], []]
        var_constant_coord_indices = {0: {}, 1: {}}
        
        result = MultiVarRecordGenerator.generate_without_overlap(
            shape, records, 2, var_num_obs, var_constant_dims,
            var_constant_coord_indices, 42
        )
        
        count0 = np.count_nonzero(~np.isnan(result['var0']))
        count1 = np.count_nonzero(~np.isnan(result['var1']))
        
        assert count0 == 15
        assert count1 == 25
    
    def test_all_records_have_correct_shape(self):
        """Should maintain correct shape for all records."""
        shape = [8, 12]
        records = {
            'var0': np.full(shape, np.nan),
            'var1': np.full(shape, np.nan),
            'var2': np.full(shape, np.nan)
        }
        var_num_obs = np.array([5, 5, 5])
        var_constant_dims = [[], [], []]
        var_constant_coord_indices = {0: {}, 1: {}, 2: {}}
        
        result = MultiVarRecordGenerator.generate_without_overlap(
            shape, records, 3, var_num_obs, var_constant_dims,
            var_constant_coord_indices, 42
        )
        
        for var_idx in range(3):
            assert result[f'var{var_idx}'].shape == tuple(shape)
    
    def test_reproducible_with_seed(self):
        """Should produce same results with same seed."""
        shape = [10, 10]
        var_num_obs = np.array([10, 10])
        var_constant_dims = [[], []]
        var_constant_coord_indices = {0: {}, 1: {}}
        
        records1 = {
            'var0': np.full(shape, np.nan),
            'var1': np.full(shape, np.nan)
        }
        result1 = MultiVarRecordGenerator.generate_without_overlap(
            shape, records1, 2, var_num_obs, var_constant_dims,
            var_constant_coord_indices, 42
        )
        
        records2 = {
            'var0': np.full(shape, np.nan),
            'var1': np.full(shape, np.nan)
        }
        result2 = MultiVarRecordGenerator.generate_without_overlap(
            shape, records2, 2, var_num_obs, var_constant_dims,
            var_constant_coord_indices, 42
        )
        
        np.testing.assert_array_equal(
            result1['var0'], result2['var0']
        )
        np.testing.assert_array_equal(
            result1['var1'], result2['var1']
        )


class TestGenerate:
    """Tests for main generate method."""
    
    def test_returns_dictionary(self):
        """Should return dictionary of records."""
        shape = [10, 10]
        records = {'var0': np.full(shape, np.nan)}
        var_num_obs = np.array([10])
        var_dims_indices = [ [0,1] ]
        var_constant_dims = [[]]
        var_constant_coord_indices = {0: {}}
        num_vars = 1
        num_dims = 2
        overlap = 'random'
        
        records, overlap_actual = MultiVarRecordGenerator.generate(
            shape, overlap, num_vars, var_num_obs, var_dims_indices,
            var_constant_dims, var_constant_coord_indices, num_dims, 42
        )
        
        assert isinstance(records, dict)
        assert 'var0' in records
        assert overlap_actual is None
    
    def test_random_overlap_calls_without_overlap(self):
        """Should use without_overlap for random overlap."""
        shape = [10, 10]
        records = {
            'var0': np.full(shape, np.nan),
            'var1': np.full(shape, np.nan)
        }
        var_num_obs = np.array([10, 10])
        var_dims_indices = [ [0,1], [0,1] ]
        var_constant_dims = [[], []]
        var_constant_coord_indices = {0: {}, 1: {}}
        num_vars = 2
        num_dims = 2
        overlap = 'random'
        
        records, overlap_actual = MultiVarRecordGenerator.generate(
            shape, overlap, num_vars, var_num_obs, var_dims_indices,
            var_constant_dims, var_constant_coord_indices, num_dims, 42
        )
        
        # Should complete without error
        assert isinstance(records, dict)
        assert 'var0' in records
        assert 'var1' in records
    
    def test_numeric_overlap_calls_with_overlap(self):
        """Should use with_overlap for numeric overlap."""
        shape = [15, 15]
        records = {
            'var0': np.full(shape, np.nan),
            'var1': np.full(shape, np.nan)
        }
        var_num_obs = np.array([10, 10])
        var_dims_indices = [ [1], [1] ]
        var_constant_dims = [[0], [0]]
        var_constant_coord_indices = {
            0: {0: np.random.default_rng(42)},
            1: {0: np.random.default_rng(43)}
        }
        num_vars = 2
        num_dims = 2
        overlap = 0.5
        
        records, overlap_actual = MultiVarRecordGenerator.generate(
            shape, overlap, num_vars, var_num_obs, var_dims_indices,
            var_constant_dims, var_constant_coord_indices, num_dims, 42
        )
        
        # Should complete without error
        assert isinstance(records, dict)
        assert 'var0' in records
        assert 'var1' in records

    def test_per_variable_overlap_targets(self):
        """Should support distinct overlap targets for each non-reference variable."""
        shape = [10, 10]
        var_num_obs = np.array([10, 10, 10])
        var_dims_indices = [[0, 1], [0, 1], [0, 1]]
        var_constant_dims = [[], [], []]
        var_constant_coord_indices = {0: {}, 1: {}, 2: {}}

        records, overlap_actual = MultiVarRecordGenerator.generate(
            shape=shape,
            overlap=[0.8, 0.5],
            num_vars=3,
            var_num_obs=var_num_obs,
            var_dims_indices=var_dims_indices,
            var_constant_dims=var_constant_dims,
            var_constant_coord_indices=var_constant_coord_indices,
            num_dims=2,
            seed=42
        )

        assert isinstance(overlap_actual, np.ndarray)
        assert overlap_actual.shape == (2,)
        assert overlap_actual[0] == pytest.approx(0.8)
        assert overlap_actual[1] == pytest.approx(0.5)
        for var_idx in range(3):
            count = np.count_nonzero(~np.isnan(records[f'var{var_idx}']))
            assert count == var_num_obs[var_idx]

    def test_fixed_overlap_shares_prefix_across_true_flags(self):
        """Should share the same overlap prefix for fixed-overlap variables."""
        shape = [10, 10]
        var_num_obs = np.array([12, 10, 10])
        var_dims_indices = [[0, 1], [0, 1], [0, 1]]
        var_constant_dims = [[], [], []]
        var_constant_coord_indices = {0: {}, 1: {}, 2: {}}

        records, _ = MultiVarRecordGenerator.generate(
            shape=shape,
            overlap=[0.8, 0.4],
            fixed_overlap=[True, True],
            num_vars=3,
            var_num_obs=var_num_obs,
            var_dims_indices=var_dims_indices,
            var_constant_dims=var_constant_dims,
            var_constant_coord_indices=var_constant_coord_indices,
            num_dims=2,
            seed=123
        )

        ref_coords = OverlapCalculator.extract_coordinate_set(records["var0"], 2)
        var1_coords = OverlapCalculator.extract_coordinate_set(records["var1"], 2)
        var2_coords = OverlapCalculator.extract_coordinate_set(records["var2"], 2)

        overlap1 = ref_coords.intersection(var1_coords)
        overlap2 = ref_coords.intersection(var2_coords)

        assert overlap2.issubset(overlap1)
    
    def test_single_variable_edge_case(self):
        """Should handle single variable."""
        shape = [10, 10]
        records = {'var0': np.full(shape, np.nan)}
        var_num_obs = np.array([15])
        var_dims_indices = [ [0,1] ]
        var_constant_dims = [[]]
        var_constant_coord_indices = {0: {}}
        num_vars = 1
        num_dims = 2
        overlap = 0.5
        
        records, overlap_actual = MultiVarRecordGenerator.generate(
            shape, overlap, num_vars, var_num_obs, var_dims_indices,
            var_constant_dims, var_constant_coord_indices, num_dims, 42
        )
        
        assert 'var0' in records
        count = np.count_nonzero(~np.isnan(records['var0']))
        assert count == 15


class TestSingleVariableCase:
    """Tests for single-variable generation (num_vars=1) to ensure compatibility."""
    
    def test_single_variable_basic(self):
        """Should generate single variable correctly."""
        shape = [10, 10]
        num_obs = 20
        
        records, overlap = MultiVarRecordGenerator.generate(
            shape=shape,
            overlap='random',
            num_vars=1,
            var_num_obs=np.array([num_obs]),
            var_dims_indices=[list(range(2))],
            var_constant_dims=[[]],
            var_constant_coord_indices={},
            num_dims=2,
            seed=42
        )
        
        assert len(records) == 1
        assert 'var0' in records
        assert records['var0'].shape == tuple(shape)
        assert np.sum(~np.isnan(records['var0'])) == num_obs
    
    def test_single_variable_all_dims(self):
        """Should handle single variable in 3D space."""
        shape = [5, 6, 7]
        num_obs = 42
        
        records, overlap = MultiVarRecordGenerator.generate(
            shape=shape,
            overlap='random',
            num_vars=1,
            var_num_obs=np.array([num_obs]),
            var_dims_indices=[list(range(3))],
            var_constant_dims=[[]],
            var_constant_coord_indices={},
            num_dims=3,
            seed=123
        )
        
        assert records['var0'].shape == tuple(shape)
        assert np.sum(~np.isnan(records['var0'])) == num_obs
        
        # Verify observations are in valid range
        obs_values = records['var0'][~np.isnan(records['var0'])]
        assert np.all(obs_values >= 0)
        assert np.all(obs_values <= 1)
    
    def test_single_variable_reproducible(self):
        """Should produce same results with same seed."""
        shape = [8, 8]
        num_obs = 16
        
        records1, _ = MultiVarRecordGenerator.generate(
            shape=shape, overlap='random', num_vars=1,
            var_num_obs=np.array([num_obs]),
            var_dims_indices=[list(range(2))],
            var_constant_dims=[[]],
            var_constant_coord_indices={},
            num_dims=2, seed=42
        )
        
        records2, _ = MultiVarRecordGenerator.generate(
            shape=shape, overlap='random', num_vars=1,
            var_num_obs=np.array([num_obs]),
            var_dims_indices=[list(range(2))],
            var_constant_dims=[[]],
            var_constant_coord_indices={},
            num_dims=2, seed=42
        )
        
        np.testing.assert_array_equal(records1['var0'], records2['var0'])
    
    def test_single_variable_overlap_ignored(self):
        """Should ignore overlap parameter for single variable."""
        shape = [10, 10]
        num_obs = 30
        
        # Try with different overlap values - should behave identically
        records1, overlap1 = MultiVarRecordGenerator.generate(
            shape=shape, overlap='random', num_vars=1,
            var_num_obs=np.array([num_obs]),
            var_dims_indices=[list(range(2))],
            var_constant_dims=[[]],
            var_constant_coord_indices={},
            num_dims=2, seed=999
        )
        
        records2, overlap2 = MultiVarRecordGenerator.generate(
            shape=shape, overlap=0.5, num_vars=1,
            var_num_obs=np.array([num_obs]),
            var_dims_indices=[list(range(2))],
            var_constant_dims=[[]],
            var_constant_coord_indices={},
            num_dims=2, seed=999
        )
        
        # Both should use 'random' path due to num_vars==1
        np.testing.assert_array_equal(records1['var0'], records2['var0'])
