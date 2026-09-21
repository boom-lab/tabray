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
            var_constant_coord_indices, 42, dim_split=0
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
        # both counts must be >= max(shape), otherwise no placement can use
        # every coordinate of the longest axis and generation is refused
        var_num_obs = np.array([20, 25])
        var_constant_dims = [[], []]
        var_constant_coord_indices = {0: {}, 1: {}}

        result = MultiVarRecordGenerator.generate_without_overlap(
            shape, records, 2, var_num_obs, var_constant_dims,
            var_constant_coord_indices, 42, dim_split=0
        )

        count0 = np.count_nonzero(~np.isnan(result['var0']))
        count1 = np.count_nonzero(~np.isnan(result['var1']))

        assert count0 == 20
        assert count1 == 25
    
    def test_all_records_have_correct_shape(self):
        """Should maintain correct shape for all records."""
        shape = [8, 12]
        records = {
            'var0': np.full(shape, np.nan),
            'var1': np.full(shape, np.nan),
            'var2': np.full(shape, np.nan)
        }
        # >= max(shape) = 12, so every coordinate of every axis can be used
        var_num_obs = np.array([12, 12, 12])
        var_constant_dims = [[], [], []]
        var_constant_coord_indices = {0: {}, 1: {}, 2: {}}
        
        result = MultiVarRecordGenerator.generate_without_overlap(
            shape, records, 3, var_num_obs, var_constant_dims,
            var_constant_coord_indices, 42, dim_split=0
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
            var_constant_coord_indices, 42, dim_split=0
        )
        
        records2 = {
            'var0': np.full(shape, np.nan),
            'var1': np.full(shape, np.nan)
        }
        result2 = MultiVarRecordGenerator.generate_without_overlap(
            shape, records2, 2, var_num_obs, var_constant_dims,
            var_constant_coord_indices, 42, dim_split=0
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
            var_constant_dims, var_constant_coord_indices, num_dims, 42,
            dim_split=0
        )
        
        assert isinstance(records, dict)
        assert 'var0' in records
        # one variable has nothing to overlap with: an empty per-variable array,
        # matching the multi-variable return type
        assert isinstance(overlap_actual, np.ndarray)
        assert overlap_actual.size == 0
    
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
            var_constant_dims, var_constant_coord_indices, num_dims, 42,
            dim_split=0
        )
        
        # Should complete without error
        assert isinstance(records, dict)
        assert 'var0' in records
        assert 'var1' in records
    
    def test_numeric_overlap_uses_the_stratified_path(self):
        """A numeric target goes through the stratified placement.

        num_obs must reach max(shape): the placement refuses a count that
        cannot put every coordinate of the longest axis to use.
        """
        shape = [15, 15]
        records = {
            'var0': np.full(shape, np.nan),
            'var1': np.full(shape, np.nan)
        }
        var_num_obs = np.array([15, 15])
        var_dims_indices = [ [0, 1], [0, 1] ]
        var_constant_dims = [[], []]
        var_constant_coord_indices = {0: {}, 1: {}}
        num_vars = 2
        num_dims = 2
        overlap = 0.5
        
        records, overlap_actual = MultiVarRecordGenerator.generate(
            shape, overlap, num_vars, var_num_obs, var_dims_indices,
            var_constant_dims, var_constant_coord_indices, num_dims, 42,
            dim_split=0
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
            seed=42,
            dim_split=0
        )

        assert isinstance(overlap_actual, np.ndarray)
        assert overlap_actual.shape == (2,)
        assert overlap_actual[0] == pytest.approx(0.8)
        assert overlap_actual[1] == pytest.approx(0.5)
        for var_idx in range(3):
            count = np.count_nonzero(~np.isnan(records[f'var{var_idx}']))
            assert count == var_num_obs[var_idx]

    def test_fixed_overlap_shares_prefix_across_true_flags(self):
        """Variables opting into fixed_overlap draw from one shared ordering.

        var2 asks for half of var1's overlap, so with a shared ordering its
        sites are a prefix of var1's and therefore a subset.

        Exercises the stratified path (dim_split given), which is what
        GenerateData uses. The previous version called the pre-stratified path,
        where the non-overlapping fill can land on var0 by accident (A3) and
        inflate both sets -- the subset property held there for one lucky seed
        and fails for 30 others.
        """
        shape = [20, 20]
        records, _ = MultiVarRecordGenerator.generate(
            shape=shape,
            overlap=[0.5, 0.25],
            fixed_overlap=[True, True],
            num_vars=3,
            var_num_obs=np.array([80, 40, 40]),
            var_dims_indices=[[0, 1]] * 3,
            var_constant_dims=[[], [], []],
            var_constant_coord_indices={0: {}, 1: {}, 2: {}},
            num_dims=2,
            seed=123,
            dim_split=1,
        )

        ref_coords = OverlapCalculator.extract_coordinate_set(records["var0"], 2)
        overlap1 = ref_coords & OverlapCalculator.extract_coordinate_set(records["var1"], 2)
        overlap2 = ref_coords & OverlapCalculator.extract_coordinate_set(records["var2"], 2)

        assert len(overlap2) < len(overlap1), "var2 asks for less, so this is not trivial"
        assert overlap2.issubset(overlap1)

    def test_independent_overlap_when_flags_are_false(self):
        """Without fixed_overlap the variables draw independently.

        The counterpart to the test above: the same configuration with the
        flags off gives var2 sites that are not a subset of var1's.
        """
        shape = [20, 20]
        records, _ = MultiVarRecordGenerator.generate(
            shape=shape,
            overlap=[0.5, 0.25],
            fixed_overlap=[False, False],
            num_vars=3,
            var_num_obs=np.array([80, 40, 40]),
            var_dims_indices=[[0, 1]] * 3,
            var_constant_dims=[[], [], []],
            var_constant_coord_indices={0: {}, 1: {}, 2: {}},
            num_dims=2,
            seed=123,
            dim_split=1,
        )

        ref_coords = OverlapCalculator.extract_coordinate_set(records["var0"], 2)
        overlap1 = ref_coords & OverlapCalculator.extract_coordinate_set(records["var1"], 2)
        overlap2 = ref_coords & OverlapCalculator.extract_coordinate_set(records["var2"], 2)

        assert not overlap2.issubset(overlap1)


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
            var_constant_dims, var_constant_coord_indices, num_dims, 42,
            dim_split=0
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
            seed=42,
            dim_split=0
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
            seed=123,
            dim_split=0
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
            num_dims=2, seed=42, dim_split=0
        )
        
        records2, _ = MultiVarRecordGenerator.generate(
            shape=shape, overlap='random', num_vars=1,
            var_num_obs=np.array([num_obs]),
            var_dims_indices=[list(range(2))],
            var_constant_dims=[[]],
            var_constant_coord_indices={},
            num_dims=2, seed=42, dim_split=0
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
            num_dims=2, seed=999, dim_split=0
        )
        
        records2, overlap2 = MultiVarRecordGenerator.generate(
            shape=shape, overlap=0.5, num_vars=1,
            var_num_obs=np.array([num_obs]),
            var_dims_indices=[list(range(2))],
            var_constant_dims=[[]],
            var_constant_coord_indices={},
            num_dims=2, seed=999, dim_split=0
        )
        
        # Both should use 'random' path due to num_vars==1
        np.testing.assert_array_equal(records1['var0'], records2['var0'])


class TestOverlapIsApportionedAcrossStrata:
    """The overlap total is decided once, not rounded in every stratum.

    Rounding target * p_j per stratum and summing is not the same as rounding
    the total. Each stratum rounds to a whole cell, and on a coarse grid one
    cell is a large share of the variable: a 3x3 grid asking for 7 of 8 shared
    cells used to get 8, and one asking for 1 of 3 used to get 0.
    """

    @staticmethod
    def achieved(shape, obs, target, seed=12345, num_vars=2, var_dims=None,
                 split_dim=0):
        """Generate, then measure F1 from the occupancy masks."""
        num_dims = len(shape)
        dims = var_dims or [list(range(num_dims))] * num_vars
        records = MultiVarRecordGenerator.generate_multivar_stratified(
            global_shape=shape,
            num_vars=num_vars,
            var_num_obs=np.array([obs] * num_vars),
            var_dims_indices=dims,
            var_constant_coords={v: {} for v in range(num_vars)},
            overlap_targets=[target] * (num_vars - 1),
            fixed_overlap_flags=[False] * (num_vars - 1),
            seed=seed,
            split_dim=split_dim,
        )
        sites = []
        for var in range(num_vars):
            indices, _ = records[f"var{var}"]
            sites.append(set(zip(*[np.asarray(a) for a in indices])))
        return len(sites[0] & sites[1]) / len(sites[0]), len(sites[0])

    def test_the_case_that_rendered_as_complete_overlap(self):
        """3x3, 8 observations, target 7/8. Per-stratum rounding gave 8 of 8."""
        f1, reference = self.achieved([3, 3], 8, 7 / 8)
        assert reference == 8
        assert f1 == pytest.approx(7 / 8)

    def test_the_case_that_rendered_as_no_overlap(self):
        """3x3, 3 observations, target 1/3. Each stratum rounded 0.33 to 0."""
        f1, reference = self.achieved([3, 3], 3, 1 / 3)
        assert reference == 3
        assert f1 == pytest.approx(1 / 3)

    @pytest.mark.parametrize("side,obs", [(3, 8), (6, 32), (9, 72), (20, 350)])
    def test_target_is_met_at_every_grid_size(self, side, obs):
        """Within one cell, which is the resolution F1 has on that grid.

        0.875 of 350 reference cells is 306.25, so 306/350 is as close as the
        grid allows. What must not happen is the drift to an endpoint.
        """
        f1, reference = self.achieved([side, side], obs, 7 / 8)
        assert abs(f1 - 7 / 8) <= 1 / reference

    def test_unreachable_target_lands_on_the_nearest_whole_cell(self):
        """0.75 of 6 cells is 4.5. Neither 4 nor 5 is the target; the result
        must be one of them rather than an endpoint."""
        f1, reference = self.achieved([3, 3], 6, 0.75)
        assert reference == 6
        assert f1 in (pytest.approx(4 / 6), pytest.approx(5 / 6))

    def test_reduced_dimension_variables_keep_the_per_stratum_rule(self):
        """p_j counts distinct PROJECTED cells there, which depends on where
        the reference landed -- a worker cannot know it for strata it does not
        own. Falling back keeps serial and parallel identical."""
        records = MultiVarRecordGenerator.generate_multivar_stratified(
            global_shape=[3, 3], num_vars=2, var_num_obs=np.array([8, 3]),
            var_dims_indices=[[0, 1], [1]],
            var_constant_coords={0: {}, 1: {0: 1}},   # var1 sits on x0 index 1
            overlap_targets=[1 / 8], fixed_overlap_flags=[False],
            seed=12345, split_dim=0)
        indices, _ = records["var1"]
        assert len(indices[0]) == 3                  # it ran, and placed its own
        assert set(np.asarray(indices[0])) == {1}    # all in its one stratum

    def test_serial_and_parallel_agree(self):
        """The counts come from globally known quantities, so a worker holding
        two strata derives the same numbers as a serial run."""
        shape, obs = [6, 6], 32
        whole = MultiVarRecordGenerator.generate_multivar_stratified(
            global_shape=shape, num_vars=2, var_num_obs=np.array([obs, obs]),
            var_dims_indices=[[0, 1], [0, 1]],
            var_constant_coords={0: {}, 1: {}}, overlap_targets=[7 / 8],
            fixed_overlap_flags=[False], seed=7, split_dim=0)
        pieces = [
            MultiVarRecordGenerator.generate_multivar_stratified(
                global_shape=shape, num_vars=2, var_num_obs=np.array([obs, obs]),
                var_dims_indices=[[0, 1], [0, 1]],
                var_constant_coords={0: {}, 1: {}}, overlap_targets=[7 / 8],
                fixed_overlap_flags=[False], seed=7, split_dim=0, strata=chunk)
            for chunk in ([0, 1, 2], [3, 4, 5])
        ]
        for var in ("var0", "var1"):
            full = set(zip(*[np.asarray(a) for a in whole[var][0]]))
            split = set()
            for piece in pieces:
                split |= set(zip(*[np.asarray(a) for a in piece[var][0]]))
            assert full == split


class TestStratumCounts:
    """The helper a worker uses to learn counts for strata it does not hold."""

    @pytest.mark.parametrize("shape,obs", [
        ([3, 3], 8), ([3, 3], 3), ([6, 6], 32), ([30, 30], 800), ([100, 40], 1500),
    ])
    def test_matches_what_placement_produces(self, shape, obs):
        """If these drift apart, the apportioned overlap counts are wrong."""
        from data_sparsity.generators.record_generator import RecordGenerator

        counts = RecordGenerator.stratum_counts(shape, obs, 12345, 0)
        indices, _ = RecordGenerator.generate_stratified_indices(
            global_shape=shape, num_obs=obs, seed=12345, split_dim=0)
        placed = np.bincount(np.asarray(indices[0]), minlength=shape[0])
        assert np.array_equal(counts, placed)
        assert counts.sum() == obs
