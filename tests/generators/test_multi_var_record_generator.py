"""Tests for MultiVarRecordGenerator class."""

import pytest
import numpy as np
from data_sparsity.generators.multi_var_record_generator import MultiVarRecordGenerator
from data_sparsity.generators.overlap_calculator import OverlapCalculator


class TestSelectConstantCoords:
    """Tests for _select_constant_coords method."""

    def test_selects_valid_coordinate_indices(self):
        """Should select indices within valid range."""
        shape = [10, 20, 30]
        var_constant_dims = [[0], [1]]

        # Create RNGs
        var_constant_coord_indices = {
            0: {0: np.random.default_rng(42)},
            1: {1: np.random.default_rng(43)},
        }

        result = MultiVarRecordGenerator._select_constant_coords(
            shape,
            var_constant_dims,
            var_constant_coord_indices,
            2,
        )

        assert 0 <= result[0][0] < 10
        assert 0 <= result[1][1] < 20

    def test_uses_preseeded_rngs(self):
        """Should use provided RNGs for selection."""
        shape = [100, 100]
        var_constant_dims = [[0]]

        var_constant_coord_indices = {0: {0: np.random.default_rng(999)}}

        result1 = MultiVarRecordGenerator._select_constant_coords(
            shape,
            var_constant_dims,
            var_constant_coord_indices,
            1,
        )

        # Reset same seed
        var_constant_coord_indices = {0: {0: np.random.default_rng(999)}}

        result2 = MultiVarRecordGenerator._select_constant_coords(
            shape,
            var_constant_dims,
            var_constant_coord_indices,
            1,
        )

        assert result1[0][0] == result2[0][0]

    def test_no_constant_dims_empty_dict(self):
        """Should return empty dict for no constant dims."""
        shape = [10, 20]
        var_constant_dims = [[]]
        var_constant_coord_indices = {0: {}}

        result = MultiVarRecordGenerator._select_constant_coords(
            shape,
            var_constant_dims,
            var_constant_coord_indices,
            1,
        )

        assert result[0] == {}

    def test_multiple_constant_dims_per_var(self):
        """Should handle multiple constant dims per variable."""
        shape = [10, 20, 30]
        var_constant_dims = [[0, 2]]

        var_constant_coord_indices = {
            0: {
                0: np.random.default_rng(42),
                2: np.random.default_rng(43),
            }
        }

        result = MultiVarRecordGenerator._select_constant_coords(
            shape,
            var_constant_dims,
            var_constant_coord_indices,
            1,
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
            1: {0: np.random.default_rng(999)},
        }

        result = MultiVarRecordGenerator._select_constant_coords(
            shape,
            var_constant_dims,
            var_constant_coord_indices,
            2,
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
            multi_indices,
            var_constant_coords,
            num_obs,
        )

        np.testing.assert_array_equal(result[0], [7, 7, 7])
        np.testing.assert_array_equal(result[1], [3, 4, 5])

    def test_varying_dims_unchanged(self):
        """Should keep varying dimensions unchanged."""
        multi_indices = (np.array([1, 2, 3]), np.array([4, 5, 6]))
        var_constant_coords = {}
        num_obs = 3

        result = MultiVarRecordGenerator._expand_to_full_coords(
            multi_indices,
            var_constant_coords,
            num_obs,
        )

        np.testing.assert_array_equal(result[0], [1, 2, 3])
        np.testing.assert_array_equal(result[1], [4, 5, 6])

    def test_correct_tuple_length(self):
        """Should return tuple with correct length."""
        multi_indices = (np.array([0]), np.array([1]), np.array([2]))
        var_constant_coords = {1: 5}
        num_obs = 1

        result = MultiVarRecordGenerator._expand_to_full_coords(
            multi_indices,
            var_constant_coords,
            num_obs,
        )

        assert len(result) == 3

    def test_correct_array_lengths(self):
        """Should return arrays with correct lengths."""
        num_obs = 5
        multi_indices = (np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7, 8, 9]))
        var_constant_coords = {0: 10}

        result = MultiVarRecordGenerator._expand_to_full_coords(
            multi_indices,
            var_constant_coords,
            num_obs,
        )

        assert len(result[0]) == num_obs
        assert len(result[1]) == num_obs

    def test_multiple_constant_dims(self):
        """Should handle multiple constant dimensions."""
        multi_indices = (np.array([0, 1]), np.array([2, 3]), np.array([4, 5]))
        var_constant_coords = {0: 10, 2: 20}
        num_obs = 2

        result = MultiVarRecordGenerator._expand_to_full_coords(
            multi_indices,
            var_constant_coords,
            num_obs,
        )

        np.testing.assert_array_equal(result[0], [10, 10])
        np.testing.assert_array_equal(result[1], [2, 3])
        np.testing.assert_array_equal(result[2], [20, 20])


class TestGenerateWithoutOverlap:
    """Tests for generate_without_overlap, the single-variable placement."""

    @staticmethod
    def _place(shape, num_obs, seed=42, **kwargs):
        records = {"var0": np.full(shape, np.nan)}
        return MultiVarRecordGenerator.generate_without_overlap(
            shape,
            records,
            1,
            np.array([num_obs]),
            [[]],
            {0: {}},
            seed,
            **kwargs,
        )

    def test_places_the_requested_count(self):
        """The record keeps the grid shape and holds num_obs values."""
        result = self._place([20, 20], 25, dim_split=0)

        assert result["var0"].shape == (20, 20)
        assert np.count_nonzero(~np.isnan(result["var0"])) == 25

    def test_reproducible_with_seed(self):
        """Same seed, same record."""
        result1 = self._place([10, 10], 10, dim_split=0)
        result2 = self._place([10, 10], 10, dim_split=0)

        np.testing.assert_array_equal(result1["var0"], result2["var0"])

    def test_rejects_several_variables(self):
        """Several variables belong to generate_multivar_stratified."""
        shape = [10, 10]
        records = {
            "var0": np.full(shape, np.nan),
            "var1": np.full(shape, np.nan),
        }
        with pytest.raises(ValueError, match="one variable"):
            MultiVarRecordGenerator.generate_without_overlap(
                shape,
                records,
                2,
                np.array([10, 10]),
                [[], []],
                {0: {}, 1: {}},
                42,
                dim_split=0,
            )

    def test_rejects_missing_dim_split(self):
        """Placement is stratified, so it needs the split dimension."""
        with pytest.raises(ValueError, match="dim_split"):
            self._place([10, 10], 10)


class TestGenerate:
    """Tests for main generate method."""

    def test_returns_dictionary(self):
        """Should return dictionary of records."""
        shape = [10, 10]
        records = {"var0": np.full(shape, np.nan)}
        var_num_obs = np.array([10])
        var_dims_indices = [[0, 1]]
        var_constant_dims = [[]]
        var_constant_coord_indices = {0: {}}
        num_vars = 1
        num_dims = 2
        overlap = "random"

        records, overlap_actual = MultiVarRecordGenerator.generate(
            shape,
            overlap,
            num_vars,
            var_num_obs,
            var_dims_indices,
            var_constant_dims,
            var_constant_coord_indices,
            num_dims,
            42,
            dim_split=0,
        )

        assert isinstance(records, dict)
        assert "var0" in records
        # one variable has nothing to overlap with: an empty per-variable array,
        # matching the multi-variable return type
        assert isinstance(overlap_actual, np.ndarray)
        assert overlap_actual.size == 0

    def test_random_overlap_uses_the_stratified_path(self):
        """overlap='random' goes through the stratified placement too."""
        shape = [10, 10]
        records = {
            "var0": np.full(shape, np.nan),
            "var1": np.full(shape, np.nan),
        }
        var_num_obs = np.array([10, 10])
        var_dims_indices = [[0, 1], [0, 1]]
        var_constant_dims = [[], []]
        var_constant_coord_indices = {0: {}, 1: {}}
        num_vars = 2
        num_dims = 2
        overlap = "random"

        records, overlap_actual = MultiVarRecordGenerator.generate(
            shape,
            overlap,
            num_vars,
            var_num_obs,
            var_dims_indices,
            var_constant_dims,
            var_constant_coord_indices,
            num_dims,
            42,
            dim_split=0,
        )

        # Should complete without error
        assert isinstance(records, dict)
        assert "var0" in records
        assert "var1" in records

    def test_numeric_overlap_uses_the_stratified_path(self):
        """A numeric target goes through the stratified placement.

        num_obs must reach max(shape): the placement refuses a count that
        cannot put every coordinate of the longest axis to use.
        """
        shape = [15, 15]
        records = {
            "var0": np.full(shape, np.nan),
            "var1": np.full(shape, np.nan),
        }
        var_num_obs = np.array([15, 15])
        var_dims_indices = [[0, 1], [0, 1]]
        var_constant_dims = [[], []]
        var_constant_coord_indices = {0: {}, 1: {}}
        num_vars = 2
        num_dims = 2
        overlap = 0.5

        records, overlap_actual = MultiVarRecordGenerator.generate(
            shape,
            overlap,
            num_vars,
            var_num_obs,
            var_dims_indices,
            var_constant_dims,
            var_constant_coord_indices,
            num_dims,
            42,
            dim_split=0,
        )

        # Should complete without error
        assert isinstance(records, dict)
        assert "var0" in records
        assert "var1" in records

    def test_per_variable_overlap_targets(self):
        """Should support distinct overlap targets for each non-reference variable.

        Targets are met per stratum, each rounded to whole sites. On a 2x10
        grid split along x0, var0's 10 sites fall 5 per stratum, so 0.8 and 0.4
        ask for exactly 4 and 2 sites in each and no rounding is involved.
        """
        shape = [2, 10]
        var_num_obs = np.array([10, 10, 10])
        var_dims_indices = [[0, 1], [0, 1], [0, 1]]
        var_constant_dims = [[], [], []]
        var_constant_coord_indices = {0: {}, 1: {}, 2: {}}

        records, overlap_actual = MultiVarRecordGenerator.generate(
            shape=shape,
            overlap=[0.8, 0.4],
            num_vars=3,
            var_num_obs=var_num_obs,
            var_dims_indices=var_dims_indices,
            var_constant_dims=var_constant_dims,
            var_constant_coord_indices=var_constant_coord_indices,
            num_dims=2,
            seed=42,
            dim_split=0,
        )

        assert isinstance(overlap_actual, np.ndarray)
        assert overlap_actual.shape == (2,)
        assert overlap_actual[0] == pytest.approx(0.8)
        assert overlap_actual[1] == pytest.approx(0.4)
        for var_idx in range(3):
            count = np.count_nonzero(~np.isnan(records[f"var{var_idx}"]))
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
        overlap1 = ref_coords & OverlapCalculator.extract_coordinate_set(
            records["var1"], 2
        )
        overlap2 = ref_coords & OverlapCalculator.extract_coordinate_set(
            records["var2"], 2
        )

        assert len(overlap2) < len(
            overlap1
        ), "var2 asks for less, so this is not trivial"
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
        overlap1 = ref_coords & OverlapCalculator.extract_coordinate_set(
            records["var1"], 2
        )
        overlap2 = ref_coords & OverlapCalculator.extract_coordinate_set(
            records["var2"], 2
        )

        assert not overlap2.issubset(overlap1)

    def test_single_variable_edge_case(self):
        """Should handle single variable."""
        shape = [10, 10]
        records = {"var0": np.full(shape, np.nan)}
        var_num_obs = np.array([15])
        var_dims_indices = [[0, 1]]
        var_constant_dims = [[]]
        var_constant_coord_indices = {0: {}}
        num_vars = 1
        num_dims = 2
        overlap = 0.5

        records, overlap_actual = MultiVarRecordGenerator.generate(
            shape,
            overlap,
            num_vars,
            var_num_obs,
            var_dims_indices,
            var_constant_dims,
            var_constant_coord_indices,
            num_dims,
            42,
            dim_split=0,
        )

        assert "var0" in records
        count = np.count_nonzero(~np.isnan(records["var0"]))
        assert count == 15


class TestSingleVariableCase:
    """Tests for single-variable generation (num_vars=1) to ensure compatibility."""

    def test_single_variable_basic(self):
        """Should generate single variable correctly."""
        shape = [10, 10]
        num_obs = 20

        records, overlap = MultiVarRecordGenerator.generate(
            shape=shape,
            overlap="random",
            num_vars=1,
            var_num_obs=np.array([num_obs]),
            var_dims_indices=[list(range(2))],
            var_constant_dims=[[]],
            var_constant_coord_indices={},
            num_dims=2,
            seed=42,
            dim_split=0,
        )

        assert len(records) == 1
        assert "var0" in records
        assert records["var0"].shape == tuple(shape)
        assert np.sum(~np.isnan(records["var0"])) == num_obs

    def test_single_variable_all_dims(self):
        """Should handle single variable in 3D space."""
        shape = [5, 6, 7]
        num_obs = 42

        records, overlap = MultiVarRecordGenerator.generate(
            shape=shape,
            overlap="random",
            num_vars=1,
            var_num_obs=np.array([num_obs]),
            var_dims_indices=[list(range(3))],
            var_constant_dims=[[]],
            var_constant_coord_indices={},
            num_dims=3,
            seed=123,
            dim_split=0,
        )

        assert records["var0"].shape == tuple(shape)
        assert np.sum(~np.isnan(records["var0"])) == num_obs

        # Verify observations are in valid range
        obs_values = records["var0"][~np.isnan(records["var0"])]
        assert np.all(obs_values >= 0)
        assert np.all(obs_values <= 1)

    def test_single_variable_reproducible(self):
        """Should produce same results with same seed."""
        shape = [8, 8]
        num_obs = 16

        records1, _ = MultiVarRecordGenerator.generate(
            shape=shape,
            overlap="random",
            num_vars=1,
            var_num_obs=np.array([num_obs]),
            var_dims_indices=[list(range(2))],
            var_constant_dims=[[]],
            var_constant_coord_indices={},
            num_dims=2,
            seed=42,
            dim_split=0,
        )

        records2, _ = MultiVarRecordGenerator.generate(
            shape=shape,
            overlap="random",
            num_vars=1,
            var_num_obs=np.array([num_obs]),
            var_dims_indices=[list(range(2))],
            var_constant_dims=[[]],
            var_constant_coord_indices={},
            num_dims=2,
            seed=42,
            dim_split=0,
        )

        np.testing.assert_array_equal(records1["var0"], records2["var0"])

    def test_single_variable_overlap_ignored(self):
        """Should ignore overlap parameter for single variable."""
        shape = [10, 10]
        num_obs = 30

        # Try with different overlap values - should behave identically
        records1, overlap1 = MultiVarRecordGenerator.generate(
            shape=shape,
            overlap="random",
            num_vars=1,
            var_num_obs=np.array([num_obs]),
            var_dims_indices=[list(range(2))],
            var_constant_dims=[[]],
            var_constant_coord_indices={},
            num_dims=2,
            seed=999,
            dim_split=0,
        )

        records2, overlap2 = MultiVarRecordGenerator.generate(
            shape=shape,
            overlap=0.5,
            num_vars=1,
            var_num_obs=np.array([num_obs]),
            var_dims_indices=[list(range(2))],
            var_constant_dims=[[]],
            var_constant_coord_indices={},
            num_dims=2,
            seed=999,
            dim_split=0,
        )

        # A single variable has nothing to overlap with
        np.testing.assert_array_equal(records1["var0"], records2["var0"])
