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
    """Two-element density: [max, min].

    var0 takes the maximum and some other variable takes the minimum, so both
    prescribed values are used. With two variables the range is exactly the two
    densities. Any further variables are drawn from [min, max].
    """

    def test_two_variables_are_exactly_the_two_densities(self, fixed_rng):
        """With two variables the range is the two prescribed densities."""
        result = MultiVarSparsityConfig.from_two_element_list([0.8, 0.2], 2, fixed_rng)
        assert list(result) == [0.8, 0.2]

    def test_minimum_is_always_used_by_some_variable(self):
        """The lower bound is a density that some variable takes, not only a bound.

        Drawing every non-reference variable from the range could leave the
        minimum unused.
        """
        for num_vars in (2, 3, 5, 10):
            for seed in range(20):
                result = MultiVarSparsityConfig.from_two_element_list(
                    [0.9, 0.1],
                    num_vars,
                    np.random.default_rng(seed),
                )
                assert result[0] == 0.9, "var0 takes the maximum"
                assert 0.1 in result, "some variable takes the minimum"

    def test_reference_is_the_maximum_for_every_seed(self):
        """var0 takes the maximum for every seed.

        If the maximum went to a variable chosen at random, the seeds that give
        it to another variable would produce a var0 smaller than another
        variable, which the overlap reference must never be.
        """
        for seed in range(50):
            result = MultiVarSparsityConfig.from_two_element_list(
                [0.9, 0.1],
                3,
                np.random.default_rng(seed),
            )
            assert result[0] == 0.9

    def test_all_densities_lie_within_the_range(self, fixed_rng):
        result = MultiVarSparsityConfig.from_two_element_list([0.9, 0.1], 5, fixed_rng)
        assert len(result) == 5
        assert all(0.1 <= value <= 0.9 for value in result)

    def test_array_length_matches_num_vars(self, fixed_rng):
        for num_vars in (2, 3, 10):
            result = MultiVarSparsityConfig.from_two_element_list(
                [0.8, 0.2],
                num_vars,
                fixed_rng,
            )
            assert len(result) == num_vars

    def test_reproducible_with_same_seed(self):
        a = MultiVarSparsityConfig.from_two_element_list(
            [0.8, 0.2], 5, np.random.default_rng(123)
        )
        b = MultiVarSparsityConfig.from_two_element_list(
            [0.8, 0.2], 5, np.random.default_rng(123)
        )
        np.testing.assert_array_equal(a, b)

    def test_different_with_different_seed(self):
        a = MultiVarSparsityConfig.from_two_element_list(
            [0.8, 0.2], 10, np.random.default_rng(123)
        )
        b = MultiVarSparsityConfig.from_two_element_list(
            [0.8, 0.2], 10, np.random.default_rng(456)
        )
        assert not np.array_equal(a, b)
        assert a[0] == b[0] == 0.8, "only the non-reference densities vary"

    def test_minimum_goes_to_different_variables(self):
        """The non-reference densities are shuffled, so var1 is not always the minimum."""
        holders = set()
        for seed in range(50):
            result = MultiVarSparsityConfig.from_two_element_list(
                [0.9, 0.1],
                4,
                np.random.default_rng(seed),
            )
            holders.add(int(np.argmin(result[1:])))
        assert len(holders) > 1


class TestFromFullList:
    """Tests for from_full_list method."""

    def test_correct_length_matches(self):
        """Correct length should pass through."""
        sparsity_list = [0.2, 0.5, 0.8]
        result = MultiVarSparsityConfig.from_full_list(sparsity_list)
        np.testing.assert_array_equal(result, sparsity_list)

    def test_values_preserved(self):
        """Values should be preserved exactly."""
        sparsity_list = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = MultiVarSparsityConfig.from_full_list(sparsity_list)
        for i, val in enumerate(sparsity_list):
            assert result[i] == val

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
        assert result[0] == 50  # Half of max

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

    def test_density_is_a_share_of_the_own_grid(self):
        """var1 on a quarter of the grid: 0.1 of its own 64 cells, not of 256.

        var0 holds 102 of a 256-cell grid at density 0.4; var1 asks for 0.1
        on its own 64 cells, so 6.4 -> 6 observations.
        """
        densities = np.array([0.4, 0.1])
        result = MultiVarSparsityConfig.compute_var_num_obs(
            densities,
            102,
            np.array([1.0, 0.25]),
        )
        np.testing.assert_array_equal(result, [102, 6])

    def test_full_grid_fractions_change_nothing(self):
        """A variable on every dimension gets the count it always had."""
        densities = np.array([0.4, 0.3, 0.1])
        plain = MultiVarSparsityConfig.compute_var_num_obs(densities, 102)
        ones = MultiVarSparsityConfig.compute_var_num_obs(
            densities,
            102,
            np.ones(3),
        )
        np.testing.assert_array_equal(plain, ones)


class TestSetupFromParameter:
    """Tests for setup_from_parameter integration."""

    def test_scalar_input(self, fixed_rng):
        """Scalar input should work."""
        sparsities, num_obs = MultiVarSparsityConfig.setup_from_parameter(
            0.5,
            3,
            100,
            0.1,
            fixed_rng,
        )
        assert len(sparsities) == 3
        assert len(num_obs) == 3
        assert all(s >= 0.1 for s in sparsities)

    def test_list_input(self, fixed_rng):
        """List input should work. [max, min]: the reference density leads."""
        sparsities, num_obs = MultiVarSparsityConfig.setup_from_parameter(
            [0.8, 0.2],
            2,
            100,
            0.1,
            fixed_rng,
        )
        assert len(sparsities) == 2
        assert len(num_obs) == 2
