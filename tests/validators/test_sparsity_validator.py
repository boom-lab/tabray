"""Tests for SparsityValidator class."""

import pytest
import numpy as np
from data_sparsity.validators import SparsityValidator


class TestComputeMinDensity:
    """Tests for compute_min_density method.

    The bound is max(shape)/prod(shape): the fewest observations that can still
    put every coordinate on every axis to use is the length of the LONGEST
    axis, because each observation supplies one coordinate per axis.
    """

    def test_uniform_dimensions(self):
        """On a cubic grid the bound is 1/(n^(d-1)), as derived by hand."""
        nb_coords = np.array([10, 10, 10])
        result = SparsityValidator.compute_min_density(nb_coords)
        assert result == 1 / (10**(3 - 1))
        assert result == 10 / 1000

    def test_non_uniform_dimensions(self):
        """Non-cubic grids are set by the longest axis, not the shortest."""
        nb_coords = np.array([5, 10, 20])
        result = SparsityValidator.compute_min_density(nb_coords)
        assert result == 20 / 1000

    def test_single_dimension(self):
        """One dimension of length n needs all n observations."""
        nb_coords = np.array([8])
        result = SparsityValidator.compute_min_density(nb_coords)
        assert result == 1

    def test_four_dimensions(self):
        """The longest axis sets the bound at any dimensionality."""
        nb_coords = np.array([3, 7, 5, 9])
        result = SparsityValidator.compute_min_density(nb_coords)
        assert result == 9 / (3 * 7 * 5 * 9)

    def test_large_dimensions(self):
        """Large dimensions should give small density."""
        nb_coords = np.array([1000, 2000])
        result = SparsityValidator.compute_min_density(nb_coords)
        assert result == 2000 / 2_000_000

    def test_min_observations_equals_longest_axis(self):
        """The bound expressed as a count, which is how it is derived."""
        for shape in ([10, 10, 10], [5, 10, 20], [4, 7, 10], [3, 7, 5, 9]):
            nb_coords = np.array(shape)
            min_obs = SparsityValidator.compute_min_density(nb_coords) * np.prod(shape)
            assert round(min_obs) == max(shape)

    def test_never_stricter_than_the_old_bound(self):
        """Every configuration accepted before is still accepted.

        The old bound was 1/nmin**(d-1). Relaxing must not reject anything it
        used to allow, so the new value is never larger.
        """
        for shape in ([10, 10, 10], [5, 10, 20], [4, 7, 10], [3, 7, 5, 9],
                      [1000, 2000], [8], [2, 3, 5, 7], [365, 2041, 4320]):
            nb_coords = np.array(shape)
            old = 1.0 / (min(shape) ** (len(shape) - 1))
            assert SparsityValidator.compute_min_density(nb_coords) <= old

    def test_agrees_with_the_old_bound_on_cubic_grids(self):
        """The hand-derived formula was exact there, and stays exact."""
        for n, d in ((10, 3), (7, 2), (4, 5), (32, 2)):
            nb_coords = np.array([n] * d)
            assert SparsityValidator.compute_min_density(nb_coords) == pytest.approx(
                1.0 / (n ** (d - 1))
            )


class TestValidateDensityBounds:
    """Tests for validate_density_bounds method."""

    def test_valid_density_in_range_passes(self):
        """Valid density in range should pass."""
        result = SparsityValidator.validate_density_bounds(0.5, 0.1)
        assert result == 0.5

    def test_density_zero_returns_minimum(self):
        """Density = 0 should return minimum."""
        result = SparsityValidator.validate_density_bounds(0.0, 0.1)
        assert result == 0.1

    def test_density_less_than_minimum_not_zero_raises_value_error(self):
        """Density < minimum (not 0) should raise ValueError."""
        with pytest.raises(ValueError, match="lower than minimum value"):
            SparsityValidator.validate_density_bounds(0.05, 0.1)

    def test_density_equals_minimum_passes(self):
        """Density = minimum should pass."""
        result = SparsityValidator.validate_density_bounds(0.1, 0.1)
        assert result == 0.1

    def test_density_equals_one_passes(self):
        """Density = 1 should pass."""
        result = SparsityValidator.validate_density_bounds(1.0, 0.1)
        assert result == 1.0

    def test_edge_case_minimum_plus_epsilon(self):
        """Density = minimum + epsilon should pass."""
        result = SparsityValidator.validate_density_bounds(0.10001, 0.1)
        assert result == pytest.approx(0.10001)

    def test_edge_case_minimum_minus_epsilon(self):
        """Density = minimum - epsilon should raise ValueError."""
        with pytest.raises(ValueError, match="lower than minimum value"):
            SparsityValidator.validate_density_bounds(0.09999, 0.1)


class TestValidateNumObsConsistency:
    """Tests for validate_num_obs_consistency method."""

    def test_matching_num_obs_passes_unchanged(self):
        """Matching num_obs should pass unchanged."""
        nb_coords = np.array([10, 10])
        num_obs, density = SparsityValidator.validate_num_obs_consistency(
            50, 0.5, nb_coords
        )
        assert num_obs == 50
        assert density == 0.5

    def test_non_matching_num_obs_adjusted(self):
        """Non-matching num_obs should be adjusted."""
        nb_coords = np.array([10, 10])
        num_obs, density = SparsityValidator.validate_num_obs_consistency(
            51, 0.5, nb_coords
        )
        assert num_obs == 50
        assert density == 0.5

    def test_returns_integer(self):
        """Should return integer num_obs."""
        nb_coords = np.array([10, 10])
        num_obs, _ = SparsityValidator.validate_num_obs_consistency(
            50.7, 0.5, nb_coords
        )
        assert isinstance(num_obs, int)

    def test_returns_adjusted_density(self):
        """Should return adjusted density."""
        nb_coords = np.array([10, 10])
        num_obs, density = SparsityValidator.validate_num_obs_consistency(
            55, 0.5, nb_coords
        )
        assert density == pytest.approx(0.5, rel=0.1)

    def test_adjusted_density_in_bounds(self):
        """Adjusted density should be in valid bounds."""
        nb_coords = np.array([10, 10])
        _, density = SparsityValidator.validate_num_obs_consistency(
            50, 0.5, nb_coords
        )
        assert 0.0 <= density <= 1.0

    def test_edge_case_large_grid(self):
        """Large grid should work."""
        nb_coords = np.array([100, 100, 100])
        num_obs, density = SparsityValidator.validate_num_obs_consistency(
            100000, 0.1, nb_coords
        )
        assert num_obs == 100000
        assert density == 0.1

    def test_edge_case_small_grid(self):
        """Small grid should work."""
        nb_coords = np.array([2, 2])
        num_obs, density = SparsityValidator.validate_num_obs_consistency(
            2, 0.5, nb_coords
        )
        assert num_obs == 2
        assert density == 0.5

    def test_out_of_bounds_after_adjustment_raises_value_error(self):
        """Out of bounds after adjustment should raise ValueError."""
        # Create a case where adjusted density would be > 1
        nb_coords = np.array([2, 2])
        with pytest.raises(ValueError, match="out of bounds"):
            SparsityValidator.validate_num_obs_consistency(
                5, 1.5, nb_coords
            )

    def test_density_recalculation_correct(self):
        """Density recalculation should be correct."""
        nb_coords = np.array([10, 10])
        num_obs, density = SparsityValidator.validate_num_obs_consistency(
            30, 0.3, nb_coords
        )
        expected_density = num_obs / np.prod(nb_coords)
        assert density == pytest.approx(expected_density)
