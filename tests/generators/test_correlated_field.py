"""Tests for CorrelatedField, the smooth value field.

The field replaces each occupied cell's uniform draw by a smooth Gaussian
field evaluated at the cell's global grid indices. See
docs/correlated_values.md.
"""

import numpy as np
import pytest

from tabray.generators import CorrelatedField


def full_grid_indices(grid_shape):
    """Global indices of every cell of a grid, one array per dim."""
    return tuple(axis.ravel() for axis in np.indices(grid_shape))


def field_on_grid(grid_shape, lengths, seed=0, noise=0.0, varying_dims=None):
    """The field's values on every cell, shaped like the grid."""
    num_dims = len(grid_shape)
    if varying_dims is None:
        varying_dims = range(num_dims)
    field = CorrelatedField(seed, 0, num_dims, varying_dims, lengths, noise)
    uniform = np.random.default_rng(seed).uniform(0, 1, int(np.prod(grid_shape)))
    return field.apply(uniform, full_grid_indices(grid_shape)).reshape(grid_shape)


def lag_correlation(values, lag, axis):
    """Correlation between cells ``lag`` apart along ``axis``."""
    first = np.take(values, np.arange(values.shape[axis] - lag), axis=axis)
    second = np.take(values, np.arange(lag, values.shape[axis]), axis=axis)
    return np.corrcoef(first.ravel(), second.ravel())[0, 1]


class TestFieldStatistics:
    """The field has the autocorrelation and the marginal it promises."""

    def test_correlation_at_one_length(self):
        """Cells L apart: Gaussian correlation e^-1/2, seen through ndtr."""
        # ndtr turns a normal correlation rho into (6/pi) arcsin(rho/2)
        expected = 6 / np.pi * np.arcsin(np.exp(-0.5) / 2)
        measured = [
            lag_correlation(field_on_grid([400, 400], [10, 10], seed=s), 10, axis=1)
            for s in range(5)
        ]
        assert np.mean(measured) == pytest.approx(expected, abs=0.1)

    def test_zero_length_is_uncorrelated(self):
        measured = [
            lag_correlation(field_on_grid([400, 400], [10, 0], seed=s), 1, axis=1)
            for s in range(5)
        ]
        assert abs(np.mean(measured)) < 0.15

    def test_none_length_is_constant(self):
        values = field_on_grid([30, 40], [None, 5])
        assert np.all(values == values[:1, :])

    def test_constant_dims_are_ignored(self):
        """A variable pinned on dim 0 does not vary along it."""
        values = field_on_grid([30, 40], [5, 5], varying_dims=[1])
        assert np.all(values == values[:1, :])

    def test_marginal_is_uniform(self):
        values = field_on_grid([400, 400], [2, 2])
        counts, _ = np.histogram(values, bins=10, range=(0, 1))
        assert np.all(np.abs(counts / values.size - 0.1) < 0.03)

    def test_full_noise_is_the_uniform_draw(self):
        grid_shape = [100, 100]
        uniform = np.random.default_rng(0).uniform(0, 1, 10_000)
        field = CorrelatedField(0, 0, 2, [0, 1], 10, noise=1.0)
        values = field.apply(uniform, full_grid_indices(grid_shape))
        np.testing.assert_allclose(values, uniform, atol=1e-9)

    def test_noise_lowers_neighbour_correlation(self):
        smooth = lag_correlation(field_on_grid([200, 200], 10), 1, axis=1)
        noisy = lag_correlation(field_on_grid([200, 200], 10, noise=0.5), 1, axis=1)
        assert noisy < smooth - 0.1


class TestFieldDeterminism:
    """A cell's value depends only on (seed, variable, global indices, draw)."""

    def test_split_evaluation_is_bit_identical(self):
        """What a parallel chunk does: evaluate a slice of the cells."""
        indices = full_grid_indices([37, 53])
        uniform = np.random.default_rng(1).uniform(0, 1, indices[0].size)
        field = CorrelatedField(3, 1, 2, [0, 1], [4, 7], noise=0.2)
        whole = field.apply(uniform, indices)
        cut = 999
        parts = np.concatenate(
            [
                field.apply(uniform[:cut], tuple(axis[:cut] for axis in indices)),
                field.apply(uniform[cut:], tuple(axis[cut:] for axis in indices)),
            ]
        )
        assert np.array_equal(whole, parts)

    def test_one_length_does_not_change_other_dims(self):
        """Every draw is made whatever the lengths."""
        first = CorrelatedField(0, 0, 3, [0, 1, 2], [4, 5, 6])
        second = CorrelatedField(0, 0, 3, [0, 1, 2], [4, 50, 6])
        assert np.array_equal(
            first.wavenumbers[:, [0, 2]], second.wavenumbers[:, [0, 2]]
        )

    def test_variables_get_different_fields(self):
        first = CorrelatedField(0, 0, 2, [0, 1], 5)
        second = CorrelatedField(0, 1, 2, [0, 1], 5)
        assert not np.array_equal(first.phases, second.phases)

    def test_empty_input(self):
        field = CorrelatedField(0, 0, 2, [0, 1], 5)
        empty = (np.empty(0, dtype=np.int64),) * 2
        assert field.apply(np.empty(0), empty).size == 0
