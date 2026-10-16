"""Tests for data_structure="correlated".

The field replaces each occupied site's uniform draw by a smooth Gaussian
field evaluated at the site's global grid indices.
"""

import contextlib
import io
import os
import warnings

import numpy as np
import pytest
import xarray as xr

from tabray.generate_data import GenerateData
from tabray.generators import CorrelatedField


def full_grid_indices(grid_shape):
    """Global indices of every site of a grid, one array per dim."""
    return tuple(axis.ravel() for axis in np.indices(grid_shape))


def field_on_grid(grid_shape, lengths, seed=0, noise=0.0, varying_dims=None):
    """The field's values on every site, shaped like the grid."""
    num_dims = len(grid_shape)
    if varying_dims is None:
        varying_dims = range(num_dims)
    field = CorrelatedField(seed, 0, num_dims, varying_dims, lengths, noise)
    uniform = np.random.default_rng(seed).uniform(0, 1, int(np.prod(grid_shape)))
    return field.apply(uniform, full_grid_indices(grid_shape)).reshape(grid_shape)


def lag_correlation(values, lag, axis):
    """Correlation between sites ``lag`` apart along ``axis``."""
    first = np.take(values, np.arange(values.shape[axis] - lag), axis=axis)
    second = np.take(values, np.arange(lag, values.shape[axis]), axis=axis)
    return np.corrcoef(first.ravel(), second.ravel())[0, 1]


def generate(temp_dir, name, **params):
    """Run GenerateData quietly and return the written netCDF dataset."""
    path = os.path.join(temp_dir, name, "ds.nc")
    with contextlib.redirect_stdout(io.StringIO()):
        GenerateData(**params).generate(
            netcdf_filepath=path,
            parquet_filepath=os.path.join(temp_dir, name, "df"),
        )
    return xr.open_dataset(path).load()


SINGLE = dict(num_obs=600, num_dims=2, ratio_dims=[3, 2], density=0.4, seed=4)
MULTI = dict(
    num_obs=600,
    num_dims=3,
    ratio_dims=[3, 2, 2],
    density=[0.4, 0.3, 0.2],
    seed=4,
    num_vars=3,
    var_dims=[3, 2, 3],
    overlap=[0.6, 0.5],
)


class TestFieldStatistics:
    """The field has the autocorrelation and the marginal it promises."""

    def test_correlation_at_one_length(self):
        """Sites L apart: Gaussian correlation e^-1/2, seen through ndtr."""
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
    """A site's value depends only on (seed, variable, global indices, draw)."""

    def test_split_evaluation_is_bit_identical(self):
        """What a parallel chunk does: evaluate a slice of the sites."""
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


class TestGenerateData:
    """The option through the public API."""

    @pytest.mark.parametrize("params", [SINGLE, MULTI], ids=["single", "multi"])
    def test_default_is_uncorrelated(self, temp_dir, params):
        default = generate(temp_dir, "default", **params)
        explicit = generate(
            temp_dir, "explicit", **params, data_structure="uncorrelated"
        )
        xr.testing.assert_identical(default, explicit)

    @pytest.mark.parametrize("params", [SINGLE, MULTI], ids=["single", "multi"])
    def test_sites_do_not_move(self, temp_dir, params):
        plain = generate(temp_dir, "plain", **params)
        smooth = generate(
            temp_dir, "smooth", **params, data_structure="correlated", noise=0.1
        )
        for name in plain.data_vars:
            assert np.array_equal(plain[name].notnull(), smooth[name].notnull())
            assert not np.array_equal(
                plain[name].fillna(-1).values, smooth[name].fillna(-1).values
            )

    def test_per_variable_structure(self, temp_dir):
        plain = generate(temp_dir, "plain", **MULTI)
        mixed = generate(
            temp_dir,
            "mixed",
            **MULTI,
            data_structure=["correlated", "uncorrelated", "correlated"],
        )
        xr.testing.assert_identical(plain["var1"], mixed["var1"])
        assert not plain["var0"].equals(mixed["var0"])

    def test_encodings_keep_sites(self, temp_dir):
        """dtype and packing apply to the field as to the uniform draw."""
        encoded = dict(dtype=["float64", "int16", "S4"], pack=[None, None, None])
        plain = generate(temp_dir, "plain", **MULTI, data_structure="correlated")
        typed = generate(
            temp_dir, "typed", **MULTI, **encoded, data_structure="correlated"
        )
        packed = generate(
            temp_dir, "packed", **MULTI, pack="int16", data_structure="correlated"
        )
        for name in plain.data_vars:
            sites = plain[name].notnull().values
            assert np.array_equal(sites, packed[name].notnull().values)
            if typed[name].dtype.kind == "f":
                assert np.array_equal(sites, typed[name].notnull().values)
        assert typed["var2"].dtype.kind in "SO"

    def test_long_length_warns(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with pytest.warns(UserWarning, match="exceeds"):
                GenerateData(
                    **SINGLE, data_structure="correlated", correlation_length=1000
                )

    def test_no_warning_on_dims_no_correlated_variable_varies_along(self):
        # Only var1 is correlated and it varies along two of the three dims;
        # the length on the third dim is never used.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fields = CorrelatedField.per_variable(
                ["uncorrelated", "correlated"],
                [5, 5, 1000],
                None,
                num_vars=2,
                grid_shape=[20, 20, 20],
                var_dims_indices=[[0, 1, 2], [0, 1]],
                seed=0,
            )
        assert fields[0] is None
        with pytest.warns(UserWarning, match="x1"):
            CorrelatedField.per_variable(
                ["uncorrelated", "correlated"],
                [5, 1000, 1000],
                None,
                num_vars=2,
                grid_shape=[20, 20, 20],
                var_dims_indices=[[0, 1, 2], [0, 1]],
                seed=0,
            )

    def test_default_length_warning_says_default(self):
        with pytest.warns(UserWarning, match="default correlation_length"):
            CorrelatedField.per_variable(
                "correlated", None, None, 1, [5, 40], [[0, 1]], seed=0
            )


class TestValidation:
    """Unusable settings raise before anything is generated."""

    @pytest.mark.parametrize(
        "settings, message",
        [
            (dict(data_structure="smooth"), "Unknown data_structure"),
            (dict(data_structure="correlated", correlation_length=0.5), "at least"),
            (dict(data_structure="correlated", correlation_length=-2), "at least"),
            (dict(data_structure="correlated", correlation_length=np.nan), "at least"),
            (
                dict(data_structure="correlated", correlation_length=[np.nan, 3]),
                "at least",
            ),
            (dict(data_structure="correlated", correlation_length=np.inf), "at least"),
            (dict(data_structure="correlated", correlation_length=[3]), "one entry"),
            (dict(data_structure="correlated", correlation_length=["a", 2]), "numbers"),
            (dict(data_structure="correlated", noise=1.5), "between 0 and 1"),
            (dict(correlation_length=5), "data_structure='correlated'"),
            (dict(noise=0.1), "data_structure='correlated'"),
        ],
    )
    def test_single_variable(self, settings, message):
        with contextlib.redirect_stdout(io.StringIO()):
            with pytest.raises(ValueError, match=message):
                GenerateData(**SINGLE, **settings)

    def test_wrong_number_of_structures(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with pytest.raises(ValueError, match="one per variable"):
                GenerateData(**MULTI, data_structure=["correlated", "correlated"])

    def test_noise_on_an_uncorrelated_variable(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with pytest.raises(ValueError, match="already all noise"):
                GenerateData(
                    **MULTI,
                    data_structure=["correlated", "uncorrelated", "correlated"],
                    noise=[0.1, 0.1, 0.1],
                )

    def test_single_noise_applies_to_correlated_variables_only(self):
        with contextlib.redirect_stdout(io.StringIO()):
            data = GenerateData(
                **MULTI,
                data_structure=["correlated", "uncorrelated", "correlated"],
                noise=0.1,
            )
        fields = data.correlated_fields
        assert fields[1] is None
        assert fields[0].noise == pytest.approx(0.1)
        assert fields[2].noise == pytest.approx(0.1)

    def test_numpy_scalar_noise(self):
        with contextlib.redirect_stdout(io.StringIO()):
            data = GenerateData(
                **SINGLE, data_structure="correlated", noise=np.float32(0.1)
            )
        assert data.correlated_fields[0].noise == pytest.approx(0.1)
