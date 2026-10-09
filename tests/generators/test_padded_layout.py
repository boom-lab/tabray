"""Tests for layout="padded".

Argo and CrocoLake fill the first k levels of each profile; the generator's
default scatters its occupancy, which at a fixed density changes compressed
array size by 2.3x and reverses which format is smaller. See
docs/layout_plan.md.
"""

import contextlib
import io

import numpy as np
import xarray as xr
import pytest

from data_sparsity.generators.record_generator import RecordGenerator
from data_sparsity.generate_data import GenerateData
from data_sparsity.validators import SparsityValidator


def occupancy(shape, indices):
    grid = np.zeros(shape, dtype=bool)
    grid[indices] = True
    return grid


class TestPaddedStratumCounts:
    """Counts come out exact, and the guards fire before anything is placed."""

    @pytest.mark.parametrize(
        "shape,obs",
        [
            ([20, 520], 10296),
            ([2000, 1042], 310516),
            ([4, 5, 6], 100),
        ],
    )
    def test_counts_sum_to_the_request(self, shape, obs):
        counts, _ = RecordGenerator.padded_stratum_counts(
            shape,
            obs,
            0,
            len(shape) - 1,
        )
        assert counts.sum() == obs

    def test_every_line_gets_at_least_one(self):
        shape = [4, 5, 6]
        lines = 5  # the middle dimension
        counts, _ = RecordGenerator.padded_stratum_counts(shape, 100, 0, 2)
        assert (counts >= lines).all()

    def test_one_stratum_can_afford_the_full_length(self):
        shape, obs = [20, 520], 10296
        counts, carrier = RecordGenerator.padded_stratum_counts(
            shape,
            obs,
            0,
            1,
        )
        assert counts[carrier] >= shape[1]

    def test_too_few_observations_raises(self):
        """Fewer than one per line plus one full run cannot cover the axes."""
        with pytest.raises(ValueError, match="for a padded layout"):
            RecordGenerator.padded_stratum_counts([20, 520], 100, 0, 1)

    def test_more_observations_than_cells_raises(self):
        """The apportionment would cap it and come out short with no error."""
        with pytest.raises(ValueError, match="exceeds the"):
            RecordGenerator.padded_stratum_counts([5, 4, 3], 80, 0, 2)


class TestPaddedPlacement:

    @pytest.mark.parametrize(
        "shape,obs,padded",
        [
            ([20, 520], 10296, 1),
            ([2000, 1042], 310516, 1),
            ([4, 5, 6], 100, 2),
        ],
    )
    def test_exact_count_and_no_duplicates(self, shape, obs, padded):
        indices, values = RecordGenerator.generate_padded_indices(
            shape,
            obs,
            42,
            0,
            padded,
        )
        assert len(values) == obs
        assert len(set(zip(*indices))) == obs

    @pytest.mark.parametrize(
        "shape,obs,padded",
        [
            ([20, 520], 10296, 1),
            ([2000, 1042], 310516, 1),
            ([4, 5, 6], 100, 2),
        ],
    )
    def test_every_coordinate_of_every_axis_is_used(self, shape, obs, padded):
        """Coverage comes from the construction, not from an LHS stage: every
        line holds an observation and one line runs the full length."""
        indices, _ = RecordGenerator.generate_padded_indices(
            shape,
            obs,
            42,
            0,
            padded,
        )
        grid = occupancy(shape, indices)
        for dim, size in enumerate(shape):
            others = tuple(d for d in range(len(shape)) if d != dim)
            assert grid.any(axis=others).all(), f"axis {dim} left a coordinate unused"

    def test_every_line_fills_its_first_k_cells(self):
        shape, obs = [2000, 1042], 310516
        indices, _ = RecordGenerator.generate_padded_indices(shape, obs, 7, 0, 1)
        grid = occupancy(shape, indices)
        lengths = grid.sum(axis=1)
        for row, length in enumerate(lengths):
            assert grid[row, :length].all()
            assert not grid[row, length:].any()

    def test_lengths_are_even(self):
        """Equal weights: every line but the full-length one gets the same k,
        give or take one cell."""
        indices, _ = RecordGenerator.generate_padded_indices(
            [2000, 1042],
            310516,
            7,
            0,
            1,
        )
        lengths = occupancy([2000, 1042], indices).sum(axis=1)
        assert lengths.max() == 1042
        others = np.sort(lengths)[:-1]
        assert others.max() - others.min() <= 1

    def test_serial_and_parallel_agree(self):
        """A caller asking for a subset of strata gets the same slices."""
        shape, obs = [2000, 1042], 310516
        whole, whole_values = RecordGenerator.generate_padded_indices(
            shape,
            obs,
            7,
            0,
            1,
        )
        pieces = [
            RecordGenerator.generate_padded_indices(
                shape,
                obs,
                7,
                0,
                1,
                strata=range(a, b),
            )
            for a, b in ((0, 500), (500, 1200), (1200, 2000))
        ]
        split = set()
        for indices, _ in pieces:
            split |= set(zip(*indices))
        assert set(zip(*whole)) == split
        assert np.array_equal(
            np.sort(whole_values),
            np.sort(np.concatenate([v for _, v in pieces])),
        )

    def test_padded_dim_cannot_be_the_split_dim(self):
        with pytest.raises(ValueError, match="both 0"):
            RecordGenerator.generate_padded_indices([20, 520], 10296, 42, 0, 0)

    def test_dispatch_needs_a_padded_dim(self):
        with pytest.raises(ValueError, match="needs padded_dim"):
            RecordGenerator.generate_stratified_indices(
                [20, 520],
                10296,
                42,
                0,
                layout="padded",
            )

    def test_unknown_layout_raises(self):
        with pytest.raises(ValueError, match="Unknown layout"):
            RecordGenerator.generate_stratified_indices(
                [20, 520],
                10296,
                42,
                0,
                layout="sideways",
            )


class TestPaddedMinimumDensity:

    def test_bound_is_higher_than_scattered(self):
        """Every line needs an observation and one must run the full length."""
        shape = np.array([20, 520])
        scattered = SparsityValidator.compute_min_density(shape)
        padded = SparsityValidator.compute_min_density(shape, padded_dim=1)
        assert padded > scattered
        assert round(padded * 20 * 520) == 20 + 520 - 1

    def test_split_dimension_does_not_enter(self):
        """n_lines is prod(shape)/n_padded whichever axis the strata run along."""
        shape = np.array([4, 5, 6])
        assert SparsityValidator.compute_min_density(
            shape,
            padded_dim=2,
        ) == pytest.approx((4 * 5 + 6 - 1) / 120)


class TestEveryVariableIsPadded:
    """layout applies to the whole dataset, not to var0 alone."""

    PARAMS = dict(
        num_obs=100,
        num_dims=3,
        ratio_dims=[1, 2, 2],
        seed=77,
        density=[0.4, 0.2],
        num_vars=2,
        layout="padded",
    )

    @staticmethod
    def build(tmp_path, tag, max_obs=None, **kwargs):
        out = tmp_path / tag
        with contextlib.redirect_stdout(io.StringIO()):
            gen = GenerateData(**kwargs, max_obs=max_obs)
            extra = {}
            if max_obs:
                extra = dict(parquet_tmp=str(out / "tmp"), merge_nc=True)
            gen.generate(
                netcdf_filepath=str(out / "d.nc"),
                parquet_filepath=str(out / "pq" / "d.parquet"),
                **extra,
            )
        dataset = xr.open_dataset(out / "d.nc")
        return gen, dataset

    @staticmethod
    def fills_first_k(values, axis):
        occupied = np.moveaxis(~np.isnan(values), axis, -1)
        rows = occupied.reshape(-1, occupied.shape[-1])
        lengths = rows.sum(axis=1)
        return all(row[:k].all() and not row[k:].any() for row, k in zip(rows, lengths))

    def test_a_variable_with_the_padded_axis_fills_its_first_k_cells(self, tmp_path):
        gen, dataset = self.build(tmp_path, "a", var_dims=[3, [1, 2]], **self.PARAMS)
        var1 = dataset["var1"]
        assert "x2" in var1.dims
        assert int(var1.notnull().sum()) == gen.var_num_obs[1]
        assert self.fills_first_k(var1.values, var1.get_axis_num("x2"))

    def test_a_variable_without_it_keeps_its_count(self, tmp_path):
        """No padded axis to fill: the variable is placed as under scattered."""
        gen, dataset = self.build(tmp_path, "b", var_dims=[3, [0, 1]], **self.PARAMS)
        var1 = dataset["var1"]
        assert "x2" not in var1.dims
        assert int(var1.notnull().sum()) == gen.var_num_obs[1]

    def test_serial_and_parallel_agree(self, tmp_path):
        params = dict(var_dims=[3, [1, 2]], **self.PARAMS)
        serial, ser = self.build(tmp_path, "s", **params)
        parallel, par = self.build(tmp_path, "p", max_obs=40, **params)
        assert parallel.NTASKS > 1
        for name in ("var0", "var1"):
            np.testing.assert_array_equal(ser[name].values, par[name].values)

    def test_the_printout_states_the_bound(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            GenerateData(var_dims=[3, [1, 2]], **self.PARAMS)
        assert "Minimum density:" in out.getvalue()
        assert "n_padded" in out.getvalue()
