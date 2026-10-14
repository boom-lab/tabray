"""Tests for DatasetDescription."""

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tabray.generate_data import GenerateData
from tabray.output import DatasetDescription


@pytest.fixture(name="known")
def fixture_known():
    """var0 on (x0, x1), var1 on x1 only.

    var0 occupies x1 = 0, 1; var1 occupies x1 = 1, 2
    -> overlap         = |{0, 1} & {1, 2}| / |{0, 1}| = 0.5
    -> overlap_reverse = |{0, 1} & {1, 2}| / |{1, 2}| = 0.5
    """
    var0 = np.full((2, 3), np.nan)
    var0[0, 0] = var0[1, 1] = 1.0
    var1 = np.array([np.nan, 2.0, 3.0])
    return xr.Dataset(
        {"var0": (("x0", "x1"), var0), "var1": (("x1",), var1)},
        coords={"x0": [10, 20], "x1": [1, 2, 3]},
    )


def test_counts_from_the_data(known):
    table = DatasetDescription.describe_dataset(known)
    assert table.attrs["grid"] == {"x0": 2, "x1": 3}
    assert table.loc["var0", "var_sites"] == 6
    assert table.loc["var0", "var_sites_occupied"] == 2
    assert table.loc["var0", "unused_coords"] == 1  # x1 = 3
    assert np.isnan(table.loc["var0", "overlap"])
    assert np.isnan(table.loc["var0", "overlap_reverse"])
    assert table.loc["var1", "dims"] == ("x1",)
    assert table.loc["var1", "density"] == pytest.approx(2 / 3)
    assert table.loc["var1", "overlap"] == pytest.approx(0.5)
    assert table.loc["var1", "overlap_reverse"] == pytest.approx(0.5)


COLUMNS = DatasetDescription.COLUMNS[:-1]  # all but dtype


def test_parquet_agrees_with_netcdf(known, tmp_path):
    nc = DatasetDescription.describe_dataset(known)
    pq = DatasetDescription.describe_dataset(
        known_parquet(tmp_path), coords=["x0", "x1"]
    )
    pd.testing.assert_frame_equal(nc[COLUMNS], pq[COLUMNS])


def test_parquet_with_repeated_variables_agrees_with_netcdf(tmp_path):
    """var1 on (x0, x1) repeats along x2, var2 on x0 along x1 and x2.

    No row has NaN, so both are assumed constant along the dims they repeat
    along, with a warning.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        gen = GenerateData(
            num_obs=400,
            num_dims=3,
            ratio_dims=(1, 2, 3),
            # var2 on several x0 sites: one value on a single site would be
            # ambiguous (a constant along x1, x2 gives the same rows)
            density=[0.5, 0.3, 0.5],
            seed=7,
            num_vars=3,
            var_dims=[3, [0, 1], [0]],
        )
        gen.generate(
            netcdf_filepath=str(tmp_path / "d.nc"),
            parquet_filepath=str(tmp_path / "pq" / "d.parquet"),
        )
    printed = io.StringIO()
    with (
        contextlib.redirect_stdout(printed),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        pq = DatasetDescription.describe_dataset(
            str(tmp_path / "pq" / "d_0.parquet"), coords=["x0", "x1", "x2"]
        )
    pd.testing.assert_frame_equal(gen.description[COLUMNS], pq[COLUMNS])
    messages = [str(w.message) for w in caught]
    assert any(m.startswith("var1 is ASSUMED constant along ['x2']") for m in messages)
    assert any(
        m.startswith("var2 is ASSUMED constant along ['x1', 'x2']") for m in messages
    )
    assert "WARNING: dims of var1 are not certain" in printed.getvalue()


def known_parquet(tmp_path):
    """``known`` as ParquetBuilder writes it: var1 repeated along x0.

    var1 at x1 = 3 matches no var0 row, so its row has x0 = NaN.
    """
    frame = pd.DataFrame(
        {
            "x0": [10, 20, np.nan],
            "x1": [1, 2, 3],
            "var0": [1.0, 1.0, np.nan],
            "var1": [np.nan, 2.0, 3.0],
        }
    )
    path = str(tmp_path / "d.parquet")
    frame.to_parquet(path)
    return path


def test_reference_passed(known, tmp_path):
    """var1 as reference: proj(var1) = {2, 3}, var0 on x1 = {1, 2}.

    -> overlap         = |{2, 3} & {1, 2}| / |{2, 3}| = 0.5
    -> overlap_reverse = |{2, 3} & {1, 2}| / |{1, 2}| = 0.5
    """
    nc = DatasetDescription.describe_dataset(known, reference="var1")
    pq = DatasetDescription.describe_dataset(
        known_parquet(tmp_path), coords=["x0", "x1"], reference="var1"
    )
    for table in (nc, pq):
        assert list(table.index) == ["var1", "var0"]
        assert np.isnan(table.loc["var1", "overlap"])
        assert table.loc["var0", "overlap"] == pytest.approx(0.5)
        assert table.loc["var0", "overlap_reverse"] == pytest.approx(0.5)


def test_reference_inferred_from_occupied_sites(known, tmp_path):
    """var1 gains a third site, so it becomes the reference although listed second."""
    known["var1"][0] = 1.0
    table = DatasetDescription.describe_dataset(known)
    assert list(table.index) == ["var1", "var0"]
    assert np.isnan(table.loc["var1", "overlap"])


def test_unknown_reference_raises(known, tmp_path):
    with pytest.raises(ValueError):
        DatasetDescription.describe_dataset(known, reference="var9")
    with pytest.raises(ValueError):
        DatasetDescription.describe_dataset(
            known_parquet(tmp_path), coords=["x0", "x1"], reference="var9"
        )


def test_parquet_needs_coords(tmp_path):
    with pytest.raises(ValueError):
        DatasetDescription.describe_dataset(str(tmp_path / "d.parquet"))


def test_serial_and_parallel_describe_the_same_data(tmp_path):
    params = dict(
        num_obs=200,
        num_dims=3,
        ratio_dims=[1, 2, 2],
        density=0.4,
        seed=7,
        num_vars=2,
        var_dims=[3, 2],
        overlap=0.5,
    )
    tables = []
    for tag, max_obs in (("s", None), ("p", 80)):
        out = tmp_path / tag
        with contextlib.redirect_stdout(io.StringIO()):
            gen = GenerateData(**params, max_obs=max_obs)
            gen.generate(
                netcdf_filepath=str(out / "d.nc"),
                parquet_filepath=str(out / "pq" / "d.parquet"),
                parquet_tmp=str(out / "tmp"),
            )
        tables.append(gen.description)
    assert gen.NTASKS > 1
    pd.testing.assert_frame_equal(tables[0], tables[1])


def test_constant_full_dims_variable_keeps_its_dims(tmp_path):
    """A constant has one value at every site, but misses rows held by var0."""
    with contextlib.redirect_stdout(io.StringIO()):
        gen = GenerateData(
            num_obs=150,
            num_dims=3,
            ratio_dims=[1, 1, 1],
            density=[0.2, 0.2],
            seed=7,
            num_vars=2,
            dtype=["float64", "int16"],
            value_range=[None, (5, 5)],
        )
        gen.generate(
            netcdf_filepath=str(tmp_path / "d.nc"),
            parquet_filepath=str(tmp_path / "pq" / "d.parquet"),
        )
    pq = DatasetDescription.describe_dataset(
        str(tmp_path / "pq" / "d_0.parquet"), coords=["x0", "x1", "x2"]
    )
    assert pq.loc["var1", "dims"] == ("x0", "x1", "x2")
    pd.testing.assert_frame_equal(gen.description[COLUMNS], pq[COLUMNS])


def line_parquet(tmp_path, var1):
    """var0 on six cells; three of them on the line (x0, x1) = (0, 0)."""
    frame = pd.DataFrame(
        {
            "x0": [0, 0, 0, 0, 1, 1],
            "x1": [0, 0, 0, 1, 0, 0],
            "x2": [0, 4, 8, 2, 5, 6],
            "var0": [1.3, 2.1, 0.5, 0.9, 0.2, 0.7],
            "var1": var1,
        }
    )
    path = str(tmp_path / "d.parquet")
    frame.to_parquet(path)
    return path


@pytest.mark.parametrize(
    "var1, dims",
    [
        # one value on every row at each (x0, x1), 7 on three rows: assumed
        # constant along x2
        ([7.0, 7.0, 7.0, 4.0, 3.0, 3.0], ("x0", "x1")),
        # missing from a row at (0, 0): varies along x2
        ([7.0, 7.0, np.nan, 4.0, 3.0, 3.0], ("x0", "x1", "x2")),
    ],
)
def test_dims_from_rows(tmp_path, var1, dims):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(io.StringIO()):
            pq = DatasetDescription.describe_dataset(
                line_parquet(tmp_path, var1), coords=["x0", "x1", "x2"]
            )
    assert pq.loc["var1", "dims"] == dims
    assumed = [
        str(w.message)
        for w in caught
        if str(w.message).startswith("var1 is ASSUMED constant")
    ]
    assert bool(assumed) == (dims == ("x0", "x1"))


def test_constant_read_as_varying(tmp_path):
    """One value on every row: fits var1 on any dims, so every dim is kept."""
    with (
        pytest.warns(UserWarning, match="var1 is ASSUMED to vary along"),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        pq = DatasetDescription.describe_dataset(
            line_parquet(tmp_path, [7.0] * 6), coords=["x0", "x1", "x2"]
        )
    assert pq.loc["var1", "dims"] == ("x0", "x1", "x2")


def test_length_one_coord_kept(tmp_path):
    """x0 holds one value: kept among the dims, as netCDF keeps it."""
    path = line_parquet(tmp_path, [7.0, 7.0, 7.0, 4.0, 3.0, 3.0])
    # x1 numbers the (x0, x1) lines, so var1 keeps one value per line
    frame = pd.read_parquet(path)
    frame.assign(x1=2 * frame["x0"] + frame["x1"], x0=0).to_parquet(path)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pq = DatasetDescription.describe_dataset(path, coords=["x0", "x1", "x2"])
    assert pq.loc["var1", "dims"] == ("x0", "x1")
