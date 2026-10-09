"""Tests for DatasetDescription."""

import contextlib
import io

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from data_sparsity.generate_data import GenerateData
from data_sparsity.output import DatasetDescription


@pytest.fixture(name="known")
def fixture_known():
    """var0 on (x0, x1), var1 on x1 only.

    var0 occupies x1 = 0, 1; var1 occupies x1 = 1, 2
    -> F1 = |{0, 1} & {1, 2}| / |{0, 1}| = 0.5
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
    assert table.loc["var0", "own_grid"] == 6
    assert table.loc["var0", "occupied"] == 2
    assert table.loc["var0", "unused"] == 1  # x1 = 3
    assert np.isnan(table.loc["var0", "f1"])
    assert table.loc["var1", "dims"] == ("x1",)
    assert table.loc["var1", "density"] == pytest.approx(2 / 3)
    assert table.loc["var1", "f1"] == pytest.approx(0.5)


def test_parquet_agrees_with_netcdf(known, tmp_path):
    """Rows as ParquetBuilder writes them: var1 pinned to x0 = 10."""
    frame = pd.DataFrame(
        {
            "x0": [10, 10, 10, 20],
            "x1": [1, 2, 3, 2],
            "var0": [1.0, np.nan, np.nan, 1.0],
            "var1": [np.nan, 2.0, 3.0, np.nan],
        }
    )
    path = str(tmp_path / "d.parquet")
    frame.to_parquet(path)
    nc = DatasetDescription.describe_dataset(known)
    pq = DatasetDescription.describe_dataset(path, coords=["x0", "x1"])
    columns = ["dims", "own_grid", "occupied", "density", "f1", "unused"]
    pd.testing.assert_frame_equal(nc[columns], pq[columns])


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
