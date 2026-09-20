"""Tests for CompressionSettings."""

import numpy as np
import pytest
import xarray as xr

from data_sparsity.output import CompressionSettings


class TestCodecValidation:
    """Only codecs both formats can honour are accepted."""

    @pytest.mark.parametrize("codec", [None, "none", "off", "NONE"])
    def test_off_spellings(self, codec):
        assert CompressionSettings(codec).enabled is False
        assert CompressionSettings(codec).codec is None

    @pytest.mark.parametrize("codec", ["gzip", "zlib", "deflate", "GZIP"])
    def test_deflate_spellings(self, codec):
        """netCDF calls it zlib, parquet calls it gzip; same algorithm."""
        settings = CompressionSettings(codec)
        assert settings.enabled is True
        assert settings.codec == "gzip"

    @pytest.mark.parametrize("codec", ["snappy", "zstd", "lz4", "brotli"])
    def test_parquet_only_codecs_are_refused(self, codec):
        """Accepting one would write the two formats on different terms."""
        with pytest.raises(ValueError, match="not to netCDF"):
            CompressionSettings(codec)

    def test_unknown_codec_raises(self):
        with pytest.raises(ValueError, match="Unknown compression"):
            CompressionSettings("bogus")

    @pytest.mark.parametrize("level", [0, 10, -1])
    def test_level_out_of_range_raises(self, level):
        with pytest.raises(ValueError, match="complevel"):
            CompressionSettings("gzip", level)

    def test_level_ignored_when_off(self):
        """No codec, no level to validate."""
        assert CompressionSettings(None, 99).enabled is False


class TestNetCDFEncoding:
    """The encoding handed to to_netcdf."""

    @staticmethod
    def dataset():
        return xr.Dataset(
            {"var0": ("x0", np.arange(3.0)), "var1": ("x0", np.arange(3.0))},
            coords={"x0": np.arange(3.0)},
        )

    def test_off_gives_empty_encoding(self):
        assert CompressionSettings(None).netcdf_encoding(self.dataset()) == {}

    def test_every_data_var_is_encoded(self):
        enc = CompressionSettings("gzip", 5).netcdf_encoding(self.dataset())
        assert set(enc) == {"var0", "var1"}
        assert enc["var0"] == {"zlib": True, "complevel": 5}

    def test_coords_are_left_alone(self):
        """Small arrays, where per-chunk overhead outweighs the saving."""
        enc = CompressionSettings("gzip").netcdf_encoding(self.dataset())
        assert "x0" not in enc

    def test_named_dataarray(self):
        da = xr.DataArray(np.arange(3.0), dims="x0", name="record")
        assert set(CompressionSettings("gzip").netcdf_encoding(da)) == {"record"}

    def test_unnamed_dataarray_gives_no_encoding(self):
        """Nothing to key the encoding on; writing must still work."""
        da = xr.DataArray(np.arange(3.0), dims="x0")
        assert CompressionSettings("gzip").netcdf_encoding(da) == {}


class TestParquetKwargs:
    """The keywords handed to to_parquet."""

    def test_off_passes_none_explicitly(self):
        """Omitting it would leave dask's own default, which is Snappy."""
        assert CompressionSettings(None).parquet_kwargs() == {"compression": None}

    def test_on_passes_codec_and_level(self):
        kwargs = CompressionSettings("gzip", 6).parquet_kwargs()
        assert kwargs == {"compression": "gzip", "compression_level": 6}


class TestEquality:
    def test_same_settings_are_equal(self):
        assert CompressionSettings("zlib", 4) == CompressionSettings("gzip", 4)

    def test_level_matters_when_enabled(self):
        assert CompressionSettings("gzip", 4) != CompressionSettings("gzip", 5)

    def test_level_does_not_matter_when_off(self):
        assert CompressionSettings(None, 1) == CompressionSettings(None, 9)

    def test_repr_says_which(self):
        assert repr(CompressionSettings(None)) == "CompressionSettings(none)"
        assert "gzip" in repr(CompressionSettings("gzip"))


class TestCompressionReachesBothFiles:
    """End to end: the codec must land in both files, and change no data.

    The point of the parameter is that the two formats are comparable, so a
    test that only checked one of them would miss the failure it exists for.
    """

    @staticmethod
    def generate(tmp_path, codec):
        import contextlib
        import io

        from data_sparsity.generate_data import GenerateData

        out = tmp_path / str(codec)
        (out / "nc").mkdir(parents=True)
        (out / "pq").mkdir(parents=True)
        with contextlib.redirect_stdout(io.StringIO()):
            gen = GenerateData(num_obs=2000, num_dims=2, ratio_dims=(1, 1),
                               density=0.2, seed=3, compression=codec)
            gen.generate(netcdf_filepath=str(out / "nc" / "d.nc"),
                         parquet_filepath=str(out / "pq" / "d.parquet"))
        return out

    @staticmethod
    def netcdf_filters(out):
        import netCDF4

        with netCDF4.Dataset(out / "nc" / "d.nc") as ds:
            name = "record" if "record" in ds.variables else "var0"
            return ds.variables[name].filters() or {}

    @staticmethod
    def parquet_codec(out):
        import glob

        import pyarrow.parquet as pq

        first = sorted(glob.glob(str(out / "pq" / "*.parquet")))[0]
        return pq.ParquetFile(first).metadata.row_group(0).column(0).compression

    def test_uncompressed_by_default(self, tmp_path):
        """Dask would otherwise write Snappy parquet beside plain netCDF."""
        out = self.generate(tmp_path, None)
        assert not self.netcdf_filters(out).get("zlib")
        assert self.parquet_codec(out) == "UNCOMPRESSED"

    def test_gzip_reaches_both(self, tmp_path):
        out = self.generate(tmp_path, "gzip")
        filters = self.netcdf_filters(out)
        assert filters.get("zlib") is True
        assert filters.get("complevel") == 4
        assert self.parquet_codec(out) == "GZIP"

    def test_compression_changes_no_data(self, tmp_path):
        import pandas as pd

        plain, gzipped = (self.generate(tmp_path, c) for c in (None, "gzip"))
        with xr.open_dataset(plain / "nc" / "d.nc") as a, \
                xr.open_dataset(gzipped / "nc" / "d.nc") as b:
            xr.testing.assert_identical(a, b)
        pd.testing.assert_frame_equal(
            pd.read_parquet(plain / "pq", engine="pyarrow"),
            pd.read_parquet(gzipped / "pq", engine="pyarrow"),
        )

    def test_gzip_is_smaller(self, tmp_path):
        """A sparse grid is mostly one repeated NaN pattern."""
        import os

        plain = self.generate(tmp_path, None) / "nc" / "d.nc"
        gzipped = self.generate(tmp_path, "gzip") / "nc" / "d.nc"
        assert os.path.getsize(gzipped) < os.path.getsize(plain)
