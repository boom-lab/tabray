"""Tests for VariableEncoding.

The encodings mirror two real files in the repository: GLORYS12 packs every
variable to int16 with scale_factor/add_offset, Argo writes plain float32 with
a 99999.0 sentinel. Both must be expressible, and so must plain float64.
"""

import numpy as np
import pytest
import xarray as xr

from data_sparsity.output import VariableEncoding


class TestDtypeAndPackAreSeparate:
    """dtype says what the variable holds; pack says how it is stored."""

    @pytest.mark.parametrize("dtype", ["float64", "float32"])
    def test_supported_dtypes(self, dtype):
        assert VariableEncoding(dtype).dtype == dtype

    def test_case_insensitive(self):
        assert VariableEncoding("Float32").dtype == "float32"

    def test_unsupported_dtype_raises(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            VariableEncoding("float16")

    @pytest.mark.parametrize("dtype", ["int8", "int16", "int32"])
    def test_integer_dtype_holds_integers(self, dtype):
        """An integer dtype is a variable of counts or flags, not a packing
        instruction. Argo's CYCLE_NUMBER is a plain int32."""
        enc = VariableEncoding(dtype)
        assert enc.integer is True
        assert enc.packed is False
        assert enc.storage_dtype == dtype

    def test_packing_an_integer_is_refused(self):
        """There is nothing to compact: the values are already integers."""
        with pytest.raises(ValueError, match="already holds integers"):
            VariableEncoding("int16", "int8")

    @pytest.mark.parametrize("pack", ["int8", "int16", "int32"])
    def test_pack_types(self, pack):
        enc = VariableEncoding("float64", pack)
        assert enc.packed is True
        assert enc.storage_dtype == pack

    def test_unpacked_by_default(self):
        enc = VariableEncoding("float32")
        assert enc.packed is False
        assert enc.storage_dtype == "float32"

    def test_pack_must_be_an_integer_type(self):
        with pytest.raises(ValueError, match="Cannot pack into"):
            VariableEncoding("float64", "float32")


class TestPacking:
    """The scale must leave the fill code unused."""

    def test_fill_code_is_reserved(self):
        """Mapping onto the full range puts the minimum ON the fill code.

        Those cells then read back as missing -- silent data loss. GLORYS
        avoids it too, reporting valid_min = -32760 rather than -32767.
        """
        enc = VariableEncoding("float64", "int16")
        lowest = round((enc.value_range[0] - enc.add_offset) / enc.scale_factor)
        assert lowest > enc.fill_value
        assert lowest == enc.fill_value + 1

    def test_range_maps_onto_the_top_code(self):
        enc = VariableEncoding("float64", "int16")
        highest = round((enc.value_range[1] - enc.add_offset) / enc.scale_factor)
        assert highest == np.iinfo("int16").max

    def test_scale_derives_from_declared_range_not_data(self):
        """A parallel chunk never sees the whole array, so it cannot measure."""
        a, b = VariableEncoding("float64", "int16"), VariableEncoding("float64", "int16")
        assert (a.scale_factor, a.add_offset) == (b.scale_factor, b.add_offset)

    def test_custom_value_range(self):
        enc = VariableEncoding("float64", "int16", value_range=(-3.0, 45.0))
        assert enc.scale_factor == pytest.approx(48.0 / 65533, rel=1e-6)

    def test_int32_keeps_float64_parameters(self):
        """Its step is finer than float32 spacing, so float32 would lose it."""
        assert VariableEncoding("float64", "int32").param_dtype is np.float64
        assert VariableEncoding("float64", "int16").param_dtype is np.float32


class TestQuantize:

    @staticmethod
    def values():
        rng = np.random.default_rng(0)
        v = np.full(500, np.nan)
        v[:300] = rng.uniform(0, 1, 300)
        v[0], v[1] = 0.0, 1.0          # the exact extremes
        return v

    def test_float64_is_unchanged(self):
        v = self.values()
        assert np.array_equal(VariableEncoding("float64").to_stored(v), v,
                              equal_nan=True)

    def test_nan_is_preserved(self):
        for pack in (None, "int16", "int8"):
            q = VariableEncoding("float64", pack).to_stored(self.values())
            assert np.array_equal(np.isnan(q), np.isnan(self.values()))

    def test_packed_values_land_within_half_a_step(self):
        enc = VariableEncoding("float64", "int16")
        v = self.values()
        err = np.nanmax(np.abs(enc.to_stored(v) - v))
        assert err <= enc.scale_factor / 2 + 1e-12

    def test_quantize_is_idempotent(self):
        """The second pass must not move anything, or the formats disagree."""
        enc = VariableEncoding("float64", "int16")
        once = enc.to_stored(self.values())
        assert np.array_equal(enc.to_stored(once), once, equal_nan=True)

    def test_does_not_mutate_its_argument(self):
        v = self.values()
        before = v.copy()
        VariableEncoding("float64", "int16").to_stored(v)
        assert np.array_equal(v, before, equal_nan=True)


class TestNetCDFEncoding:

    def test_default_says_nothing(self):
        """float64 with NaN is what xarray does anyway; adding an encoding
        entry changes its write path for no benefit."""
        assert VariableEncoding("float64").netcdf_encoding() == {}

    def test_float32_sets_dtype(self):
        assert VariableEncoding("float32").netcdf_encoding() == {"dtype": "float32"}

    def test_sentinel_fill_is_carried(self):
        """The Argo idiom: a real value marking absence, not NaN."""
        enc = VariableEncoding("float32", fill_value=99999.0).netcdf_encoding()
        assert enc["_FillValue"] == np.float32(99999.0)

    def test_packed_carries_scale_offset_and_fill(self):
        enc = VariableEncoding("float64", "int16").netcdf_encoding()
        assert enc["dtype"] == "int16"
        assert enc["_FillValue"] == np.int16(-32767)
        assert enc["scale_factor"].dtype == np.float32
        assert enc["add_offset"].dtype == np.float32


class TestPandasDtype:
    """What parquet stores: the decoded type, not the packed one."""

    @pytest.mark.parametrize("dtype,pack,expected", [
        ("float64", None, "float64"),
        ("float32", None, "float32"),
        ("float64", "int16", "float32"),
        ("float64", "int8", "float32"),
        ("float64", "int32", "float64"),
    ])
    def test_decoded_types(self, dtype, pack, expected):
        assert VariableEncoding(dtype, pack).pandas_dtype() == expected


class TestPerVariable:

    def test_scalar_applies_to_all(self):
        encs = VariableEncoding.per_variable("float64", "int16", None, 3)
        assert len(encs) == 3 and all(e.pack == "int16" for e in encs)

    def test_sequence_gives_one_each(self):
        encs = VariableEncoding.per_variable(
            ["float64", "float64", "float32"], ["int16", None, None],
            [None, None, 99999.0], 3)
        assert [e.storage_dtype for e in encs] == ["int16", "float64", "float32"]
        assert encs[2].fill_value == 99999.0

    def test_none_means_float64(self):
        assert all(e.dtype == "float64"
                   for e in VariableEncoding.per_variable(None, None, None, 2))

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="one per variable"):
            VariableEncoding.per_variable(["float64"], None, None, 3)


class TestRoundTripThroughNetCDF:
    """The encodings must survive a real write and read."""

    @staticmethod
    def values():
        rng = np.random.default_rng(1)
        v = np.full((50, 50), np.nan)
        idx = rng.choice(v.size, 800, replace=False)
        v.flat[idx] = rng.uniform(0, 1, 800)
        v.flat[idx[0]], v.flat[idx[1]] = 0.0, 1.0
        return v

    @pytest.mark.parametrize("dtype,pack,fill,on_disk,in_memory", [
        ("float64", None, None, "float64", "float64"),
        ("float32", None, None, "float32", "float32"),
        ("float32", None, 99999.0, "float32", "float32"),
        ("float64", "int16", None, "int16", "float32"),
        ("float64", "int8", None, "int8", "float32"),
        ("float64", "int32", None, "int32", "float64"),
    ])
    def test_round_trip(self, tmp_path, dtype, pack, fill, on_disk, in_memory):
        enc = VariableEncoding(dtype, pack, fill)
        values = enc.to_stored(self.values())
        da = xr.DataArray(values, dims=("y", "x"), name="v")
        da.encoding = enc.netcdf_encoding()
        path = tmp_path / f"{dtype}_{pack}_{fill}.nc"
        da.to_netcdf(path)

        with xr.open_dataset(path, mask_and_scale=False) as raw:
            assert str(raw["v"].dtype) == on_disk
        with xr.open_dataset(path) as back:
            assert str(back["v"].dtype) == in_memory
            read = back["v"].values.astype("float64")

        lost = (~np.isnan(values)) & np.isnan(read)
        assert not lost.any(), f"{lost.sum()} values read back as missing"
        assert np.array_equal(np.isnan(read), np.isnan(values))
        assert np.nanmax(np.abs(read - values)) == pytest.approx(0, abs=1e-6)


class TestIntegerValues:
    """Integer variables: what they hold, and how the values are made."""

    @staticmethod
    def draws(n=200_000):
        """Raw uniform(0, 1), which is all ObservationGenerator ever makes."""
        u = np.full(n + 1, np.nan)
        u[:n] = np.random.default_rng(0).uniform(0, 1, n)
        return u

    def test_default_range(self):
        assert VariableEncoding("int16").value_range == (0, 100)

    def test_values_stay_inside_the_range(self):
        enc = VariableEncoding("int16", value_range=(0, 8))
        out = enc.to_stored(self.draws())
        assert np.nanmin(out) == 0 and np.nanmax(out) == 8

    def test_values_are_whole_numbers(self):
        out = VariableEncoding("int16", value_range=(3, 11)).to_stored(self.draws())
        assert np.array_equal(out[~np.isnan(out)] % 1, np.zeros(200_000))

    def test_every_value_in_the_range_is_equally_likely(self):
        """floor over (hi - lo + 1) bins, not round over (hi - lo).

        Rounding gives the two end bins half the width of the others, so the
        extremes come out at half the frequency -- a quiet bias in any
        histogram of the result.
        """
        enc = VariableEncoding("int16", value_range=(0, 8))
        out = enc.to_stored(self.draws())
        counts = np.bincount(out[~np.isnan(out)].astype(int), minlength=9)
        share = counts / counts.sum()
        assert share.min() > 0.9 / 9 and share.max() < 1.1 / 9

    def test_range_width_sets_cardinality(self):
        """The knob that matters for storage: a narrow range is flag-like
        data, which both formats encode very differently from noise."""
        narrow = VariableEncoding("int16", value_range=(0, 4)).to_stored(self.draws())
        wide = VariableEncoding("int32", value_range=(0, 50_000)).to_stored(self.draws())
        assert len(np.unique(narrow[~np.isnan(narrow)])) == 5
        assert len(np.unique(wide[~np.isnan(wide)])) > 10_000

    def test_nan_is_preserved(self):
        out = VariableEncoding("int16").to_stored(self.draws())
        assert np.isnan(out[-1])

    def test_placement_does_not_depend_on_dtype(self):
        """The reason integers transform the uniform draw rather than calling
        rng.integers: a different RNG method consumes the stream differently,
        which would move the occupied sites and compare different datasets."""
        u = self.draws(1000)
        for enc in (VariableEncoding("float64"), VariableEncoding("int16"),
                    VariableEncoding("float64", "int16")):
            assert np.array_equal(np.isnan(enc.to_stored(u)), np.isnan(u))

    def test_fill_defaults_outside_the_range(self):
        assert VariableEncoding("int16").fill_value == -32767

    def test_fill_inside_the_range_raises(self):
        with pytest.raises(ValueError, match="lies inside value_range"):
            VariableEncoding("int16", fill_value=5)

    def test_range_must_fit_the_dtype(self):
        with pytest.raises(ValueError, match="does not fit int8"):
            VariableEncoding("int8", value_range=(0, 1000))

    def test_inverted_range_raises(self):
        with pytest.raises(ValueError, match="max above min"):
            VariableEncoding("int16", value_range=(10, 2))

    def test_parquet_column_is_nullable(self):
        """A vacant site must be a real null; NaN would force it to float."""
        assert VariableEncoding("int16").pandas_dtype() == "Int16"

    def test_netcdf_encoding(self):
        enc = VariableEncoding("int32", fill_value=99999).netcdf_encoding()
        assert enc == {"dtype": "int32", "_FillValue": np.int32(99999)}


class TestFloatValueRange:
    """value_range spreads a float variable too, and drives its packing."""

    def test_default_leaves_the_draws_alone(self):
        u = np.random.default_rng(0).uniform(0, 1, 100)
        assert np.array_equal(VariableEncoding("float64").to_stored(u), u)

    def test_custom_range_spreads_the_values(self):
        u = np.array([0.0, 0.5, 1.0])
        out = VariableEncoding("float64", value_range=(-3.0, 45.0)).to_stored(u)
        assert np.allclose(out, [-3.0, 21.0, 45.0])

    def test_packing_uses_the_same_range(self):
        """Derived from one range, so a value cannot fall outside the grid
        that packs it."""
        enc = VariableEncoding("float64", "int16", value_range=(-3.0, 45.0))
        u = np.array([0.0, 0.5, 1.0])
        out = enc.to_stored(u)
        assert np.abs(out - np.array([-3.0, 21.0, 45.0])).max() < enc.scale_factor
