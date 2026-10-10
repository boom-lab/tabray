# Per-variable encoding: why it works the way it does

`.claude/CLAUDE.md` lists the rules. This file gives the reasons.

## Why per variable, not a mode

Scientific netCDF has no single convention. Two real files in the repo disagree on nearly everything:

| | GLORYS12 (`GLOBAL_MULTIYEAR_PHY_001_030/`) | Argo Sprof (`argo/`) |
|---|---|---|
| dtype | `int16` packed, scale + offset | `float32` plain |
| missing | `-32767` | `99999.0`, not NaN |
| compression | zlib **1**, no shuffle | zlib **4**, with shuffle |
| coordinates | `float32` | `float64` |

## Why `dtype` and `pack` are separate

`dtype` is what a variable holds; `pack` is how a float is compacted on disk. One file holds both
kinds: Argo stores `TEMP` as a plain `float32` and `CYCLE_NUMBER` as a plain `int32`.

## `value_range`

- Packed float: sets the packing scale, so a value cannot land outside its own grid.
- Integer: the width is the column's cardinality. `(0, 8)` gives flag-like data that dictionary-
  and run-length-encodes; a wide range behaves like noise.

## Integers from the uniform draw

`lo + floor(u * (hi - lo + 1))`, applied to `ObservationGenerator`'s `uniform(0, 1)` draw.

- Not `rng.integers`: a different RNG method consumes the stream differently, so a dtype change
  would shift every later draw and move the occupied sites. With the transform, dtype changes
  values and leaves placement identical.
- `floor` over `hi - lo + 1` bins: rounding over `hi - lo` gives the end bins half width, so the
  extremes appear at half the frequency.

## Strings

`S<k>` draws integer codes through the same transform, so a string variable occupies the same
sites as an integer one. The codes become text only on write (`VariableEncoding.to_text`):
base 36, zero-padded to k characters, so every value has exactly k characters in both formats.

- netCDF: `S1` characters along a `string<k>` dimension, `_FillValue = " "`, as Argo stores
  `PLATFORM_NUMBER`. A vacant site goes in as NaN, which xarray writes as the fill followed by
  null bytes and reads back as NaN. A string of k spaces would read back as a value.
- `merge_nc`: strings read back from chunk files are objects, which dask loads whole to find a
  width; `NetCDFBuilder.fix_text_width` casts them to `S<k>` first, with the same bytes on disk.
- parquet: pyarrow-backed strings (`string[pyarrow]`), one buffer per column. Python-object
  strings would cost one object per row in memory.
- The data `generate()` returns keep the codes in the Dataset; the DataFrame holds the text.

## Constants

`min == max` gives every occupied site the same value: the integer transform maps every draw to
`lo`, the float transform multiplies by a zero span. A packed constant takes `scale_factor = 1`.
Parquet description reads a constant as dropping every coordinate (`()`, one site): a value that
never changes carries no position.

## Packing

- The fill code is reserved. Mapping the value range onto the full integer range puts the minimum
  value on `_FillValue`, and those cells read back as missing. GLORYS reports
  `valid_min = -32760` for this reason.
- The dtype of `scale_factor`/`add_offset` decides what xarray decodes to. `float32` parameters
  give a `float32` array instead of a promoted `float64` one. `int32` keeps `float64` parameters:
  its step is finer than `float32` spacing.

## Integer fill and parquet

A plain integer variable is a nullable parquet column (`Int16`), so a vacant site is a null and
the column stays integer. Its fill value must sit outside `value_range`: the same collision the
packing scale avoids by reserving a code.

## Quantising once

`VariableEncoding.to_stored` runs once (`GenerateData._to_stored` and its worker counterpart),
before either format is written, so netCDF and parquet hold the same numbers
and the serial/parallel digest check stays valid. The scale comes from
`VariableEncoding.VALUE_RANGE`, not from the data, because a parallel chunk never sees the whole
array.
