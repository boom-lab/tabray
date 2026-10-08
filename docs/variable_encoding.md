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
