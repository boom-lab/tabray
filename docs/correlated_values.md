# Correlated values: why and how

`.claude/CLAUDE.md` lists the rules. This file gives the reasons.

## Why

With the default `data_structure="uncorrelated"`, every value is an independent `uniform(0, 1)`
draw: white noise. Two consequences for a netCDF vs parquet comparison:

- **Size.** Neighbouring values share nothing, so no codec compresses them. Packed to int16 and
  written with zlib, the "compressed" netCDF is 99% of the uncompressed size.
- **Speed.** On incompressible input, deflate stores its blocks raw and snappy stores literals, so
  both decode at close to memory speed. On real data, zlib inflates at roughly 440 MB/s per core
  and snappy at 950 MB/s (single-core measurement on one smooth field).

So with white noise a benchmark measures neither the I/O that compression saves nor the decoding
it costs. Real fields change little between neighbouring cells, and that is what compression
exploits. `data_structure="correlated"` produces values like that.

## What the field is

For variable *v*, the value at a cell with global grid indices `i` comes from a sum of `M = 64`
random plane waves (random Fourier features):

```
S(i) = sqrt(2/M) * sum_m cos( sum_d k[m,d] * i_d + phi[m] )
z    = sqrt(1 - noise^2) * S(i) + noise * ndtri(u)
u'   = ndtr(z)
```

- The phases `phi[m]` are uniform on `(0, 2 pi)`. The wavenumber `k[m,d]` is drawn from
  `N(0, 1/L_d^2)`.
- Averaged over the random waves, `S` then has the Gaussian autocorrelation
  `exp(-1/2 * sum_d (lag_d / L_d)^2)`. Cells `L_d` apart along dim `d` have correlation 0.61,
  and cells `3 L_d` apart have almost none.
- One realisation scatters around that average by about `1/sqrt(M)`, and with long lengths its
  pattern can look like interference fringes. That is enough to set how much a value changes
  between neighbouring cells, which is what compression responds to.
- `S` is close to a standard normal, so `ndtr`, the normal CDF, maps it back to `uniform(0, 1)`.
  The distribution of values is the same as for uncorrelated values; only the dependence between
  cells changes. As a result `value_range`, integer dtypes, strings and packing apply unchanged,
  and a `stats` task returns comparable results either way.
- `ndtr` slightly lowers the correlation: a normal correlation `rho` becomes
  `(6/pi) arcsin(rho/2)`, at most 0.02 lower.

### `correlation_length`

The length is measured in grid cells (index space), not in coordinate units, for two reasons:

- Compression depends on the neighbours in storage order.
- In a scaling experiment where one axis grows, a length in cells keeps every chunk equally
  compressible. A length in coordinate units would make the field smoother as the axis grows
  and mix that into the size trend.

Each entry of `correlation_length` is one of:

- a number of cells, at least 1;
- `0`: the wavenumbers along that dim are drawn uniformly on `(-pi, pi)`, the whole band an
  integer grid can hold, so cells along it are uncorrelated;
- `None`: the wavenumbers are 0, so the value is constant along that dim.

Lengths between 0 and 1 are rejected. Their Gaussian spectrum reaches past the highest
wavenumber the grid can hold and folds back, so the field is no longer Gaussian-correlated. A
length longer than its axis gives a warning, because the values are then nearly constant along
that axis. A variable's constant (pinned) dims get no wavenumber.

### `noise`

`noise` is the standard deviation of white noise as a share of the total. It reuses the existing
uniform draw through `ndtri`. Real data carry measurement and model noise far below the
variable's range, and at int16 resolution that noise sets how well the data compress.

## Effect on compression

Measured on a grid of 4 x 20 x 100 x 100 at density 1, int16-packed and rewritten with zlib
level 1 + shuffle, one `x0` step per HDF5 chunk. The table gives the compressed size as a share
of the uncompressed size.

| `data_structure` | `correlation_length` (x0, x1, x2, x3) | `noise` | size |
|---|---|---|---|
| uncorrelated | – | – | 1.01 |
| correlated | default (10 on every dim) | 0 | 0.88 |
| correlated | None, 2, 50, 50 | 0 | 0.70 |
| correlated | None, 2, 200, 200 | 0 | 0.57 |
| correlated | None, 200, 200, 2 | 0 | 0.74 |
| correlated | None, 2, 200, 200 | 0.01 | 0.75 |
| correlated | None, 2, 200, 200 | 0.1 | 0.91 |

What the table shows:

- **The contiguous axis matters most.** `x3` is contiguous on disk, and a long length along it
  matters most. Along `x0`, nothing changes while each chunk holds one `x0` step, because chunks
  are compressed independently.
- **Even a little noise erases much of the gain.** `noise=0.01` already costs more than half of
  the gain from a long length.

## Determinism

`u'` depends only on `(seed, variable, global grid indices, u)`:

- The waves come from `stream(seed, Stream.CORRELATED_FIELD, var_idx)`. Every draw is made in a
  fixed order, whatever the lengths, so changing one dim's length leaves the waves along the
  other dims unchanged.
- The field is applied in `MultiVarRecordGenerator.generate` and `generate_without_overlap`
  while the indices are still global, before the split axis is rebased to the chunk. A
  parallel chunk therefore computes the same values as the serial run.
- `u` is still drawn, after the sites, on the same stream. Correlated and uncorrelated runs
  therefore occupy the same sites.
- Each cell's value comes from operations on that cell only, one wave at a time with dims added
  in a fixed order. Serial and parallel results are bit-identical, whatever the length of the
  array being evaluated. A matrix product could reorder the sums with the array size.

## Cost

There are 64 cosines per occupied cell. They are taken in float32, after reducing the angle to
`[-pi, pi]` in float64. That is about seven times faster than in float64, and the values move by
under 1e-7, far below any packing step. Measured cost: about 3 s per 10^7 cells on 4 varying
dims, on one core. A dim with length `None` costs nothing.
