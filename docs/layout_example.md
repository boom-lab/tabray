# How a dataset is generated: scattered against padded

One small two-variable call, generated twice, once per layout. Every number and
grid below is the output of a run on `feat/add-data-occupancy-layout`, with
equal weights for the padded counts.

```python
GenerateData(num_obs=12, num_dims=3, ratio_dims=[1, 2, 2],
             density=[0.5, 0.25], seed=7, num_vars=2, var_dims=[3, 3],
             layout="scattered")   # then layout="padded"
```

## 1. Validation and configuration (same for both layouts)

| step | result |
|---|---|
| grid from `num_obs`, `density`, `ratio_dims` | shape `[2, 4, 4]` (x0, x1, x2), 32 sites |
| `num_obs` rounded to the grid | 12 -> 16 (`adjusted` in the report) |
| per-variable counts, density x own grid | var0 0.5 x 32 = **16**, var1 0.25 x 32 = **8** (both on all 3 dims) |
| split dimension (strata) | x1: 4 strata, `x1 = 0..3`, each a 2 x 4 plane (x0 rows, x2 columns) |
| padded axis (padded only) | x2, the last dim; a **line** is one x0 value inside a stratum, so 2 lines of 4 cells per stratum |
| minimum density for var0 | scattered `max(shape)/sites` = 4/32 = 0.125; padded `(lines + n_padded - 1)/sites` = (8 + 4 - 1)/32 = 0.344. var0's 0.5 passes both |

## 2. Scattered

**Stage A -- LHS (var0).** `max(shape)` = 4 points from `stream(seed, LHS)`,
every coordinate of every axis used once at least:

```
(x0, x1, x2) = (0, 1, 1), (1, 0, 3), (1, 2, 2), (0, 3, 0)
```

One LHS point falls in each stratum.

**Stage B -- fill counts (var0).** The other 16 - 4 = 12 observations are
apportioned over the strata by free sites: 3 per stratum. Total per stratum:
1 LHS + 3 fill = **4, 4, 4, 4**.

**Stage C -- per stratum, var0.** The 3 fill sites are drawn uniformly from the
stratum's free cells, from `stream(seed, STRATUM, j)`.

**Stage D -- per stratum, var1.** 8 apportioned over 4 equal planes: **2, 2, 2, 2**.
With `overlap="random"` each stratum draws its 2 cells uniformly from the whole
plane, from `stream(seed, VAR, 1, j)`; they may land on var0's cells or not.

Result (`A` var0 only, `B` var1 only, `X` both, `.` empty; lower case marks
var0's LHS point):

```
stratum x1=0        stratum x1=1        stratum x1=2        stratum x1=3
      x2: 0 1 2 3         x2: 0 1 2 3         x2: 0 1 2 3         x2: 0 1 2 3
x0=0      . A . A   x0=0      A a . A   x0=0      X . . A   x0=0      a A . B
x0=1      . X . x   x0=1      . A B B   x0=1      A . x .   x0=1      X A . .
```

## 3. Padded (x2)

**No LHS stage.** Coverage comes from the construction instead: every line
holds at least its cell `x2 = 0`, and one line fills all 4 cells of x2.

**Stage B -- counts per stratum (var0), `padded_stratum_counts`.** A floor of
one observation per line (2 per stratum, 8 in all); the other 8 apportioned
with equal weights, 2 per stratum, capped at 2 lines x 4 cells: 4, 4, 4, 4.
The stratum with the largest count (the first, on a tie) carries the
full-length line; it needs 4 + 2 - 1 = 5, so it is raised by one and the
difference is taken from the others' slack:

```
var0 per stratum: 5, 4, 4, 3      full-length line in stratum x1=0
```

**Stage C -- per stratum, var0, `_first_occupied_per_line`.** One cell per
line, the rest split with equal weights; the full-length line gets k = 4.
Each line fills its first k cells of x2:

| stratum | count | k for x0=0, x0=1 |
|---|---|---|
| x1=0 | 5 | 4, 1 |
| x1=1 | 4 | 2, 2 |
| x1=2 | 4 | 2, 2 |
| x1=3 | 3 | 1, 2 |

**Stage D -- per stratum, var1, `_padded_cells`.** Same counts as scattered,
2 per stratum. var1 varies along x2, so its 2 cells are split over the 2
lines with equal weights and each line fills its first k cells; a line may
stay empty, since coverage is var0's guarantee only.

Result:

```
stratum x1=0        stratum x1=1        stratum x1=2        stratum x1=3
      x2: 0 1 2 3         x2: 0 1 2 3         x2: 0 1 2 3         x2: 0 1 2 3
x0=0      X A A A   x0=0      X A . .   x0=0      X A . .   x0=0      X . . .
x0=1      X . . .   x0=1      X A . .   x0=1      X A . .   x0=1      X A . .
```

## 4. What differs

| | scattered | padded |
|---|---|---|
| var0 per stratum | 4, 4, 4, 4 | 5, 4, 4, 3 |
| where cells sit in a line | anywhere | positions 0..k-1 |
| var0 coverage from | LHS stage | one cell per line + one full line |
| var1 placement | uniform over the plane | first k cells of each line |
| overlap | target allowed | must be `"random"` (raises otherwise) |
| var0 minimum density | 0.125 | 0.344 |

Both runs give var0 16 and var1 8 observations (density 0.5 and 0.25, `match`
in the report), every coordinate of every axis used by var0, and the same files
from a serial and a parallel run.
