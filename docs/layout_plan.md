# Plan: the shape of the emptiness

`density` says how many sites are occupied. It says nothing about where they sit, and the
generator scatters them uniformly. Every real dataset in this repository is arranged some other
way: Argo and CrocoLake fill a prefix of each profile's levels, GLORYS leaves continents empty.

## Why it matters

Measured on a 2000x500 grid, same occupancy of 0.149, same number of occupied cells, three
arrangements:

```
layout      nc zlib   parquet   which is smaller   full nc read
scattered    1.15 M    1.00 M   parquet, 1.15x          157 ms
padded       0.55 M    0.86 M   netCDF,  1.56x           89 ms
masked       0.51 M    0.84 M   netCDF,  1.65x          118 ms
```

Two things follow. Compressed array size moves by 2.3x, because scattered fill interrupts the
runs DEFLATE collapses. And the answer to "which format is smaller" reverses. Since that answer
is what this package produces, a sweep run today describes a regime none of the three real files
occupies.

Working memory does not change: the array is dense once loaded and the row count is fixed by the
occupancy, so layout is a disk and decompression property. Filtered reads are the exception --
there the array slab is a fixed function of the window while the tabular result follows what is
actually in the window, so layout decides which queries are cheap.

## The parameter

Two orthogonal parameters rather than three layouts.

`layout="scattered"` (default, today's behaviour) or `"padded"` says how the data fills the cells
available to it. `mask_fraction` says which cells are available at all. They compose: a mask with
scattered fill is GLORYS, a mask with padded fill is a profile dataset confined to a region, and
no mask with padded fill is Argo.

On the names. CF's Discrete Sampling Geometries chapter reserves *ragged* for the contiguous and
indexed ragged array representations, which pack variable-length profiles into a 1-D array with a
count or index variable -- a layout this package does not produce. The padded two-dimensional
form that Argo uses is what CF calls the *incomplete multidimensional array representation*.
`padded` says what the array holds and collides with neither.

`aligned` and `clustered` would name all three layouts after the data rather than after the
absence, which is more consistent, but `aligned` already means index alignment in pandas and
sequence alignment in bioinformatics, and `clustered` implies several groups of similar points
where a mask leaves one connected region. `padded` and `masked` have one meaning each and it is
the right one. Both are general array-computing words -- `numpy.ma`, and padding a batch of
variable-length sequences -- rather than oceanographic ones.

## Padded

Implemented as the distribution the **fill** stage draws from inside a stratum. Today that draw
is uniform over the stratum's free cells; padded biases it to a prefix along one named axis, so
each stratum fills positions `0..k-1`.

This fits the stratum model without disturbing it. A stratum is already one profile's worth of
cells when the split runs along the profile axis, its count is already apportioned, and `k`
follows from `(seed, stratum)` -- so serial still equals parallel by construction. Density stays
exact because the layout chooses where the apportioned count goes, not how much.

**The padded axis cannot be the split dimension.** A stratum holds one index of the split
dimension, so a prefix along it is meaningless. `_choose_split_dim` picks the split axis on its
own, so the two can collide; the constructor should resolve or refuse it rather than generate
something silently wrong.

**Overlap must raise.** With both variables filling a prefix of the same axis, the intersection
in a stratum is `min(k_0, k_i)`, so `F1 = n_i / n_0` and nothing is left to choose. Argo confirms
this: NITRATE sits at occupancy 0.161 against TEMP's 0.986 and its measured F1 is 0.163, the
density ratio. Passing `overlap` alongside `layout="padded"` should raise, the way packing an
integer dtype does, because the target cannot be honoured. The achieved value still goes in the
generation report.

**Coverage comes from the layout, not from the LHS.** Keeping the LHS would place `max(shape)`
points wherever coverage demands, which is outside the prefix, leaving a padded dataset with a
thin scatter through it -- 0.6% of the cells on CrocoLake's shape, 5.1% on Argo's. A prefix
construction can guarantee coverage on its own, so the LHS stage is skipped under `padded`.

Within a stratum, call each combination of the remaining axes a **line**, and fill positions
`0..k-1` of the padded axis on each line. Coverage then needs two things:

* every line holds at least one observation, `k >= 1`. This covers the split axis and every axis
  other than the padded one, since each line is one combination of them.
* one line somewhere reaches the last coordinate, `k = n_padded`. This covers the padded axis.

Both are constraints on the vector of prefix lengths, which is what the layout is choosing
anyway, so coverage costs nothing extra and no cell sits outside the pattern.

The full-length line has to be chosen without the workers talking to each other. Take the stratum
with the largest apportioned count -- that vector is global, so every worker names the same one --
raise one of its lines to `n_padded`, and take the difference back from the other strata.

Raising rather than hoping the chosen stratum already affords `n_padded` is what makes this
feasible rather than probable. The redistribution is bounded: no donor may drop below its own
coverage floor, which is the number of lines it holds, so it is an apportionment over each
donor's slack above that floor and `ChunkUtils.apportion` already takes weights and capacities in
that shape. The total stays exact, and the donors follow from the global count vector plus one
seeded stream, so every worker computes the same correction. On CrocoLake's shape it moves about
887 observations of 310,516, or 0.3%, so the per-stratum counts stop being a pure apportionment
and carry a correction on top.

Two knock-ons, neither large but both real:

* **The minimum density rises.** Coverage under a prefix needs `n_lines + n_padded - 1`
  observations, against `max(shape)` for scattered. On Argo's shape that is 539 rather than 520;
  on CrocoLake's, 3,041 rather than 2,000. `compute_min_density` needs a padded variant, and the
  generation report should state which bound applied.
* **`stratum_counts` needs a padded variant.** It currently derives the per-stratum counts as the
  LHS points that fell in each stratum plus an apportioned fill. With no LHS the counts are a
  plain apportionment of `num_obs` over stratum capacity. The overlap apportionment reads that
  function, but overlap raises under `padded`, so nothing else depends on the change.

## The mask

`mask_fraction` excludes a fraction of the cells on a subset of dimensions, the same exclusion in
every stratum. Generate it by thresholding a smoothed random field over those dimensions, seeded
from `seed`, which gives blob-like shapes rather than rectangles. The field is defined over the
masked dimensions only, so it costs `prod(masked dims)` rather than the grid, and every worker
derives the same mask from the seed.

**`density` is measured against the unmasked cells.** Within a level GLORYS fills every wet cell:
`zos` and `thetao` at level 0 both read 69.71%, so it reads as `density=1.0,
mask_fraction=0.303`. Under the other reading the caller has to keep `density` in step with the
mask by hand, and any `density` above `1 - mask_fraction` becomes impossible rather than merely
unusual. Global density is `density x (1 - mask_fraction)`, which is deterministic, and the report
prints both because they have different denominators.

**Overlap stays free under a mask**, unlike padded. The mask removes cells from play but leaves
the choice of which remaining cells each variable takes. What `mask_fraction` fixes is the
achievable range: with `A = (1 - mask_fraction) x prod(shape)` cells available, F1 is bounded
below by `max(0, (n_0 + n_i - A) / n_0)` and above by `min(1, n_i / n_0)`. That lower bound is the
forced-overlap term the placement code already computes per stratum as
`lo = max(0, n_here - free_cells)`.

**Coverage applies to reachable coordinates.** A coordinate is reachable if the mask leaves it at
least one cell. The rule becomes *every reachable coordinate is used*, which is the existing rule
wherever there is no mask.

A mask can make a coordinate unreachable by emptying its whole row, and GLORYS does: 38 of its
2041 latitudes hold no ocean at the surface and 44 hold none at 2225 m, at the Antarctic and polar
ends of the range. Those do not fill in on another day the way the empty deepest level does --
an interior Antarctic latitude is land in every file.

GLORYS is right to keep them. Its grid is defined by its 1/12 degree spacing, not by where the
water is, and dropping those latitudes would make the axis irregular. That is a case
`docs/explainer.md` does not cover: its argument for requiring every coordinate to be used is that
an unused coordinate could be dropped for a smaller grid, which holds for an irregular grid and
fails for a regular one where the spacing is the thing being represented. **The explainer needs a
sentence for this**, and it is a change to its premise rather than to the code, so it should be
made deliberately.

So: `allow_unused_coordinates=False` by default, rejecting a mask that strands a coordinate.
Set it and such a mask is accepted, those coordinates count as unreachable, and the report names
them. Nothing other than a mask can produce one, since the LHS covers the rest.

**The LHS draws from unmasked cells**, whether or not the opt-in is set, or it would place
observations on land while trying to cover coordinates. Its target is the reachable coordinates,
and it can always reach them: a coordinate is reachable exactly when it has a cell the LHS may
use.

**The minimum density is computed over unmasked cells**, and its coverage term counts reachable
coordinates rather than all of them. Covering an axis needs one observation per reachable
coordinate, and a masked cell cannot host one, so the bound rises where the mask thins an axis and
falls where the opt-in lets it strand one.

**What a constant mask does not reproduce** is GLORYS's depth gradient: its wet fraction falls
from 69.71% at the surface to 54.22% at 2225 m, because the empty set is bathymetry rather than a
land mask. A mask that varies with the stratum would need a profile of fractions and nested masks,
since a cell excluded at one depth stays excluded deeper. Recorded as an extension. The blockiness
is what the compression and read-time measurements respond to, and a constant mask has that.

## What must not change

* density and per-variable observation counts stay exact
* every coordinate of every axis stays used, except ones a mask strands under
  `allow_unused_coordinates`
* a stratum still depends only on `(seed, stratum)`, so serial equals parallel
* overlap keeps working under `scattered`, with or without a mask

The golden digests will move for any case that asks for a non-default layout, and must not move
for any case that does not.

## Report rows to add

* the achieved layout, as a measurable: the fraction of occupied cells whose neighbour along the
  padded axis is also occupied. Scattered and padded separate sharply on this, so it confirms the
  parameter did what was asked.
* occupancy against the full grid alongside the requested density, whenever a mask is in play,
  since the two have different denominators there.
* the achieved mask fraction against the requested one.
* the coordinates a mask left unreachable, named rather than counted, so the relaxation is
  visible instead of silent.
* achieved F1 under padded, which is derived rather than requested.

## Stages

1. `layout="padded"`: the prefix construction with its own coverage guarantee, the padded
   variants of `compute_min_density` and `stratum_counts`, the split-dimension conflict, and the
   `overlap` refusal.
2. The report rows.
3. `mask_fraction`, the density denominator, the reachable-coordinate coverage rule and
   `allow_unused_coordinates`. Composes with either layout. Needs the sentence in
   `docs/explainer.md` first, since it changes a definition there.
4. Optional: a mask that varies with the stratum, and a `layout` per variable rather than one for
   the dataset.

## Open questions

* Which axis is padded by default? Argo and CrocoLake want the level axis, which is the non-split
  one. On more than two dimensions a prefix along one axis inside a hyperplane generalises, but
  the default needs stating.
* **One padded axis only, for now.** Two would nest: a prefix along A, then within each filled
  position of A a prefix along B, so the occupied set is `{(i, j, l) : j < k_i, l < m_ij}` and the
  counts need two levels of apportionment rather than one. It also needs
  `n_dims >= n_padded + 1`, since the split dimension cannot be padded, so two padded axes means
  at least three dimensions and two dimensions can never have both padded. None of the three
  reference datasets needs it -- Argo and CrocoLake pad one axis, GLORYS pads none -- so this is
  recorded rather than built. A spectral instrument sampling variable wavelengths at variable
  depths would be the case that asks for it.
* Should `layout` be per variable? Argo has both kinds in one file -- padded measurements beside
  per-profile scalars that are fully occupied. Stage 4, unless it turns out to be needed earlier.
* What distribution for `k`? Argo's levels per profile have median 70 and maximum 1042, so a
  lognormal is closer than a uniform. The measured quartiles are worth copying.

## Verification owed

`pylint` is not installed in the environment this work was written in, and CLAUDE.md requires a
score of 8 or better per file. Nothing added for the layouts, nor anything else written alongside
them, has been checked against it. A pass over the whole package is owed rather than a pass over
the new files.
