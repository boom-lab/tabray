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

`layout="scattered"` (default, today's behaviour) or `"padded"`. A mask was considered as an
orthogonal parameter and declined; see below.

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

## The mask: declined

A mask would exclude a fraction of the cells, the same exclusion in every stratum, with
blob-like shapes from a thresholded smoothed field. It is not being built. The measurements are
why.

Same grid, same occupancy of 0.149, same number of occupied cells:

```
arrangement    occupancy   nc zlib   parquet   smaller
scattered          0.149     1.15M     1.00M   parquet
blobs r=3          0.149     0.92M     0.97M    netCDF
blobs r=10         0.149     0.77M     0.93M    netCDF
blobs r=40         0.149     0.67M     0.90M    netCDF
padded             0.148     0.54M     0.83M    netCDF
one band           0.149     0.51M     0.84M    netCDF
```

Blobs land between 0.67 and 0.92, bracketed by `scattered` at 1.15 and `padded` at 0.54, and the
patch scale is the dial that slides them along. So a mask is an interpolation between two
arrangements the package already produces, not a third regime. Every figure it could give is
reachable from the two endpoints.

Against that: it needs a fraction *and* a scale, since the fraction fixes how much is empty and
the scale fixes how compressible that emptiness is. `GenerateData` already takes twenty
parameters. It also drags in the density denominator question -- whether `density` means the
unmasked cells or the whole grid, which would give `density` a third meaning after the
full-grid convention for reduced-dimension variables -- and a change to the premise in
`docs/explainer.md`, because 38 of GLORYS's 2041 latitudes hold no ocean at all and a mask that
strands a coordinate contradicts a definition stated there.

**What would justify revisiting.** The one thing a mask does that `padded` cannot is reproduce a
geometry: land is two-dimensional blobs, a profile file is one-dimensional runs, and handing a
downstream tool something land-shaped needs the former even where the latter brackets its
compressed size. That is a different purpose from the occupancy sweep. If the question becomes
"does this reader handle a land mask", build it; if it stays "how does occupancy change the
array-versus-tabular answer", the two implemented layouts bracket it.

**A measurement that did not settle anything**, recorded so it is not repeated: the real GLORYS
land mask on a 2000x500 window gave 3.11 MB against 3.31 MB for scattered occupancy, 6% apart.
That window is 93% ocean, and at that occupancy there is too little emptiness for its shape to
matter. The effect needs sparse data, so the synthetic rows above are the evidence and the real
mask is not.

## What must not change

* density and per-variable observation counts stay exact
* every coordinate of every axis stays used, except ones a mask strands under
  `allow_unused_coordinates`
* a stratum still depends only on `(seed, stratum)`, so serial equals parallel
* overlap keeps working under `scattered`

The golden digests will move for any case that asks for a non-default layout, and must not move
for any case that does not.

## Report rows to add

* the achieved layout, as a measurable: the fraction of occupied cells whose neighbour along the
  padded axis is also occupied. Scattered and padded separate sharply on this, so it confirms the
  parameter did what was asked.
* occupancy against the full grid alongside the requested density.
* achieved F1 under padded, which is derived rather than requested.

## Stages

1. `layout="padded"`: the prefix construction with its own coverage guarantee, the padded
   variants of `compute_min_density` and `stratum_counts`, the split-dimension conflict, and the
   `overlap` refusal.
2. The report rows.
3. **Declined.** The mask, for the reasons above. `docs/explainer.md` needs no change as a
   result, since nothing else can strand a coordinate.
4. Optional: a `layout` per variable rather than one for the dataset.

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
