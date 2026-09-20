# Plan: a generation report

Every run should state what was asked for, what was produced, and where the two differ, with
the evidence for the difference. Today that information exists but is scattered: the validator
prints the corrections it makes, the overlap step prints achieved F1 beside its target, and
everything else is left to the reader to measure.

## What decides whether this is useful

**Measure from the written data, not from the generator's own bookkeeping.** A report that
re-reads `self.var_num_obs` will agree with a bug in placement; one that counts non-fill cells
in the array will not. The overlap problem that prompted this was found by reading occupancy
masks, and would have been invisible to a report built on internal state.

**Offer evidence, not verdicts.** Where the numbers separate two causes, say which. Where they
do not, list the candidates rather than choose.

## What it covers

| property | achieved from | flagged when |
|---|---|---|
| shape / `ratio_dims` | coordinate lengths | axis rounding moved it |
| `num_obs` per variable | count of non-fill cells | differs from the request |
| `density` per variable | occupied / grid | differs from the request |
| `overlap` per variable | F1 and F2 from the occupancy masks | differs from the target |
| coverage | coordinates used per axis | any coordinate unused |
| `dtype`, `fill_value`, compression | read back from the file | differs from the request |
| row consistency | parquet rows against occupied cells | the two formats disagree |

## Status, not a bare number

Three values, so the validator's own corrections do not read as faults:

* `match`
* `adjusted` -- the validator changed the input on purpose and said so
* `differs` -- everything else

A report that flags expected rounding on every run trains the reader to ignore it.

## Evidence

Two cases separate cleanly, because the quantities are computable:

* overlap off **with clipped strata** -- consistent with density and overlap being
  over-determined for that variable
* overlap off **with no clipping** -- consistent with per-stratum rounding; the achieved value
  can be predicted before placement and quoted

Everything else lists candidates.

## The parallel path

`generate()` returns `(None, None)` under chunking and the data lives on disk. Reading the
output back would be the most faithful measurement and the most expensive. It is not needed:
overlap never spans strata, by the construction in
`MultiVarRecordGenerator.generate_multivar_stratified`, so per-chunk intersection and occupancy
counts sum to the global figures. Coverage needs a union rather than a sum, which is one boolean
vector per axis -- cheap for a worker to return. Workers return counts, the parent aggregates,
nothing is re-read.

## Where the report goes

Printed at the end of `generate()`, and returned as an object so notebooks and tests can assert
on it.

Not into the netCDF attributes at first. That would change every output file and invalidate the
golden baseline, so it belongs in a later opt-in step rather than as a side effect of this work.

## Stages

1. The report class, the serial path, printing and returning. Nothing written changes, so the
   golden digests stay put.
2. Parallel aggregation from worker counts.
3. The overlap apportionment fix. Stage 1 becomes its test: rows that read `differs` on the
   tutorial 3 cases should read `match` afterwards.
4. Optional, opt-in: the report in the file attributes.

## Risks

* **Cost.** Measuring occupancy is a pass over the masks, which `ParquetBuilder` already makes.
  The report should reuse that rather than add a pass.
* **Noise.** See the `adjusted` status above.
* **Circularity.** See the first section. If the report ever starts reading internal state for
  convenience, it stops being a check.
