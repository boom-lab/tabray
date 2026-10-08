# Review diagrams for the report branch

Four diagrams for `feat/add-reporting-of-stats` (PR #23): a baseline at the
commit the rebased branch starts from, then one per review unit. Each unit is
a contiguous commit range, so `git diff <from> <to>` shows what it changed.
Each diagram shows the flow **as it stands at the end of that unit** and
highlights what changed against the previous one.

Render with `plantuml -tsvg docs/review_report_*.puml`. (`-tpdf` fails on
this PlantUML build; the PDFs were made from the SVGs with inkscape.)

## Colour key

| colour | meaning |
|---|---|
| green | added or changed by this unit |
| red | removed by this unit |
| orange | single-variable flow touched by this unit |
| yellow | generated data, a file, or printed output that changed as a result |
| plain | unchanged since the previous unit |

## The worked example

The call from the PR #18 and PR #22 series:

```python
GenerateData(
    num_obs=100, num_dims=3, ratio_dims=[1, 2, 2],
    density=[0.4, 0.1], seed=77,
    num_vars=2, var_dims=[3, 2],
    overlap=0.5, fixed_overlap=False,
    max_obs=None,   # case A, serial;  case B: max_obs=40 -> NTASKS=3, merge_nc=True
)
```

| case | change to the call | why |
|---|---|---|
| A/B | none | the report on a normal run, serial and parallel |
| single | `num_vars=1, density=0.4` | the single-variable report |
| S | `density=[0.4, 0.4], var_dims=[[0, 1, 2], [1, 2]]` | saturation: var1 asks for 102 sites on a 64-cell grid |

Every printed report and number in the diagrams comes from running these
cases at the unit's end commit. Output files are byte-identical at all three
checkpoints, and serial equals parallel in every run.

## The units

| # | range (`git diff`) | lines | what it does |
|---|---|---|---|
| 00 | `f87a0e9` | — | baseline: main after PR #22 |
| 01 | `f87a0e9 a786b4a` | +472/-7 (src +304, tests +159) | `GenerationReport`, measured from the arrays, serial path |
| 02 | `a786b4a 4998269` | +221/-6 (src +178, tests +40) | the chunked path: `measure_chunk` in workers, `from_chunks` in the parent |
| 03 | `4998269 e059ed9` | +285/-110 (src +127/-85, tests +136/-24) | density as a share of the variable's own grid; rounding always reads `differs`; coverage for var0 only |

The original single commit `5e31df1` was split in two during the rebase onto
main and formatted with black. The final tree equals the rebased commit apart
from formatting. Unit 01 is over the 400-line target by its test file.

## Found by running the example

Running the worked example at unit 02 showed the report disagreeing with
itself. The rows are the same at the original `5e31df1`, so these predate the
rebase.

1. **Density, serial path** -- fixed in unit 03. Nothing defined density for a
   variable on fewer dimensions; the generator treated it as a share of the
   full grid, `from_arrays` as a share of the variable's own grid. Unit 03
   makes it the own grid everywhere and documents it.
2. **Density, serial vs parallel** -- fixed in unit 03: `from_chunks` now
   divides by the own grid too.
3. **Coverage of non-reference variables** -- fixed in unit 03. Only var0's
   coverage is guaranteed, by the LHS stage, so the coverage row is now var0's
   alone. That also removes the chunked path checking axes a variable does
   not have.
4. **Rounding** -- fixed in unit 03: density rows compare against the density
   as requested, at the default tolerance, so the validator's rounding reads
   `differs` in every run, single variable included.
