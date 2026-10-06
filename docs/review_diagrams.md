# Review diagrams for the parallel multi-variable branch

Fifteen diagrams, one per review unit, covering all 34 commits of
`feat/add-parallel-support-for-multivar` (+7,958 / -7,624 as a single PR).
Each diagram shows the whole generation flow **as it stands at the end of that
unit**, so a diagram can be read on its own, and highlights what the unit
changed against the previous one.

Render any of them with `plantuml docs/review_NN_*.puml`; the committed PDFs
were produced with `plantuml -tpdf docs/review_*.puml`.

## Colour key, used identically in every diagram

| colour | meaning |
|---|---|
| green | added or changed by this unit |
| red | removed by this unit |
| orange | single-variable flow touched by this unit |
| yellow | generated data or a file that changed as a result |
| plain | unchanged since the previous unit |

## The worked example

Every diagram follows the same call, so the numbers are comparable across the
series. Two cases run side by side in each diagram, plus a third where the unit
needs one:

```python
GenerateData(
    num_obs=100, num_dims=3, ratio_dims=[1, 2, 2],
    density=[0.4, 0.1], seed=77,
    num_vars=2, var_dims=[3, 2],
    overlap=0.5, fixed_overlap=False,
    max_obs=None,   # case A, serial;  case B uses max_obs=40 -> NTASKS=3
).generate(netcdf_filepath, parquet_filepath, parquet_tmp)
```

It resolves to a grid of `[4, 8, 8]` (256 sites), `num_obs` corrected to 102,
`var_num_obs` 102 and 26, with `var1` varying along two of the three dimensions
and pinned to one coordinate on the third. The single-variable counterpart
(`num_vars=1, density=0.4`, same grid) runs alongside, serial and parallel, so
each diagram can say whether the unit touched that path.

Seed 77 was chosen because the example has to run at **every** checkpoint: with
a two-element density, the code before unit 11 tossed a coin for which variable
got the maximum, and half of all seeds (12345 among them) raised
`ValueError: var0 must be the largest variable`.

## How the numbers were produced

Every figure in the diagrams — shapes, per-stratum counts, achieved overlap,
row counts, file sizes — comes from running the cases at that commit, not from
reading the code. One `git worktree` per checkpoint, the `tabray` conda
environment, and a probe script that records the instance attributes, the
occupied index sets, the DataFrame, the netCDF attributes and the files on
disk. The probe and its per-checkpoint output live in this session's scratch
directory; say the word and it can be committed under `tests/`.

## The units

| # | commits | endpoint | src lines | what it does |
|---|---|---|---|---|
| 01 | b646f9a, 18a85ef, 48950f2, adb82be | adb82be | +424/-84 | parallel support for multi-variable runs, `max_workers`, unsqueezed chunk files |
| 02 | 2b14258, 86e9e2e | 86e9e2e | +304/-75 | stratify the grid by hyperplane; `apportion`; single-variable path moves over |
| 03 | 08e0cfa, 4aac88d, c0393ef | c0393ef | +564/-127 | per-stratum placement for every variable; `_choose_split_dim`; LHS takes `max(shape)` |
| 04 | 5199b5e, 2368f7c | 2368f7c | +262/-57 | overlap reported as F1, with F2 alongside |
| 05 | 6390501 | 6390501 | +98/-22 | both paths write the same self-describing attributes |
| 06 | a2bc4fb | a2bc4fb | +95/-9 | opt-in `merge_nc` concatenates the chunk files |
| 07 | 3e31654 | 3e31654 | +83/-67 | identical parquet row order in both paths |
| 08 | 86479f5, 2748cf8, b46195b | b46195b | +14/-293 | one overlap entry point, pooled metric dropped |
| 09 | e9c789c, 9a4841c, f88be14 | f88be14 | +217/-4598 | remove the machinery the stratum model replaced |
| 10 | d49efbc, b218154 | b218154 | +172/-74 | scratch directory of its own, opt-in worker logging |
| 11 | d2c79af, 870b9de, 7fa9e8e | 7fa9e8e | +292/-110 | explicit `var_dims` indices, deterministic two-element density |
| 12 | 77d4b72 | 77d4b72 | +349/-94 | every random stream from one tagged registry |
| 13 | 8525f0c | 8525f0c | +171/-43 | minimum density from the longest axis |
| 14 | ab60aee | ab60aee | +195/-7 | merge one variable at a time, single-threaded |
| 15 | 7424f42, 3e4f87f, d3e9329, 0f070a1, 5d3cb12 | 5d3cb12 | +419/-14 plus formatting | one codec for both formats; comment trimming, pylint, black |

Units 08, 09 and 13 leave the example's output byte-identical, which is the
claim to check rather than the code. Unit 12 changes the most data: it moves
`var1` onto a different constant dimension and with it the split axis, the
coordinates and the achieved overlap.

Reading order is the order above; it is also chronological, so
`git log -p --reverse main..HEAD` walks the same ground commit by commit.
