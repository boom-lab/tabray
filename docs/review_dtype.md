# Review diagrams for the dtype branch

Five diagrams for `feat/add-dtype-param-option` (PR #22): a baseline at the
commit the branch starts from, then one per review unit. Each unit is a
contiguous range of commits, so `git diff <from> <to>` shows exactly what the
unit changed. Each diagram shows the generation flow **as it stands at the end
of that unit** and highlights what changed against the previous one.

Render with `plantuml -tpdf docs/review_dtype_*.puml`.

## Colour key, the same in every diagram

| colour | meaning |
|---|---|
| green | added or changed by this unit |
| red | removed by this unit |
| orange | single-variable flow touched by this unit |
| yellow | generated data or a file that changed as a result |
| plain | unchanged since the previous unit |

## The worked example

The call from the PR #18 series, so the numbers compare across both:

```python
GenerateData(
    num_obs=100, num_dims=3, ratio_dims=[1, 2, 2],
    density=[0.4, 0.1], seed=77,
    num_vars=2, var_dims=[3, 2],
    overlap=0.5, fixed_overlap=False,
    max_obs=None,   # serial;  max_obs=40 -> NTASKS=3, merge_nc=True
)
```

It resolves to a `[4, 8, 8]` grid, `num_obs` 102, `var_num_obs` 102 and 26,
`dim_split` 1. Four variants run alongside, each serial and parallel:

| case | change to the call | used by |
|---|---|---|
| A/B | none | every unit, to show the default output does not move |
| E | `dtype=["int16", "float32"], fill_value=[None, 99999.0]` | units 01, 02 |
| P | `dtype=["float32", "int16"], pack=["int16", None], value_range=[None, (0, 8)]` | unit 02 on |
| O | `var_dims=[3, 3], overlap=0.05` | unit 03 |
| single | `num_vars=1, density=0.4` | every unit |

Every figure in the diagrams (sizes, dtypes, scale factors, per-stratum
counts, F1) comes from running these cases at the unit's end commit, one
`git worktree` per checkpoint, not from reading the code. Serial equals
parallel in every run.

## The units

| # | range (`git diff`) | commits | lines | what it does |
|---|---|---|---|---|
| 00 | `088efdd` | — | — | baseline: main after PR #18 |
| 01 | `088efdd b6776f5` | b6776f5 | +584/-9 (src +361, tests +201) | `dtype` and `fill_value` per variable; packed integers; `VariableEncoding` |
| 02 | `b6776f5 e482c4b` | 261ef73, e482c4b | +394/-115 | `pack` separate from `dtype`; integer variables from `value_range`; nullable parquet |
| 03 | `e482c4b 3b23ae6` | 3b23ae6 | +359/-6 (src +105) | overlap count decided once and apportioned across strata |
| 04 | `3b23ae6 8ac3fa4` | 420d224, 78cc3e0, 8ac3fa4 | +328/-209, mostly docs and formatting | unused parameter, `storage_dtype`, CLAUDE.md trimmed, black |

Unit 01 is a single commit over the 400-line target; 201 of its lines are the
new test file. Splitting it would mean rewriting the commit.

Units 03 and 04 leave cases A/B, E and P byte-identical. Unit 03 changes case O
only (F1 0.0784 -> 0.0490 for a target of 0.05), and unit 04 changes no output.
Unit 02 changes what `dtype="int16"` means: in unit 01 it packed a float, from
unit 02 on it makes an integer variable.
