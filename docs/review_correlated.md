# Review diagrams for the correlated-values branch

Diagrams for `feat/smooth-var-fields`: a baseline at the commit the branch
starts from, one per review unit, and a final diagram of the whole branch
against the starting point. Each unit is one commit, so `git show <commit>`
shows what it changed.

Render with `plantuml -tsvg docs/review_correlated_*.puml`.

## Colour key

| colour | meaning |
|---|---|
| green | added or changed by this unit |
| red | removed by this unit |
| orange | single-variable flow touched by this unit |
| yellow | generated data, a file, or printed output that changed as a result |
| plain | unchanged since the previous unit |

## The worked example

`GenerateData(num_obs=100, num_dims=3, ratio_dims=[1, 2, 2], seed=77, ...)`,
serial and with `max_obs=40` (3 chunks, `merge_nc=True`), the call of the
earlier review series:

| case | rest of the call | why |
|---|---|---|
| S | `density=0.4` | single variable; must not change |
| A | `density=[0.4, 0.1], num_vars=2, var_dims=[3, 2], overlap=0.5` | several variables; must not change |
| CS | case S + `data_structure="correlated"` | single variable, default length |
| CA | case A + `data_structure="correlated", correlation_length=[None, 2, 4], noise=0.1` | several variables, per-dim lengths, noise |
| Z | `num_obs=80000, num_dims=4, ratio_dims=[4, 20, 100, 100], density=1, pack="int16"`, rewritten with zlib level 1 + shuffle | what correlation does to compressed size |

Every figure comes from running these cases at the unit's commit, one
`git worktree` per checkpoint, with `PYTHONPATH` set to the worktree so the
editable install does not shadow it.

## The split

The branch was one commit, "temporary smooth values commit" (+1021/-17),
rebased onto main `57f873a`. It is now five commits with the same final tree:

| # | range (`git diff`) | lines | what it does |
|---|---|---|---|
| 00 | `57f873a` | — | baseline: main |
| 01 | `57f873a 7eb2025` | +69/-15 (src) | the per-variable broadcast rule moves to `utils/per_variable.py`; NumPy scalars count as one value |
| 02 | `7eb2025 df6419f` | +471 (src +354, tests +117) | `CorrelatedField`, new stream tag, scipy; not wired |
| 03 | `df6419f ca25382` | +299/-2 (src +70/-1, tests +229/-1) | `data_structure`, `correlation_length`, `noise` in `GenerateData`; applied on global indices before the chunk rebase |
| — | `ca25382 3309cb7` | +171/-1 (docs) | README, `docs/correlated_values.md`, `docs/variable_encoding.md`, CLAUDE.md |
| — | `3309cb7 70fa64d` | +12 (docs) | HANDOFF: parquet dims bug found in review |
| 04 | `57f873a 70fa64d` | +1021/-17 | the whole branch, final against initial |

The full test suite passes at 7eb2025, df6419f and ca25382; the last two
commits change no code. Unit 02 is over the 400-line target on its own, and the
branch is at +1021.

## Found by running the example

1. **Unit 01 is not behaviour-neutral.** On main, a NumPy scalar such as
   `fill_value=np.float32(-1.0)` raises `TypeError: 'numpy.float32' object is
   not iterable`; from 7eb2025 it applies to every variable. The commit
   message says so.
2. **The default length is longer than every axis of the example.** Case CS
   warns three times (10 cells on axes of 4 and 8). The default is in cells,
   not a share of the axis, so small grids are nearly constant unless a
   length is given.
3. **Uncompressed files do not change size.** Cases CS and CA write the same
   bytes as S and A: the package writes netCDF uncompressed and float64. The
   effect shows only once compression is on (case Z: 1.01 to 0.854).
