# Review diagrams for the layout branch

Diagrams for `feat/add-data-occupancy-layout` (PR #24): a baseline at the commit
the rebased branch starts from, one per review unit, and a final diagram of the
whole PR against the starting point. Each unit is a contiguous commit range, so
`git diff <from> <to>` shows what it changed.

Render with `plantuml -tsvg docs/review_layout_*.puml` (`-tpdf` fails on this
PlantUML build; the PDFs were made from the SVGs with inkscape).

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
serial and with `max_obs=40` (3 chunks, `merge_nc=True`):

| case | rest of the call | why |
|---|---|---|
| A | `density=[0.4, 0.1], num_vars=2, var_dims=[3, 2], overlap=0.5` | the PR #22/#23 call; must not change |
| S | `density=0.4` | single variable, scattered |
| P1 | `density=0.4, layout="padded"` | single variable, padded |
| P2 | `density=[0.4, 0.1], num_vars=2, var_dims=[3, 2], layout="padded"` | several variables, padded |
| P3 | `density=[0.4, 0.2], num_vars=2, var_dims=[3, [1, 2]], layout="padded"` | var1 has the padded axis |
| P4 | `density=[0.4, 0.2], num_vars=2, var_dims=[3, [0, 1]], layout="padded"` | var1 lacks the padded axis |
| E1 | P2 with `overlap=0.5` | must raise |

Every figure comes from running these cases at the unit's end commit, one
`git worktree` per checkpoint.

## The rebase

PR #24 sat on the pre-rebase `5e31df1` from PR #23. Its commits were replayed
onto main (`94efaaf`), and five commits fixing what the worked example showed
were added on top:

| original | rebased | change |
|---|---|---|
| `fe6d696` plan the layout parameter | `7f83754` | none |
| `3749c01` generate the padded layout | `14d7be1` | 8 conflicts resolved; `ChunkUtils.apportion`'s `capacity` argument restored; black. Its run-fraction report row was dropped (tests check the same thing) |
| `ca7d6cc` require the conda env | `012f99e` | shortened to a CLAUDE.md rule |
| `6875619` declare scipy | dropped | main already names the env `tabray` and lists pylint; scipy was for the declined mask |
| `f3889d3` decline the mask layout | `8b8bfdc` | "full-grid convention" updated to the own-grid density |
| — | `9eed344` | new: pad every variable that has the padded axis |
| — | `a5ffa7d` | new: the report states which minimum-density bound applied |
| — | `17aeb80` | new: clip only var0 to the minimum density |
| — | `42b032c` | new: "fills its first k cells" replaces "prefix" |
| — | `294e031` | new: equal weights for the padded counts (were lognormal, never requested) |

## The units

| # | range (`git diff`) | lines | what it does |
|---|---|---|---|
| 00 | `94efaaf` | — | baseline: main after PR #23 |
| 01 | `94efaaf 7f83754` | +230 (docs) | the layout plan |
| 02 | `7f83754 14d7be1` | +623/-5 (src +390, tests +210) | `layout="padded"`: placement, minimum density, split dim, overlap guard |
| 03 | `14d7be1 8b8bfdc` | +52/-74 (docs) | conda-env rule; mask layout declined |
| 04 | `8b8bfdc 294e031` | +233/-107 (src +107/-58, tests +83/-15) | every variable with the padded axis padded; minimum-density row; var0-only clip; wording; equal weights |
| 05 | `94efaaf 294e031` | +967/-15 | the whole PR, final against initial |

Unit 02 is over the 400-line target; 210 of its lines are the new test file.

## Found by running the example

None of these was intended: the plan and the commit messages say otherwise.

1. **Only var0 was padded** -- fixed in unit 04. The plan's reason for refusing
   `overlap` assumes every variable fills its first k cells, and the
   per-variable layout it deferred was about variables lacking the padded
   axis. Now one layout applies to the dataset: every variable with the padded
   axis fills its first k cells, one without it is scattered.
2. **The bound in force was not reported** -- fixed in unit 04. The plan asked
   that "the generation report should state which bound applied".
3. **The minimum density clipped every variable** -- fixed in unit 04. The
   bound is var0's coverage requirement, but `validate_and_clip` raised every
   variable to it (P2: var1 0.1 -> 0.152, 10 cells instead of 6). Only var0 is
   clipped now.
4. **Sizes do not move** -- open. The layout changes compressed size; this
   package writes uncompressed files, so P1 and S write the same 14061 B.
