# How the grid gets filled: the stratified split, step by step

A worked trace of the generation algorithm under the stratified design (global LHS, stratified
fill), at both ends of the density range, serially and in parallel. All figures below are real
output, not illustrations.

## Setup

```
grid   (4, 3, 6)          dims x0=4, x1=3, x2=6
split  dim 2              the largest dimension, chosen by argmax
strata 6 hyperplanes      one per index j along x2, each of shape (4,3) = 12 sites
chunks 3                  2 strata per chunk
total  72 sites
```

The **stratum** is one hyperplane at a fixed index of the split dimension. It is set by the
grid and never changes. The **chunk** is a worker's unit of work, a contiguous run of strata,
and is set by `max_obs`. Decoupling them is what makes the output independent of how the run
is parallelised.

Panels below show one hyperplane each, rows = x0, columns = x1.

```
legend   L = var0 placed by the LHS stage     0 = var0 placed by the fill stage
         1 = var1 only                        B = both variables
         X = an LHS point also held by var1   . = empty
```

## The five steps

| step | what happens | scope | cost |
|---|---|---|---|
| 1 | draw each coordinate axis, `sort(rng_d.uniform(0,1,shape[d]))` | global | `shape[d]` floats |
| 2 | LHS stage: `n_s = min(num_obs, min(shape))` points, per-axis permutation or choice, paired positionally | **global** | `O(min(shape))` |
| 3 | apportion the remaining `num_obs - n_s` across strata (largest remainder); likewise var1's count and its overlap share | global, deterministic | `O(num_strata)` |
| 4 | in each stratum, draw var0's fill sites from `rng(seed, j)`, avoiding the LHS points already there | **per stratum** | `O(m_j)` |
| 5 | in each stratum, var1 takes its overlap share from var0's sites there, then fills the rest from sites held by neither | **per stratum** | `O(m_j)` |

Steps 1–3 are global but cheap: one axis, `min(shape)` LHS points, and a vector of per-stratum
counts. Steps 4–5 touch one hyperplane at a time. Nothing anywhere is proportional to the total
grid, which is what removes the S1 wall.

Step 2 stays global deliberately. At the sparse end the LHS *is* the sample, and its guarantee
spans strata, so it cannot be reproduced stratum by stratum. It costs `min(shape)` values, so
every worker can recompute it identically and keep the points that land in its own strata.

## Case A — high density

```
var0 54 obs (density 0.750), var1 24 obs, overlap target 0.5

STEP 2  global LHS: n_s = min(54, min(4,3,6)) = 3   points [(0,1,5), (3,2,4), (1,0,2)]
        -> LHS is 5.6% of var0's observations
STEP 3  fill per stratum  [8, 8, 8, 9, 9, 9]   (sum 51 = 54 - 3)
        var1 per stratum  [4, 4, 4, 4, 4, 4], of which overlap [2, 2, 2, 2, 2, 2]
                                               (sum 12 = round(0.5 x 24))
STEP 4/5 result per stratum:
        j=0      j=1      j=2      j=3      j=4      j=5
        0 1 .   1 0 0   0 . 0   0 0 .   0 0 0   0 L B
        0 1 0   . 1 B   L B 0   1 B 0   0 B 0   B 0 0
        B 0 0   0 0 0   0 1 B   0 0 B   0 1 1   0 1 0
        0 . B   B . 0   0 1 0   1 0 0   0 0 X   1 0 0

        totals: var0 54, var1 24, coincident 12 -> overlap 0.500
```

The three LHS points land in strata 5, 4 and 2 (their x2 coordinates), so those strata carry
one LHS point each and the fill makes up the rest. The LHS is 5.6% of the sample and does
nothing the random fill would not have done anyway at this density.

## Case B — high sparsity

```
var0 6 obs (density 0.083), var1 6 obs, overlap target 0.5

STEP 2  global LHS: n_s = min(6, min(4,3,6)) = 3   points [(0,1,5), (3,2,4), (1,0,2)]
        -> LHS is 50.0% of var0's observations
STEP 3  fill per stratum  [0, 0, 0, 1, 1, 1]   (sum 3 = 6 - 3)
        var1 per stratum  [1, 1, 1, 1, 1, 1], of which overlap [0, 0, 0, 1, 1, 1]
                                               (sum 3 = round(0.5 x 6))
STEP 4/5 result per stratum:
        j=0      j=1      j=2      j=3      j=4      j=5
        . . .   . . .   . . .   B . .   . . .   . X .
        . . 1   . 1 .   L . .   . . .   . . .   . . .
        . . .   . . .   1 . .   . . .   B . .   . . .
        . . .   . . .   . . .   . . .   . . L   . 0 .

        totals: var0 6, var1 6, coincident 3 -> overlap 0.500
```

Same three LHS points, but now they are **half the dataset**. Three strata hold nothing but an
LHS point, three hold nothing but a fill point. This is why step 2 cannot be stratified: the
LHS guarantee (every x1 coordinate used exactly once, since x1 is the shortest axis) is a
statement about the whole grid, and no stratum can enforce it alone.

## In parallel

Each worker runs steps 1–3 itself — they are cheap and deterministic — then runs steps 4–5 for
its own strata only.

```
chunk 0 -> strata 0, 1      chunk 1 -> strata 2, 3      chunk 2 -> strata 4, 5
```

```
dense:  every chunk reproduces the serial strata exactly -> True
sparse: every chunk reproduces the serial strata exactly -> True
```

The strata are byte-identical to the serial run because each is generated from `(seed, j)` and
nothing else. Regroup them 2-per-worker, 6-per-worker or 1-per-worker and the output does not
move, so `max_obs` becomes purely a memory and scheduling knob with no effect on the data.

## Constraint: the fill pool can run out

Step 5 draws var1's non-overlapping sites from sites held by neither variable, so per stratum
it needs

```
stratum_sites - m0  >=  (1 - t) * m1
```

With 12 sites and var0 holding 10, a var1 of 6 at `t=0.5` needs 3 free sites and has 2:
infeasible. That is why Case A uses var1 = 24 rather than 36. The current code's
`if len(available_flat) >= num_separate` branch skips silently in this situation, dropping
observations without warning; this bound should be checked at validation time instead.

## Where the current code differs

- **Step 1**, split axis: chunks draw within their own value sub-range instead of slicing the
  global sorted axis, which stratifies the coordinates and makes them depend on `NTASKS` (D1).
- **Steps 2 and 4**, in parallel: each chunk rebuilds the *entire global* index set and discards
  the part outside its range, via a Python comprehension over every grid point (S1).
- **Steps 4 and 5**, multi-variable: placement is re-seeded per chunk rather than per stratum,
  so chunk boundaries change the output (D4).
- **Step 5**: the fill pool excludes only var1's own overlap sites, not var0's, so achieved
  overlap exceeds the target by about `(1-t) * density(var0)` (A3). The trace above uses the
  corrected pool, which is why both cases hit 0.500 exactly.
