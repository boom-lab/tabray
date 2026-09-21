# Serial / parallel equivalence: diagnostics

Branch: `feat/add-parallel-support-for-multivar`
Date: 2026-09-19
Scope: diagnosis. Entries marked **[done]** have since been fixed; everything else is unchanged
and still describes current behaviour.

**Status.** Stages A, B and C of `docs/explainer_stratified_generation.md` are implemented,
plus the coverage fix (A5).

* **Stage A** (step 1, plus `num_obs` preservation) closed **D1** and **D3**. Serial output
  untouched.
* **Stage B** (steps 2 and 4: global LHS, stratified fill, per-stratum values) closed **D2** and
  **S1** for the single-variable path. Single-variable parallel output is now *identical* to
  serial -- coordinates, occupancy, values and parquet content -- across 1D-6D, minimum density,
  density 1.0, and split dimensions other than dim 0. Peak memory for one GLORYS12 stratum went
  from ~160 GB to no measurable allocation, 0.02 s.
* **A5** made the coverage guarantee hold for every axis, and it is now enforced by explicit
  guards rather than merely achieved.

* **Stage C** (step 5: per-stratum multi-variable placement) closed **D4** and **A3**, and the
  semantic half of **D5**. Overlap follows F1 and the split dimension is restricted to
  dimensions every variable varies along.

* **D6** closed the on-disk differences: one overlap definition reported everywhere (F1, with
  F2 alongside), the same self-describing attributes from both paths, an opt-in `merge_nc`, and
  a parquet row order that is identical as written rather than only after sorting. **S2** and
  the rest of **D5** fell out of it.

**Serial and parallel now produce identical files on disk** -- coordinates, dimensions, values,
parquet rows and attributes -- verified across single- and multi-variable, 2D to 6D, minimum
density, reduced-dimension variables, `overlap='random'` and `fixed_overlap`.

Still open: **S6** and **A4**. Sections describing a closed item record the behaviour *before* the change.

## Question

The parallel workflow exists to generate larger-than-memory cases on HPC, so it must satisfy the
same constraints as the serial workflow: given infinite resources, a single-threaded run and a
multi-threaded run with the same parameters should produce the same data — identical netCDF and
identical parquet content on disk.

## Finding

**Not currently the case.** With `max_obs` set so that `NTASKS > 1`, parallel output never
matches serial output. With `NTASKS == 1` the two paths are bit-identical (verified for single-
and multi-variable), so every divergence originates in the chunked path.

- Single-variable: occupancy in *index* space already matches serial in most configurations.
  Three causes separate it from exact equivalence (D1, D2, D3). **All three are now closed; the
  single-variable paths agree exactly.**
- Multi-variable: placement is re-randomised per chunk, and one configuration class
  (constant dimension == split dimension) is semantically wrong, not merely different.

## What each path does

| step | serial | parallel (per chunk) |
|---|---|---|
| coords, non-split dims | `sort(rng_d.uniform(0,1,n_d))` | same RNG, same draw -> **identical** |
| coords, split dim | `sort(rng_s.uniform(0,1,N))` | `sort(rng_s.uniform(a_k,b_k,n_k))`, RNG restarted per chunk |
| sites, 1 var | global hybrid LHS from `rng(seed+1000)` | same global LHS, filtered to the chunk's index range |
| values, 1 var | next draws of that **same** stream | fresh `rng(seed+1000+100*chunk_id)` |
| sites+values, multi-var | global placement, `rng(seed+9999)` / `rng(seed+var+2000)` | chunk-local placement, `rng(seed+9999+10000*chunk_id)` / `rng(seed+var+2000+100*chunk_id)` |
| output | 1 `.nc`, 1 parquet dir | N `.nc` chunk files (no merge step), parquet consolidated |

## What matches

- Coordinates on every non-split dimension: bit-identical (`utils/chunk_utils.py:162`).
- Single-variable occupancy mask: identical in 1D/2D/3D/4D, at minimum density (pure LHS), at
  density 1.0, and for NTASKS up to 8 — whenever chunking does not change `num_obs` (see D3).
  The global-LHS-then-filter design in `generators/record_generator.py:261` holds.
- Per-variable observation totals in multi-var: preserved exactly by
  `get_multi_var_observations_per_chunk`.
- Achieved overlap: 0.6524 serial vs 0.6524 parallel on a 2-variable case.
- Parallel output is deterministic: two runs byte-identical; `max_workers=1` equals the default.

---

## Equivalence divergences

### [done] D1 — split-dimension coordinates differ, and come from a different distribution

Each chunk draws `n_k` uniforms confined to `[div_k/N, div_{k+1}/N)`; serial draws `N` iid
uniforms over `[0,1)` and sorts. Parallel coordinates are *stratified*, not iid — this is a
distributional difference, not only a difference in values.

Measured (2000 seeds, N=32, sections [11,11,10]): count of `x0` points in the first stratum is
Binomial in serial (mean 10.97, sd 2.65, range 3-20) and exactly 11 every time in parallel
(sd 0.00). Mean max-gap 0.1226 serial vs 0.1145 parallel.

Location: `workers/parallel_worker.py` coordinate loop; `utils/chunk_utils.py:162`.

**Fixed in stage A.** The worker now draws every axis over the global shape with the serial
call and slices the split axis by `task_range`, so the concatenated axis reproduces serial
exactly. Chunk blocks remain ordered and non-overlapping, so `xr.concat` and
`open_mfdataset(combine="by_coords")` both still reconstruct the dataset (verified).

### [done] D2 — observation values always differ, in every chunk including chunk 0

Serial draws indices and values from one stream, so values come from the advanced state. The
worker takes indices from `lhs_rng` and values from a fresh generator with the same base seed,
so the value stream restarts from the beginning.

Measured: serial first values `0.398, 0.590, 0.860`; chunk 0 `0.357, 0.372, 0.576`.

Location: `generators/multi_var_record_generator.py`, `generate_without_overlap` obs_seed branch.

**Fixed in stage B.** Sites and values are now drawn together per stratum from
`(seed, STRATUM_STREAM, j)`, so the pairing survives any regrouping of strata into chunks.

### [done] D3 — chunking silently changes `num_obs` and `density`, which changes *which* sites are occupied

`ChunkUtils.get_observations_per_chunk` rounds per chunk. The correction branch only touches the
last chunk, and only when the total overshoots — undershoots are never corrected.

Measured: sweep of 480 configurations (num_dims 1-4, various `ratio_dims`, num_obs 500-5000,
density 0.05-1.0, NTASKS 2-7) -> **81 of 416 valid configs (19%)** produced fewer observations
than serial (e.g. 1750 -> 1749, 1012 -> 1010).

The consequence is not a one-site difference. `num_obs_global` feeds the global LHS, so for
shape `[9,36,18]`, n=1750 vs n=1749 share only **1006 of 1750 sites (57%)**.

The per-chunk counts this function computes are then discarded: actual placement comes from the
LHS filter (measured chunk counts 515/517/527/457 against planned 511/511/511/483). So the
computation that perturbs `num_obs` is not the one that places the observations.

Location: `utils/chunk_utils.py` (`get_observations_per_chunk`), `generate_data.py:_generate_par`.

**Fixed in stage A.** `get_observations_per_chunk` now apportions by largest remainder, so the
total is preserved exactly; the 1750 -> 1749 case is now 1750/1750.

### [done] D4 — multi-variable placement is re-randomised per chunk

Chunk-local seeds plus chunk-local site selection. The occupancy mask differed in every
multi-var case run (2 vars full dims, `overlap='random'`, 3 vars with `fixed_overlap`). Totals
and overlap fraction are preserved; the actual sites are not.

Distributional side effect: the per-chunk hybrid LHS guarantees every coordinate of every
non-split dimension is used at least once *per chunk*, so globally each such coordinate is hit
at least `NTASKS` times — more LHS structure than serial produces.

Location: `generators/multi_var_record_generator.py`, `generate_with_overlap` seeding block.

**Fixed in stage C.** `generate_multivar_stratified` places every variable one hyperplane at a
time from `(seed, stratum)`. Verified identical to serial -- coordinates, occupancy and values --
for 2 and 3 variables, scalar and per-variable targets, `overlap='random'`, `fixed_overlap`, and
6D anisotropic grids. Overlap now follows F1 (`docs/explainer_multivar.md`): the per-stratum
target `t * |proj_j(S_0)|` is local, so the global denominator is never needed.

### [done] D5 — variables on fewer dimensions: structurally different, and wrong when the constant dim is the split dim

Two distinct causes:

1. The worker calls `build_dataset(..., squeeze_constant_dims=False)`
   (`workers/parallel_worker.py:253`) while serial defaults to `True`. Serial writes `var1` with
   dims `(x1,x2)`, parallel writes `(x0,x1,x2)`.
2. `_select_constant_coords` (`generators/multi_var_record_generator.py:48`) is handed the
   **task** shape, so the constant coordinate is drawn from `[0, task_size)` instead of
   `[0, max_dim_size)`.

Measured, 3 chunks, `var1` constant on the split dimension:

```
serial   var1 dims=('x1','x2')       -> x0 squeezed out (one coordinate)
parallel var1 dims=('x0','x1','x2')  -> non-nan at x0 = [7, 19, 31], 27 obs each
```

A variable that should live at one `x0` lives at three — one per chunk, each at the same
chunk-local index. When the constant dim is *not* the split dim the value is consistent
(index 3 only, 324 obs) and only the squeeze difference remains.

**Cause 2 fixed in stage C, and the case is now impossible by construction.** The constant
coordinate is drawn from the global shape, and the split dimension is chosen only among
dimensions *every* variable varies along (`_choose_split_dim`), so no variable can be constant
along it. That constraint is required anyway for projected overlap to stay stratum-local.

**Cause 1 fixed in D6** by the merge step: chunk files keep every dimension so they can
concatenate, and `generate(..., merge_nc=True)` squeezes on merge. Merging is opt-in because
large datasets are routinely served as many files.

### [done] D6 — on-disk layout

- N `.nc` chunk files with no merge step (serial writes one file).
- Multi-var chunk attributes differ from serial's: `chunk_id`, per-chunk `num_obs`, and a
  `density` of 0.699 that is `(var0+var1 obs)/chunk_grid`, which is not a meaningful quantity.
- Parquet **row order** differs whenever the split dimension is not dim 0. Serial rows are
  lexicographic in `(x0,x1,...)`; parallel chunks are concatenated in split-dim-block order.
  Measured: shape `[20,60]`, split dim 1, identical occupancy, serial lexsorted, parallel not.
- **Metadata is incomplete and inconsistent between the paths.** Serial writes neither
  `var_densities` nor `var_num_obs`, so a serial multi-variable file cannot say that var1 was at
  0.3 and var2 at 0.2 — only `density: 0.4`, the reference variable's. `num_obs` means var0's
  count in serial (879) and the sum over all variables in a chunk (754). `density` is the
  *requested* value in serial and the *realised* one in a chunk. And no file records which
  overlap convention `overlap_target` uses, which is how the F1/F2 ambiguity propagates onto
  disk. Both paths should write the per-variable arrays the code already carries, agree on what
  `num_obs` and `density` mean, and state the overlap convention alongside achieved F1 and F2.
- **Squeezing constant dimensions** (see D5): serial squeezes, chunk files cannot.

---

## Scalability observations (independent of equivalence)

These bear on the stated purpose of the parallel path rather than on serial/parallel agreement.

### [done] S1 — single-variable worker peak memory scales with the *global* grid, not the chunk

`generate_hybrid_indices` builds `[i for i in range(total_points) if i not in lhs_set]`
(`generators/record_generator.py:213`), and each chunk runs it on the **global** shape.

Measured: global grid 10^7 points, 4 chunks, dense chunk array 0.020 GB, one worker peaked at
**0.49 GB above baseline** — 25x its own chunk. Cost is ~50 bytes and ~0.14 s per global grid
point, in every worker concurrently. Extrapolating to 10^10 points: ~500 GB and ~20 min per
worker. For single-variable runs, chunking currently raises peak memory rather than lowering it.

#### Worked instance of S1: a GLORYS12-shaped grid

Reported independently while preparing a real generation run, and confirmed here.

```
num_dims   = 3
ratio_dims = (365, 2041, 4320)          multiplier resolves to exactly 1
density    = 0.69
num_obs    = 2,220,591,672              shape (365, 2041, 4320), 3,218,248,800 sites
max_obs    = 10,281,517                 NTASKS = 216, split dim 2, 4320/216 = 20 per chunk
```

Every figure verified: sites, `num_obs`, `NTASKS`, and the split all land exactly.

Cost of the comprehension at this shape, per worker, over the **global** grid:

```
transient Python list   ~36 B/pt  = 115.86 GB
measured constant       ~50 B/pt  = 160.91 GB    <- peak; list and array coexist
resulting int64 array     8 B/pt  =  25.75 GB
```

`rng.choice` on that array does not add a further population-sized buffer (measured: peak RSS
tracks the array itself), so ~160 GB is the peak.

Against what the config actually needs:

```
dense chunk record array  =  119.19 MB
serial dense array        =   25.75 GB
```

So the chunking is well sized — 119 MB per worker — and S1 is the only obstacle. Because the
term scales with the global grid, more chunks do not reduce it, and serial hits the same wall.

**Fixed in stage B.** `RecordGenerator._ranks_to_local` maps ranks in a stratum's free sites to
local indices without materialising the complement, so peak memory is one hyperplane. Measured
on this exact config: one stratum, 514,025 observations, **0.02 s and no measurable allocation**,
against ~160 GB before.

Two further properties of this config, both verified:

- **No D3 drift.** `10,280,517 x 216 = 2,220,591,672`, exactly `num_obs`. The rounding is exact
  here, so the occupancy pattern would match serial.
- **The LHS stage contributes 365 of 2,220,591,672 observations (0.0000164%).** At this density
  the coverage guarantee it enforces is satisfied by chance with probability
  `1 - 10^-378,997` (see the appendix on departures from the idealised model), so the global
  machinery that S1 pays for buys nothing in this regime.


### [done] S2 — `build_multi_var_dataframe` cost is O(num_obs x grid_points)

`record[non_nan_mask][obs_idx]` re-masks the whole array inside the row loop
(`output/parquet_builder.py:99`). Holding observations fixed at 4000/var and growing the grid:
40k points 0.31 s, 360k 0.57 s, 1.44M 1.92 s.

**Fixed in D6d**, as a side effect of making the row order deterministic. Vectorised with one
`unique` over the union of the variables' flat indices and a `searchsorted` per variable:
40k 0.001 s, 360k 0.003 s, 1.44M 0.008 s, 9M 0.043 s -- 240x at 1.44M, and no longer scaling
with the grid.

### [done] S3 — the parallel path deletes the parquet output directory recursively

`shutil.rmtree(tmp_dir)` in `generate_data.py:842` resolves `tmp_dir` to
`dirname(parquet_tmp)`, which is the parquet output directory itself.

Measured: a `NOTES.md` and an `important_subdir/results.csv` placed in that directory survive a
serial run and are destroyed by a parallel one. Serial only removes `.nc`/`.parquet`/`_metadata`
files.

**Fixed together with S4**, same root cause. `parquet_tmp` names the scratch directory -- which
is what its docstring always said -- instead of being read with `os.path.dirname`, which
resolved to the output directory. Scratch chunks now live in their own directory, removed once
the merge succeeds. Cleanup goes through `PathManager.remove_scratch_dir`, which refuses to
delete a directory holding any file this run did not write, or any subdirectory, and says so
rather than proceeding.

### [done] S4 — workers race on shared parquet metadata

`ParquetBuilder.save_to_file` writes `_metadata`/`_common_metadata` when `chunk_id is None`, and
the worker passes `chunk_id=None` for its temporary chunk. Every worker therefore writes those
two files into the same directory concurrently, contradicting the "don't write metadata file"
comment three lines above.

**Fixed.** `save_to_file` takes an explicit `write_metadata` instead of inferring it from
`chunk_id`, and the workers pass `False`. The output directory ends up with exactly one
`_metadata`/`_common_metadata` pair, written by the consolidation.

### [done] S5 — worker logging

DEBUG level, dumps full coordinate arrays, and writes `worker_<id>.log` into the current working
directory rather than the output directory (`workers/parallel_worker.py:82`). It also used
`logging.basicConfig`, which configures the ROOT logger, so every library in the worker process
logged at DEBUG too. With 216 chunks that is 216 files dropped wherever the job was launched
from.

**Fixed.** A named per-worker logger, silent unless `TABRAY_WORKER_LOG=debug` (or `info`) is
set, writing beside the netCDF output rather than into the working directory. The coordinate
dump logs axis sizes instead of the arrays.

### S6 — chunk count is capped by the longest axis, and `max_obs` is not a memory knob

The grid is split along one dimension only, so the maximum number of chunks is the length of the
longest axis. The code enforces it:

```
ValueError: Dimension has size 54 but 55 chunks should be generated?
```

Measured on `[54,54,18,18,9,9]` (ratio 6:6:2:2:1:1): 11 of 48 swept configurations were refused
for this reason.

The consequence is a floor under per-worker memory. A 9.57 TB grid of shape
`[270,270,90,90,45,45]` cannot be cut finer than 270 chunks, so the dense record array is at
least 35.43 GB per worker, per variable, however low `max_obs` is set. A block decomposition
over two dimensions would allow 270x270 = 72,900 chunks and a 131 MB floor.

Related: chunk memory is `8 * prod(shape)/NTASKS * num_vars` bytes, and since
`num_obs ~= density * prod(shape)` and `NTASKS = ceil(num_obs/max_obs)`, this is approximately
**`8 * max_obs / density * num_vars` bytes**. At density 0.01 the README's default
`max_obs=10_000_000` implies ~8 GB per chunk per variable; at density 0.001, ~80 GB. The README
documents `max_obs` as "set based on available memory" without the density factor.

Location: `generate_data.py:_multiprocessing_setup` (`max_dim_size < NTASKS` check).


### [done] S7 — the merged netCDF is reproducible in content but not in bytes

`merge_nc` concatenates through `xr.open_mfdataset`, so the merge is dask-backed. Dask's
completion order varies between runs and netCDF4/HDF5 allocates object headers in write order,
so identical content lands at different file addresses.

Ten identical runs of one configuration produced two distinct file hashes, 8/2. Diffing them:

```
nc/d.nc          35740 vs 35740 bytes, identical=False
                 first differing byte at 4605, inside an OCHK marker
                 V0 ...\x03\x01\x9c'\x00...   V1 ...\x03\x01\x9cY\x00...
pq/d_0.parquet   19605 bytes, identical=True
pq/_metadata     identical=True
```

`OCHK` is an HDF5 object header continuation block and the differing bytes are file addresses
(`0x279c` against `0x599c`). Coordinates, occupancy masks, values and parquet rows were
bit-identical between the two variants.

Serial and the unmerged chunk files write from plain numpy and are unaffected; only the merged
file goes through dask.

Consequence for verification: **byte hashing is the wrong tool for merged output.** Equivalence
checks must compare coordinates, values and parquet rows, not file digests -- otherwise HDF5
layout noise reads as a behaviour change. This caused a false alarm during the dead-code
removal.

Also note this refines the D6 claim: serial and parallel agree on *content*, not on bytes.
Besides the HDF5 layout, some attributes still differ -- single-variable serial writes its
attributes on the variable (it saves a DataArray) while the merged file has them at dataset
level, the merge does not carry achieved overlap, and `description` differs.

**Root cause.** Not dask's completion order, as first recorded. `to_netcdf` on a dask-backed
dataset issues one store per *data variable* and runs them concurrently; HDF5 allocates each
variable's space on first write, so the variables land at the same addresses in a permuted order:

```
run0  var0:49212  var1:31636  var2:14060
run1  var0:31636  var1:14060  var2:49212
run4  var0:14060  var1:49212  var2:31636
```

This is why it is **multi-variable only**. Over 8 identical runs: every single-variable case
(`sv2d`, `sv3d`, `sv6d`) gave 1 hash, `mv2` gave 2, `mv3` gave 5. Parquet was always stable, and
coordinates always landed at the same addresses.

Forcing a synchronous dask scheduler does **not** fix it (5 distinct hashes of 6 became 3 of 6) --
the store order comes out of graph construction, not execution. `merged.load()` fixes it
completely but materialises the whole dataset.

**Fixed, for the memory rather than the bytes.** `_merge_netcdf_files` now writes one variable at
a time: `merged[[name]].to_netcdf(path, mode="w")` then `mode="a"` for the rest. Each call has a
single store, so allocation order is fixed. Every case above is now 1 hash over 8 runs.

The reason to take it is memory. Measured by lowering `RLIMIT_AS` after opening and merging a
382 MB, 3-variable dataset through the real method:

```
single to_netcdf call     fails at 900 MB   writes at 1200 MB
one variable at a time    fails at 600 MB   writes at  900 MB
```

Each write is still lazy and still streams; only the concurrency across variables is given up,
and the writes share one HDF5 file lock anyway. `RLIMIT_AS` caps virtual address space, which
numpy and HDF5 reserve generously, so only the comparison between the two is meaningful.

**A measurement note worth keeping.** Peak RSS (`ru_maxrss`) is useless here. It counts memory
freed back to the allocator but not returned to the OS, so sequential allocate-and-free looks the
same as holding everything: it reported peak rising with dataset size and *not* falling when the
dask chunk size was cut 8x, which would have suggested no streaming at all. The address-space cap
is the instrument that distinguishes them.

**Still true after the fix:** the merge is not free of the dataset size -- ~900 MB of address
space for 382 MB of data. `merge_nc` defaults to `False`, and for output too large for that the
per-chunk files remain the answer. The docstring previously claimed the merge "streams rather than
being materialised"; it now states the measured cost.



### [done] S8 — `merge_nc` could deadlock and hang the run

Found while verifying S7: the verification harness stopped dead twice in four runs, in the first
parallel case, with every worker finished and the parent using no CPU. A `faulthandler` dump gave
the reason -- two dask worker threads, each waiting on xarray's netCDF4 lock, one reading and one
writing:

```
Thread A  netCDF4_.py:114 _getitem    -> locks.py:64 __enter__   (reading a chunk file)
Thread B  netCDF4_.py:80  __setitem__ -> locks.py:64 __enter__   (writing the merged output)
```

`to_netcdf` on a dask-backed dataset opened from netCDF files takes that one lock on both sides.
With the default thread pool the read side and the write side can block each other permanently.

Not reproducible on demand: 0 hangs in 25 runs of the same case in a fresh process, and 0 in 12
rounds of serial-then-parallel inside one process. Both observed hangs happened while the machine
was busy with other work, which fits a timing-dependent race rather than a deterministic ordering
bug.

**Fixed structurally rather than by testing it away.** The merge write now runs under
`dask.config.set(scheduler="synchronous")`, so dask executes the graph in the calling thread.
With one thread there is no second thread to contend for the lock, and the deadlock cannot occur.
Nothing is materialised -- the graph is still walked chunk by chunk -- and the merge timing was
unchanged in measurement (1.67 s against 1.76 s).

Because the race could not be reproduced on demand, the argument for the fix is structural, not
statistical: absence of hangs in N runs would not have proved anything either way. Supporting
evidence: the harness completed cleanly on every run afterwards.

Scope: `merge_nc=True` only, which is not the default. The unmerged per-chunk output and the
parquet consolidation are not involved -- the whole stack is netCDF read/write.



### [done] A9 — the overlap target was rounded in every stratum

Found by comparing the tutorial notebooks against the code they branched from. `round(t_i * p_j)`
was computed in each stratum and the results summed, which is not the same as rounding the total.
Each stratum rounds to a whole cell, and on a coarse grid one cell is a large share of the
variable:

```
3x3, 8 observations, target 7/8   ->  p_j = 2,3,3   round gives 2,3,3 = 8 of 8   F1 1.000
3x3, 3 observations, target 1/3   ->  p_j = 1,1,1   round gives 0,0,0 = 0 of 3   F1 0.000
```

Both are reachable: 7 of 8 and 1 of 3 are valid placements. The docstring named the cost --
"a drift of order sqrt(num_strata)/2 on the global numerator" -- and the bound holds, one cell in
both cases. What it did not say is that one cell of eight is 0.125 of F1, which is the whole
illustration in `notebooks/tutorial3.ipynb`.

**Predictable before placement.** Given `p_j`, each variable's per-stratum counts, and the plane
size, the achieved overlap is computable exactly, clipping included -- verified against five grids
from 3x3 to 60x60. This is what the generation report in `docs/generation_report_plan.md` can use
to attribute a deviation rather than guess at it.

**Fixed for variables that vary along every dimension.** The total is decided once and apportioned
across strata by largest remainder within each stratum's bounds, the same treatment observation
counts already get from `ChunkUtils.apportion`. `RecordGenerator.stratum_counts` rederives var0's
per-stratum counts from the global LHS without placing anything, so a worker can apportion
globally.

**Not fixed for variables that drop a dimension**, on purpose. Their `p_j` counts distinct
*projected* cells, which depends on where var0 landed; a worker cannot know it for strata it does
not own. Applying the fix in serial and not in parallel would break the equivalence this whole
document is about, so both paths keep the per-stratum rule for those.

Achieved overlap, after:

```
3x3  target 0.875 -> 0.875     30x30 target 0.875 -> 0.875   (was 0.8875)
3x3  target 0.333 -> 0.333     mv3   target 0.500 -> 0.5006  (was 0.5028)
6x6  target 0.875 -> 0.875     mv3   target 0.300 -> 0.3003  (was 0.2958)
```

Two golden digests moved, `mv3` and `mvfixed`; `mv2` was already exact and `mvred` uses reduced
dimensions and falls back. Serial still equals parallel in all ten.


---

## Adjacent findings (not parallel-specific)

### [done] A1 — `validate_var_dims` crashes on list-of-lists `var_dims`

`if num_dims > var_dims[0]` compares int to list (`validators/parameter_validator.py:125`). The
README's own multi-variable examples use that form and raise `TypeError`. This also blocks
testing explicit per-variable dimension lists in the parallel path.

**Fixed.** The reference check now branches on the entry's type: a list of indices is completed
to every dimension, an int is compared as a number. Two further defects in the same function
went with it -- the caller's list was assigned rather than copied and then written at index 0,
and an empty sequence indexed out of range.

Verified on the README's own example: `num_dims=4`, `var_dims=[[0,1,2],[1,2,3],[0,2,3]]`. The
reference is completed to `[0,1,2,3]`, the split dimension lands on 2 (shared by all three
variables), and serial and parallel agree on values and parquet rows.

### [done] A2 — seed offsets collide

- `generate_without_overlap` uses `seed + var_idx + 1000`, which for `var0` is exactly the
  dimension-1 coordinate seed `seed + 1*1000`.
- `generate_lhs_rng` uses `seed + 1000` as well; its comment claims 10000.
- With the chunk offset `+100*chunk_id`, chunk 10 lands on `seed+2000`, the dimension-2
  coordinate seed.
- CLAUDE.md documents placement as `seed + var_idx + 2000`; that holds only for the overlap path.

Three further collisions were measured on live configurations, by drawing from both streams and
comparing the bits:

```
seed 42:    coord x0  ==  the density-range draw     (seed + 0*1000 is seed)
seed 5042:  coord x5  ==  var_dims selection, var0   (live at 6+ dimensions)
seed 6042:  coord x6  ==  constant coords, var0      (live at 7+ dimensions)
```

The first is live at any dimensionality. With a two-element `density` range, `var1`'s density came
out as `min + (max-min)*u` for the same `u` that became the first x0 coordinate.

A fourth was structural rather than arithmetic: `seed + 6000 + var_idx` carried no dimension
index, so a variable constant on two dimensions drew the same coordinate index on both.

**Fixed.** Every stream now comes from `stream(seed, tag, *index)` in
`data_sparsity/utils/streams.py`, with tags in a `Stream` class. Adding a purpose means adding a
tag where the existing values are visible, instead of picking an offset and hoping it misses.
Constant coordinates gained the missing dimension index. Verified: the seven streams involved in
the collisions above are pairwise distinct.

**A latent collision inside the new scheme, found while verifying it.** `SeedSequence` ignores
trailing zeros, so `default_rng([42,0,0])`, `default_rng([42,0])` and `default_rng(42)` are one
stream. With `COORDINATE = 0`, the x0 axis was therefore still `default_rng(seed)`, and
`stream(s, tag, 0)` equalled `stream(s, tag)`. No two *used* purposes collided — tags differ, and
same-tag index tuples all have the same length — so nothing generated was wrong, but the registry
did not deliver the guarantee it was written for. `stream` now shifts every component up by one,
so no component is ever zero and no tuple is a zero-extension of another; negative indices raise,
since they would cancel the shift. `tests/utils/test_streams.py` pins this, including a sweep over
every tag against every index tuple up to depth two.

**Cost: a full rebaseline.** Moving the coordinate seeds changes every coordinate axis and so every
output. Nine of the ten golden cases changed; `sv1d` did not, because it is one-dimensional with one
variable, no density range and no constant dims, so its only moved stream was x0 — which, under the
trailing-zero collapse, was still the old `seed + 0*1000`. The shift then changed all ten. Serial
and parallel remained identical in all ten cases throughout.

### [done] A3 — achieved overlap exceeds the target, in **both** modes

`overlap=0.5` with two variables produced 0.652. The non-overlapping draws are sampled from all
sites unused *by that variable*, rather than excluding the reference's sites, so they collide
with the reference at roughly its density. Serial and parallel agree, so this is not an
equivalence divergence — but the `overlap` parameter does not deliver what it states.

**Fixed in stage C**, as part of the stratified placement: the non-overlapping remainder is
drawn from cells held by *neither* variable, so achieved overlap equals the target instead of
picking up accidental coincidences at var0's density.

Confirmed a bug rather than a design choice: `docs/explainer_multivar.md` defines overlap as
realised coincidence, commit `7eee1c9` pinned that definition, and the code's own comment at
`multi_var_record_generator.py:391` labels the colliding draws "non-overlapping". It survives
because no test runs a generation and compares `overlap_actual` against `overlap_target` — the
suite tests the measurement function and the parameter parsing separately.

Magnitude: `achieved = t + (1-t)*(m0 - t*m1)/(S - t*m1)`, about `t + (1-t)*density(var0)`
(predicted 0.6476 vs 0.6524 measured). Always upward, worst at low targets with a dense
reference; `overlap=0` yields coincidence at var0's density rather than disjoint variables.
Not traced: whether `overlap='random'`, which takes `generate_without_overlap`, has the same
defect.

### A4 — README drift

Says the parallel path uses Dask for generation (it uses `ProcessPoolExecutor`; Dask is only
used for parquet consolidation), and its multi-variable examples use the `var_dims` form that
A1 rejects.


### [done] A5 — the LHS covered only the shortest axis, leaving unused coordinates

`docs/explainer.md` ("Sparse vs dense") only considers grids where every
coordinate is occupied at least once: an unused coordinate carries no
information and should not be part of the grid. The generator did not enforce
this.

`generate_lhs_indices` sized itself by `n_s = min(num_obs, min(shape))`. Since
each observation uses exactly one coordinate per axis, `n_s` points can touch at
most `n_s` coordinates on any axis, so only an axis of length `min(shape)` was
fully covered. Longer axes were left to the random fill, which at low density is
too sparse to cover them by chance.

Measured on shape `[4,7,10]` at the validator's own minimum density (18
observations), 300 seeds, average unused coordinates per run:

```
                          x0 (len 4)   x1 (len 7)   x2 (len 10)   seeds affected
before                         0.000        0.317         1.230          259/300
after stage A+B                0.000        0.373         0.000           69/300
after this fix                 0.000        0.000         0.000            0/300
```

x0 was covered by the LHS (shortest), x2 by stage B's stratum apportionment
(one observation per stratum), and x1 by nothing at all -- it was neither the
shortest axis nor the split dimension. In 2D no such axis exists, which is why
this only shows up from 3D with three distinct sizes.

**Fixed** by sizing the LHS with `max(shape)` and tiling permutations on axes
shorter than `n_s`, so every coordinate of every axis is used at least once.
This is affordable precisely because `compute_min_density` already guarantees at
least `max(shape)` observations.

### [done] A6 — `compute_min_density` is sufficient but not tight for non-cubic grids

`1/nmin**(d-1)` is exactly the coverage bound on hyper-cubic grids, where
`n**d / n**(d-1) = n = max(shape) = min(shape)` -- the same formula written two
ways. It was derived by hand for that case and is correct there.

On non-cubic grids the tight bound is `max(shape)/prod(shape)`: necessary
(covering an axis of length L needs at least L observations) and sufficient
(`max(shape)` points can cover every axis by tiling permutations). The current
formula sits above it, so there is a band of densities where coverage is
achievable but the validator refuses:

```
shape                   current -> num_obs   tight bound   ratio
[4, 7, 10]                            17.5            10   1.75x
[3, 9]                                 9.0             9   1.00x
[365, 2041, 4320]                 24,156.5         4,320   5.59x
[2, 3, 5, 7]                          26.2             7   3.75x
```

Not a defect -- a stricter bound is safe, and it is deliberately kept. But it
narrows the sparse end of the density sweep, by 5.59x for a GLORYS12-shaped
grid: the sparsest dataset the package can currently generate there has 24,157
observations where 4,320 would satisfy the definition.

**Applied.** `compute_min_density` now returns `max(shape)/prod(shape)`.

*The definitional call.* `explainer.md` stated the rule two ways: the worked
3x3 example says coordinates are used "once and only once", while footnote 3
says "all coordinates are occupied at least once". On a cubic grid these
describe the same point set, which is why the difference went unnoticed. Off
cubic they come apart, and "once and only once" has no solution at all: on
`[4,7,10]`, a set using each of the 10 coordinates of axis 2 exactly once has
10 points, and 10 points cannot use each of the 4 coordinates of axis 0 exactly
once. **At least once** is therefore the condition, which is what the footnote
already said and what the rest of the package assumes. No new convention was
invented; the general statement in the explainer was corrected to match the
footnote, and the bound follows from it.

*Verified reachable, not just permitted.* Generating at exactly the new minimum
and checking that every coordinate on every axis carries at least one
observation, serial and parallel:

```
grid              obs   max(axis)   serial     parallel
[10, 10, 10]       10          10   covered    covered
[4, 7, 10]         10          10   covered    covered
[5, 20, 10]        20          20   covered    covered
[50, 10, 1, 1]     50          50   covered    covered
[2, 3, 5, 7]        7           7   covered    covered
```

Three of these were rejected before: `[4,7,10]` demanded 17.5 observations,
`[2,3,5,7]` demanded 26.25, `[5,20,10]` demanded 40.

*The sharpest case was not in the original table.* Any grid with an axis of
length 1 has `nmin = 1`, so the old bound was `1/1**(d-1) = 1.0` -- the only
admissible density was 1.0. Such grids could be generated purely gridded and no
other way, which for a package built to sweep occupancy removed the entire
sweep. `[50,10,1,1]` now runs at 50 observations of 500.

*No data change.* The bound only gates validation, so all ten golden digests
are unchanged and serial still equals parallel in all ten. 461 tests pass.

Documented in `docs/explainer.md` ("Minimum density, on any grid", which also
fills the `[TD ADD FORMULA]` placeholder that stood there), in the README under
the `density` parameter, and in CLAUDE.md.



### [done] A7 — per-variable dtype is not a parameter

Everything is float64: the record arrays, the observation values and the coordinate axes. Real
data is not. GLORYS12 and most CMEMS products are float32, and many are packed `int16` with
`scale_factor`/`add_offset`, i.e. 2 bytes per value.

**Proposed**: a `var_dtype` parameter accepting a scalar applied to every variable, or one entry
per variable, e.g. `["float64", "str", "float32", "int", "int"]`.

**Why it is not just a memory knob.** Changing dtype moves the array/tabular crossover, which is
the headline result of the comparison. Measured on a 50x50x50 grid, netCDF zlib-4 against
parquet snappy:

```
 density |  ----- float64 -----  |  ----- float32 -----
         |    nc.z    pq  ncz/pq |    nc.z    pq  ncz/pq
    0.01 |   0.05M  0.02M   2.59 |   0.03M  0.01M   2.44
    0.10 |   0.23M  0.14M   1.60 |   0.12M  0.09M   1.30
    0.20 |   0.36M  0.28M   1.31 |   0.18M  0.18M   1.04
    0.40 |   0.57M  0.55M   1.04 |   0.28M  0.34M   0.82
    0.90 |   0.85M  1.18M   0.72 |   0.42M  0.72M   0.58
```

The array format is smaller where `ncz/pq < 1`: around density 0.4 in float64, around 0.2 in
float32. netCDF halves exactly (dense array, measured 0.503) while parquet only falls to ~0.65,
because it stores coordinates as well as values and encodes float32 less efficiently per value.
Different scaling on the two sides, so dtype is an axis of the comparison rather than a constant
factor on it.

**Constraints for whoever implements it:**

* **Coordinates must stay float64.** Sorted uniforms in [0,1) collide in float32 from about
  N = 100,000, because the minimum gap among N order statistics goes as 1/N^2 while float32
  resolution near 1.0 is 1.19e-7. Measured duplicates: 0 at N=4,320, 189 at N=100,000, 19,613
  at N=1,000,000. xarray dimension coordinates must be unique and the tests assert it. Axes cost
  `sum(shape)` values, so there is no memory reason to shrink them.
* **Values lose uniqueness.** float32 has only ~1.07e9 representable values in [0,1), so more
  than that many observations cannot be distinct, and duplicates become certain above ~1e5.
  Several tests assert distinct observation values; they encode a float64 assumption.
* **Integer and string types have no NaN**, and the empty-site marker is `np.nan` throughout
  (`RecordGenerator.initialize_record`, and `~np.isnan(record)` in every consumer). These would
  need a masked array or a `_FillValue` sentinel, which itself changes the array/tabular
  comparison -- netCDF stores the fill value at every vacant site, parquet stores no row at all.
* **Strings in netCDF** are either variable-length or fixed-width char arrays, with storage
  behaviour unlike any numeric type. Worth treating as its own case rather than one more dtype.

Memory is not the motivating reason: float32 halves the record array, where stage B already cut
the binding cost by ~1300x. A GLORYS12 chunk is 119 MB; float32 makes it 60 MB. The case for
this parameter is realism and coverage of the comparison, not footprint.



### [done] A8 — the two-element density form collapsed, and var0 was assigned by a coin flip

Three coupled defects in how `density=[a, b]` was handled.

* `validate_density_refvar` overwrote `density[0]` with the maximum. For a two-element list --
  a `[max, min]` range, not per-variable densities -- that collapsed the range to a point, so
  every variable got the same density. Measured: `[0.3, 0.5]`, `num_vars=2`, 100 seeds ->
  accepted 100, densities collapsed to equal 100.
* `from_two_element_list` then assigned min and max to the two variables with a coin flip, so
  var0 depended on the seed and was not reproducible from the arguments.
* `validate_reference_is_largest` rejected the half where var0 drew the minimum. Measured:
  `[0.5, 0.3]`, 100 seeds -> accepted 57, rejected 43.

So the form the README documented as "multiple variables with different densities" either gave
every variable the same density, or worked on roughly half of seeds.

**Fixed.** The two-element form is `[max, min]`: var0 takes the maximum, some other variable
takes the minimum, and any further variables are drawn uniformly from `[min, max]`. With two
variables the range is exactly the two prescribed densities. `validate_density_refvar` raises
instead of overwriting when `density[0]` is not the largest, for both the range form and the
full per-variable list, so a user's numbers are never silently rewritten.

No data change: all ten golden digests are unchanged. What changed is that var0 is now the
maximum for every seed rather than for about half of them, and the rejections are gone.

Ascending input raises rather than being silently reinterpreted. Nine of the repository's
thirteen two-element density literals were already descending; the four ascending ones (two
README examples, two tests) were corrected.


---

## How this was measured

### Coverage

1D-4D; densities 0.0157-1.0; NTASKS 1-8; 1-3 variables; `overlap` numeric / list / `'random'`;
`fixed_overlap` on and off; constant dims both on and off the split dimension; `max_workers` 1
vs default; grids up to 10^7 points.

Not covered: true HPC scale; float `ratio_dims`; explicit per-variable dimension lists (blocked
by A1).

### Harness

Session-scoped, not checked in. Shape of it:

- `run(tag, params, max_obs=None)` — builds `GenerateData`, generates into
  `<tag>/nc/data.nc` and `<tag>/pq/data.parquet`, captures stdout.
- `load(g, d)` — reads back from disk; for parallel, `xr.concat` of `data_*.nc` along
  `x{dim_split}` plus `pd.read_parquet` of the parquet directory.
- `case(name, params, max_obs)` — runs both modes and reports, per coordinate: identical /
  max abs diff / sortedness; per variable: obs count, NaN-mask equality, value equality; per
  parquet: row count, column list, equality as written and after sorting.

The pool uses the spawn context, so a driver script needs an `if __name__ == "__main__":` guard
or the children re-execute it.

---

## Appendix: grid shape and memory scaling

Reference for reasoning about which of the findings above actually bind for a given
`ratio_dims`. All figures are float64 (8 B/point) and per worker unless stated.

### Quantities

| column | definition | why it matters |
|---|---|---|
| `k` | base unit; axes are `ratio * k`. The code derives `k` from `num_obs`, `density` and the ratios (`nb_coords_dim1`) and rounds it to an integer | the only free parameter once the ratio is fixed |
| `grid points` | `prod(shape)` — every cell, occupied or not | grows as `k^num_dims` |
| `dense grid` | `grid points x 8 B` | what a serial run allocates; the size the parallel path exists to get past |
| `axis` | `max(shape)`, the longest dimension | the dimension the code splits along |
| `axis bytes` | `axis x 8 B` — one full coordinate axis | the unit for any per-axis handling of the split dimension (D1) |
| `max chunks` | `= axis` | hard ceiling; exceeding it raises (S6) |
| `min chunk` | `dense grid / max chunks` | smallest dense record array a worker can get, per variable, reached only at maximum chunk count. Counts the record array only — not the DataFrame or write buffers |
| `S1/worker` | `grid points x ~50 B` (measured constant) | cost of the global-grid comprehension each single-variable worker builds today (S1) |

Because `k` is rounded to an integer, the realised grid lands near but not on any target size.

### Ratio 6:6:2:2:1:1 (6D), shape `[6k, 6k, 2k, 2k, k, k]`

```
   k  shape                        grid points  dense grid    axis  axis bytes  max chunks  min chunk  S1/worker
   9  [54,54,18,18,9,9]             76,527,504    612.2 MB      54      0.4 KB          54    11.3 MB    3.83 GB
  20  [120,120,40,40,20,20]      9,216,000,000    73.73 GB     120      1.0 KB         120   614.4 MB  460.80 GB
  21  [126,126,42,42,21,21]     12,350,321,424    98.80 GB     126      1.0 KB         126   784.1 MB  617.52 GB
  45  [270,270,90,90,45,45]  1,195,742,250,000     9.57 TB     270      2.2 KB         270   35.43 GB   59.79 TB
```

Worked example, k=45:

```
shape         = [270, 270, 90, 90, 45, 45]
grid points   = 270*270*90*90*45*45   = 1,195,742,250,000
dense grid    = 1.196e12 x 8 B        = 9.57 TB
axis          = max(shape)            = 270
axis bytes    = 270 x 8 B             = 2,160 B
max chunks    = axis                  = 270
min chunk     = 9.57 TB / 270         = 35.43 GB   (per variable)
S1/worker     = 1.196e12 x ~50 B      = 59.79 TB
```

Reading: in 6D the axis is negligible (2.2 KB against 9.57 TB) because the grid is a product of
six axes while the axis is one of them. What binds instead is S6 — 270 chunks maximum, 35 GB
minimum per worker — and then S1, which is three orders of magnitude worse again.

Anisotropy helps the chunking here: at 9.57 TB this ratio gives a 270-long axis, while a 6D
hypercube of the same size gives 104.

### Ratio 10000:10:1:1 (4D), shape `[10000k, 10k, k, k]`

```
   k  shape                     grid points  dense grid     axis  axis bytes  max chunks  min chunk  S1/worker
   6  [60000,60,6,6]            129,600,000     1.04 GB    60000    480.0 KB      60,000    17.3 KB    6.48 GB
  18  [180000,180,18,18]     10,497,600,000    83.98 GB   180000     1.44 MB     180,000   466.6 KB  524.88 GB
  19  [190000,190,19,19]     13,032,100,000   104.26 GB   190000     1.52 MB     190,000   548.7 KB  651.61 GB
  59  [590000,590,59,59]  1,211,736,100,000     9.69 TB   590000     4.72 MB     590,000   16.43 MB   60.59 TB
```

The opposite regime. At 9.69 TB this ratio chunks 2,185x finer than 6:6:2:2:1:1 did at the same
size: 590,000 chunks against 270, and a 16.43 MB floor against 35.43 GB. S6 stops binding.

In exchange, the axis stops being free. It is still small in absolute terms (4.72 MB at 9.69 TB),
but it is no longer negligible *relative to a chunk* once the chunk count is high:

```
k=59, shape [590000, 590, 59, 59]
   NTASKS=     16  chunk= 605.87 GB  axis/chunk = 0.0000
   NTASKS=    256  chunk=  37.87 GB  axis/chunk = 0.0001
   NTASKS=  8,192  chunk=   1.18 GB  axis/chunk = 0.0040
   NTASKS= 65,536  chunk= 147.92 MB  axis/chunk = 0.0319
   NTASKS=590,000  chunk=  16.43 MB  axis/chunk = 0.2873
```

At small `k` the axis can exceed the chunk outright:

```
k=6, shape [60000, 60, 6, 6], grid 1.04 GB, axis bytes 480.0 KB
   NTASKS=     16  chunk=  64.80 MB  axis/chunk =  0.007
   NTASKS=    256  chunk=   4.05 MB  axis/chunk =  0.119
   NTASKS=  2,160  chunk=  480.0 KB  axis/chunk =  1.000   <- crossover
   NTASKS=  8,192  chunk=  126.6 KB  axis/chunk =  3.793
   NTASKS= 60,000  chunk=   17.3 KB  axis/chunk = 27.778
```

### Crossover rule

One axis costs more than one chunk when `NTASKS > prod(shape)/axis`, i.e. more chunks than the
product of the non-split dimensions. Since `NTASKS <= axis` (S6), the crossover is reachable at
all only when `axis^2 > prod(shape)`:

```
ratio (10000, 10, 1, 1)    reachable iff k^2 < 1000    -> yes, for k < 32
ratio (6, 6, 2, 2, 1, 1)   reachable iff k^4 < 0.25    -> never
ratio (1, 3)               reachable iff 3 > 1         -> yes, at any k
```

So the two ratios sit on opposite sides: 6:6:2:2:1:1 is chunk-limited and axis-free;
10000:10:1:1 is axis-sensitive and chunks freely. In both, S1 exceeds every other quantity in
the table by three to six orders of magnitude.

---

## Appendix: departures from the idealised model

The obvious mental model of the generator is: "`num_obs` sites drawn uniformly without
replacement from the grid, with each axis carrying iid U(0,1) coordinates, sorted." The code
departs from that model in the ways below. Listed so that generated datasets can be predicted
rather than inspected after the fact. Ordered by how large the effect is.

### 1. Placement is a hybrid Latin hypercube, not a uniform sample

`generate_hybrid_indices` draws in two stages:

- Stage 1, `n_s = max(shape)` points forming a Latin hypercube. The longest axis gets
  `permutation(dim_size)`; shorter axes get tiled permutations, truncated to `n_s` and shuffled.
  The per-axis lists are paired positionally. **Every coordinate of every axis is used at least
  once**, and on the longest axis exactly once.

  (Before the A5 fix this was `n_s = min(num_obs, min(shape))`, which covered only the shortest
  axis. The paragraph below records that behaviour.)
- Stage 2, the remaining `num_obs - n_s` points uniformly without replacement over the
  complement.

So the sample is a mixture, and the mixing weight is `n_s/num_obs`:

- At minimum density (`num_obs == min(shape)`) the sample is **entirely** a Latin hypercube. It
  is not a uniform random subset and must not be modelled as one.
- At high density the LHS share becomes negligible — 0.000195% for the GLORYS12 config
  (`n_s = 4320` of 2,220,591,672 observations).

**[done]** This used to be weaker than `generate_lhs_indices`' docstring claimed: "each
coordinate in each dimension is used at least once" held only for axes of length `n_s`, i.e. the
shortest. For `(365, 2041, 4320)` at minimum density, dim 0 was covered while 1,676 of dim 1's
coordinates and 3,955 of dim 2's were empty. Sizing the LHS by `max(shape)` fixed this, and
`generate_lhs_indices`, `generate_hybrid_indices` and `generate_stratified_indices` now all raise
rather than emit a placement that cannot cover every axis. See A5.

### 2. [done] Coordinates on the split axis are stratified in parallel mode

Serial draws `N` iid uniforms per axis and sorts, so counts in any sub-interval are Binomial.
Parallel confines each chunk to its own sub-interval with a fixed count, so those counts have
zero variance. This is D1. For the GLORYS12 config it means the longitude axis carries exactly
20 coordinates in each 1/216 of [0,1), rather than a Binomial(4320, 1/216) number of them.

Non-split axes are unaffected and match the serial model exactly.

Closed by stage A: the split axis is now the serial draw, sliced by index, so sub-interval
counts are Binomial again.

### 3. Realised `num_obs` and `density` can differ from the requested values

Validation is deliberately part-soft: `num_obs` is adjusted for consistency with `density` and
the grid, and the corrected configuration is printed. Any prediction must use the
post-validation attributes, not the constructor arguments. The additional parallel-only drift
from chunk rounding (D3) is **[done]** — `num_obs` now survives chunking unchanged.

### 4. [changed] Per-chunk observation counts are now deterministic

**Before stage B**, with global LHS filtering, a chunk received however many of the global sites
fell in its index range — a hypergeometric count. For the GLORYS12 config: mean 10,280,517,
sd 1,781 (0.0173% of the mean), so roughly +/-5,343 at three sigma.

**After stage B** the fill is apportioned across strata by largest remainder, so per-stratum and
therefore per-chunk counts are exact and have **zero variance**. This is the one deliberate
statistical change of the stratified design, and it applies to serial as much as to parallel, so
the two still agree exactly.

What it costs, measured against an ideal simple random sample: identical marginal occupancy for
every site and identical totals; variance removed from counts along the split axis; variance
along other axes inflated by a factor `1 + 1/H` where `H` is the sites per stratum — 5% on a toy
20-site stratum, 0.00013% for GLORYS12. Chunk-aligned range queries now return deterministic row
counts, which removes a nuisance variable from read benchmarks but means per-stratum counts can
no longer be treated as a random variable.

### 5. [fixed] Coordinate values and site placement are not drawn from independent streams

The additive seed offsets made the x1 axis and the placement indices functions of one PCG64
stream. A2 replaced them with `stream(seed, tag, *index)`, so the two are independent by
construction.

### 6. [removed] Silent duplicate fallback at extreme density

`generate_hybrid_indices` drew the remainder **with replacement** when the complement ran short,
and the duplicates collapsed on assignment, so fewer distinct sites were occupied than `num_obs`
asked for, with no warning. It was unreachable -- `dim_split` is always set, so placement went
through `generate_stratified_indices` -- and the function has been deleted.

### 7. [fixed] Multi-variable only

Achieved overlap exceeding the target (A3), placement re-randomised per chunk (D4), and a
variable constant on the split dimension landing at `NTASKS` coordinates instead of one (D5) are
all closed. What remains is the granularity in A9: a reduced-dimension variable keeps per-stratum
rounding, and no grid can express a target finer than one cell of the reference footprint.

### Computing coverage probabilities

For "does any coordinate end up with no observations", fix an axis `k` and one of its
coordinates. The slice perpendicular to it holds `m = prod(shape)/shape[k]` sites. Sampling
`n` sites without replacement from `N` total, the exact probability that none falls in the
slice is hypergeometric:

```
P = C(N-m, n) / C(N, n) = prod_{j=0}^{m-1} (N - n - j) / (N - j)
```

The Bernoulli approximation `P ~= (1-density)^m` is accurate to about four significant figures
in log10 at these sizes, and is slightly conservative — sampling without replacement spreads
points marginally more evenly than iid occupancy, so the exact value is smaller.

For shape `(365, 2041, 4320)` at density 0.69:

```
 dim  coords  slice sites m   exact log10 P   approx log10 P
   0     365      8,817,120      -4,496,446       -4,484,725
   1    2041      1,576,800        -802,395         -802,021
   2    4320        744,965        -379,001         -378,918
```

A union bound over every coordinate of every axis adds `log10(shape[k])` to each row and takes
the largest: `log10 P(any empty coordinate anywhere) <= -378,997`, dominated by dim 2 because
the longest axis has the thinnest slices. Stage 1 of the LHS makes dim 0 certain rather than
merely near-certain, since an axis of length `n_s` receives a full permutation.
