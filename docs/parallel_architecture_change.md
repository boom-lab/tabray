# Parallel generation: the architecture that was, the one that is, and why it changed

This records an architecture that was removed, so that the deletions in the history read as a
replacement rather than a disappearance. It covers only the parallel generation path.

## The problem both designs solve

The parallel path exists to generate larger-than-memory datasets on HPC. It must produce the
same data as a single-threaded run: the same occupied sites, the same values, the same
coordinates. Chunks are generated in separate processes with no shared state, so equivalence has
to come from each worker deriving its share from the seed alone.

## The old architecture: regenerate the global draw, keep your slice

Every worker reconstructed the whole global random draw and discarded the part outside its
chunk.

```
worker k:
    coordinates   draw n_k uniforms inside [div_k/N, div_{k+1}/N), the chunk's value sub-range
    sites         regenerate the ENTIRE global hybrid LHS over the global grid,
                  then keep the points whose split-dim index falls in the chunk
    values        fresh RNG seeded seed + 1000 + 100*chunk_id
    multi-var     place each variable within the chunk, seeds offset by chunk_id
```

Supporting machinery, all now removed:

| piece | what it did |
|---|---|
| `generate_global_lhs_indices_for_chunk` | regenerate the global LHS, filter to the chunk's index range |
| `generate_lhs_rng` | hand the worker an RNG "advanced" to its position -- in the end a no-op |
| `calculate_lhs_draws_per_generation` | count the draws one LHS generation consumes, so a worker could skip ahead |
| `generate_rngs(chunk_id, dim_split, obs_in_chunk)` | fast-forward the split dimension's stream by `chunk_id * obs_in_chunk` draws |
| `generate_split_dimension_range` | compute the chunk's value sub-range `[a_k, b_k)` |
| `assign_rngs_to_dimensions` | pass-through left over from an earlier signature |
| `OverlapIndexMapper` | map the reference variable's indices onto another variable's compatible coordinates |

### Why it was built that way, and why it could not stay

The dead code records two successive attempts.

**Attempt one: advance the stream.** Give each worker the same generator and skip the draws the
previous chunks would have made. `calculate_lhs_draws_per_generation` exists to count exactly
how many draws to skip. It was abandoned, and it could not have worked: serial coordinates are
the **order statistics** of the whole sample -- the sorted result -- so chunk k's coordinates are
not draws `k*n` through `(k+1)*n` of any stream. No amount of skipping recovers them.

**Attempt two: regenerate and filter.** Correct, and it did deliver matching occupancy for the
single-variable case. Two costs killed it:

* **Memory.** Regenerating the global draw means touching the global grid. The fill stage built
  `[i for i in range(total_points) if i not in lhs_set]`, roughly 50 bytes per grid point, in
  every worker at once. On a GLORYS12-shaped grid (3.2e9 sites) that is ~160 GB per worker
  against a 119 MB chunk. Chunking *raised* peak memory instead of lowering it, and serial hit
  the same wall.
* **It only ever worked for one variable.** Multi-variable placement was chunk-local with
  chunk-offset seeds, so the sites differed from serial entirely.

Coordinates had a third problem. Confining a chunk to its own value sub-range and requiring
exactly `n_k` points in it makes the axis *stratified* rather than iid: the count in an interval
had zero variance where serial gives a Binomial count.

## The new architecture: the stratum

The grid is partitioned into **strata** -- one hyperplane per index along the split dimension.
A **chunk** is a contiguous run of strata. The stratum is fixed by the grid; the chunk is set by
`max_obs`. Separating them is what makes the output independent of how the run is parallelised.

```
global, and cheap:
    coordinate axes     sort(rng_d.uniform(0,1,shape[d]))      one vector per dimension
    LHS stage           max(shape) points covering every axis  O(max(shape))
    apportionment       how many sites each stratum gets       O(num_strata)

per stratum j, from (seed, j) and nothing else:
    fill sites          drawn inside the hyperplane
    values              drawn from the same stream, after the sites
    overlap             each variable takes its share of var0's footprint in THIS hyperplane
```

A worker runs the global steps itself -- they are a vector, a few hundred points and a list of
counts -- then runs the per-stratum steps for its own strata. Serial does the same for all of
them.

### Why this is equivalent by construction

Two facts about strata do the work:

* **Sites in different strata cannot collide**, because they differ in the split-dimension index.
  So sampling without replacement never has to span strata: the only global decision is *how
  many* sites each stratum gets, which is a vector of integers.
* **Two observations coincide only if every index matches**, including the split one. So overlap
  is always within a stratum, and a per-stratum overlap target aggregates to the global one.

Because a stratum depends only on `(seed, stratum)`, any caller producing a subset of strata
produces exactly the slices a caller producing all of them would. Serial and parallel agree
without arranging for streams to line up, and regrouping strata into different numbers of chunks
does not move a single value.

Two things still have to stay global, and both are cheap:

* the **coordinate axes**, because a chunk slices the sorted global axis by index rather than
  drawing its own -- `shape[d]` floats;
* the **LHS stage**, because its guarantee (every coordinate of every axis used at least once)
  spans strata and no stratum can enforce it alone -- `max(shape)` points.

### What replaced what

```
generate_split_dimension_range          -> slice the sorted global axis by index, in the worker
generate_global_lhs_indices_for_chunk   -> RecordGenerator.generate_stratified_indices
generate_lhs_rng                        -> per-stratum RNG, default_rng([seed, tag, stratum])
calculate_lhs_draws_per_generation      -> nothing; no stream is skipped any more
generate_rngs(chunk_id, ...)            -> generate_rngs(seed, num_dims); no advancement
assign_rngs_to_dimensions               -> nothing; it returned its argument
OverlapIndexMapper                      -> the per-stratum overlap step inside
                                           MultiVarRecordGenerator.generate_multivar_stratified
```

The complement array that made the old design unaffordable is gone too: picking sites that avoid
an excluded set is done by mapping ranks through `RecordGenerator._ranks_to_local`, which never
materialises the complement.

## What it bought

* One GLORYS12 stratum: ~160 GB and minutes per worker, to 0.02 s and no measurable allocation.
* Single- and multi-variable parallel output matches serial: coordinates, occupancy, values, and
  parquet rows as written.
* `max_obs` became purely a memory and scheduling knob. It no longer changes the data.

## What it cost

* Per-stratum counts are exact rather than hypergeometric, so counts along the split axis have
  no variance where simple random sampling gives some. Marginal occupancy of every site and the
  totals are unchanged; variance along other axes is inflated by `1 + 1/H`, with `H` the sites
  per stratum -- 0.00013% for GLORYS12.
* The split dimension must be one every variable varies along, so a grid whose longest axis is
  dropped by some variable is split on a shorter one, lowering the maximum chunk count.

## Writing the merged file

`generate(merge_nc=True)` concatenates the chunk netCDF files into one. The write is shaped by
two constraints that are not visible from the code.

**One variable per `to_netcdf` call.** Handing the whole dataset to `to_netcdf` issues one dask
store per data variable and runs them concurrently. HDF5 allocates each variable's space on first
write, so the variables land at the same addresses in whatever order the stores finish, and
identical data produces a different file on every run. Over 8 identical runs, single-variable
cases gave 1 hash, a 2-variable case gave 2 and a 3-variable case gave 5. A synchronous scheduler
does not fix this — the order comes out of graph construction, not execution — but one store per
call does.

Writing them in sequence also costs less memory, which is the better reason to do it. Merging a
382 MB, 3-variable dataset under a lowered `RLIMIT_AS`:

```
    single to_netcdf call     fails at 900 MB   writes at 1200 MB
    one variable at a time    fails at 600 MB   writes at  900 MB
```

**Single-threaded.** xarray guards the netCDF4 library with one lock, and this write takes it on
both sides: it reads the chunk files and writes the output through the same lock. With dask's
default thread pool, a thread reading a chunk and a thread writing the output can block each other
permanently. It is rare and load-dependent — twice in four runs of the verification harness, and
never in 25 runs of the same case on its own — but a hang cannot be recovered from, so the write
runs in the calling thread, where two threads cannot contend. Nothing is materialised: dask still
walks the graph chunk by chunk.

**What this does not buy.** The merge is still not free of the dataset size — roughly 900 MB of
address space for 382 MB of data. For output larger than that, leave `merge_nc` at its default
`False` and keep the per-chunk files.
