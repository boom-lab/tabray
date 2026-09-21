# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Synthetic-data generator used to compare read/manipulation performance of **array** storage (netCDF/xarray) against **tabular** storage (parquet/pandas) across a range of grid occupancies, from purely gridded to maximally sparse. `docs/explainer.md` and `docs/explainer_multivar.md` define the terminology the code uses (site, density/sparsity, gridded vs irregular, overlap) — read them before changing generation semantics.

The distribution is named `tabray` (`pyproject.toml`); the import package is `data_sparsity`.

## Commands

```bash
# environment (conda, installs the package editable via pip)
conda env create -f environment.yml && conda activate tabray
# or
pip install -e ".[dev]"          # dev = pytest, pytest-cov, pylint

pytest                            # 524 tests, ~50s; -v and --tb=short come from pyproject
pytest tests/generators/test_overlap_calculator.py
pytest tests/generators/test_coordinate_generator.py::TestGenerateDimensionCoords
pytest -k "overlap"
bash tests/run_all_tests.sh       # full suite + HTML/terminal coverage

pylint data_sparsity/<module>.py  # every file must score >= 8
```

Notebooks under `notebooks/` must be run with `notebooks/` as the working directory: `run_case(..., base_dir="./tutorial1")` writes into `notebooks/tutorial1/<case>/{netCDF,parquet}/`.

`tests/diagnostic_coordinate_identity*.py` are standalone scripts, not collected by pytest (they don't match `test_*.py`). Run them with `python` when investigating serial/parallel coordinate divergence.

## Architecture

`GenerateData` (`data_sparsity/generate_data.py`, the only user-facing class) is an orchestrator. It holds no generation logic; it validates, configures, then delegates:

```
GenerateData.__init__  ->  validators/   ParameterValidator, DimensionValidator, SparsityValidator
                       ->  config/       MultiVarSparsityConfig, MultiVarDimensionsConfig, MultiVarOverlapConfig
GenerateData.generate  ->  generators/   CoordinateGenerator, MultiVarRecordGenerator (-> RecordGenerator,
                                         ObservationGenerator, OverlapIndexMapper, OverlapCalculator)
                       ->  output/       NetCDFBuilder, ParquetBuilder, PathManager
                       ->  workers/      generate_chunk  (parallel path only)
```

### Per-variable encoding

`dtype`, `pack` and `fill_value` (constructor, scalar or one per variable) go through
`VariableEncoding` (`data_sparsity/output/variable_encoding.py`). Default is `float64` with NaN,
which emits **no** xarray encoding entry at all, so the default output is byte-identical to what
the code wrote before the parameter existed.

`dtype` says what a variable holds (`float64`, `float32`, or `int8`/`int16`/`int32` for counts
and flags); `pack` says how a *float* is compacted on disk (`int8`, `int16`, `int32`, or None).
They are separate axes because a file holds both kinds: Argo stores `TEMP` as a plain `float32`
and `CYCLE_NUMBER` as a plain `int32`. Packing an integer dtype raises.

`value_range` spans the values, inclusive: `(0, 1)` for floats, `(0, 100)` for integers. For a
packed float it also sets the packing scale, so a value cannot land outside its own grid. For an
integer variable the width is the column's cardinality, which is the knob that matters for
storage — `(0, 8)` gives flag-like data that dictionary- and run-length-encodes, while a wide
range behaves like noise.

`VariableEncoding.to_stored` maps `ObservationGenerator`'s raw `uniform(0, 1)` draws onto what a
variable holds, once, before either format is written. Integers come from transforming that draw
(`lo + floor(u * (hi - lo + 1))`), never from `rng.integers`: a different RNG method consumes the
stream differently, so changing a variable's dtype would shift every later draw and move the
occupied sites. Under the transform, dtype changes values and leaves placement identical, which
is what a storage comparison needs. The `floor` over `hi - lo + 1` bins matters too — rounding
over `hi - lo` gives the end bins half width and the extremes half the frequency.

A plain integer variable becomes a **nullable** parquet column (`Int16`), so a vacant site is a
real null rather than forcing the column to float. Its fill value must sit outside `value_range`,
checked in the constructor: the same collision the packing scale avoids by reserving a code.

Scientific netCDF has no single convention, which is why this is per variable rather than a mode.
Two real files sit in the repo and disagree on nearly everything:

| | GLORYS12 (`GLOBAL_MULTIYEAR_PHY_001_030/`) | Argo Sprof (`argo/`) |
|---|---|---|
| dtype | `int16` packed, scale + offset | `float32` plain |
| missing | `-32767` | `99999.0`, not NaN |
| compression | zlib **1**, no shuffle | zlib **4**, with shuffle |
| coordinates | `float32` | `float64` |

An integer dtype implies packing. Two things there are easy to get wrong and are covered by
tests: the fill code must be **reserved** (mapping the value range onto the full integer range
puts the minimum value on `_FillValue`, and those cells read back as missing — GLORYS reports
`valid_min = -32760` for this reason), and the dtype of `scale_factor`/`add_offset` decides what
xarray decodes *to*, so they are written as `float32` where that is precise enough, giving a
`float32` array rather than a promoted `float64` one. `int32` is the exception: its step is finer
than `float32` spacing, so it keeps `float64` parameters.

Values are quantised once, by `GenerateData._quantize` and its worker counterpart, before either
format is written — so netCDF and parquet hold the same numbers and the serial/parallel digest
check still means something. The scale is derived from `VariableEncoding.VALUE_RANGE`, matching
`ObservationGenerator`'s `uniform(0, 1)`, never measured from the data, because a parallel chunk
never sees the whole array.

### Layout

`layout` (`"scattered"` default, or `"padded"`) and `padded_dim` decide where the occupied cells
sit, which `density` says nothing about. It matters: at a fixed occupancy of 0.149 the compressed
array is 1.15 MB scattered against 0.55 MB padded, and parquet is smaller than netCDF under
scattered while netCDF is smaller under padded — the sweep's headline answer depends on it.
`docs/layout_plan.md` carries the measurements and the design.

`padded` is implemented as its own placement, `RecordGenerator.generate_padded_indices`, dispatched
from `generate_stratified_indices`. Each combination of the dimensions other than the split and the
padded one is a **line**, and a line holds positions `0..k-1`. Coverage comes from the construction
rather than from the LHS stage, which is skipped: every line holds at least one observation, which
uses every coordinate of the split axis and of every axis but the padded one, and one line runs the
full length, which uses the padded axis. So nothing sits outside the pattern.

Three consequences. The **minimum density is higher** — `prod(shape)/n_padded + n_padded - 1`
observations, so `compute_min_density` takes `padded_dim`. **`overlap` raises**, because with every
variable filling a prefix of the same axis the intersection is `min(k_0, k_i)` and F1 is the ratio
of the densities; Argo confirms it, NITRATE at 0.161 occupancy against TEMP's 0.986 measuring
F1 = 0.163. And **the padded axis cannot be the split dimension**, so `_choose_split_dim` excludes
it alongside the dimensions not shared by every variable.

Per-stratum counts come from `RecordGenerator.padded_stratum_counts`, a plain apportionment with a
floor of one per line, lognormally weighted so profile lengths vary — uniform weights gave every
profile the same length. One stratum is raised so a line can run the full length and the difference
comes back from the others' slack. The report measures the achieved layout as a run fraction, the
share of occupied cells whose neighbour towards 0 is also occupied: near 1 under padded, near the
density under scattered.

### Compression

`compression` (default `None`) and `complevel` go through `CompressionSettings`
(`data_sparsity/output/compression_settings.py`), which hands netCDF `{"zlib": True,
"complevel": n}` per data variable and parquet `compression="gzip"`. One setting drives both,
because the comparison the package exists to make is only meaningful if the two formats are
written on the same terms — and they were not: dask defaults to Snappy parquet while
`to_netcdf` writes uncompressed, so the array side was being measured in its worst
configuration against a compressed tabular side. At 5% density on a 20k-observation grid that
was netCDF 3.26 MB against parquet 0.44 MB; with DEFLATE on both, 0.42 against 0.29.

Only codecs both formats support are accepted. This netCDF build offers DEFLATE alone —
`zstd` and `blosc` are refused by the python binding despite libnetcdf 4.9.4 — so `snappy`,
`zstd`, `lz4` and `brotli` raise rather than silently applying to parquet only. Passing
`compression=None` writes `compression=None` to `to_parquet` explicitly; omitting it would
leave dask's Snappy default.

Compression is also the only mechanism that shrinks a scattered sparse grid. HDF5 does not
elide all-fill data that you write: a dense array of NaNs costs a full 8 bytes per vacant
site. The "unwritten chunks cost nothing" property needs chunks that are *entirely* empty,
which for randomly scattered occupancy requires fewer than about one point per chunk.

### Minimum density

`SparsityValidator.compute_min_density` returns `max(shape) / prod(shape)`: the sparsest grid in
which every coordinate on every axis is still used at least once. Each observation supplies one
coordinate per axis, so the LONGEST axis sets the floor, and `max(shape)` observations can reach
it (the LHS stage takes `n_s = max(shape)` for exactly this reason). `density=0.0` requests this
minimum. `docs/explainer.md`, "Minimum density, on any grid", derives it; the README documents it
under the `density` parameter.

This was `1 / nmin**(d-1)`, keyed to the SHORTEST axis. The two agree on cubic grids
(`n**d / n**(d-1) = n`), where the old formula was hand-derived and exact. Off cubic they diverge,
and the old one refused achievable densities: 5.6x too strict on a GLORYS12-shaped grid, and on any
grid with an axis of length 1 it returned 1.0, so such grids could only be generated fully gridded.

Validation is deliberately part-hard, part-soft: unusable inputs raise, rounding-level inconsistencies are silently corrected and the corrected configuration is printed (`_print_input_config` / `_print_updated_config`). Code that reads back `num_obs`, `density`, or `shape` must read the post-validation instance attributes, not the constructor arguments.

### Naming conventions baked into the data

Dimensions are `x0 … x{num_dims-1}`, variables are `var0 … var{num_vars-1}`. These strings are hardcoded across the generators, the parquet builder, and the overlap calculator; renaming requires touching all of them. A single-variable netCDF DataArray is named `record`, but its record dict key is still `var0`.

### Single variable is a special case of multi-variable

There is one generation path. `generate()` calls `_generate_multi_var_records` for every case, which calls `MultiVarRecordGenerator.generate`; with `num_vars=1` all dimensions vary and there are no constant dims. Do not reintroduce a separate single-variable placement routine. `num_vars == 1` only changes the output type: `xr.DataArray` + per-observation DataFrame, versus `xr.Dataset` + one-row-per-coordinate DataFrame with a column per variable.

### Variables on fewer dimensions

The grid is always `num_dims`-dimensional. A variable with fewer dimensions varies along `var_dims_indices[i]` and is pinned to a single coordinate on each dimension in `var_constant_dims[i]`. Records are generated in the reduced shape (constant dims set to size 1) and expanded to full coordinates before assignment; `NetCDFBuilder.build_dataset` drops and squeezes those dims back out on write.

### Overlap

`var0` is the reference variable, placed first. Overlap is **F1**, the definition in `docs/explainer_multivar.md`: `|proj(S0) & proj(Si)| / |proj(S0)|`, the share of *var0's* sites that also carry variable i, measured on the dimensions the two share. `OverlapCalculator.compute_overlap_report` is the single entry point: it returns `f1`, `f2` and the set sizes behind them.

Placement is per stratum: `MultiVarRecordGenerator.generate_multivar_stratified` takes cells from var0's footprint *within each hyperplane*, then fills the rest from cells held by neither variable, so the achieved overlap equals the target rather than picking up accidental coincidences. Where the target is unreachable — a reduced-dimension variable whose projected reference saturates — it warns and density takes precedence.

**How many cells per stratum** depends on the variable. One that varies along every dimension has its total decided once, `round(t_i * sum_j p_j)`, and apportioned across strata by largest remainder inside each stratum's bounds — the same treatment observation counts get. Rounding `t_i * p_j` in every stratum and summing is not the same number: each stratum rounds to a whole cell, and on a coarse grid one cell is a large share of the variable. A 3x3 grid asking for 7 of 8 shared cells got 8; one asking for 1 of 3 got 0.

A variable that **drops a dimension** keeps the per-stratum rounding. Its `p_j` counts distinct *projected* cells, which depends on where var0 landed, so a worker holding one chunk cannot know it for strata it does not own. Both paths apply the same rule per variable, which is what keeps serial and parallel identical. `RecordGenerator.stratum_counts` is what makes the apportioned case work under chunking: it rederives var0's per-stratum counts from the global LHS without placing anything.

`var0` is always the largest variable: `validate_reference_is_largest` rejects anything else, and `validate_density_refvar` forces `density[0]` to the maximum (mutating the caller's list, and collapsing a 2-element range to a single value).

`fixed_overlap` (per-variable bool) makes variables draw their overlapping sites from one shared permutation of the reference sites, so opted-in variables overlap each other as well as the reference; with `False` each variable draws independently.

### Determinism and the serial/parallel contract

Every RNG comes from `stream(seed, tag, *index)` in `data_sparsity/utils/streams.py`, never
from sequential consumption of one stream. The tuple goes to `numpy.random.default_rng`, which
runs it through `SeedSequence`, so distinct tuples give independent generators. Tags live in the
`Stream` class:

| tag | indexed by | used for |
|---|---|---|
| `COORDINATE` | dimension | one coordinate axis |
| `LHS` | — | the global Latin hypercube stage |
| `STRATUM` | stratum | per-stratum fill and values |
| `DENSITY` | — | drawing per-variable densities from a range |
| `VAR_DIMS` | variable | choosing which dimensions a variable varies along |
| `CONST_COORD` | variable, constant dim | a variable's coordinate on a constant dimension |
| `VAR` | variable, stratum | per-variable placement |
| `SHARED_OVERLAP` | — | the shared ordering behind `fixed_overlap` |

No tuple contains a chunk id, which is what makes parallel chunks reproduce serial output: serial
and every worker derive the same generator for the same purpose. Chunks differ only in *which*
strata they generate, so a chunk slices the global sorted coordinate axis by index rather than
advancing a stream. Any change to a tag value, or to the order in which draws are taken, changes
the data — `tests/test_scenarios_parallel.py` mirrors the serial scenarios specifically to catch
divergence.

Streams used to be derived by adding offsets to the seed (`seed + dim_idx * 1000` for coordinates
and so on). That collided: `seed + 0*1000` is `seed`, so the x0 axis and the density range were
one stream; `seed + 5000` is `seed + 5*1000`, so at six dimensions the x5 axis and var0's
dimension selection were one stream. Adding a purpose now means adding a tag where the existing
values are visible, rather than picking an offset and hoping it misses.

`seed` has a default of `None` in the signature but is required; passing nothing raises `TypeError`.

### Parallel path (experimental)

`max_obs` set below `num_obs` switches `NTASKS > 1`. The grid is split along its largest dimension, chunks run in a `ProcessPoolExecutor` (spawn context, at most 4 workers) via the module-level `generate_chunk`, which takes every parameter explicitly so nothing needs to pickle `GenerateData`. Each chunk writes its own netCDF file (`<base>_<zero-padded chunk_id>.nc`) and a temporary parquet chunk into the scratch directory named by `parquet_tmp`; worker logs are off unless `TABRAY_WORKER_LOG=debug` is set, and then land beside the netCDF output; `_consolidate_parquet_files` then merges the parquet chunks with dask into one 300MB-partitioned dataset and deletes the temporaries. In this mode `generate()` returns `(None, None)` — results exist only on disk. Chunking rounds observation counts per chunk, so it also rewrites `self.num_obs` and `self.density`.

`generate(merge_nc=True)` concatenates the chunk netCDF files into one and deletes them; the
default `False` leaves the per-chunk files, which is the only option for output too large to
merge. The merge writes one data variable at a time, under a synchronous dask scheduler. Both
matter: writing all variables in one `to_netcdf` call runs one dask store per variable
concurrently, and HDF5 then allocates their space in completion order, so identical data produces
a different file on every run (S7); and with the default thread pool a thread reading a chunk and
a thread writing the output can deadlock on xarray's netCDF4 lock, hanging the run (S8). Neither
is a memory trade -- the write still streams chunk by chunk, and one variable at a time needs less
memory than all of them at once.

`generate()` always calls `PathManager.setup_output_paths(overwrite=True)`, so existing `.nc`/`.parquet`/`_metadata` files in the target directories are deleted.

## Conventions (from `.github/copilot-instructions.md`)

- Python 3.12+; PEP 8, PEP 257, PEP 484 type hints; NumPy or Google-style docstrings; pylint >= 8 per file.
- One class per module, module name matching the class (`MyClass` -> `my_class.py`); subclass names extend the parent's name.
- Keep `environment.yml` and `pyproject.toml` in sync when touching dependencies, and justify new ones.
- Keep `README.md` current with new features and parameters.
- Non-expert Python users read this code: prefer clear over clever, and comment any non-obvious design decision.
- Never commit generated data (`.nc`, `.parquet`, `.csv`, …) — `.gitignore` already covers them.

## Known drift in the docs

`README.md` claims `from data_sparsity import GenerateData` and `342/426 tests passing`. Both are stale: `data_sparsity/__init__.py` is empty, so the import is `from data_sparsity.generate_data import GenerateData` (what the tests and notebook helpers use), and the suite is fully green at 524 tests. It also describes multi-variable data_vars as `record0, record1, …`; they are `var0, var1, …`.

# Claude Persona & Output Constraints

Mannered prose substitutes metaphor and flourish for direct statement. Instead of "a parameter worth varying," the mannered writer produces "a dial worth turning." Instead of "this point still matters," they write "this point earns its keep." The phrases exist to display the writer, not to convey the idea, and readers can tell. That is why mannered prose irritates: it makes the reader work harder so the writer can perform. It is also imprecise. Metaphors drag in connotations the writer did not choose and cannot control. The fix is to say what you mean. When a literal phrase is available, use it. This applies to both chat and every document you write.
