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

Validation is deliberately part-hard, part-soft: unusable inputs raise, rounding-level inconsistencies are silently corrected and the corrected configuration is printed (`_print_input_config` / `_print_updated_config`). Code that reads back `num_obs`, `density`, or `shape` must read the post-validation instance attributes, not the constructor arguments.

### Naming conventions baked into the data

Dimensions are `x0 … x{num_dims-1}`, variables are `var0 … var{num_vars-1}`. These strings are hardcoded across the generators, the parquet builder, and the overlap calculator; renaming requires touching all of them. A single-variable netCDF DataArray is named `record`, but its record dict key is still `var0`.

### Single variable is a special case of multi-variable

There is one generation path. `_generate_record` and `generate()` both call `MultiVarRecordGenerator.generate` with `num_vars=1`, all dimensions varying, no constant dims. Do not reintroduce a separate single-variable placement routine. `num_vars == 1` only changes the output type: `xr.DataArray` + per-observation DataFrame, versus `xr.Dataset` + one-row-per-coordinate DataFrame with a column per variable.

### Variables on fewer dimensions

The grid is always `num_dims`-dimensional. A variable with fewer dimensions varies along `var_dims_indices[i]` and is pinned to a single coordinate on each dimension in `var_constant_dims[i]`. Records are generated in the reduced shape (constant dims set to size 1) and expanded to full coordinates before assignment; `NetCDFBuilder.build_dataset` drops and squeezes those dims back out on write.

### Overlap

`var0` is the reference variable in `MultiVarRecordGenerator`: it is placed first, and each other variable takes `round(overlap_target[i] * var_num_obs[i])` of its sites from `var0`'s occupied sites (filtered to those compatible with its own constant coordinates), then fills the rest from unused sites. So overlap is the fraction of the *non-reference* variable's sites that coincide with reference sites — `OverlapCalculator.compute_actual_overlaps_against_reference` measures exactly that. `compute_actual_overlap` is an older, different metric that picks its reference by observation count instead; note that `MultiVarSparsityConfig` does not sort densities, so `var0` is not necessarily the densest variable.

`fixed_overlap` (per-variable bool) makes variables draw their overlapping sites from one shared permutation of the reference sites, so opted-in variables overlap each other as well as the reference; with `False` each variable draws independently.

### Determinism and the serial/parallel contract

Every RNG is derived from `seed` by a fixed offset, never by sequential consumption of one stream:

- coordinates: `seed + dim_idx * 1000` (`ChunkUtils.generate_rngs`)
- overlap site selection: `seed + 9999` (+ `chunk_id * 10000`)
- per-variable values and non-overlapping placement: `seed + var_idx + 2000` (+ `chunk_id * 100`)

The point of this scheme is that parallel chunks reproduce serial output: shared dimensions get identical coordinates across chunks, and the split dimension's RNG is fast-forwarded by `chunk_id * obs_in_chunk` draws. Any change to seeding, or to the order in which draws are taken, breaks that equivalence — `tests/test_scenarios_parallel.py` mirrors the serial scenarios specifically to catch it.

`seed` has a default of `None` in the signature but is required; passing nothing raises `TypeError`.

### Parallel path (experimental)

`max_obs` set below `num_obs` switches `NTASKS > 1`. The grid is split along its largest dimension, chunks run in a `ProcessPoolExecutor` (spawn context, at most 4 workers) via the module-level `generate_chunk`, which takes every parameter explicitly so nothing needs to pickle `GenerateData`. Each chunk writes its own netCDF file (`<base>_<zero-padded chunk_id>.nc`), a temporary parquet chunk, and a `worker_<chunk_id>.log`; `_consolidate_parquet_files` then merges the parquet chunks with dask into one 300MB-partitioned dataset and deletes the temporaries. In this mode `generate()` returns `(None, None)` — results exist only on disk. Chunking rounds observation counts per chunk, so it also rewrites `self.num_obs` and `self.density`.

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
