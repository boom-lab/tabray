# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Synthetic-data generator that compares **array** storage (netCDF/xarray) against **tabular** storage (parquet/pandas) across grid occupancies, from gridded to maximally sparse. `docs/explainer.md` and `docs/explainer_multivar.md` define the terms the code uses (site, density/sparsity, gridded vs irregular, overlap); read them before changing generation semantics.

Distribution and import package, repository and local conda env: `tabray`.

## Commands

```bash
conda env create -f environment.yml && conda activate tabray
pip install -e ".[dev]"          # alternative; dev = pytest, pytest-cov, pylint, black

pytest                            # full suite, ~85s
pytest tests/generators/test_overlap_calculator.py
pytest -k "overlap"
bash tests/run_all_tests.sh       # suite + coverage

pylint tabray/<module>.py  # every file must score >= 8
black tabray tests                # config in pyproject.toml, target py312
```

- Run Python through the `tabray` env (`$(conda info --base)/envs/tabray/bin/python`), an editable install of the working tree. A bare `python` can resolve to another interpreter holding a stale installed copy of the package, with no sign in the output.
- Notebooks run with `notebooks/` as the working directory.
- `tests/diagnostic_coordinate_identity*.py` are standalone scripts, not collected by pytest.

## Architecture

`GenerateData` (`tabray/generate_data.py`, the only user-facing class) validates, configures, then delegates; it holds no generation logic:

```
GenerateData.__init__  ->  validators/   ParameterValidator, DimensionValidator, SparsityValidator
                       ->  config/       MultiVarSparsityConfig, MultiVarDimensionsConfig, MultiVarOverlapConfig
GenerateData.generate  ->  generators/   CoordinateGenerator, MultiVarRecordGenerator (-> RecordGenerator,
                                         ObservationGenerator, OverlapCalculator)
                       ->  output/       NetCDFBuilder, ParquetBuilder, PathManager, VariableEncoding,
                                         DatasetDescription (measures the written netCDF)
                       ->  workers/      generate_chunk  (parallel path only)
```

### Validation

- Unusable inputs raise; rounding-level inconsistencies are corrected and printed.
- Read `num_obs`, `density`, `shape` from post-validation instance attributes, not constructor arguments.
- Minimum density = `max(shape) / prod(shape)` (`SparsityValidator.compute_min_density`): the longest axis sets it. `density=0.0` raises. Derivation: `docs/explainer.md`, "Minimum density, on any grid".
- `seed` defaults to `None` in the signature but is required.

### Naming baked into the data

- Dimensions `x0 … x{num_dims-1}`, variables `var0 … var{num_vars-1}`, hardcoded across generators, parquet builder and overlap calculator.
- Single-variable netCDF DataArray is named `record`; its record dict key is still `var0`.

### One generation path

- `generate()` → `_generate_multi_var_records` → `MultiVarRecordGenerator.generate` for every case. Do not add a separate single-variable routine.
- `num_vars == 1` changes only the output type: `xr.DataArray` + per-observation DataFrame, versus `xr.Dataset` + one row per coordinate.
- The grid is always `num_dims`-dimensional. A variable on fewer dims varies along `var_dims_indices[i]` and is pinned to one coordinate on each of `var_constant_dims[i]`; `NetCDFBuilder.build_dataset` squeezes those dims out on write.
- A density is a share of the variable's OWN grid (`prod` of its varying dims): `compute_var_num_obs` scales by `grid_fractions`, and both report paths divide by the own grid.

### Overlap

- `var0` is the reference, placed first, and must be the largest variable (`validate_density_refvar` raises otherwise).
- Overlap (`docs/explainer_multivar.md`): `overlap = |proj(S0) & proj(Si)| / |proj(S0)|`, the targeted quantity; `overlap_reverse` divides by `|proj(Si)|`. Single entry point: `OverlapCalculator.compute_overlap_report`.
- Placement is per stratum (`generate_multivar_stratified`): overlap cells from var0's footprint in each hyperplane, the rest from cells held by neither. Unreachable target → warning, density wins.
- Overlap count per stratum:
  - variable on all dims: total `round(t_i * sum_j p_j)`, apportioned across strata by largest remainder (`RecordGenerator.stratum_counts` rederives var0's counts under chunking);
  - variable that drops a dim: `round(t_i * p_j)` per stratum, because `p_j` depends on where var0 landed and a worker cannot see other chunks.
- `fixed_overlap=True`: opted-in variables draw from one shared permutation of reference sites, so they also overlap each other.

### Layout

`layout` (`"scattered"` default, or `"padded"`) and `padded_dim` set where occupied cells sit; `density` sets how many. Reasons and measurements: `docs/layout_plan.md`.

- One layout per dataset. var0: `RecordGenerator.generate_padded_indices`; other variables with the padded axis: `MultiVarRecordGenerator._padded_cells` per stratum; a variable without the padded axis stays scattered.
- A line (one combination of the dims other than split and padded) holds positions `0..k-1` of the padded axis.
- Coverage comes from the construction, not the LHS stage, which `padded` skips: every line holds an observation; one line runs the full length.
- Minimum density = `(prod(shape)/n_padded + n_padded - 1) / prod(shape)` (`compute_min_density(..., padded_dim)`).
- `overlap` raises under `padded`: each line of every variable fills its first k cells of one axis, so the overlap follows from the densities.
- The padded axis cannot be `dim_split` (`_choose_split_dim` excludes it).
- Stratum counts: `padded_stratum_counts`, equal weights capped per stratum (`ChunkUtils.apportion(..., capacity)`), one stratum raised to carry the full-length line; k per line also from equal weights.
- `_print_updated_config` prints which minimum-density bound applied (var0's coverage bound).

### Per-variable encoding

`dtype`, `pack`, `fill_value`, `value_range` → `VariableEncoding` (`output/variable_encoding.py`). Reasons: `docs/variable_encoding.md`.

- Default `float64` + NaN emits no encoding entry; default output must stay byte-identical.
- `dtype` = what a variable holds; `pack` = how a float is compacted on disk. Packing an integer dtype raises.
- Integers = `lo + floor(u * (hi - lo + 1))` from the uniform draw, never `rng.integers`: a dtype change must not move occupied sites.
- Packed float: fill code reserved; `scale_factor`/`add_offset` are `float32`, except for `int32` (`float64`).
- Integer `fill_value` must sit outside `value_range`; the parquet column is nullable (`Int16`).
- `to_stored` runs once (`GenerateData._to_stored` + worker counterpart) before both writers; scale from `VALUE_RANGE`, never from the data.

### Determinism and the serial/parallel contract

Every RNG comes from `stream(seed, tag, *index)` (`utils/streams.py`); tags live in the `Stream` class.

| tag | indexed by | used for |
|---|---|---|
| `COORDINATE` | dimension | one coordinate axis |
| `LHS` | — | the global Latin hypercube stage |
| `STRATUM` | stratum | per-stratum fill and values |
| `DENSITY` | — | per-variable densities from a range |
| `VAR_DIMS` | variable | which dims a variable varies along |
| `CONST_COORD` | variable, constant dim | a variable's coordinate on a constant dim |
| `VAR` | variable, stratum | per-variable placement |
| `SHARED_OVERLAP` | — | the shared ordering behind `fixed_overlap` |

- No tuple contains a chunk id: serial and workers derive the same generator for the same purpose.
- Chunks slice the global sorted coordinate axis by index; they never advance a stream.
- New purpose → new tag. Never derive streams by seed offsets (`streams.py` docstring says why).
- Changing a tag value or the order of draws changes the data. `test_parallel_reproduces_serial_exactly` (`tests/utils/test_parallel_comparison.py`) checks exact equality.

### Parallel path

- `max_obs < num_obs` → `NTASKS > 1`. Split along `dim_split` (`_choose_split_dim`: longest dim every variable varies along); `NTASKS <= len(dim_split)`.
- `ProcessPoolExecutor`, spawn context, `min(NTASKS, max_workers or os.cpu_count())` workers, module-level `generate_chunk` with explicit parameters.
- Scripts need an `if __name__ == "__main__":` guard (spawn re-imports them). README does not mention this yet.
- Each chunk writes `<base>_<chunk_id>.nc` and a parquet chunk into `parquet_tmp`; `_consolidate_parquet_files` merges them (300MB partitions) and deletes the temporaries.
- `generate()` returns `(None, None)`; results exist only on disk. Single variable: `self.num_obs` and `self.density` are rewritten from per-chunk counts.
- Chunk netCDF files keep every dim (`squeeze_constant_dims=False`). Apply `NetCDFBuilder.squeeze_constant_dims` before comparing with serial output.
- `merge_nc=True` concatenates chunks one variable at a time under a synchronous dask scheduler. Do not change either: see S7 (non-reproducible bytes) and S8 (deadlock) in `.claude/serial_parallel_equivalence_diagnostics.md`.
- Worker logs: off unless `TABRAY_WORKER_LOG=debug`, then in `generate(log_dir=...)` (default `./logs`).
- `generate()` calls `PathManager.setup_output_paths(overwrite=True)`: existing `.nc`/`.parquet`/`_metadata` files in the target directories are deleted.

## Conventions (from `.github/copilot-instructions.md`)

- Python 3.12+; PEP 8, PEP 257, PEP 484 type hints; NumPy or Google-style docstrings; pylint >= 8 per file.
- One class per module, module name matching the class (`MyClass` -> `my_class.py`); subclass names extend the parent's name.
- Keep `environment.yml` and `pyproject.toml` in sync when touching dependencies, and justify new ones.
- Keep `README.md` current with new features and parameters.
- Non-expert Python users read this code: prefer clear over clever, and comment any non-obvious design decision.
- Never commit generated data (`.nc`, `.parquet`, `.csv`, …).
- A module-level import of the package is `import tabray as tr`, used as `tr.<name>` (like `numpy as np`).
- `docs/`: never track `.pdf` renders (`*.pdf` is in `.gitignore`).
- Review diagrams (`docs/review_*.puml`): commit them to the PR branch so the PR history keeps them, and delete them before the PR merges.
- Dataflow examples (`docs/dataflow_example_*.puml`) stay in the repository. They show the code's mechanics and architecture through worked examples from real runs: detailed enough to follow each step, no more. Each generation step shows a grid of the sites filled at that point. Take every number from a run of the current code (scratch tracer: call `RecordGenerator._global_stage`, `generate_stratified_indices(strata=[j])`, `generate_multivar_stratified`), and rerun when generation semantics change: the examples do not fail when they go stale. `notebooks/dataflow_examples.ipynb` runs the same calls; keep the two in step.
- Writing a `.puml` activity diagram: a line ending in `]` closes the activity early, `[[x]]` is a link, `__x__` underlines, and a line starting with `=` becomes a heading. Escape with `~` (`~__main~__`).

## Open items

`.claude/HANDOFF.md` lists unfinished work: lint exceptions to agree, diagnostic S6. Delete entries as they are done.

# Claude Persona & Output Constraints

Mannered prose substitutes metaphor and flourish for direct statement. Instead of "a parameter worth varying," the mannered writer produces "a dial worth turning." Instead of "this point still matters," they write "this point earns its keep." The phrases exist to display the writer, not to convey the idea, and readers can tell. That is why mannered prose irritates: it makes the reader work harder so the writer can perform. It is also imprecise. Metaphors drag in connotations the writer did not choose and cannot control. The fix is to say what you mean. When a literal phrase is available, use it. This applies to both chat and every document you write.
