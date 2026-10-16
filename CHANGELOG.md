# Changelog

## 0.2.0 (2026-10-16)

### Breaking changes

- The import package is renamed from `data_sparsity` to `tabray`: `from tabray.generate_data import GenerateData`.
- The same parameters and seed produce different data than 0.1.0. Every random draw now comes from one registry of streams keyed by the seed, a purpose tag and a grid position, and placement runs per stratum (hyperplane of the split dimension).
- `overlap` is the share of var0's sites, projected on the dimensions two variables share, that also carry the other variable. An unreachable target prints a warning, and the density takes precedence.
- var0 is the overlap reference and must hold the most observations; the constructor raises otherwise.
- A two-element density is a `[max, min]` range with the maximum first.
- The minimum density is set by the longest axis, `max(shape) / prod(shape)`.
- A density is a share of the variable's own grid (the product of the dimensions it varies along).
- Parquet layout of a variable on fewer dimensions: its value repeats on every row matching it on its own dimensions (like `xarray.Dataset.to_dataframe()`).

### Added

- Parallel generation reproduces the serial output exactly. `max_workers` caps the number of processes; `generate(merge_nc=True)` merges the chunk netCDF files into one; `generate(log_dir=...)` and `TABRAY_WORKER_LOG` control worker logs.
- Parallel `generate()` returns a dask-backed dataset (DataArray for one variable) and dask DataFrame, read from the written files, instead of `(None, None)`.
- `var_dims` accepts explicit dimension indices (`[[0, 1, 2], [1, 2]]`).
- Density `0.0` for var1 onward in a per-variable list: an empty variable.
- Per-variable encoding: `dtype` (`float64`, `float32`, `int8`, `int16`, `int32`, `S<k>` strings), `pack` (integer packing with `scale_factor`/`add_offset`), `fill_value`, `value_range` (`min == max` gives a constant).
- Correlated values: `data_structure="correlated"`, `correlation_length`, `noise`. Values change from site to site as a smooth field; occupied sites are the same as in an uncorrelated run.
- Experimental `layout="padded"` and `padded_dim`: each line fills the first k cells of one axis, as profile data do.
- `DatasetDescription.describe_dataset`: dimensions, sites, density, overlap and dtype of each variable of any netCDF or parquet dataset. `generate()` prints this description of what it wrote and keeps it as `gen.description`. For parquet, a variable's dimensions are inferred from its rows, and every assumption is printed and warned.
- Notebooks: tutorial for serial and parallel generation (`tutorial4`), dataset analysis (`analyse_dataset`), and `dataflow_examples`.
- Docs: `docs/dataflow.puml` and worked examples `docs/dataflow_example_*.puml`.
- Dependency: `scipy>=1.10.0`.

### Fixed

- Some coordinates could stay unused.

### Removed

- `docs/explainer.md` and `docs/explainer_multivar.md`, moved to branch `docs/explainers` for a rewrite.

## 0.1.0 (2026-08-07)

First release on PyPI, with import package `data_sparsity`.
