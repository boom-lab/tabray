# tabray

tabray is a Python Package for generating tabular and array data structures with different topologies. It currently writes to netCDF for arrays and parquet for tabular, but I'd like to add Zarr in the near future. Besides generating data, the package can describe any netCDF or parquet dataset: dimensions, density and overlap of each variable. Describing a real dataset gives the parameters to generate a synthetic one like it (see **Describing a Dataset**).

## Installation

### Using pip

```bash
# Install the package
pip install -e .

# Install with test dependencies
pip install -e ".[test]"

# Install with all development dependencies
pip install -e ".[dev]"
```

### Using conda

```bash
conda env create -f environment.yml
conda activate tabray
```

The conda environment includes pytest and other test dependencies by default.

## Usage

### Basic Usage

Generate synthetic observation data and save to both NetCDF and Parquet formats:

```python
from tabray.generate_data import GenerateData

# Create a data generator
gen = GenerateData(
    num_obs=1000,           # Number of observations
    num_dims=3,             # Number of dimensions
    ratio_dims=(1.0, 1.0, 1.0),  # Relative size of each dimension
    density=0.1,            # 10% of grid points contain observations
    seed=42                 # Random seed for reproducibility
)

# Generate data and save to files
dataarray, dataframe = gen.generate(
    netcdf_filepath="output_data.nc",
    parquet_filepath="output_data.parquet"
)

print(f"Generated {len(dataframe)} observations")
print(f"Array shape: {dataarray.shape}")
```

**`generate()` deletes files.** Before writing, it deletes every `.nc`, `.parquet` and `_metadata` file in the netCDF and parquet output directories. Write to directories that hold nothing else you want to keep.

### Default Output Paths

```python
from tabray.generate_data import GenerateData

gen = GenerateData(
    num_obs=500,
    num_dims=2,
    ratio_dims=(2.0, 1.0),  # First dimension twice as large as second
    density=0.2,
    seed=123
)

# No paths: writes ./test_nc/test.nc and ./test_parquet/data_*.parquet
# (after deleting the .nc, .parquet and _metadata files in both directories)
dataarray, dataframe = gen.generate()

# Access the data
print(dataarray)
print(dataframe.head())
```

### Parallel Generation for Large Datasets

Setting `max_obs` below `num_obs` splits the grid into chunks and generates each chunk in its own process. A parallel run writes the same data as a serial run with the same parameters and seed.

```python
from tabray.generate_data import GenerateData

# Required in a script: each chunk runs in a new process, which imports this file again
if __name__ == "__main__":
    # Single variable: 100 observations in 4 chunks of 25
    gen = GenerateData(
        num_obs=100,
        num_dims=1,
        ratio_dims=1,
        density=1.0,
        seed=34,
        max_obs=25,           # at most 25 observations per chunk
    )
    gen.generate(
        netcdf_filepath="single/data.nc",
        parquet_filepath="single/data.parquet",
    )

    # Two variables on an 8x8x8 grid, 4 chunks
    gen_multi = GenerateData(
        num_obs=256,          # observations of var0
        num_dims=3,
        ratio_dims=1,
        density=0.5,
        seed=42,
        num_vars=2,
        var_dims=2,           # var1 varies along two of the three dimensions
        overlap=0.5,          # half of var0's sites, projected on var1's dimensions, also carry var1
        max_obs=64,
    )
    gen_multi.generate(
        netcdf_filepath="multi/data.nc",
        parquet_filepath="multi/data.parquet",
    )
```

**How it works:**
- The run is split into `ceil(num_obs / max_obs)` chunks along the longest dimension that every variable varies along.
- Each chunk runs in its own process (`concurrent.futures.ProcessPoolExecutor`). `max_workers` caps the number of processes running at once (default: the CPU count).
- Every random draw depends on the seed and on the position in the grid, not on the chunk, so the chunks together reproduce the serial output
- `generate()` returns the dataset and dataframe dask-backed: they read from the written files, and nothing is loaded until it is used
- For NetCDF: it generates one file per chunk (e.g., `data_0.nc`, `data_1.nc`, etc.), which are merged if `merge_nc=True` (note that this needs more memory).
- For Parquet: each chunk is written to a scratch directory (`generate(parquet_tmp=...)`), then Dask merges the chunks into one dataset with ~300 MB partitions and deletes the scratch files.
- `max_obs` refers to the observations of `var0`, the variable with the most observations
- Each process holds one chunk, so peak memory grows with chunk size times the number of processes running at once
- Lower `max_obs` (smaller chunks) or `max_workers` (fewer processes) to reduce memory usage

### Multi-Variable Datasets

```python
from tabray.generate_data import GenerateData

# Three variables on a 3x3 grid
gen = GenerateData(
    num_obs=8,
    num_dims=2,
    ratio_dims=1,
    density=8/9,                 # var0 holds 8 of the 9 sites
    seed=12345,
    num_vars=3,
    var_dims=[2, 2, 1],          # var2 varies along one dimension, chosen at random
    overlap=[1, 1],              # every var0 site also carries var1 and var2
    fixed_overlap=[True, True],  # var1 and var2 draw from one shared ordering of var0's sites
)

dataset, dataframe = gen.generate(
    netcdf_filepath="multi_var_data.nc",
    parquet_filepath="multi_var_data.parquet"
)

print(list(dataset.data_vars))   # ['var0', 'var1', 'var2']
print(list(dataframe.columns))   # ['x0', 'x1', 'var0', 'var1', 'var2']
```

**How it works:**
- The result is an `xarray.Dataset`, and the DataFrame has one row per site holding at least one variable (NaN where a variable is absent)
- A variable on fewer dimensions repeats its value on every row matching its own dimensions
- var0 is the reference: it varies along every dimension and must hold the most observations
- A density is a share of the variable's own grid (the product of the dimensions it varies along)

**Density options:**
- Scalar (e.g., `0.1`): all variables have the same density
- `[max, min]` (e.g., `[0.3, 0.1]`): var0 gets max, one other variable gets min, the rest are drawn from the range. With two variables: one density per variable
- One per variable (e.g., `[0.3, 0.2, 0.1]`): `0.0` for var1 onward gives an empty variable

**Variable dimensions options:**
- Int (e.g., `2`): each non-reference variable uses 2 randomly selected dimensions
- List of ints (e.g., `[3, 3, 2]`): number of dimensions per variable, randomly selected
- List of lists (e.g., `[[0,1,2], [1,2], [0,2]]`): explicit dimensions per variable
- At least one dimension must be shared by every variable

**Overlap control:**

```
overlap_i = |proj(S_0) & proj(S_i)| / |proj(S_0)|
```

- Share of var0's sites that also carry variable *i*, projected onto the dimensions the two share
- `'random'` (default): no constraint
- Float in `[0, 1]`: one target for every non-reference variable; a list sets one per variable
- Unreachable targets print a warning, and density takes precedence
- `fixed_overlap=True`: variables draw from one shared ordering of var0's sites, so they also overlap each other

```python
# Example: 3 variables with different dimensions and controlled overlap
gen = GenerateData(
    num_obs=3,
    num_dims=2,
    ratio_dims=1,
    density=3/9,
    seed=12345,
    num_vars=3,
    var_dims=[2, 2, 1],            # var2 varies along one dimension
    overlap=[1/3, 1/3],            # one of var0's 3 sites carries each variable, on shared dimensions
    fixed_overlap=[True, False],   # only var1 draws from the shared ordering
)

dataset, df = gen.generate()
```

### Smooth (Correlated) Values

By default values are white noise. `data_structure="correlated"` makes them change smoothly from site to site:

```python
gen = GenerateData(
    num_obs=20000,
    num_dims=3,
    ratio_dims=(1, 2, 2),
    density=0.9,
    seed=42,
    data_structure="correlated",  # occupied sites are unchanged
    correlation_length=[None, 5, 30],  # optional: constant along x0, 5 sites along x1, 30 along x2
    noise=0.01,                   # optional: share of white noise
)
```

### Describing a Dataset

```python
from tabray.output import DatasetDescription

DatasetDescription.describe_dataset("data.nc")
DatasetDescription.describe_dataset("parquet_dir", coords=["x0", "x1", "x2"])
DatasetDescription.describe_dataset("data.nc", reference="temp")  # overlap against temp
```

- One row per variable: dimensions, sites, occupied sites, density, overlap with the reference, unused coordinates, dtype
- The reference is `reference` if given, else the variable with the most occupied sites
- Parquet has no axes: it requires `coords` to be specified, and a variable's dimensions are inferred from its rows (with some assumptions in ambiguous cases)
- `notebooks/analyse_dataset.ipynb` compares the descriptions of the netCDF and parquet files of one dataset
- When generating a synthetic dataset, `generate()` prints this description of what it generate and keeps it as `gen.description`

### How the Data Are Generated

- `docs/dataflow.puml` traces a run from parameters to files
- `docs/dataflow_example_*.puml` are worked examples from real runs; `notebooks/dataflow_examples.ipynb` runs the same calls
- Render with [PlantUML](https://plantuml.com): `plantuml -tpdf docs/dataflow*.puml`

### Parameters

- **num_obs** (int): number of observations of var0
- **num_dims** (int): number of dimensions
- **ratio_dims** (tuple): relative size of each dimension
- **density** (float, list, or tuple): share of grid points holding an observation, in (0, 1] (see **Density options**)
  - Minimum: `max(shape) / prod(shape)`, so every coordinate is used at least once
- **sparsity** (float, list, or tuple): `1 - density`; give `density` or `sparsity`, not both
- **seed** (int): random seed, required
- **max_obs** (int): maximum observations per chunk; below `num_obs` it turns on parallel generation
- **max_workers** (int): maximum processes running at once (default: the CPU count)
- **num_vars** (int): number of variables (default: 1)
- **var_dims** (int or list): dimensions of each variable (see **Variable dimensions options**; default: all)
- **overlap** (float, list, or `'random'`): overlap target per non-reference variable (see **Overlap control**)
- **fixed_overlap** (bool or list): share overlap draws across variables (default: `False`)
- **dtype** (str or list): `float64` (default), `float32`, `int8`, `int16`, `int32`, or `S<k>` for strings of k characters
- **value_range** (tuple or list): (min, max) of the values (default: `(0, 1)` for floats, `(0, 100)` for integers); `min == max` gives a constant
- **pack** (str or list): integer type to pack floats into on disk with `scale_factor`/`add_offset` (default: `None`)
- **fill_value** (float or list): marks a vacant site (default: NaN for floats, a reserved code for integers)
- **data_structure** (str or list): `"uncorrelated"` (default) or `"correlated"`
- **correlation_length** (float or list): correlation length in sites, per dimension (default: 10); `0` for none, `None` for constant
- **noise** (float or list): share of white noise in a correlated field, 0 (default) to 1
- (experimental) **layout** (str): `"scattered"` (default) or `"padded"`, where each line fills the first k cells of one axis, as profiles do
- (experimental) **padded_dim** (int): the padded axis (default: the last); `overlap` cannot be set with `layout="padded"`

Per-variable parameters take one value for all variables, or a list with one per variable.

## Notebooks

Run them with `notebooks/` as the working directory.

- `tutorial1.ipynb`: one variable on a 3x3 grid: gridded, irregular, and sparse on a grid
- `tutorial2.ipynb`: the same cases on a 100x100 grid
- `tutorial3.ipynb`: two and three variables, on two dimensions or fewer, with overlap 1 and below 1
- `tutorial4.ipynb`: serial and parallel generation of the same dataset, and how to read the chunked output
- `dataflow_examples.ipynb`: the runs behind `docs/dataflow_example_*.puml`
- `analyse_dataset.ipynb`: describes the netCDF and parquet files of a generated dataset and compares them
  
## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=tabray --cov-report=html

# Run specific test file
pytest tests/validators/test_parameter_validator.py

# Run specific test class
pytest tests/generators/test_coordinate_generator.py::TestGenerateAllCoords

# Run tests matching a pattern
pytest -k "test_2d"
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Apache 2.0

## Authors

Enrico Milanese (enrico.milanese@whoi.edu)
