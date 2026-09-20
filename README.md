# Data Sparsity Analysis

The goal of this repository is to study the performance of _read_ operations of different data when stored in tabular or array structure. I'll start using netCDF for arrays and parquet for tabular, as they are the most commonly discussed. The number of permutations on how to do this are possibly infinite, so I'm picking a few guided by the following principles:

1. I want to illustrate limit cases: extremely regular ('gridded') data and extremely sparse data;
2. I want to capture some kind of transition between the two limit cases;
3. The metric of comparison is the user experience in the form of time to read or manipulate a given amount of observations;
4. I want to explore both smaller- and larger-than-memory cases.

Notes:

- Points 3 and 4 are related: I'll use read time as the metric for small enough data, and manipulation time (e.g. computing the mean, plotting, or so) for larger-than-memory data

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
conda activate data_sparsity
```

The conda environment includes pytest and other test dependencies by default.

## Usage

### Basic Usage

Generate synthetic observation data and save to both NetCDF and Parquet formats:

```python
from data_sparsity.generate_data import GenerateData

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

### Generate Data Without Saving

```python
from data_sparsity.generate_data import GenerateData

gen = GenerateData(
    num_obs=500,
    num_dims=2,
    ratio_dims=(2.0, 1.0),  # First dimension twice as large as second
    density=0.2,
    seed=123
)

# Generate data without saving to disk
dataarray, dataframe = gen.generate()

# Access the data
print(dataarray)
print(dataframe.head())
```

### Parallel Generation for Large Datasets

Setting `max_obs` below `num_obs` splits the grid into chunks and generates each chunk in its own process. A parallel run writes the same data as a serial run with the same parameters and seed.

```python
from data_sparsity.generate_data import GenerateData

# Generate a large single-variable dataset with parallel processing
gen = GenerateData(
    num_obs=50_000_000,     # 50 million observations
    num_dims=3,
    ratio_dims=(2.0, 1.5, 1.0),
    density=0.05,
    seed=42,
    max_obs=10_000_000      # Split into chunks of 10M observations each
)

# Data is automatically split and processed in parallel
gen.generate(
    netcdf_filepath="large_data.nc",
    parquet_filepath="large_data.parquet"
)

# Generate a large multi-variable dataset with parallel processing
gen_multi = GenerateData(
    num_obs=30_000_000,     # 30 million observations (for highest density variable)
    num_dims=4,
    ratio_dims=(2.0, 1.5, 1.0, 1.0),
    density=[0.10, 0.05],   # [max, min]: var0 takes the max, the rest are drawn from the range
    seed=42,
    max_obs=10_000_000,
    num_vars=3,
    var_dims=[[0,1,2,3], [1,2,3], [0,2,3]],  # var0 always uses every dimension
    overlap=0.5             # half of var0's sites also carry each other variable
)

gen_multi.generate(
    netcdf_filepath="large_multivar_data.nc",
    parquet_filepath="large_multivar_data.parquet"
)
```

**How it works:**
- The run is split into `ceil(num_obs / max_obs)` chunks along the longest dimension that every variable varies along. The chunk count cannot exceed that dimension's length; a `max_obs` that asks for more chunks raises.
- Each chunk runs in its own process (`concurrent.futures.ProcessPoolExecutor`). `max_workers` caps the number of processes running at once (default: the CPU count).
- Every random draw depends on the seed and on the position in the grid, never on the chunk, so the chunks together reproduce the serial output: same coordinates, sites, values and overlap. Overlap targets apply to the whole grid, not to each chunk.
- `generate()` returns `(None, None)`; the results exist only on disk.
- For NetCDF: one file per chunk (e.g., `large_data_0.nc`, `large_data_1.nc`, etc.), which `xr.open_mfdataset` combines. Chunk files keep every dimension so that they concatenate, so a variable defined on fewer dimensions appears on the full grid, NaN outside its constant coordinate.
- `generate(merge_nc=True)` merges the chunk files into one file, with those constant dimensions removed as in a serial run, and deletes the chunk files. The merge needs more memory than the data size; leave it off for output that does not fit in memory.
- For Parquet: each chunk is written to a scratch directory (`generate(parquet_tmp=...)`), then Dask merges the chunks into one dataset with ~300 MB partitions and deletes the scratch files.
- Worker logs: off by default. Set `TABRAY_WORKER_LOG=debug` (or `info`) to write one
  `worker_<id>.log` per chunk into `generate(log_dir=...)`, default `./logs`

**Memory:**
- `max_obs` refers to the observations of `var0`, the variable with the highest density
- Each process holds one chunk, so peak memory grows with chunk size times the number of processes running at once
- Lower `max_obs` (smaller chunks) or `max_workers` (fewer processes) to reduce it

### Multi-Variable Datasets

Generate datasets with multiple observation variables measured at potentially different points and dimensions:

```python
from data_sparsity.generate_data import GenerateData

# Generate a dataset with 3 variables
gen = GenerateData(
    num_obs=1000,
    num_dims=3,
    ratio_dims=(1.0, 1.0, 1.0),
    density=[0.3, 0.1],   # [max, min]: var0 takes 0.3, var1 and var2 are drawn from [0.1, 0.3]
    seed=42,
    num_vars=3,           # Number of variables
    var_dims=3,           # Each variable uses all 3 dimensions
    overlap=0.5,          # half of var0's sites also carry each other variable
    fixed_overlap=False   # Share overlap draws only when enabled
)

dataarray, dataframe = gen.generate(
    netcdf_filepath="multi_var_data.nc",
    parquet_filepath="multi_var_data.parquet"
)

# The result is an xarray.Dataset (not DataArray) with multiple data variables
print(dataarray.data_vars)  # ['var0', 'var1', 'var2']

# The DataFrame has one row per site holding at least one variable and one
# column per variable, NaN where that variable is absent
print(dataframe.columns)  # ['x0', 'x1', 'x2', 'var0', 'var1', 'var2']
```

**Density Options for Multiple Variables:**
- Scalar (e.g., `0.1`): All variables have the same density
- 2-element list/tuple `[max, min]` (e.g., `[0.3, 0.1]`): var0 gets max, one other variable gets min, the rest are drawn uniformly from `[min, max]`
- num_vars-element list/tuple (e.g., `[0.3, 0.2, 0.1]`): Each variable gets its specified density

var0 is the overlap reference and must have the largest density, so the largest value comes first; otherwise the constructor raises.

**Variable Dimensions Options:**
- Int (e.g., `2`): Each non-reference variable uses 2 randomly selected dimensions
- List of ints (e.g., `[3, 3, 2]`): Each variable uses the specified number of dimensions (randomly selected)
- List of lists/tuples (e.g., `[[0,1,2], [1,2], [0,2]]`): Each variable uses explicitly specified dimensions

var0 always uses every dimension: a smaller value for it is replaced by `num_dims`, and a note is printed. At least one dimension must be shared by every variable, otherwise the constructor raises.

**Overlap Control:**

```
overlap_i = |proj(S_0) & proj(S_i)| / |proj(S_0)|
```

Share of var0's sites that also carry variable *i*. `S_i`: sites of variable *i*; `proj`:
projection onto the dimensions the two share.

- `'random'` (default): No constraint, measurements placed independently
- Float `0.0` to `1.0`: Target overlap for every non-reference variable
  - `0.0`: No site of var0 carries variable *i*
  - `0.5`: Half of var0's sites carry variable *i*
  - `1.0`: Every site of var0 carries variable *i*
  - Maximum: `|proj(S_i)| / |proj(S_0)|`
  - Minimum (variable on every dimension): `max(0, n_0 + n_i - N) / n_0` for `N` grid points;
    a lower target raises
  - Unreachable targets: the generator warns and density takes precedence
- List of length `num_vars - 1`: One target per non-reference variable, in
  `var1`, `var2`, ... order
- `fixed_overlap` (bool or list of bools, default `False`): Share overlap draws
  across variables when enabled

```python
# Example: 3 variables with different dimensions and controlled overlap
gen = GenerateData(
    num_obs=500,
    num_dims=4,
    ratio_dims=(1.0, 1.0, 1.0, 1.0),
    density=0.2,
    seed=42,
    num_vars=3,
    var_dims=[[0,1,2,3], [1,2,3], [0,3]],  # Different dimensions per variable
    overlap=0.7  # 70% of var0's sites carry each variable, on shared dimensions
)

dataset, df = gen.generate()
```

### Parameters

- **density** and **sparsity** are both valid, independent ways to specify grid occupancy (`sparsity = 1 - density`). Provide exactly one; if both are passed, `density` takes precedence and a warning is raised.

- **num_obs** (int): Number of observations to generate (must be positive). For multi-variable datasets, this refers to the observations in the variable with the highest density.
- **num_dims** (int): Number of dimensions in the coordinate space (must be positive)
- **ratio_dims** (tuple): Tuple with `num_dims` elements defining the relative size of each dimension (all must be positive)
- **density** (float, list, or tuple, optional): 
  - Float: Fraction of grid points that contain observations, range (0.0, 1.0]
  - 2-element list/tuple `[max, min]`: var0 gets max, one other variable gets min, the rest are drawn from the range
  - num_vars-element list/tuple: Specific density for each variable, largest first (see **Density Options for Multiple Variables**)

  There is a **lower bound**, because a dataset is only generated if every coordinate on every
  axis is used at least once — an unused coordinate would be stored without describing any data
  point. Each observation supplies one coordinate per axis, so covering the longest axis takes at
  least that many observations:

  ```
  minimum observations = max(shape)        minimum density = max(shape) / prod(shape)
  ```

  A density below this raises, and so does `density=0.0` (or `sparsity=1.0`): the grid size is
  `num_obs / density`, so the density must be positive. On a grid of
  `d` equal axes of size `n` it works out to the familiar `1/n^(d-1)`. Note that it is the
  **longest** axis that sets the bound, not the shortest, and that axes of length 1 cost nothing:
  a `50x10x1x1` grid needs the same 50 observations as `50x10`. `docs/explainer.md` derives this.
- **sparsity** (float, list, or tuple, optional): Fraction of grid points that are vacant, `sparsity = 1 - density`. Accepts the same float/list/tuple forms as `density` and is converted to `density` internally. Provide either `density` or `sparsity` (not both).
- **seed** (int): Random seed for reproducibility (non-negative integer). Required.
- **max_obs** (int, optional): Maximum number of observations per chunk (default: `None`, serial generation). When `num_obs` exceeds this value, the dataset is split into chunks and generated in parallel (see **Parallel Generation for Large Datasets**).
- **max_workers** (int, optional): Maximum number of processes running at once in parallel generation (default: the CPU count)
- **dtype** (str or list, optional): What each variable holds: `float64` (default) or
  `float32`. One value for all variables, or one per variable.
- **pack** (str or list, optional): Integer type to compact the values into on disk — `int8`,
  `int16`, `int32` — or `None` (default) to store them plain. Values are carried as
  `scale_factor * code + add_offset`, the convention GLORYS12 and most reanalysis products use.
  The scale comes from the value range, not from the data, so parallel chunks agree with a
  serial run. Parquet stores the decoded type (`float32` for a packed `int16`), because packing
  is a netCDF device and parquet's idiom is the natural type.

  `dtype` and `pack` are separate because a file holds both kinds. Argo stores `TEMP` as a plain
  `float32` and `CYCLE_NUMBER` as a plain `int32`, neither packed.
- **fill_value** (float or list, optional): What marks a vacant site (default: NaN for float
  dtypes, the reserved code for integers). One value for all variables, or one per variable.
  Argo files use `99999.0` in a `float32` variable, in preference to NaN; that is expressible
  here. For packed dtypes the fill code is reserved, so no real value can collide with it.
- **num_vars** (int, optional): Number of variables in the dataset (default: 1)
- **var_dims** (int, list, or tuple, optional): 
  - Int: Number of dimensions for each variable (randomly selected if less than num_dims)
  - List of ints: Number of dimensions for each variable
  - List of lists/tuples: Explicit dimension indices for each variable
  - Default: All variables use all dimensions
- **overlap** (float, list, or str, optional): Control overlap between variables (default: 'random')
  - `'random'`: No constraint on overlap
  - Float [0.0, 1.0]: Target share of var0's sites that also carry each non-reference variable (see **Overlap Control**)
  - List of length `num_vars - 1`: One overlap target per non-reference variable
- **fixed_overlap** (bool or list of bools, optional): When `True`, variables draw their overlapping sites from one shared ordering of var0's sites, so they also overlap each other (default: `False`)

## Testing

The repository includes a test suite covering all modules.

### Running Tests

After installing the environment with test dependencies, run the test suite:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=data_sparsity --cov-report=html

# Run specific test file
pytest tests/validators/test_parameter_validator.py

# Run specific test class
pytest tests/generators/test_coordinate_generator.py::TestGenerateDimensionCoords

# Run tests matching a pattern
pytest -k "test_2d"
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Apache 2.0

## Authors

Enrico Milanese (enrico.milanese@whoi.edu)
