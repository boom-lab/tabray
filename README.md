# Data Sparsity Analysis

The goal of this repository is to study the performance of _read_ operations of different data when stored in tabular or array structure. I'll start using netCDF for arrays and parquet for tabular, as they are the most commonly discussed. The number of permutations on how to do this are possibly infinite, so I'm picking a few guided by the following principles:

1. I want to illustrate limit cases: extremely regular ('gridded') data and extremely sparse data;
2. I want to capture some kind of transition between the two limit cases;
3. The metric of comparison is the user experience in the form of time to read or manipulate a given amount of observations;
4. I want to explore both smaller- and larger-than-memory cases.

Notes:

- Points 3 and 4 are related: I'll use read time as the metric for small enough data, and manipulation time (e.g. computing the mean, plotting, or so) for larger-than-memory data

## Repository structure

The repo is split in two main components. Python tools used to generate the data in array and tabular format, and a `benchmark` folder containing jupyter notebooks and codes in any language that can be used to read and benchmark performance.

## Installation

### Using pip

```bash
pip install -e .
```

### Using conda

```bash
conda env create -f environment.yml
conda activate data_sparsity
```

## Usage

### Basic Usage

Generate synthetic observation data and save to both NetCDF and Parquet formats:

```python
from data_sparsity import GenerateData

# Create a data generator
gen = GenerateData(
    num_obs=1000,           # Number of observations
    num_dims=3,             # Number of dimensions
    ratio_dims=(1.0, 1.0, 1.0),  # Relative size of each dimension
    sparsity=0.1,           # 10% of grid points contain observations
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
from data_sparsity import GenerateData

gen = GenerateData(
    num_obs=500,
    num_dims=2,
    ratio_dims=(2.0, 1.0),  # First dimension twice as large as second
    sparsity=0.2,
    seed=123
)

# Generate data without saving to disk
dataarray, dataframe = gen.generate()

# Access the data
print(dataarray)
print(dataframe.head())
```

### Parallel Generation for Large Datasets

When generating datasets larger than available memory, the tool automatically uses parallel processing to split the data into manageable chunks. **This now works for both single-variable and multi-variable datasets:**

```python
from data_sparsity import GenerateData

# Generate a large single-variable dataset with parallel processing
gen = GenerateData(
    num_obs=50_000_000,     # 50 million observations
    num_dims=3,
    ratio_dims=(2.0, 1.5, 1.0),
    sparsity=0.05,
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
    num_obs=30_000_000,     # 30 million observations (for highest sparsity variable)
    num_dims=4,
    ratio_dims=(2.0, 1.5, 1.0, 1.0),
    sparsity=[0.05, 0.10],  # Multiple variables with different sparsities
    seed=42,
    max_obs=10_000_000,
    num_vars=3,
    var_dims=[[0,1,2], [1,2,3], [0,2,3]],
    overlap=0.5             # 50% overlap between variables
)

gen_multi.generate(
    netcdf_filepath="large_multivar_data.nc",
    parquet_filepath="large_multivar_data.parquet"
)
```

**How it works:**
- The dataset is split along the largest dimension into multiple chunks
- Each chunk is generated independently using parallel processing with Dask
- For NetCDF: Multiple files are created (e.g., `large_data_0.nc`, `large_data_1.nc`, etc.)
- For Parquet: Chunks are generated, then repartitioned and consolidated into a single dataset
- Coordinates along shared dimensions are identical across chunks (using the same seed)
- Coordinates along the split dimension are unique per chunk
- **Multi-variable support:** All variables are generated for each chunk with proper overlap control
- **Overlap in parallel:** When overlap is specified, it's maintained within each chunk using consistent RNG strategies

**Memory Management:**
- Set `max_obs` based on available memory (default: 10 million observations)
- For multi-variable datasets, `max_obs` refers to the observations of the variable with highest sparsity
- Lower values create more chunks but use less memory per chunk
- Higher values reduce overhead but require more memory

### Multi-Variable Datasets

Generate datasets with multiple observation variables measured at potentially different points and dimensions:

```python
from data_sparsity import GenerateData

# Generate a dataset with 3 variables
gen = GenerateData(
    num_obs=1000,
    num_dims=3,
    ratio_dims=(1.0, 1.0, 1.0),
    sparsity=[0.1, 0.3],  # Variable 0 gets min, variable 2 gets max, variable 1 gets random
    seed=42,
    num_vars=3,           # Number of variables
    var_dims=3,           # Each variable uses all 3 dimensions
    overlap=0.5           # 50% of measurements overlap across variables
)

dataarray, dataframe = gen.generate(
    netcdf_filepath="multi_var_data.nc",
    parquet_filepath="multi_var_data.parquet"
)

# The result is an xarray.Dataset (not DataArray) with multiple data variables
print(dataarray.data_vars)  # ['record0', 'record1', 'record2']

# The DataFrame includes a 'variable' column identifying each observation
print(dataframe['variable'].value_counts())
```

**Sparsity Options for Multiple Variables:**
- Scalar (e.g., `0.2`): All variables have the same sparsity
- 2-element list/tuple (e.g., `[0.1, 0.3]`): One variable gets min, one gets max, rest are random
- num_vars-element list/tuple (e.g., `[0.1, 0.2, 0.3]`): Each variable gets its specified sparsity

**Variable Dimensions Options:**
- Int (e.g., `2`): Each variable randomly uses 2 dimensions
- List of ints (e.g., `[2, 3, 2]`): Each variable uses the specified number of dimensions (randomly selected)
- List of lists/tuples (e.g., `[[0,1], [1,2], [0,2]]`): Each variable uses explicitly specified dimensions

**Overlap Control:**
- `'random'` (default): No constraint, measurements placed independently
- Float `0.0` to `1.0`: Target percentage of measurements at same coordinates
  - `0.0`: No overlap (measurements at different points)
  - `1.0`: Complete overlap (all measurements at same points)
  - `0.5`: Half the measurements overlap

```python
# Example: 3 variables with different dimensions and controlled overlap
gen = GenerateData(
    num_obs=500,
    num_dims=4,
    ratio_dims=(1.0, 1.0, 1.0, 1.0),
    sparsity=0.2,
    seed=42,
    num_vars=3,
    var_dims=[[0,1,2], [1,2,3], [0,3]],  # Different dimensions per variable
    overlap=0.7  # 70% overlap along shared dimensions
)

dataset, df = gen.generate()
```

### Parameters

- **num_obs** (int): Number of observations to generate (must be positive). For multi-variable datasets, this refers to the observations in the variable with the highest sparsity.
- **num_dims** (int): Number of dimensions in the coordinate space (must be positive)
- **ratio_dims** (tuple): Tuple with `num_dims` elements defining the relative size of each dimension (all must be positive)
- **sparsity** (float, list, or tuple): 
  - Float: Fraction of grid points that contain observations, range [0.0, 1.0]
  - 2-element list/tuple: Min and max sparsity values; one variable gets min, one gets max, rest are random
  - num_vars-element list/tuple: Specific sparsity for each variable
- **seed** (int): Random seed for reproducibility (non-negative integer)
- **max_obs** (int, optional): Maximum number of observations per chunk when using parallel generation (default: 10,000,000). When `num_obs` exceeds this value, the dataset is automatically split into chunks and generated in parallel.
- **num_vars** (int, optional): Number of variables in the dataset (default: 1)
- **var_dims** (int, list, or tuple, optional): 
  - Int: Number of dimensions for each variable (randomly selected if less than num_dims)
  - List of ints: Number of dimensions for each variable
  - List of lists/tuples: Explicit dimension indices for each variable
  - Default: All variables use all dimensions
- **overlap** (float or str, optional): Control overlap between variables (default: 'random')
  - `'random'`: No constraint on overlap
  - Float [0.0, 1.0]: Target overlap percentage along shared dimensions
