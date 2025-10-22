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

### Parameters

- **num_obs** (int): Number of observations to generate (must be positive)
- **num_dims** (int): Number of dimensions in the coordinate space (must be positive)
- **ratio_dims** (tuple): Tuple with `num_dims` elements defining the relative size of each dimension (all must be positive)
- **sparsity** (float): Fraction of grid points that contain observations, range [0.0, 1.0]
- **seed** (int): Random seed for reproducibility (non-negative integer)
