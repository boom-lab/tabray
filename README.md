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
- **aws_config_path** (str, optional): Path to AWS configuration YAML file for S3 access

## AWS S3 Support

The package supports storing data directly to AWS S3 buckets. You can use either environment variables or a configuration file for AWS credentials.

### Using Environment Variables

Set the following environment variables before running your code:

```bash
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="us-east-1"  # optional
```

Then use S3 paths in your code:

```python
from data_sparsity import GenerateData

gen = GenerateData(
    num_obs=1000,
    num_dims=3,
    ratio_dims=(1.0, 1.0, 1.0),
    sparsity=0.1,
    seed=42
)

# Save to S3
dataarray, dataframe = gen.generate(
    netcdf_filepath="s3://my-bucket/data/output_data.nc",
    parquet_filepath="s3://my-bucket/data/output_data_parquet"
)
```

### Using a Configuration File

1. Copy the template configuration file:
   ```bash
   cp aws_config_template.yaml aws_config.yaml
   ```

2. Edit `aws_config.yaml` with your AWS credentials:
   ```yaml
   aws_access_key_id: YOUR_ACCESS_KEY_ID
   aws_secret_access_key: YOUR_SECRET_ACCESS_KEY
   aws_default_region: us-east-1
   ```

3. Use the configuration file in your code:
   ```python
   from data_sparsity import GenerateData

   gen = GenerateData(
       num_obs=1000,
       num_dims=3,
       ratio_dims=(1.0, 1.0, 1.0),
       sparsity=0.1,
       seed=42,
       aws_config_path="aws_config.yaml"
   )

   # Save to S3
   dataarray, dataframe = gen.generate(
       netcdf_filepath="s3://my-bucket/data/output_data.nc",
       parquet_filepath="s3://my-bucket/data/output_data_parquet"
   )
   ```

**Note:** The `aws_config.yaml` file is gitignored and will not be committed to version control. Always use the template file (`aws_config_template.yaml`) as a reference and never commit your actual credentials.
