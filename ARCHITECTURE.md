# GenerateData Class - Architecture Documentation

## Overview

The `GenerateData` class provides a clean, modular interface for generating synthetic observation data in both array (NetCDF) and tabular (Parquet) formats. This document describes the architectural decisions made during implementation.

## Design Principles

1. **Modularity**: Each operation (validation, coordinate generation, observation generation, etc.) is separated into its own method
2. **Clarity**: Code is written to be understandable by non-expert Python users
3. **Robustness**: Comprehensive input validation with clear error messages
4. **Reproducibility**: Random seed parameter ensures reproducible results
5. **Style Consistency**: Follows the coding style of the CrocoCamp project

## Class Structure

### Public Interface

- `__init__(num_obs, num_dims, ratio_dims, sparsity, seed)`: Initialize with validation
- `generate(netcdf_filepath, parquet_filepath)`: Main orchestration method
- `save_to_netcdf(filepath, overwrite)`: Save array data to NetCDF
- `save_to_parquet(dirname, filename, overwrite)`: Save tabular data to Parquet dataset

### Private Methods

- `_validate_parameters()`: Performs 8 validation checks
- `_generate_coordinates()`: Creates coordinate arrays for each dimension
- `_generate_observations()`: Generates random observation values
- `_generate_record()`: Creates sparse multi-dimensional array
- `_create_dataarray()`: Builds xarray DataArray
- `_create_dataframe()`: Builds pandas DataFrame

## Parameter Validation

The `_validate_parameters()` method performs comprehensive validation with both hard and soft checks:

### Hard Checks (raises errors):
1. **num_obs type and value**: Must be a positive integer
2. **sparsity type and range**: Must be a number in [0.0, 1.0]
3. **num_dims type and value**: Must be a positive integer
4. **ratio_dims type and consistency**: Must be int (value 1), or list/tuple/array matching num_dims length
5. **Minimum dimension size**: All dimensions must have at least 1 element
6. **Integer dimension sizes**: All dimensions must have integer number of elements
7. **Minimum sparsity**: Sparsity must be above minimum theoretical value
8. **seed type and value**: Must be a non-negative integer

### Soft Checks (auto-corrects with warnings):
- **Dimension size rounding**: Non-integer dimension sizes are rounded to nearest integer
- **num_obs consistency**: Adjusted if inconsistent with sparsity and dimension sizes due to rounding
- **Zero sparsity**: Automatically set to minimum allowed value if input is exactly 0

The validation includes extensive print statements to inform users of any auto-corrections, ensuring transparency in parameter adjustments.

## Data Generation Pipeline

The `generate()` method orchestrates the following steps:

1. **Generate Coordinates**: Creates coordinate arrays for each dimension based on ratio_dims
   - Uses normalized ratios to determine relative dimension sizes
   - Ensures total grid points accommodate the specified sparsity level
   - Generates evenly-spaced coordinates in [0, 1) for each dimension

2. **Generate Observations**: Creates num_obs random values in [0, 1]
   - Uses numpy's random generator with the specified seed
   - Values are uniformly distributed

3. **Generate Record**: Creates sparse multi-dimensional array
   - Initializes full grid with NaN values
   - Randomly selects num_obs grid points (without replacement)
   - Assigns observation values to selected points

4. **Create DataArray**: Builds xarray structure
   - Includes all coordinates and dimension names
   - Adds metadata attributes (num_obs, sparsity, seed)
   - Names the data variable "record"

5. **Create DataFrame**: Builds pandas tabular structure
   - One row per observation (num_obs rows total)
   - One column per dimension plus one for the record value
   - Coordinate values are mapped from grid indices

6. **Save Files** (optional): Writes to disk if paths provided
   - NetCDF format preserves array structure
   - Parquet format preserves tabular structure

## Design Rationale

### Why xarray and pandas?

- **xarray**: Natural representation for gridded/array data with labeled dimensions
- **pandas**: Standard for tabular data in Python ecosystem
- Both are well-established, widely-used libraries with excellent I/O support

### Why separate coordinate and record generation?

Separation allows for:
- Independent testing of each component
- Easier debugging when issues arise
- Future extensions (e.g., different coordinate schemes)
- Clear conceptual boundaries in the code

### Why store intermediate results?

Storing `_coordinates`, `_observations`, `_record`, etc. as instance variables:
- Allows inspection of intermediate states for debugging
- Enables saving without re-generation
- Supports future methods that might need access to these components
- Minimal memory overhead for typical use cases

### Why use private methods?

The underscore prefix (`_method_name`) indicates internal implementation:
- Users should interact through the public `generate()` method
- Internal methods can be refactored without breaking user code
- Follows Python conventions for encapsulation

## Code Quality

- **PEP 8 Compliance**: Code follows Python style guidelines
- **PEP 257 Compliance**: All modules, classes, and functions have docstrings
- **PEP 484 Compliance**: Type hints on all method signatures
- **Pylint Score**: 9.69/10 (exceeds required 8.0)
- **Python Version**: Compatible with Python 3.12+

## Dependencies

Chosen for stability, wide adoption, and performance:

- `numpy>=1.24.0`: Array operations and random generation
- `pandas>=2.0.0`: DataFrame handling
- `dask`: Parallel computing and distributed Parquet datasets
- `pyarrow>=12.0.0`: Parquet format backend
- `xarray>=2023.1.0`: Labeled array operations
- `netCDF4>=1.6.0`: NetCDF format I/O
- `jupyterlab`: For running example notebooks

## Utility Module

The `data_sparsity.utils` module provides helper functions:

- `check_or_create_folder()`: Validates/creates directories with overwrite control
- `check_nc()`: Validates NetCDF file paths and creates necessary directories
- `check_parquet()`: Validates Parquet dataset directories

These utilities ensure proper file handling and prevent accidental overwrites unless explicitly requested.

## Future Extensions

The architecture supports easy extensions:

1. **Additional file formats**: New `save_to_*` methods
2. **Custom coordinate schemes**: Alternative `_generate_coordinates` implementations
3. **Different observation distributions**: Modify `_generate_observations`
4. **Metadata enhancement**: Extend DataArray/DataFrame attributes
5. **Validation customization**: Additional checks in `_validate_parameters`

## Usage Examples

See `examples.py` (Python script) or `notebooks/examples.ipynb` (Jupyter notebook) for comprehensive usage demonstrations including:
- Basic generation and saving to both NetCDF and Parquet formats
- Different sparsity levels and their effect on grid size
- Non-uniform dimensions with custom ratios
- Data inspection and attribute viewing
- Reproducibility verification with random seeds

## Key API Changes from Original Design

The implementation has evolved to include:

1. **Flexible ratio_dims**: Can now be specified as `int` (value `1` for equal dimensions) or as array-like
2. **Enhanced validation**: Includes automatic corrections with user notifications for rounding issues
3. **Dask integration**: Parquet datasets use Dask for better scalability
4. **Overwrite control**: All save methods now include explicit `overwrite` parameters
5. **Parquet as dataset**: `save_to_parquet()` creates a directory with multiple files rather than a single file
6. **Coordinate naming**: Dimensions are named `x0`, `x1`, etc. instead of `dim_0`, `dim_1`
7. **Verbose output**: Extensive print statements during initialization and validation for transparency
