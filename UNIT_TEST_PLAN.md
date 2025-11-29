# Unit Test Plan for data_sparsity

## Overview

This plan covers comprehensive unit testing for all refactored modules. The goal is deterministic testing with >90% code coverage using pytest.

---

## Test Structure

```
tests/
├── conftest.py                          # Shared fixtures
├── validators/
│   ├── __init__.py
│   ├── test_parameter_validator.py      # 30 tests
│   ├── test_dimension_validator.py      # 35 tests
│   └── test_sparsity_validator.py       # 25 tests
├── config/
│   ├── __init__.py
│   ├── test_multi_var_sparsity.py       # 35 tests
│   ├── test_multi_var_dimensions.py     # 40 tests
│   └── test_multi_var_overlap.py        # 25 tests
├── generators/
│   ├── __init__.py
│   ├── test_coordinate_generator.py     # 15 tests
│   ├── test_observation_generator.py    # 10 tests
│   ├── test_record_generator.py         # 30 tests
│   ├── test_single_var_generator.py     # 20 tests
│   ├── test_multi_var_generator.py      # 35 tests
│   ├── test_overlap_mapper.py           # 30 tests
│   └── test_overlap_calculator.py       # 25 tests
├── output/
│   ├── __init__.py
│   ├── test_path_manager.py             # 30 tests
│   ├── test_netcdf_builder.py           # 25 tests
│   └── test_parquet_builder.py          # 25 tests
└── test_generate_data.py                # 50 integration tests

Total: ~460 tests
```

---

## Testing Strategy

### 1. Test Categories

#### A. Unit Tests (Isolated)
- Test individual methods with mocked dependencies
- Use synthetic data with known outputs
- Deterministic: same input = same output (fixed seeds)

#### B. Integration Tests
- Test module interactions
- Test GenerateData class end-to-end
- Verify file I/O with temporary directories

#### C. Edge Case Tests
- Boundary conditions
- Error handling
- Invalid inputs

#### D. Regression Tests
- Ensure refactored code matches original behavior
- Compare outputs with original implementation

---

## Module-by-Module Test Plan

### validators/test_parameter_validator.py (30 tests)

**Method: validate_num_obs (6 tests)**
- ✓ Valid positive integer
- ✓ Zero raises ValueError
- ✓ Negative raises ValueError
- ✓ Float raises TypeError
- ✓ String raises TypeError
- ✓ None raises TypeError

**Method: validate_sparsity_type (8 tests)**
- ✓ Valid float returns float
- ✓ Valid int returns float
- ✓ Valid list returns max
- ✓ Valid tuple returns max
- ✓ Negative raises ValueError
- ✓ > 1.0 raises ValueError
- ✓ String raises TypeError
- ✓ Empty list raises ValueError

**Method: validate_num_dims (6 tests)**
- ✓ Valid positive integer
- ✓ Zero raises ValueError
- ✓ Negative raises ValueError
- ✓ Float raises TypeError
- ✓ String raises TypeError
- ✓ None raises TypeError

**Method: validate_ratio_dims (6 tests)**
- ✓ Integer 1 converts to list
- ✓ List passes through as array
- ✓ Tuple converts to array
- ✓ Length mismatch raises ValueError
- ✓ Invalid type raises TypeError
- ✓ Integer != 1 raises ValueError

**Method: validate_seed (2 tests)**
- ✓ Valid non-negative integer
- ✓ Negative raises ValueError

**Method: validate_num_vars (2 tests)**
- ✓ Valid positive integer
- ✓ Zero/negative raises ValueError

---

### validators/test_dimension_validator.py (35 tests)

**Method: compute_nb_coords_dim1 (8 tests)**
- ✓ Valid parameters return expected value
- ✓ Different num_obs values
- ✓ Different sparsity values
- ✓ Different ratio_dims_prod values
- ✓ Different num_dims values
- ✓ Result < 1 raises ValueError
- ✓ Large values handled correctly
- ✓ Edge case: sparsity=1

**Method: round_to_integer (5 tests)**
- ✓ Rounds up correctly (e.g., 10.6 -> 11)
- ✓ Rounds down correctly (e.g., 10.4 -> 10)
- ✓ Exact integers unchanged
- ✓ Prints correct message (capture stdout)
- ✓ Returns integer type

**Method: compute_nb_coords_per_dim (5 tests)**
- ✓ Uniform ratio_dims
- ✓ Non-uniform ratio_dims
- ✓ Single dimension
- ✓ Many dimensions (10+)
- ✓ Result shape matches ratio_dims shape

**Method: validate_min_elements_per_dim (6 tests)**
- ✓ All valid (>= 1) passes
- ✓ One dimension < 1 raises ValueError
- ✓ Multiple dimensions < 1 raises ValueError
- ✓ Exactly 1 passes
- ✓ Error message contains dimension index
- ✓ Zero raises ValueError

**Method: validate_integer_elements (6 tests)**
- ✓ Integer values pass through unchanged
- ✓ Close-to-integer values rounded
- ✓ Non-integer values raise ValueError
- ✓ Prints old and new values
- ✓ Returns integer array
- ✓ Multiple non-integer values error message

**Method: compute_shape_and_grid_points (5 tests)**
- ✓ 1D array
- ✓ 2D array
- ✓ 3D array
- ✓ High-dimensional array (10D)
- ✓ Grid points = product of shape

---

### validators/test_sparsity_validator.py (25 tests)

**Method: compute_min_sparsity (5 tests)**
- ✓ Uniform dimensions
- ✓ Non-uniform dimensions
- ✓ Single dimension
- ✓ Returns 1/min(dims)
- ✓ Large dimensions

**Method: validate_sparsity_bounds (10 tests)**
- ✓ Valid sparsity in range passes
- ✓ Sparsity = 0 returns minimum
- ✓ Sparsity < minimum (not 0) raises ValueError
- ✓ Sparsity = minimum passes
- ✓ Sparsity = 1 passes
- ✓ Sparsity > 1 raises ValueError (from type validation)
- ✓ Prints minimum sparsity
- ✓ Prints adjustment message for 0
- ✓ Edge case: sparsity = minimum + epsilon
- ✓ Edge case: sparsity = minimum - epsilon

**Method: validate_num_obs_consistency (10 tests)**
- ✓ Matching num_obs passes unchanged
- ✓ Non-matching num_obs adjusted
- ✓ Prints adjustment message
- ✓ Returns integer
- ✓ Returns adjusted sparsity
- ✓ Adjusted sparsity in bounds
- ✓ Edge case: large grid
- ✓ Edge case: small grid
- ✓ Out of bounds after adjustment raises ValueError
- ✓ Sparsity recalculation correct

---

### config/test_multi_var_sparsity.py (35 tests)

**Method: from_scalar (5 tests)**
- ✓ Single variable
- ✓ Two variables
- ✓ Many variables
- ✓ Array shape correct
- ✓ All elements equal

**Method: from_two_element_list (10 tests)**
- ✓ Two variables: random assignment (test with seeds)
- ✓ Three variables: includes min, max, random
- ✓ Many variables: distribution correct
- ✓ All elements in [min, max]
- ✓ Contains exactly one min
- ✓ Contains exactly one max
- ✓ Reproducible with same seed
- ✓ Different with different seed
- ✓ Order shuffled (not always sorted)
- ✓ Array length matches num_vars

**Method: from_full_list (5 tests)**
- ✓ Correct length matches
- ✓ Values preserved
- ✓ Wrong length raises ValueError
- ✓ Empty list raises ValueError
- ✓ Single element for single var

**Method: validate_and_clip (8 tests)**
- ✓ All valid values pass
- ✓ Value below minimum clipped
- ✓ Multiple values clipped
- ✓ Prints warning for clipping
- ✓ Value > 1 raises ValueError
- ✓ Negative value raises ValueError
- ✓ Zero handled correctly
- ✓ Returns modified array

**Method: compute_var_num_obs (5 tests)**
- ✓ Single variable
- ✓ Equal sparsities
- ✓ Different sparsities
- ✓ Max sparsity gets full num_obs
- ✓ Minimum 1 observation per variable

**Method: setup_from_parameter (2 tests)**
- ✓ Integration test: scalar input
- ✓ Integration test: list input

---

### config/test_multi_var_dimensions.py (40 tests)

**Method: select_random_dims (8 tests)**
- ✓ Correct count returned
- ✓ All elements in valid range
- ✓ No duplicates
- ✓ Sorted output
- ✓ Reproducible with same seed
- ✓ Different with different seed
- ✓ Edge case: select 1 dim
- ✓ Edge case: select all dims

**Method: from_int (8 tests)**
- ✓ All dims: all variables identical
- ✓ Subset dims: random selection
- ✓ Exceeds num_dims raises ValueError
- ✓ Single dimension per variable
- ✓ Reproducible with seed
- ✓ num_vars copies for all dims
- ✓ Length matches num_vars
- ✓ Each element sorted

**Method: from_list_element (10 tests)**
- ✓ Integer element: random selection
- ✓ Integer element: all dims
- ✓ List element: preserved
- ✓ Tuple element: converted to list
- ✓ Empty list raises ValueError
- ✓ Out of range raises ValueError
- ✓ Duplicates raise ValueError
- ✓ Negative index raises ValueError
- ✓ Invalid type raises TypeError
- ✓ Result sorted

**Method: from_list (6 tests)**
- ✓ All integers
- ✓ All lists
- ✓ Mixed integers and lists
- ✓ Wrong length raises ValueError
- ✓ Each element validated
- ✓ Length matches num_vars

**Method: compute_constant_dims (5 tests)**
- ✓ No varying dims: all constant
- ✓ All varying dims: no constant
- ✓ Mixed: correct complement
- ✓ Single dimension varying
- ✓ Length matches num_vars

**Method: preselect_constant_coord_indices (3 tests)**
- ✓ Creates RNG for each constant dim
- ✓ Empty dict for no constant dims
- ✓ Correct structure returned

---

### config/test_multi_var_overlap.py (25 tests)

**Method: validate_overlap_value (8 tests)**
- ✓ Valid float in [0,1] passes
- ✓ Float 0 passes
- ✓ Float 1 passes
- ✓ Float < 0 raises ValueError
- ✓ Float > 1 raises ValueError
- ✓ String 'random' passes
- ✓ Invalid string raises ValueError
- ✓ Invalid type raises TypeError

**Method: compute_min_overlap (10 tests)**
- ✓ No shared dimensions: 0
- ✓ All shared dimensions: 1
- ✓ Partial overlap: correct ratio
- ✓ Two variables
- ✓ Many variables
- ✓ Different dimension combinations
- ✓ Single variable: 0
- ✓ Empty list: 0
- ✓ Complex scenario
- ✓ Edge case: one var 1 dim, other all dims

**Method: validate_overlap_feasibility (5 tests)**
- ✓ Feasible overlap passes
- ✓ Infeasible raises ValueError
- ✓ Single variable passes (no check)
- ✓ Random overlap passes (no check)
- ✓ Exact minimum passes

**Method: setup_from_parameter (2 tests)**
- ✓ Integration test: float overlap
- ✓ Integration test: 'random' overlap

---

### generators/test_coordinate_generator.py (15 tests)

**Method: generate_dimension_coords (8 tests)**
- ✓ Correct length
- ✓ Values in [0,1]
- ✓ Sorted ascending
- ✓ Reproducible with seed
- ✓ Different with different seed
- ✓ Single coordinate
- ✓ Many coordinates (1000+)
- ✓ Returns numpy array

**Method: generate_all_coords (7 tests)**
- ✓ 1D grid
- ✓ 2D grid
- ✓ 3D grid
- ✓ List length matches shape length
- ✓ Each array correct length
- ✓ All arrays sorted
- ✓ Reproducible with seed

---

### generators/test_observation_generator.py (10 tests)

**Method: generate_observations (10 tests)**
- ✓ Correct length
- ✓ Values in [0,1]
- ✓ Reproducible with seed
- ✓ Different with different seed
- ✓ Single observation
- ✓ Many observations (10000+)
- ✓ Returns numpy array
- ✓ Float dtype
- ✓ No NaN values
- ✓ Statistical distribution approximately uniform

---

### generators/test_record_generator.py (30 tests)

**Method: initialize_record (5 tests)**
- ✓ 1D shape
- ✓ 2D shape
- ✓ 3D shape
- ✓ All values NaN
- ✓ Correct dtype (float)

**Method: generate_flat_indices (8 tests)**
- ✓ Correct count
- ✓ All unique (no duplicates)
- ✓ All in valid range
- ✓ Reproducible with seed
- ✓ Different with different seed
- ✓ num_obs = total_points (all points)
- ✓ num_obs = 1 (single point)
- ✓ Large grid

**Method: convert_to_multi_indices (6 tests)**
- ✓ 1D shape
- ✓ 2D shape
- ✓ 3D shape
- ✓ Tuple length matches shape length
- ✓ Each array length matches num_indices
- ✓ Round-trip: ravel then unravel

**Method: assign_observations (5 tests)**
- ✓ Values assigned at indices
- ✓ Other values remain NaN
- ✓ Modifies in-place
- ✓ Correct observation values
- ✓ Multiple assignments

**Method: validate_sparsity (6 tests)**
- ✓ Matching sparsity passes
- ✓ Close sparsity passes (within tolerance)
- ✓ Different sparsity raises ValueError
- ✓ Error message contains values
- ✓ Edge case: sparsity = 1
- ✓ Edge case: sparsity very small

---

### generators/test_single_var_generator.py (20 tests)

**Method: generate (20 tests)**
- ✓ Basic 2D generation
- ✓ 3D generation
- ✓ 1D generation
- ✓ High-dimensional (5D+)
- ✓ Correct shape
- ✓ Correct number of observations
- ✓ Values in [0,1]
- ✓ Reproducible with seed
- ✓ Different with different seed
- ✓ Pre-provided observations used
- ✓ Sparsity validation passes
- ✓ Sparsity validation fails
- ✓ All grid points used (sparsity=1)
- ✓ Single observation
- ✓ Large grid with low sparsity
- ✓ Small grid with high sparsity
- ✓ No observation overlap (uniqueness)
- ✓ Returns numpy array
- ✓ NaN at non-observation points
- ✓ Non-NaN at observation points

---

### generators/test_multi_var_generator.py (35 tests)

**Test _compute_var_shapes (5 tests)**
- ✓ No constant dims: same as full shape
- ✓ With constant dims: size 1 for constant
- ✓ Multiple variables
- ✓ All constant dims
- ✓ Dictionary keys correct

**Test _select_constant_coords (5 tests)**
- ✓ Selects valid coordinate indices
- ✓ Uses pre-seeded RNGs
- ✓ No constant dims: empty dict
- ✓ Multiple constant dims per var
- ✓ Different values per variable

**Test _expand_to_full_coords (5 tests)**
- ✓ Constant dims filled with constant value
- ✓ Varying dims unchanged
- ✓ Correct tuple length
- ✓ Correct array lengths
- ✓ Multiple constant dims

**Test generate_without_overlap (8 tests)**
- ✓ Two variables
- ✓ Many variables
- ✓ Different observation counts
- ✓ Different varying dimensions
- ✓ All records have correct shape
- ✓ Correct number of observations per var
- ✓ Independent placement (low overlap expected)
- ✓ Reproducible with seed

**Test generate_with_overlap (8 tests)**
- ✓ Two variables with overlap
- ✓ Many variables with overlap
- ✓ Overlap target respected (approximately)
- ✓ Reference variable fully generated
- ✓ Secondary variables have overlap portion
- ✓ Secondary variables have separate portion
- ✓ Reproducible with seed
- ✓ High overlap (0.9+)

**Test generate (main method) (4 tests)**
- ✓ Random overlap uses without_overlap path
- ✓ Numeric overlap uses with_overlap path
- ✓ Single variable edge case
- ✓ Returns correct tuple

---

### generators/test_overlap_mapper.py (30 tests)

**Test identify_random_assign_dims (5 tests)**
- ✓ Source=1, target>1: identified
- ✓ Source>1, target=1: not identified
- ✓ Both same size: not identified
- ✓ Multiple matching dimensions
- ✓ No matching dimensions

**Test map_single_coordinate (10 tests)**
- ✓ Direct mapping (both vary)
- ✓ Target constant (target=1)
- ✓ Source constant (None marker)
- ✓ Out of range: invalid
- ✓ Multiple dimensions
- ✓ All valid scenarios
- ✓ All constant in target
- ✓ All constant in source
- ✓ Mixed scenarios
- ✓ Tuple length matches

**Test try_random_assignment (8 tests)**
- ✓ Finds unused index
- ✓ Multiple attempts if needed
- ✓ Returns None if exhausted
- ✓ Uses RNG correctly
- ✓ Respects max_attempts
- ✓ Avoids used indices
- ✓ Single random dim
- ✓ Multiple random dims

**Test map_indices_for_overlap (7 tests)**
- ✓ Full mapping possible
- ✓ Partial mapping (some invalid)
- ✓ No random dimensions
- ✓ With random dimensions
- ✓ Insufficient mappings warning
- ✓ Returns correct count when possible
- ✓ Reproducible with seed

---

### generators/test_overlap_calculator.py (25 tests)

**Test extract_coordinate_set (6 tests)**
- ✓ 2D record
- ✓ 3D record
- ✓ Sparse record
- ✓ Dense record
- ✓ Single observation
- ✓ Empty record (all NaN)

**Test project_coordinates (6 tests)**
- ✓ Project to single dimension
- ✓ Project to multiple dimensions
- ✓ Project to all dimensions (identity)
- ✓ Dimension order matters
- ✓ Duplicates eliminated
- ✓ Empty set handling

**Test compute_pairwise_overlap (8 tests)**
- ✓ Full overlap
- ✓ No overlap
- ✓ Partial overlap
- ✓ No shared dimensions: 0
- ✓ All shared dimensions
- ✓ Different dimension combinations
- ✓ One observation each
- ✓ Many observations

**Test compute_actual_overlap (5 tests)**
- ✓ Two variables
- ✓ Multiple variables
- ✓ Single variable: None returned
- ✓ No overlap: 0.0
- ✓ Full overlap: 1.0

---

### output/test_path_manager.py (30 tests)

**Test check_or_create_folder (12 tests)**
- ✓ Creates non-existent folder
- ✓ Existing empty folder passes
- ✓ Existing non-empty folder without overwrite raises ValueError
- ✓ Existing non-empty folder with overwrite cleans
- ✓ Removes .nc files
- ✓ Removes .parquet files
- ✓ Removes _metadata files
- ✓ Keeps other files
- ✓ Not a directory raises NotADirectoryError
- ✓ Permission error raises OSError
- ✓ Nested folder creation
- ✓ Prints appropriate messages

**Test prepare_netcdf_path (8 tests)**
- ✓ Adds .nc extension if missing
- ✓ Keeps .nc extension if present
- ✓ Creates parent directory
- ✓ Existing file without overwrite raises ValueError
- ✓ Existing file with overwrite passes
- ✓ No directory specified works
- ✓ Nested directories created
- ✓ Returns corrected path

**Test prepare_parquet_path (5 tests)**
- ✓ Creates parent directory
- ✓ No directory specified works
- ✓ Nested directories created
- ✓ Extension not added (unlike NetCDF)
- ✓ Returns path unchanged

**Test setup_output_paths (5 tests)**
- ✓ All paths specified
- ✓ Some paths None (skipped)
- ✓ All paths None
- ✓ Creates all directories
- ✓ Returns tuple of paths

---

### output/test_netcdf_builder.py (25 tests)

**Test create_default_attrs (5 tests)**
- ✓ All fields present
- ✓ Correct types
- ✓ Values match inputs
- ✓ ratio_dims converted to list
- ✓ Returns dictionary

**Test build_dataarray (8 tests)**
- ✓ 2D array
- ✓ 3D array
- ✓ Correct shape
- ✓ Correct coordinates
- ✓ Correct dimensions
- ✓ Correct name
- ✓ Attributes attached
- ✓ Returns xr.DataArray

**Test build_dataset (7 tests)**
- ✓ Single variable
- ✓ Multiple variables
- ✓ Correct data_vars
- ✓ Shared coordinates
- ✓ Attributes attached
- ✓ Each variable correct shape
- ✓ Returns xr.Dataset

**Test save_to_file (5 tests)**
- ✓ DataArray saved successfully
- ✓ Dataset saved successfully
- ✓ File exists after save
- ✓ Overwrite=False raises error
- ✓ Overwrite=True succeeds

---

### output/test_parquet_builder.py (25 tests)

**Test extract_non_nan_points (6 tests)**
- ✓ 2D record
- ✓ 3D record
- ✓ Correct columns
- ✓ Correct row count
- ✓ No NaN values in output
- ✓ Coordinate values correct

**Test build_single_var_dataframe (5 tests)**
- ✓ Calls extract_non_nan_points
- ✓ Returns DataFrame
- ✓ Correct shape
- ✓ Has 'record' column
- ✓ Has coordinate columns

**Test build_multi_var_dataframe (9 tests)**
- ✓ Two variables
- ✓ Multiple variables
- ✓ Correct columns (coords + vars)
- ✓ Shared coordinates merged
- ✓ Non-shared coords have pd.NA
- ✓ One row per unique coordinate
- ✓ Correct variable values
- ✓ Returns DataFrame
- ✓ No duplicate coordinates

**Test save_to_file (5 tests)**
- ✓ pandas DataFrame saved
- ✓ dask DataFrame saved
- ✓ File exists after save
- ✓ chunk_id modifies filename
- ✓ Overwrite handling

---

### test_generate_data.py (50 integration tests)

**Initialization tests (10 tests)**
- ✓ Minimal valid parameters
- ✓ Full parameters specified
- ✓ Invalid parameters raise errors
- ✓ Validation runs automatically
- ✓ Attributes set correctly
- ✓ Single variable defaults
- ✓ Multi-variable setup
- ✓ max_obs triggers parallel mode
- ✓ Prints configuration
- ✓ RNG initialized

**Single-variable generation (10 tests)**
- ✓ 1D generation
- ✓ 2D generation
- ✓ 3D generation
- ✓ High-dimensional generation
- ✓ Low sparsity
- ✓ High sparsity
- ✓ Different seeds produce different results
- ✓ Same seed reproduces results
- ✓ NetCDF output created
- ✓ Parquet output created

**Multi-variable generation (15 tests)**
- ✓ Two variables
- ✓ Many variables (5+)
- ✓ Different sparsities per variable
- ✓ Different dimensions per variable
- ✓ Random overlap
- ✓ Zero overlap
- ✓ Full overlap
- ✓ Partial overlap (0.5)
- ✓ Overlap calculation correct
- ✓ Dataset has all variables
- ✓ DataFrame has all columns
- ✓ Constant dimensions respected
- ✓ Varying dimensions correct
- ✓ Reproducible with seed
- ✓ Files created correctly

**Edge cases (10 tests)**
- ✓ Single observation
- ✓ Maximum sparsity (1.0)
- ✓ Minimum sparsity
- ✓ Single dimension
- ✓ Many dimensions (10+)
- ✓ Non-uniform dimension ratios
- ✓ Very small grid (2x2)
- ✓ Very large num_obs with adjustment
- ✓ All variables same dimensions
- ✓ All variables different dimensions

**Error handling (5 tests)**
- ✓ Invalid sparsity raises error
- ✓ Invalid overlap raises error
- ✓ Infeasible overlap raises error
- ✓ Invalid file paths raise error
- ✓ Type errors caught

---

## Testing Tools & Setup

### Required Packages
```bash
pip install pytest pytest-cov pytest-mock numpy pandas xarray
```

### conftest.py - Shared Fixtures
```python
import pytest
import numpy as np
import tempfile
import shutil

@pytest.fixture
def fixed_rng():
    """Provide a fixed random number generator for reproducibility."""
    return np.random.default_rng(42)

@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after test."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)

@pytest.fixture
def sample_coordinates_2d():
    """Provide sample 2D coordinates."""
    return {
        'x0': np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
        'x1': np.array([0.2, 0.4, 0.6, 0.8])
    }

@pytest.fixture
def sample_record_2d():
    """Provide sample 2D record with sparse data."""
    record = np.full((5, 4), np.nan)
    record[0, 0] = 0.5
    record[2, 1] = 0.7
    record[4, 3] = 0.3
    return record
```

---

## Coverage Goals

| Module                      | Target Coverage |
|-----------------------------|----------------|
| validators/*                | 95%+           |
| config/*                    | 90%+           |
| generators/* (except mapper)| 90%+           |
| generators/overlap_mapper   | 85%+ (complex) |
| output/*                    | 90%+           |
| generate_data.py            | 85%+           |
| **Overall**                 | **90%+**       |

---

## Test Execution Strategy

### Phase 1: Validators (Week 1)
- Start with simplest modules
- Establish testing patterns
- Set up CI/CD integration

### Phase 2: Config (Week 1-2)
- More complex logic
- More edge cases
- Random behavior testing

### Phase 3: Generators (Week 2-3)
- Most complex modules
- Careful fixture design
- Performance considerations

### Phase 4: Output (Week 3)
- File I/O testing
- Temp directory management
- Format validation

### Phase 5: Integration (Week 4)
- End-to-end tests
- Regression tests
- Performance benchmarks

---

## Success Criteria

- ✅ All 460 tests pass
- ✅ 90%+ code coverage
- ✅ All edge cases covered
- ✅ All error paths tested
- ✅ Tests run in < 60 seconds
- ✅ No test interdependencies
- ✅ Deterministic (no flaky tests)
- ✅ Well-documented fixtures
- ✅ Clear test names
- ✅ CI/CD integrated

---

## Implementation Priority

**High Priority (Must Have)**
1. Validators - Foundation for everything
2. Config - Complex logic needs verification
3. Single-var generators - Core functionality
4. Output builders - User-facing

**Medium Priority (Should Have)**
5. Multi-var generators - Complex but well-isolated
6. Overlap calculator - Can be tested independently
7. Integration tests - Verify overall system

**Lower Priority (Nice to Have)**
8. Overlap mapper - Complex, well-tested via integration
9. Edge cases - Additional confidence
10. Performance tests - Optimization opportunity

---

## Notes

- **Determinism**: All tests use fixed seeds for reproducibility
- **Isolation**: Each test is independent, no shared state
- **Speed**: Fast tests (<60s total) encourage frequent running
- **Coverage**: Focus on high-value code paths first
- **Documentation**: Each test has clear docstring explaining what it verifies

---

**Total Estimated Tests: ~460**  
**Estimated Implementation Time: 4 weeks**  
**Expected Coverage: 90%+**
