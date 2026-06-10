# Bug Fix Summary: Multi-Variable Observation Count Issue

**Date:** December 2, 2024  
**Bug Report:** Variables getting zero or incorrect observation counts in multi-variable generation

## Problem Identified

The test `test_different_sparsities_per_variable` was failing because variables were getting incorrect observation counts:
- Expected: var0=131, var1=109, var2=87
- Actual: var0=211, var1=33, var2=0

## Root Cause

**Primary Issue:** The overlap feasibility validation only checked if the reference variable had enough grid space, but did not validate that NON-reference variables had sufficient grid points for their observations.

**Secondary Issue:** When non-reference variables have constrained grid space (e.g., varying in only 2 of 3 dimensions), they may not have enough grid points to accommodate both their overlap AND separate observations.

### Example Failure Case
```
Configuration:
- var0: 131 obs, varies in [0,1,2], grid = 9×9×9 = 729 points ✓
- var1: 109 obs, varies in [1,2], grid = 9×9 = 81 points ✗
- Overlap: 30%

What happens:
- var1 needs 109 total observations
- 30% overlap = 33 observations  
- Separate = 76 observations
- Total grid points = 81
- After overlap: 81 - 33 = 48 points available
- Problem: 48 < 76, so separate observations can't be generated!
```

The code in `multi_var_record_generator.py` would silently skip generating separate observations when `len(available_flat) < num_separate`, resulting in the variable having only the overlap observations.

## Solution Implemented

### 1. Added Per-Variable Grid Space Validation

Added `validate_per_variable_space()` method to `MultiVarOverlapConfig` that validates each variable has enough grid space for its observations:

```python
def validate_per_variable_space(
    shape: List[int],
    var_num_obs: np.ndarray,
    var_dims_indices: List[List[int]]
) -> None:
    """Validate that each variable has enough grid space for its observations."""
    for var_idx in range(len(var_num_obs)):
        var_varying_dims = var_dims_indices[var_idx]
        
        # Compute effective grid points
        if len(var_varying_dims) == 0 or len(var_varying_dims) == len(shape):
            var_grid_points = int(np.prod(shape))
        else:
            var_grid_points = int(np.prod([shape[d] for d in var_varying_dims]))
        
        if var_grid_points < var_num_obs[var_idx]:
            raise ValueError(
                f"Variable {var_idx} needs {var_num_obs[var_idx]} observations "
                f"but only has {var_grid_points} grid points..."
            )
```

### 2. Updated API Signature

Updated `MultiVarOverlapConfig.setup_from_parameter()` to accept `shape` and `var_dims_indices`:

**Before:**
```python
setup_from_parameter(overlap, num_vars, total_grid_points, var_num_obs)
```

**After:**
```python
setup_from_parameter(overlap, num_vars, shape, var_num_obs, var_dims_indices)
```

### 3. Updated Call Site

Updated `generate_data.py` to pass the new parameters:
```python
self.overlap_target = MultiVarOverlapConfig.setup_from_parameter(
    self.overlap, self.num_vars, self.shape, self.var_num_obs, self.var_dims_indices
)
```

### 4. Updated Tests

Updated all test configurations to use feasible parameter combinations:

| Test | Original Sparsity | Updated Sparsity | Reason |
|------|-------------------|------------------|--------|
| `test_two_variables` | [0.1, 0.15] | [0.15, 0.07] | var1 needs ≤81 obs |
| `test_many_variables` | [0.1, 0.12, 0.08, 0.15] | [0.10, 0.06, 0.04, 0.05] | All non-ref vars need ≤121 obs |
| `test_different_sparsities_per_variable` | [0.18, 0.15, 0.12] | [0.20, 0.07, 0.05] | var1, var2 need ≤81 obs |
| `test_different_dimensions_per_variable` | sparsity=0.15 | sparsity=0.08 | var1 needs ≤81 obs |
| `test_zero_overlap` | [0.15, 0.15] | [0.15, 0.04] | var1 with 1 dim needs ≤9 obs |
| `test_dataset_has_all_variables` | sparsity=0.1 | sparsity=0.05 | 5 vars need space |
| `test_dataframe_has_all_columns` | [0.1, 0.12] | [0.10, 0.07] | var1 needs ≤81 obs |
| `test_reproducible_with_seed` | [0.15, 0.18] | [0.15, 0.08] | var1 needs ≤64 obs |

## Files Modified

1. `data_sparsity/config/multi_var_overlap.py`
   - Added `validate_per_variable_space()` method
   - Updated `setup_from_parameter()` signature

2. `data_sparsity/generate_data.py`
   - Updated call to `setup_from_parameter()` with new parameters

3. `tests/test_generate_data.py`
   - Updated 8 test configurations with feasible sparsity values

4. `tests/config/test_multi_var_overlap.py`
   - Updated 2 tests to use new API signature

## Testing Status

- Before fix: 15 failing tests (including critical bug)
- After fix: 13 failing tests (bug fixed, remaining failures are parameter-related)
- Tests passing: 394/407 (96.8%)

## Remaining Work

Some tests still need parameter adjustments to ensure:
1. Non-reference variables have sufficient grid space
2. Sparsity values don't get clipped to minimum (which maximizes observations)
3. Edge cases are properly handled

## Benefits

1. **Bug Prevention:** Early validation prevents silent data generation failures
2. **Clear Error Messages:** Users get actionable feedback about infeasible configurations
3. **Correctness:** Ensures generated datasets match specified parameters
4. **Maintainability:** Validation logic is centralized and well-documented

## Related Documentation

- `BUG_ANALYSIS.md` - Detailed analysis of the original bug
- `REFACTOR_VAR_DIMS_PLAN.md` - Original refactoring plan (partially implemented)
