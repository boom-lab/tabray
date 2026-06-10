# Refactoring Plan: Fix var_dims for Reference Variable

## Problem Statement

Currently, the `var_dims` parameter is used to compute `total_grid_points` which assumes all variables use the full grid. However, when computing minimum overlap feasibility, we need to consider that the reference variable (with most observations) only has access to grid points in its varying dimensions.

## Root Cause

The computation of minimum overlap in `MultiVarOverlapConfig.compute_min_overlap()` uses `total_grid_points` (product of all dimensions), but the reference variable might only vary in a subset of dimensions, giving it fewer available grid points.

For example:
- Grid shape: [9, 9, 9] → total_grid_points = 729
- Var 0 varying dims: [0, 2] → effective grid points = 9*9 = 81
- Var 0 observations: 109

This causes: 109 observations cannot fit in 81 grid points without overlap, but the code thinks there are 729 points available.

## Solution Approach

We need to:
1. Ensure the reference variable's `var_dims_indices` entry is `[]` (empty) to indicate it uses ALL dimensions
2. Update `compute_min_overlap` to compute effective grid points for the reference variable
3. Pass `var_dims_indices` to the overlap calculation so it can compute the correct grid size

### Option 1: Always make reference variable use all dimensions (RECOMMENDED)

**Rationale**: The reference variable should use all dimensions to maximize available grid space and avoid sampling issues.

**Changes**:
- Modify `MultiVarDimensionsConfig.setup_from_parameter()` to ensure the variable with most observations gets `var_dims_indices = []` (meaning all dims)
- Update `compute_min_overlap()` to calculate effective grid points per variable based on their varying dimensions
- Pass `var_dims_indices` to overlap validation

### Option 2: Compute effective grid points for reference variable

**Rationale**: Keep current behavior but fix the overlap calculation.

**Changes**:
- Pass `var_dims_indices` to `compute_min_overlap()`
- Calculate effective grid points for reference variable based on its varying dimensions
- Use this reduced grid size for minimum overlap calculation

## Implementation Steps

### Step 1: Modify MultiVarOverlapConfig

Update `compute_min_overlap()` signature to accept `var_dims_indices`:

```python
@staticmethod
def compute_min_overlap(
    shape: List[int],
    var_num_obs: np.ndarray,
    var_dims_indices: List[List[int]]
) -> float:
    # Compute effective grid points for reference variable
    sorted_indices = np.argsort(var_num_obs)[::-1]
    refvar_idx = int(sorted_indices[0])
    refvar_varying_dims = var_dims_indices[refvar_idx]
    
    if len(refvar_varying_dims) == 0:
        # Empty list means all dimensions vary
        refvar_grid_points = int(np.prod(shape))
    else:
        refvar_grid_points = int(np.prod([shape[d] for d in refvar_varying_dims]))
    
    # Rest of logic using refvar_grid_points instead of total_grid_points
```

### Step 2: Update setup_from_parameter() signature

Update `MultiVarOverlapConfig.setup_from_parameter()`:

```python
@staticmethod
def setup_from_parameter(
    overlap: Union[float, str],
    num_vars: int,
    shape: List[int],
    var_num_obs: np.ndarray,
    var_dims_indices: List[List[int]]
) -> Union[float, str]:
```

### Step 3: Update generate_data.py

Update the call site in `_configure_multi_var()`:

```python
self.overlap_target = MultiVarOverlapConfig.setup_from_parameter(
    self.overlap, self.num_vars, self.shape, self.var_num_obs, self.var_dims_indices
)
```

### Step 4: Update tests

Update all tests that call `MultiVarOverlapConfig.compute_min_overlap()` or `setup_from_parameter()` to pass the new parameter.

## Files to Modify

1. `data_sparsity/config/multi_var_overlap.py` - Update methods
2. `data_sparsity/generate_data.py` - Update call site
3. `tests/config/test_multi_var_overlap.py` - Update test calls
4. Documentation strings - Update parameter descriptions

## Testing Strategy

1. Run existing tests to verify no regressions
2. Add specific test case with var_dims=[2, 2] and verify overlap calculation
3. Run the failing test case to verify it passes
4. Add edge case tests for empty var_dims_indices

## Backward Compatibility

This change modifies the API signature of `MultiVarOverlapConfig` methods. Since this is an internal API (not exposed to end users), this is acceptable. External API (`GenerateData` constructor) remains unchanged.
