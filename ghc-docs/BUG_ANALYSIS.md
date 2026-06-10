# Bug Analysis: Zero Observations in Multi-Variable Generation

## Issue
Test `test_different_sparsities_per_variable` fails because variables get incorrect observation counts:
- Expected: var0=131, var1=109, var2=87
- Actual: var0=211, var1=33, var2=0

## Root Cause

The overlap feasibility validation (`compute_min_overlap`) only checks if the reference variable has enough space, but doesn't validate that NON-reference variables have sufficient grid points for their observations given the overlap requirement.

### Detailed Analysis

**Configuration:**
- var0: 131 obs, varies in [0,1,2], grid points = 9×9×9 = 729 ✓
- var1: 109 obs, varies in [1,2], grid points = 9×9 = 81 ✗
- var2: 87 obs, varies in [0,2], grid points = 9×9 = 81 ✗
- Overlap: 30%

**What happens:**
1. var1 needs 109 observations with 30% overlap:
   - Overlap: 33 observations
   - Separate: 76 observations  
   - Total needed: 33 + 76 = 109
   - Available: 81 grid points
   - After overlap: 81 - 33 = 48 points for separate
   - **Problem: 48 < 76, so separate observations can't be generated!**

2. var2 needs 87 observations with 30% overlap:
   - Overlap: 26 observations
   - Separate: 61 observations
   - Total needed: 26 + 61 = 87
   - Available: 81 grid points  
   - After overlap: 81 - 26 = 55 points for separate
   - **Problem: 55 < 61, so separate observations can't be generated!**

### Why This Happens

The code in `multi_var_record_generator.py` (lines 308-324) has:
```python
if num_separate > 0:
    used_flat = set(overlap_target_flat) if num_overlap > 0 else set()
    available_flat = np.array([
        i for i in range(var_total_points) if i not in used_flat
    ])
    
    if len(available_flat) >= num_separate:  # <-- This condition fails!
        # Generate separate observations
    # ELSE: Silently skip generating separate observations!
```

When the condition fails, no separate observations are generated, resulting in the variable having fewer observations than expected.

## Solution

The overlap feasibility validation needs to check that EACH variable has enough grid space for its observations:

```python
For each variable i with grid_points[i] and obs[i]:
    overlap_obs = overlap * obs[i]
    separate_obs = obs[i] - overlap_obs
    required_space = overlap_obs + separate_obs = obs[i]
    
    # But overlap observations can share space with reference variable,
    # so actual space needed in variable's own grid is just obs[i]
    # (since overlap can reuse any of the grid points)
    
    if grid_points[i] < obs[i]:
        FAIL: Not enough space
```

Actually, the constraint is simpler: each variable needs `obs[i] <= grid_points[i]`, regardless of overlap, because overlap just means some observations are at the same spatial locations as the reference variable, but they still occupy grid points in that variable's space.

Wait, let me reconsider... The issue is more subtle. With overlap:
- Overlap observations must map to valid locations in BOTH reference and target variable spaces
- Separate observations only need space in the target variable's space
- So: `num_overlap + num_separate = obs[i]` must fit in `grid_points[i]`

But overlap observations don't "use up" grid points in a special way - they're just observations that happen to be at locations that overlap with the reference variable.

The actual constraint is: **`obs[i] <= grid_points[i]`** for each variable i.

## Fix Required

1. Update `compute_min_overlap()` in `multi_var_overlap.py` to validate each variable's grid space
2. Add validation that each variable has `grid_points[i] >= obs[i]`
3. If validation fails, either:
   - Raise an error with clear message
   - Automatically adjust observation counts
   - Adjust overlap to make it feasible

The cleaner solution is to raise an error during validation, so users know their configuration is infeasible.

## Files to Modify

1. `data_sparsity/config/multi_var_overlap.py` - Add per-variable grid space validation
2. Tests - Update test parameters to be feasible
