# Parallel Workflow Analysis Report

**Date**: 2025-12-02  
**Analysis Type**: Comprehensive codebase review to determine parallel workflow functionality

## Executive Summary

The parallel workflow in the `data_sparsity` codebase **IS BROKEN** and does not generate equivalent output to the serial workflow. The primary issue is with the Dask cluster initialization, preventing any parallel tasks from executing.

## Key Findings

### 1. **Dask Cluster Startup Failure** (CRITICAL)

**Issue**: The `LocalCluster` initialization in `_generate_par()` fails immediately with a `RuntimeError`.

**Error**:
```
RuntimeError: An attempt has been made to start a new process before the
current process has finished its bootstrapping phase.
```

**Root Cause**: The Dask LocalCluster is being started in a non-main module context without proper multiprocessing guards. Python's multiprocessing requires the `if __name__ == '__main__':` idiom when using spawn or forkserver start methods.

**Location**: `data_sparsity/generate_data.py`, line 747-748:
```python
cluster = LocalCluster(n_workers=4, threads_per_worker=1, processes=True)
client = Client(cluster)
```

**Impact**: No parallel tasks can execute. The entire `_generate_par()` method fails immediately.

### 2. **Parallel Mode Triggering** (WORKING)

✓ The logic to trigger parallel mode works correctly:
- `NTASKS` is properly calculated when `num_obs > max_obs`
- Dimension splitting setup is correct
- Chunk size calculations work as expected

**Example**: With `num_obs=504` and `max_obs=250`, the system correctly computes:
```
NTASKS: 3
dim_split: 0
section_sizes: [24, 24, 23]
```

### 3. **RNG Strategy for Parallel** (POTENTIALLY CORRECT)

The RNG strategy appears sound in theory:
- **Global RNG**: Shared across all chunks for non-split dimensions (same seed)
- **Task RNG**: Unique per chunk for split dimension and observations (seed + chunk_id)

**Location**: `data_sparsity/utils/chunk_utils.py`, lines 48-63:
```python
def generate_rngs(seed: int, chunk_id: int):
    global_rng = np.random.default_rng(seed)
    task_rng = np.random.default_rng(seed + chunk_id + 1)
    return global_rng, task_rng
```

**Concern**: Cannot verify correctness without actual execution due to Dask failure.

### 4. **Multi-Variable Parallel Support** (IMPLEMENTED BUT UNTESTED)

The code includes support for multi-variable datasets in parallel mode:
- `MultiVarRecordGenerator.generate()` accepts optional `chunk_id`, `max_dim_size`, and `dim_split` parameters
- Observation counts are scaled per chunk: `chunk_fraction = shape[dim_split] / max_dim_size`
- Overlap control is maintained per-chunk with chunk-specific RNG seeds

**Location**: `data_sparsity/generators/multi_var_record_generator.py`, lines 140-159

**Status**: Implementation exists but cannot be validated due to Dask startup failure.

### 5. **Coordinate Generation** (POTENTIALLY CORRECT)

Coordinates are generated with dimension-specific ranges and RNGs:
- Non-split dimensions: Range [0, 1), global RNG → **same across all chunks**
- Split dimension: Normalized range per chunk, task RNG → **unique per chunk**

**Example for chunk 0 of 3**:
```python
dim_ranges = {
    0: (0/71, 24/71)  # First 24 elements of dimension 0
}
dim_rngs = {
    0: task_rng,      # Unique RNG for split dim
    1: global_rng     # Shared RNG for non-split dims
}
```

**Expected Behavior**: Coordinates along non-split dimensions should be identical across chunks (same global RNG). Coordinates along split dimension should be non-overlapping and cover the full range when combined.

**Status**: Logic appears correct but untested due to execution failure.

### 6. **File Output Structure** (INCONSISTENT)

**NetCDF**: Multiple files created (one per chunk)
- Format: `{filepath}_{chunk_id}.nc`
- Example: `test_parallel_0.nc`, `test_parallel_1.nc`, `test_parallel_2.nc`

**Parquet**: Single consolidated file intended
- Chunks written to temporary directory
- Dask DataFrame consolidation at end
- **Issue**: Consolidation step never executes due to Dask failure

**Observation**: The README claims "the number of output files generated might differ" between serial and parallel, which is true for NetCDF but should NOT be true for Parquet (single file intended).

## Comparison: Serial vs Parallel

### What Should Match (if working):

1. **Total observation count per variable** (allowing small variance due to rounding in chunk calculations)
2. **Coordinate indices occupied** (which grid points have data)
3. **Grid shape and dimensions**
4. **Variable structure and names**

### What Can Differ:

1. **Observation values** (due to different RNG streams for values vs positions)
2. **Coordinate values at occupied indices** (different RNG for split dimension)
3. **Number of output files** (multiple NetCDF files in parallel vs single in serial)

### Current Reality:

❌ **Nothing matches** because parallel execution never completes successfully.

## Test Status

### Existing Tests:

**File**: `tests/test_generate_data.py`

**Classes**:
- `TestParallelSingleVariable` (7 tests)
- `TestParallelMultiVariable` (5 tests)
- `TestParallelErrorHandling` (5 tests)

**Status**: 
- 15/15 parallel tests **FAILING**
- 13 fail with `FileExistsError` (leftover test files)
- 2 fail with `NameError: name 'shape' is not defined` (bug in test code)

**Root Cause**: Tests attempt to call `_generate_record_par()` directly, which:
1. Creates leftover files that aren't cleaned up
2. May expose bugs in the multi-variable parallel generation (undefined `shape` variable on line 916)

### Test Coverage Gaps:

1. No end-to-end parallel generation tests
2. No validation of coordinate consistency across chunks
3. No RNG determinism tests for parallel mode
4. No comparison tests between serial and parallel outputs

## Bugs Identified

### Bug #1: Dask Cluster Startup (CRITICAL)
**Severity**: CRITICAL - Blocks all parallel execution  
**Location**: `data_sparsity/generate_data.py:747-748`  
**Fix Required**: Add multiprocessing guard or switch to alternative parallelization

### Bug #2: Undefined Variable in Multi-Variable Parallel
**Severity**: MEDIUM - Affects multi-variable parallel generation  
**Location**: `data_sparsity/generate_data.py:916` (in `_generate_record_par`)  
**Issue**: Reference to undefined `shape` variable
**Code**:
```python
records, overlap_actual = MultiVarRecordGenerator.generate(
    shape, self.overlap_target, ...  # ← 'shape' not defined in this scope
)
```
**Should be**: `task_shape` (defined earlier in the function)

### Bug #3: generate() Method File Saving
**Severity**: LOW - Only affects file saving path  
**Location**: `data_sparsity/generate_data.py:684-704`  
**Issue**: Variable reassignment issue
```python
netcdf_filepath, _, _ = PathManager.setup_output_paths(...)
self.save_to_netcdf(self.netcdf_filepath, ...)  # ← references self attribute that doesn't exist
```
**Fix**: Should use local variable `netcdf_filepath`, not `self.netcdf_filepath`

### Bug #4: Test File Cleanup
**Severity**: LOW - Test hygiene issue  
**Location**: `tests/test_generate_data.py` (all parallel tests)  
**Issue**: Generated files aren't cleaned up, causing subsequent test failures

## Recommendations

### Immediate Fixes (Priority 1):

1. **Fix Dask Startup**: 
   - Option A: Add proper `if __name__ == '__main__':` guards
   - Option B: Switch to `concurrent.futures.ProcessPoolExecutor` (simpler, no external dependencies for parallelization logic)
   - Option C: Use `multiprocessing.Pool` directly

2. **Fix Bug #2**: Change `shape` to `task_shape` in `_generate_record_par`

3. **Fix Bug #3**: Use local variable instead of non-existent self attribute

### Testing (Priority 2):

1. Create end-to-end integration tests comparing serial vs parallel outputs
2. Add coordinate consistency validation
3. Test RNG determinism across runs
4. Add proper test cleanup

### Documentation (Priority 3):

1. Update `PARALLEL_GENERATION_STATUS.md` with current findings
2. Document expected differences between serial and parallel outputs
3. Add troubleshooting guide for parallel mode

## Conclusion

**Is the parallel workflow broken?** **YES**

**Does it generate the same output as serial?** **CANNOT DETERMINE** - execution fails before generating any output

**Main Issue**: Dask cluster initialization failure prevents any parallel execution

**Secondary Issues**:
- Bug in multi-variable parallel code path
- Bug in file saving logic
- Insufficient test coverage
- Test cleanup issues

**Path Forward**: Fix Dask startup issue (or replace with simpler parallelization), then validate output equivalence.

---

## Appendix: Test Execution Log

```
### Test 2: Attempting Parallel Generation ###
Input configuration:
  Number of observations: 500
  ...
Parallel generation: 3 tasks, 250 obs per task
  Dataset split along dimension 0
  Block sizes along it: [24, 24, 23]

Parallel Setup:
  NTASKS: 3
  max_dim_size: 71
  dim_split: 0
  section_sizes: [24, 24, 23]

✓ Parallel mode triggered with 3 tasks

Attempting parallel generation...
2025-12-02 12:21:53,652 - distributed.nanny - ERROR - Failed to start process
Traceback (most recent call last):
  ...
RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.
```

The parallel workflow setup is correct, but execution fails immediately at the Dask worker startup phase.
