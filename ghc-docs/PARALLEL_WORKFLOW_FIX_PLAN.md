# Parallel Workflow Fix and Validation Plan

**Date**: 2025-12-02  
**Status**: Planning Phase  
**Priority**: HIGH  
**Estimated Effort**: 3-5 days

## Problem Statement

The parallel data generation workflow in `data_sparsity` is currently non-functional due to a critical Dask cluster initialization failure. While the workflow logic appears sound, it cannot be validated until the execution framework is fixed.

## Goals

1. **Primary**: Make parallel generation execute successfully
2. **Secondary**: Validate that parallel output is equivalent to serial output (within expected tolerances)
3. **Tertiary**: Ensure both single-variable and multi-variable datasets work in parallel mode

## Current State Analysis

### What Works ✓
- Parallel mode detection and NTASKS calculation
- Dimension splitting logic
- Chunk size computation
- RNG strategy (in theory)
- Multi-variable data structure support

### What's Broken ✗
- **CRITICAL**: Dask LocalCluster fails to start workers
- **HIGH**: Multi-variable parallel has undefined variable bug (`shape` vs `task_shape`)
- **MEDIUM**: File saving in `generate()` method references non-existent attribute
- **LOW**: Test files not cleaned up, causing cascade failures

## Technical Requirements

### Functional Requirements

#### FR1: Parallel Execution Must Complete
- Parallel workflow must generate output files without errors
- Support 2+ chunks with configurable `max_obs`
- Work for both single-variable and multi-variable datasets

#### FR2: Output Equivalence (with Tolerances)
- **Must Match**:
  - Total observation count per variable (±1% tolerance for rounding)
  - Grid shape and coordinate dimensions
  - Coordinate indices that contain observations
  - Variable names and structure
  
- **May Differ**:
  - Observation values (different RNG stream)
  - Coordinate values along split dimension (different RNG)
  - Number of NetCDF files (multiple in parallel, single in serial)
  
- **Must Be Consistent**:
  - Coordinates along non-split dimensions (same global RNG)
  - Overlap percentage in multi-variable mode (±5% tolerance)

#### FR3: Determinism and Reproducibility
- Same seed + same parameters → identical outputs (within RNG differences)
- Parallel runs with same seed should produce same coordinate occupancy patterns
- Serial and parallel with same seed should have same total observation counts

### Non-Functional Requirements

#### NFR1: Performance
- Parallel execution should be faster than serial for `num_obs > 10M`
- Overhead should be <20% for `max_obs` chunks
- Memory usage should scale linearly with chunk size, not total size

#### NFR2: Backward Compatibility
- Serial workflow must remain unaffected
- Existing API must not change
- Existing serial tests must continue passing

#### NFR3: Maintainability
- Code should be testable without requiring Dask cluster
- Errors should be clear and actionable
- Parallel logic should be isolated from serial logic

## Proposed Solution

### Approach A: Fix Dask (Keep Current Design)

**Description**: Add proper multiprocessing guards to enable Dask LocalCluster startup.

**Changes Required**:
1. Wrap `_generate_par()` call with `if __name__ == '__main__':` guard
2. Move cluster initialization to a separate function that can be guarded
3. Add `freeze_support()` for Windows compatibility

**Pros**:
- Minimal code changes
- Dask provides robust distributed computing features
- Dashboard useful for debugging

**Cons**:
- Adds complexity (Dask dependency)
- Harder to test and debug
- Serialization issues with `self` reference in `_generate_record_par`
- Requires running from main module or special setup

**Estimated Effort**: 2 days (including testing)

### Approach B: Replace with ProcessPoolExecutor (RECOMMENDED)

**Description**: Replace Dask with `concurrent.futures.ProcessPoolExecutor` for simpler, more robust parallelization.

**Changes Required**:
1. Replace `LocalCluster`/`Client` with `ProcessPoolExecutor`
2. Make `_generate_record_par` a standalone function (not method)
3. Pass all necessary parameters explicitly (no `self` reference)
4. Simplify parquet consolidation (use pandas directly)

**Pros**:
- Standard library (no external dependency for core parallelization)
- Simpler to understand and debug
- Easier to test (can use `max_workers=1` for deterministic testing)
- Works in any context (no `if __name__` required for simple cases)
- No serialization issues with self

**Cons**:
- No built-in dashboard
- Less sophisticated than Dask for truly distributed workloads
- Manual load balancing

**Estimated Effort**: 3 days (more refactoring but simpler result)

### Approach C: Multiprocessing.Pool (Alternative)

**Description**: Use `multiprocessing.Pool` directly.

**Changes Required**:
- Similar to Approach B but with `multiprocessing.Pool`
- Requires same refactoring to standalone function

**Pros**:
- Standard library
- Well-established and stable
- Good for CPU-bound tasks

**Cons**:
- Less modern API than `concurrent.futures`
- Similar complexity to Approach B

**Estimated Effort**: 3 days

## Recommended Approach: B (ProcessPoolExecutor)

**Rationale**:
1. **Simplicity**: Standard library, modern API, clear semantics
2. **Testability**: Can run with `max_workers=1` for deterministic tests
3. **Maintainability**: No external dependencies for core functionality
4. **Robustness**: Avoids Dask serialization and startup issues

## Implementation Plan

### Phase 1: Critical Bug Fixes (1 day)

**Priority**: CRITICAL  
**Goal**: Fix bugs preventing execution

#### Task 1.1: Fix Multi-Variable Parallel Bug
**File**: `data_sparsity/generate_data.py`  
**Line**: ~916  
**Change**:
```python
# Before:
records, overlap_actual = MultiVarRecordGenerator.generate(
    shape, self.overlap_target, ...
)

# After:
records, overlap_actual = MultiVarRecordGenerator.generate(
    task_shape, self.overlap_target, ...
)
```

#### Task 1.2: Fix File Saving Bug
**File**: `data_sparsity/generate_data.py`  
**Lines**: 684-704  
**Change**:
```python
# Before:
netcdf_filepath, _, _ = PathManager.setup_output_paths(...)
self.save_to_netcdf(self.netcdf_filepath, ...)

# After:
nc_path, pq_path, pq_tmp = PathManager.setup_output_paths(...)
self.save_to_netcdf(nc_path, ...)
```

### Phase 2: Replace Dask with ProcessPoolExecutor (2 days)

**Priority**: HIGH  
**Goal**: Enable parallel execution

#### Task 2.1: Create Standalone Worker Function
**New File**: `data_sparsity/workers/parallel_worker.py`

```python
"""Standalone worker function for parallel generation."""

def generate_chunk(
    chunk_id: int,
    obs_in_chunk: int,
    # All generation parameters as explicit arguments
    seed: int,
    shape: tuple,
    sparsity: float,
    num_vars: int,
    var_sparsities: np.ndarray,
    var_num_obs: np.ndarray,
    var_dims_indices: list,
    var_constant_dims: list,
    var_constant_coord_indices: dict,
    overlap_target: Union[float, str],
    # Parallel-specific parameters
    dim_split: int,
    max_dim_size: int,
    div_points: list,
    section_sizes: list,
    # Output paths
    netcdf_filepath: str,
    parquet_tmp: str,
) -> Tuple[int, int, str]:
    """Generate a single chunk of data in parallel.
    
    Returns:
        (chunk_id, total_observations, parquet_chunk_path)
    """
    # Implementation moved from _generate_record_par
    # No reference to self - all parameters passed explicitly
    pass
```

**Benefits**:
- No serialization of self
- Testable in isolation
- Clear interface

#### Task 2.2: Refactor _generate_par() Method
**File**: `data_sparsity/generate_data.py`

```python
def _generate_par(self) -> None:
    """Generate data using parallel processing."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from data_sparsity.workers.parallel_worker import generate_chunk
    
    # Setup (existing code)
    ...
    
    # Prepare arguments for all chunks
    chunk_args = []
    for chunk_id, chunk_obs in zip(range(self.NTASKS), per_chunk_obs):
        args = {
            'chunk_id': chunk_id,
            'obs_in_chunk': chunk_obs,
            'seed': self.seed,
            'shape': self.shape,
            # ... all other parameters
        }
        chunk_args.append(args)
    
    # Execute in parallel
    with ProcessPoolExecutor(max_workers=min(self.NTASKS, 4)) as executor:
        futures = [executor.submit(generate_chunk, **args) for args in chunk_args]
        
        for future in as_completed(futures):
            chunk_id, obs_num, chunk_path = future.result()
            print(f"Completed chunk {chunk_id} with {obs_num} observations")
    
    # Consolidate parquet files
    self._consolidate_parquet_files()
```

#### Task 2.3: Implement Parquet Consolidation
**File**: `data_sparsity/generate_data.py`

```python
def _consolidate_parquet_files(self) -> None:
    """Consolidate temporary parquet files into single output."""
    import pandas as pd
    import glob
    
    tmp_pattern = os.path.join(os.path.dirname(self.parquet_tmp), "*.parquet")
    tmp_files = sorted(glob.glob(tmp_pattern))
    
    if not tmp_files:
        raise RuntimeError("No temporary parquet files found")
    
    # Read and concatenate all chunks
    dfs = [pd.read_parquet(f) for f in tmp_files]
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Write consolidated file
    df_combined.to_parquet(self.parquet_filepath)
    
    # Cleanup temporary files
    for f in tmp_files:
        os.remove(f)
```

### Phase 3: Validation and Testing (1-2 days)

**Priority**: HIGH  
**Goal**: Verify output equivalence

#### Task 3.1: Create Comparison Test Suite
**New File**: `tests/test_parallel_equivalence.py`

```python
"""Tests to verify parallel output matches serial output."""

class TestParallelSerialEquivalence:
    """Compare parallel and serial generation outputs."""
    
    def test_single_var_observation_count(self):
        """Total observations should match between serial and parallel."""
        pass
    
    def test_single_var_grid_occupancy(self):
        """Same grid cells should be occupied."""
        pass
    
    def test_single_var_coordinate_consistency(self):
        """Non-split dimension coordinates should be identical."""
        pass
    
    def test_multi_var_observation_counts(self):
        """Each variable should have same observation count."""
        pass
    
    def test_multi_var_overlap_preserved(self):
        """Overlap percentage should be maintained."""
        pass
    
    def test_determinism_across_runs(self):
        """Same seed should produce same results."""
        pass
```

#### Task 3.2: Fix Existing Parallel Tests
**File**: `tests/test_generate_data.py`

- Add proper cleanup (fixtures with teardown)
- Remove direct calls to `_generate_record_par` (test through `generate()`)
- Add assertions for output validation
- Fix undefined variable issues

#### Task 3.3: Add Integration Tests
**New File**: `tests/integration/test_end_to_end_parallel.py`

```python
"""End-to-end tests for parallel generation workflow."""

def test_small_parallel_dataset():
    """Generate complete dataset with 2 chunks."""
    pass

def test_multi_var_parallel_dataset():
    """Generate multi-variable dataset with 3 chunks."""
    pass

def test_very_small_chunks():
    """Handle edge case of many small chunks."""
    pass
```

### Phase 4: Documentation and Cleanup (0.5 days)

**Priority**: MEDIUM  
**Goal**: Update docs and finalize

#### Task 4.1: Update Documentation
- Update `README.md` with corrected parallel usage
- Update `PARALLEL_GENERATION_STATUS.md` with current status
- Add troubleshooting section

#### Task 4.2: Add Inline Documentation
- Document RNG strategy in detail
- Explain coordinate generation per chunk
- Clarify output format differences

#### Task 4.3: Code Cleanup
- Remove unused Dask imports
- Remove obsolete code paths
- Add type hints to new functions

## Validation Criteria

### Success Criteria (Must Pass)

1. ✓ Parallel execution completes without errors for single-variable datasets
2. ✓ Parallel execution completes without errors for multi-variable datasets
3. ✓ Total observation count matches serial (±1%)
4. ✓ Grid occupancy patterns match serial
5. ✓ Coordinates on non-split dimensions match serial exactly
6. ✓ All existing serial tests still pass
7. ✓ New parallel tests pass consistently

### Performance Criteria (Should Pass)

1. ✓ Parallel is faster than serial for num_obs > 10M
2. ✓ Memory usage scales with chunk size
3. ✓ Overhead < 20% for small datasets

## Risk Assessment

### High Risks

**Risk**: Refactoring breaks serial workflow  
**Mitigation**: Run full test suite after each change; no changes to serial code paths

**Risk**: Output equivalence cannot be achieved  
**Mitigation**: Document expected differences clearly; adjust tolerance ranges

**Risk**: New bugs introduced during refactoring  
**Mitigation**: Comprehensive testing; code review; phased rollout

### Medium Risks

**Risk**: Performance degradation  
**Mitigation**: Benchmark before and after; optimize chunk size

**Risk**: Edge cases not covered  
**Mitigation**: Add edge case tests; handle errors gracefully

### Low Risks

**Risk**: Documentation becomes outdated  
**Mitigation**: Update docs as part of each task

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Bug Fixes | 1 day | None |
| Phase 2: Replace Dask | 2 days | Phase 1 complete |
| Phase 3: Testing | 1-2 days | Phase 2 complete |
| Phase 4: Docs | 0.5 days | Phase 3 complete |
| **Total** | **4.5-5.5 days** | |

## Next Steps

1. Review and approve this plan
2. Create feature branch: `fix/parallel-workflow`
3. Execute Phase 1 (critical bugs)
4. Execute Phase 2 (ProcessPoolExecutor refactor)
5. Execute Phase 3 (validation)
6. Execute Phase 4 (documentation)
7. Merge to main after all tests pass

## Open Questions

1. **Q**: Should we keep Dask as an optional dependency for users who want it?  
   **A**: TBD - requires discussion

2. **Q**: What should be the default `max_obs` value?  
   **A**: Current is 10M, seems reasonable; may adjust based on testing

3. **Q**: Should parallel mode be opt-in or automatic based on `num_obs`?  
   **A**: Current automatic behavior is good; keep it

4. **Q**: How to handle overlap in parallel mode for multi-variable?  
   **A**: Current per-chunk approach seems correct; validate in testing

## Related Documents

- `PARALLEL_GENERATION_STATUS.md` - Current status (outdated)
- `PARALLEL_WORKFLOW_ANALYSIS.md` - Detailed analysis of broken state
- `README.md` - User-facing documentation
- `tests/test_generate_data.py` - Existing tests

---

**Author**: Planning Specialist Agent  
**Last Updated**: 2025-12-02  
**Status**: DRAFT - Awaiting Review
