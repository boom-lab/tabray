# Parallel Generation Status

## Current State

The parallel generation feature (`_generate_par()` method) has been partially restored but is **not fully functional** and requires additional work.

## Known Issues

1. **Dask Cluster Startup**: The LocalCluster fails to start workers with `RuntimeError: Nanny failed to start worker process`
2. **Parquet Corruption**: Reading back parquet files created during parallel generation results in `OSError: Couldn't deserialize thrift`
3. **Missing Attributes**: The `max_dim_size` and `section_sizes` attributes need to be properly initialized

## What Works

- Directory creation for output files
- Basic parallel task distribution logic
- NetCDF and Parquet file naming with chunk IDs

## What Needs Work

1. **Fix Dask Setup**: Investigate why LocalCluster workers fail to start
   - May need different process/thread configuration
   - Could be related to serialization of self reference

2. **Fix Parquet Writing**: The parquet files being written in parallel are corrupted
   - May need to avoid dask.dataframe conversion
   - Consider using pandas directly for chunk writes

3. **Refactor Worker Function**: The `_generate_record_par()` function references `self` but is called in a distributed context
   - Should be a standalone function or static method
   - Parameters should be explicitly passed rather than accessing instance attributes

4. **Add Tests**: Parallel generation has no automated tests
   - Need integration tests for parallel mode
   - Need tests for chunk merging logic

## Workaround

For now, users should:
- Set `max_obs` to a large value (or omit it) to ensure `NTASKS == 1`
- This will use the serial generation path (`_generate_serial()`) which is fully functional

## Recommendation

The parallel generation feature should be:
1. Marked as experimental in documentation
2. Refactored to use a cleaner multiprocessing approach
3. Properly tested before being considered production-ready

## Next Steps

To fix parallel generation:

1. Refactor `_generate_record_par()` to be a standalone function
2. Use `concurrent.futures` or `multiprocessing.Pool` instead of Dask
3. Write chunks directly with pandas (not dask.dataframe)
4. Add comprehensive tests for parallel mode
5. Document performance characteristics and when to use parallel vs serial

---

*Last Updated: 2025-11-29*
