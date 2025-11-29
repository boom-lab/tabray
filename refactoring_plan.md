# 
   Refactoring Plan for generate_data.py

   Based on my analysis of the 2325-line generate_data.py file, here's a comprehensive refactoring plan that prioritizes:

     - DRY (Don't Repeat Yourself) - eliminate code duplication
     - Testability - break down complex methods into testable units
     - Readability - clear, single-purpose methods

   -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

   Phase 1: Extract Validation Logic into Separate Classes

   Current Issue: _validate_parameters() is 206 lines with multiple validation concerns mixed together.

   Solution: Create validation classes in separate modules:

     - data_sparsity/validators/parameter_validator.py
       - ParameterValidator class with focused methods:
         - validate_num_obs(num_obs: int) -> None
         - validate_sparsity_type(sparsity) -> float  # Returns representative value
         - validate_num_dims(num_dims: int) -> None
         - validate_ratio_dims(ratio_dims, num_dims: int) -> np.ndarray
         - validate_seed(seed: int) -> None
         - validate_num_vars(num_vars: int) -> None
     - data_sparsity/validators/dimension_validator.py
       - DimensionValidator class:
         - compute_nb_coords_dim1(num_obs, sparsity, ratio_dims_prod, num_dims) -> int
         - compute_nb_coords_per_dim(ratio_dims, nb_coords_dim1) -> np.ndarray
         - validate_min_elements_per_dim(nb_coords_per_dim) -> None
         - validate_integer_elements(nb_coords_per_dim) -> np.ndarray  # Returns rounded
         - compute_shape_and_grid_points(nb_coords_per_dim) -> Tuple[List[int], int]
     - data_sparsity/validators/sparsity_validator.py
       - SparsityValidator class:
         - compute_min_sparsity(nb_coords_per_dim) -> float
         - validate_sparsity_bounds(sparsity, sparsity_min) -> float
         - validate_num_obs_consistency(num_obs, sparsity, shape) -> int  # Returns adjusted

   -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

   Phase 2: Extract Multi-Variable Configuration Classes

   Current Issue: Multi-variable setup scattered across 3 methods (76, 115, and 44 lines respectively).

   Solution: Create configuration classes:

     - data_sparsity/config/multi_var_sparsity.py
       - MultiVarSparsityConfig class:
         - from_scalar(sparsity, num_vars) -> np.ndarray
         - from_two_element_list(sparsity, num_vars, rng) -> np.ndarray
         - from_full_list(sparsity, num_vars) -> np.ndarray
         - validate_and_clip(var_sparsities, sparsity_min) -> np.ndarray
         - compute_var_num_obs(var_sparsities, num_obs) -> np.ndarray
     - data_sparsity/config/multi_var_dimensions.py
       - MultiVarDimensionsConfig class:
         - from_int(var_dims, num_vars, num_dims, seed) -> List[List[int]]
         - from_list(var_dims, num_vars, num_dims, seed) -> List[List[int]]
         - compute_constant_dims(var_dims_indices, num_dims) -> List[List[int]]
         - preselect_constant_coord_indices(var_constant_dims, num_dims, seed) -> dict
     - data_sparsity/config/multi_var_overlap.py
       - MultiVarOverlapConfig class:
         - validate_overlap_value(overlap) -> Union[float, str]
         - compute_min_overlap(var_dims_indices) -> float
         - validate_overlap_feasibility(overlap, min_overlap, num_vars) -> None

   -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

   Phase 3: Extract Coordinate Generation Logic

   Current Issue: Coordinate generation mixed with observation generation.

   Solution: Create dedicated coordinate module:

     - data_sparsity/generators/coordinate_generator.py
       - CoordinateGenerator class:
         - generate_dimension_coords(dim_size: int, rng) -> np.ndarray
         - generate_all_coords(shape: List[int], rng) -> List[np.ndarray]

   -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

   Phase 4: Split Observation Generation by Strategy

   Current Issue: Multiple large methods (193, 190, 149, 103, 99 lines) for record generation with overlapping logic.

   Solution: Create strategy pattern for generation:

     - data_sparsity/generators/observation_generator.py
       - ObservationGenerator (base class):
         - generate_observations(num_obs: int, rng) -> np.ndarray
     - data_sparsity/generators/record_generator.py
       - RecordGenerator (base class):
         - initialize_record(shape) -> np.ndarray
         - generate_flat_indices(total_points, num_obs, rng) -> np.ndarray
         - convert_to_multi_indices(flat_indices, shape) -> Tuple
         - assign_observations(record, multi_indices, observations) -> None
     - data_sparsity/generators/single_var_record_generator.py
       - SingleVarRecordGenerator (inherits RecordGenerator):
         - generate(shape, num_obs, observations, rng) -> np.ndarray
     - data_sparsity/generators/multi_var_record_generator.py
       - MultiVarRecordGenerator (inherits RecordGenerator):
         - generate_without_overlap(shape, records, var_config) -> dict
         - generate_with_overlap(shape, records, var_config, rng) -> dict
     - data_sparsity/generators/overlap_index_mapper.py
       - OverlapIndexMapper class:
         - map_indices_for_overlap(indices, source_dims, target_dims, shape) -> Tuple
         - Extract the 103-line method into smaller pieces:
           - identify_shared_dims(source_dims, target_dims) -> List[int]
           - identify_separate_dims(all_dims, shared_dims, other_dims) -> Tuple
           - build_coordinate_mapping(indices, shared_dims, sep_dims, shape) -> dict
           - reconstruct_target_indices(mapping, target_dims, num_points) -> Tuple
     - data_sparsity/generators/overlap_calculator.py
       - OverlapCalculator class:
         - compute_actual_overlap(records, shape, var_config) -> float
         - compute_pairwise_overlap(indices1, indices2) -> float

   -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

   Phase 5: Separate Parallel Generation Logic

   Current Issue: Parallel generation methods duplicate serial logic with slight modifications.

   Solution: Create parallel-specific modules:

     - data_sparsity/parallel/parallel_coordinator.py
       - ParallelCoordinator class:
         - setup_cluster(n_workers, threads_per_worker, memory_limit) -> Client
         - compute_chunk_distribution(num_obs, max_obs, n_tasks) -> List[Tuple]
         - submit_tasks(client, task_func, chunks) -> List[Future]
         - collect_results(futures) -> List
     - data_sparsity/parallel/chunk_generator.py
       - ChunkGenerator class (uses composition of single-process generators):
         - generate_chunk(chunk_id, obs_in_chunk, config) -> Tuple
         - Reuses validators and generators from previous phases

   -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

   Phase 6: Extract Output Format Creation

   Current Issue: DataArray, DataFrame, Dataset creation methods (82, 62, 84, 87 lines) repeat similar patterns.

   Solution: Create format-specific builders:

     - data_sparsity/output/netcdf_builder.py
       - NetCDFBuilder class:
         - build_dataarray(record, coordinates, var_name) -> xr.DataArray
         - build_dataset(records, coordinates, var_names) -> xr.Dataset
         - save_to_file(data, filepath, overwrite) -> None
     - data_sparsity/output/parquet_builder.py
       - ParquetBuilder class:
         - build_single_var_dataframe(record, coordinates) -> pd.DataFrame
         - build_multi_var_dataframe(records, coordinates) -> pd.DataFrame
         - extract_non_nan_points(record, coordinates) -> pd.DataFrame
         - save_to_file(dataframe, filepath, overwrite, chunk_id) -> None

   -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

   Phase 7: Refactor Main GenerateData Class

   Solution: Slim down GenerateData to be a coordinator that:

     - Simplified __init__: Store parameters, instantiate validators/configurators
     - Simplified _validate_parameters: Delegate to validator classes (5-10 lines)
     - Simplified generation methods: Use generator classes
     - Simplified generate: Orchestrate the workflow (20-30 lines)

   -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

   Phase 8: Create Comprehensive Test Suite

   After refactoring, create test files:

     - tests/validators/test_parameter_validator.py
     - tests/validators/test_dimension_validator.py
     - tests/validators/test_sparsity_validator.py
     - tests/config/test_multi_var_sparsity.py
     - tests/config/test_multi_var_dimensions.py
     - tests/config/test_multi_var_overlap.py
     - tests/generators/test_coordinate_generator.py
     - tests/generators/test_record_generator.py
     - tests/generators/test_overlap_index_mapper.py
     - tests/generators/test_overlap_calculator.py
     - tests/output/test_netcdf_builder.py
     - tests/output/test_parquet_builder.py
     - tests/test_generate_data_integration.py (end-to-end tests)

   -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

   Benefits of This Refactoring:

     - Testability: Each class has 3-10 small methods (10-30 lines each) that are easy to unit test
     - DRY: Eliminates duplication between serial and parallel paths, single/multi-var paths
     - Readability: Clear separation of concerns - validation, configuration, generation, output
     - Maintainability: Changes to one aspect (e.g., validation) don't affect generation logic
     - Extensibility: Easy to add new validation rules, generation strategies, or output formats

   Implementation Order:

     - Start with validators (Phase 1) - no dependencies
     - Then configurators (Phase 2) - depend on validators
     - Then generators (Phases 3-4) - depend on configurators
     - Then parallel (Phase 5) - depends on generators
     - Then outputs (Phase 6) - independent
     - Finally refactor main class (Phase 7) - uses all above
     - Add tests throughout (Phase 8)

   Would you like me to proceed with this plan? I can start with Phase 1 (validators) if you approve.


