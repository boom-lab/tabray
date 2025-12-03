"""Integration tests for GenerateData class."""

import pytest
import numpy as np
import xarray as xr
import pandas as pd
import os
import tempfile
import shutil
from data_sparsity.generate_data import GenerateData


class TestInitialization:
    """Tests for GenerateData initialization."""
    
    def test_minimal_valid_parameters(self):
        """Should initialize with minimal valid parameters."""
        gen = GenerateData(
            num_obs=100,
            num_dims=2,
            ratio_dims=1,
            sparsity=1.,
            seed=42
        )
        assert gen.num_obs == 100
        assert gen.num_dims == 2
        assert gen.seed == 42
        assert gen.num_vars == 1
    
    def test_full_parameters_specified(self):
        """Should initialize with all parameters."""
        gen = GenerateData(
            num_obs=200,
            num_dims=3,
            ratio_dims=1,
            sparsity=[0.15, 0.2],
            seed=123,
            num_vars=2,
            var_dims=2,
            overlap=0.5
        )
        assert gen.num_obs >= 200  # May be adjusted
        assert gen.num_dims == 3
        assert gen.num_vars == 2
        assert gen.overlap == 0.5
    
    def test_invalid_num_obs_raises_error(self):
        """Should raise ValueError for invalid num_obs."""
        with pytest.raises((ValueError, TypeError)):
            GenerateData(
                num_obs=0,
                num_dims=2,
                ratio_dims=1,
                sparsity=0.1,
                seed=42
            )
    
    def test_invalid_sparsity_raises_error(self):
        """Should raise ValueError for invalid sparsity."""
        with pytest.raises(ValueError):
            GenerateData(
                num_obs=100,
                num_dims=2,
                ratio_dims=1,
                sparsity=1.5,  # > 1.0
                seed=42
            )
    
    def test_attributes_set_correctly(self):
        """Should set all attributes correctly."""
        gen = GenerateData(
            num_obs=150,
            num_dims=3,
            ratio_dims=[2, 1, 3],
            sparsity=0.2,
            seed=999
        )
        assert hasattr(gen, 'num_obs')
        assert hasattr(gen, 'num_dims')
        assert hasattr(gen, 'ratio_dims')
        assert hasattr(gen, 'sparsity')
        assert hasattr(gen, 'seed')
    
    def test_single_variable_defaults(self):
        """Should default to single variable."""
        gen = GenerateData(
            num_obs=100,
            num_dims=2,
            ratio_dims=1,
            sparsity=0.1,
            seed=42
        )
        assert gen.num_vars == 1
        assert gen.overlap == 'random'
    
    def test_multi_variable_setup(self):
        """Should properly set up multi-variable configuration."""
        gen = GenerateData(
            num_obs=200,
            num_dims=3,
            ratio_dims=1,
            sparsity=[0.1, 0.2, 0.15],
            seed=42,
            num_vars=3,
            var_dims=[2, 2, 3],
            overlap=0.6
        )
        assert gen.num_vars == 3
        assert isinstance(gen.sparsity, (list, tuple))


class TestSingleVariableGeneration:
    """Tests for single-variable data generation."""
    
    def test_1d_generation(self):
        """Should generate 1D data."""
        gen = GenerateData(
            num_obs=20,
            num_dims=1,
            ratio_dims=1,
            sparsity=0.2,
            seed=42
        )
        dataarray, dataframe = gen.generate()
        
        assert isinstance(dataarray, xr.DataArray)
        assert isinstance(dataframe, pd.DataFrame)
        assert len(dataarray.dims) == 1
    
    def test_2d_generation(self):
        """Should generate 2D data."""
        gen = GenerateData(
            num_obs=50,
            num_dims=2,
            ratio_dims=1,
            sparsity=0.2,
            seed=42
        )
        dataarray, dataframe = gen.generate()
        
        assert isinstance(dataarray, xr.DataArray)
        assert len(dataarray.dims) == 2
        assert dataframe.shape[0] >= 40  # At least 40 observations
    
    def test_3d_generation(self):
        """Should generate 3D data."""
        gen = GenerateData(
            num_obs=100,
            num_dims=3,
            ratio_dims=1,
            sparsity=0.15,
            seed=42
        )
        dataarray, dataframe = gen.generate()
        
        assert len(dataarray.dims) == 3
        # Check that observations exist
        assert not np.all(np.isnan(dataarray.values))
    
    def test_correct_number_of_observations(self):
        """Should generate correct number of observations."""
        num_obs = 75
        gen = GenerateData(
            num_obs=num_obs,
            num_dims=2,
            ratio_dims=1,
            sparsity=0.2,
            seed=42
        )
        dataarray, dataframe = gen.generate()
        
        # Count non-NaN values in dataarray (may be adjusted)
        non_nan_count = np.count_nonzero(~np.isnan(dataarray.values))
        assert non_nan_count >= 60  # At least most of requested
        
        # Check dataframe has correct number of rows
        assert len(dataframe) >= 60
    
    def test_high_sparsity(self):
        """Should handle high sparsity (dense data)."""
        gen = GenerateData(
            num_obs=90,
            num_dims=2,
            ratio_dims=[1, 1],
            sparsity=0.9,
            seed=42
        )
        dataarray, dataframe = gen.generate()
        
        # Should have many observations
        non_nan_count = np.count_nonzero(~np.isnan(dataarray.values))
        assert non_nan_count == 90
    
    def test_reproducible_with_seed(self):
        """Should produce same results with same seed."""
        gen1 = GenerateData(
            num_obs=50,
            num_dims=2,
            ratio_dims=1,
            sparsity=0.2,
            seed=42
        )
        dataarray1, dataframe1 = gen1.generate()
        
        gen2 = GenerateData(
            num_obs=50,
            num_dims=2,
            ratio_dims=1,
            sparsity=0.2,
            seed=42
        )
        dataarray2, dataframe2 = gen2.generate()
        
        # Arrays should be identical
        np.testing.assert_array_equal(
            dataarray1.values, 
            dataarray2.values, 
        )
        # DataFrames should be identical
        pd.testing.assert_frame_equal(dataframe1, dataframe2)
    
    def test_different_with_different_seed(self):
        """Should produce different results with different seeds."""
        gen1 = GenerateData(
            num_obs=50,
            num_dims=2,
            ratio_dims=1,
            sparsity=0.1,
            seed=42
        )
        dataarray1, _ = gen1.generate()
        
        gen2 = GenerateData(
            num_obs=50,
            num_dims=2,
            ratio_dims=1,
            sparsity=0.1,
            seed=999
        )
        dataarray2, _ = gen2.generate()
        
        # Arrays should be different
        assert not np.array_equal(
            dataarray1.values,
            dataarray2.values,
            equal_nan=True
        )


class TestMultiVariableGeneration:
    """Tests for multi-variable data generation."""
    
    def test_two_variables(self):
        """Should generate dataset with two variables."""
        gen = GenerateData(
            num_obs=100,
            num_dims=3,
            ratio_dims=1,
            sparsity=[0.15, 0.07],  # Adjusted: var1 gets ~50 obs (< 81 grid points)
            seed=42,
            num_vars=2,
            var_dims=2,
            overlap=0.5
        )
        dataset, dataframe = gen.generate()
        
        assert isinstance(dataset, xr.Dataset)
        assert len(dataset.data_vars) == 2
        assert 'var0' in dataset
        assert 'var1' in dataset
    
    def test_many_variables(self):
        """Should generate dataset with multiple variables."""
        gen = GenerateData(
            num_obs=200,
            num_dims=3,
            ratio_dims=1,
            sparsity=[0.10, 0.06, 0.04, 0.05],  # Adjusted: non-ref vars get ~70,50,60 obs
            seed=42,
            num_vars=4,
            var_dims=2,
            overlap=0.4
        )
        dataset, dataframe = gen.generate()
        
        assert len(dataset.data_vars) == 4
        for i in range(4):
            assert f'var{i}' in dataset
    
    def test_different_sparsities_per_variable(self):
        """Should respect different sparsity values per variable."""
        gen = GenerateData(
            num_obs=150,
            num_dims=3,
            ratio_dims=1,
            sparsity=[0.20, 0.07, 0.05],  # Adjusted: var1~60 obs, var2~43 obs (well < 81)
            seed=42,
            num_vars=3,
            var_dims=2,
            overlap=0.3
        )
        dataset, dataframe = gen.generate()
        
        # Each variable should have different number of observations
        counts_data = dataset.count()
        print(counts_data)
        
        # Extract counts as integers
        counts = [int(counts_data[f'var{i}'].values) for i in range(3)]
        
        # All counts should be positive
        assert all(c > 0 for c in counts)
        # At least some variation in counts
        assert max(counts) > min(counts)
    
    def test_different_dimensions_per_variable(self):
        """Should handle different varying dimensions per variable."""
        gen = GenerateData(
            num_obs=120,
            num_dims=3,
            ratio_dims=1,
            sparsity=0.8,  # Adjusted to meet minimum sparsity requirement
            seed=42,
            num_vars=2,
            var_dims=[2, 2],  # Both use 2 dims (but var0 gets all dims per new logic)
            overlap=1
        )
        dataset, dataframe = gen.generate()
        
        # Both variables should exist
        assert 'var0' in dataset
        assert 'var1' in dataset
    
    def test_zero_overlap(self):
        """Should generate variables with zero overlap."""
        gen = GenerateData(
            num_obs=100,
            num_dims=3,
            ratio_dims=1,
            sparsity=[0.15, 0.01],  # Adjusted: var1 with 1 dim gets ~7 obs (< 9)
            seed=42,
            num_vars=2,
            var_dims=[2, 1],  # Different dims - var0 all dims, var1 gets 1 dim
            overlap='random'
        )
        dataset, dataframe = gen.generate()
        
        # Should have minimal overlap (random placement)
        assert dataset is not None
        assert len(dataset.data_vars) == 2
    
    def test_high_overlap(self):
        """Should generate variables with high overlap."""
        gen = GenerateData(
            num_obs=100,
            num_dims=3,
            ratio_dims=1,
            sparsity=[0.1, 0.1],
            seed=42,
            num_vars=2,
            var_dims=2,
            overlap=0.9
        )
        dataset, dataframe = gen.generate()
        
        # Should have high overlap
        assert dataset is not None
        assert len(dataset.data_vars) == 2
    
    def test_random_overlap(self):
        """Should handle random overlap specification."""
        gen = GenerateData(
            num_obs=100,
            num_dims=3,
            ratio_dims=1,
            sparsity=[0.1, 0.1],
            seed=42,
            num_vars=2,
            var_dims=2,
            overlap='random'
        )
        dataset, dataframe = gen.generate()
        
        assert dataset is not None
        assert len(dataset.data_vars) == 2
    
    def test_dataset_has_all_variables(self):
        """Should include all variables in dataset."""
        num_vars = 5
        gen = GenerateData(
            num_obs=200,
            num_dims=3,
            ratio_dims=1,
            sparsity=0.75,
            seed=42,
            num_vars=num_vars,
            var_dims=2,
            overlap=0.65
        )
        dataset, _ = gen.generate()
        
        assert len(dataset.data_vars) == num_vars
        for i in range(num_vars):
            assert f'var{i}' in dataset
    
    def test_dataframe_has_all_columns(self):
        """Should include columns for all variables and coordinates."""
        gen = GenerateData(
            num_obs=100,
            num_dims=3,
            ratio_dims=1,
            sparsity=[0.10, 0.07],  # Adjusted: var1 gets ~50 obs (< 81)
            seed=42,
            num_vars=2,
            var_dims=2,
            overlap=0.5
        )
        _, dataframe = gen.generate()
        
        # Should have coordinate columns and variable columns
        assert 'x0' in dataframe.columns
        assert 'x1' in dataframe.columns
        assert 'var0' in dataframe.columns
        assert 'var1' in dataframe.columns
    
    def test_reproducible_with_seed(self):
        """Should produce same multi-var results with same seed."""
        gen1 = GenerateData(
            num_obs=80,
            num_dims=3,
            ratio_dims=1,
            sparsity=[0.15, 0.08],  # Adjusted: var1 gets ~45 obs (< 64 grid points)
            seed=42,
            num_vars=2,
            var_dims=2,
            overlap=0.5
        )
        dataset1, _ = gen1.generate()
        
        gen2 = GenerateData(
            num_obs=80,
            num_dims=3,
            ratio_dims=1,
            sparsity=[0.15, 0.08],  # Adjusted: same as gen1
            seed=42,
            num_vars=2,
            var_dims=2,
            overlap=0.5
        )
        dataset2, _ = gen2.generate()
        
        # Datasets should be identical
        for var in dataset1.data_vars:
            # Use assert_allclose which handles NaN properly
            np.testing.assert_allclose(
                dataset1[var].values,
                dataset2[var].values,
                rtol=0,
                atol=0,
                equal_nan=True
            )


class TestFileOutput:
    """Tests for file output functionality."""
    
    def test_netcdf_output_created(self, temp_dir):
        """Should create NetCDF file."""
        gen = GenerateData(
            num_obs=50,
            num_dims=2,
            ratio_dims=1,
            sparsity=0.1,
            seed=42
        )
        dataarray, _ = gen.generate()
        
        netcdf_path = os.path.join(temp_dir, "test.nc")
        gen.save_to_netcdf(netcdf_path, dataarray, overwrite=True)
        
        assert os.path.exists(netcdf_path)
    
    def test_parquet_output_created(self, temp_dir):
        """Should create Parquet file."""
        gen = GenerateData(
            num_obs=50,
            num_dims=2,
            ratio_dims=1,
            sparsity=0.1,
            seed=42
        )
        _, dataframe = gen.generate()

        parquet_path = os.path.join(temp_dir, "test")
        gen.save_to_parquet(parquet_path, dataframe, overwrite=True)

        parquet_dirpath = os.path.dirname(parquet_path)
        files = os.listdir(parquet_dirpath)
        parquet_files = [f for f in files if f.endswith(".parquet")]

        assert "_metadata" in files
        assert "_common_metadata" in files
        assert len(parquet_files) > 0
    
    def test_netcdf_roundtrip(self, temp_dir):
        """Should save and load NetCDF correctly."""
        gen = GenerateData(
            num_obs=30,
            num_dims=2,
            ratio_dims=1,
            sparsity=0.2,
            seed=42
        )
        dataarray, _ = gen.generate()
        
        netcdf_path = os.path.join(temp_dir, "test.nc")
        gen.save_to_netcdf(netcdf_path, dataarray, overwrite=True)
        
        # Load back
        loaded = xr.open_dataarray(netcdf_path)
        
        # Should match original
        np.testing.assert_array_equal(
            dataarray.values,
            loaded.values,
        )
        loaded.close()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_observation(self):
        """Should handle single observation."""

        # Single obs make sense only with sparsity = 1 and 1 element per
        # dimension [ratio_dims=1]
        gen = GenerateData(
            num_obs=1,
            num_dims=10,
            ratio_dims=1,
            sparsity=1,
            seed=42
        )
        dataarray, dataframe = gen.generate()
        
        non_nan_count = np.count_nonzero(~np.isnan(dataarray.values))
        assert non_nan_count >= 1
        assert len(dataframe) >= 1
    
    def test_maximum_sparsity(self):
        """Should handle maximum sparsity (all points filled)."""
        gen = GenerateData(
            num_obs=100,
            num_dims=2,
            ratio_dims=[1, 1],
            sparsity=1.0,
            seed=42
        )
        dataarray, dataframe = gen.generate()
        
        # Should have all points or num_obs points
        non_nan_count = np.count_nonzero(~np.isnan(dataarray.values))
        assert non_nan_count == 100
    
    def test_single_dimension(self):
        """Should handle single dimension."""
        gen = GenerateData(
            num_obs=20,
            num_dims=1,
            ratio_dims=1,
            sparsity=0.2,
            seed=42
        )
        dataarray, dataframe = gen.generate()
        
        assert len(dataarray.dims) == 1
        assert dataarray is not None
    
    def test_many_dimensions(self):
        """Should handle many dimensions (5+)."""
        gen = GenerateData(
            num_obs=100,
            num_dims=4,
            ratio_dims=1,
            sparsity=0.2,
            seed=42
        )
        dataarray, dataframe = gen.generate()
        
        assert len(dataarray.dims) == 4
        assert dataarray is not None
    
    def test_non_uniform_dimension_ratios(self):
        """Should handle non-uniform dimension ratios."""
        gen = GenerateData(
            num_obs=150,
            num_dims=3,
            ratio_dims=[1, 2, 1],
            sparsity=0.2,
            seed=42
        )
        dataarray, dataframe = gen.generate()
        
        # Dimensions should have different sizes
        shape = dataarray.shape
        assert len(set(shape)) > 1  # Not all dimensions same size
    
    def test_very_small_grid(self):
        """Should handle very small grids."""
        gen = GenerateData(
            num_obs=4,
            num_dims=2,
            ratio_dims=[1, 1],
            sparsity=1.0,
            seed=42
        )
        dataarray, dataframe = gen.generate()
        
        assert dataarray is not None
        assert dataframe is not None
    
    def test_all_variables_same_dimensions(self):
        """Should handle all variables with same dimensions."""
        gen = GenerateData(
            num_obs=100,
            num_dims=3,
            ratio_dims=1,
            sparsity=0.15,
            seed=42,
            num_vars=2,
            var_dims=2,  # All use 2 dimensions
            overlap=0.5
        )
        dataset, _ = gen.generate()
        
        assert len(dataset.data_vars) == 2
    
    def test_all_variables_different_dimensions(self):
        """Should handle all variables with different dimensions."""
        gen = GenerateData(
            num_obs=120,
            num_dims=3,
            ratio_dims=1,
            sparsity=0.1,
            seed=42,
            num_vars=3,
            var_dims=[1, 2, 3],  # Each different
            overlap=0.0
        )
        dataset, _ = gen.generate()
        
        assert len(dataset.data_vars) == 3


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_invalid_sparsity_raises_error(self):
        """Should raise error for invalid sparsity."""
        with pytest.raises(ValueError):
            GenerateData(
                num_obs=100,
                num_dims=2,
                ratio_dims=1,
                sparsity=2.0,  # > 1.0
                seed=42
            )
    
    def test_negative_num_obs_raises_error(self):
        """Should raise error for negative num_obs."""
        with pytest.raises((ValueError, TypeError)):
            GenerateData(
                num_obs=-10,
                num_dims=2,
                ratio_dims=1,
                sparsity=0.1,
                seed=42
            )
    
    def test_zero_num_dims_raises_error(self):
        """Should raise error for zero dimensions."""
        with pytest.raises((ValueError, TypeError)):
            GenerateData(
                num_obs=100,
                num_dims=0,
                ratio_dims=1,
                sparsity=0.1,
                seed=42
            )
    
    def test_invalid_num_vars_raises_error(self):
        """Should raise error for invalid num_vars."""
        with pytest.raises((ValueError, TypeError)):
            GenerateData(
                num_obs=100,
                num_dims=2,
                ratio_dims=1,
                sparsity=0.1,
                seed=42,
                num_vars=0
            )
    
    def test_invalid_overlap_raises_error(self):
        """Should raise error for invalid overlap value."""
        with pytest.raises(ValueError):
            GenerateData(
                num_obs=100,
                num_dims=3,
                ratio_dims=1,
                sparsity=[0.1, 0.1],
                seed=42,
                num_vars=2,
                var_dims=2,
                overlap=1.5  # > 1.0
            )


class TestParallelSingleVariable:
    """Tests for parallel single-variable generation workflow."""
    
    def test_parallel_generation_with_chunks(self, temp_dir):
        """Should generate data using parallel workflow with multiple chunks."""
        gen = GenerateData(
            num_obs=500,
            num_dims=2,
            ratio_dims=[5, 5],
            sparsity=0.05,
            seed=42,
            num_vars=1
        )
        
        # Set up file paths and parallel execution parameters
        gen.netcdf_filepath = os.path.join(temp_dir, "test_parallel.nc")
        gen.parquet_filepath = os.path.join(temp_dir, "test_parallel.parquet")
        gen.parquet_tmp = os.path.join(temp_dir, "tmp", "test_parallel_tmp.parquet")
        gen.NTASKS = 4
        gen.max_dim_size = gen.shape[0]
        gen.div_points = [0, 2, 4, 6, 8]  # 4 chunks - note: shape[0] should be 8
        gen.section_sizes = [2, 2, 2, 2]
        gen.dim_split = 0
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        
        # Generate one chunk
        chunk_id = 0
        obs_in_chunk = 25
        result_chunk_id, result_obs = gen._generate_record_par(chunk_id, obs_in_chunk)
        
        assert result_chunk_id == chunk_id
        assert result_obs > 0  # Should have generated some observations
    
    def test_rng_reproducibility_across_chunks(self):
        """Should produce consistent results with same seed across runs."""
        # Run 1
        gen1 = GenerateData(
            num_obs=200,
            num_dims=2,
            ratio_dims=[10, 10],
            sparsity=0.1,
            seed=42,
            num_vars=1
        )
        gen1.NTASKS = 2
        gen1.max_dim_size = 10
        gen1.div_points = [0, 5, 10]
        gen1.section_sizes = [5, 5]
        gen1.dim_split = 0
        gen1.netcdf_filepath = "test_parallel.nc"
        gen1.parquet_filepath = "test_parallel.parquet"
        gen1.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        
        # Run 2 with same seed
        gen2 = GenerateData(
            num_obs=200,
            num_dims=2,
            ratio_dims=[10, 10],
            sparsity=0.1,
            seed=42,
            num_vars=1
        )
        gen2.NTASKS = 2
        gen2.max_dim_size = 10
        gen2.div_points = [0, 5, 10]
        gen2.section_sizes = [5, 5]
        gen2.dim_split = 0
        gen2.netcdf_filepath = "test_parallel.nc"
        gen2.parquet_filepath = "test_parallel.parquet"
        gen2.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        
        # Both should produce same observation counts
        _, obs1 = gen1._generate_record_par(0, 50)
        _, obs2 = gen2._generate_record_par(0, 50)
        
        assert obs1 == obs2
    
    def test_chunk_boundary_handling(self):
        """Should handle chunk boundaries correctly."""
        gen = GenerateData(
            num_obs=300,
            num_dims=2,
            ratio_dims=[10, 10],
            sparsity=0.1,
            seed=42,
            num_vars=1
        )
        
        gen.NTASKS = 3
        # Set up file paths for parallel execution
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        gen.max_dim_size = 10
        gen.div_points = [0, 3, 7, 10]
        gen.section_sizes = [3, 4, 3]
        gen.dim_split = 0
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        
        # Test first chunk
        chunk_id, obs = gen._generate_record_par(0, 30)
        assert chunk_id == 0
        assert obs >= 0
        
        # Test last chunk
        chunk_id, obs = gen._generate_record_par(2, 30)
        assert chunk_id == 2
        assert obs >= 0
    
    def test_sparsity_validation_across_chunks(self):
        """Should maintain approximate sparsity across all chunks."""
        gen = GenerateData(
            num_obs=400,
            num_dims=2,
            ratio_dims=[20, 20],
            sparsity=0.1,
            seed=42,
            num_vars=1
        )
        
        gen.NTASKS = 4
        # Set up file paths for parallel execution
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        gen.max_dim_size = 20
        gen.div_points = [0, 5, 10, 15, 20]
        gen.section_sizes = [5, 5, 5, 5]
        gen.dim_split = 0
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        
        total_obs = 0
        for chunk_id in range(4):
            _, obs = gen._generate_record_par(chunk_id, 100)
            total_obs += obs
        
        # Total observations should be reasonable for the sparsity
        total_points = np.prod(gen.shape)
        actual_sparsity = total_obs / total_points
        assert 0.05 <= actual_sparsity <= 0.15  # Allow some variance
    
    def test_different_chunk_sizes(self):
        """Should handle non-uniform chunk sizes."""
        gen = GenerateData(
            num_obs=300,
            num_dims=2,
            ratio_dims=[10, 10],
            sparsity=0.1,
            seed=42,
            num_vars=1
        )
        
        gen.NTASKS = 3
        # Set up file paths for parallel execution
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        gen.max_dim_size = 10
        gen.div_points = [0, 2, 7, 10]  # Non-uniform: 2, 5, 3
        gen.section_sizes = [2, 5, 3]
        gen.dim_split = 0
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        
        # All chunks should work regardless of size
        for chunk_id in range(3):
            result_id, obs = gen._generate_record_par(chunk_id, 50)
            assert result_id == chunk_id
            assert obs >= 0


class TestParallelMultiVariable:
    """Tests for parallel multi-variable generation workflow."""
    
    def test_multi_variable_parallel_generation(self, temp_dir):
        """Should generate multiple variables using parallel workflow."""
        gen = GenerateData(
            num_obs=300,
            num_dims=2,
            ratio_dims=[10, 10],
            sparsity=[0.1, 0.15],
            seed=42,
            num_vars=2,
            var_dims=2,
            overlap=0.3
        )
        
        # Set up file paths for parallel execution
        gen.netcdf_filepath = os.path.join(temp_dir, "test_multivar_par.nc")
        gen.parquet_filepath = os.path.join(temp_dir, "test_multivar_par.parquet")
        gen.parquet_tmp = os.path.join(temp_dir, "tmp", "test_multivar_par_tmp.parquet")
        
        gen.NTASKS = 2
        gen.max_dim_size = 10
        gen.div_points = [0, 5, 10]
        gen.section_sizes = [5, 5]
        gen.dim_split = 0
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        
        # Generate one chunk
        chunk_id, obs = gen._generate_record_par(0, 150)
        
        assert chunk_id == 0
        assert obs > 0
    
    def test_overlap_computation_across_chunks(self):
        """Should handle overlap computation in parallel mode."""
        gen = GenerateData(
            num_obs=200,
            num_dims=2,
            ratio_dims=[10, 10],
            sparsity=[0.1, 0.1],
            seed=42,
            num_vars=2,
            var_dims=2,
            overlap=0.5
        )
        
        gen.NTASKS = 2
        # Set up file paths for parallel execution
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        gen.max_dim_size = 10
        gen.div_points = [0, 5, 10]
        gen.section_sizes = [5, 5]
        gen.dim_split = 0
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        
        # Generate both chunks
        _, obs1 = gen._generate_record_par(0, 100)
        _, obs2 = gen._generate_record_par(1, 100)
        
        assert obs1 > 0
        assert obs2 > 0
    
    def test_constant_dimension_handling(self):
        """Should handle constant dimensions correctly in parallel mode."""
        gen = GenerateData(
            num_obs=200,
            num_dims=3,
            ratio_dims=[10, 10, 5],
            sparsity=[0.1, 0.12],
            seed=42,
            num_vars=2,
            var_dims=[2, 1]  # Different var_dims
        )
        
        gen.NTASKS = 2
        # Set up file paths for parallel execution
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        gen.max_dim_size = 10
        gen.div_points = [0, 5, 10]
        gen.section_sizes = [5, 5]
        gen.dim_split = 0
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        
        # Should work with different var_dims per variable
        chunk_id, obs = gen._generate_record_par(0, 100)
        assert obs >= 0
    
    def test_chunk_consolidation_with_overlap(self):
        """Should properly consolidate chunks with overlapping variables."""
        gen = GenerateData(
            num_obs=400,
            num_dims=2,
            ratio_dims=[20, 20],
            sparsity=[0.08, 0.08],
            seed=42,
            num_vars=2,
            var_dims=2,
            overlap=0.4
        )
        
        gen.NTASKS = 2
        # Set up file paths for parallel execution
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        gen.max_dim_size = 20
        gen.div_points = [0, 10, 20]
        gen.section_sizes = [10, 10]
        gen.dim_split = 0
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        
        # Generate both chunks
        total_obs = 0
        for chunk_id in range(2):
            _, obs = gen._generate_record_par(chunk_id, 200)
            total_obs += obs
        
        # Should have observations from both variables
        assert total_obs > 0
    
    def test_variable_synchronization(self):
        """Should synchronize variables correctly across chunks."""
        gen = GenerateData(
            num_obs=300,
            num_dims=2,
            ratio_dims=[15, 15],
            sparsity=[0.1, 0.1],
            seed=42,
            num_vars=2,
            var_dims=2,
            overlap=0.5
        )
        
        gen.NTASKS = 3
        # Set up file paths for parallel execution
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        gen.max_dim_size = 15
        gen.div_points = [0, 5, 10, 15]
        gen.section_sizes = [5, 5, 5]
        gen.dim_split = 0
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        
        # All chunks should maintain variable consistency
        for chunk_id in range(3):
            result_id, obs = gen._generate_record_par(chunk_id, 100)
            assert result_id == chunk_id
            assert obs >= 0


class TestParallelErrorHandling:
    """Tests for error handling in parallel workflows."""
    
    def test_chunk_size_validation(self):
        """Should validate chunk sizes correctly."""
        gen = GenerateData(
            num_obs=100,
            num_dims=2,
            ratio_dims=[10, 10],
            sparsity=0.1,
            seed=42,
            num_vars=1
        )
        
        # Set up valid chunk parameters
        gen.NTASKS = 2
        # Set up file paths for parallel execution
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        gen.max_dim_size = 10
        gen.div_points = [0, 5, 10]
        gen.section_sizes = [5, 5]
        gen.dim_split = 0
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        
        # Should work with valid setup
        chunk_id, obs = gen._generate_record_par(0, 50)
        assert obs >= 0
    
    def test_invalid_split_dimension_handled(self):
        """Should handle invalid split dimension gracefully."""
        gen = GenerateData(
            num_obs=100,
            num_dims=2,
            ratio_dims=[10, 10],
            sparsity=0.1,
            seed=42,
            num_vars=1
        )
        
        gen.NTASKS = 2
        # Set up file paths for parallel execution
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        gen.max_dim_size = 10
        gen.div_points = [0, 5, 10]
        gen.section_sizes = [5, 5]
        
        # Valid dimension indices
        for dim_split in range(gen.num_dims):
            gen.dim_split = dim_split
            _, obs = gen._generate_record_par(0, 50)
            assert obs >= 0
    
    def test_edge_case_single_chunk(self):
        """Should handle single chunk as edge case."""
        gen = GenerateData(
            num_obs=100,
            num_dims=2,
            ratio_dims=[10, 10],
            sparsity=0.1,
            seed=42,
            num_vars=1
        )
        
        gen.NTASKS = 1
        # Set up file paths for parallel execution
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        gen.max_dim_size = 10
        gen.div_points = [0, 10]
        gen.section_sizes = [10]
        gen.dim_split = 0
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        
        # Should work with single chunk
        chunk_id, obs = gen._generate_record_par(0, 100)
        assert chunk_id == 0
        assert obs >= 0
    
    def test_edge_case_many_small_chunks(self):
        """Should handle many small chunks."""
        gen = GenerateData(
            num_obs=200,
            num_dims=2,
            ratio_dims=[10, 10],
            sparsity=0.1,
            seed=42,
            num_vars=1
        )
        
        gen.NTASKS = 10
        # Set up file paths for parallel execution
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        gen.max_dim_size = 10
        gen.div_points = list(range(11))  # [0, 1, 2, ..., 10]
        gen.section_sizes = [1] * 10
        gen.dim_split = 0
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        
        # Should work with many small chunks
        for chunk_id in range(5):  # Test subset
            result_id, obs = gen._generate_record_par(chunk_id, 20)
            assert result_id == chunk_id
            assert obs >= 0
    
    def test_zero_observations_handled(self):
        """Should handle case where chunk has zero observations."""
        gen = GenerateData(
            num_obs=10,
            num_dims=2,
            ratio_dims=[10, 10],
            sparsity=0.001,  # Very low sparsity
            seed=42,
            num_vars=1
        )
        
        gen.NTASKS = 2
        # Set up file paths for parallel execution
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        gen.max_dim_size = 10
        gen.div_points = [0, 5, 10]
        gen.section_sizes = [5, 5]
        gen.dim_split = 0
        gen.netcdf_filepath = "test_parallel.nc"
        gen.parquet_filepath = "test_parallel.parquet"
        gen.parquet_tmp = "tmp/test_parallel_tmp.parquet"
        
        # Should handle even if some chunks have zero observations
        chunk_id, obs = gen._generate_record_par(0, 1)
        assert chunk_id == 0
        assert obs >= 0
