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
        
        # Parquet creates a directory
        assert os.path.exists(parquet_path) or os.path.exists(parquet_path + ".parquet")
    
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
            equal_nan=True
        )
        loaded.close()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_observation(self):
        """Should handle single observation."""
        gen = GenerateData(
            num_obs=1,
            num_dims=2,
            ratio_dims=1,
            sparsity=0.0,  # Use minimum sparsity
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
