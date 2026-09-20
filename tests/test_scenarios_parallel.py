"""Parallel versions of TestScenarios - testing with max_obs to trigger parallel execution.

All tests should produce identical results to their serial counterparts,
except that data is read from disk (NetCDF/Parquet) after generation.
"""

import pytest
import numpy as np
import xarray as xr
import pandas as pd
import os
import glob
from data_sparsity.generate_data import GenerateData


class TestScenariosParallel:
    """Parallel versions of TestScenarios with max_obs."""
    
    @staticmethod
    def load_generated_data(gen: GenerateData):
        """Helper to load data generated in parallel mode.
        
        Args:
            gen: GenerateData instance that has executed parallel generation
            
        Returns:
            Tuple of (dataarray/dataset, dataframe) loaded from disk
        """
        import dask.dataframe as dd
        import glob
        
        # Load NetCDF - parallel mode creates multiple chunk files
        # Find all chunk files with pattern: base_0.nc, base_1.nc, etc.
        base_path = gen.netcdf_filepath[:-3]  # Remove '.nc'
        netcdf_pattern = f"{base_path}_*.nc"
        netcdf_files = sorted(glob.glob(netcdf_pattern))
        
        if not netcdf_files:
            raise FileNotFoundError(
                f"No NetCDF chunk files found matching pattern {netcdf_pattern}"
            )
        
        # Open and concatenate all chunks along the split dimension
        if gen.num_vars == 1:
            # For DataArray, open all chunks and concatenate
            dataarrays = [xr.open_dataarray(f) for f in netcdf_files]
            dataarray = xr.concat(dataarrays, dim=f'x{gen.dim_split}')
            # Close individual files
            for da in dataarrays:
                da.close()
        else:
            # For Dataset, open all chunks and concatenate
            datasets = [xr.open_dataset(f) for f in netcdf_files]
            dataarray = xr.concat(datasets, dim=f'x{gen.dim_split}')
            # Close individual files
            for ds in datasets:
                ds.close()
        
        # Load Parquet - Dask saves to directory-based storage with multiple part files
        # The filepath "/path/to/file.parquet" actually creates files in "/path/to/"
        # with names like "file_0.parquet", "file_1.parquet", etc.
        dirpath = os.path.dirname(gen.parquet_filepath)
        filename = os.path.basename(gen.parquet_filepath)
        if filename.endswith('.parquet'):
            filename = filename[:-8]  # Remove .parquet extension
        
        # Read all partition files
        parquet_pattern = os.path.join(dirpath, f"{filename}_*.parquet")
        
        try:
            # Try reading as pandas first (for smaller files)
            dataframe = pd.read_parquet(dirpath, engine='pyarrow')
        except Exception:
            # Fall back to Dask for larger files
            try:
                dataframe = dd.read_parquet(parquet_pattern).compute()
            except Exception:
                # Last resort: try reading the directory directly
                dataframe = dd.read_parquet(dirpath).compute()
        
        return dataarray, dataframe
    
    def test_scenario_1a_parallel(self, temp_dir):
        """Should generate a full 1D array (parallel execution)"""
        
        gen = GenerateData(
            num_obs=100,
            num_dims=1,
            ratio_dims=1,
            density=1.,
            seed=34,
            max_obs=25  # Trigger parallel with 4 chunks
        )
        
        # Parallel generation returns None, None
        result = gen.generate(
            netcdf_filepath=os.path.join(temp_dir, "test_1a_p.nc"),
            parquet_filepath=os.path.join(temp_dir, "test_1a_p.parquet"),
            parquet_tmp=os.path.join(temp_dir, "tmp_1a_p")
        )
        assert result == (None, None), "Parallel generation should return (None, None)"
        
        # Load generated data
        dataarray, dataframe = self.load_generated_data(gen)
        
        # Same checks as test_scenario_1a
        assert len(dataarray.dims) == 1, "DataArray should have exactly one dimension"
        assert 'x0' in dataarray.dims, "Dimension should be named 'x0'"
        assert len(dataarray.coords['x0']) == 100, "'x0' should have 100 coordinates"
        
        non_nan_count = np.count_nonzero(~np.isnan(dataarray.values))
        assert non_nan_count == 100, "All 100 coordinates should have non-NaN values"
        
        assert len(dataarray.coords['x0']) == len(np.unique(dataarray.coords['x0'])), \
            "All coordinates in x0 should be unique"
        
        valid_values = dataarray.values[~np.isnan(dataarray.values)]
        assert len(valid_values) == len(np.unique(valid_values)), \
            "All record values should be unique"
        
        # DataFrame checks
        assert list(dataframe.columns) == ['x0', 'record'], \
            "DataFrame should have exactly two columns: 'x0' and 'record'"
        assert len(dataframe['x0']) == 100, "'x0' column should have 100 rows"
        assert len(dataframe['record']) == 100, "'record' column should have 100 rows"
        assert len(dataframe['x0']) == len(dataframe['x0'].unique()), \
            "All coordinates in 'x0' should be unique"
        assert len(dataframe['record']) == len(dataframe['record'].unique()), \
            "All record values should be unique"
    
    def test_scenario_1b_parallel(self, temp_dir):
        """Should generate a full 2D array (parallel execution)"""
        
        m = 100
        n = 33
        gen = GenerateData(
            num_obs=m*n,
            num_dims=2,
            ratio_dims=[m,n],
            density=1.,
            seed=76,
            max_obs=800  # Trigger parallel with ~4 chunks
        )
        
        result = gen.generate(
            netcdf_filepath=os.path.join(temp_dir, "test_1b_p.nc"),
            parquet_filepath=os.path.join(temp_dir, "test_1b_p.parquet"),
            parquet_tmp=os.path.join(temp_dir, "tmp_1b_p")
        )
        assert result == (None, None)
        
        dataarray, dataframe = self.load_generated_data(gen)
        
        # Same checks as test_scenario_1b
        assert len(dataarray.dims) == 2, "DataArray should have exactly two dimensions"
        assert list(dataarray.dims) == ['x0', 'x1'], \
            "Dimensions should be named 'x0' and 'x1' in that order"
        assert len(dataarray.coords['x0']) == m, f"'x0' should have {m} coordinates"
        assert len(dataarray.coords['x1']) == n, f"'x1' should have {n} coordinates"
        
        non_nan_count = np.count_nonzero(~np.isnan(dataarray.values))
        assert non_nan_count == m * n, \
            f"All {m * n} coordinate pairs should have non-NaN values"
        
        assert len(dataarray.coords['x0']) == len(np.unique(dataarray.coords['x0'])), \
            "All coordinates in x0 should be unique"
        assert len(dataarray.coords['x1']) == len(np.unique(dataarray.coords['x1'])), \
            "All coordinates in x1 should be unique"
        
        valid_values = dataarray.values[~np.isnan(dataarray.values)].flatten()
        assert len(valid_values) == len(np.unique(valid_values)), \
            "All record values should be unique"
        
        # DataFrame checks
        assert list(dataframe.columns) == ['x0', 'x1', 'record'], \
            "DataFrame should have exactly three columns"
        assert len(dataframe['x0']) == m * n
        assert len(dataframe['x0'].unique()) == m
        assert len(dataframe['x1']) == m * n
        assert len(dataframe['x1'].unique()) == n
        
        unique_combinations = dataframe[['x0', 'x1']].drop_duplicates()
        assert len(unique_combinations) == m * n
        
        assert len(dataframe['record']) == m * n
        assert len(dataframe['record']) == len(dataframe['record'].unique())
    
    def test_scenario_1c_parallel(self, temp_dir):
        """Should generate a full 10-D array (parallel execution)"""
        
        m = [10,5,6,8,2,4,5,3,3,7]
        
        gen = GenerateData(
            num_obs=int(np.prod(m)),
            num_dims=len(m),
            ratio_dims=m,
            density=1.,
            seed=10,
            max_obs=int(np.floor(np.prod(m)/np.max(m)))  # Trigger parallel
        )
        
        result = gen.generate(
            netcdf_filepath=os.path.join(temp_dir, "test_1c_p.nc"),
            parquet_filepath=os.path.join(temp_dir, "test_1c_p.parquet"),
            parquet_tmp=os.path.join(temp_dir, "tmp_1c_p")
        )
        assert result == (None, None)
        
        dataarray, dataframe = self.load_generated_data(gen)
        
        # Same checks as test_scenario_1c
        assert len(dataarray.dims) == len(m)
        
        expected_dims = [f'x{i}' for i in range(len(m))]
        assert list(dataarray.dims) == expected_dims
        
        for i, size in enumerate(m):
            dim_name = f'x{i}'
            assert len(dataarray.coords[dim_name]) == size
        
        total_points = np.prod(m)
        non_nan_count = np.count_nonzero(~np.isnan(dataarray.values))
        assert non_nan_count == total_points
        
        for i, size in enumerate(m):
            dim_name = f'x{i}'
            coords = dataarray.coords[dim_name]
            assert len(coords) == len(np.unique(coords))
        
        valid_values = dataarray.values[~np.isnan(dataarray.values)].flatten()
        assert len(valid_values) == len(np.unique(valid_values))
        
        # DataFrame checks
        expected_columns = [f'x{i}' for i in range(len(m))] + ['record']
        assert list(dataframe.columns) == expected_columns
        
        total_rows = np.prod(m)
        assert len(dataframe) == total_rows
        
        for i, size in enumerate(m):
            col_name = f'x{i}'
            unique_count = len(dataframe[col_name].unique())
            assert unique_count == size
        
        coord_cols = [f'x{i}' for i in range(len(m))]
        unique_combinations = dataframe[coord_cols].drop_duplicates()
        assert len(unique_combinations) == total_rows
        
        assert len(dataframe['record']) == total_rows
        assert len(dataframe['record']) == len(dataframe['record'].unique())
    
    def test_scenario_2b_parallel(self, temp_dir):
        """Should generate a sparse 2D array (parallel execution)"""
        
        gen = GenerateData(
            num_obs=5,
            num_dims=2,
            ratio_dims=1,
            density=1/5,
            seed=35,
            max_obs=2  # Very small chunks to test chunking with minimal data
        )
        
        result = gen.generate(
            netcdf_filepath=os.path.join(temp_dir, "test_2b_p.nc"),
            parquet_filepath=os.path.join(temp_dir, "test_2b_p.parquet"),
            parquet_tmp=os.path.join(temp_dir, "tmp_2b_p")
        )
        assert result == (None, None)
        
        dataarray, dataframe = self.load_generated_data(gen)
        
        # Same checks as test_scenario_2b
        assert len(dataarray.dims) == 2
        assert list(dataarray.dims) == ['x0', 'x1']
        assert len(dataarray.coords['x0']) == 5
        assert len(dataarray.coords['x1']) == 5
        
        non_nan_mask = ~np.isnan(dataarray.values)
        coords = np.argwhere(non_nan_mask)
        for i, j in coords:
            row_count = int(non_nan_mask[i, :].sum())
            col_count = int(non_nan_mask[:, j].sum())
            assert row_count == 1
            assert col_count == 1
        
        assert len(dataarray.coords['x0']) == len(np.unique(dataarray.coords['x0']))
        assert len(dataarray.coords['x1']) == len(np.unique(dataarray.coords['x1']))
        
        valid_values = dataarray.values[~np.isnan(dataarray.values)].flatten()
        assert len(valid_values) == len(np.unique(valid_values))
        
        # DataFrame checks
        assert list(dataframe.columns) == ['x0', 'x1', 'record']
        assert len(dataframe) == 5
        assert len(dataframe['x0'].unique()) == 5
        assert len(dataframe['x1'].unique()) == 5
        
        unique_combinations = dataframe[['x0', 'x1']].drop_duplicates()
        assert len(unique_combinations) == 5
        
        assert len(dataframe['record']) == 5
        assert len(dataframe['record']) == len(dataframe['record'].unique())
        assert not dataframe.isnull().any().any()
    
    def test_scenario_2c_parallel(self, temp_dir):
        """Should generate a 10-D array with minimum sparsity (parallel execution)"""
        
        gen = GenerateData(
            num_obs=3,
            num_dims=10,
            ratio_dims=1,
            density=1/(3**9),
            seed=11,
            max_obs=1  # Each observation in separate chunk
        )
        
        result = gen.generate(
            netcdf_filepath=os.path.join(temp_dir, "test_2c_p.nc"),
            parquet_filepath=os.path.join(temp_dir, "test_2c_p.parquet"),
            parquet_tmp=os.path.join(temp_dir, "tmp_2c_p")
        )
        assert result == (None, None)
        
        dataarray, dataframe = self.load_generated_data(gen)
        
        # Same checks as test_scenario_2c
        assert len(dataarray.dims) == 10
        
        expected_dims = [f'x{i}' for i in range(10)]
        assert list(dataarray.dims) == expected_dims
        
        dim_sizes = [len(dataarray.coords[f'x{i}']) for i in range(10)]
        assert len(set(dim_sizes)) == 1
        
        # Check "exactly once" semantics
        non_nan_mask = ~np.isnan(dataarray.values)
        non_nan_count = np.count_nonzero(non_nan_mask)
        non_nan_indices = np.argwhere(non_nan_mask)
        ndim = dataarray.ndim
        dim_names = list(dataarray.dims)
        
        counts_by_axis = []
        for axis in range(ndim):
            axis_indices = non_nan_indices[:, axis]
            counts = np.bincount(axis_indices, minlength=dataarray.shape[axis])
            counts_by_axis.append(counts)
        
        for idx_tuple in map(tuple, non_nan_indices):
            for axis, coord_idx in enumerate(idx_tuple):
                cnt = counts_by_axis[axis][coord_idx]
                if cnt != 1:
                    dim = dim_names[axis]
                    coord_label = dataarray.coords[dim][coord_idx]
                    raise AssertionError(
                        f"Dimension '{dim}' coordinate index {coord_idx} (label={coord_label!r}) "
                        f"is used {int(cnt)} times; expected exactly 1"
                    )
        
        for i in range(10):
            dim_name = f'x{i}'
            coords = dataarray.coords[dim_name]
            assert len(coords) == len(np.unique(coords))
        
        valid_values = dataarray.values[~np.isnan(dataarray.values)].flatten()
        assert len(valid_values) == len(np.unique(valid_values))
        
        # DataFrame checks
        expected_columns = [f'x{i}' for i in range(10)] + ['record']
        assert list(dataframe.columns) == expected_columns
        assert len(dataframe) == gen.num_obs
        
        for i in range(10):
            col_name = f'x{i}'
            unique_count = len(dataframe[col_name].unique())
            assert unique_count == gen.num_obs
        
        coord_cols = [f'x{i}' for i in range(10)]
        unique_combinations = dataframe[coord_cols].drop_duplicates()
        assert len(unique_combinations) == gen.num_obs
        
        assert len(dataframe['record']) == gen.num_obs
        assert len(dataframe['record']) == len(dataframe['record'].unique())
        assert not dataframe.isnull().any().any()
    
    def test_scenario_3a_parallel(self, temp_dir):
        """Should generate a sparse 2D array (parallel execution)"""
        
        gen = GenerateData(
            num_obs=10,
            num_dims=2,
            ratio_dims=[7,5],
            density=10/35,
            seed=31,
            max_obs=5
        )
        
        result = gen.generate(
            netcdf_filepath=os.path.join(temp_dir, "test_3a_p.nc"),
            parquet_filepath=os.path.join(temp_dir, "test_3a_p.parquet"),
            parquet_tmp=os.path.join(temp_dir, "tmp_3a_p")
        )
        assert result == (None, None)
        
        dataarray, dataframe = self.load_generated_data(gen)
        
        # Same checks as test_scenario_3a
        assert len(dataarray.dims) == 2
        assert list(dataarray.dims) == ['x0', 'x1']
        assert len(dataarray.coords['x0']) == 7
        assert len(dataarray.coords['x1']) == 5
        
        x0_with_data = np.any(~np.isnan(dataarray.values), axis=1)
        x1_with_data = np.any(~np.isnan(dataarray.values), axis=0)
        assert np.any(x0_with_data)
        assert np.any(x1_with_data)
        
        assert len(dataarray.coords['x0']) == len(np.unique(dataarray.coords['x0']))
        assert len(dataarray.coords['x1']) == len(np.unique(dataarray.coords['x1']))
        
        non_nan_count = np.count_nonzero(~np.isnan(dataarray.values))
        total_points = 7 * 5
        assert non_nan_count < total_points
        
        valid_values = dataarray.values[~np.isnan(dataarray.values)].flatten()
        assert len(valid_values) == len(np.unique(valid_values))
        
        actual_sparsity = non_nan_count / total_points
        expected_sparsity = 10/35
        assert abs(actual_sparsity - expected_sparsity) < 0.01
        
        # DataFrame checks
        assert list(dataframe.columns) == ['x0', 'x1', 'record']
        assert len(dataframe) == 10
        
        x0_unique = len(dataframe['x0'].unique())
        assert x0_unique <= 7
        assert x0_unique >= 1
        
        x1_unique = len(dataframe['x1'].unique())
        assert x1_unique <= 5
        assert x1_unique >= 1
        
        assert len(dataframe['record']) == len(dataframe['record'].unique())
        assert not dataframe.isnull().any().any()
    
    def test_scenario_3b_parallel(self, temp_dir):
        """Should generate a 2D array with all sites full except one (parallel execution)"""
        
        m = 100
        n = 4
        gen = GenerateData(
            num_obs=(m*n-1),
            num_dims=2,
            ratio_dims=[m,n],
            density=(m*n-1)/(m*n),
            seed=20,
            max_obs=100  # ~3 chunks
        )
        
        result = gen.generate(
            netcdf_filepath=os.path.join(temp_dir, "test_3b_p.nc"),
            parquet_filepath=os.path.join(temp_dir, "test_3b_p.parquet"),
            parquet_tmp=os.path.join(temp_dir, "tmp_3b_p")
        )
        assert result == (None, None)
        
        dataarray, dataframe = self.load_generated_data(gen)
        
        # Same checks as test_scenario_3b
        assert len(dataarray.dims) == 2
        assert list(dataarray.dims) == ['x0', 'x1']
        assert len(dataarray.coords['x0']) == m
        assert len(dataarray.coords['x1']) == n
        
        x0_with_data = np.any(~np.isnan(dataarray.values), axis=1)
        assert np.all(x0_with_data)
        x1_with_data = np.any(~np.isnan(dataarray.values), axis=0)
        assert np.all(x1_with_data)
        
        assert len(dataarray.coords['x0']) == len(np.unique(dataarray.coords['x0']))
        assert len(dataarray.coords['x1']) == len(np.unique(dataarray.coords['x1']))
        
        non_nan_count = np.count_nonzero(~np.isnan(dataarray.values))
        total_points = m * n
        #assert non_nan_count < total_points
        assert non_nan_count == m * n - 1
        
        valid_values = dataarray.values[~np.isnan(dataarray.values)].flatten()
        assert len(valid_values) == len(np.unique(valid_values))
        
        actual_sparsity = non_nan_count / total_points
        expected_sparsity = (m * n - 1) / (m * n)
        assert abs(actual_sparsity - expected_sparsity) < 0.001
        
        # DataFrame checks
        assert list(dataframe.columns) == ['x0', 'x1', 'record']
        assert len(dataframe) == m * n - 1
        assert len(dataframe['x0'].unique()) == m
        assert len(dataframe['x1'].unique()) == n
        assert len(dataframe['record']) == len(dataframe['record'].unique())
        assert not dataframe.isnull().any().any()
    
    def test_scenario_3c_parallel(self, temp_dir):
        """Should generate a 10D array with all sites full except one (parallel execution)"""
        
        m = [3,5,6,9,2,4,5]
        num_obs = int(np.prod(m)-1)
        #max_obs = int(np.floor(num_obs/np.max(m)+1))
        max_obs = int(np.ceil(num_obs/4))
        
        gen = GenerateData(
            num_obs=num_obs,
            num_dims=len(m),
            ratio_dims=m,
            density=(np.prod(m)-1)/np.prod(m),
            seed=20,
            max_obs=max_obs
        )
        
        result = gen.generate(
            netcdf_filepath=os.path.join(temp_dir, "test_3c_p.nc"),
            parquet_filepath=os.path.join(temp_dir, "test_3c_p.parquet"),
            parquet_tmp=os.path.join(temp_dir, "tmp_3c_p")
        )
        assert result == (None, None)
        
        dataarray, dataframe = self.load_generated_data(gen)
        
        # Same checks as test_scenario_3c
        assert len(dataarray.dims) == len(m)
        
        expected_dims = [f'x{i}' for i in range(len(m))]
        assert list(dataarray.dims) == expected_dims
        
        for i, size in enumerate(m):
            dim_name = f'x{i}'
            assert len(dataarray.coords[dim_name]) == size
        
        for i in range(len(m)):
            axes = tuple(j for j in range(len(m)) if j != i)
            dim_with_data = np.any(~np.isnan(dataarray.values), axis=axes)
            assert np.all(dim_with_data)
        
        for i, size in enumerate(m):
            dim_name = f'x{i}'
            coords = dataarray.coords[dim_name]
            assert len(coords) == len(np.unique(coords))
        
        total_points = int(np.prod(m))
        non_nan_count = np.count_nonzero(~np.isnan(dataarray.values))
        assert non_nan_count == total_points - 1
        
        valid_values = dataarray.values[~np.isnan(dataarray.values)].flatten()
        assert len(valid_values) == len(np.unique(valid_values))
        
        actual_sparsity = non_nan_count / total_points
        expected_sparsity = (np.prod(m) - 1) / np.prod(m)
        assert abs(actual_sparsity - expected_sparsity) < 0.0001
        
        # DataFrame checks
        expected_columns = [f'x{i}' for i in range(len(m))] + ['record']
        assert list(dataframe.columns) == expected_columns
        
        expected_rows = int(np.prod(m) - 1)
        assert len(dataframe) == expected_rows
        
        for i, size in enumerate(m):
            col_name = f'x{i}'
            unique_count = len(dataframe[col_name].unique())
            assert unique_count == size
        
        coord_cols = [f'x{i}' for i in range(len(m))]
        unique_combinations = dataframe[coord_cols].drop_duplicates()
        assert len(unique_combinations) == expected_rows
        
        assert len(dataframe['record']) == len(dataframe['record'].unique())
        assert not dataframe.isnull().any().any()
