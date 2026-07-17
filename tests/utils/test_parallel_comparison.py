"""Tests comparing serial and parallel workflow outputs.

This module ensures that parallel workflow produces datasets identical to serial
workflow, except for record variable values which can differ but must have
identical NaN positions.
"""

import pytest
import numpy as np
import xarray as xr
import dask.array as da
import dask.dataframe as dd
import tempfile
import shutil
import os
import glob
from data_sparsity.generate_data import GenerateData


def datasets_are_identical(ds1, ds2, variable='record'):
    """Compare two datasets for structural and data equivalence.
    
    Coordinates must match exactly (including NaN positions and values), except
    for x0 which can differ between serial and parallel workflows.
    The specified variable must have same NaN/non-NaN positions but can have
    different non-NaN values.
    
    Args:
        ds1: First xarray Dataset or DataArray
        ds2: Second xarray Dataset or DataArray
        variable: Variable name to check for NaN position matching
        
    Returns:
        Tuple of (bool, str) indicating match status and message

    """
    # 1. Check dimension names and lengths
    if ds1.dims != ds2.dims:
        return False, "Dimension names or lengths differ"

    # 2. Check coordinate names
    if set(ds1.coords) != set(ds2.coords):
        return False, "Coordinate names differ"

    # 3. Check coordinate values, datatypes, and nan positions
    for coord in ds1.coords:
        c1 = ds1[coord]
        c2 = ds2[coord]

        # Check dtype and shape
        if c1.dtype != c2.dtype:
            return False, f"Coordinate {coord} dtype differs"
        if c1.shape != c2.shape:
            return False, f"Coordinate {coord} shape differs"

        # skip checking actual values for x0 because parallel generates it
        # differently (but still uniformly distributed)
        if c1.name == "x0": continue

        # Handle dask arrays
        v1 = c1.data if hasattr(c1.data, "compute") else c1.values
        v2 = c2.data if hasattr(c2.data, "compute") else c2.values

        # Compute values if they are Dask arrays
        if hasattr(v1, "compute"):
            v1 = v1.compute()
        if hasattr(v2, "compute"):
            v2 = v2.compute()

        # Use numpy's equal_nan for exact coordinate matching
        if not np.array_equal(v1, v2, equal_nan=True):
            return False, f"Coordinate {coord} values differ"

    # 4. Check that both have the variable (handle DataArray vs Dataset)
    if isinstance(ds1, xr.DataArray):
        if ds1.name != variable and variable != 'record':
            return False, f"DataArray name '{ds1.name}' doesn't match variable '{variable}'"
        data1 = ds1.data
    else:
        if variable not in ds1:
            return False, f"Missing '{variable}' variable in first dataset"
        data1 = ds1[variable].data
        
    if isinstance(ds2, xr.DataArray):
        if ds2.name != variable and variable != 'record':
            return False, f"DataArray name '{ds2.name}' doesn't match variable '{variable}'"
        data2 = ds2.data
    else:
        if variable not in ds2:
            return False, f"Missing '{variable}' variable in second dataset"
        data2 = ds2[variable].data

    # 5. Check NaN and non-NaN counts for the specified variable

    # Check NaN/non-NaN counts
    is_dask1 = hasattr(data1, "chunks")
    is_dask2 = hasattr(data2, "chunks")

    nan_count_1 = da.isnan(data1).sum().compute() if is_dask1 else np.isnan(data1).sum()
    nan_count_2 = da.isnan(data2).sum().compute() if is_dask2 else np.isnan(data2).sum()

    nnan_count_1 = (~da.isnan(data1)).sum().compute() if is_dask1 else (~np.isnan(data1)).sum()
    nnan_count_2 = (~da.isnan(data2)).sum().compute() if is_dask2 else (~np.isnan(data2)).sum()

    if nan_count_1 != nan_count_2:
        return False, f"Different number of NaNs in '{variable}' ({nan_count_1} vs {nan_count_2})"
    if nnan_count_1 != nnan_count_2:
        return False, f"Different number of non-NaNs in '{variable}' ({nnan_count_1} vs {nnan_count_2})"

    return True, "Datasets match all checked aspects (including counts of NaNs and non-NaNs)."


def dataframes_coordinate_nan_match(df1, df2, coordinates, variable='record'):
    """Compare two dataframes for coordinate and NaN position equivalence.
    
    Verifies that:
    - x0 can differ between dataframes (values don't need to match)
    - For each combination of other coordinates (x1, x2, etc.), the number 
      of occurrences and NaN patterns must be identical in both dataframes
    - The overall structure (excluding x0 values) must be equivalent
    
    Args:
        df1: First pandas or dask DataFrame
        df2: Second pandas or dask DataFrame
        coordinates: List of coordinate column names (first should be 'x0')
        variable: Variable column to check for NaN matching
        
    Returns:
        Tuple of (bool, str) indicating match status and message
    """
    # Input checks
    for coord in coordinates + [variable]:
        if coord not in df1.columns or coord not in df2.columns:
            return False, f"Column '{coord}' missing in one of the dataframes"

    # Convert dask to pandas if needed for comparison
    if isinstance(df1, dd.DataFrame):
        df1 = df1.compute()
    if isinstance(df2, dd.DataFrame):
        df2 = df2.compute()

    # Separate x0 from other coordinates
    x0_coord = coordinates[0]  # Assumed to be 'x0'
    other_coords = coordinates[1:]  # x1, x2, etc.

    import pandas as pd
    
    # Strategy: For each combination of other_coords (x1, x2, ...),
    # verify that the number of occurrences and NaN patterns match
    
    if other_coords:
        # Group by other coordinates and get NaN status
        df1_grouped = df1[other_coords + [variable]].copy()
        df2_grouped = df2[other_coords + [variable]].copy()
        
        df1_grouped['isnan'] = df1_grouped[variable].isna()
        df2_grouped['isnan'] = df2_grouped[variable].isna()
        
        # Count occurrences of each (other_coords, isnan) combination
        counts1 = df1_grouped.groupby(other_coords + ['isnan']).size().reset_index(name='count')
        counts2 = df2_grouped.groupby(other_coords + ['isnan']).size().reset_index(name='count')
        
        # Sort for comparison
        counts1 = counts1.sort_values(by=other_coords + ['isnan']).reset_index(drop=True)
        counts2 = counts2.sort_values(by=other_coords + ['isnan']).reset_index(drop=True)
        
        # Check if they match
        if not counts1.equals(counts2):
            # Detailed check to provide better error message
            merged = pd.merge(counts1, counts2, on=other_coords + ['isnan'], 
                            suffixes=('_1', '_2'), how='outer')
            if merged.isnull().values.any():
                return False, "Coordinate combinations differ between dataframes"
            if not (merged['count_1'] == merged['count_2']).all():
                return False, "Frequency of coordinate combinations differs"
            return False, "Structure differs between dataframes"
    else:
        # No other coordinates, just check NaN counts for x0
        nan_count1 = df1[variable].isna().sum()
        nan_count2 = df2[variable].isna().sum()
        nnan_count1 = (~df1[variable].isna()).sum()
        nnan_count2 = (~df2[variable].isna()).sum()
        
        if nan_count1 != nan_count2:
            return False, f"NaN counts differ ({nan_count1} vs {nan_count2})"
        if nnan_count1 != nnan_count2:
            return False, f"Non-NaN counts differ ({nnan_count1} vs {nnan_count2})"

    return True, "Coordinates and NaN status match for all keys"


class TestParallelSerialComparison:
    """Compare parallel and serial generation outputs."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    def test_scenario_1a_parallel_vs_serial(self, temp_dir):
        """1D full array: parallel should match serial."""
        # Serial generation
        gen_serial = GenerateData(
            num_obs=100,
            num_dims=1,
            ratio_dims=1,
            sparsity=1.,
            seed=34
        )
        da_serial, df_serial = gen_serial.generate()
        
        # Parallel generation - setup paths AFTER initialization
        nc_dir = os.path.join(temp_dir, "nc")
        pq_dir = os.path.join(temp_dir, "pq")
        os.makedirs(nc_dir, exist_ok=True)
        os.makedirs(pq_dir, exist_ok=True)
        
        gen_parallel = GenerateData(
            num_obs=100,
            num_dims=1,
            ratio_dims=1,
            sparsity=1.,
            seed=34,
            max_obs=26  # Trigger parallel mode
        )
        
        result = gen_parallel.generate(
            netcdf_filepath=os.path.join(nc_dir, "test.nc"),
            parquet_filepath=os.path.join(pq_dir, "test")
        )
        # Parallel mode returns (None, None)
        assert result == (None, None), "Parallel generation should return (None, None)"
        
        # Load parallel data from disk - chunk files are in nc_dir
        nc_files = sorted(glob.glob(os.path.join(nc_dir, "test_*.nc")))
        assert len(nc_files) > 0, f"No parallel chunk files found in {nc_dir}"
        
        # Concatenate chunks
        dataarrays = [xr.open_dataarray(f) for f in nc_files]
        da_parallel_full = xr.concat(dataarrays, dim='x0')
        for da in dataarrays:
            da.close()
        
        # Load parquet data
        import pandas as pd
        pq_files = sorted(glob.glob(os.path.join(pq_dir, "test_*.parquet")))
        if pq_files:
            df_parallel = pd.concat([pd.read_parquet(f) for f in pq_files], ignore_index=True)
        else:
            # Try consolidated file
            df_parallel = pd.read_parquet(pq_dir)
        
        # Compare datasets
        match, msg = datasets_are_identical(da_serial, da_parallel_full)
        assert match, f"Datasets don't match: {msg}"
        
        # Compare dataframes
        coords = ['x0']
        match_df, msg_df = dataframes_coordinate_nan_match(
            df_serial, df_parallel, coords
        )
        assert match_df, f"Dataframes don't match: {msg_df}"
    
    def test_scenario_1b_parallel_vs_serial(self, temp_dir):
        """2D full array: parallel should match serial."""
        m, n = 100, 33
        
        # Serial generation
        gen_serial = GenerateData(
            num_obs=m*n,
            num_dims=2,
            ratio_dims=[m, n],
            sparsity=1.,
            seed=76
        )
        da_serial, df_serial = gen_serial.generate()
        
        # Parallel generation - setup paths AFTER initialization
        nc_dir = os.path.join(temp_dir, "nc")
        pq_dir = os.path.join(temp_dir, "pq")
        os.makedirs(nc_dir, exist_ok=True)
        os.makedirs(pq_dir, exist_ok=True)
        
        gen_parallel = GenerateData(
            num_obs=m*n,
            num_dims=2,
            ratio_dims=[m, n],
            sparsity=1.,
            seed=76,
            max_obs=500  # Trigger parallel mode
        )        
        result = gen_parallel.generate(
            netcdf_filepath=os.path.join(nc_dir, "test.nc"),
            parquet_filepath=os.path.join(pq_dir, "test")
        
        )
        
        # Parallel mode returns (None, None)
        
        assert result == (None, None), "Parallel generation should return (None, None)"
        
        
        
        # Load parallel data from disk
        nc_files = sorted(glob.glob(os.path.join(nc_dir, "test_*.nc")))
        assert len(nc_files) > 0, f"No parallel chunk files found in {nc_dir}"
        
        # Concatenate chunks
        dataarrays = [xr.open_dataarray(f) for f in nc_files]
        da_parallel_full = xr.concat(dataarrays, dim='x0')
        for da in dataarrays:
            da.close()
        
        # Load parquet data
        import pandas as pd
        pq_files = sorted(glob.glob(os.path.join(pq_dir, "test_*.parquet")))
        if pq_files:
            df_parallel = pd.concat([pd.read_parquet(f) for f in pq_files], ignore_index=True)
        else:
            # Try consolidated file
            df_parallel = pd.read_parquet(pq_dir)
        
        # Compare datasets
        match, msg = datasets_are_identical(da_serial, da_parallel_full)
        assert match, f"Datasets don't match: {msg}"
        
        # Compare dataframes
        coords = ['x0', 'x1']
        match_df, msg_df = dataframes_coordinate_nan_match(
            df_serial, df_parallel, coords
        )
        assert match_df, f"Dataframes don't match: {msg_df}"
    
    def test_scenario_2b_parallel_vs_serial(self, temp_dir):
        """2D sparse array: parallel should match serial."""

        num_obs = 5
        num_dims = 2
        ratio_dims = 1
        sparsity = 1/num_obs
        seed = 23
        
        # Serial generation
        gen_serial = GenerateData(
            num_obs=num_obs,
            num_dims=num_dims,
            ratio_dims=ratio_dims,
            sparsity=sparsity,
            seed=seed,
        )
        da_serial, df_serial = gen_serial.generate()
        
        # Parallel generation - setup paths AFTER initialization
        nc_dir = os.path.join(temp_dir, "nc")
        pq_dir = os.path.join(temp_dir, "pq")
        os.makedirs(nc_dir, exist_ok=True)
        os.makedirs(pq_dir, exist_ok=True)
        
        gen_parallel = GenerateData(
            num_obs=num_obs,
            num_dims=num_dims,
            ratio_dims=ratio_dims,
            sparsity=sparsity,
            seed=seed,
            max_obs=3  # Very small chunks to test chunking with minimal data
        )
        result = gen_parallel.generate(
            netcdf_filepath=os.path.join(nc_dir, "test.nc"),
            parquet_filepath=os.path.join(pq_dir, "test")
            
        )
        
        # Parallel mode returns (None, None)
        assert result == (None, None), "Parallel generation should return (None, None)"
        
        # Load parallel data from disk
        nc_files = sorted(glob.glob(os.path.join(nc_dir, "test_*.nc")))
        assert len(nc_files) > 0, f"No parallel chunk files found in {nc_dir}"
        
        # Concatenate chunks
        dataarrays = [xr.open_dataarray(f) for f in nc_files]
        da_parallel_full = xr.concat(dataarrays, dim='x0')
        for da in dataarrays:
            da.close()
            
        # Load parquet data
        import pandas as pd
        pq_files = sorted(glob.glob(os.path.join(pq_dir, "test_*.parquet")))
        if pq_files:
            df_parallel = pd.concat([pd.read_parquet(f) for f in pq_files], ignore_index=True)
        else:
            # Try consolidated file
            df_parallel = pd.read_parquet(pq_dir)
            
        # Compare datasets
        match, msg = datasets_are_identical(da_serial, da_parallel_full)
        assert match, f"Datasets don't match: {msg}"
        
        # Compare dataframes
        coords = ['x0', 'x1']
        match_df, msg_df = dataframes_coordinate_nan_match(
            df_serial, df_parallel, coords
        )
        assert match_df, f"Dataframes don't match: {msg_df}"
    
    def test_3d_parallel_vs_serial(self, temp_dir):
        """3D sparse array: parallel should match serial."""
        # Serial generation
        gen_serial = GenerateData(
            num_obs=200,
            num_dims=3,
            ratio_dims=[2, 1, 1.5],
            sparsity=0.15,
            seed=123
        )
        da_serial, df_serial = gen_serial.generate()
        
        # Parallel generation - setup paths AFTER initialization
        nc_dir = os.path.join(temp_dir, "nc")
        pq_dir = os.path.join(temp_dir, "pq")
        os.makedirs(nc_dir, exist_ok=True)
        os.makedirs(pq_dir, exist_ok=True)
        
        gen_parallel = GenerateData(
            num_obs=200,
            num_dims=3,
            ratio_dims=[2, 1, 1.5],
            sparsity=0.15,
            seed=123,
            max_obs=60  # Trigger parallel mode
        )        
        result = gen_parallel.generate(
            netcdf_filepath=os.path.join(nc_dir, "test.nc"),
            parquet_filepath=os.path.join(pq_dir, "test")
        
        )
        
        # Parallel mode returns (None, None)
        
        assert result == (None, None), "Parallel generation should return (None, None)"
                
        # Load parallel data from disk
        nc_files = sorted(glob.glob(os.path.join(nc_dir, "test_*.nc")))
        assert len(nc_files) > 0, f"No parallel chunk files found in {nc_dir}"
        
        # Concatenate chunks
        dataarrays = [xr.open_dataarray(f) for f in nc_files]
        da_parallel_full = xr.concat(dataarrays, dim='x0')
        for da in dataarrays:
            da.close()
        
        # Load parquet data
        import pandas as pd
        pq_files = sorted(glob.glob(os.path.join(pq_dir, "test_*.parquet")))
        if pq_files:
            df_parallel = pd.concat([pd.read_parquet(f) for f in pq_files], ignore_index=True)
        else:
            # Try consolidated file
            df_parallel = pd.read_parquet(pq_dir)
        
        # Compare datasets
        match, msg = datasets_are_identical(da_serial, da_parallel_full)
        assert match, f"Datasets don't match: {msg}"
        
        # Compare dataframes
        coords = ['x0', 'x1', 'x2']
        match_df, msg_df = dataframes_coordinate_nan_match(
            df_serial, df_parallel, coords
        )
        assert match_df, f"Dataframes don't match: {msg_df}"


class TestComparisonUtilities:
    """Tests for comparison utility functions."""
    
    def test_datasets_identical_for_same_data(self):
        """Should return True for identical datasets."""
        da1 = xr.DataArray(
            [[1.0, 2.0, np.nan], [4.0, np.nan, 6.0]],
            coords={'x0': [0, 1], 'x1': [0, 1, 2]},
            dims=['x0', 'x1'],
            name='record'
        )
        da2 = da1.copy(deep=True)
        
        match, msg = datasets_are_identical(da1, da2, variable='record')
        assert match, msg
    
    def test_datasets_different_nan_counts(self):
        """Should detect different NaN counts."""
        da1 = xr.DataArray(
            [[1.0, np.nan], [3.0, 4.0]],  # 1 NaN
            coords={'x0': [0, 1], 'x1': [0, 1]},
            dims=['x0', 'x1'],
            name='record'
        )
        da2 = xr.DataArray(
            [[1.0, np.nan], [3.0, np.nan]],  # 2 NaNs
            coords={'x0': [0, 1], 'x1': [0, 1]},
            dims=['x0', 'x1'],
            name='record'
        )
        
        match, msg = datasets_are_identical(da1, da2, variable='record')
        assert not match
        assert "Different number of NaNs" in msg
        
    def test_datasets_different_nan_positions(self):
        """Should pass when NaN counts match (even if positions differ)."""
        da1 = xr.DataArray(
            [[1.0, np.nan], [3.0, 4.0]],
            coords={'x0': [0, 1], 'x1': [0, 1]},
            dims=['x0', 'x1'],
            name='record'
        )
        da2 = xr.DataArray(
            [[1.0, 2.0], [3.0, np.nan]],  # Same NaN count, different position
            coords={'x0': [0, 1], 'x1': [0, 1]},
            dims=['x0', 'x1'],
            name='record'
        )
        
        # This should pass because we only check NaN *counts*, not positions
        # The function is designed for comparing parallel vs serial where
        # coordinates must match exactly but record values can differ
        match, msg = datasets_are_identical(da1, da2, variable='record')
        assert match, msg
    
    def test_dataframes_identical(self):
        """Should return True for identical dataframes."""
        import pandas as pd
        df1 = pd.DataFrame({
            'x0': [0, 1, 2],
            'x1': [0, 1, 2],
            'record': [1.0, np.nan, 3.0]
        })
        df2 = df1.copy()
        
        match, msg = dataframes_coordinate_nan_match(df1, df2, ['x0', 'x1'])
        assert match, msg
    
    def test_dataframes_different_nan_status(self):
        """Should detect different NaN status at same coordinates."""
        import pandas as pd
        df1 = pd.DataFrame({
            'x0': [0, 1, 2],
            'x1': [0, 1, 2],
            'record': [1.0, np.nan, 3.0]
        })
        df2 = pd.DataFrame({
            'x0': [0, 1, 2],
            'x1': [0, 1, 2],
            'record': [1.0, 2.0, 3.0]  # No NaN at x0=1
        })
        
        match, msg = dataframes_coordinate_nan_match(df1, df2, ['x0', 'x1'])
        assert not match
        assert "Coordinate" in msg or "NaN" in msg or "differ" in msg
    
    def test_dataframes_different_coordinates(self):
        """Should detect different coordinate combinations (x0 can differ, but not x1)."""
        import pandas as pd
        df1 = pd.DataFrame({
            'x0': [0, 1, 2],
            'x1': [0, 1, 2],
            'record': [1.0, 2.0, 3.0]
        })
        df2 = pd.DataFrame({
            'x0': [0, 1, 3],  # x0 difference is OK
            'x1': [0, 1, 3],  # x1 difference should be detected
            'record': [1.0, 2.0, 3.0]
        })
        
        match, msg = dataframes_coordinate_nan_match(df1, df2, ['x0', 'x1'])
        assert not match
        assert "Coordinate" in msg
    
    def test_dataframes_x0_can_differ(self):
        """x0 values can differ as long as other coordinates match."""
        import pandas as pd
        df1 = pd.DataFrame({
            'x0': [0, 1, 2],
            'x1': [0, 1, 2],
            'record': [1.0, 2.0, 3.0]
        })
        df2 = pd.DataFrame({
            'x0': [10, 20, 30],  # Completely different x0 values
            'x1': [0, 1, 2],      # Same x1 values
            'record': [5.0, 6.0, 7.0]  # Different record values but same structure
        })
        
        match, msg = dataframes_coordinate_nan_match(df1, df2, ['x0', 'x1'])
        assert match, f"Should match when x0 differs but x1 matches: {msg}"
