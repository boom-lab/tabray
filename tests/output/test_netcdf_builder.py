"""Tests for NetCDFBuilder class."""

import pytest
import numpy as np
import xarray as xr
from data_sparsity.output.netcdf_builder import NetCDFBuilder


class TestCreateDefaultAttrs:
    """Tests for create_default_attrs method."""
    
    def test_all_fields_present(self):
        """Should include all required fields."""
        attrs = NetCDFBuilder.create_default_attrs(
            num_obs=100,
            num_dims=3,
            ratio_dims=np.array([1, 2, 1]),
            density=0.5,
            seed=42
        )
        
        assert "description" in attrs
        assert "num_obs" in attrs
        assert "num_dims" in attrs
        assert "ratio_dims" in attrs
        assert "density" in attrs
        assert "density" in attrs
        assert "seed" in attrs
    
    def test_correct_values(self):
        """Should store correct values."""
        attrs = NetCDFBuilder.create_default_attrs(
            num_obs=150,
            num_dims=2,
            ratio_dims=np.array([1, 1]),
            density=0.3,
            seed=999
        )
        
        assert attrs["num_obs"] == 150
        assert attrs["num_dims"] == 2
        assert attrs["ratio_dims"] == [1, 1]
        assert attrs["density"] == 0.3
        assert attrs["seed"] == 999
    
    def test_ratio_dims_as_array(self):
        """Should handle ratio_dims as numpy array."""
        ratio_dims = np.array([1, 2, 3])
        attrs = NetCDFBuilder.create_default_attrs(
            num_obs=100, num_dims=3, ratio_dims=ratio_dims,
            density=0.5, seed=42
        )
        
        assert isinstance(attrs["ratio_dims"], list)
        assert attrs["ratio_dims"] == [1, 2, 3]
    
    def test_ratio_dims_as_list(self):
        """Should handle ratio_dims as list."""
        ratio_dims = [2, 3, 1]
        attrs = NetCDFBuilder.create_default_attrs(
            num_obs=100, num_dims=3, ratio_dims=ratio_dims,
            density=0.5, seed=42
        )
        
        assert attrs["ratio_dims"] == [2, 3, 1]
    
    def test_density_as_float(self):
        """Should convert density to float."""
        attrs = NetCDFBuilder.create_default_attrs(
            num_obs=100, num_dims=2, ratio_dims=[1, 1],
            density=0.5, seed=42
        )
        
        assert isinstance(attrs["density"], float)


class TestBuildDataarray:
    """Tests for build_dataarray method."""
    
    def test_1d_record(self):
        """Should build DataArray from 1D record."""
        record = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        coordinates = {'x0': np.array([0.0, 0.25, 0.5, 0.75, 1.0])}
        
        result = NetCDFBuilder.build_dataarray(record, coordinates)
        
        assert isinstance(result, xr.DataArray)
        assert result.dims == ('x0',)
    
    def test_2d_record(self):
        """Should build DataArray from 2D record."""
        record = np.ones((5, 4))
        coordinates = {
            'x0': np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            'x1': np.array([0.0, 0.33, 0.67, 1.0])
        }
        
        result = NetCDFBuilder.build_dataarray(record, coordinates)
        
        assert isinstance(result, xr.DataArray)
        assert result.dims == ('x0', 'x1')
        assert result.shape == (5, 4)
    
    def test_3d_record(self):
        """Should build DataArray from 3D record."""
        record = np.ones((3, 4, 5))
        coordinates = {
            'x0': np.array([0.0, 0.5, 1.0]),
            'x1': np.array([0.0, 0.33, 0.67, 1.0]),
            'x2': np.array([0.0, 0.2, 0.4, 0.6, 0.8])
        }
        
        result = NetCDFBuilder.build_dataarray(record, coordinates)
        
        assert result.dims == ('x0', 'x1', 'x2')
        assert result.shape == (3, 4, 5)
    
    def test_custom_var_name(self):
        """Should use custom variable name."""
        record = np.ones((5, 4))
        coordinates = {
            'x0': np.linspace(0, 1, 5),
            'x1': np.linspace(0, 1, 4)
        }
        
        result = NetCDFBuilder.build_dataarray(
            record, coordinates, var_name="temperature"
        )
        
        assert result.name == "temperature"
    
    def test_default_var_name(self):
        """Should use default variable name."""
        record = np.ones((5, 4))
        coordinates = {
            'x0': np.linspace(0, 1, 5),
            'x1': np.linspace(0, 1, 4)
        }
        
        result = NetCDFBuilder.build_dataarray(record, coordinates)
        
        assert result.name == "record"
    
    def test_with_attributes(self):
        """Should include provided attributes."""
        record = np.ones((5,))
        coordinates = {'x0': np.linspace(0, 1, 5)}
        attrs = {"description": "test data", "units": "meters"}
        
        result = NetCDFBuilder.build_dataarray(record, coordinates, attrs=attrs)
        
        assert result.attrs["description"] == "test data"
        assert result.attrs["units"] == "meters"
    
    def test_without_attributes(self):
        """Should work without attributes."""
        record = np.ones((5,))
        coordinates = {'x0': np.linspace(0, 1, 5)}
        
        result = NetCDFBuilder.build_dataarray(record, coordinates)
        
        assert isinstance(result.attrs, dict)
    
    def test_coordinate_values_preserved(self):
        """Should preserve coordinate values."""
        record = np.ones((3,))
        coordinates = {'x0': np.array([0.1, 0.5, 0.9])}
        
        result = NetCDFBuilder.build_dataarray(record, coordinates)
        
        np.testing.assert_array_equal(result.coords['x0'].values, [0.1, 0.5, 0.9])
    
    def test_data_values_preserved(self):
        """Should preserve data values including NaN."""
        record = np.array([1.0, np.nan, 3.0])
        coordinates = {'x0': np.array([0.0, 0.5, 1.0])}
        
        result = NetCDFBuilder.build_dataarray(record, coordinates)
        
        np.testing.assert_array_equal(result.values, record)


class TestBuildDataset:
    """Tests for build_dataset method."""
    
    def test_single_variable(self):
        """Should build Dataset from single variable."""
        records = {'var0': np.ones((5, 4))}
        coordinates = {
            'x0': np.linspace(0, 1, 5),
            'x1': np.linspace(0, 1, 4)
        }
        
        result = NetCDFBuilder.build_dataset(records, coordinates)
        
        assert isinstance(result, xr.Dataset)
        assert 'var0' in result.data_vars
    
    def test_multiple_variables(self):
        """Should build Dataset from multiple variables."""
        records = {
            'var0': np.ones((5, 4)),
            'var1': np.ones((5, 4)) * 2,
            'var2': np.ones((5, 4)) * 3
        }
        coordinates = {
            'x0': np.linspace(0, 1, 5),
            'x1': np.linspace(0, 1, 4)
        }
        
        result = NetCDFBuilder.build_dataset(records, coordinates)
        
        assert len(result.data_vars) == 3
        assert 'var0' in result
        assert 'var1' in result
        assert 'var2' in result
    
    def test_shared_coordinates(self):
        """Should share coordinates across variables."""
        records = {
            'var0': np.ones((5, 4)),
            'var1': np.ones((5, 4))
        }
        coordinates = {
            'x0': np.linspace(0, 1, 5),
            'x1': np.linspace(0, 1, 4)
        }
        
        result = NetCDFBuilder.build_dataset(records, coordinates)
        
        np.testing.assert_array_equal(
            result['var0'].coords['x0'].values,
            result['var1'].coords['x0'].values
        )
    
    def test_with_attributes(self):
        """Should include dataset-level attributes."""
        records = {'var0': np.ones((5, 4))}
        coordinates = {
            'x0': np.linspace(0, 1, 5),
            'x1': np.linspace(0, 1, 4)
        }
        attrs = {"title": "Test Dataset", "version": "1.0"}
        
        result = NetCDFBuilder.build_dataset(records, coordinates, attrs=attrs)
        
        assert result.attrs["title"] == "Test Dataset"
        assert result.attrs["version"] == "1.0"
    
    def test_without_attributes(self):
        """Should work without attributes."""
        records = {'var0': np.ones((5, 4))}
        coordinates = {
            'x0': np.linspace(0, 1, 5),
            'x1': np.linspace(0, 1, 4)
        }
        
        result = NetCDFBuilder.build_dataset(records, coordinates)
        
        assert isinstance(result.attrs, dict)
    
    def test_preserves_data_values(self):
        """Should preserve data values for all variables."""
        records = {
            'var0': np.array([[1.0, np.nan], [3.0, 4.0]]),
            'var1': np.array([[5.0, 6.0], [np.nan, 8.0]])
        }
        coordinates = {
            'x0': np.array([0.0, 1.0]),
            'x1': np.array([0.0, 1.0])
        }
        
        result = NetCDFBuilder.build_dataset(records, coordinates)
        
        np.testing.assert_array_equal(
            result['var0'].values, records['var0']
        )
        np.testing.assert_array_equal(
            result['var1'].values, records['var1']
        )
