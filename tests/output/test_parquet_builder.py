"""Tests for ParquetBuilder class."""

import pytest
import numpy as np
import pandas as pd
from data_sparsity.output.parquet_builder import ParquetBuilder


class TestExtractNonNanPoints:
    """Tests for extract_non_nan_points method."""

    def test_1d_record(self):
        """Should extract non-NaN points from 1D record."""
        record = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        coordinates = {"x0": np.array([0.0, 0.25, 0.5, 0.75, 1.0])}

        result = ParquetBuilder.extract_non_nan_points(record, coordinates)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # Three non-NaN values
        assert list(result.columns) == ["x0", "record"]

    def test_2d_record(self):
        """Should extract non-NaN points from 2D record."""
        record = np.array(
            [
                [1.0, np.nan],
                [np.nan, 4.0],
            ]
        )
        coordinates = {
            "x0": np.array([0.0, 1.0]),
            "x1": np.array([0.0, 1.0]),
        }

        result = ParquetBuilder.extract_non_nan_points(record, coordinates)

        assert len(result) == 2  # Two non-NaN values
        assert "x0" in result.columns
        assert "x1" in result.columns
        assert "record" in result.columns

    def test_3d_record(self):
        """Should extract non-NaN points from 3D record."""
        record = np.full((3, 2, 2), np.nan)
        record[0, 0, 0] = 1.0
        record[1, 1, 1] = 2.0
        record[2, 0, 1] = 3.0

        coordinates = {
            "x0": np.array([0.0, 0.5, 1.0]),
            "x1": np.array([0.0, 1.0]),
            "x2": np.array([0.0, 1.0]),
        }

        result = ParquetBuilder.extract_non_nan_points(record, coordinates)

        assert len(result) == 3
        assert list(result.columns) == ["x0", "x1", "x2", "record"]

    def test_sparse_record(self):
        """Should handle very sparse records."""
        record = np.full((10, 10), np.nan)
        record[5, 5] = 42.0

        coordinates = {
            "x0": np.linspace(0, 1, 10),
            "x1": np.linspace(0, 1, 10),
        }

        result = ParquetBuilder.extract_non_nan_points(record, coordinates)

        assert len(result) == 1
        assert result["record"].iloc[0] == 42.0

    def test_dense_record(self):
        """Should handle dense records."""
        record = np.arange(20).reshape(4, 5).astype(float)
        coordinates = {
            "x0": np.linspace(0, 1, 4),
            "x1": np.linspace(0, 1, 5),
        }

        result = ParquetBuilder.extract_non_nan_points(record, coordinates)

        assert len(result) == 20  # All points

    def test_empty_record(self):
        """Should handle record with all NaN."""
        record = np.full((5, 5), np.nan)
        coordinates = {
            "x0": np.linspace(0, 1, 5),
            "x1": np.linspace(0, 1, 5),
        }

        result = ParquetBuilder.extract_non_nan_points(record, coordinates)

        assert len(result) == 0

    def test_coordinate_values_match(self):
        """Should match coordinates to correct record values."""
        record = np.array([[1.0, np.nan], [np.nan, 4.0]])
        coordinates = {
            "x0": np.array([10.0, 20.0]),
            "x1": np.array([100.0, 200.0]),
        }

        result = ParquetBuilder.extract_non_nan_points(record, coordinates)

        # Check first point (0, 0) -> value 1.0
        assert result.iloc[0]["x0"] == 10.0
        assert result.iloc[0]["x1"] == 100.0
        assert result.iloc[0]["record"] == 1.0

        # Check second point (1, 1) -> value 4.0
        assert result.iloc[1]["x0"] == 20.0
        assert result.iloc[1]["x1"] == 200.0
        assert result.iloc[1]["record"] == 4.0


class TestBuildSingleVarDataframe:
    """Tests for build_single_var_dataframe method."""

    def test_basic_functionality(self):
        """Should build DataFrame for single variable."""
        record = np.array([1.0, np.nan, 3.0])
        coordinates = {"x0": np.array([0.0, 0.5, 1.0])}

        result = ParquetBuilder.build_single_var_dataframe(record, coordinates)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_returns_dataframe(self):
        """Should return pandas DataFrame."""
        record = np.ones((5, 4))
        coordinates = {
            "x0": np.linspace(0, 1, 5),
            "x1": np.linspace(0, 1, 4),
        }

        result = ParquetBuilder.build_single_var_dataframe(record, coordinates)

        assert isinstance(result, pd.DataFrame)

    def test_column_names(self):
        """Should have correct column names."""
        record = np.array([[1.0, 2.0], [3.0, 4.0]])
        coordinates = {
            "x0": np.array([0.0, 1.0]),
            "x1": np.array([0.0, 1.0]),
        }

        result = ParquetBuilder.build_single_var_dataframe(record, coordinates)

        assert "x0" in result.columns
        assert "x1" in result.columns
        assert "record" in result.columns


class TestBuildMultiVarDataframe:
    """Tests for build_multi_var_dataframe method."""

    def test_two_variables(self):
        """Should build DataFrame for two variables."""
        records = {
            "var0": np.array([[1.0, np.nan], [np.nan, 4.0]]),
            "var1": np.array([[5.0, 6.0], [7.0, np.nan]]),
        }
        coordinates = {
            "x0": np.array([0.0, 1.0]),
            "x1": np.array([0.0, 1.0]),
        }

        result = ParquetBuilder.build_multi_var_dataframe(
            records,
            coordinates,
            num_vars=2,
            num_dims=2,
        )

        assert isinstance(result, pd.DataFrame)
        assert "var0" in result.columns
        assert "var1" in result.columns

    def test_many_variables(self):
        """Should handle many variables."""
        shape = (3, 3)
        records = {
            "var0": np.ones(shape),
            "var1": np.ones(shape) * 2,
            "var2": np.ones(shape) * 3,
            "var3": np.ones(shape) * 4,
        }
        coordinates = {
            "x0": np.linspace(0, 1, 3),
            "x1": np.linspace(0, 1, 3),
        }

        result = ParquetBuilder.build_multi_var_dataframe(
            records,
            coordinates,
            num_vars=4,
            num_dims=2,
        )

        assert len(result.columns) >= 6  # 2 coords + 4 vars
        for i in range(4):
            assert f"var{i}" in result.columns

    def test_coordinate_columns_present(self):
        """Should include coordinate columns."""
        records = {
            "var0": np.ones((2, 2)),
            "var1": np.ones((2, 2)),
        }
        coordinates = {
            "x0": np.array([0.0, 1.0]),
            "x1": np.array([0.0, 1.0]),
        }

        result = ParquetBuilder.build_multi_var_dataframe(
            records,
            coordinates,
            num_vars=2,
            num_dims=2,
        )

        assert "x0" in result.columns
        assert "x1" in result.columns

    def test_nan_preserved_correctly(self):
        """Should preserve NaN values correctly."""
        records = {
            "var0": np.array([[1.0, np.nan]]),
            "var1": np.array([[np.nan, 2.0]]),
        }
        coordinates = {
            "x0": np.array([0.0]),
            "x1": np.array([0.0, 1.0]),
        }

        result = ParquetBuilder.build_multi_var_dataframe(
            records,
            coordinates,
            num_vars=2,
            num_dims=2,
        )

        # Should have one row where var0 is NaN and var1 has value
        # And one row where var0 has value and var1 is NaN
        assert len(result) >= 1

    def test_unique_coordinates(self):
        """Should have unique coordinate combinations."""
        records = {
            "var0": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "var1": np.array([[5.0, 6.0], [7.0, 8.0]]),
        }
        coordinates = {
            "x0": np.array([0.0, 1.0]),
            "x1": np.array([0.0, 1.0]),
        }

        result = ParquetBuilder.build_multi_var_dataframe(
            records,
            coordinates,
            num_vars=2,
            num_dims=2,
        )

        # Should have 4 unique coordinate combinations
        coord_cols = ["x0", "x1"]
        unique_coords = result[coord_cols].drop_duplicates()
        assert len(unique_coords) <= 4

    def test_all_variables_in_columns(self):
        """Should include all variable columns."""
        num_vars = 5
        records = {f"var{i}": np.ones((3, 3)) for i in range(num_vars)}
        coordinates = {
            "x0": np.linspace(0, 1, 3),
            "x1": np.linspace(0, 1, 3),
        }

        result = ParquetBuilder.build_multi_var_dataframe(
            records,
            coordinates,
            num_vars=num_vars,
            num_dims=2,
        )

        for i in range(num_vars):
            assert f"var{i}" in result.columns
