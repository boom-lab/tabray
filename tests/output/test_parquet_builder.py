"""Tests for ParquetBuilder class."""

import pytest
import numpy as np
import pandas as pd
from tabray.output.parquet_builder import ParquetBuilder


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


class TestRepeatFewerDims:
    """A variable on fewer dims repeats across the dims it drops."""

    @staticmethod
    def build(var_constant_dims):
        """var0 full; var1, var2 on (x0, x1); var3 on x0.

        - var1 (0, 0) -> 5 repeats on both var0 rows at (0, 0); (1, 1) -> 6
          matches no var0 row -> row (1, 1, NaN)
        - var2 (1, 1) -> 7 shares that row
        - var3 x0 = 1 -> 8 fills every row at x0 = 1
        """
        var0 = np.full((2, 2, 2), np.nan)
        var0[0, 0, 0], var0[0, 0, 1], var0[1, 0, 0] = 1.0, 2.0, 3.0
        var1 = np.full((2, 2, 2), np.nan)
        var1[0, 0, 1], var1[1, 1, 1] = 5.0, 6.0
        var2 = np.full((2, 2, 2), np.nan)
        var2[1, 1, 0] = 7.0
        var3 = np.full((2, 2, 2), np.nan)
        var3[1, 0, 0] = 8.0
        records = {"var0": var0, "var1": var1, "var2": var2, "var3": var3}
        coordinates = {
            "x0": np.array([0.0, 1.0]),
            "x1": np.array([10.0, 20.0]),
            "x2": np.array([100.0, 200.0]),
        }
        return ParquetBuilder.build_multi_var_dataframe(
            records, coordinates, 4, 3, var_constant_dims=var_constant_dims
        )

    def test_values_repeat_and_unmatched_cells_get_nan_coordinates(self):
        nan = np.nan
        expected = pd.DataFrame(
            {
                "x0": [0.0, 0.0, 1.0, 1.0],
                "x1": [10.0, 10.0, 10.0, 20.0],
                "x2": [100.0, 200.0, 100.0, nan],
                "var0": [1.0, 2.0, 3.0, nan],
                "var1": [5.0, 5.0, nan, 6.0],
                "var2": [nan, nan, nan, 7.0],
                "var3": [nan, nan, 8.0, 8.0],
            }
        )
        result = self.build([[], [2], [2], [1, 2]])
        pd.testing.assert_frame_equal(result, expected)

    def test_no_constant_dims_keeps_one_row_per_cell(self):
        """None and all-empty give the full-dims layout."""
        pd.testing.assert_frame_equal(self.build(None), self.build([[]] * 4))
        assert len(self.build(None)) == 5  # 3 var0 cells + (1, 1, 0), (1, 1, 1)
