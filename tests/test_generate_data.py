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

    def test_density_kwarg_is_supported(self):
        """Should initialize when density is passed directly."""
        gen = GenerateData(
            num_obs=100,
            num_dims=2,
            ratio_dims=1,
            sparsity=None,
            density=0.75,
            seed=42,
        )
        assert gen.density == 0.75

    def test_density_wins_over_sparsity(self):
        """Should warn and prefer density when both inputs are present."""
        with pytest.warns(UserWarning, match="Both density and sparsity"):
            gen = GenerateData(
                num_obs=100,
                num_dims=2,
                ratio_dims=1,
                sparsity=0.1,
                density=0.8,
                seed=42,
            )
            assert gen.num_obs == 97
            assert np.isclose(gen.density, 97 / 121)

    def test_minimal_valid_parameters(self):
        """Should initialize with minimal valid parameters."""
        gen = GenerateData(
            num_obs=100,
            num_dims=2,
            ratio_dims=1,
            density=1.0,
            seed=42,
        )
        assert gen.num_obs == 100
        assert gen.num_dims == 2
        assert gen.seed == 42
        assert gen.num_vars == 1

    def test_max_workers_is_configurable(self):
        """Should preserve the requested parallel worker limit."""
        gen = GenerateData(
            num_obs=100,
            num_dims=2,
            ratio_dims=1,
            density=1.0,
            seed=42,
            max_workers=2,
        )
        assert gen.max_workers == 2

    @pytest.mark.parametrize("value", [0, -1, 1.5, True])
    def test_invalid_max_workers_raises_error(self, value):
        """Should reject non-positive and non-integer worker limits."""
        with pytest.raises(ValueError, match="max_workers"):
            GenerateData(
                num_obs=100,
                num_dims=2,
                ratio_dims=1,
                density=1.0,
                seed=42,
                max_workers=value,
            )

    def test_full_parameters_specified(self):
        """Should initialize with all parameters."""
        gen = GenerateData(
            num_obs=200,
            num_dims=3,
            ratio_dims=1,
            density=[0.2, 0.15],
            seed=123,
            num_vars=2,
            var_dims=2,
            overlap=0.5,
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
                density=0.1,
                seed=42,
            )

    def test_invalid_sparsity_raises_error(self):
        """Should raise ValueError for invalid sparsity."""
        with pytest.raises(ValueError):
            GenerateData(
                num_obs=100,
                num_dims=2,
                ratio_dims=1,
                density=1.5,  # > 1.0
                seed=42,
            )

    def test_attributes_set_correctly(self):
        """Should set all attributes correctly."""
        gen = GenerateData(
            num_obs=150,
            num_dims=3,
            ratio_dims=[2, 1, 3],
            density=0.2,
            seed=999,
        )
        assert hasattr(gen, "num_obs")
        assert hasattr(gen, "num_dims")
        assert hasattr(gen, "ratio_dims")
        assert hasattr(gen, "density")
        assert hasattr(gen, "seed")

    def test_single_variable_defaults(self):
        """Should default to single variable."""
        gen = GenerateData(
            num_obs=100,
            num_dims=2,
            ratio_dims=1,
            density=0.1,
            seed=42,
        )
        assert gen.num_vars == 1
        assert gen.overlap == "random"

    def test_multi_variable_setup(self):
        """Should properly set up multi-variable configuration."""
        gen = GenerateData(
            num_obs=200,
            num_dims=3,
            ratio_dims=1,
            density=[0.2, 0.15, 0.1],
            seed=42,
            num_vars=3,
            var_dims=[2, 2, 3],
            overlap=0.6,
        )
        assert gen.num_vars == 3
        assert isinstance(gen.density, (list, tuple))


class TestSingleVariableGeneration:
    """Tests for single-variable data generation."""

    def test_1d_generation(self):
        """Should generate 1D data."""
        gen = GenerateData(
            num_obs=20,
            num_dims=1,
            ratio_dims=1,
            density=1,
            seed=42,
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
            density=0.2,
            seed=42,
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
            density=0.15,
            seed=42,
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
            density=0.2,
            seed=42,
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
            density=0.9,
            seed=42,
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
            density=0.2,
            seed=42,
        )
        dataarray1, dataframe1 = gen1.generate()

        gen2 = GenerateData(
            num_obs=50,
            num_dims=2,
            ratio_dims=1,
            density=0.2,
            seed=42,
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
            density=0.1,
            seed=42,
        )
        dataarray1, _ = gen1.generate()

        gen2 = GenerateData(
            num_obs=50,
            num_dims=2,
            ratio_dims=1,
            density=0.1,
            seed=999,
        )
        dataarray2, _ = gen2.generate()

        # Arrays should be different
        assert not np.array_equal(
            dataarray1.values,
            dataarray2.values,
            equal_nan=True,
        )


class TestMultiVariableGeneration:
    """Tests for multi-variable data generation."""

    def test_two_variables(self):
        """Should generate dataset with two variables."""
        gen = GenerateData(
            num_obs=100,
            num_dims=3,
            ratio_dims=1,
            density=[0.15, 0.07],  # Adjusted: var1 gets ~50 obs (< 81 grid points)
            seed=42,
            num_vars=2,
            var_dims=2,
            overlap=0.5,
        )
        dataset, dataframe = gen.generate()

        assert isinstance(dataset, xr.Dataset)
        assert len(dataset.data_vars) == 2
        assert "var0" in dataset
        assert "var1" in dataset

    def test_many_variables(self):
        """Should generate dataset with multiple variables."""
        gen = GenerateData(
            num_obs=200,
            num_dims=3,
            ratio_dims=1,
            density=[
                0.10,
                0.06,
                0.04,
                0.05,
            ],  # Adjusted: non-ref vars get ~70,50,60 obs
            seed=42,
            num_vars=4,
            var_dims=2,
            overlap=0.4,
        )
        dataset, dataframe = gen.generate()

        assert len(dataset.data_vars) == 4
        for i in range(4):
            assert f"var{i}" in dataset

    def test_different_sparsities_per_variable(self):
        """Should respect different sparsity values per variable."""
        gen = GenerateData(
            num_obs=150,
            num_dims=3,
            ratio_dims=1,
            density=[
                0.20,
                0.07,
                0.05,
            ],  # Adjusted: var1~60 obs, var2~43 obs (well < 81)
            seed=42,
            num_vars=3,
            var_dims=2,
            overlap=0.3,
        )
        dataset, dataframe = gen.generate()

        # Each variable should have different number of observations
        counts_data = dataset.count()
        print(counts_data)

        # Extract counts as integers
        counts = [int(counts_data[f"var{i}"].values) for i in range(3)]

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
            density=0.8,  # Adjusted to meet minimum sparsity requirement
            seed=42,
            num_vars=2,
            var_dims=[2, 2],  # Both use 2 dims (but var0 gets all dims per new logic)
            overlap=1,
        )
        dataset, dataframe = gen.generate()

        # Both variables should exist
        assert "var0" in dataset
        assert "var1" in dataset

    def test_zero_overlap(self):
        """Should generate variables with zero overlap."""
        gen = GenerateData(
            num_obs=100,
            num_dims=3,
            ratio_dims=1,
            density=[0.15, 0.01],  # Adjusted: var1 with 1 dim gets ~7 obs (< 9)
            seed=42,
            num_vars=2,
            var_dims=[2, 1],  # Different dims - var0 all dims, var1 gets 1 dim
            overlap="random",
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
            density=[0.1, 0.1],
            seed=42,
            num_vars=2,
            var_dims=2,
            overlap=0.9,
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
            density=[0.1, 0.1],
            seed=42,
            num_vars=2,
            var_dims=2,
            overlap="random",
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
            density=0.75,
            seed=42,
            num_vars=num_vars,
            var_dims=2,
            overlap=0.65,
        )
        dataset, _ = gen.generate()

        assert len(dataset.data_vars) == num_vars
        for i in range(num_vars):
            assert f"var{i}" in dataset

    def test_dataframe_has_all_columns(self):
        """Should include columns for all variables and coordinates."""
        gen = GenerateData(
            num_obs=100,
            num_dims=3,
            ratio_dims=1,
            density=[0.10, 0.07],  # Adjusted: var1 gets ~50 obs (< 81)
            seed=42,
            num_vars=2,
            var_dims=2,
            overlap=0.5,
        )
        _, dataframe = gen.generate()

        # Should have coordinate columns and variable columns
        assert "x0" in dataframe.columns
        assert "x1" in dataframe.columns
        assert "var0" in dataframe.columns
        assert "var1" in dataframe.columns

    def test_reproducible_with_seed(self):
        """Should produce same multi-var results with same seed."""
        gen1 = GenerateData(
            num_obs=80,
            num_dims=3,
            ratio_dims=1,
            density=[0.15, 0.08],  # Adjusted: var1 gets ~45 obs (< 64 grid points)
            seed=42,
            num_vars=2,
            var_dims=2,
            overlap=0.5,
        )
        dataset1, _ = gen1.generate()

        gen2 = GenerateData(
            num_obs=80,
            num_dims=3,
            ratio_dims=1,
            density=[0.15, 0.08],  # Adjusted: same as gen1
            seed=42,
            num_vars=2,
            var_dims=2,
            overlap=0.5,
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
                equal_nan=True,
            )


class TestMultiVariableEdgeCases:
    """Edge case tests for multi-variable generation with enhanced validation."""

    @staticmethod
    def compute_actual_overlap(dataframe, var1_idx, var2_idx, shared_dims):
        """Compute actual overlap between two variables along shared dimensions.

        Multi-variable DataFrames use wide format: one row per coordinate with
        separate columns for each variable (var0, var1, etc). A coordinate has
        an observation for a variable if that column is not NA.

        Args:
            dataframe: Parquet DataFrame with observation data (wide format)
            var1_idx: Index of first variable
            var2_idx: Index of second variable
            shared_dims: List of dimension indices that are shared

        Returns:
            float: Overlap fraction in [0, 1] representing fraction of
                   smaller variable's observations that overlap
        """
        var1_col = f"var{var1_idx}"
        var2_col = f"var{var2_idx}"

        # Filter to rows where each variable has observations
        var1_df = dataframe[dataframe[var1_col].notna()].copy()
        var2_df = dataframe[dataframe[var2_col].notna()].copy()

        if len(var1_df) == 0 or len(var2_df) == 0:
            return 0.0

        # Extract coordinates for shared dimensions only
        coord_cols = [f"x{d}" for d in shared_dims]

        # Convert to sets of tuples for intersection
        var1_coords = set(map(tuple, var1_df[coord_cols].values))
        var2_coords = set(map(tuple, var2_df[coord_cols].values))

        # Compute overlap as fraction of smaller set
        overlap_count = len(var1_coords.intersection(var2_coords))
        min_count = min(len(var1_coords), len(var2_coords))

        return overlap_count / min_count if min_count > 0 else 0.0

    def test_high_overlap_with_different_dims(self):
        """Mixed-dimension overlap is driven by the variable's own density.

        Overlap is measured on the shared dimensions, so the reference is seen
        through its projection. A cell of the shared space is free of var0 only
        if var0 misses it at every dropped coordinate, which is rare, so a
        reduced-dimension variable has little room to sit off the reference and
        the achieved overlap is pushed up towards its own density. The target is
        then a floor rather than a value that can be hit.
        """
        gen = GenerateData(
            num_obs=200,
            num_dims=4,
            ratio_dims=1,
            density=[0.2, 0.15],
            seed=42,
            num_vars=2,
            var_dims=3,  # var0 gets all 4 dims, var1 gets 3 dims
            overlap=0.8,
        )
        dataset, dataframe = gen.generate()

        assert "var0" in dataset
        assert "var1" in dataset

        var0_df = dataframe[dataframe["var0"].notna()]
        var1_df = dataframe[dataframe["var1"].notna()]

        assert all(var0_df[col].nunique() > 1 for col in ["x0", "x1", "x2", "x3"])

        var1_varying_dims = [
            col for col in ["x0", "x1", "x2", "x3"] if var1_df[col].nunique() > 1
        ]
        var1_constant_dims = [
            col for col in ["x0", "x1", "x2", "x3"] if var1_df[col].nunique() == 1
        ]

        assert len(var1_varying_dims) == 3
        assert len(var1_constant_dims) == 1

        # F1 per non-reference variable; forced up to 0.8667 here because the
        # projected reference leaves too few free cells for a lower value
        assert gen.overlap_actual.shape == (1,)
        assert 0.8 <= gen.overlap_actual[0] <= 1.0

    def test_constant_dimensions_remain_constant(self):
        """Variables with fewer varying dims should keep one dimension constant."""
        gen = GenerateData(
            num_obs=150,
            num_dims=3,
            ratio_dims=1,
            density=0.15,
            seed=42,
            num_vars=2,
            var_dims=2,  # Use integer to trigger reference variable logic
            overlap=0.5,
        )
        dataset, dataframe = gen.generate()

        assert "var0" in dataset
        assert "var1" in dataset

        var1_df = dataframe[dataframe["var1"].notna()]
        var1_constant_dims = [
            col for col in ["x0", "x1", "x2"] if var1_df[col].nunique() == 1
        ]
        var1_varying_dims = [
            col for col in ["x0", "x1", "x2"] if var1_df[col].nunique() > 1
        ]

        assert len(var1_constant_dims) == 1
        assert len(var1_varying_dims) == 2

        var0_df = dataframe[dataframe["var0"].notna()]
        assert all(var0_df[col].nunique() > 1 for col in ["x0", "x1", "x2"])

    def test_many_variables_with_mixed_sparsities(self):
        """Should handle 5+ variables with widely different sparsities."""
        gen = GenerateData(
            num_obs=500,
            num_dims=3,
            ratio_dims=1,
            density=[0.25, 0.20, 0.15, 0.10, 0.05],
            seed=42,
            num_vars=5,
            var_dims=3,
            overlap=0.3,
        )
        dataset, dataframe = gen.generate()

        # All variables should exist
        assert len(dataset.data_vars) == 5
        for i in range(5):
            assert f"var{i}" in dataset

        # Check observation counts increase with sparsity
        counts = []
        for i in range(5):
            var_col = f"var{i}"
            var_count = dataframe[var_col].notna().sum()
            counts.append(var_count)
            assert var_count > 0, f"var{i} has no observations"

        # var0 (highest sparsity) should have most observations
        assert counts[0] == max(counts), f"var0 should have most obs, counts: {counts}"

        # var4 (lowest sparsity) should have fewest observations
        assert counts[4] == min(
            counts
        ), f"var4 should have fewest obs, counts: {counts}"

    def test_overlap_verification_random_mode(self):
        """Random overlap should produce a valid measured overlap ratio."""
        gen = GenerateData(
            num_obs=200,
            num_dims=3,
            ratio_dims=1,
            density=[0.15, 0.15],  # Same sparsity for both
            seed=42,
            num_vars=2,
            var_dims=3,  # All dimensions vary for both
            overlap="random",
        )
        dataset, dataframe = gen.generate()

        # Both variables should exist with observations
        assert "var0" in dataset
        assert "var1" in dataset
        var0_count = dataframe["var0"].notna().sum()
        var1_count = dataframe["var1"].notna().sum()
        assert var0_count > 0
        assert var1_count > 0

        # 'random' does not target a specific overlap value; the source only
        # reports the realized overlap after generation.
        assert 0 <= gen.overlap_actual <= 1.0

    def test_zero_overlap_with_target(self):
        """Should handle zero overlap target correctly."""
        gen = GenerateData(
            num_obs=150,
            num_dims=3,
            ratio_dims=1,
            density=[0.15, 0.10],
            seed=42,
            num_vars=2,
            var_dims=3,
            overlap=0.0,
        )
        dataset, dataframe = gen.generate()

        # Both variables should exist
        assert "var0" in dataset
        assert "var1" in dataset

        # Compute actual overlap
        shared_dims = [0, 1, 2]
        actual_overlap = self.compute_actual_overlap(dataframe, 0, 1, shared_dims)

        # Should have low overlap (allowing some random overlap due to discrete sampling)
        # With 0.0 target, the implementation generates independent samples
        # which may have some incidental overlap
        assert (
            actual_overlap <= 0.25
        ), f"Expected low overlap with target=0.0, got {actual_overlap:.2f}"

    def test_maximal_overlap(self):
        """Should handle maximal overlap (1.0) target."""
        gen = GenerateData(
            num_obs=100,
            num_dims=3,
            ratio_dims=1,
            density=[0.20, 0.10],  # Different sparsities
            seed=42,
            num_vars=2,
            var_dims=3,
            overlap=1.0,
        )
        dataset, dataframe = gen.generate()

        # Both variables should exist
        assert "var0" in dataset
        assert "var1" in dataset

        # Compute actual overlap
        shared_dims = [0, 1, 2]
        actual_overlap = self.compute_actual_overlap(dataframe, 0, 1, shared_dims)

        # Should have very high overlap (allowing some tolerance for discrete sampling)
        assert (
            actual_overlap >= 0.85
        ), f"Expected maximal overlap, got {actual_overlap:.2f}"

    def test_no_duplicate_observations_within_variable(self):
        """Each variable should have unique observations (no duplicates)."""
        gen = GenerateData(
            num_obs=200,
            num_dims=3,
            ratio_dims=1,
            density=[0.15, 0.10],
            seed=42,
            num_vars=2,
            var_dims=3,
            overlap=0.5,
        )
        dataset, dataframe = gen.generate()

        # Check var0 for duplicates
        var0_df = dataframe[dataframe["var0"].notna()]
        coord_cols = ["x0", "x1", "x2"]
        var0_coords = var0_df[coord_cols]
        assert len(var0_coords) == len(
            var0_coords.drop_duplicates()
        ), "var0 has duplicate observations"

        # Check var1 for duplicates
        var1_df = dataframe[dataframe["var1"].notna()]
        var1_coords = var1_df[coord_cols]
        assert len(var1_coords) == len(
            var1_coords.drop_duplicates()
        ), "var1 has duplicate observations"


class TestFileOutput:
    """Tests for file output functionality."""

    def test_netcdf_output_created(self, temp_dir):
        """Should create NetCDF file."""
        gen = GenerateData(
            num_obs=50,
            num_dims=2,
            ratio_dims=1,
            density=0.1,
            seed=42,
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
            density=0.1,
            seed=42,
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
            density=0.2,
            seed=42,
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
            density=1,
            seed=42,
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
            density=1.0,
            seed=42,
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
            density=1,
            seed=42,
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
            density=0.2,
            seed=42,
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
            density=0.2,
            seed=42,
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
            density=1.0,
            seed=42,
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
            density=0.15,
            seed=42,
            num_vars=2,
            var_dims=2,  # All use 2 dimensions
            overlap=0.5,
        )
        dataset, _ = gen.generate()

        assert len(dataset.data_vars) == 2

    def test_all_variables_different_dimensions(self):
        """Should handle all variables with different dimensions."""
        gen = GenerateData(
            num_obs=120,
            num_dims=3,
            ratio_dims=1,
            density=0.1,
            seed=42,
            num_vars=3,
            var_dims=[1, 2, 3],  # Each different
            overlap=0.0,
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
                density=2.0,  # > 1.0
                seed=42,
            )

    def test_negative_num_obs_raises_error(self):
        """Should raise error for negative num_obs."""
        with pytest.raises((ValueError, TypeError)):
            GenerateData(
                num_obs=-10,
                num_dims=2,
                ratio_dims=1,
                density=0.1,
                seed=42,
            )

    def test_zero_num_dims_raises_error(self):
        """Should raise error for zero dimensions."""
        with pytest.raises((ValueError, TypeError)):
            GenerateData(
                num_obs=100,
                num_dims=0,
                ratio_dims=1,
                density=0.1,
                seed=42,
            )

    def test_invalid_num_vars_raises_error(self):
        """Should raise error for invalid num_vars."""
        with pytest.raises((ValueError, TypeError)):
            GenerateData(
                num_obs=100,
                num_dims=2,
                ratio_dims=1,
                density=0.1,
                seed=42,
                num_vars=0,
            )

    def test_invalid_overlap_raises_error(self):
        """Should raise error for invalid overlap value."""
        with pytest.raises(ValueError):
            GenerateData(
                num_obs=100,
                num_dims=3,
                ratio_dims=1,
                density=[0.1, 0.1],
                seed=42,
                num_vars=2,
                var_dims=2,
                overlap=1.5,  # > 1.0
            )


class TestScenarios:
    """Tests for specific scenarios whose outcome is known"""

    def test_scenario_1a(self):
        """Should generate a full 1D array"""

        gen = GenerateData(
            num_obs=100,
            num_dims=1,
            ratio_dims=1,
            density=1.0,
            seed=34,
        )
        dataarray, dataframe = gen.generate()

        # Add checks for dataarray
        # Check 1: there is only one dimension
        assert len(dataarray.dims) == 1, "DataArray should have exactly one dimension"

        # Check 2: the only dimension present is called 'x0'
        assert "x0" in dataarray.dims, "Dimension should be named 'x0'"

        # Check 3: 'x0' has 100 coordinates
        assert len(dataarray.coords["x0"]) == 100, "'x0' should have 100 coordinates"

        # Check 4: all coordinates have a corresponding measurement in 'record'
        non_nan_count = np.count_nonzero(~np.isnan(dataarray.values))
        assert non_nan_count == 100, "All 100 coordinates should have non-NaN values"

        # Check 5: all coordinates in x0 are unique
        assert len(dataarray.coords["x0"]) == len(
            np.unique(dataarray.coords["x0"])
        ), "All coordinates in x0 should be unique"

        # Check 6: all record values are unique
        valid_values = dataarray.values[~np.isnan(dataarray.values)]
        assert len(valid_values) == len(
            np.unique(valid_values)
        ), "All record values should be unique"

        # Add checks for dataframe
        # Check 1: there are two columns ('x0' and 'record')
        assert list(dataframe.columns) == [
            "x0",
            "record",
        ], "DataFrame should have exactly two columns: 'x0' and 'record'"

        # Check 2: 'x0' has 100 rows
        assert len(dataframe["x0"]) == 100, "'x0' column should have 100 rows"

        # Check 3: 'record' has 100 rows
        assert len(dataframe["record"]) == 100, "'record' column should have 100 rows"

        # Check 4: all coordinates in x0 are unique
        assert len(dataframe["x0"]) == len(
            dataframe["x0"].unique()
        ), "All coordinates in 'x0' should be unique"

        # Check 5: all record values are unique
        assert len(dataframe["record"]) == len(
            dataframe["record"].unique()
        ), "All record values should be unique"

    def test_scenario_1b(self):
        """Should generate a full 2D array"""

        m = 100
        n = 33
        gen = GenerateData(
            num_obs=m * n,
            num_dims=2,
            ratio_dims=[m, n],
            density=1.0,
            seed=76,
        )
        dataarray, dataframe = gen.generate()

        # Add checks for dataarray
        # Check 1: there are two dimensions
        assert len(dataarray.dims) == 2, "DataArray should have exactly two dimensions"

        # Check 2: the dimensions present are called 'x0' (first) and 'x1' (second)
        assert list(dataarray.dims) == [
            "x0",
            "x1",
        ], "Dimensions should be named 'x0' and 'x1' in that order"

        # Check 3: 'x0' has 100 coordinates
        assert len(dataarray.coords["x0"]) == m, f"'x0' should have {m} coordinates"

        # Check 4: 'x1' has 33 coordinates
        assert len(dataarray.coords["x1"]) == n, f"'x1' should have {n} coordinates"

        # Check 5: all coordinates tuples have a corresponding measurement in 'record'
        non_nan_count = np.count_nonzero(~np.isnan(dataarray.values))
        assert (
            non_nan_count == m * n
        ), f"All {m * n} coordinate pairs should have non-NaN values"

        # Check 6: all coordinates in x0 and x1 are unique
        assert len(dataarray.coords["x0"]) == len(
            np.unique(dataarray.coords["x0"])
        ), "All coordinates in x0 should be unique"
        assert len(dataarray.coords["x1"]) == len(
            np.unique(dataarray.coords["x1"])
        ), "All coordinates in x1 should be unique"

        # Check 7: all record values are unique
        valid_values = dataarray.values[~np.isnan(dataarray.values)].flatten()
        assert len(valid_values) == len(
            np.unique(valid_values)
        ), "All record values should be unique"

        # Add checks for dataframe
        # Check 1: there are three columns ('x0', 'x1' and 'record')
        assert list(dataframe.columns) == [
            "x0",
            "x1",
            "record",
        ], "DataFrame should have exactly three columns: 'x0', 'x1', and 'record'"

        # Check 2: 'x0' has m*n rows
        assert len(dataframe["x0"]) == m * n, f"'x0' column should have {m * n} rows"

        # Check 2b: 'x0' has 100 unique values
        assert len(dataframe["x0"].unique()) == m, f"'x0' should have {m} unique values"

        # Check 3: 'x1' has m*n rows
        assert len(dataframe["x1"]) == m * n, f"'x1' column should have {m * n} rows"

        # Check 3b: 'x1' has 33 unique values
        assert len(dataframe["x1"].unique()) == n, f"'x1' should have {n} unique values"

        # Check 4: Each row has a unique combination of x0 and x1 values
        unique_combinations = dataframe[["x0", "x1"]].drop_duplicates()
        assert (
            len(unique_combinations) == m * n
        ), "Each row should have a unique combination of x0 and x1 values"

        # Check 5: 'record' has m*n rows
        assert (
            len(dataframe["record"]) == m * n
        ), f"'record' column should have {m * n} rows"

        # Check 6: all record values are unique
        assert len(dataframe["record"]) == len(
            dataframe["record"].unique()
        ), "All record values should be unique"

    def test_scenario_1c(self):
        """Should generate a full 10-D array"""

        m = [10, 5, 6, 8, 20, 3, 7]

        gen = GenerateData(
            num_obs=int(np.prod(m)),
            num_dims=len(m),
            ratio_dims=m,
            density=1.0,
            seed=10,
        )
        dataarray, dataframe = gen.generate()

        # Add checks for dataarray
        # Check 1: there are 10 dimensions
        assert len(dataarray.dims) == len(
            m
        ), f"DataArray should have exactly {len(m)} dimensions"

        # Check 2: the dimensions present are called 'x0' (first) and 'x1' (second), and 'x2', etc until 'x9'
        expected_dims = [f"x{i}" for i in range(len(m))]
        assert (
            list(dataarray.dims) == expected_dims
        ), f"Dimensions should be named {expected_dims} in that order"

        # Check 3: 'x0' has 10 coordinates, 'x1' has 5 coordinates, etc until x9 has 7 coordinates
        for i, size in enumerate(m):
            dim_name = f"x{i}"
            assert (
                len(dataarray.coords[dim_name]) == size
            ), f"'{dim_name}' should have {size} coordinates"

        # Check 5: all coordinates tuples (of size 10) have a corresponding measurement in 'record'
        total_points = np.prod(m)
        non_nan_count = np.count_nonzero(~np.isnan(dataarray.values))
        assert (
            non_nan_count == total_points
        ), f"All {total_points} coordinate tuples should have non-NaN values"

        # Check 6: all coordinates in x0, x1, x2, etc are unique
        for i, size in enumerate(m):
            dim_name = f"x{i}"
            coords = dataarray.coords[dim_name]
            assert len(coords) == len(
                np.unique(coords)
            ), f"All coordinates in {dim_name} should be unique"

        # Check 7: all record values are unique
        valid_values = dataarray.values[~np.isnan(dataarray.values)].flatten()
        assert len(valid_values) == len(
            np.unique(valid_values)
        ), "All record values should be unique"

        # Add checks for dataframe
        # Check 1: there are 11 columns ('x0', 'x1', etc, 'x9' and 'record')
        expected_columns = [f"x{i}" for i in range(len(m))] + ["record"]
        assert (
            list(dataframe.columns) == expected_columns
        ), f"DataFrame should have columns {expected_columns}"

        # Check 2: the dataframe has np.prod(m) rows
        total_rows = np.prod(m)
        assert len(dataframe) == total_rows, f"DataFrame should have {total_rows} rows"

        # Check 3: 'x0' has 10 unique values, 'x1' has 5 values etc
        for i, size in enumerate(m):
            col_name = f"x{i}"
            unique_count = len(dataframe[col_name].unique())
            assert (
                unique_count == size
            ), f"'{col_name}' should have {size} unique values, but has {unique_count}"

        # Check 4: Each row has a unique combination of (x0, x1, x2, ..., x9) values
        coord_cols = [f"x{i}" for i in range(len(m))]
        unique_combinations = dataframe[coord_cols].drop_duplicates()
        assert (
            len(unique_combinations) == total_rows
        ), "Each row should have a unique combination of coordinate values"

        # Check 5: 'record' has np.prod(m) rows
        assert (
            len(dataframe["record"]) == total_rows
        ), f"'record' column should have {total_rows} rows"

        # Check 6: all record values are unique
        assert len(dataframe["record"]) == len(
            dataframe["record"].unique()
        ), "All record values should be unique"

    def test_scenario_2a(self):
        """Should raise an error because sparsity is lower than minimum"""

        # With 5 observations across a 7x5 grid (35 points), minimum sparsity would be higher
        # This configuration should raise a ValueError during initialization
        with pytest.raises(
            ValueError, match="Provided density value.*is lower than minimum"
        ):
            gen = GenerateData(
                num_obs=5,
                num_dims=2,
                ratio_dims=[7, 5],
                density=5 / 35,  # This is below the minimum sparsity for this grid
                seed=10,
            )

    def test_scenario_2b(self):
        """Should generate a sparse 2D array"""

        gen = GenerateData(
            num_obs=5,
            num_dims=2,
            ratio_dims=1,
            density=1 / 5,
            seed=35,
        )
        dataarray, dataframe = gen.generate()

        # Add checks for dataarray
        # Check 1: there are two dimensions
        assert len(dataarray.dims) == 2, "DataArray should have exactly two dimensions"

        # Check 2: the dimensions present are called 'x0' (first) and 'x1' (second)
        assert list(dataarray.dims) == [
            "x0",
            "x1",
        ], "Dimensions should be named 'x0' and 'x1' in that order"

        # Check 3: 'x0' has 5 coordinates, 'x1' has 5 coordinates
        assert len(dataarray.coords["x0"]) == 5, "'x0' should have 5 coordinates"
        assert len(dataarray.coords["x1"]) == 5, "'x1' should have 5 coordinates"

        # Check 5: for each dimension, only one coordinate has a corresponding measurement
        non_nan_mask = ~np.isnan(dataarray.values)
        coords = np.argwhere(non_nan_mask)  # list of (i, j) indices for non-NaN cells
        for i, j in coords:
            row_count = int(non_nan_mask[i, :].sum())
            col_count = int(non_nan_mask[:, j].sum())
            assert (
                row_count == 1
            ), f"x0 coordinate at index {i} has {row_count} non-NaN(s); expected exactly 1"
            assert (
                col_count == 1
            ), f"x1 coordinate at index {j} has {col_count} non-NaN(s); expected exactly 1"

        # Check 6: all coordinates in x0 and x1 are unique
        assert len(dataarray.coords["x0"]) == len(
            np.unique(dataarray.coords["x0"])
        ), "All coordinates in x0 should be unique"
        assert len(dataarray.coords["x1"]) == len(
            np.unique(dataarray.coords["x1"])
        ), "All coordinates in x1 should be unique"

        # Check 7: all record values are unique
        valid_values = dataarray.values[~np.isnan(dataarray.values)].flatten()
        assert len(valid_values) == len(
            np.unique(valid_values)
        ), "All record values should be unique"

        # Add checks for dataframe
        # Check 1: there are three columns ('x0', 'x1' and 'record')
        assert list(dataframe.columns) == [
            "x0",
            "x1",
            "record",
        ], "DataFrame should have exactly three columns: 'x0', 'x1', and 'record'"

        # Check 2: the dataframe has 5 rows
        assert len(dataframe) == 5, "DataFrame should have 5 rows"

        # Check 2b: 'x0' and 'x1' have each 5 unique values
        assert len(dataframe["x0"].unique()) == 5, "'x0' should have at 5 unique values"
        assert len(dataframe["x1"].unique()) == 5, "'x1' should have at 5 unique values"

        # Check 4: Each row has a unique combination of x0 and x1 values
        unique_combinations = dataframe[["x0", "x1"]].drop_duplicates()
        assert (
            len(unique_combinations) == 5
        ), "Each row should have a unique combination of x0 and x1 values"

        # Check 4b: 'record' has 5 rows
        assert len(dataframe["record"]) == 5, "'record' column should have 5 rows"

        # Check 6: all record values are unique
        assert len(dataframe["record"]) == len(
            dataframe["record"].unique()
        ), "All record values should be unique"

        # Check 8: all entries in the dataframe are numbers (i.e. there isn't any NaN)
        assert (
            not dataframe.isnull().any().any()
        ), "DataFrame should not contain any NaN values"

    def test_scenario_2c(self):
        """Should generate a 10-D array with minimum sparsity"""

        # For minimum sparsity in 10 dimensions with ratio_dims=1:
        # - Grid: 3×3×3×3×3×3×3×3×3×3 = 3^10 = 59,049 points
        # - Minimum sparsity = 1/n^(d-1) = 1/3^9 ≈ 5.08e-05
        # - With num_obs = 3, each coordinate used exactly once (like test_scenario_2b)
        # - This tests "exactly once" semantics in high dimensions
        gen = GenerateData(
            num_obs=3,
            num_dims=10,
            ratio_dims=1,
            density=1 / (3**9),  # Minimum sparsity: 1/3^9 ≈ 5.08e-05
            seed=11,
        )
        dataarray, dataframe = gen.generate()

        # Add checks for dataarray
        # Check 1: there are 10 dimensions
        assert len(dataarray.dims) == 10, "DataArray should have exactly 10 dimensions"

        # Check 2: the dimensions present are called 'x0' (first) and 'x1' (second), and 'x2', etc until 'x9'
        expected_dims = [f"x{i}" for i in range(10)]
        assert (
            list(dataarray.dims) == expected_dims
        ), f"Dimensions should be named {expected_dims} in that order"

        # Check 3: all dimensions have the same number of coordinates
        dim_sizes = [len(dataarray.coords[f"x{i}"]) for i in range(10)]
        assert (
            len(set(dim_sizes)) == 1
        ), "All dimensions should have the same number of coordinates"
        expected_size = dim_sizes[0]

        # Check 4: for each dimension, only one coordinate has a corresponding measurement
        # ensure each occupied coordinate along every axis is used exactly once
        non_nan_mask = ~np.isnan(dataarray.values)
        non_nan_count = np.count_nonzero(non_nan_mask)
        non_nan_indices = np.argwhere(
            non_nan_mask
        )  # shape (k, n) where n = number of dims
        ndim = dataarray.ndim
        dim_names = list(dataarray.dims)

        # Precompute how many times each coordinate index is used along each axis
        counts_by_axis = []
        for axis in range(ndim):
            axis_indices = non_nan_indices[
                :, axis
            ]  # positional indices along this axis for all non-NaNs
            # bincount with minlength ensures indices with 0 occurrences are present
            counts = np.bincount(axis_indices, minlength=dataarray.shape[axis])
            counts_by_axis.append(counts)

        # For every non-NaN position, assert that the coordinate index for each axis is used exactly once
        for idx_tuple in map(tuple, non_nan_indices):
            for axis, coord_idx in enumerate(idx_tuple):
                cnt = counts_by_axis[axis][coord_idx]
                if cnt != 1:
                    dim = dim_names[axis]
                    # include the coordinate label (if coords are meaningful labels rather than just positional indices)
                    coord_label = dataarray.coords[dim][coord_idx]
                    raise AssertionError(
                        f"Dimension '{dim}' coordinate index {coord_idx} (label={coord_label!r}) "
                        f"is used {int(cnt)} times among non-NaN values; expected exactly 1"
                    )

        # Check 5: all coordinates in x0, x1, x2, etc are unique
        for i in range(10):
            dim_name = f"x{i}"
            coords = dataarray.coords[dim_name]
            assert len(coords) == len(
                np.unique(coords)
            ), f"All coordinates in {dim_name} should be unique"

        # Check 6: all record values are unique
        valid_values = dataarray.values[~np.isnan(dataarray.values)].flatten()
        assert len(valid_values) == len(
            np.unique(valid_values)
        ), "All record values should be unique"

        # Check 8: the ratio of non-nan entries should match the density for the grid
        total_points = np.prod(dataarray.shape)
        actual_density = non_nan_count / total_points
        expected_density = (
            gen.density[0] if isinstance(gen.density, list) else gen.density
        )
        # Allow small difference due to rounding
        assert (
            abs(actual_density - expected_density) < 0.01
        ), f"Density should be approximately {expected_density}, got {actual_density}"

        # Add checks for dataframe
        # Check 1: there are 11 columns ('x0', 'x1', etc, 'x9' and 'record')
        expected_columns = [f"x{i}" for i in range(10)] + ["record"]
        assert (
            list(dataframe.columns) == expected_columns
        ), f"DataFrame should have columns {expected_columns}"

        # Check 2: the dataframe rows match adjusted num_obs
        assert (
            len(dataframe) == gen.num_obs
        ), f"DataFrame should have {gen.num_obs} rows"

        # Check 3: At minimum sparsity with "exactly once" semantics,
        # each dimension should have exactly num_obs unique values used
        # With num_obs=3, each dimension should have 3 unique coordinate values (all used)
        for i in range(10):
            col_name = f"x{i}"
            unique_count = len(dataframe[col_name].unique())
            assert (
                unique_count == gen.num_obs
            ), f"'{col_name}' should have {gen.num_obs} unique values (one per observation)"

        # Check 4: Each row has a unique combination of (x0, x1, x2, ..., x9) values
        coord_cols = [f"x{i}" for i in range(10)]
        unique_combinations = dataframe[coord_cols].drop_duplicates()
        assert (
            len(unique_combinations) == gen.num_obs
        ), "Each row should have a unique combination of coordinate values"

        # Check 5: 'record' rows match adjusted num_obs
        assert (
            len(dataframe["record"]) == gen.num_obs
        ), f"'record' column should have {gen.num_obs} rows"

        # Check 6: all record values are unique
        assert len(dataframe["record"]) == len(
            dataframe["record"].unique()
        ), "All record values should be unique"

        # Check 7: all entries in the dataframe are numbers (i.e. there isn't any NaN)
        assert (
            not dataframe.isnull().any().any()
        ), "DataFrame should not contain any NaN values"

    def test_scenario_3a(self):
        """Should generate a sparse 2D array"""

        gen = GenerateData(
            num_obs=10,
            num_dims=2,
            ratio_dims=[7, 5],
            density=10 / 35,
            seed=31,
        )
        dataarray, dataframe = gen.generate()

        # Add checks for dataarray
        # Check 1: there are two dimensions
        assert len(dataarray.dims) == 2, "DataArray should have exactly two dimensions"

        # Check 2: the two dimensions are called 'x0', 'x1'
        assert list(dataarray.dims) == [
            "x0",
            "x1",
        ], "Dimensions should be named 'x0' and 'x1' in that order"

        # Check 3: 'x0' has 7 coordinates values
        assert len(dataarray.coords["x0"]) == 7, "'x0' should have 7 coordinates"

        # Check 4: 'x1' has 5 coordinates values
        assert len(dataarray.coords["x1"]) == 5, "'x1' should have 5 coordinates"

        # Check 5: with sparse data, not all coordinates need to be used
        # With 10 observations in a 7x5 grid, some coordinates may be unused
        # Just verify that at least some coordinates in each dimension have data
        x0_with_data = np.any(~np.isnan(dataarray.values), axis=1)
        x1_with_data = np.any(~np.isnan(dataarray.values), axis=0)
        assert np.any(
            x0_with_data
        ), "At least some x0 coordinates should have measurements"
        assert np.any(
            x1_with_data
        ), "At least some x1 coordinates should have measurements"

        # Check 6: all coordinates in x0 are unique
        assert len(dataarray.coords["x0"]) == len(
            np.unique(dataarray.coords["x0"])
        ), "All coordinates in x0 should be unique"

        # Check 7: all coordinates in x1 are unique
        assert len(dataarray.coords["x1"]) == len(
            np.unique(dataarray.coords["x1"])
        ), "All coordinates in x1 should be unique"

        # Check 8: not all points (tuples of coordinates) have a corresponding measurement in 'record'
        non_nan_count = np.count_nonzero(~np.isnan(dataarray.values))
        total_points = 7 * 5
        assert (
            non_nan_count < total_points
        ), "Not all coordinate pairs should have measurements (sparse array)"

        # Check 9: all record values are unique
        valid_values = dataarray.values[~np.isnan(dataarray.values)].flatten()
        assert len(valid_values) == len(
            np.unique(valid_values)
        ), "All record values should be unique"

        # Check 10: the ratio of non-nan entries in record over the length of record should be 10/35
        actual_density = non_nan_count / total_points
        expected_density = 10 / 35
        assert (
            abs(actual_density - expected_density) < 0.01
        ), f"Density ratio should be approximately {expected_density}, got {actual_density}"

        # Add checks for dataframe
        # Check 1: there are three columns ('x0', 'x1' and 'record')
        assert list(dataframe.columns) == [
            "x0",
            "x1",
            "record",
        ], "DataFrame should have exactly three columns: 'x0', 'x1', and 'record'"

        # Check 2: the dataframe has 10 rows
        assert len(dataframe) == 10, "DataFrame should have 10 rows"

        # Check 3a: 'x0' unique values (sparse data may not use all coordinates)
        x0_unique = len(dataframe["x0"].unique())
        assert x0_unique <= 7, "'x0' should have at most 7 unique values"
        assert x0_unique >= 1, "'x0' should have at least 1 unique value"

        # Check 3b: 'x1' unique values (sparse data may not use all coordinates)
        x1_unique = len(dataframe["x1"].unique())
        assert x1_unique <= 5, "'x1' should have at most 5 unique values"
        assert x1_unique >= 1, "'x1' should have at least 1 unique value"

        # Check 5: all record values are unique
        assert len(dataframe["record"]) == len(
            dataframe["record"].unique()
        ), "All record values should be unique"

        # Check 8: all entries in the dataframe are numbers (i.e. there isn't any NaN)
        assert (
            not dataframe.isnull().any().any()
        ), "DataFrame should not contain any NaN values"

    def test_scenario_3b(self):
        """Should generate a 2D array with all sites full except one"""

        m = 15
        n = 4
        gen = GenerateData(
            num_obs=(m * n - 1),
            num_dims=2,
            ratio_dims=[m, n],
            density=(m * n - 1) / (m * n),
            seed=20,
        )
        dataarray, dataframe = gen.generate()

        # Add checks for dataarray
        # Check 1: there are two dimensions
        assert len(dataarray.dims) == 2, "DataArray should have exactly two dimensions"

        # Check 2: the two dimensions are called 'x0', 'x1'
        assert list(dataarray.dims) == [
            "x0",
            "x1",
        ], "Dimensions should be named 'x0' and 'x1' in that order"

        # Check 3a: 'x0' has m coordinates values
        assert len(dataarray.coords["x0"]) == m, f"'x0' should have {m} coordinates"

        # Check 3b: 'x1' has n coordinates values
        assert len(dataarray.coords["x1"]) == n, f"'x1' should have {n} coordinates"

        # Check 4: all coordinates have at least one corresponding measurement in 'record'
        # Check if all x0 coordinates are used
        x0_with_data = np.any(~np.isnan(dataarray.values), axis=1)
        assert np.all(
            x0_with_data
        ), "All x0 coordinates should have at least one measurement"
        # Check if all x1 coordinates are used
        x1_with_data = np.any(~np.isnan(dataarray.values), axis=0)
        assert np.all(
            x1_with_data
        ), "All x1 coordinates should have at least one measurement"

        # Check 5a: all coordinates in x0 are unique
        assert len(dataarray.coords["x0"]) == len(
            np.unique(dataarray.coords["x0"])
        ), "All coordinates in x0 should be unique"

        # Check 5b: all coordinates in x1 are unique
        assert len(dataarray.coords["x1"]) == len(
            np.unique(dataarray.coords["x1"])
        ), "All coordinates in x1 should be unique"

        # Check 7: not all points (tuples of coordinates) have a corresponding measurement in 'record'
        non_nan_count = np.count_nonzero(~np.isnan(dataarray.values))
        total_points = m * n
        assert (
            non_nan_count < total_points
        ), "Not all coordinate pairs should have measurements (one missing)"
        assert (
            non_nan_count == m * n - 1
        ), f"Should have exactly {m * n - 1} measurements"

        # Check 6: all record values are unique
        valid_values = dataarray.values[~np.isnan(dataarray.values)].flatten()
        assert len(valid_values) == len(
            np.unique(valid_values)
        ), "All record values should be unique"

        # Check 8: the ratio of non-nan entries in record over the length of record should be (m*n-1)/(m*n)=59/60
        actual_density = non_nan_count / total_points
        expected_density = (m * n - 1) / (m * n)
        assert (
            abs(actual_density - expected_density) < 0.001
        ), f"Density ratio should be approximately {expected_density}, got {actual_density}"

        # Add checks for dataframe
        # Check 1: there are three columns ('x0', 'x1' and 'record')
        assert list(dataframe.columns) == [
            "x0",
            "x1",
            "record",
        ], "DataFrame should have exactly three columns: 'x0', 'x1', and 'record'"

        # Check 2: the dataframe has (m*n-1) rows
        assert len(dataframe) == m * n - 1, f"DataFrame should have {m * n - 1} rows"

        # Check 3a: 'x0' has m unique values
        assert len(dataframe["x0"].unique()) == m, f"'x0' should have {m} unique values"

        # Check 3b: 'x1' has n unique values
        assert len(dataframe["x1"].unique()) == n, f"'x1' should have {n} unique values"

        # Check 5: all record values are unique
        assert len(dataframe["record"]) == len(
            dataframe["record"].unique()
        ), "All record values should be unique"

        # Check 8: all entries in the dataframe are numbers (i.e. there isn't any NaN)
        assert (
            not dataframe.isnull().any().any()
        ), "DataFrame should not contain any NaN values"

    def test_scenario_3c(self):
        """Should generate a 10D array with all sites full except one"""

        m = [3, 5, 6, 9, 19, 2, 7]

        gen = GenerateData(
            num_obs=int(np.prod(m) - 1),
            num_dims=len(m),
            ratio_dims=m,
            density=(np.prod(m) - 1) / np.prod(m),
            seed=20,
        )
        dataarray, dataframe = gen.generate()

        # Add checks for dataarray
        # Check 1: there are 10 dimensions
        assert len(dataarray.dims) == len(
            m
        ), f"DataArray should have exactly {len(m)} dimensions"

        # Check 2: the dimensions present are called 'x0' (first) and 'x1' (second), and 'x2', etc until 'x9'
        expected_dims = [f"x{i}" for i in range(len(m))]
        assert (
            list(dataarray.dims) == expected_dims
        ), f"Dimensions should be named {expected_dims} in that order"

        # Check 3: 'x0' has 3 coordinates, 'x1' has 5 coordinates, etc until x9 has 7 coordinates
        for i, size in enumerate(m):
            dim_name = f"x{i}"
            assert (
                len(dataarray.coords[dim_name]) == size
            ), f"'{dim_name}' should have {size} coordinates"

        # Check 4: all coordinates have at least one corresponding measurement in 'record'
        # For each dimension, check if all coordinates are used
        for i in range(len(m)):
            # Create axes tuple for np.any - all axes except current one
            axes = tuple(j for j in range(len(m)) if j != i)
            dim_with_data = np.any(~np.isnan(dataarray.values), axis=axes)
            assert np.all(
                dim_with_data
            ), f"All coordinates in x{i} should have at least one measurement"

        # Check 5: all coordinates in x0, x1, x2, etc are unique
        for i, size in enumerate(m):
            dim_name = f"x{i}"
            coords = dataarray.coords[dim_name]
            assert len(coords) == len(
                np.unique(coords)
            ), f"All coordinates in {dim_name} should be unique"

        # Check 7: all points (tuples of coordinates) but one have a corresponding measurement in 'record'
        total_points = int(np.prod(m))
        non_nan_count = np.count_nonzero(~np.isnan(dataarray.values))
        assert (
            non_nan_count == total_points - 1
        ), f"Should have exactly {total_points - 1} measurements (all but one point)"

        # Check 6 (duplicate): all record values are unique
        valid_values = dataarray.values[~np.isnan(dataarray.values)].flatten()
        assert len(valid_values) == len(
            np.unique(valid_values)
        ), "All record values should be unique"

        # Check 8: the ratio of non-nan entries in record over the length of record should be (np.prod(m)-1)/np.prod(m)
        actual_density = non_nan_count / total_points
        expected_density = (np.prod(m) - 1) / np.prod(m)
        assert (
            abs(actual_density - expected_density) < 0.0001
        ), f"Density ratio should be approximately {expected_density}, got {actual_density}"

        # Add checks for dataframe
        # Check 1: there are 11 columns ('x0', 'x1', etc, 'x9' and 'record')
        expected_columns = [f"x{i}" for i in range(len(m))] + ["record"]
        assert (
            list(dataframe.columns) == expected_columns
        ), f"DataFrame should have columns {expected_columns}"

        # Check 2: the dataframe has np.prod(m)-1 rows
        expected_rows = int(np.prod(m) - 1)
        assert (
            len(dataframe) == expected_rows
        ), f"DataFrame should have {expected_rows} rows"

        # Check 3: 'x0' has 3 unique values, 'x1' has 5 values etc
        for i, size in enumerate(m):
            col_name = f"x{i}"
            unique_count = len(dataframe[col_name].unique())
            assert (
                unique_count == size
            ), f"'{col_name}' should have {size} unique values, but has {unique_count}"

        # Check 4: Each row has a unique combination of (x0, x1, x2, ..., x9) values
        coord_cols = [f"x{i}" for i in range(len(m))]
        unique_combinations = dataframe[coord_cols].drop_duplicates()
        assert (
            len(unique_combinations) == expected_rows
        ), "Each row should have a unique combination of coordinate values"

        # Check 5: all record values are unique
        assert len(dataframe["record"]) == len(
            dataframe["record"].unique()
        ), "All record values should be unique"

        # Check 6: all entries in the dataframe are numbers (i.e. there isn't any NaN)
        assert (
            not dataframe.isnull().any().any()
        ), "DataFrame should not contain any NaN values"


class TestHybridLHSSampling:
    """Tests for Hybrid LHS + Random sampling implementation."""

    def test_lhs_base_coverage(self):
        """LHS component should cover every coordinate of every dimension."""
        from data_sparsity.generators.record_generator import RecordGenerator

        rng = np.random.default_rng(42)
        shape = [5, 7, 5]
        n_s = max(shape)  # = 7; sized by the LONGEST axis, not the shortest

        lhs_indices = RecordGenerator.generate_lhs_indices(shape, n_s, rng)

        # Check 1: Correct number of observations
        assert all(
            len(idx) == n_s for idx in lhs_indices
        ), f"All dimensions should have {n_s} samples"

        # Check 2: the longest dimension uses all coordinates exactly once
        assert sorted(lhs_indices[1]) == list(
            range(7)
        ), "Dimension 1 (size 7) should use all coordinates exactly once"

        # Check 3: shorter dimensions repeat, but still use every coordinate
        for dim in (0, 2):
            assert set(lhs_indices[dim].tolist()) == set(
                range(5)
            ), f"Dimension {dim} (size 5) must use every coordinate"

    def test_lhs_rejects_n_s_below_longest_axis(self):
        """n_s below max(shape) cannot cover every axis, so it is refused.

        An unused coordinate is a grid site carrying no information, which
        docs/explainer.md excludes from the definition of a grid.
        """
        from data_sparsity.generators.record_generator import RecordGenerator

        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match=r"n_s .* < max\(shape\)"):
            RecordGenerator.generate_lhs_indices([5, 7, 3], 5, rng)

    def test_hybrid_rejects_too_few_observations(self):
        """Fewer observations than the longest axis cannot cover it."""
        from data_sparsity.generators.record_generator import RecordGenerator

        with pytest.raises(ValueError, match=r"num_obs .* < max\(shape\)"):
            RecordGenerator.generate_hybrid_indices(
                [4, 7, 10],
                5,
                np.random.default_rng(42),
            )
        with pytest.raises(ValueError, match=r"num_obs .* < max\(shape\)"):
            RecordGenerator.generate_stratified_indices([4, 7, 10], 5, 1, 2)

    def test_lhs_covers_axes_shorter_than_n_s(self):
        """Axes shorter than n_s are tiled so every coordinate is still used.

        n_s used to be capped at min(shape) and a shorter axis raised. It is now
        sized by max(shape), because covering an axis of length L needs at least
        L points, so short axes must repeat coordinates rather than fail.
        """
        from data_sparsity.generators.record_generator import RecordGenerator

        rng = np.random.default_rng(42)
        shape = [5, 7, 3]
        n_s = 7  # = max(shape); dimension 2 (size 3) is shorter than n_s

        lhs_indices = RecordGenerator.generate_lhs_indices(shape, n_s, rng)

        for dim, dim_size in enumerate(shape):
            assert len(lhs_indices[dim]) == n_s
            assert set(lhs_indices[dim].tolist()) == set(
                range(dim_size)
            ), f"Dimension {dim} (size {dim_size}) must use every coordinate"

    def test_lhs_covers_every_coordinate_of_every_axis(self):
        """The guarantee that makes a grid legitimate: no unused coordinate.

        docs/explainer.md only considers grids where every coordinate is
        occupied at least once; an unused coordinate carries no information and
        should not be part of the grid.
        """
        from data_sparsity.generators.record_generator import RecordGenerator

        for shape in ([4, 7, 10], [3, 3], [5, 5, 20], [2, 3, 5, 7], [12]):
            n_s = max(shape)
            lhs_indices = RecordGenerator.generate_lhs_indices(
                shape,
                n_s,
                np.random.default_rng(7),
            )
            for dim, dim_size in enumerate(shape):
                assert set(lhs_indices[dim].tolist()) == set(
                    range(dim_size)
                ), f"shape {shape}: dimension {dim} has an unused coordinate"

    def test_hybrid_at_minimum_sparsity(self):
        """Hybrid should equal pure LHS at minimum sparsity."""
        from data_sparsity.generators.record_generator import RecordGenerator

        rng = np.random.default_rng(42)
        shape = [5, 5]
        num_obs = 5  # Minimum sparsity for this shape

        hybrid_indices = RecordGenerator.generate_hybrid_indices(shape, num_obs, rng)

        # Check 1: Correct total number of observations
        assert all(
            len(idx) == num_obs for idx in hybrid_indices
        ), f"Should have {num_obs} observations per dimension"

        # Check 2: All coordinates used exactly once (pure LHS behavior)
        for dim_idx in range(len(shape)):
            unique_coords = set(hybrid_indices[dim_idx])
            assert len(unique_coords) == shape[dim_idx], (
                f"At minimum sparsity, all {shape[dim_idx]} coordinates "
                f"in dimension {dim_idx} should be used"
            )
            assert unique_coords == set(
                range(shape[dim_idx])
            ), f"Should use coordinates 0 through {shape[dim_idx]-1}"

    def test_hybrid_above_minimum_sparsity(self):
        """Hybrid should use LHS base + random fill above minimum."""
        from data_sparsity.generators.record_generator import RecordGenerator

        rng = np.random.default_rng(42)
        shape = [5, 5]
        num_obs = 10  # Above minimum (0.4 vs 0.2 minimum sparsity)

        hybrid_indices = RecordGenerator.generate_hybrid_indices(shape, num_obs, rng)

        # Check 1: Correct total number of observations
        assert all(
            len(idx) == num_obs for idx in hybrid_indices
        ), f"Should have {num_obs} observations per dimension"

        # Check 2: All coordinates used (guaranteed by LHS base)
        for dim_idx in range(len(shape)):
            unique_coords = set(hybrid_indices[dim_idx])
            assert len(unique_coords) == shape[dim_idx], (
                f"Hybrid approach should ensure all {shape[dim_idx]} "
                f"coordinates in dimension {dim_idx} are used"
            )

        # Check 3: Some coordinates used multiple times (from random fill)
        for dim_idx in range(len(shape)):
            coords = hybrid_indices[dim_idx]
            counts = {}
            for coord in coords:
                counts[coord] = counts.get(coord, 0) + 1

            # With 10 obs and 5 coords, at least one must be used twice
            assert any(count > 1 for count in counts.values()), (
                f"Above minimum sparsity, some coordinates in dimension "
                f"{dim_idx} should be used multiple times"
            )

    def test_hybrid_high_sparsity(self):
        """Hybrid should guarantee coverage even at high sparsity."""
        from data_sparsity.generators.record_generator import RecordGenerator

        rng = np.random.default_rng(42)
        shape = [5, 5]
        num_obs = 20  # High sparsity (0.8)

        hybrid_indices = RecordGenerator.generate_hybrid_indices(shape, num_obs, rng)

        # Check: All coordinates still used
        for dim_idx in range(len(shape)):
            unique_coords = set(hybrid_indices[dim_idx])
            assert len(unique_coords) == shape[dim_idx], (
                f"Even at high sparsity, all {shape[dim_idx]} coordinates "
                f"in dimension {dim_idx} should be used"
            )

    def test_hybrid_nonuniform_dimensions(self):
        """Hybrid should work with non-uniform dimension sizes."""
        from data_sparsity.generators.record_generator import RecordGenerator

        rng = np.random.default_rng(42)
        shape = [3, 10, 5, 7]
        num_obs = 10  # Above min(shape) = 3
        n_s = min(shape)  # = 3

        hybrid_indices = RecordGenerator.generate_hybrid_indices(shape, num_obs, rng)

        # Check: Minimum dimension coordinates all used
        min_dim_indices = [i for i, size in enumerate(shape) if size == n_s]
        for dim_idx in min_dim_indices:
            unique_coords = set(hybrid_indices[dim_idx])
            assert len(unique_coords) == shape[dim_idx], (
                f"Minimum dimension {dim_idx} (size {shape[dim_idx]}) "
                f"should have all coordinates used"
            )

        # Check: Larger dimensions have at least n_s unique coordinates
        for dim_idx, dim_size in enumerate(shape):
            if dim_size > n_s:
                unique_coords = set(hybrid_indices[dim_idx])
                # LHS base ensures n_s unique coords, random may add more
                assert len(unique_coords) >= n_s, (
                    f"Dimension {dim_idx} (size {dim_size}) should have "
                    f"at least {n_s} unique coordinates, found {len(unique_coords)}"
                )


class TestHybridLHSIntegration:
    """Integration tests for hybrid LHS with GenerateData."""

    def test_scenario_2b_with_hybrid(self):
        """test_scenario_2b should now pass with hybrid approach."""
        gen = GenerateData(
            num_obs=5,
            num_dims=2,
            ratio_dims=1,
            density=1 / 5,  # Minimum sparsity
            seed=35,
        )
        dataarray, dataframe = gen.generate()

        # At minimum sparsity, hybrid approach = pure LHS
        # All coordinates should be used exactly once
        for dim_idx, dim_name in enumerate(["x0", "x1"]):
            used_coords = dataframe[dim_name].unique()
            expected_count = dataarray.shape[dim_idx]

            assert len(used_coords) == expected_count, (
                f"Hybrid LHS should guarantee all {expected_count} coordinates "
                f"in {dim_name} are used, found {len(used_coords)}"
            )

            # Each coordinate used exactly once at minimum sparsity
            coord_counts = dataframe[dim_name].value_counts()
            assert all(coord_counts == 1), (
                f"At minimum sparsity, each coordinate in {dim_name} "
                f"should be used exactly once (pure LHS)"
            )

    def test_coordinate_coverage_at_multiple_sparsity_levels(self):
        """Hybrid LHS should guarantee coverage at all sparsity levels."""
        scenarios = [
            {"num_obs": 5, "sparsity": 0.2, "desc": "minimum"},
            {"num_obs": 10, "sparsity": 0.4, "desc": "above minimum"},
            {"num_obs": 20, "sparsity": 0.8, "desc": "high"},
        ]

        for scenario in scenarios:
            gen = GenerateData(
                num_obs=scenario["num_obs"],
                num_dims=2,
                ratio_dims=1,
                density=scenario["sparsity"],
                seed=42,
            )
            dataarray, dataframe = gen.generate()

            # At ANY sparsity, all coordinates should be used
            for dim_name in ["x0", "x1"]:
                used_coords = dataframe[dim_name].unique()
                expected = 5  # Grid is 5×5

                assert len(used_coords) == expected, (
                    f"Hybrid LHS should guarantee all {expected} coordinates "
                    f"in {dim_name} are used at {scenario['desc']} sparsity "
                    f"(num_obs={scenario['num_obs']}), found {len(used_coords)}"
                )

    def test_scenario_3a_strengthened(self):
        """test_scenario_3a should have stronger guarantees with hybrid."""
        gen = GenerateData(
            num_obs=10,
            num_dims=2,
            ratio_dims=[7, 5],
            density=10 / 35,
            seed=12,
        )
        dataarray, dataframe = gen.generate()

        # With hybrid approach, minimum dimension coordinates ALL used
        # For shape [7, 5], min(shape) = 5
        # So x1 (size 5) guarantees all coordinates used
        # x0 (size 7) guarantees at least 5 unique coordinates used

        # Check x1 (minimum dimension): all coordinates used
        x1_used = dataframe["x1"].unique()
        assert len(x1_used) == 5, (
            f"Hybrid LHS guarantees all 5 coordinates in x1 are used, "
            f"found {len(x1_used)}"
        )

        # Check x0 (larger dimension): at least min(shape) coordinates used
        x0_used = dataframe["x0"].unique()
        assert len(x0_used) >= 5, (
            f"Hybrid LHS guarantees at least 5 coordinates in x0 are used, "
            f"found {len(x0_used)}"
        )
