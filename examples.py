#!/usr/bin/env python
"""Example usage of the GenerateData class.

This example demonstrates how to use the GenerateData class to create
synthetic observation data in both array (NetCDF) and tabular (Parquet)
formats for performance comparison studies.
"""

import tempfile
import os
from data_sparsity import GenerateData


def example_basic_generation():
    """Example: Basic data generation and saving."""
    print("=" * 70)
    print("Example 1: Basic Data Generation")
    print("=" * 70)

    # Create a temporary directory for output files
    with tempfile.TemporaryDirectory() as tmpdir:
        netcdf_path = os.path.join(tmpdir, "sparse_data.nc")
        parquet_dir = os.path.join(tmpdir, "sparse_data_parquet")

        # Create data generator with 1000 observations in 3D space
        gen = GenerateData(
            num_obs=1000,
            num_dims=3,
            ratio_dims=(1.0, 1.0, 1.0),  # Equal size dimensions
            sparsity=0.1,                 # 10% of grid points have data
            seed=42
        )

        # Generate data
        dataarray, dataframe = gen.generate()
        
        # Save to files
        gen.save_to_netcdf(netcdf_path, overwrite=True)
        gen.save_to_parquet(parquet_dir, filename="sparse_data", overwrite=True)

        print(f"\nGenerated {len(dataframe)} observations")
        print(f"Array shape: {dataarray.shape}")
        print(f"Array dimensions: {list(dataarray.dims)}")
        print(f"DataFrame shape: {dataframe.shape}")
        print(f"DataFrame columns: {list(dataframe.columns)}")
        
        # Calculate sizes
        netcdf_size = os.path.getsize(netcdf_path)
        parquet_size = sum(
            os.path.getsize(os.path.join(parquet_dir, f))
            for f in os.listdir(parquet_dir)
        )
        
        print(f"\nNetCDF file size: {netcdf_size:,} bytes")
        print(f"Parquet dataset size: {parquet_size:,} bytes")
        print()


def example_different_sparsity_levels():
    """Example: Generate data with different sparsity levels."""
    print("=" * 70)
    print("Example 2: Different Sparsity Levels")
    print("=" * 70)

    sparsity_levels = [0.01, 0.1, 0.5, 0.9]

    for sparsity in sparsity_levels:
        print(f"\n--- Sparsity {sparsity:.0%} ---")
        gen = GenerateData(
            num_obs=100,
            num_dims=2,
            ratio_dims=1,  # Use simplified API: ratio_dims=1 means all dims equal
            sparsity=sparsity,
            seed=42
        )

        dataarray, dataframe = gen.generate()

        print(f"\nResults:")
        print(f"  Grid size: {dataarray.shape}")
        print(f"  Total grid points: {dataarray.size}")
        print(f"  Observations: {len(dataframe)}")
        print(f"  Filled ratio: {len(dataframe) / dataarray.size:.2%}")
        print()


def example_non_uniform_dimensions():
    """Example: Generate data with non-uniform dimension ratios."""
    print("=" * 70)
    print("Example 3: Non-Uniform Dimensions")
    print("=" * 70)

    # Create data with different dimension ratios (e.g., representing x, y, z or time, lat, lon)
    # Note: ratio_dims represents relative sizes where first element is normalized to 1
    gen = GenerateData(
        num_obs=500,
        num_dims=3,
        ratio_dims=[3.0, 2.0, 1.0],  # First dimension 3x larger than third
        sparsity=0.2,
        seed=123
    )

    dataarray, dataframe = gen.generate()

    print("\nDimension ratios: [3.0, 2.0, 1.0]")
    print(f"Generated array shape: {dataarray.shape}")
    print(f"Generated {len(dataframe)} observations")
    print()

    # Show coordinate ranges
    for dim_name, coords in dataarray.coords.items():
        print(f"{dim_name}: {len(coords)} points, range [{coords.min().item():.3f}, {coords.max().item():.3f}]")
    print()


def example_inspect_data():
    """Example: Inspect generated data."""
    print("=" * 70)
    print("Example 4: Inspecting Generated Data")
    print("=" * 70)

    gen = GenerateData(
        num_obs=50,
        num_dims=2,
        ratio_dims=1,  # All dimensions equal size
        sparsity=0.3,
        seed=999
    )

    dataarray, dataframe = gen.generate()

    print("\nDataArray attributes:")
    for key, value in dataarray.attrs.items():
        if isinstance(value, (list, tuple)) or hasattr(value, '__iter__') and not isinstance(value, str):
            print(f"  {key}: {list(value) if hasattr(value, '__iter__') else value}")
        else:
            print(f"  {key}: {value}")
    print()

    print("DataFrame sample (first 5 rows):")
    print(dataframe.head())
    print()

    print("DataFrame statistics:")
    print(dataframe.describe())
    print()


def example_reproducibility():
    """Example: Demonstrate reproducibility with seeds."""
    print("=" * 70)
    print("Example 5: Reproducibility with Seeds")
    print("=" * 70)

    print("\n--- Generating first dataset with seed=42 ---")
    gen1 = GenerateData(num_obs=100, num_dims=2, ratio_dims=1, sparsity=0.2, seed=42)
    _, df1 = gen1.generate()

    print("\n--- Generating second dataset with seed=42 ---")
    gen2 = GenerateData(num_obs=100, num_dims=2, ratio_dims=1, sparsity=0.2, seed=42)
    _, df2 = gen2.generate()

    print("\nGenerated two datasets with the same seed (42)")
    print(f"Dataset 1 shape: {df1.shape}")
    print(f"Dataset 2 shape: {df2.shape}")
    print(f"Datasets are identical: {df1.equals(df2)}")
    print()

    # Generate with different seed
    print("--- Generating third dataset with seed=123 ---")
    gen3 = GenerateData(num_obs=100, num_dims=2, ratio_dims=1, sparsity=0.2, seed=123)
    _, df3 = gen3.generate()

    print("\nGenerated third dataset with different seed (123)")
    print(f"Dataset 3 shape: {df3.shape}")
    print(f"Dataset 1 and 3 are identical: {df1.equals(df3)}")
    print()


if __name__ == "__main__":
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  GenerateData Class - Usage Examples".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print()

    example_basic_generation()
    example_different_sparsity_levels()
    example_non_uniform_dimensions()
    example_inspect_data()
    example_reproducibility()

    print("*" * 70)
    print("All examples completed successfully!")
    print("*" * 70)
