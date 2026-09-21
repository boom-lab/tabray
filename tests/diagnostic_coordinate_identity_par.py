"""Diagnostic test to verify coordinate identity across chunks."""

import numpy as np
from data_sparsity.generate_data import GenerateData
import tempfile
import os
import glob
import xarray as xr


def test_coordinate_arrays_match():
    """Verify that non-split dimension coordinates are identical across chunks."""

    temp_dir = tempfile.mkdtemp()

    # Generate parallel data
    gen = GenerateData(
        num_obs=5,
        num_dims=2,
        ratio_dims=1,
        density=1 / 5,
        seed=35,
        max_obs=2,
    )

    gen.generate(
        netcdf_filepath=os.path.join(temp_dir, "test.nc"),
        parquet_filepath=os.path.join(temp_dir, "test"),
    )

    # Load all chunks
    nc_files = sorted(glob.glob(os.path.join(temp_dir, "test_*.nc")))
    chunks = [xr.open_dataarray(f) for f in nc_files]

    print(f"\nFound {len(chunks)} chunks")

    # Check x1 coordinates (non-split dimension)
    x1_arrays = [chunk.coords["x1"].values for chunk in chunks]

    print(f"\nChunk 0 x1 shape: {x1_arrays[0].shape}")
    print(f"Chunk 0 x1: {x1_arrays[0][:5]}")
    print(f"Chunk 0 x1: {x1_arrays[0][-5:]}")

    for i in range(1, len(chunks)):
        print(f"\nChunk {i} x1 shape: {x1_arrays[i].shape}")
        print(f"Chunk {i} x1: {x1_arrays[i][:5]}")
        print(f"Chunk {i} x1: {x1_arrays[i][-5:]}")

        # Check if identical
        if np.array_equal(x1_arrays[0], x1_arrays[i]):
            print(f"✓ Chunk {i} x1 IDENTICAL to Chunk 0")
        else:
            print(f"✗ Chunk {i} x1 DIFFERS from Chunk 0")
            # Show differences
            diff_indices = np.where(x1_arrays[0] != x1_arrays[i])[0]
            print(f"  Different at {len(diff_indices)} indices")
            if len(diff_indices) > 0:
                print(f"  First difference at index {diff_indices[0]}:")
                print(f"    Chunk 0: {x1_arrays[0][diff_indices[0]]}")
                print(f"    Chunk {i}: {x1_arrays[i][diff_indices[0]]}")

    # Check x0 coordinates (split dimension) - should differ
    x0_arrays = [chunk.coords["x0"].values for chunk in chunks]
    print(f"\nChunk 0 x0 shape: {x0_arrays[0].shape} (should be chunk-specific)")
    print(f"Chunk 1 x0 shape: {x0_arrays[1].shape} (should be chunk-specific)")

    # Cleanup
    for chunk in chunks:
        chunk.close()


if __name__ == "__main__":
    test_coordinate_arrays_match()
