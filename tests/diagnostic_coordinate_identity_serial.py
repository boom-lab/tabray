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
    gen_serial = GenerateData(
        num_obs=5,
        num_dims=2,
        ratio_dims=1,
        density=1/5,
        seed=35,
    )
    
    da_serial, df_serial = gen_serial.generate()

    for coord in da_serial.coords:
        print(coord)
        print(da_serial[coord].values)

    print(da_serial.data)
    
    

if __name__ == "__main__":
    test_coordinate_arrays_match()
