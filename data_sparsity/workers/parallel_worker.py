"""Standalone worker function for parallel data generation.

This module provides a standalone worker function that can be executed
in parallel processes without serialization issues. All parameters are
passed explicitly rather than through object references.
"""

import gc
import logging
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from numpy.typing import ArrayLike

from data_sparsity.generators import CoordinateGenerator, MultiVarRecordGenerator
from data_sparsity.output import NetCDFBuilder, ParquetBuilder
from data_sparsity.utils import ChunkUtils


def generate_chunk(
    chunk_id: int,
    obs_in_chunk: int,
    seed: int,
    shape: Tuple[int, ...],
    sparsity: float,
    num_vars: int,
    num_dims: int,
    ratio_dims: Tuple[float, ...],
    num_obs: int,
    var_sparsities: Optional[ArrayLike],
    var_num_obs: Optional[ArrayLike],
    var_dims_indices: Optional[List[List[int]]],
    var_constant_dims: Optional[List[List[int]]],
    var_constant_coord_indices: Optional[Dict[int, Dict[int, int]]],
    overlap_target: Union[float, str],
    dim_split: int,
    max_dim_size: int,
    div_points: List[int],
    section_sizes: List[int],
    netcdf_filepath: str,
    parquet_tmp: str,
    ntasks: int,
) -> Tuple[int, int, str]:
    """Generate a single chunk of data in parallel.
    
    This function is designed to be called by ProcessPoolExecutor and contains
    no references to the parent GenerateData object. All necessary parameters
    are passed explicitly.
    
    Args:
        chunk_id: Identifier for this chunk
        obs_in_chunk: Number of observations to generate in this chunk
        seed: Random seed for the base RNG
        shape: Full shape of the dataset
        sparsity: Sparsity level (0-1)
        num_vars: Number of variables (1 for single-var)
        num_dims: Number of dimensions
        ratio_dims: Ratio of dimensions
        num_obs: Total number of observations across all chunks
        var_sparsities: Array of sparsity per variable (multi-var only)
        var_num_obs: Array of observation counts per variable (multi-var only)
        var_dims_indices: List of dimension indices per variable (multi-var only)
        var_constant_dims: List of constant dimensions per variable (multi-var only)
        var_constant_coord_indices: Dict of constant coordinates (multi-var only)
        overlap_target: Target overlap fraction or 'maximal' (multi-var only)
        dim_split: Dimension along which to split chunks
        max_dim_size: Maximum size of the split dimension
        div_points: Division points for chunks along split dimension
        section_sizes: Size of each chunk along split dimension
        netcdf_filepath: Base path for NetCDF output
        parquet_tmp: Path template for temporary parquet chunks
        ntasks: Total number of tasks (for formatting)
        
    Returns:
        Tuple of (chunk_id, total_observations, parquet_chunk_path)
    """
    # Set up logging for this worker
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(process)d %(levelname)s %(message)s',
        filename=f'worker_{chunk_id}.log',
    )
    logging.debug("")
    logging.debug("######------ NEW CHUNK ------######")
    
    # Generate two distinct generators:
    # global_rng: identical across tasks, used for coordinates along non-split dimensions
    # task_rng: unique per task, used for split dimension coordinates and observations
    global_rng, task_rng = ChunkUtils.generate_rngs(seed, chunk_id)
    
    # Determine chunk dimensions and range along split dimension
    task_range = (div_points[chunk_id], div_points[chunk_id + 1])
    task_size = section_sizes[chunk_id]
    task_shape = ChunkUtils.update_chunk_shape(shape, dim_split, task_size)
    
    logging.debug("task_range: %s", task_range)
    logging.debug("task_size: %s", task_size)
    logging.debug("task_shape: %s", task_shape)
    
    total_chunk_points = ChunkUtils.validate_chunk_points(task_shape)
    logging.debug("total_chunk_points: %s", total_chunk_points)
    
    # Build dimension ranges and RNGs for coordinate generation
    dim_ranges = ChunkUtils.generate_split_dimension_range(
        dim_split,
        task_range,
        max_dim_size
    )
    
    dim_rngs = ChunkUtils.assign_rngs_to_dimensions(
        dim_split,
        task_shape,
        global_rng,
        task_rng
    )
    
    # Generate coordinates with optional dimension-specific ranges and RNGs
    coordinates = CoordinateGenerator.generate_all_coords(
        task_shape,
        global_rng,
        dim_ranges,
        dim_rngs
    )
    
    logging.debug("obs in chunk: %s", obs_in_chunk)
    logging.debug("total chunk points: %s", total_chunk_points)
    
    # Generate records for single or multiple variables
    if num_vars == 1:
        # Single variable mode - use MultiVarRecordGenerator with num_vars=1 for consistency
        records, overlap_actual = MultiVarRecordGenerator.generate(
            shape=task_shape,
            overlap='random',  # Irrelevant for single variable
            num_vars=1,
            var_num_obs=np.array([obs_in_chunk]),
            var_dims_indices=[list(range(len(task_shape)))],  # All dims vary
            var_constant_dims=[[]],  # No constant dims
            var_constant_coord_indices={},  # No constant coords
            num_dims=num_dims,
            seed=seed,
            chunk_id=chunk_id,
            max_dim_size=max_dim_size,
            dim_split=dim_split
        )
        
        # Extract the single record from the dictionary
        record = records['var0']
        
        logging.debug("chunk id: %s", chunk_id)
        logging.debug("record.shape: %s", record.shape)
        logging.debug("num obs in chunk: %s", obs_in_chunk)
        logging.debug("non-nans in chunk: %s", np.sum(~np.isnan(record)))
        logging.debug("dims: %s", list(coordinates.keys()))
        logging.debug("coords: %s", coordinates)
        
        # Create DataArray with chunk-specific attributes
        chunk_attrs = NetCDFBuilder.create_default_attrs(
            num_obs, num_dims, ratio_dims, sparsity, seed
        )
        chunk_attrs["chunk_id"] = chunk_id
        chunk_attrs["description"] = "Sparse observation data (chunk)"
        
        dataarray = NetCDFBuilder.build_dataarray(
            record, coordinates, attrs=chunk_attrs
        )
        
        # Save to NetCDF
        nb_digits = len(str(ntasks))
        fpath = f"{netcdf_filepath[:-3]}_{chunk_id:0{nb_digits}d}.nc"
        NetCDFBuilder.save_to_file(dataarray, fpath, overwrite=False)
        del dataarray
        gc.collect()
        
        # Create DataFrame
        dataframe = ParquetBuilder.build_single_var_dataframe(record, coordinates)
        
        # Save to temporary parquet file
        import os
        tmp_dir = os.path.dirname(parquet_tmp)
        parquet_chunk_path = os.path.join(tmp_dir, f"chunk_{chunk_id:04d}.parquet")
        ParquetBuilder.save_to_file(dataframe, parquet_chunk_path, overwrite=False, chunk_id=None)
        
        total_obs = np.sum(~np.isnan(record))
        
    else:
        # Multi-variable mode
        records, overlap_actual = MultiVarRecordGenerator.generate(
            task_shape, overlap_target, num_vars, var_num_obs,
            var_dims_indices, var_constant_dims,
            var_constant_coord_indices, num_dims, seed,
            chunk_id, max_dim_size, dim_split
        )
        
        logging.debug("chunk id: %s", chunk_id)
        total_obs = 0
        for var_idx in range(num_vars):
            var_name = f"var_{var_idx}"
            var_obs = np.sum(~np.isnan(records[var_name]))
            total_obs += var_obs
            logging.debug("%s obs in chunk: %s", var_name, var_obs)
        
        # Create Dataset with chunk-specific attributes
        chunk_attrs = NetCDFBuilder.create_default_attrs(
            num_obs, num_dims, ratio_dims,
            float(var_sparsities[0]), seed
        )
        chunk_attrs.update({
            "chunk_id": chunk_id,
            "description": "Multi-variable sparse observation data (chunk)",
            "num_vars": num_vars,
            "var_sparsities": var_sparsities.tolist() if var_sparsities is not None else [],
            "var_num_obs": var_num_obs.tolist() if var_num_obs is not None else [],
            "overlap_target": overlap_target if isinstance(
                overlap_target, str
            ) else float(overlap_target)
        })
        
        dataset = NetCDFBuilder.build_dataset(
            records, coordinates, attrs=chunk_attrs
        )
        
        # Save to NetCDF
        nb_digits = len(str(ntasks))
        fpath = f"{netcdf_filepath[:-3]}_{chunk_id:0{nb_digits}d}.nc"
        NetCDFBuilder.save_to_file(dataset, fpath, overwrite=False)
        del dataset
        gc.collect()
        
        # Create DataFrame
        dataframe = ParquetBuilder.build_multi_var_dataframe(
            records, coordinates, num_vars, num_dims
        )
        
        # Save to temporary parquet file
        import os
        tmp_dir = os.path.dirname(parquet_tmp)
        parquet_chunk_path = os.path.join(tmp_dir, f"chunk_{chunk_id:04d}.parquet")
        ParquetBuilder.save_to_file(dataframe, parquet_chunk_path, overwrite=False, chunk_id=None)
    
    return chunk_id, total_obs, parquet_chunk_path
