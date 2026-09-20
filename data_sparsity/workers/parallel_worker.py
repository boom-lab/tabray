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
    density: float,
    num_vars: int,
    num_dims: int,
    ratio_dims: Tuple[float, ...],
    num_obs: int,
    var_densities: Optional[ArrayLike],
    var_num_obs: Optional[ArrayLike],
    var_dims_indices: Optional[List[List[int]]],
    var_constant_dims: Optional[List[List[int]]],
    var_constant_coord_indices: Optional[Dict[int, Dict[int, int]]],
    overlap_target: Union[float, str, List[float]],
    dim_split: int,
    max_dim_size: int,
    div_points: List[int],
    section_sizes: List[int],
    netcdf_filepath: str,
    parquet_tmp: str,
    ntasks: int,
    num_obs_global: Optional[int] = None,
    fixed_overlap: Union[bool, List[bool]] = False,
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
        density: Density level (0-1)
        num_vars: Number of variables (1 for single-var)
        num_dims: Number of dimensions
        ratio_dims: Ratio of dimensions
        num_obs: Total number of observations across all chunks
        var_densities: Array of density per variable (multi-var only)
        var_num_obs: Array of observation counts per variable (multi-var only)
        var_dims_indices: List of dimension indices per variable (multi-var only)
        var_constant_dims: List of constant dimensions per variable (multi-var only)
        var_constant_coord_indices: Dict of constant coordinates (multi-var only)
        overlap_target: Target overlap fraction or 'maximal' (multi-var only)
        fixed_overlap: Whether overlapping sites should be shared across
            variables that opt in
        dim_split: Dimension along which to split chunks
        max_dim_size: Maximum size of the split dimension
        div_points: Division points for chunks along split dimension
        section_sizes: Size of each chunk along split dimension
        netcdf_filepath: Base path for NetCDF output
        parquet_tmp: Path template for temporary parquet chunks
        ntasks: Total number of tasks (for formatting)
        num_obs_global: Total observations globally (for LHS filtering and RNG advancement)
        
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
    
    # Determine chunk dimensions and range along split dimension FIRST
    task_range = (div_points[chunk_id], div_points[chunk_id + 1])
    task_size = section_sizes[chunk_id]
    task_shape = ChunkUtils.update_chunk_shape(shape, dim_split, task_size)
    
    logging.debug("task_range: %s", task_range)
    logging.debug("task_size: %s", task_size)
    logging.debug("task_shape: %s", task_shape)
    
    total_chunk_points = ChunkUtils.validate_chunk_points(task_shape)
    logging.debug("total_chunk_points: %s", total_chunk_points)
    
    # Generate dimension-specific RNGs for coordinates (no chunk-specific parameters)
    # All chunks use the same base RNGs to ensure coordinate alignment
    coord_dim_rngs = ChunkUtils.generate_rngs(
        seed=seed,
        num_dims=num_dims
    )
    
    # Draw every coordinate axis exactly as the serial path does, over the GLOBAL
    # shape, then keep this chunk's slice of the split axis. The axis is sorted,
    # so elements [task_range[0]:task_range[1]] are precisely the coordinates
    # whose global index falls in this chunk -- the same index partition that the
    # site filtering uses. Concatenating the chunks in order therefore reproduces
    # the serial axis exactly, and every non-split axis is identical in every file.
    #
    # Drawing only this chunk's values within its own value sub-range would
    # instead stratify the axis: each chunk would hold exactly section_sizes[k]
    # coordinates in its interval, where the serial draw gives a Binomial count.
    coordinates = CoordinateGenerator.generate_all_coords(
        list(shape),
        rng=None,  # unused when dim_rngs is supplied
        dim_ranges=None,
        dim_rngs=coord_dim_rngs,
    )
    split_dim_name = f"x{dim_split}"
    coordinates[split_dim_name] = coordinates[split_dim_name][
        task_range[0]:task_range[1]
    ]
    
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
            dim_split=dim_split,
            lhs_shape=list(shape),  # Pass global shape for LHS
            num_obs_global=num_obs_global,  # Pass global observation count
            div_points=div_points  # Pass division points for chunk filtering
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
            num_obs, num_dims, ratio_dims, density, seed
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
        dataframe = ParquetBuilder.build_single_var_dataframe(
            record, coordinates, order_dim=dim_split
        )
        
        # Save to temporary parquet file
        import os
        tmp_dir = os.path.dirname(parquet_tmp)
        parquet_chunk_path = os.path.join(tmp_dir, f"chunk_{chunk_id:04d}.parquet")
        ParquetBuilder.save_to_file(dataframe, parquet_chunk_path, overwrite=False, chunk_id=None)
        
        total_obs = np.sum(~np.isnan(record))
        
    else:
        # Multi-variable mode
        # Generate the chunk directly so the workflow can scale past RAM.
        chunk_var_num_obs = np.asarray(var_num_obs, dtype=int)
        records, overlap_actual = MultiVarRecordGenerator.generate(
            task_shape, overlap_target, num_vars, chunk_var_num_obs,
            var_dims_indices, var_constant_dims,
            var_constant_coord_indices, num_dims, seed,
            chunk_id=chunk_id,
            max_dim_size=max_dim_size,
            dim_split=dim_split,
            lhs_shape=list(shape),        # GLOBAL shape
            div_points=div_points,        # which strata belong to this chunk
            num_obs_global=num_obs_global,
            fixed_overlap=fixed_overlap
        )

        logging.debug("chunk id: %s", chunk_id)
        # Count what was actually placed. var_num_obs is the GLOBAL per-variable
        # count -- the strata apportion it internally -- so the chunk's own
        # figures have to be measured, not inherited from the arguments.
        chunk_var_counts = []
        total_obs = 0
        for var_idx in range(num_vars):
            var_name = f"var{var_idx}"
            var_obs = int(np.sum(~np.isnan(records[var_name])))
            chunk_var_counts.append(var_obs)
            total_obs += var_obs
            logging.debug("%s obs in chunk: %s", var_name, var_obs)

        # Create Dataset with chunk-specific attributes. The density recorded is
        # the reference variable's within this chunk, which is a meaningful
        # quantity; summing every variable's observations over one grid was not.
        # Same quantities as serial: num_obs and density describe the reference
        # variable within this chunk, per-variable counts are measured.
        chunk_attrs = NetCDFBuilder.create_default_attrs(
            chunk_var_counts[0], num_dims, ratio_dims,
            float(chunk_var_counts[0] / np.prod(task_shape)), seed
        )
        chunk_attrs.update(NetCDFBuilder.create_multivar_attrs(
            num_vars=num_vars,
            var_densities=var_densities,
            var_num_obs=chunk_var_counts,
            overlap_target=overlap_target,
            fixed_overlap=fixed_overlap,
        ))
        chunk_attrs.update({
            "chunk_id": chunk_id,
            "description": "Multi-variable sparse observation data (chunk)",
        })
        # Achieved overlap is deliberately not recorded here: a chunk only sees
        # its own strata, so any figure it computed would be chunk-local and
        # would not aggregate to the dataset's overlap. It belongs to the merge.
        
        dataset = NetCDFBuilder.build_dataset(
            records,
            coordinates,
            attrs=chunk_attrs,
            var_constant_dims=var_constant_dims,
            squeeze_constant_dims=False,
        )
        
        # Save to NetCDF
        nb_digits = len(str(ntasks))
        fpath = f"{netcdf_filepath[:-3]}_{chunk_id:0{nb_digits}d}.nc"
        NetCDFBuilder.save_to_file(dataset, fpath, overwrite=False)
        del dataset
        gc.collect()
        
        # Create DataFrame
        dataframe = ParquetBuilder.build_multi_var_dataframe(
            records, coordinates, num_vars, num_dims, order_dim=dim_split
        )
        
        # Save to temporary parquet file
        import os
        tmp_dir = os.path.dirname(parquet_tmp)
        parquet_chunk_path = os.path.join(tmp_dir, f"chunk_{chunk_id:04d}.parquet")
        ParquetBuilder.save_to_file(dataframe, parquet_chunk_path, overwrite=False, chunk_id=None)
    
    return chunk_id, total_obs, parquet_chunk_path
