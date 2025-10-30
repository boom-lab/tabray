#!/usr/bin/env python3

import os
from typing import Any, Dict, List, Tuple

def check_or_create_folder(folder_path: str, overwrite: bool = False) -> None:
    """Check if folder exists, if not, create it."""
    if os.path.exists(folder_path):
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"{folder_path} exists but is not a directory")
        if os.listdir(folder_path):
            if not overwrite:
                raise ValueError(f"'{folder_path}' exists but is not empty")
            else:
                print(f"'{folder_path}' exists, removing netcdf and/or parquet files")
                rm_files = [
                    filename
                    for filename in os.listdir(folder_path)
                    if filename.endswith('.nc') or filename.endswith('.parquet') or filename.endswith('_metadata')
                ]
                for filename in rm_files.sort():
                    file_path = os.path.join(folder_path, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.remove(file_path)
                            print(f'  Deleted file: {file_path}')
                    except OSError as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        try:
            os.makedirs(folder_path, exist_ok=True)
        except OSError as e:
            raise OSError(f"Could not create {folder_path}: {e}") from e


def check_nc(file_path: str, overwrite: bool = False) -> None:
    """Check if tree to path exists or generate it"""
    if not file_path.endswith('.nc'):
        file_path = file_path+'.nc'
    if os.path.isfile(file_path) and not overwrite:
        raise ValueError(
            f"File {file_path} exists already. Pass "
            "overwrite=True to overwrite it."
        )

    netcdf_dir = os.path.dirname(file_path)
    # create folder to store nc file in
    check_or_create_folder(
        netcdf_dir,
        overwrite
    )


def check_parquet(file_path: str, overwrite: bool = False) -> None:
    """Check if tree to path exists or generate it"""
    # create folder to store parquet dataset in
    parquet_dir = os.path.dirname(file_path)
    check_or_create_folder(
        parquet_dir,
        overwrite
    )


def set_up_paths(netcdf_filepath: str = None, parquet_filepath: str = None, parquet_tmp: str = None) -> None:

    print(f"Setting up netcdf paths {netcdf_filepath}")
    check_nc(
        netcdf_filepath,
        overwrite=True
    )

    print(f"Setting up parquet paths {parquet_filepath}")
    check_parquet(
        parquet_filepath,
        overwrite=True
    )

    print(f"Setting up temporary parquet paths {parquet_tmp}")
    check_parquet(
        parquet_tmp,
        overwrite=True
    )
