#!/usr/bin/env python3

import os
from typing import Any, Dict, List, Tuple


def check_or_create_folder(folder_path: str, overwrite: bool = False) -> None:
    """Check if folder exists, if not, create it."""
    if os.path.exists(folder_path):
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"{folder_path} exists but is not a directory")
        if os.listdir(folder_path) and not overwrite:
            raise ValueError(f"'{folder_path}' exists but is not empty")
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

    # create folder to store nc file in
    check_or_create_folder(
        os.path.dirname(file_path),
        overwrite
    )


def check_parquet(file_path: str, overwrite: bool = False) -> None:
    """Check if tree to path exists or generate it"""
    # create folder to store parquet dataset in
    check_or_create_folder(
        os.path.dirname(file_path),
        overwrite
    )
