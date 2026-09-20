"""Path management for output files.

This module handles directory creation and file path validation for
output files (NetCDF and Parquet).
"""

import os
from typing import Optional


class PathManager:
    """Manager for output file paths and directories.
    
    This class handles directory creation, path validation, and cleanup
    for output files.
    """

    @staticmethod
    def check_or_create_folder(folder_path: str, overwrite: bool = False) -> None:
        """Check if folder exists, create if needed, clean if requested.
        
        Args:
            folder_path: Path to folder
            overwrite: If True, remove existing .nc, .parquet, and _metadata files
            
        Raises:
            NotADirectoryError: If path exists but is not a directory
            ValueError: If folder exists, is not empty, and overwrite=False
            OSError: If folder cannot be created
        """
        if os.path.exists(folder_path):
            if not os.path.isdir(folder_path):
                raise NotADirectoryError(
                    f"{folder_path} exists but is not a directory"
                )
            if os.listdir(folder_path):
                if not overwrite:
                    raise ValueError(f"'{folder_path}' exists but is not empty")
                else:
                    print(
                        f"'{folder_path}' exists, removing netcdf and/or "
                        "parquet files"
                    )
                    rm_files = [
                        filename
                        for filename in os.listdir(folder_path)
                        if filename.endswith('.nc') or 
                           filename.endswith('.parquet') or 
                           filename.endswith('_metadata')
                    ]
                    for filename in sorted(rm_files):
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

    @staticmethod
    def prepare_netcdf_path(file_path: str, overwrite: bool = False) -> str:
        """Prepare directory for NetCDF file and validate path.
        
        Args:
            file_path: Path to NetCDF file
            overwrite: If True, allow overwriting existing files
            
        Returns:
            Validated file path (with .nc extension if missing)
            
        Raises:
            ValueError: If file exists and overwrite=False
        """
        if not file_path.endswith('.nc'):
            if file_path.endswith('/'):
                file_path = file_path + 'test'
            file_path = file_path + '.nc'
        
        if os.path.isfile(file_path) and not overwrite:
            raise ValueError(
                f"File {file_path} exists already. Pass "
                "overwrite=True to overwrite it."
            )
        
        netcdf_dir = os.path.dirname(file_path)
        if netcdf_dir:
            PathManager.check_or_create_folder(netcdf_dir, overwrite)
        
        return file_path

    @staticmethod
    def prepare_parquet_path(file_path: str, overwrite: bool = False) -> str:
        """Prepare directory for Parquet file.
        
        Args:
            file_path: Path to Parquet file
            overwrite: If True, allow overwriting existing files
            
        Returns:
            Validated file path
        """
        parquet_dir = os.path.dirname(file_path)
        if parquet_dir:
            PathManager.check_or_create_folder(parquet_dir, overwrite)
        
        return file_path

    @staticmethod
    def setup_output_paths(
        netcdf_filepath: Optional[str] = None,
        parquet_filepath: Optional[str] = None,
        parquet_tmp: Optional[str] = None,
        overwrite: bool = True,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Setup all output paths for a generation run.
        
        Args:
            netcdf_filepath: Path for NetCDF output
            parquet_filepath: Path for Parquet output
            parquet_tmp: Path for temporary Parquet files
            overwrite: If True, allow overwriting existing files
            
        Returns:
            Tuple of (netcdf_filepath, parquet_filepath, parquet_tmp)
        """

        if netcdf_filepath is None:
            netcdf_filepath = "./test_nc/"
        if parquet_filepath is None:
            parquet_filepath = "./test_parquet/"
        if parquet_tmp is None:
            parquet_dir = os.path.dirname(parquet_filepath) or '.'
            parquet_base = os.path.basename(parquet_filepath).replace('.parquet', '')
            parquet_tmp = os.path.join(parquet_dir, f"tmp_{parquet_base}")

        print(f"Setting up netcdf paths {netcdf_filepath}")
        netcdf_filepath = PathManager.prepare_netcdf_path(
            netcdf_filepath, overwrite
        )

        print(f"Setting up parquet paths {parquet_filepath}")
        parquet_filepath = PathManager.prepare_parquet_path(
            parquet_filepath, overwrite
        )

        # parquet_tmp names the scratch DIRECTORY itself, not a file inside
        # one -- taking its dirname lands on the real output directory, which
        # the scratch cleanup then deletes.
        print(f"Setting up temporary parquet directory {parquet_tmp}")
        os.makedirs(parquet_tmp, exist_ok=True)

        return netcdf_filepath, parquet_filepath, parquet_tmp

    SCRATCH_PATTERNS = ("chunk_", "_metadata", "_common_metadata")

    @staticmethod
    def remove_scratch_dir(dir_path: str) -> bool:
        """Delete a scratch directory, but only if it holds nothing else.

        Refuses to remove anything containing a file this run did not write, or
        a subdirectory. A mistyped path then fails loudly instead of destroying
        data: the previous code called shutil.rmtree on whatever it was given.

        Args:
            dir_path: Directory to remove

        Returns:
            True if it was removed, False if it was kept
        """
        import shutil

        if not dir_path or not os.path.isdir(dir_path):
            return False
        unexpected = []
        for entry in os.listdir(dir_path):
            full = os.path.join(dir_path, entry)
            if os.path.isdir(full):
                unexpected.append(entry + "/")
            elif not entry.startswith(PathManager.SCRATCH_PATTERNS):
                unexpected.append(entry)
        if unexpected:
            print(
                f"Keeping {dir_path}: it holds files this run did not write "
                f"({', '.join(sorted(unexpected)[:5])}). Remove it by hand if "
                "that is what you want."
            )
            return False
        shutil.rmtree(dir_path)
        return True
