"""Utility functions for data_sparsity module.

This module provides helper functions for file/folder validation,
S3 path detection, and AWS configuration management.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import s3fs
import yaml


def is_s3_path(path: str) -> bool:
    """Check if a path is an S3 path.

    Args:
        path: Path to check

    Returns:
        True if path starts with 's3://', False otherwise
    """
    return path.startswith('s3://')


def load_aws_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load AWS configuration from YAML file.

    AWS credentials and configuration can be provided via:
    1. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.)
    2. YAML configuration file (specified by config_path)
    3. AWS credentials file (~/.aws/credentials)

    Environment variables take precedence over the config file.

    Args:
        config_path: Path to AWS config YAML file. If None, looks for
                    'aws_config.yaml' in the current directory.

    Returns:
        Dictionary with AWS configuration settings
    """
    config = {}

    # Try to load from config file if it exists
    if config_path is None:
        config_path = 'aws_config.yaml'

    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            file_config = yaml.safe_load(f) or {}
            config.update(file_config)

    # Environment variables override config file
    env_mappings = {
        'AWS_ACCESS_KEY_ID': 'aws_access_key_id',
        'AWS_SECRET_ACCESS_KEY': 'aws_secret_access_key',
        'AWS_SESSION_TOKEN': 'aws_session_token',
        'AWS_DEFAULT_REGION': 'aws_default_region',
        'AWS_PROFILE': 'aws_profile',
    }

    for env_var, config_key in env_mappings.items():
        env_value = os.environ.get(env_var)
        if env_value:
            config[config_key] = env_value

    return config


def get_s3_filesystem(config_path: Optional[str] = None) -> s3fs.S3FileSystem:
    """Get configured S3 filesystem object.

    Args:
        config_path: Path to AWS config YAML file

    Returns:
        Configured S3FileSystem instance
    """
    config = load_aws_config(config_path)

    # Build s3fs configuration
    s3fs_config = {}

    if 'aws_access_key_id' in config:
        s3fs_config['key'] = config['aws_access_key_id']
    if 'aws_secret_access_key' in config:
        s3fs_config['secret'] = config['aws_secret_access_key']
    if 'aws_session_token' in config:
        s3fs_config['token'] = config['aws_session_token']
    if 'aws_profile' in config:
        s3fs_config['profile'] = config['aws_profile']
    if 's3_endpoint_url' in config:
        s3fs_config['endpoint_url'] = config['s3_endpoint_url']
    if 's3_use_ssl' in config:
        s3fs_config['use_ssl'] = config['s3_use_ssl']

    # Let boto3 handle default region from environment or ~/.aws/config
    # Only set if explicitly provided
    if 'aws_default_region' in config:
        s3fs_config['client_kwargs'] = {'region_name': config['aws_default_region']}

    return s3fs.S3FileSystem(**s3fs_config)


def check_or_create_folder(
    folder_path: str,
    overwrite: bool = False,
    s3_filesystem: Optional[s3fs.S3FileSystem] = None
) -> None:
    """Check if folder exists, if not, create it.

    Supports both local filesystem and S3 paths.

    Args:
        folder_path: Path to folder (local or S3)
        overwrite: If True, allow non-empty folders
        s3_filesystem: Optional S3FileSystem instance for S3 paths
    """
    if is_s3_path(folder_path):
        if s3_filesystem is None:
            s3_filesystem = get_s3_filesystem()

        # S3 doesn't have explicit directory creation
        # Directories are created implicitly when files are uploaded
        # Just check if path exists and is not empty if overwrite is False
        if not overwrite:
            try:
                files = s3_filesystem.ls(folder_path)
                if files:
                    raise ValueError(
                        f"S3 path '{folder_path}' exists and is not empty"
                    )
            except FileNotFoundError:
                # Path doesn't exist, which is fine
                pass
    else:
        # Local filesystem
        if os.path.exists(folder_path):
            if not os.path.isdir(folder_path):
                raise NotADirectoryError(
                    f"{folder_path} exists but is not a directory"
                )
            if os.listdir(folder_path) and not overwrite:
                raise ValueError(f"'{folder_path}' exists but is not empty")
        else:
            try:
                os.makedirs(folder_path, exist_ok=True)
            except OSError as e:
                raise OSError(f"Could not create {folder_path}: {e}") from e


def check_nc(
    file_path: str,
    overwrite: bool = False,
    s3_filesystem: Optional[s3fs.S3FileSystem] = None
) -> None:
    """Check if tree to path exists or generate it.

    Supports both local filesystem and S3 paths.

    Args:
        file_path: Path to NetCDF file (local or S3)
        overwrite: If True, allow overwriting existing files
        s3_filesystem: Optional S3FileSystem instance for S3 paths
    """
    if not file_path.endswith('.nc'):
        file_path = file_path + '.nc'

    if is_s3_path(file_path):
        if s3_filesystem is None:
            s3_filesystem = get_s3_filesystem()

        if s3_filesystem.exists(file_path) and not overwrite:
            raise ValueError(
                f"File {file_path} exists already. Pass "
                "overwrite=True to overwrite it."
            )

        # For S3, ensure parent "directory" exists (conceptually)
        parent_dir = str(Path(file_path).parent)
        check_or_create_folder(parent_dir, overwrite, s3_filesystem)
    else:
        # Local filesystem
        if os.path.isfile(file_path) and not overwrite:
            raise ValueError(
                f"File {file_path} exists already. Pass "
                "overwrite=True to overwrite it."
            )

        # Create folder to store nc file in
        check_or_create_folder(
            os.path.dirname(file_path),
            overwrite
        )


def check_parquet(
    file_path: str,
    overwrite: bool = False,
    s3_filesystem: Optional[s3fs.S3FileSystem] = None
) -> None:
    """Check if tree to path exists or generate it.

    Supports both local filesystem and S3 paths.

    Args:
        file_path: Path to Parquet directory (local or S3)
        overwrite: If True, allow non-empty directories
        s3_filesystem: Optional S3FileSystem instance for S3 paths
    """
    # Create folder to store parquet dataset in
    check_or_create_folder(
        os.path.dirname(file_path) if not is_s3_path(file_path) else str(Path(file_path).parent),
        overwrite,
        s3_filesystem
    )
