"""Tests for PathManager class."""

import pytest
import os
import tempfile
import shutil
from data_sparsity.output.path_manager import PathManager


class TestCheckOrCreateFolder:
    """Tests for check_or_create_folder method."""
    
    def test_creates_non_existent_folder(self, temp_dir):
        """Should create folder when it doesn't exist."""
        new_folder = os.path.join(temp_dir, "new_folder")
        PathManager.check_or_create_folder(new_folder)
        assert os.path.exists(new_folder)
        assert os.path.isdir(new_folder)
    
    def test_existing_empty_folder_passes(self, temp_dir):
        """Should accept existing empty folder."""
        # Should not raise
        PathManager.check_or_create_folder(temp_dir)
    
    def test_existing_non_empty_without_overwrite_raises(self, temp_dir):
        """Should raise ValueError for non-empty folder without overwrite."""
        # Create a file in the folder
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        
        with pytest.raises(ValueError, match="exists but is not empty"):
            PathManager.check_or_create_folder(temp_dir, overwrite=False)
    
    def test_existing_non_empty_with_overwrite_cleans(self, temp_dir):
        """Should clean .nc and .parquet files with overwrite=True."""
        # Create test files
        nc_file = os.path.join(temp_dir, "data.nc")
        pq_file = os.path.join(temp_dir, "data.parquet")
        meta_file = os.path.join(temp_dir, "_metadata")
        txt_file = os.path.join(temp_dir, "keep.txt")
        
        for f in [nc_file, pq_file, meta_file, txt_file]:
            with open(f, 'w') as file:
                file.write("test")
        
        PathManager.check_or_create_folder(temp_dir, overwrite=True)
        
        # .nc, .parquet, and _metadata should be removed
        assert not os.path.exists(nc_file)
        assert not os.path.exists(pq_file)
        assert not os.path.exists(meta_file)
        # Other files should remain
        assert os.path.exists(txt_file)
    
    def test_removes_nc_files(self, temp_dir):
        """Should remove .nc files."""
        nc_file = os.path.join(temp_dir, "data.nc")
        with open(nc_file, 'w') as f:
            f.write("test")
        
        PathManager.check_or_create_folder(temp_dir, overwrite=True)
        assert not os.path.exists(nc_file)
    
    def test_removes_parquet_files(self, temp_dir):
        """Should remove .parquet files."""
        pq_file = os.path.join(temp_dir, "data.parquet")
        with open(pq_file, 'w') as f:
            f.write("test")
        
        PathManager.check_or_create_folder(temp_dir, overwrite=True)
        assert not os.path.exists(pq_file)
    
    def test_removes_metadata_files(self, temp_dir):
        """Should remove _metadata files."""
        meta_file = os.path.join(temp_dir, "_metadata")
        with open(meta_file, 'w') as f:
            f.write("test")
        
        PathManager.check_or_create_folder(temp_dir, overwrite=True)
        assert not os.path.exists(meta_file)
    
    def test_keeps_other_files(self, temp_dir):
        """Should keep files with other extensions."""
        txt_file = os.path.join(temp_dir, "keep.txt")
        csv_file = os.path.join(temp_dir, "keep.csv")
        
        for f in [txt_file, csv_file]:
            with open(f, 'w') as file:
                file.write("test")
        
        PathManager.check_or_create_folder(temp_dir, overwrite=True)
        assert os.path.exists(txt_file)
        assert os.path.exists(csv_file)
    
    def test_not_directory_raises(self, temp_dir):
        """Should raise NotADirectoryError if path is a file."""
        file_path = os.path.join(temp_dir, "file.txt")
        with open(file_path, 'w') as f:
            f.write("test")
        
        with pytest.raises(NotADirectoryError):
            PathManager.check_or_create_folder(file_path)
    
    def test_nested_folder_creation(self, temp_dir):
        """Should create nested directories."""
        nested_path = os.path.join(temp_dir, "level1", "level2", "level3")
        PathManager.check_or_create_folder(nested_path)
        assert os.path.exists(nested_path)
        assert os.path.isdir(nested_path)


class TestPrepareNetcdfPath:
    """Tests for prepare_netcdf_path method."""
    
    def test_adds_nc_extension(self, temp_dir):
        """Should add .nc extension if missing."""
        base_path = os.path.join(temp_dir, "data")
        result = PathManager.prepare_netcdf_path(base_path, overwrite=False)
        assert result.endswith(".nc")
    
    def test_keeps_nc_extension(self, temp_dir):
        """Should keep .nc extension if present."""
        path_with_nc = os.path.join(temp_dir, "data.nc")
        result = PathManager.prepare_netcdf_path(path_with_nc, overwrite=False)
        assert result == path_with_nc
    
    def test_creates_parent_directory(self, temp_dir):
        """Should create parent directory if needed."""
        nested_path = os.path.join(temp_dir, "subdir", "data.nc")
        PathManager.prepare_netcdf_path(nested_path, overwrite=False)
        assert os.path.exists(os.path.dirname(nested_path))
    
    def test_existing_file_without_overwrite_raises(self, temp_dir):
        """Should raise ValueError if file exists and overwrite=False."""
        file_path = os.path.join(temp_dir, "data.nc")
        with open(file_path, 'w') as f:
            f.write("test")
        
        with pytest.raises(ValueError, match="exists already"):
            PathManager.prepare_netcdf_path(file_path, overwrite=False)
    
    def test_existing_file_with_overwrite_passes(self, temp_dir):
        """Should pass if file exists and overwrite=True."""
        file_path = os.path.join(temp_dir, "data.nc")
        with open(file_path, 'w') as f:
            f.write("test")
        
        # Should not raise
        result = PathManager.prepare_netcdf_path(file_path, overwrite=True)
        assert result == file_path
    
    def test_returns_corrected_path(self, temp_dir):
        """Should return corrected path with .nc extension."""
        base_path = os.path.join(temp_dir, "data")
        result = PathManager.prepare_netcdf_path(base_path, overwrite=False)
        assert result == base_path + ".nc"


class TestPrepareParquetPath:
    """Tests for prepare_parquet_path method."""
    
    def test_creates_parent_directory(self, temp_dir):
        """Should create parent directory if needed."""
        nested_path = os.path.join(temp_dir, "subdir", "data")
        PathManager.prepare_parquet_path(nested_path)
        assert os.path.exists(os.path.dirname(nested_path))
    
    def test_no_extension_added(self, temp_dir):
        """Should not add extension (unlike NetCDF)."""
        base_path = os.path.join(temp_dir, "data")
        result = PathManager.prepare_parquet_path(base_path)
        assert result == base_path
        assert not result.endswith(".parquet")
    
    def test_returns_path_unchanged(self, temp_dir):
        """Should return path unchanged."""
        path = os.path.join(temp_dir, "data.parquet")
        result = PathManager.prepare_parquet_path(path)
        assert result == path
