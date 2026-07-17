"""Shared pytest fixtures for data_sparsity tests."""

import pytest
import numpy as np
import tempfile
import shutil
import os


@pytest.fixture
def fixed_rng():
    """Provide a fixed random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after test."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_coordinates_2d():
    """Provide sample 2D coordinates."""
    return {
        'x0': np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
        'x1': np.array([0.2, 0.4, 0.6, 0.8])
    }


@pytest.fixture
def sample_coordinates_3d():
    """Provide sample 3D coordinates."""
    return {
        'x0': np.array([0.1, 0.5, 0.9]),
        'x1': np.array([0.2, 0.6]),
        'x2': np.array([0.3, 0.7])
    }


@pytest.fixture
def sample_record_2d():
    """Provide sample 2D record with sparse data."""
    record = np.full((5, 4), np.nan)
    record[0, 0] = 0.5
    record[2, 1] = 0.7
    record[4, 3] = 0.3
    return record


@pytest.fixture
def sample_record_3d():
    """Provide sample 3D record with sparse data."""
    record = np.full((3, 2, 2), np.nan)
    record[0, 0, 0] = 0.1
    record[1, 1, 0] = 0.5
    record[2, 0, 1] = 0.9
    return record


@pytest.fixture
def sample_multi_var_records_2d():
    """Provide sample multi-variable 2D records."""
    records = {}
    
    # Variable 0
    rec0 = np.full((4, 4), np.nan)
    rec0[0, 0] = 0.1
    rec0[1, 1] = 0.2
    rec0[2, 2] = 0.3
    records['var0'] = rec0
    
    # Variable 1
    rec1 = np.full((4, 4), np.nan)
    rec1[0, 0] = 0.4  # Overlaps with var0
    rec1[1, 2] = 0.5
    rec1[3, 3] = 0.6
    records['var1'] = rec1
    
    return records
