"""Tests for RecordGenerator class."""

import pytest
import numpy as np
from tabray.generators.record_generator import RecordGenerator


class TestInitializeRecord:
    """Tests for initialize_record method."""

    def test_1d_shape(self):
        """Should initialize 1D array with NaN."""
        shape = (10,)
        record = RecordGenerator.initialize_record(shape)
        assert record.shape == shape
        assert np.all(np.isnan(record))

    def test_2d_shape(self):
        """Should initialize 2D array with NaN."""
        shape = (5, 8)
        record = RecordGenerator.initialize_record(shape)
        assert record.shape == shape
        assert np.all(np.isnan(record))

    def test_3d_shape(self):
        """Should initialize 3D array with NaN."""
        shape = (4, 6, 3)
        record = RecordGenerator.initialize_record(shape)
        assert record.shape == shape
        assert np.all(np.isnan(record))

    def test_all_values_nan(self):
        """Should fill entire array with NaN."""
        shape = (10, 10)
        record = RecordGenerator.initialize_record(shape)
        assert np.isnan(record).sum() == np.prod(shape)

    def test_float_dtype(self):
        """Should return float dtype."""
        shape = (5, 5)
        record = RecordGenerator.initialize_record(shape)
        assert np.issubdtype(record.dtype, np.floating)


class TestAssignObservations:
    """Tests for assign_observations method."""

    def test_values_assigned_at_indices(self):
        """Should assign values at specified indices."""
        record = np.full((5, 5), np.nan)
        multi_indices = (np.array([0, 2, 4]), np.array([1, 3, 0]))
        observations = np.array([0.5, 0.7, 0.3])
        RecordGenerator.assign_observations(record, multi_indices, observations)
        assert record[0, 1] == 0.5
        assert record[2, 3] == 0.7
        assert record[4, 0] == 0.3

    def test_other_values_remain_nan(self):
        """Should leave other positions as NaN."""
        record = np.full((5, 5), np.nan)
        multi_indices = (np.array([0, 2]), np.array([1, 3]))
        observations = np.array([0.5, 0.7])
        RecordGenerator.assign_observations(record, multi_indices, observations)
        # Count NaN values
        nan_count = np.isnan(record).sum()
        assert nan_count == 25 - 2  # 25 total - 2 assigned

    def test_modifies_in_place(self):
        """Should modify record array in-place."""
        record = np.full((3, 3), np.nan)
        original_id = id(record)
        multi_indices = (np.array([1]), np.array([1]))
        observations = np.array([0.5])
        RecordGenerator.assign_observations(record, multi_indices, observations)
        assert id(record) == original_id

    def test_correct_observation_values(self):
        """Should assign correct observation values."""
        record = np.full((4, 4), np.nan)
        multi_indices = (np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))
        observations = np.array([0.1, 0.2, 0.3, 0.4])
        RecordGenerator.assign_observations(record, multi_indices, observations)
        np.testing.assert_array_equal(np.diag(record), observations)

    def test_multiple_assignments(self):
        """Should handle multiple observations correctly."""
        record = np.full((6, 6), np.nan)
        multi_indices = (np.array([0, 1, 2, 3, 4, 5]), np.array([5, 4, 3, 2, 1, 0]))
        observations = np.arange(6) * 0.1
        RecordGenerator.assign_observations(record, multi_indices, observations)
        for i, obs in enumerate(observations):
            assert record[i, 5 - i] == obs


class TestGenerateStratifiedIndicesSubsets:
    """A subset of strata reproduces exactly the slices of a full run.

    Parallel workers each generate only their own strata, so this is what makes
    the concatenated chunks equal the serial output.
    """

    @staticmethod
    def _sorted_by_site(shape, indices, values):
        flat = np.ravel_multi_index(indices, shape)
        order = np.argsort(flat)
        return flat[order], values[order]

    @pytest.mark.parametrize(
        "shape, num_obs, split_dim",
        [
            ([6, 5], 12, 0),
            ([6, 5], 29, 1),
            ([4, 7, 3], 40, 0),
            ([4, 7, 3], 40, 1),
            ([3, 10, 5], 10, 2),
        ],
    )
    def test_partitioned_strata_equal_full_run(self, shape, num_obs, split_dim):
        seed = 7
        full_idx, full_vals = RecordGenerator.generate_stratified_indices(
            shape, num_obs, seed, split_dim
        )

        # uneven partition of the strata, as chunking can produce
        strata = list(range(shape[split_dim]))
        groups = [strata[:1], strata[1:3], strata[3:]]
        parts = [
            RecordGenerator.generate_stratified_indices(
                shape, num_obs, seed, split_dim, strata=group
            )
            for group in groups
        ]
        part_idx = tuple(
            np.concatenate([idx[dim] for idx, _ in parts]) for dim in range(len(shape))
        )
        part_vals = np.concatenate([vals for _, vals in parts])

        full_flat, full_sorted = self._sorted_by_site(shape, full_idx, full_vals)
        part_flat, part_sorted = self._sorted_by_site(shape, part_idx, part_vals)

        assert full_flat.size == num_obs
        np.testing.assert_array_equal(part_flat, full_flat)
        np.testing.assert_array_equal(part_sorted, full_sorted)
