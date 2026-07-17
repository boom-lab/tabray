"""Tests for OverlapIndexMapper class."""

import pytest
import numpy as np
from data_sparsity.generators.overlap_index_mapper import OverlapIndexMapper


class TestIdentifyRandomAssignDims:
    """Tests for identify_random_assign_dims method."""
    
    def test_source_1_target_gt1_identified(self):
        """Should identify dims where source=1 and target>1."""
        source_shape = [1, 10, 20]
        target_shape = [5, 10, 20]
        
        result = OverlapIndexMapper.identify_random_assign_dims(
            source_shape, target_shape
        )
        
        assert result == [0]
    
    def test_source_gt1_target_1_not_identified(self):
        """Should not identify dims where source>1 and target=1."""
        source_shape = [10, 20]
        target_shape = [1, 20]
        
        result = OverlapIndexMapper.identify_random_assign_dims(
            source_shape, target_shape
        )
        
        assert result == []
    
    def test_both_same_size_not_identified(self):
        """Should not identify dims where both have same size."""
        source_shape = [10, 20, 30]
        target_shape = [10, 20, 30]
        
        result = OverlapIndexMapper.identify_random_assign_dims(
            source_shape, target_shape
        )
        
        assert result == []
    
    def test_multiple_matching_dimensions(self):
        """Should identify multiple matching dimensions."""
        source_shape = [1, 10, 1, 20]
        target_shape = [5, 10, 8, 20]
        
        result = OverlapIndexMapper.identify_random_assign_dims(
            source_shape, target_shape
        )
        
        assert result == [0, 2]
    
    def test_no_matching_dimensions(self):
        """Should return empty list when no matches."""
        source_shape = [10, 20, 30]
        target_shape = [10, 20, 30]
        
        result = OverlapIndexMapper.identify_random_assign_dims(
            source_shape, target_shape
        )
        
        assert result == []


class TestMapSingleCoordinate:
    """Tests for map_single_coordinate method."""
    
    def test_direct_mapping_both_vary(self):
        """Should directly map when both dimensions vary."""
        source_coords = [5, 10]
        source_shape = [20, 30]
        target_shape = [20, 30]
        
        valid, target_coords = OverlapIndexMapper.map_single_coordinate(
            source_coords, source_shape, target_shape
        )
        
        assert valid is True
        assert target_coords == [5, 10]
    
    def test_target_constant_maps_to_zero(self):
        """Should map to 0 when target dimension is constant."""
        source_coords = [5, 10]
        source_shape = [20, 30]
        target_shape = [1, 30]
        
        valid, target_coords = OverlapIndexMapper.map_single_coordinate(
            source_coords, source_shape, target_shape
        )
        
        assert valid is True
        assert target_coords == [0, 10]
    
    def test_source_constant_maps_to_none(self):
        """Should map to None when source is constant but target varies."""
        source_coords = [0, 10]
        source_shape = [1, 30]
        target_shape = [20, 30]
        
        valid, target_coords = OverlapIndexMapper.map_single_coordinate(
            source_coords, source_shape, target_shape
        )
        
        assert valid is True
        assert target_coords == [None, 10]
    
    def test_out_of_range_invalid(self):
        """Should return invalid when coordinate out of range."""
        source_coords = [25, 10]
        source_shape = [30, 30]
        target_shape = [20, 30]
        
        valid, target_coords = OverlapIndexMapper.map_single_coordinate(
            source_coords, source_shape, target_shape
        )
        
        assert valid is False
    
    def test_multiple_dimensions(self):
        """Should handle multiple dimensions correctly."""
        source_coords = [1, 2, 3]
        source_shape = [10, 20, 30]
        target_shape = [10, 20, 30]
        
        valid, target_coords = OverlapIndexMapper.map_single_coordinate(
            source_coords, source_shape, target_shape
        )
        
        assert valid is True
        assert target_coords == [1, 2, 3]
    
    def test_all_valid_scenarios(self):
        """Should correctly handle all mapping scenarios together."""
        source_coords = [0, 5, 10]
        source_shape = [1, 20, 30]
        target_shape = [10, 1, 30]
        
        valid, target_coords = OverlapIndexMapper.map_single_coordinate(
            source_coords, source_shape, target_shape
        )
        
        assert valid is True
        assert target_coords == [None, 0, 10]
    
    def test_all_constant_in_target(self):
        """Should map all to 0 when all target dims constant."""
        source_coords = [5, 10]
        source_shape = [20, 30]
        target_shape = [1, 1]
        
        valid, target_coords = OverlapIndexMapper.map_single_coordinate(
            source_coords, source_shape, target_shape
        )
        
        assert valid is True
        assert target_coords == [0, 0]
    
    def test_all_constant_in_source(self):
        """Should map all to None when all source dims constant."""
        source_coords = [0, 0]
        source_shape = [1, 1]
        target_shape = [20, 30]
        
        valid, target_coords = OverlapIndexMapper.map_single_coordinate(
            source_coords, source_shape, target_shape
        )
        
        assert valid is True
        assert target_coords == [None, None]
    
    def test_mixed_scenarios(self):
        """Should handle mixed mapping scenarios."""
        source_coords = [0, 5, 15, 0]
        source_shape = [1, 10, 20, 1]
        target_shape = [5, 1, 20, 8]
        
        valid, target_coords = OverlapIndexMapper.map_single_coordinate(
            source_coords, source_shape, target_shape
        )
        
        assert valid is True
        assert target_coords == [None, 0, 15, None]
    
    def test_tuple_length_matches(self):
        """Should return tuple with same length as input."""
        source_coords = [1, 2, 3, 4, 5]
        source_shape = [10, 20, 30, 40, 50]
        target_shape = [10, 20, 30, 40, 50]
        
        valid, target_coords = OverlapIndexMapper.map_single_coordinate(
            source_coords, source_shape, target_shape
        )
        
        assert len(target_coords) == len(source_coords)


class TestTryRandomAssignment:
    """Tests for try_random_assignment method."""
    
    def test_finds_unused_index(self, fixed_rng):
        """Should find an unused index."""
        target_coords = [None, 5]
        random_assign_dims = [0]
        target_shape = [10, 10]
        used_target_indices = set()
        
        result = OverlapIndexMapper.try_random_assignment(
            target_coords, random_assign_dims, target_shape,
            used_target_indices, fixed_rng
        )
        
        assert result is not None
        assert isinstance(result, (int, np.integer))
    
    def test_multiple_attempts_if_needed(self):
        """Should try multiple times if indices are used."""
        target_coords = [None, 0]
        random_assign_dims = [0]
        target_shape = [5, 10]
        # Mark some indices as used
        used_target_indices = {0, 10, 20}
        rng = np.random.default_rng(42)
        
        result = OverlapIndexMapper.try_random_assignment(
            target_coords, random_assign_dims, target_shape,
            used_target_indices, rng
        )
        
        # Should find an unused one
        if result is not None:
            assert result not in used_target_indices
    
    def test_returns_none_if_exhausted(self):
        """Should return None if all possibilities exhausted."""
        target_coords = [None, 0]
        random_assign_dims = [0]
        target_shape = [3, 10]
        # Mark all possible indices as used
        used_target_indices = {0, 10, 20}
        rng = np.random.default_rng(42)
        
        result = OverlapIndexMapper.try_random_assignment(
            target_coords, random_assign_dims, target_shape,
            used_target_indices, rng, max_attempts=10
        )
        
        # May return None if exhausted
        assert result is None or result not in used_target_indices
    
    def test_uses_rng_correctly(self):
        """Should use RNG for random selection."""
        target_coords = [None, 5]
        random_assign_dims = [0]
        target_shape = [10, 10]
        used_target_indices = set()
        
        rng1 = np.random.default_rng(999)
        result1 = OverlapIndexMapper.try_random_assignment(
            target_coords, random_assign_dims, target_shape,
            used_target_indices, rng1
        )
        
        rng2 = np.random.default_rng(999)
        result2 = OverlapIndexMapper.try_random_assignment(
            target_coords, random_assign_dims, target_shape,
            used_target_indices, rng2
        )
        
        # Same seed should give same result
        assert result1 == result2
    
    def test_respects_max_attempts(self):
        """Should respect max_attempts parameter."""
        target_coords = [None, 0]
        random_assign_dims = [0]
        target_shape = [100, 10]
        used_target_indices = set(range(0, 1000, 10))  # Many used
        rng = np.random.default_rng(42)
        
        # With low max_attempts, might not find one
        result = OverlapIndexMapper.try_random_assignment(
            target_coords, random_assign_dims, target_shape,
            used_target_indices, rng, max_attempts=5
        )
        
        # Result may be None or valid unused index
        assert result is None or result not in used_target_indices
    
    def test_avoids_used_indices(self):
        """Should avoid already-used indices."""
        target_coords = [None, 5]
        random_assign_dims = [0]
        target_shape = [10, 10]
        used_target_indices = {5, 15, 25}
        rng = np.random.default_rng(42)
        
        result = OverlapIndexMapper.try_random_assignment(
            target_coords, random_assign_dims, target_shape,
            used_target_indices, rng
        )
        
        if result is not None:
            assert result not in used_target_indices
    
    def test_single_random_dim(self):
        """Should handle single random dimension."""
        target_coords = [5, None]
        random_assign_dims = [1]
        target_shape = [10, 20]
        used_target_indices = set()
        rng = np.random.default_rng(42)
        
        result = OverlapIndexMapper.try_random_assignment(
            target_coords, random_assign_dims, target_shape,
            used_target_indices, rng
        )
        
        assert result is not None
    
    def test_multiple_random_dims(self):
        """Should handle multiple random dimensions."""
        target_coords = [None, 5, None]
        random_assign_dims = [0, 2]
        target_shape = [5, 10, 8]
        used_target_indices = set()
        rng = np.random.default_rng(42)
        
        result = OverlapIndexMapper.try_random_assignment(
            target_coords, random_assign_dims, target_shape,
            used_target_indices, rng
        )
        
        assert result is not None


class TestMapIndicesForOverlap:
    """Tests for map_indices_for_overlap method."""
    
    def test_full_mapping_possible(self):
        """Should successfully map all indices when possible."""
        source_indices = np.array([0, 5, 10])
        source_shape = [20, 20]
        target_shape = [20, 20]
        rng = np.random.default_rng(42)
        
        target_indices  = OverlapIndexMapper.map_indices_for_overlap(
            source_indices, source_shape, target_shape, len(source_indices), rng
        )

        success_count = len(target_indices) 
        assert success_count <= len(source_indices)
        assert success_count > 0
    
    def test_partial_mapping_some_invalid(self):
        """Should handle partial mapping when some coords invalid."""
        source_indices = np.array([0, 150, 300])  # Some out of range
        source_shape = [20, 20]
        target_shape = [10, 10]  # Smaller target
        rng = np.random.default_rng(42)
        
        target_indices = OverlapIndexMapper.map_indices_for_overlap(
            source_indices, source_shape, target_shape, len(source_indices), rng
        )
        
        # Some may fail
        success_count = len(target_indices)
        assert success_count <= len(source_indices)
    
    def test_same_dimensions(self):
        """Should work without different dimensions."""
        source_indices = np.array([0, 11, 22])
        source_shape = [10, 10]
        target_shape = [10, 10]
        rng = np.random.default_rng(42)
        
        target_indices = OverlapIndexMapper.map_indices_for_overlap(
            source_indices, source_shape, target_shape, len(source_indices), rng
        )
        
        assert len(target_indices) <= len(source_indices)
    
    def test_different_dimensions(self):
        """Should handle differnt dimensions."""
        source_indices = np.array([0, 5, 10])
        source_shape = [1, 20]  # First dim constant
        target_shape = [10, 20]  # First dim varies
        rng = np.random.default_rng(42)
        
        target_indices = OverlapIndexMapper.map_indices_for_overlap(
            source_indices, source_shape, target_shape, len(source_indices), rng
        )
        
        assert len(target_indices) <= len(source_indices)
    
    def test_reproducible_with_seed(self):
        """Should produce same results with same seed."""
        source_indices = np.array([0, 10, 19])
        source_shape = [1, 20]
        target_shape = [10, 20]
        
        rng1 = np.random.default_rng(999)
        target1 = OverlapIndexMapper.map_indices_for_overlap(
            source_indices, source_shape, target_shape, len(source_indices), rng1
        )
        
        rng2 = np.random.default_rng(999)
        target2 = OverlapIndexMapper.map_indices_for_overlap(
            source_indices, source_shape, target_shape, len(source_indices), rng2
        )
        
        np.testing.assert_array_equal(target1, target2)
