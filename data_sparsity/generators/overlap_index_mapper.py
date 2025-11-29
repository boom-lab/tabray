"""Index mapping for overlap between variables.

This module handles the complex task of mapping observation indices from
one variable's space to another variable's space for overlap control.
"""

from typing import List, Optional, Set, Tuple
import numpy as np


class OverlapIndexMapper:
    """Mapper for indices between variable spaces for overlap.
    
    This class handles mapping observation locations from one variable's
    dimensional space to another's, accounting for differences in which
    dimensions vary vs. remain constant.
    """

    @staticmethod
    def identify_random_assign_dims(
        source_shape: List[int],
        target_shape: List[int]
    ) -> List[int]:
        """Identify dimensions needing random assignment.
        
        These are dimensions where source is constant (size 1) but target varies.
        
        Args:
            source_shape: Shape of source variable
            target_shape: Shape of target variable
            
        Returns:
            List of dimension indices requiring random assignment
        """
        return [
            d for d in range(len(source_shape))
            if source_shape[d] == 1 and target_shape[d] > 1
        ]

    @staticmethod
    def map_single_coordinate(
        source_coords: List[int],
        source_shape: List[int],
        target_shape: List[int]
    ) -> Tuple[bool, List[Optional[int]]]:
        """Map a single coordinate from source to target space.
        
        Args:
            source_coords: Coordinates in source space
            source_shape: Shape of source variable
            target_shape: Shape of target variable
            
        Returns:
            Tuple of (is_valid, target_coords) where target_coords may contain
            None for dimensions requiring random assignment
        """
        target_coords = []
        valid = True
        
        for d in range(len(source_coords)):
            if target_shape[d] == 1:
                target_coords.append(0)
            elif source_shape[d] == 1:
                target_coords.append(None)
            else:
                if source_coords[d] < target_shape[d]:
                    target_coords.append(source_coords[d])
                else:
                    valid = False
                    break
        
        return valid, target_coords

    @staticmethod
    def try_random_assignment(
        target_coords: List[Optional[int]],
        random_assign_dims: List[int],
        target_shape: List[int],
        used_target_indices: Set[int],
        rng: np.random.Generator,
        max_attempts: int = 100
    ) -> Optional[int]:
        """Try to find an unused target index with random assignment.
        
        Args:
            target_coords: Target coordinates with None for random dims
            random_assign_dims: Dimensions needing random values
            target_shape: Shape of target variable
            used_target_indices: Set of already-used target indices
            rng: Random number generator
            max_attempts: Maximum number of attempts
            
        Returns:
            Target flat index if successful, None otherwise
        """
        max_attempts = min(
            max_attempts,
            int(np.prod([target_shape[d] for d in random_assign_dims]))
        )
        
        for _ in range(max_attempts):
            trial_coords = target_coords.copy()
            for d in random_assign_dims:
                trial_coords[d] = int(rng.integers(0, target_shape[d]))
            
            target_flat = int(np.ravel_multi_index(trial_coords, target_shape))
            
            if target_flat not in used_target_indices:
                return target_flat
        
        return None

    @staticmethod
    def map_indices_for_overlap(
        source_flat_indices: np.ndarray,
        source_shape: List[int],
        target_shape: List[int],
        num_needed: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Map indices from source to target variable space for overlap.
        
        This works with full shapes where constant dimensions have size 1.
        
        Args:
            source_flat_indices: Flat indices in source variable's space
            source_shape: Shape of source variable (size 1 for constant dims)
            target_shape: Shape of target variable (size 1 for constant dims)
            num_needed: Number of overlapping observations needed
            rng: Random number generator for selection
            
        Returns:
            Array of flat indices in target variable's space
        """
        source_multi = np.unravel_index(source_flat_indices, source_shape)
        
        random_assign_dims = OverlapIndexMapper.identify_random_assign_dims(
            source_shape, target_shape
        )
        
        used_target_indices = set()
        corresponding_target_flat = []
        
        for i in range(len(source_flat_indices)):
            source_coords = [source_multi[d][i] for d in range(len(source_multi))]
            
            valid, target_coords = OverlapIndexMapper.map_single_coordinate(
                source_coords, source_shape, target_shape
            )
            
            if not valid:
                continue
            
            if random_assign_dims:
                target_flat = OverlapIndexMapper.try_random_assignment(
                    target_coords, random_assign_dims, target_shape,
                    used_target_indices, rng
                )
                if target_flat is not None:
                    used_target_indices.add(target_flat)
                    corresponding_target_flat.append(target_flat)
            else:
                target_flat = int(np.ravel_multi_index(target_coords, target_shape))
                if target_flat not in used_target_indices:
                    used_target_indices.add(target_flat)
                    corresponding_target_flat.append(target_flat)
        
        if len(corresponding_target_flat) >= num_needed:
            selected_indices = rng.choice(
                len(corresponding_target_flat),
                size=num_needed,
                replace=False
            )
            return np.array(
                [corresponding_target_flat[i] for i in selected_indices],
                dtype=np.int64
            )
        else:
            print(
                f"WARNING: Only {len(corresponding_target_flat)} valid overlap "
                f"mappings found, needed {num_needed}. Using all available."
            )
            return np.array(corresponding_target_flat, dtype=np.int64)
