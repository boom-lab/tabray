"""Single-variable record generation.

This module generates sparse record arrays for a single variable.
"""

from typing import List, Optional
import numpy as np
from data_sparsity.generators.record_generator import RecordGenerator
from data_sparsity.generators.observation_generator import ObservationGenerator


class SingleVarRecordGenerator:
    """Generator for single-variable record arrays.
    
    This class creates sparse record arrays for a single variable,
    using the base RecordGenerator functionality.
    """

    @staticmethod
    def generate(
        shape: List[int],
        num_obs: int,
        rng: np.random.Generator,
        observations: Optional[np.ndarray] = None,
        expected_sparsity: Optional[float] = None
    ) -> np.ndarray:
        """Generate a sparse record array for a single variable.
        
        Args:
            shape: Shape of the record array
            num_obs: Number of observations to place
            rng: Random number generator
            observations: Pre-generated observation values (optional)
            expected_sparsity: Expected sparsity for validation (optional)
            
        Returns:
            Sparse record array with observations placed at random locations
            
        Raises:
            ValueError: If sparsity validation fails
        """
        total_grid_points = int(np.prod(shape))
        
        if expected_sparsity is not None:
            RecordGenerator.validate_sparsity(
                num_obs, total_grid_points, expected_sparsity
            )
        
        record = RecordGenerator.initialize_record(shape)
        
        flat_indices = RecordGenerator.generate_flat_indices(
            total_grid_points, num_obs, rng
        )
        multi_indices = RecordGenerator.convert_to_multi_indices(flat_indices, shape)
        
        if observations is None:
            observations = ObservationGenerator.generate_observations(num_obs, rng)
        
        RecordGenerator.assign_observations(record, multi_indices, observations)
        
        return record
