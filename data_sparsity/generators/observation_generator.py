"""Observation value generation.

This module generates random observation values.
"""

import numpy as np


class ObservationGenerator:
    """Generator for observation values.
    
    This class creates random observation values for data points.
    """

    @staticmethod
    def generate_observations(
        num_obs: int,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Generate random observation values.
        
        Args:
            num_obs: Number of observations to generate
            rng: Random number generator
            
        Returns:
            Array of random observation values in [0, 1]
        """
        return rng.uniform(0, 1, size=num_obs)
