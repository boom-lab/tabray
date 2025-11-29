"""Multi-variable overlap configuration.

This module handles overlap validation and configuration for multiple
variables, including computing minimum feasible overlap.
"""

from typing import List, Union


class MultiVarOverlapConfig:
    """Configuration manager for multi-variable overlap.
    
    This class validates overlap specifications and computes feasibility
    constraints based on dimensional configuration.
    """

    @staticmethod
    def validate_overlap_value(overlap: Union[float, str]) -> Union[float, str]:
        """Validate overlap parameter value.
        
        Args:
            overlap: Overlap specification (0-1 or 'random')
            
        Returns:
            Validated overlap value
            
        Raises:
            TypeError: If overlap is not float or string
            ValueError: If float overlap is outside [0, 1] or string is not 'random'
        """
        if isinstance(overlap, (float, int)):
            overlap = float(overlap)
            if not 0.0 <= overlap <= 1.0:
                raise ValueError(
                    f"overlap must be between 0 and 1, got {overlap}"
                )
            return overlap
        elif isinstance(overlap, str):
            if overlap != 'random':
                raise ValueError(
                    f"overlap string must be 'random', got '{overlap}'"
                )
            return overlap
        else:
            raise TypeError(
                f"overlap must be float or 'random', got {type(overlap)}"
            )

    @staticmethod
    def compute_min_overlap(var_dims_indices: List[List[int]]) -> float:
        """Compute minimum possible overlap given variable dimensions.
        
        The minimum overlap occurs when variables share the fewest dimensions.
        It is computed as: (# shared dims) / (# total dims used by any variable)
        
        Args:
            var_dims_indices: List of varying dimension indices per variable
            
        Returns:
            Minimum feasible overlap (0 to 1)
        """
        if not var_dims_indices or len(var_dims_indices) < 2:
            return 0.0
        
        # Find all pairs and compute their shared dimensions
        num_vars = len(var_dims_indices)
        min_shared = float('inf')
        max_union = 0
        
        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                dims_i = set(var_dims_indices[i])
                dims_j = set(var_dims_indices[j])
                
                shared = len(dims_i & dims_j)
                union = len(dims_i | dims_j)
                
                if shared < min_shared:
                    min_shared = shared
                if union > max_union:
                    max_union = union
        
        if max_union == 0:
            return 0.0
        
        return min_shared / max_union

    @staticmethod
    def validate_overlap_feasibility(
        overlap: Union[float, str],
        min_overlap: float,
        num_vars: int
    ) -> None:
        """Validate that requested overlap is feasible.
        
        Args:
            overlap: Requested overlap value
            min_overlap: Minimum feasible overlap
            num_vars: Number of variables
            
        Raises:
            ValueError: If requested overlap is below minimum feasible
        """
        if num_vars < 2:
            return
        
        if isinstance(overlap, float):
            if overlap < min_overlap:
                raise ValueError(
                    f"Requested overlap {overlap} is below minimum feasible "
                    f"overlap {min_overlap} given the dimension configuration. "
                    f"Either increase overlap or adjust var_dims."
                )

    @staticmethod
    def setup_from_parameter(
        overlap: Union[float, str],
        num_vars: int,
        var_dims_indices: List[List[int]]
    ) -> Union[float, str]:
        """Setup overlap configuration from parameter.
        
        Main entry point that validates overlap parameter and checks feasibility.
        
        Args:
            overlap: Overlap specification (0-1 or 'random')
            num_vars: Number of variables
            var_dims_indices: Varying dimension indices per variable
            
        Returns:
            Validated overlap value
            
        Raises:
            TypeError: If overlap is not a valid type
            ValueError: If overlap is infeasible
        """
        overlap = MultiVarOverlapConfig.validate_overlap_value(overlap)
        
        if num_vars > 1:
            min_overlap = MultiVarOverlapConfig.compute_min_overlap(var_dims_indices)
            print(
                f"Minimum feasible overlap given dimension configuration: "
                f"{min_overlap}"
            )
            MultiVarOverlapConfig.validate_overlap_feasibility(
                overlap, min_overlap, num_vars
            )
        
        return overlap
