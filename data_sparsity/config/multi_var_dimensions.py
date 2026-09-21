"""Multi-variable dimension configuration.

This module handles dimension setup for multiple variables, including
which dimensions vary for each variable and which are held constant.
"""

from typing import Dict, List, Tuple, Union
import numpy as np

from data_sparsity.utils.streams import Stream, stream


class MultiVarDimensionsConfig:
    """Configuration manager for multi-variable dimensions.

    This class processes the var_dims parameter which can be:
    - An int: all variables have the same number of VARYING dimensions (which
      are randomly selected)
    - A list/tuple of ints: each var has a different number of VARYING dims
      (which randomly selected)
    - A list/tuple of lists/tuples: each var has explicitly specified VARYING dims

    VARYING dimensions are those dimensions along which the variable's record
    changes. Dimensions not in var_dims_indices are held constant for that
    variable, as if the variable is measured only once (or never and assigned a
    constant value) along that dimensions. Numerically, all variables have
    components in the full num_dims space.

    Sets self.var_dims_indices (varying dimensions), self.var_constant_dims,
    and self.var_constant_coord_indices (pre-selected constant coordinate indices)

    """

    @staticmethod
    def select_random_dims(
        num_dims: int,
        var_dims: int,
        seed: int,
        var_idx: int,
    ) -> List[int]:
        """Select random varying dimensions for a variable.

        Args:
            num_dims: Total number of dimensions
            var_dims: Number of varying dimensions to select
            seed: Base random seed
            var_idx: Which variable this is, so each gets its own stream

        Returns:
            Sorted list of dimension indices
        """
        var_rng = stream(seed, Stream.VAR_DIMS, var_idx)
        dims = sorted(var_rng.choice(num_dims, size=var_dims, replace=False).tolist())
        return dims

    @staticmethod
    def from_int(
        var_dims: int,
        num_vars: int,
        num_dims: int,
        seed: int,
    ) -> List[List[int]]:
        """Create dimension indices from integer specification.

        All variables get the same number of varying dimensions,
        but which ones vary is randomly selected per variable.

        Args:
            var_dims: Number of varying dimensions per variable
            num_vars: Number of variables
            num_dims: Total number of dimensions
            seed: Base random seed

        Returns:
            List of dimension index lists, one per variable

        Raises:
            ValueError: If var_dims exceeds num_dims
        """
        if var_dims > num_dims:
            raise ValueError(f"var_dims {var_dims} cannot exceed num_dims {num_dims}")

        if var_dims == num_dims:
            return [list(range(num_dims))] * num_vars

        var_dims_indices = []
        for var_idx in range(num_vars):
            dims = MultiVarDimensionsConfig.select_random_dims(
                num_dims,
                var_dims,
                seed,
                var_idx,
            )
            var_dims_indices.append(dims)

        return var_dims_indices

    @staticmethod
    def from_list_element(
        var_dims_elem: Union[int, List, Tuple],
        var_idx: int,
        num_dims: int,
        seed: int,
    ) -> List[int]:
        """Process a single element from var_dims list.

        Args:
            var_dims_elem: Element for one variable (int or list of indices)
            var_idx: Variable index (for error messages and seeding)
            num_dims: Total number of dimensions
            seed: Base random seed

        Returns:
            List of dimension indices for this variable

        Raises:
            TypeError: If element is not int or list/tuple
            ValueError: If dimension specifications are invalid
        """
        if isinstance(var_dims_elem, int):
            if var_dims_elem > num_dims:
                raise ValueError(
                    f"Variable {var_idx} var_dims {var_dims_elem} cannot exceed "
                    f"num_dims {num_dims}"
                )
            if var_dims_elem == num_dims:
                return list(range(num_dims))
            else:
                return MultiVarDimensionsConfig.select_random_dims(
                    num_dims,
                    var_dims_elem,
                    seed,
                    var_idx,
                )
        elif isinstance(var_dims_elem, (list, tuple)):
            dims = list(var_dims_elem)
            if len(dims) == 0 or len(dims) > num_dims:
                raise ValueError(
                    f"Variable {var_idx} dim indices must have 1 to "
                    f"{num_dims} elements, got {len(dims)}"
                )
            if not all(0 <= d < num_dims for d in dims):
                raise ValueError(
                    f"Variable {var_idx} dim indices {dims} must be in "
                    f"range [0, {num_dims})"
                )
            if len(set(dims)) != len(dims):
                raise ValueError(
                    f"Variable {var_idx} dim indices {dims} contain duplicates"
                )
            return sorted(dims)
        else:
            raise TypeError(
                f"Variable {var_idx} var_dims must be int or list/tuple, "
                f"got {type(var_dims_elem)}"
            )

    @staticmethod
    def from_list(
        var_dims: Union[List, Tuple],
        num_vars: int,
        num_dims: int,
        seed: int,
    ) -> List[List[int]]:
        """Create dimension indices from list specification.

        Each element can be an int (number of dims) or list (explicit indices).

        Args:
            var_dims: List/tuple with one element per variable
            num_vars: Number of variables
            num_dims: Total number of dimensions
            seed: Base random seed

        Returns:
            List of dimension index lists, one per variable

        Raises:
            ValueError: If list length doesn't match num_vars
        """
        if len(var_dims) != num_vars:
            raise ValueError(
                f"var_dims list must have {num_vars} elements, " f"got {len(var_dims)}"
            )

        var_dims_indices = []
        for var_idx, var_dims_elem in enumerate(var_dims):
            dims = MultiVarDimensionsConfig.from_list_element(
                var_dims_elem,
                var_idx,
                num_dims,
                seed,
            )
            var_dims_indices.append(dims)

        return var_dims_indices

    @staticmethod
    def compute_constant_dims(
        var_dims_indices: List[List[int]],
        num_dims: int,
    ) -> List[List[int]]:
        """Compute which dimensions are constant for each variable.

        Args:
            var_dims_indices: Varying dimension indices per variable
            num_dims: Total number of dimensions

        Returns:
            List of constant dimension index lists, one per variable
        """
        var_constant_dims = []
        for varying_dims in var_dims_indices:
            constant_dims = [d for d in range(num_dims) if d not in varying_dims]
            var_constant_dims.append(constant_dims)
        return var_constant_dims

    @staticmethod
    def preselect_constant_coord_indices(
        var_constant_dims: List[List[int]],
        shape: List[int],
        seed: int,
    ) -> Dict[int, Dict[int, np.random.Generator]]:
        """Pre-select RNGs for constant coordinate indices.

        For reproducibility, we pre-create RNGs that will later be used
        to select which coordinate index to use for constant dimensions.

        Args:
            var_constant_dims: Constant dimension indices per variable
            shape: Shape of full coordinate space
            seed: Base random seed

        Returns:
            Dictionary mapping var_idx -> {const_dim -> RNG}
        """
        var_constant_coord_indices = {}

        for var_idx, const_dims in enumerate(var_constant_dims):
            if const_dims:
                const_rng_dict = {}
                for const_dim in const_dims:
                    # Indexed by the dimension too, or a variable constant on
                    # two dimensions picks the same index on both.
                    const_rng_dict[const_dim] = stream(
                        seed,
                        Stream.CONST_COORD,
                        var_idx,
                        const_dim,
                    )
                var_constant_coord_indices[var_idx] = const_rng_dict
            else:
                var_constant_coord_indices[var_idx] = {}

        return var_constant_coord_indices

    @staticmethod
    def setup_from_parameter(
        var_dims: Union[int, List, Tuple],
        num_vars: int,
        num_dims: int,
        shape: List[int],
        seed: int,
    ) -> Tuple[
        List[List[int]], List[List[int]], Dict[int, Dict[int, np.random.Generator]]
    ]:
        """Setup multi-variable dimension configuration from parameter.

        Main entry point that processes var_dims parameter and returns
        complete configuration.

        Args:
            var_dims: Dimension specification (int or list)
            num_vars: Number of variables
            num_dims: Total number of dimensions
            shape: Shape of full coordinate space
            seed: Base random seed

        Returns:
            Tuple of (var_dims_indices, var_constant_dims, var_constant_coord_indices)

        Raises:
            TypeError: If var_dims is not a valid type
        """
        if isinstance(var_dims, int):
            var_dims_indices = MultiVarDimensionsConfig.from_int(
                var_dims,
                num_vars,
                num_dims,
                seed,
            )
        elif isinstance(var_dims, (list, tuple)):
            var_dims_indices = MultiVarDimensionsConfig.from_list(
                var_dims,
                num_vars,
                num_dims,
                seed,
            )
        else:
            raise TypeError(
                f"var_dims must be int, list, or tuple, got {type(var_dims)}"
            )

        var_constant_dims = MultiVarDimensionsConfig.compute_constant_dims(
            var_dims_indices,
            num_dims,
        )
        var_constant_coord_indices = (
            MultiVarDimensionsConfig.preselect_constant_coord_indices(
                var_constant_dims,
                shape,
                seed,
            )
        )

        return var_dims_indices, var_constant_dims, var_constant_coord_indices
