"""Overlap calculation for multi-variable datasets.

This module computes the actual overlap achieved between variables
after record generation.
"""

from typing import Dict, List, Set, Tuple, Optional
import numpy as np


class OverlapCalculator:
    """Calculator for actual overlap between variables.
    
    This class computes how much observation overlap was achieved
    between multiple variables in a dataset.
    """

    @staticmethod
    def extract_coordinate_set(
        record: np.ndarray,
        num_dims: int
    ) -> Set[Tuple[int, ...]]:
        """Extract set of indices of coordinates where observations exist.
        
        Args:
            record: Record array for one variable
            num_dims: Total number of dimensions
            
        Returns:
            Set of indices tuples of coordinates where observations are non-NaN
        """
        non_nan_indices = np.where(~np.isnan(record))
        
        coords_set = set()
        for obs_idx in range(len(non_nan_indices[0])):
            coord_tuple = tuple(
                non_nan_indices[dim_idx][obs_idx]
                for dim_idx in range(num_dims)
            )
            coords_set.add(coord_tuple)
        
        return coords_set

    @staticmethod
    def project_coordinates(
        coords_set: Set[Tuple[int, ...]],
        dimensions: List[int]
    ) -> Set[Tuple[int, ...]]:
        """Project coordinates onto specific dimensions.
        
        Args:
            coords_set: Set of full coordinate tuples
            dimensions: Dimension indices to project onto
            
        Returns:
            Set of projected coordinate tuples
        """
        projected = set()
        for coord in coords_set:
            proj_coord = tuple(coord[d] for d in dimensions)
            projected.add(proj_coord)
        return projected

    @staticmethod
    def compute_pairwise_overlap_non_normalized(
        ref_coords: Set[Tuple[int, ...]],
        var_coords: Set[Tuple[int, ...]],
        ref_varying_dims: List[int],
        var_varying_dims: List[int]
    ) -> int:
        """Compute overlap count between two variables.
        
        Args:
            ref_coords: Coordinate set for reference variable
            var_coords: Coordinate set for comparison variable
            ref_varying_dims: Varying dimensions for reference variable
            var_varying_dims: Varying dimensions for comparison variable
            
        Returns:
            Number of overlapping observations
        """
        return len(ref_coords.intersection(var_coords))

    @staticmethod
    def compute_overlap_report(
        records: Dict[str, np.ndarray],
        num_vars: int,
        num_dims: int,
        var_dims_indices: Optional[List[List[int]]] = None,
        ref_var: str = "var0"
    ) -> Dict[str, np.ndarray]:
        """Measure overlap against the reference variable, both conventions.

        Overlap is measured on the dimensions the two variables share, as
        docs/explainer_multivar.md defines it, so a variable that varies along
        fewer dimensions is compared through its projection.

            F1 = |proj(S_0) & proj(S_i)| / |proj(S_0)|   <- what the generator targets
            F2 = |proj(S_0) & proj(S_i)| / |proj(S_i)|

        F1 answers "what fraction of the reference's sites also carry this
        variable", F2 the reverse. Both are reported because either can be the
        one a reader expects. They are related by the ratio of the projected set
        sizes, which is measured here rather than inferred from the observation
        counts: under projection ``|proj(S_0)|`` is smaller than ``n_0``, so
        ``F2 = F1 * n_0/n_i`` would be wrong.

        Args:
            records: Dictionary mapping variable names to record arrays
            num_vars: Number of variables
            num_dims: Total number of dimensions
            var_dims_indices: Dimensions each variable varies along. Defaults to
                every dimension for every variable, i.e. no projection.
            ref_var: Reference variable name

        Returns:
            Dict with 'f1', 'f2', 'shared', 'ref_size' and 'var_size' arrays,
            one entry per non-reference variable
        """
        empty = np.array([], dtype=float)
        if num_vars <= 1:
            return {k: empty for k in ("f1", "f2", "shared", "ref_size", "var_size")}

        if var_dims_indices is None:
            var_dims_indices = [list(range(num_dims))] * num_vars

        ref_full = OverlapCalculator.extract_coordinate_set(records[ref_var], num_dims)
        f1, f2, shared, ref_size, var_size = [], [], [], [], []

        for var_idx in range(1, num_vars):
            dims = list(var_dims_indices[var_idx])
            var_full = OverlapCalculator.extract_coordinate_set(
                records[f"var{var_idx}"], num_dims
            )
            proj_ref = OverlapCalculator.project_coordinates(ref_full, dims)
            proj_var = OverlapCalculator.project_coordinates(var_full, dims)
            common = len(proj_ref & proj_var)

            shared.append(common)
            ref_size.append(len(proj_ref))
            var_size.append(len(proj_var))
            f1.append(common / len(proj_ref) if proj_ref else 0.0)
            f2.append(common / len(proj_var) if proj_var else 0.0)

        return {
            "f1": np.asarray(f1, dtype=float),
            "f2": np.asarray(f2, dtype=float),
            "shared": np.asarray(shared, dtype=int),
            "ref_size": np.asarray(ref_size, dtype=int),
            "var_size": np.asarray(var_size, dtype=int),
        }

    @staticmethod
    def print_overlap_report(
        report: Dict[str, np.ndarray],
        targets: Optional[List] = None
    ) -> None:
        """Print achieved overlap in both conventions.

        Both are shown because either can be the one a reader expects, and the
        two differ by more than a relabelling: F1 is what the generator targets.
        """
        if report["f1"].size == 0:
            return
        print("Achieved overlap against var0 "
              "(F1 = share of var0's sites also carrying the variable; "
              "F2 = the reverse):")
        for idx in range(report["f1"].size):
            target = "" if targets is None or targets[idx] is None \
                else f"  target F1 {float(targets[idx]):.4f}"
            print(
                f"  var{idx + 1}: F1 {report['f1'][idx]:.4f}   "
                f"F2 {report['f2'][idx]:.4f}   "
                f"({report['shared'][idx]} shared of {report['ref_size'][idx]} "
                f"var0 sites, {report['var_size'][idx]} own sites){target}"
            )
