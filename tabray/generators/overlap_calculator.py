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
        num_dims: int,
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
                non_nan_indices[dim_idx][obs_idx] for dim_idx in range(num_dims)
            )
            coords_set.add(coord_tuple)

        return coords_set

    @staticmethod
    def project_coordinates(
        coords_set: Set[Tuple[int, ...]],
        dimensions: List[int],
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
    def compute_overlap_report(
        records: Dict[str, np.ndarray],
        num_vars: int,
        num_dims: int,
        var_dims_indices: List[List[int]],
    ) -> Dict[str, np.ndarray]:
        """Measure overlap against the reference variable, both conventions.

        Overlap is measured on the dimensions the two variables share, so a
        variable that varies along fewer dimensions is compared through its
        projection.

            overlap         = |proj(S_0) & proj(S_i)| / |proj(S_0)|   <- what the generator targets
            overlap_reverse = |proj(S_0) & proj(S_i)| / |proj(S_i)|

        overlap answers "what fraction of the reference's sites also carry this
        variable", overlap_reverse the reverse. Both are reported because either can be the
        one a reader expects. They are related by the ratio of the projected set
        sizes, which is measured here rather than inferred from the observation
        counts: under projection ``|proj(S_0)|`` is smaller than ``n_0``, so
        ``overlap_reverse = overlap * n_0/n_i`` would be wrong.

        Args:
            records: Dictionary mapping variable names to record arrays
            num_vars: Number of variables
            num_dims: Total number of dimensions
            var_dims_indices: Dimensions each variable varies along

        Returns:
            Dict with 'overlap', 'overlap_reverse', 'shared', 'ref_size' and
            'var_size' arrays, one entry per non-reference variable
        """
        empty = np.array([], dtype=float)
        if num_vars <= 1:
            keys = ("overlap", "overlap_reverse", "shared", "ref_size", "var_size")
            return {k: empty for k in keys}

        ref_full = OverlapCalculator.extract_coordinate_set(records["var0"], num_dims)
        overlap, reverse, shared, ref_size, var_size = [], [], [], [], []

        for var_idx in range(1, num_vars):
            dims = list(var_dims_indices[var_idx])
            var_full = OverlapCalculator.extract_coordinate_set(
                records[f"var{var_idx}"],
                num_dims,
            )
            proj_ref = OverlapCalculator.project_coordinates(ref_full, dims)
            proj_var = OverlapCalculator.project_coordinates(var_full, dims)
            common = len(proj_ref & proj_var)

            shared.append(common)
            ref_size.append(len(proj_ref))
            var_size.append(len(proj_var))
            overlap.append(common / len(proj_ref) if proj_ref else 0.0)
            reverse.append(common / len(proj_var) if proj_var else 0.0)

        return {
            "overlap": np.asarray(overlap, dtype=float),
            "overlap_reverse": np.asarray(reverse, dtype=float),
            "shared": np.asarray(shared, dtype=int),
            "ref_size": np.asarray(ref_size, dtype=int),
            "var_size": np.asarray(var_size, dtype=int),
        }

    @staticmethod
    def print_overlap_report(
        report: Dict[str, np.ndarray],
        targets: List[Optional[float]],
    ) -> None:
        """Print achieved overlap in both conventions.

        overlap is the share of var0's sites that also carry the variable, the
        quantity the generator targets; overlap_reverse is the share of the variable's sites
        that also carry var0. Both are shown because either can be the one a
        reader expects, and they are not interchangeable: they differ by the
        ratio of the projected set sizes.

        Args:
            report: Output of compute_overlap_report, at least one variable
            targets: Target overlap per non-reference variable, None for 'random'
        """
        print(
            "Achieved overlap against var0 "
            "(overlap = share of var0's sites also carrying the variable; "
            "overlap_r = share of the variable's sites also carrying var0):"
        )
        for idx in range(report["overlap"].size):
            target = (
                "" if targets[idx] is None else f"  target {float(targets[idx]):.4f}"
            )
            print(
                f"  var{idx + 1}: overlap {report['overlap'][idx]:.4f}   "
                f"overlap_r {report['overlap_reverse'][idx]:.4f}   "
                f"({report['shared'][idx]} shared of {report['ref_size'][idx]} "
                f"var0 sites, {report['var_size'][idx]} own sites){target}"
            )
