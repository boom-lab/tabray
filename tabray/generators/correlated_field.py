"""Spatially correlated values: a smooth Gaussian random field per variable.

By default every value is an independent uniform(0, 1) draw, which is white
noise: packed to int16 it does not compress, and every codec decodes it at
close to memory speed. Real fields change little between neighbouring cells,
which is what compression feeds on. ``data_structure="correlated"`` replaces
the draw at each occupied cell by a smooth field evaluated at that cell's
grid indices. docs/correlated_values.md explains the choices and measures
their effect on compression.
"""

import warnings
from typing import List, Optional, Sequence, Union

import numpy as np
from scipy.special import ndtr, ndtri

from tabray.utils.per_variable import is_single_value, spread_per_variable
from tabray.utils.streams import Stream, stream

#: One entry of ``correlation_length``: cells, 0 (white) or None (constant).
LengthEntry = Optional[float]


class CorrelatedField:
    """The smooth value field of one variable.

    The field is a sum of ``N_MODES`` random plane waves ("random Fourier
    features")::

        S(i) = sqrt(2 / M) * sum_m cos(sum_d k[m, d] * i_d + phi[m])

    where ``i`` holds a cell's GLOBAL grid indices. With ``k[m, d]`` drawn from
    N(0, 1 / L_d**2), the field has on average the Gaussian autocorrelation
    ``exp(-1/2 * sum_d (lag_d / L_d)**2)``: cells L_d apart along dim d have
    correlation 0.61, cells 3 L_d apart almost none. ``S`` is close to a
    standard normal, so ``ndtr`` (the normal CDF) maps it back to uniform(0, 1),
    the draw every encoding (integers, strings, packing) already expects.

    ``noise`` mixes in the existing uniform draw as white noise: real data
    carry measurement and model noise far below the variable's range, and that
    floor decides most of the compression.

    The value at a cell depends only on (seed, variable, global indices, the
    uniform draw), so serial and parallel runs agree, and the occupied sites
    are those of an uncorrelated run.
    """

    #: Number of plane waves. Fewer make the field visibly periodic and its
    #: autocorrelation scatter around the target (error ~ 1/sqrt(M)).
    N_MODES = 64
    #: Correlation length, in cells, of every dim when none is given.
    DEFAULT_LENGTH = 10.0
    #: Keeps ndtri finite: uniform draws of exactly 0 would map to -inf.
    U_EPS = 1e-12
    #: Cells evaluated together, sized for the CPU cache.
    BLOCK = 1 << 14

    def __init__(
        self,
        seed: int,
        var_idx: int,
        num_dims: int,
        varying_dims: Sequence[int],
        correlation_length: Union[LengthEntry, Sequence[LengthEntry]] = None,
        noise: float = 0.0,
    ) -> None:
        """Draw the plane waves of one variable.

        Args:
            seed: The run's base seed
            var_idx: Variable index; each variable gets its own field
            num_dims: Number of grid dimensions
            varying_dims: Dims the variable varies along. On its constant
                dims it sits on one coordinate, so the field ignores them.
            correlation_length: In grid cells. None gives DEFAULT_LENGTH on
                every dim, a number the same length on every dim, a list one
                entry per grid dim: a length >= 1, 0 for no correlation
                along that dim, None for a value constant along it.
            noise: Standard deviation of the white noise, as a share of the
                total (0 to 1). 0 is a perfectly smooth field, 1 the
                uncorrelated draw.

        Raises:
            ValueError: If a length or the noise is out of range, or the
                list of lengths has the wrong size
        """
        self.lengths = self.resolve_lengths(correlation_length, num_dims)
        noise = float(noise)
        if not 0.0 <= noise <= 1.0:
            raise ValueError(f"noise must be between 0 and 1, got {noise}")
        self.noise = noise
        self.varying_dims = sorted(int(d) for d in varying_dims)

        # Every draw is made, whatever the lengths, in a fixed order, so that
        # changing one dim's length leaves the waves of the others unchanged.
        rng = stream(seed, Stream.CORRELATED_FIELD, var_idx)
        self.phases = rng.uniform(0.0, 2.0 * np.pi, size=self.N_MODES)
        gaussian = rng.standard_normal(size=(self.N_MODES, num_dims))
        white_wavenumbers = rng.uniform(-np.pi, np.pi, size=(self.N_MODES, num_dims))

        # k[:, d] is the wavenumber of each wave along dim d, in radians per
        # cell. A uniform spread over (-pi, pi), the whole band an integer
        # grid can hold, decorrelates neighbouring cells exactly.
        self.wavenumbers = np.zeros((self.N_MODES, num_dims))
        for dim in self.varying_dims:
            length = self.lengths[dim]
            if length is None:
                continue
            if length == 0:
                self.wavenumbers[:, dim] = white_wavenumbers[:, dim]
            else:
                self.wavenumbers[:, dim] = gaussian[:, dim] / length

    @classmethod
    def resolve_lengths(
        cls,
        correlation_length: Union[LengthEntry, Sequence[LengthEntry]],
        num_dims: int,
    ) -> List[LengthEntry]:
        """Turn the user's ``correlation_length`` into one entry per dim.

        Args:
            correlation_length: None, one number, or one entry per dim
            num_dims: Number of grid dimensions

        Returns:
            One entry per dim: a length >= 1, 0, or None

        Raises:
            ValueError: If the list has the wrong size or an entry is
                negative, between 0 and 1, NaN or infinite
        """
        if correlation_length is None:
            return [cls.DEFAULT_LENGTH] * num_dims
        if isinstance(correlation_length, (int, float, np.integer, np.floating)):
            lengths = [correlation_length] * num_dims
        else:
            lengths = list(correlation_length)
            if len(lengths) != num_dims:
                raise ValueError(
                    "correlation_length must be one number or one entry per "
                    f"dimension ({num_dims}), got {len(lengths)}"
                )
        resolved = []
        for length in lengths:
            if length is None:
                resolved.append(None)
                continue
            if isinstance(length, bool) or not isinstance(
                length, (int, float, np.integer, np.floating)
            ):
                raise ValueError(
                    f"correlation_length entries must be numbers or None, got {length!r}"
                )
            length = float(length)
            # Below one cell the Gaussian spectrum reaches past the highest
            # wavenumber the grid can hold and folds back: the field is no
            # longer Gaussian-correlated. 0 asks for no correlation directly.
            # NaN fails every comparison, so it is caught by isfinite; inf is
            # refused too, None already says "constant".
            if not np.isfinite(length) or length < 0 or 0 < length < 1:
                raise ValueError(
                    "correlation_length must be 0 (no correlation), at least "
                    f"1 cell, or None (constant), got {length}"
                )
            resolved.append(length)
        return resolved

    def apply(
        self, uniform: np.ndarray, global_indices: Sequence[np.ndarray]
    ) -> np.ndarray:
        """Replace uniform draws by the field's values at the same cells.

        Args:
            uniform: The uniform(0, 1) draws of the occupied cells
            global_indices: One index array per grid dim, in the GLOBAL grid
                (not chunk-local: the field must not depend on the chunking)

        Returns:
            Values in (0, 1), uniform across the grid, smooth from cell to cell
        """
        uniform = np.asarray(uniform, dtype=float)
        if uniform.size == 0:
            return uniform
        smooth = np.empty(uniform.size)
        # Blocks small enough to stay in cache; every operation is per cell,
        # so the block size does not change any value.
        for start in range(0, uniform.size, self.BLOCK):
            stop = min(start + self.BLOCK, uniform.size)
            smooth[start:stop] = self._sum_waves(
                [global_indices[dim][start:stop] for dim in range(len(global_indices))]
            )
        smooth *= np.sqrt(2.0 / self.N_MODES)

        if self.noise:
            white_noise = ndtri(np.clip(uniform, self.U_EPS, 1.0 - self.U_EPS))
            smooth = np.sqrt(1.0 - self.noise**2) * smooth + self.noise * white_noise
        return ndtr(smooth)

    def _sum_waves(self, global_indices: List[np.ndarray]) -> np.ndarray:
        """Sum of the plane waves at a block of cells, before scaling.

        One wave at a time, dims summed in a fixed order: each cell's value
        comes from the same operations whatever the block's length, so a
        chunk reproduces the serial run bit for bit. A matrix product or a
        reduction over a 2-D array may reorder the sums with the size.

        The cosine is the cost: 64 per cell. It is taken in float32, about
        seven times faster than float64 here, after reducing the angle to
        [-pi, pi] in float64 so no precision is lost on large angles. The
        values move by under 1e-7, far below any packing step.

        Args:
            global_indices: One index array per grid dim, for the block

        Returns:
            The unscaled sum of the waves at each cell
        """
        size = global_indices[0].size
        positions = {
            dim: global_indices[dim].astype(float)
            for dim in self.varying_dims
            if np.any(self.wavenumbers[:, dim])
        }
        total = np.zeros(size)
        angle = np.empty(size)
        term = np.empty(size)
        angle32 = np.empty(size, dtype=np.float32)
        for mode in range(self.N_MODES):
            angle.fill(self.phases[mode])
            for dim, position in positions.items():
                wavenumber = self.wavenumbers[mode, dim]
                if wavenumber != 0.0:
                    np.multiply(position, wavenumber, out=term)
                    angle += term
            # angle - 2 pi * round(angle / 2 pi): the same angle in [-pi, pi]
            np.divide(angle, 2.0 * np.pi, out=term)
            np.rint(term, out=term)
            term *= 2.0 * np.pi
            angle -= term
            angle32[:] = angle
            np.cos(angle32, out=angle32)
            total += angle32
        return total

    @classmethod
    def per_variable(
        cls,
        data_structure: Union[str, Sequence[str]],
        correlation_length: Union[LengthEntry, Sequence[LengthEntry]],
        noise: Union[float, Sequence[float], None],
        num_vars: int,
        grid_shape: Sequence[int],
        var_dims_indices: Sequence[Sequence[int]],
        seed: int,
    ) -> List[Optional["CorrelatedField"]]:
        """Resolve one field per variable, None for an uncorrelated one.

        Args:
            data_structure: ``"uncorrelated"`` or ``"correlated"``, once or
                one per variable
            correlation_length: Shared by every correlated variable (see
                ``__init__``)
            noise: One value, applied to the correlated variables, or one
                per variable (0 or None on an uncorrelated one). None means 0.
            num_vars: Number of variables
            grid_shape: Cells along each grid dim
            var_dims_indices: Varying dims of each variable
            seed: The run's base seed

        Returns:
            List of ``num_vars`` entries, a CorrelatedField or None

        Raises:
            ValueError: On an unknown data_structure, a list of the wrong
                length, or correlation settings with no correlated variable
        """
        num_dims = len(grid_shape)
        structures = spread_per_variable(data_structure, "data_structure", num_vars)
        for structure in structures:
            if structure not in ("uncorrelated", "correlated"):
                raise ValueError(
                    f"Unknown data_structure {structure!r}. "
                    "Use 'uncorrelated' or 'correlated'."
                )
        noises = spread_per_variable(noise, "noise", num_vars)
        noises = [0.0 if value is None else float(value) for value in noises]
        is_correlated = [structure == "correlated" for structure in structures]

        if not any(is_correlated):
            if correlation_length is not None or any(noises):
                raise ValueError(
                    "correlation_length and noise shape correlated values: "
                    "set data_structure='correlated' to use them."
                )
            return [None] * num_vars
        if is_single_value(noise):
            # One noise value is meant for the correlated variables: an
            # uncorrelated variable's values are already all noise.
            noises = [
                value if correlated else 0.0
                for value, correlated in zip(noises, is_correlated)
            ]
        else:
            for value, correlated in zip(noises, is_correlated):
                if not correlated and value:
                    raise ValueError(
                        "noise is set for a variable with data_structure="
                        "'uncorrelated', whose values are already all noise."
                    )

        lengths = cls.resolve_lengths(correlation_length, num_dims)
        # Only dims a correlated variable varies along: on the others no
        # field is evaluated, so their length has no effect.
        dims_of_correlated_vars = sorted(
            {
                int(dim)
                for var_idx, correlated in enumerate(is_correlated)
                if correlated
                for dim in var_dims_indices[var_idx]
            }
        )
        length_source = "default " if correlation_length is None else ""
        for dim in dims_of_correlated_vars:
            length, size = lengths[dim], grid_shape[dim]
            if length is not None and length > size:
                warnings.warn(
                    f"{length_source}correlation_length {length:g} on x{dim} "
                    f"exceeds its {size} cells: values will be nearly constant "
                    "along it.",
                    UserWarning,
                    stacklevel=3,
                )
        return [
            (
                cls(seed, var_idx, num_dims, var_dims_indices[var_idx], lengths, value)
                if structure == "correlated"
                else None
            )
            for var_idx, (structure, value) in enumerate(zip(structures, noises))
        ]

    def __repr__(self) -> str:
        lengths = {
            f"x{dim}": ("white" if self.lengths[dim] == 0 else self.lengths[dim])
            for dim in self.varying_dims
        }
        return f"CorrelatedField(lengths={lengths}, noise={self.noise:g})"
