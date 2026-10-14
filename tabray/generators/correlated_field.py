"""Spatially correlated values: a smooth Gaussian random field per variable.

By default every value is an independent uniform(0, 1) draw, that is, white
noise. Packed to int16, white noise does not compress, and every codec
decodes it at close to memory speed. Real fields change little between
neighbouring sites, and compression relies on that similarity.
``data_structure="correlated"`` replaces the draw at each occupied site with
a smooth field evaluated at the site's grid indices.
docs/correlated_values.md explains the design and reports its effect on
compression.
"""

import warnings
from typing import List, Optional, Sequence, Union

import numpy as np
from scipy.special import ndtr, ndtri

from tabray.utils.per_variable import is_single_value, spread_per_variable
from tabray.utils.streams import Stream, stream

#: One entry of ``correlation_length``: sites, 0 (white) or None (constant).
LengthEntry = Optional[float]


class CorrelatedField:
    """The smooth value field of one variable.

    The field is a sum of ``N_MODES`` random plane waves ("random Fourier
    features")::

        S(i) = sqrt(2 / M) * sum_m cos(sum_d k[m, d] * i_d + phi[m])

    where ``i`` holds a site's GLOBAL grid indices. With ``k[m, d]`` drawn from
    N(0, 1 / L_d**2), the expected autocorrelation of the field is the
    Gaussian ``exp(-1/2 * sum_d (lag_d / L_d)**2)``: sites L_d apart along
    dim d have correlation 0.61, and sites 3 L_d apart are nearly
    uncorrelated. ``S`` is approximately standard normal, so ``ndtr`` (the
    normal CDF) maps it to uniform(0, 1), the distribution that every
    encoding (integers, strings, packing) expects as input.

    ``noise`` adds independent noise to each occupied site on top of the
    smooth field. It is the noise's standard deviation as a share of the
    total, from 0 (no noise) to 1 (no smooth field left). The noise comes from
    the uniform draw already made for each occupied site, so no new random
    numbers are drawn. It models the small site-to-site variation of real
    measurements and model output: even when it is small next to the
    variable's range, it changes the low bits of every stored value, and
    those bits do not compress. See docs/correlated_values.md for measured
    sizes.

    The value at a site depends only on the seed, the variable, the site's
    global indices and its uniform draw, so serial and parallel runs produce
    the same values. The field changes values only: sites are placed before
    any value is drawn, so the occupied sites are the same as in an
    uncorrelated run.
    """

    #: Number of plane waves. With fewer waves the field becomes visibly
    #: periodic, and its autocorrelation deviates more from the target
    #: (error of order 1/sqrt(M)).
    N_MODES = 64
    #: Correlation length, in sites, used on every dim when none is given.
    DEFAULT_LENGTH = 10.0
    #: Margin that keeps ndtri finite: a uniform draw of exactly 0 maps to -inf.
    U_EPS = 1e-12
    #: Sites evaluated at a time in apply, so its work arrays stay small.
    BLOCK = 16_384

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
            var_idx: Variable index; each variable has its own field
            num_dims: Number of grid dimensions
            varying_dims: Dims the variable varies along. The variable has
                a single coordinate on each of its constant dims, so the
                field does not depend on them.
            correlation_length: In sites. None uses DEFAULT_LENGTH on
                every dim; a number uses that length on every dim; a list
                gives one entry per grid dim: a length >= 1, 0 for no
                correlation along that dim, or None for a value that is
                constant along it.
            noise: Standard deviation of the white noise, as a share of the
                total (0 to 1). 0 gives a smooth field; 1 gives the
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

        # All draws are made in a fixed order, independently of the lengths,
        # so changing one dim's length does not change the other dims' waves.
        rng = stream(seed, Stream.CORRELATED_FIELD, var_idx)
        self.phases = rng.uniform(0.0, 2.0 * np.pi, size=self.N_MODES)
        standard_normal_draws = rng.standard_normal(size=(self.N_MODES, num_dims))
        white_wavenumbers = rng.uniform(-np.pi, np.pi, size=(self.N_MODES, num_dims))

        # k[:, d] is the wavenumber of each wave along dim d, in radians per
        # site. Wavenumbers drawn uniformly from (-pi, pi), the full band an
        # integer grid can represent, give zero correlation between sites.
        self.wavenumbers = np.zeros((self.N_MODES, num_dims))
        for dim in self.varying_dims:
            length = self.lengths[dim]
            if length is None:
                continue
            if length == 0:
                self.wavenumbers[:, dim] = white_wavenumbers[:, dim]
            else:
                self.wavenumbers[:, dim] = standard_normal_draws[:, dim] / length

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
        lengths_per_dim = []
        for length in lengths:
            if length is None:
                lengths_per_dim.append(None)
                continue
            if isinstance(length, bool) or not isinstance(
                length, (int, float, np.integer, np.floating)
            ):
                raise ValueError(
                    f"correlation_length entries must be numbers or None, got {length!r}"
                )
            length = float(length)
            # Below a length of one site, the Gaussian spectrum extends past
            # the highest wavenumber the grid can represent and is aliased, so
            # the field is no longer Gaussian-correlated; 0 is the way to
            # request no correlation. NaN fails every comparison and is
            # rejected by isfinite. inf is rejected too: None already means
            # "constant".
            if not np.isfinite(length) or length < 0 or 0 < length < 1:
                raise ValueError(
                    "correlation_length must be 0 (no correlation), at least "
                    f"1 site, or None (constant), got {length}"
                )
            lengths_per_dim.append(length)
        return lengths_per_dim

    def apply(
        self, uniform: np.ndarray, global_indices: Sequence[np.ndarray]
    ) -> np.ndarray:
        """Replace uniform draws with the field's values at the same sites.

        Args:
            uniform: The uniform(0, 1) draws of the occupied sites
            global_indices: One index array per grid dim, in the GLOBAL grid,
                not chunk-local, so that the field does not depend on the
                chunking

        Returns:
            Values in (0, 1), uniformly distributed over the grid and smooth
            between neighbouring sites
        """
        uniform = np.asarray(uniform, dtype=float)
        if uniform.size == 0:
            return uniform
        smooth_field = np.empty(uniform.size)
        # Every operation is per site, so the block size does not change any
        # value.
        for block_start in range(0, uniform.size, self.BLOCK):
            block_stop = min(block_start + self.BLOCK, uniform.size)
            smooth_field[block_start:block_stop] = self._sum_waves(
                [
                    global_indices[dim][block_start:block_stop]
                    for dim in range(len(global_indices))
                ]
            )
        smooth_field *= np.sqrt(2.0 / self.N_MODES)

        # A standard normal value per site, mapped to uniform(0, 1) by ndtr.
        normal_score = smooth_field
        if self.noise:
            white_noise = ndtri(np.clip(uniform, self.U_EPS, 1.0 - self.U_EPS))
            normal_score = (
                np.sqrt(1.0 - self.noise**2) * smooth_field + self.noise * white_noise
            )
        return ndtr(normal_score)

    def _sum_waves(self, global_indices: List[np.ndarray]) -> np.ndarray:
        """Sum of the plane waves at a block of occupied sites, before scaling.

        The waves are added one at a time, and the dims of each wave in a
        fixed order. Each site's value is therefore computed by the same
        operations independently of the block's length, and a chunk
        reproduces the serial run bit for bit. A matrix product or a
        reduction over a 2-D array may change the order of the sums with the
        array size.

        Most of the time is spent on the cosines, 64 per occupied site. They are
        computed in float32, about seven times faster than float64 here,
        after the angle is reduced to [-pi, pi] in float64, so large angles
        lose no precision. The values change by less than 1e-7, well below
        any packing step.

        Args:
            global_indices: One index array per grid dim, for the block

        Returns:
            The unscaled sum of the waves at each occupied site
        """
        num_sites = global_indices[0].size
        # Only the dims along which some wave varies; as float, the type of
        # the angle they enter.
        float_indices = {
            dim: global_indices[dim].astype(float)
            for dim in self.varying_dims
            if np.any(self.wavenumbers[:, dim])
        }
        # Work arrays, allocated once per block and overwritten for each wave.
        wave_sum = np.zeros(num_sites)
        wave_angle = np.empty(num_sites)
        angle_along_dim = np.empty(num_sites)
        full_turns = np.empty(num_sites)
        full_turns_angle = np.empty(num_sites)
        wave_angle_float32 = np.empty(num_sites, dtype=np.float32)
        wave_value = np.empty(num_sites, dtype=np.float32)
        for wave in range(self.N_MODES):
            # wave_angle = phi + sum_d k_d * i_d
            wave_angle.fill(self.phases[wave])
            for dim, index_along_dim in float_indices.items():
                wavenumber = self.wavenumbers[wave, dim]
                if wavenumber != 0.0:
                    np.multiply(index_along_dim, wavenumber, out=angle_along_dim)
                    wave_angle += angle_along_dim
            # Subtract the whole turns: angle - 2 pi * round(angle / 2 pi) is
            # the same angle in [-pi, pi].
            np.divide(wave_angle, 2.0 * np.pi, out=full_turns)
            np.rint(full_turns, out=full_turns)
            np.multiply(full_turns, 2.0 * np.pi, out=full_turns_angle)
            wave_angle -= full_turns_angle
            wave_angle_float32[:] = wave_angle
            np.cos(wave_angle_float32, out=wave_value)
            wave_sum += wave_value
        return wave_sum

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
            noise: One value, applied to the correlated variables only, or
                one value per variable (0 or None for an uncorrelated one).
                None means 0.
            num_vars: Number of variables
            grid_shape: Sites along each grid dim
            var_dims_indices: Varying dims of each variable
            seed: The run's base seed

        Returns:
            List of ``num_vars`` entries, a CorrelatedField or None

        Raises:
            ValueError: If data_structure is unknown, a list has the wrong
                length, or correlation settings are given without a
                correlated variable
        """
        num_dims = len(grid_shape)
        var_structures = spread_per_variable(data_structure, "data_structure", num_vars)
        for var_structure in var_structures:
            if var_structure not in ("uncorrelated", "correlated"):
                raise ValueError(
                    f"Unknown data_structure {var_structure!r}. "
                    "Use 'uncorrelated' or 'correlated'."
                )
        var_noises = spread_per_variable(noise, "noise", num_vars)
        var_noises = [
            0.0 if var_noise is None else float(var_noise) for var_noise in var_noises
        ]
        is_correlated = [
            var_structure == "correlated" for var_structure in var_structures
        ]

        if not any(is_correlated):
            if correlation_length is not None or any(var_noises):
                raise ValueError(
                    "correlation_length and noise apply only to correlated "
                    "values: set data_structure='correlated' to use them."
                )
            return [None] * num_vars
        if is_single_value(noise):
            # A single noise value applies to the correlated variables only:
            # the values of an uncorrelated variable are already all noise.
            var_noises = [
                var_noise if correlated else 0.0
                for var_noise, correlated in zip(var_noises, is_correlated)
            ]
        else:
            for var_noise, correlated in zip(var_noises, is_correlated):
                if not correlated and var_noise:
                    raise ValueError(
                        "noise is set for a variable with data_structure="
                        "'uncorrelated', whose values are already all noise."
                    )

        lengths = cls.resolve_lengths(correlation_length, num_dims)
        # Check only the dims that a correlated variable varies along: no
        # field is evaluated on the other dims, so their length has no effect.
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
            length, axis_size = lengths[dim], grid_shape[dim]
            if length is not None and length > axis_size:
                warnings.warn(
                    f"{length_source}correlation_length {length:g} on x{dim} "
                    f"exceeds its {axis_size} sites: values will be nearly constant "
                    "along it.",
                    UserWarning,
                    stacklevel=3,
                )
        return [
            (
                cls(
                    seed,
                    var_idx,
                    num_dims,
                    var_dims_indices[var_idx],
                    lengths,
                    var_noise,
                )
                if correlated
                else None
            )
            for var_idx, (correlated, var_noise) in enumerate(
                zip(is_correlated, var_noises)
            )
        ]

    def __repr__(self) -> str:
        lengths = {
            f"x{dim}": ("white" if self.lengths[dim] == 0 else self.lengths[dim])
            for dim in self.varying_dims
        }
        return f"CorrelatedField(lengths={lengths}, noise={self.noise:g})"
