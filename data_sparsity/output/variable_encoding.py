"""How one variable is stored on disk, in both output formats.

Scientific netCDF has no single convention, so this is a per-variable choice
rather than a mode. Two real files, both in this repository:

* GLORYS12 (model output) stores every variable as ``int16`` with
  ``scale_factor``/``add_offset`` and ``_FillValue = -32767``.
* Argo Sprof (observations) stores ``float32`` with no packing and
  ``_FillValue = 99999.0``, a sentinel even though NaN was available.

Both are normal. ``float64`` everywhere, the default, is also normal.
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


class VariableEncoding:
    """The on-disk representation of one variable.

    Two separate choices, one parameter each:

    * ``dtype`` is what the variable holds. A continuous quantity is a float.
    * ``pack`` is how a float is compacted on disk. Naming an integer type
      here carries the values as ``scale_factor * code + add_offset``, the
      convention GLORYS12 and most reanalysis products use.

    They are separate because a file can hold both. Argo stores ``TEMP`` as a
    plain float32 and ``CYCLE_NUMBER`` as a plain int32, and neither is packed.

    The packing scale comes from the declared value range, not from the data,
    so a parallel chunk derives the same scale as a serial run without seeing
    the whole array.
    """

    #: ObservationGenerator draws uniform(0, 1).
    VALUE_RANGE = (0.0, 1.0)
    #: Default span of an integer variable, inclusive at both ends. Narrow it
    #: for flag-like data: the width sets the column's cardinality, which is
    #: what drives dictionary and run-length encoding in parquet.
    INT_VALUE_RANGE = (0, 100)

    FLOAT_DTYPES = ("float64", "float32")
    #: Integer types a float can be packed into, with the fill code each
    #: reserves.
    INT_FILL = {"int8": -127, "int16": -32767, "int32": -2147483647}

    def __init__(
            self,
            dtype: str = "float64",
            pack: Optional[str] = None,
            fill_value: Optional[float] = None,
            value_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Resolve one variable's encoding.

        Args:
            dtype: What the variable holds: ``float64`` or ``float32``
            pack: Integer type to compact the values into on disk
                (``int8``, ``int16``, ``int32``), or None to store them plain
            fill_value: Value marking a vacant site. Defaults to NaN, or to
                the reserved code when packed. A float may take a sentinel
                instead, as Argo does with 99999.0.
            value_range: (min, max) the values span, used to derive the
                packing. Defaults to the generator's own range.

        Raises:
            ValueError: If the dtype or pack type is unknown, or pack is asked
                for on a type that cannot be packed
        """
        dtype = str(dtype).lower()
        if dtype not in self.FLOAT_DTYPES and dtype not in self.INT_FILL:
            raise ValueError(
                f"Unsupported dtype {dtype!r}. Use one of "
                f"{list(self.FLOAT_DTYPES) + list(self.INT_FILL)}."
            )
        self.dtype = dtype

        if pack is not None:
            pack = str(pack).lower()
            if dtype in self.INT_FILL:
                raise ValueError(
                    f"dtype={dtype!r} already holds integers, so pack={pack!r} "
                    "has nothing to do. Packing compacts float values into an "
                    "integer type."
                )
            if pack not in self.INT_FILL:
                raise ValueError(
                    f"Cannot pack into {pack!r}. Use one of "
                    f"{list(self.INT_FILL)}, or None to store the values plain."
                )
        self.pack = pack

        if value_range is not None:
            self.value_range = tuple(value_range)
        elif self.integer:
            self.value_range = self.INT_VALUE_RANGE
        else:
            self.value_range = self.VALUE_RANGE
        self._validate_range()
        self.param_dtype = np.float64  # set by _packing when packed

        if self.packed:
            self.fill_value = (
                int(fill_value) if fill_value is not None
                else self.INT_FILL[pack]
            )
            self.scale_factor, self.add_offset = self._packing()
        elif self.integer:
            self.fill_value = (
                int(fill_value) if fill_value is not None
                else self.INT_FILL[dtype]
            )
            self.scale_factor = self.add_offset = None
            self._validate_fill_outside_range()
        else:
            self.fill_value = fill_value  # None means NaN
            self.scale_factor = self.add_offset = None

    @property
    def integer(self) -> bool:
        """Whether the variable holds integers rather than a continuous
        quantity. Flags, counts and identifiers -- Argo's CYCLE_NUMBER."""
        return self.dtype in self.INT_FILL

    @property
    def packed(self) -> bool:
        """Whether values are carried as scaled integers."""
        return self.pack is not None

    @property
    def storage_dtype(self) -> str:
        """The type the file holds, which is the pack type when packing."""
        return self.pack if self.packed else self.dtype

    def _validate_range(self) -> None:
        """The range must be ordered, and must fit an integer dtype."""
        lo, hi = self.value_range
        if hi <= lo:
            raise ValueError(
                f"value_range must be (min, max) with max above min, got "
                f"({lo}, {hi})"
            )
        if self.integer:
            info = np.iinfo(self.dtype)
            if lo < info.min or hi > info.max:
                raise ValueError(
                    f"value_range ({lo}, {hi}) does not fit {self.dtype}, "
                    f"which spans {info.min} to {info.max}"
                )

    def _validate_fill_outside_range(self) -> None:
        """A fill inside the range would read back as missing.

        The same collision the packing scale avoids by reserving a code, in a
        different place: Argo's 99999 sits outside its cycle numbers, but a
        wider range would swallow it.
        """
        lo, hi = self.value_range
        if lo <= self.fill_value <= hi:
            raise ValueError(
                f"fill_value {self.fill_value} lies inside value_range "
                f"({lo}, {hi}), so those values would read back as missing. "
                "Choose a fill outside the range."
            )

    def _packing(self) -> Tuple[float, float]:
        """Derive scale and offset, leaving the fill code unused.

        Mapping the range onto the full integer range would put the minimum
        value on the fill code, and those cells would read back as missing.
        This is why GLORYS reports ``valid_min = -32760`` rather than -32767.
        The usable codes are ``fill + 1 .. imax``.
        """
        vmin, vmax = self.value_range
        imax = int(np.iinfo(self.pack).max)
        steps = imax - (self.fill_value + 1)
        scale = (vmax - vmin) / steps
        offset = vmax - imax * scale
        # The dtype of scale_factor/add_offset decides what xarray decodes to,
        # so float32 parameters give a float32 array and a quarter of the
        # memory. Only where float32 can carry them: a quantisation step near
        # float32's own spacing would be lost, which is the int32 case.
        headroom = scale / float(np.spacing(np.float32(max(abs(vmin), abs(vmax)))))
        self.param_dtype = np.float32 if headroom > 16 else np.float64
        return (float(self.param_dtype(scale)), float(self.param_dtype(offset)))

    def to_stored(self, values: np.ndarray) -> np.ndarray:
        """Turn the generator's raw draws into the values the file holds.

        ObservationGenerator draws uniform(0, 1) whatever the variable is, so
        this is where a variable becomes what it represents: spread over its
        value range, rounded to integers, or projected onto the packing grid.
        Applied once, before either format is written, so netCDF and parquet
        hold the same numbers and stay comparable.

        The integer mapping transforms the unit draw rather than calling a
        different RNG method. That matters: the determinism contract rests on
        which method is called in what order, so drawing integers separately
        would shift every later draw and move the occupied sites. Changing a
        variable's dtype leaves placement identical and changes only values,
        which is what a storage comparison wants.

        NaN is left alone; it marks a vacant site and becomes the fill on
        write.

        Args:
            values: Raw uniform(0, 1) draws, with NaN at vacant sites

        Returns:
            The values as stored
        """
        lo, hi = self.value_range

        if self.integer:
            occupied = ~np.isnan(values)
            out = values.copy()
            # floor over (hi - lo + 1) bins, not round over (hi - lo): with
            # rounding the two end bins are half-width and the extremes come
            # out at half the frequency of every other value.
            codes = lo + np.floor(values[occupied] * (hi - lo + 1))
            out[occupied] = np.clip(codes, lo, hi)
            return out

        if (lo, hi) != (0.0, 1.0):
            occupied = ~np.isnan(values)
            values = values.copy()
            values[occupied] = lo + (hi - lo) * values[occupied]

        if not self.packed:
            if self.dtype == "float32":
                return values.astype(np.float32).astype(np.float64)
            return values

        occupied = ~np.isnan(values)
        out = values.copy()
        codes = np.rint(
            (values[occupied] - self.add_offset) / self.scale_factor
        )
        codes = np.clip(codes, self.fill_value + 1, np.iinfo(self.pack).max)
        out[occupied] = self.add_offset + self.scale_factor * codes
        return out

    def netcdf_encoding(self) -> dict:
        """The per-variable part of xarray's ``encoding``."""
        if self.integer:
            return {"dtype": self.dtype,
                    "_FillValue": np.dtype(self.dtype).type(self.fill_value)}
        if not self.packed and self.dtype == "float64" and self.fill_value is None:
            # The default. Say nothing, so xarray writes as it would without
            # an encoding argument at all.
            return {}
        if not self.packed:
            encoding = {"dtype": self.dtype}
            if self.fill_value is not None:
                encoding["_FillValue"] = np.dtype(self.dtype).type(self.fill_value)
            return encoding
        return {
            "dtype": self.pack,
            "scale_factor": self.param_dtype(self.scale_factor),
            "add_offset": self.param_dtype(self.add_offset),
            "_FillValue": np.dtype(self.pack).type(self.fill_value),
        }

    def pandas_dtype(self) -> str:
        """The column type parquet stores.

        A packed variable is stored decoded: scale/offset is a netCDF device
        for a format without per-column encodings, and parquet's idiom is the
        natural type. float32 holds a decoded int16 with room to spare -- the
        quantisation step is hundreds of times coarser than float32's spacing.
        """
        if self.integer:
            # Nullable, so a vacant site is a real null rather than forcing the
            # column to float the way NaN would.
            return self.dtype.capitalize()
        if self.packed:
            return "float32" if self.param_dtype is np.float32 else "float64"
        return self.dtype

    def __repr__(self) -> str:
        if not self.packed:
            fill = "NaN" if self.fill_value is None else self.fill_value
            span = f", range={self.value_range}" if self.integer else ""
            return f"VariableEncoding({self.dtype}{span}, fill={fill})"
        return (f"VariableEncoding({self.dtype} -> {self.pack}, "
                f"scale={self.scale_factor:.6g}, fill={self.fill_value})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VariableEncoding):
            return NotImplemented
        return (self.dtype, self.pack, self.fill_value, self.value_range) == (
            other.dtype, other.pack, other.fill_value, other.value_range)

    @classmethod
    def per_variable(
            cls,
            dtype: Union[str, Sequence[str], None],
            pack: Union[str, Sequence, None],
            fill_value: Union[float, Sequence, None],
            num_vars: int,
            value_range: Union[Tuple, Sequence, None] = None,
    ) -> List["VariableEncoding"]:
        """Resolve one encoding per variable.

        A scalar applies to every variable; a sequence gives one entry each,
        so a dataset can mix them the way a real one does -- Argo carries
        float32 measurements beside integer cycle numbers.

        Args:
            dtype: One dtype, or one per variable. None means float64.
            pack: One pack type, or one per variable. None stores plain.
            fill_value: One fill value, or one per variable. None means the
                default for the dtype.
            num_vars: Number of variables
            value_range: One (min, max), or one per variable. None means the
                default for the dtype.

        Returns:
            List of ``num_vars`` encodings

        Raises:
            ValueError: If a sequence has the wrong length
        """
        def spread(value, name, atom=(str, int, float)):
            if value is None or isinstance(value, atom):
                return [value] * num_vars
            values = list(value)
            if len(values) != num_vars:
                raise ValueError(
                    f"{name} must be a single value or one per variable "
                    f"({num_vars}), got {len(values)}"
                )
            return values

        dtypes = spread(dtype, "dtype")
        packs = spread(pack, "pack")
        fills = spread(fill_value, "fill_value")
        # A bare (min, max) applies to every variable; a sequence of pairs
        # gives one each.
        if value_range is not None and not (
                len(value_range) == 2
                and all(isinstance(v, (int, float)) for v in value_range)):
            ranges = spread(list(value_range), "value_range", atom=())
        else:
            ranges = [value_range] * num_vars
        return [
            cls(d if d is not None else "float64", pk, f, r)
            for d, pk, f, r in zip(dtypes, packs, fills, ranges)
        ]
