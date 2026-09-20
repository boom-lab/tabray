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

    An integer dtype implies packing: the generator's values are fractional,
    so they are carried as ``scale_factor * code + add_offset``. The scale is
    derived from the declared value range, never from the data, so that a
    parallel chunk computes the same scale as a serial run without having to
    see the whole array.
    """

    #: ObservationGenerator draws uniform(0, 1).
    VALUE_RANGE = (0.0, 1.0)

    FLOAT_DTYPES = ("float64", "float32")
    #: Integer dtypes, with the fill code each reserves.
    INT_FILL = {"int8": -127, "int16": -32767, "int32": -2147483647}

    def __init__(
            self,
            dtype: str = "float64",
            fill_value: Optional[float] = None,
            value_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Resolve one variable's encoding.

        Args:
            dtype: ``float64``, ``float32``, ``int8``, ``int16`` or ``int32``
            fill_value: Value marking a vacant site. Defaults to NaN for float
                dtypes and to the reserved code for integer ones. A float
                dtype may take a sentinel instead, as Argo does.
            value_range: (min, max) the data can take, used to derive the
                packing. Defaults to the generator's own range.

        Raises:
            ValueError: If the dtype is unsupported, or a fill value is given
                that collides with the representable range
        """
        dtype = str(dtype).lower()
        if dtype not in self.FLOAT_DTYPES and dtype not in self.INT_FILL:
            raise ValueError(
                f"Unsupported dtype {dtype!r}. Use one of "
                f"{list(self.FLOAT_DTYPES) + list(self.INT_FILL)}."
            )
        self.dtype = dtype
        self.value_range = tuple(value_range or self.VALUE_RANGE)
        self.param_dtype = np.float64  # set by _packing for packed dtypes

        if self.packed:
            self.fill_value = (
                int(fill_value) if fill_value is not None
                else self.INT_FILL[dtype]
            )
            self.scale_factor, self.add_offset = self._packing()
        else:
            self.fill_value = fill_value  # None means NaN
            self.scale_factor = self.add_offset = None

    @property
    def packed(self) -> bool:
        """Whether values are carried as scaled integers."""
        return self.dtype in self.INT_FILL

    def _packing(self) -> Tuple[float, float]:
        """Derive scale and offset, leaving the fill code unused.

        Mapping the range onto the full integer range would put the minimum
        value on the fill code, and those cells would read back as missing.
        This is why GLORYS reports ``valid_min = -32760`` rather than -32767.
        The usable codes are ``fill + 1 .. imax``.
        """
        vmin, vmax = self.value_range
        imax = int(np.iinfo(self.dtype).max)
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

    def quantize(self, values: np.ndarray) -> np.ndarray:
        """Round values to what the file will actually hold.

        Applied before either format is written, so netCDF and parquet store
        the same numbers and remain comparable. NaN is left alone; it marks a
        vacant site and becomes the fill on write.

        Args:
            values: Generated values, with NaN at vacant sites

        Returns:
            The same array rounded to the representable grid
        """
        if self.dtype == "float64":
            return values
        if self.dtype == "float32":
            return values.astype(np.float32).astype(np.float64)

        occupied = ~np.isnan(values)
        out = values.copy()
        codes = np.rint(
            (values[occupied] - self.add_offset) / self.scale_factor
        )
        codes = np.clip(codes, self.fill_value + 1, np.iinfo(self.dtype).max)
        out[occupied] = self.add_offset + self.scale_factor * codes
        return out

    def netcdf_encoding(self) -> dict:
        """The per-variable part of xarray's ``encoding``."""
        if self.dtype == "float64" and self.fill_value is None:
            # The default. Say nothing, so xarray writes exactly as it would
            # without an encoding argument at all.
            return {}
        if not self.packed:
            encoding = {"dtype": self.dtype}
            if self.fill_value is not None:
                encoding["_FillValue"] = np.dtype(self.dtype).type(self.fill_value)
            return encoding
        return {
            "dtype": self.dtype,
            "scale_factor": self.param_dtype(self.scale_factor),
            "add_offset": self.param_dtype(self.add_offset),
            "_FillValue": np.dtype(self.dtype).type(self.fill_value),
        }

    def pandas_dtype(self) -> str:
        """The column type parquet stores.

        A packed variable is stored decoded: scale/offset is a netCDF device
        for a format without per-column encodings, and parquet's idiom is the
        natural type. float32 holds a decoded int16 with room to spare -- the
        quantisation step is hundreds of times coarser than float32's spacing.
        """
        if self.packed:
            return "float32" if self.param_dtype is np.float32 else "float64"
        return self.dtype

    def __repr__(self) -> str:
        if not self.packed:
            fill = "NaN" if self.fill_value is None else self.fill_value
            return f"VariableEncoding({self.dtype}, fill={fill})"
        return (f"VariableEncoding({self.dtype} packed, "
                f"scale={self.scale_factor:.6g}, fill={self.fill_value})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VariableEncoding):
            return NotImplemented
        return (self.dtype, self.fill_value, self.value_range) == (
            other.dtype, other.fill_value, other.value_range)

    @classmethod
    def per_variable(
            cls,
            dtype: Union[str, Sequence[str], None],
            fill_value: Union[float, Sequence, None],
            num_vars: int,
    ) -> List["VariableEncoding"]:
        """Resolve one encoding per variable.

        A scalar applies to every variable; a sequence gives one entry each,
        so a dataset can mix them the way a real one does -- Argo carries
        float32 measurements beside integer cycle numbers.

        Args:
            dtype: One dtype, or one per variable. None means float64.
            fill_value: One fill value, or one per variable. None means the
                default for the dtype.
            num_vars: Number of variables

        Returns:
            List of ``num_vars`` encodings

        Raises:
            ValueError: If a sequence has the wrong length
        """
        def spread(value, name):
            if value is None or isinstance(value, (str, int, float)):
                return [value] * num_vars
            values = list(value)
            if len(values) != num_vars:
                raise ValueError(
                    f"{name} must be a single value or one per variable "
                    f"({num_vars}), got {len(values)}"
                )
            return values

        dtypes = spread(dtype, "dtype")
        fills = spread(fill_value, "fill_value")
        return [
            cls(d if d is not None else "float64", f)
            for d, f in zip(dtypes, fills)
        ]
