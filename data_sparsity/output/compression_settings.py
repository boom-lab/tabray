"""Compression applied identically to both output formats.

The package exists to compare array storage against tabular storage, so the
two must be written on the same terms. They are not comparable by default:
dask/pyarrow compresses parquet with Snappy unless told otherwise, while
``to_netcdf`` writes uncompressed. This class settles the codec once and hands
each backend the keyword it wants.
"""

from typing import Optional, Union

import xarray as xr


class CompressionSettings:
    """A codec and level that both netCDF and parquet can honour.

    Only codecs available to both formats are accepted. netCDF4 offers DEFLATE
    (its ``zlib`` option); parquet calls the same algorithm ``gzip``. Snappy,
    zstd, lz4 and brotli are parquet-only here and are refused, because using
    one would mean the two formats were no longer measured against the same
    work.
    """

    # Spellings accepted for each codec, mapped to the canonical name.
    ALIASES = {
        None: None,
        "none": None,
        "off": None,
        "gzip": "gzip",
        "zlib": "gzip",
        "deflate": "gzip",
    }

    #: Codecs parquet can do but this netCDF build cannot.
    PARQUET_ONLY = ("snappy", "zstd", "lz4", "brotli")

    def __init__(
            self,
            codec: Optional[str] = None,
            level: int = 4,
    ) -> None:
        """Validate and store the codec.

        Args:
            codec: ``None`` (or ``"none"``) for uncompressed, or ``"gzip"``
                (``"zlib"``, ``"deflate"``) for DEFLATE on both formats
            level: Compression level 1-9, ignored when codec is None

        Raises:
            ValueError: If the codec is unknown, is parquet-only, or the level
                is out of range
        """
        key = codec.lower() if isinstance(codec, str) else codec
        if key not in self.ALIASES:
            if key in self.PARQUET_ONLY:
                raise ValueError(
                    f"compression={codec!r} is available to parquet but not to "
                    "netCDF, so the two formats would not be written on the "
                    "same terms. Use 'gzip' (DEFLATE, both) or None."
                )
            raise ValueError(
                f"Unknown compression {codec!r}. Use one of "
                f"{sorted(a for a in self.ALIASES if a)} or None."
            )

        self.codec = self.ALIASES[key]

        if self.codec is not None and not 1 <= int(level) <= 9:
            raise ValueError(f"complevel must be 1-9, got {level}")
        self.level = int(level)

    @property
    def enabled(self) -> bool:
        """Whether anything is compressed."""
        return self.codec is not None

    def netcdf_encoding(self, data: Union[xr.DataArray, xr.Dataset]) -> dict:
        """Per-variable encoding for ``to_netcdf``.

        Compression forces chunked HDF5 storage, which netCDF4 sizes itself.
        Coordinates are left alone: they are small, and compressing them costs
        more in per-chunk overhead than it saves.

        Args:
            data: The object about to be written

        Returns:
            Encoding dict, empty when compression is off
        """
        if not self.enabled:
            return {}
        if isinstance(data, xr.DataArray):
            names = [data.name] if data.name is not None else []
        else:
            names = list(data.data_vars)
        return {
            name: {"zlib": True, "complevel": self.level} for name in names
        }

    def parquet_kwargs(self) -> dict:
        """Keyword arguments for ``to_parquet``.

        Passing ``compression=None`` explicitly matters: dask's own default is
        Snappy, so leaving it out is what made the two formats incomparable.
        """
        return {"compression": self.codec, **(
            {"compression_level": self.level} if self.enabled else {}
        )}

    def __repr__(self) -> str:
        if not self.enabled:
            return "CompressionSettings(none)"
        return f"CompressionSettings({self.codec}, level={self.level})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompressionSettings):
            return NotImplemented
        return (self.codec, self.enabled and self.level) == (
            other.codec, other.enabled and other.level
        )
