"""tabray: synthetic array (netCDF) and tabular (parquet) datasets."""

from importlib.metadata import version

from tabray.generate_data import GenerateData
from tabray.output.dataset_description import DatasetDescription

__version__ = version("tabray")

__all__ = ["GenerateData", "DatasetDescription", "__version__"]
