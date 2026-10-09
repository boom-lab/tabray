"""Configuration modules for multi-variable data generation."""

from tabray.config.multi_var_sparsity import MultiVarSparsityConfig
from tabray.config.multi_var_dimensions import MultiVarDimensionsConfig
from tabray.config.multi_var_overlap import MultiVarOverlapConfig

__all__ = [
    "MultiVarSparsityConfig",
    "MultiVarDimensionsConfig",
    "MultiVarOverlapConfig",
]
