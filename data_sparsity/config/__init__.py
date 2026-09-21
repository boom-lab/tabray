"""Configuration modules for multi-variable data generation."""

from data_sparsity.config.multi_var_sparsity import MultiVarSparsityConfig
from data_sparsity.config.multi_var_dimensions import MultiVarDimensionsConfig
from data_sparsity.config.multi_var_overlap import MultiVarOverlapConfig

__all__ = [
    "MultiVarSparsityConfig",
    "MultiVarDimensionsConfig",
    "MultiVarOverlapConfig",
]
