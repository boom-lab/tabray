"""Per-variable parameters given once for all variables or once per variable.

``dtype``, ``pack``, ``fill_value``, ``value_range``, ``data_structure`` and
``noise`` all follow the same rule: a single value applies to every variable,
a sequence gives one entry per variable. Keeping the rule in one place makes
every parameter accept the same inputs.
"""

from typing import Any, List

import numpy as np

#: Types read as one value for every variable. ``np.generic`` covers NumPy
#: scalars such as ``np.float32(0.1)``, which a value taken from an array is.
SINGLE_VALUE_TYPES = (str, int, float, np.generic)


def is_single_value(value: Any, single_value_types: tuple = SINGLE_VALUE_TYPES) -> bool:
    """Whether ``value`` applies to every variable (None counts as one value).

    Args:
        value: The parameter as the user gave it
        single_value_types: Types read as one value

    Returns:
        True for None or an instance of ``single_value_types``
    """
    return value is None or isinstance(value, single_value_types)


def spread_per_variable(
    value: Any,
    name: str,
    num_vars: int,
    single_value_types: tuple = SINGLE_VALUE_TYPES,
) -> List[Any]:
    """One entry per variable from a single value or a per-variable sequence.

    Args:
        value: The parameter as the user gave it
        name: The parameter's name, for the error message
        num_vars: Number of variables
        single_value_types: Types read as one value. ``()`` makes every
            non-None value a sequence, for a parameter whose single value is
            itself a sequence (``value_range``).

    Returns:
        List of ``num_vars`` entries

    Raises:
        ValueError: If a sequence does not hold one entry per variable
    """
    if is_single_value(value, single_value_types):
        return [value] * num_vars
    values = list(value)
    if len(values) != num_vars:
        raise ValueError(
            f"{name} must be a single value or one per variable "
            f"({num_vars}), got {len(values)}"
        )
    return values
