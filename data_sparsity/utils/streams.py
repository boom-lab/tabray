"""Independent random streams derived from one seed.

Every generator draws from ``stream(seed, tag, *index)``, which hands the
tuple to ``numpy.random.default_rng``, which runs it through ``SeedSequence``,
so distinct tuples give independent generators and two purposes cannot end up
sharing a stream. See ``stream`` for the shift that makes that true even when a
tag or an index is zero.

Streams used to be derived by adding offsets to the seed -- ``seed +
dim_idx * 1000`` for coordinate axes, ``seed`` for the density range, ``seed +
5000 + var_idx`` for dimension selection, ``seed + 6000 + var_idx`` for
constant coordinates. Those arithmetics collided:

* ``seed + 0*1000`` is ``seed``, so the x0 axis and the density range were the
  same stream. A variable's density came out as ``min + (max-min)*u`` for the
  same ``u`` that became an x0 coordinate.
* ``seed + 5000`` is ``seed + 5*1000``, so on a 6-dimensional grid the x5 axis
  and var0's dimension selection were the same stream.
* ``seed + 6000`` likewise for the x6 axis and var0's constant coordinate.

Adding a purpose means adding a tag here, which is checked against the others,
rather than picking an offset and hoping it misses.
"""

from typing import Union
import numpy as np


class Stream:
    """Tags identifying what a random stream is for.

    Values are fixed: changing one changes the data generated for that
    purpose. The placement tags keep the values they were introduced with, so
    moving to this registry did not disturb them.
    """

    COORDINATE = 0      # one coordinate axis, indexed by dimension
    LHS = 1             # the global Latin hypercube stage
    STRATUM = 2         # per-stratum fill and values, indexed by stratum
    DENSITY = 3         # drawing per-variable densities from a range
    VAR_DIMS = 4        # choosing which dimensions a variable varies along
    CONST_COORD = 5     # choosing a variable's coordinate on a constant dim
    VAR = 10            # per-variable placement, indexed by variable, stratum
    SHARED_OVERLAP = 11 # the shared ordering behind fixed_overlap


def stream(seed: int, tag: int, *index: Union[int, np.integer]) -> np.random.Generator:
    """Return the generator for one purpose.

    Every component is shifted up by one before being handed to NumPy. That is
    not cosmetic: ``SeedSequence`` IGNORES TRAILING ZEROS, so without the shift
    ``default_rng([seed, 0, 0])`` -- the x0 coordinate axis -- would be the same
    stream as ``default_rng(seed)``, and ``stream(seed, tag, 0)`` the same as
    ``stream(seed, tag)``. Shifting guarantees no component is ever zero, so no
    tuple is a zero-extension of another and distinct purposes stay distinct.

    Args:
        seed: The run's base seed
        tag: A value from :class:`Stream` saying what the stream is for
        *index: Any further coordinates of the purpose, such as the dimension
            or variable it belongs to. Must be non-negative.

    Returns:
        A generator independent of every other tag and index

    Raises:
        ValueError: If any index is negative, which would undo the shift
    """
    shifted = []
    for i in index:
        i = int(i)
        if i < 0:
            raise ValueError(
                f"stream index must be non-negative, got {i}. Negative indices "
                "would cancel the shift that keeps streams distinct."
            )
        shifted.append(i + 1)
    return np.random.default_rng([int(seed), int(tag) + 1, *shifted])
