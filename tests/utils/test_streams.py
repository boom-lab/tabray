"""Tests for the random stream registry.

The point of data_sparsity.utils.streams is that two purposes never share a
stream. These tests pin that property, including the NumPy behaviour that
makes it non-obvious: SeedSequence ignores trailing zeros.
"""

import itertools

import numpy as np
import pytest

from data_sparsity.utils.streams import Stream, stream


def first_draws(rng, n=5):
    """Return the opening draws of a generator, for comparing streams."""
    return tuple(rng.uniform(0, 1, n))


class TestSeedSequenceTrailingZeros:
    """The NumPy behaviour the shift in stream() exists to defeat."""

    def test_numpy_ignores_trailing_zeros(self):
        """Documents why stream() cannot pass raw tags and indices.

        If this ever fails, NumPy changed and the shift could be dropped.
        """
        assert first_draws(np.random.default_rng(42)) == first_draws(
            np.random.default_rng([42, 0, 0])
        )

    def test_stream_is_not_fooled_by_it(self):
        """A zero tag with a zero index is still its own stream."""
        assert first_draws(stream(42, 0, 0)) != first_draws(np.random.default_rng(42))
        assert first_draws(stream(42, 0, 0)) != first_draws(stream(42, 0))


class TestStreamsAreDistinct:
    """No two purposes may share a stream."""

    def test_every_tag_differs(self):
        tags = [v for k, v in vars(Stream).items() if not k.startswith("_")]
        draws = [first_draws(stream(42, t)) for t in tags]
        assert len(set(draws)) == len(tags)

    def test_indices_differ_within_a_tag(self):
        draws = [first_draws(stream(42, Stream.COORDINATE, d)) for d in range(8)]
        assert len(set(draws)) == 8

    def test_index_depth_differs(self):
        """(var 0) and (var 0, stratum 0) are different purposes."""
        assert first_draws(stream(42, Stream.VAR, 0)) != first_draws(
            stream(42, Stream.VAR, 0, 0)
        )

    def test_multi_index_combinations_differ(self):
        combos = list(itertools.product(range(3), range(3)))
        draws = [first_draws(stream(42, Stream.VAR, v, s)) for v, s in combos]
        assert len(set(draws)) == len(combos)

    def test_seeds_differ(self):
        draws = [first_draws(stream(s, Stream.COORDINATE, 0)) for s in range(5)]
        assert len(set(draws)) == 5

    def test_no_collision_across_the_whole_live_space(self):
        """Every tag against every plausible index tuple, in one sweep."""
        seen = {}
        for tag in [v for k, v in vars(Stream).items() if not k.startswith("_")]:
            for depth in range(3):
                for index in itertools.product(range(4), repeat=depth):
                    draws = first_draws(stream(42, tag, *index))
                    assert draws not in seen, (
                        f"(tag={tag}, index={index}) collides with {seen[draws]}"
                    )
                    seen[draws] = (tag, index)


class TestStreamContract:
    """Reproducibility and input handling."""

    def test_same_arguments_give_the_same_stream(self):
        assert first_draws(stream(42, Stream.VAR, 1, 2)) == first_draws(
            stream(42, Stream.VAR, 1, 2)
        )

    def test_numpy_integers_are_accepted(self):
        """Indices often arrive as numpy scalars from enumerate over arrays."""
        assert first_draws(stream(42, Stream.VAR, np.int64(1))) == first_draws(
            stream(42, Stream.VAR, 1)
        )

    def test_negative_index_raises(self):
        """A negative index would cancel the shift and reintroduce collisions."""
        with pytest.raises(ValueError, match="non-negative"):
            stream(42, Stream.VAR, -1)
