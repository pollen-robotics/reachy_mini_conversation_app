"""Tests for the #267 empty-STOP canned-line fallback module."""

from __future__ import annotations
from typing import Iterator
from pathlib import Path

import pytest

from robot_comic import empty_stop_fallbacks as fb


@pytest.fixture(autouse=True)
def _clear_cache() -> Iterator[None]:
    """Drop the per-profile pool cache before AND after each test.

    The module-level cache is shared across tests; without this fixture a
    test that writes to ``<tmp>/empty_stop_fallbacks.txt`` poisons later
    tests' lookups via the cached tuple.
    """
    fb.clear_cache()
    yield
    fb.clear_cache()


def test_default_pool_has_at_least_three_entries() -> None:
    """The bundled default pool must be big enough for no-repeat sampling.

    With <2 distinct entries the no-immediate-repeat invariant in
    :func:`pick_fallback` degrades to "any line, possibly the same one".
    With <3 entries the sampler has effectively one alternative which
    will be picked every other turn and feels canned. Keep the floor at
    3 so operators relying on the default get reasonable variety.
    """
    pool = fb.get_default_pool()
    assert len(pool) >= 3
    # All entries must be non-empty strings (the operator should hear
    # something audible, not a synthesised empty string).
    assert all(isinstance(line, str) and line.strip() for line in pool)


def test_load_pool_returns_default_when_profile_dir_is_none() -> None:
    """No profile dir → default pool, no disk access."""
    assert fb.load_pool(None) == fb.get_default_pool()


def test_load_pool_returns_default_when_file_missing(tmp_path: Path) -> None:
    """Profile dir without the override file → default pool."""
    pool = fb.load_pool(tmp_path)
    assert pool == fb.get_default_pool()


def test_load_pool_reads_profile_override(tmp_path: Path) -> None:
    """When the file exists, its non-blank/non-comment lines win.

    Persona-flavoured operator content lives in the profile file; the
    bundled default is only the safety net. The loader strips whitespace,
    skips ``#`` comments and blank lines, and preserves order.
    """
    (tmp_path / "empty_stop_fallbacks.txt").write_text(
        "# header comment\nYeah. I'm here.\n\n   \nTry that again, friend.\n# another comment\nYou still with me?\n",
        encoding="utf-8",
    )
    pool = fb.load_pool(tmp_path)
    assert pool == ("Yeah. I'm here.", "Try that again, friend.", "You still with me?")


def test_load_pool_falls_back_to_default_when_file_all_blank(tmp_path: Path) -> None:
    """Empty/all-comment file → default pool (avoid accidental disable)."""
    (tmp_path / "empty_stop_fallbacks.txt").write_text("# only a comment\n\n   \n", encoding="utf-8")
    pool = fb.load_pool(tmp_path)
    assert pool == fb.get_default_pool()


def test_load_pool_caches_per_profile_dir(tmp_path: Path) -> None:
    """Repeated loads for the same profile don't re-read the file."""
    target = tmp_path / "empty_stop_fallbacks.txt"
    target.write_text("first\nsecond\n", encoding="utf-8")
    pool_a = fb.load_pool(tmp_path)
    # Rewrite the file; without cache invalidation the cached tuple wins.
    target.write_text("third\n", encoding="utf-8")
    pool_b = fb.load_pool(tmp_path)
    assert pool_a == pool_b == ("first", "second")
    fb.clear_cache()
    assert fb.load_pool(tmp_path) == ("third",)


def test_pick_fallback_avoids_immediate_repeat() -> None:
    """With pool size >=2, the sampler never returns ``last_spoken``."""
    pool = ("A", "B", "C")
    # Run many iterations — even though random.choice is involved, the
    # filter is exhaustive, so no run can equal ``last_spoken``.
    for _ in range(100):
        result = fb.pick_fallback(pool, last_spoken="A")
        assert result != "A"
        assert result in pool


def test_pick_fallback_handles_singleton_pool() -> None:
    """Pool with only one distinct entry returns it even if it == last."""
    assert fb.pick_fallback(("only",), last_spoken="only") == "only"
    assert fb.pick_fallback(("only",), last_spoken=None) == "only"


def test_pick_fallback_no_last_spoken_picks_anything_in_pool() -> None:
    """First-fallback case (no prior line) returns any pool entry."""
    pool = ("X", "Y", "Z")
    seen: set[str] = set()
    for _ in range(200):
        seen.add(fb.pick_fallback(pool, last_spoken=None))
    # All entries reachable from random.choice with seed-free sampling.
    assert seen == set(pool)


def test_pick_fallback_empty_pool_defensive_default() -> None:
    """Empty pool returns the first default entry instead of raising.

    Callers should never pass an empty pool (``load_pool`` falls back to
    the bundled default), but a silent hang is worse than a slightly
    off-flavour line, so the sampler never raises here.
    """
    result = fb.pick_fallback((), last_spoken=None)
    assert result == fb.get_default_pool()[0]
