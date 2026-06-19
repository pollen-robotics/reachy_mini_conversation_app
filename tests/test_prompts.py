"""Tests for prompt assembly helpers."""

from __future__ import annotations
import re

from reachy_mini_conversation_app.prompts import _current_date_line


def test_current_date_line_format() -> None:
    """The date anchor is a single line with a UTC ISO date (or a clear 'unknown')."""
    line = _current_date_line()
    assert line.startswith("The current date is ")
    assert re.fullmatch(r"The current date is (\d{4}-\d{2}-\d{2} \(UTC\)|unknown)\.", line)
