"""Tests for the release-notes tooling's commit and notes parsing."""

import pytest

from utils.release_notes.fetch_prs import is_doc_file, extract_pr_number
from utils.release_notes.validate_notes import extract_pr_references


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Show the live transcript on the Talk view (#580)", 580),
        ("Merge pull request #563 from pollen-robotics/fix/562-auto-language\n\nbody", 563),
        ("Prepare v1.1.0 release (#583)\n\nCloses #12", 583),
        ("Merge main into fix/524-mcp-v2-sdk", None),
        ("Fix #12 in the middle of a title", None),
    ],
)
def test_extract_pr_number(message: str, expected: int | None) -> None:
    """Both merge methods main allows yield the PR number; other commits yield none."""
    assert extract_pr_number(message) == expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("README.md", True),
        ("docs/tools.md", True),
        ("docs/assets/logo.png", False),
        ("src/reachy_mini_conversation_app/README.md", False),
    ],
)
def test_is_doc_file(filename: str, expected: bool) -> None:
    """Only the top-level README and docs/*.md count as documentation."""
    assert is_doc_file(filename) is expected


def test_extract_pr_references() -> None:
    """Every #number in the notes counts as a PR reference."""
    notes = "- Faster listening by @a in #569\n- Settings by @b in #552, see #569"
    assert extract_pr_references(notes) == {552, 569}
