"""Tests for the CrowdWork session state tool."""

from __future__ import annotations
import sys
import json
import asyncio
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# Load CrowdWork from its package path in the tools/ directory
_PROFILE_PATH = Path(__file__).parents[2] / "src" / "robot_comic" / "tools" / "crowd_work.py"


def _load_crowd_work():
    spec = importlib.util.spec_from_file_location("crowd_work_test_module", _PROFILE_PATH)
    assert spec and spec.loader, f"Cannot load module from {_PROFILE_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["crowd_work_test_module"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.CrowdWork


def make_deps(recent_user_transcripts: list[str] | None = None) -> MagicMock:
    """Return a minimal mock ToolDependencies.

    Seeds ``recent_user_transcripts`` with the names used by existing tests so
    the #287 hallucination guard on ``crowd_work update`` does not reject them.
    Tests that exercise the guard pass their own list (typically empty).
    """
    deps = MagicMock()
    if recent_user_transcripts is None:
        recent_user_transcripts = ["I'm Tony", "Bob here", "Tony", "Bob"]
    deps.recent_user_transcripts = recent_user_transcripts
    return deps


@pytest.fixture
def CrowdWork():
    """Fixture that returns the CrowdWork class loaded from the profile path."""
    return _load_crowd_work()


@pytest.fixture
def crowd_work(tmp_path, CrowdWork):
    """Fixture that returns a CrowdWork instance backed by a temp session directory."""
    session_dir = tmp_path / ".comedy_sessions"
    return CrowdWork(session_dir=session_dir)


# --- update action ---


@pytest.mark.asyncio
async def test_update_stores_name_job_hometown(crowd_work):
    """Update action persists name, job, and hometown in state."""
    result = await crowd_work(make_deps(), action="update", name="Tony", job="engineer", hometown="Pittsburgh")
    assert result["status"] == "updated"
    assert crowd_work._state["name"] == "Tony"
    assert crowd_work._state["job"] == "engineer"
    assert crowd_work._state["hometown"] == "Pittsburgh"


@pytest.mark.asyncio
async def test_update_appends_details_without_duplicates(crowd_work):
    """Repeated detail strings are deduplicated rather than appended twice."""
    await crowd_work(make_deps(), action="update", details=["nervous laugh"])
    await crowd_work(make_deps(), action="update", details=["nervous laugh", "wrinkled shirt"])
    assert crowd_work._state["details"] == ["nervous laugh", "wrinkled shirt"]


@pytest.mark.asyncio
async def test_update_partial_fields_does_not_overwrite_others(crowd_work):
    """Partial update only sets the supplied fields; existing fields are preserved."""
    await crowd_work(make_deps(), action="update", name="Tony", job="engineer")
    await crowd_work(make_deps(), action="update", hometown="Pittsburgh")
    assert crowd_work._state["name"] == "Tony"
    assert crowd_work._state["job"] == "engineer"
    assert crowd_work._state["hometown"] == "Pittsburgh"


# --- query action ---


@pytest.mark.asyncio
async def test_query_returns_empty_callbacks_when_no_data(crowd_work):
    """Query on an empty session returns null identity fields and an empty callbacks list."""
    result = await crowd_work(make_deps(), action="query")
    assert result["profile"]["name"] is None
    assert result["profile"]["job"] is None
    assert result["profile"]["hometown"] is None
    assert result["callbacks"] == []


@pytest.mark.asyncio
async def test_query_returns_callbacks_with_two_or_more_identity_fields(crowd_work):
    """Query surfaces a callback hint when at least two identity fields are known."""
    await crowd_work(make_deps(), action="update", name="Tony", job="engineer")
    result = await crowd_work(make_deps(), action="query")
    assert len(result["callbacks"]) >= 1
    assert "Tony" in result["callbacks"][0]
    assert "engineer" in result["callbacks"][0]


@pytest.mark.asyncio
async def test_query_includes_detail_callbacks(crowd_work):
    """Query includes at least one detail-based callback when details have been stored."""
    await crowd_work(make_deps(), action="update", details=["nervous laugh", "wrinkled shirt"])
    result = await crowd_work(make_deps(), action="query")
    assert any("nervous laugh" in cb for cb in result["callbacks"])


@pytest.mark.asyncio
async def test_query_returns_at_most_three_callbacks(crowd_work):
    """Query never returns more than three callback hints regardless of data richness."""
    await crowd_work(
        make_deps(),
        action="update",
        name="Tony",
        job="engineer",
        hometown="Pittsburgh",
        details=["nervous laugh", "wrinkled shirt", "combover"],
    )
    result = await crowd_work(make_deps(), action="query")
    assert len(result["callbacks"]) <= 3


# --- unknown action ---


@pytest.mark.asyncio
async def test_unknown_action_returns_error(crowd_work):
    """An unrecognised action returns a dict containing an 'error' key."""
    result = await crowd_work(make_deps(), action="explode")
    assert "error" in result


# --- persistence ---


@pytest.mark.asyncio
async def test_update_writes_session_file(crowd_work, tmp_path):
    """Update triggers a background write that persists the session to disk."""
    await crowd_work(make_deps(), action="update", name="Tony")
    await asyncio.sleep(0.15)  # let fire-and-forget write complete
    session_dir = tmp_path / ".comedy_sessions"
    files = list(session_dir.glob("session_*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["name"] == "Tony"
    assert "session_id" in data
    assert "started_at" in data


@pytest.mark.asyncio
async def test_load_recent_session_resumes_from_disk(tmp_path, CrowdWork):
    """A same-day session file on disk is resumed on construction."""
    session_dir = tmp_path / ".comedy_sessions"
    session_dir.mkdir()
    session_file = session_dir / "session_20260506_120000.json"
    session_file.write_text(
        json.dumps(
            {
                "session_id": "20260506_120000",
                "started_at": "2026-05-06T12:00:00",
                "name": "Tony",
                "job": "engineer",
                "hometown": "Pittsburgh",
                "details": ["nervous laugh"],
                "roast_targets_used": [],
                "last_updated": "2026-05-06T12:05:00",
            }
        )
    )
    cw = CrowdWork(session_dir=session_dir)
    assert cw._state["name"] == "Tony"
    assert cw._state["job"] == "engineer"


@pytest.mark.asyncio
async def test_unknown_action_mentions_clear(crowd_work):
    """The error message for an unknown action now lists 'clear' as a valid action."""
    result = await crowd_work(make_deps(), action="explode")
    assert "clear" in result.get("error", "")


# --- clear action ---


@pytest.mark.asyncio
async def test_clear_returns_ok(crowd_work):
    """Clear action returns {action: 'clear', ok: true}."""
    result = await crowd_work(make_deps(), action="clear")
    assert result == {"action": "clear", "ok": True}


@pytest.mark.asyncio
async def test_clear_resets_in_memory_state(crowd_work):
    """After populating state via update, clear zeroes all fields."""
    await crowd_work(
        make_deps(),
        action="update",
        name="Tony",
        job="engineer",
        hometown="Pittsburgh",
        details=["nervous laugh"],
        roast_targets_used=["hair"],
    )
    await crowd_work(make_deps(), action="clear")
    assert crowd_work._state["name"] is None
    assert crowd_work._state["job"] is None
    assert crowd_work._state["hometown"] is None
    assert crowd_work._state["details"] == []
    assert crowd_work._state["roast_targets_used"] == []


@pytest.mark.asyncio
async def test_clear_changes_session_id(crowd_work):
    """After clear, _session_id is different from the pre-clear value."""
    original_id = crowd_work._session_id
    # Ensure enough time passes for a new timestamp (strftime resolution is 1 s)
    import time

    time.sleep(1.1)
    await crowd_work(make_deps(), action="clear")
    assert crowd_work._session_id != original_id


@pytest.mark.asyncio
async def test_clear_preserves_old_file(crowd_work, tmp_path):
    """The on-disk file written before clear is not deleted."""
    await crowd_work(make_deps(), action="update", name="Tony")
    await asyncio.sleep(0.15)  # let the background write finish
    session_dir = tmp_path / ".comedy_sessions"
    files_before = list(session_dir.glob("session_*.json"))
    assert len(files_before) == 1, "expected exactly one pre-clear session file"

    await crowd_work(make_deps(), action="clear")

    files_after = list(session_dir.glob("session_*.json"))
    # The original file must still be present
    assert files_before[0] in files_after


@pytest.mark.asyncio
async def test_clear_then_update_writes_new_file(crowd_work, tmp_path):
    """After clear, an update writes a new session file (different from the pre-clear one)."""
    await crowd_work(make_deps(), action="update", name="Tony")
    await asyncio.sleep(0.15)
    session_dir = tmp_path / ".comedy_sessions"
    files_before = list(session_dir.glob("session_*.json"))
    assert len(files_before) == 1

    import time

    time.sleep(1.1)  # ensure new timestamp-based session_id
    await crowd_work(make_deps(), action="clear")
    await crowd_work(make_deps(), action="update", name="Bob")
    await asyncio.sleep(0.15)

    files_after = list(session_dir.glob("session_*.json"))
    assert len(files_after) == 2
    new_files = [f for f in files_after if f not in files_before]
    assert len(new_files) == 1
    data = json.loads(new_files[0].read_text())
    assert data["name"] == "Bob"


@pytest.mark.asyncio
async def test_load_does_not_resume_old_session(tmp_path, CrowdWork):
    """Sessions outside SESSION_WINDOW_HOURS should not be loaded."""
    session_dir = tmp_path / ".comedy_sessions"
    session_dir.mkdir()
    session_file = session_dir / "session_19990101_120000.json"
    old_data = {
        "session_id": "19990101_120000",
        "started_at": "1999-01-01T12:00:00",
        "name": "OldPerson",
        "job": "dinosaur wrangler",
        "hometown": "Jurassic Park",
        "details": [],
        "roast_targets_used": [],
        "last_updated": "1999-01-01T12:05:00",
    }
    session_file.write_text(json.dumps(old_data))
    # Force old mtime
    import os

    old_time = 946728000  # 2000-01-01 approximately
    os.utime(session_file, (old_time, old_time))
    cw = CrowdWork(session_dir=session_dir)
    assert cw._state["name"] is None


# --- name-hallucination guard (#287) ---


@pytest.mark.asyncio
async def test_update_rejects_name_not_in_recent_transcripts(crowd_work, caplog):
    """A name never spoken by the user must NOT be stored on the session."""
    import logging

    deps = make_deps(recent_user_transcripts=["Hello", "What's up"])
    with caplog.at_level(logging.WARNING):
        result = await crowd_work(deps, action="update", name="John")
    assert crowd_work._state["name"] is None
    assert result["stored"]["name"] is None
    assert result.get("name_rejected") == "John"
    assert result.get("needs_name") is True
    assert any("rejected name" in rec.message and "John" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_update_accepts_name_present_in_recent_transcripts(crowd_work):
    """A name the user actually spoke is persisted as before."""
    deps = make_deps(recent_user_transcripts=["My name is Tony"])
    result = await crowd_work(deps, action="update", name="Tony")
    assert crowd_work._state["name"] == "Tony"
    assert result["stored"]["name"] == "Tony"
    assert "name_rejected" not in result


@pytest.mark.asyncio
async def test_update_rejects_name_but_still_stores_other_fields(crowd_work):
    """Even when ``name`` is rejected, other fields on the same update still apply."""
    deps = make_deps(recent_user_transcripts=["Hello"])
    result = await crowd_work(deps, action="update", name="John", job="engineer", hometown="Pittsburgh")
    assert crowd_work._state["name"] is None
    assert crowd_work._state["job"] == "engineer"
    assert crowd_work._state["hometown"] == "Pittsburgh"
    assert result.get("name_rejected") == "John"
    assert result["stored"]["job"] == "engineer"


@pytest.mark.asyncio
async def test_update_word_boundary_rejects_substring(crowd_work):
    """User said 'Antonio', LLM passed 'Anton' — rejected (no word boundary)."""
    deps = make_deps(recent_user_transcripts=["I'm Antonio from Rome"])
    result = await crowd_work(deps, action="update", name="Anton")
    assert crowd_work._state["name"] is None
    assert result.get("name_rejected") == "Anton"


@pytest.mark.asyncio
async def test_update_case_insensitive_match(crowd_work):
    """User said 'tony' (lowercase), LLM passed 'Tony' — accepted."""
    deps = make_deps(recent_user_transcripts=["tony here"])
    result = await crowd_work(deps, action="update", name="Tony")
    assert crowd_work._state["name"] == "Tony"
    assert "name_rejected" not in result
