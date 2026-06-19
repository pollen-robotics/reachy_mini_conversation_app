"""Tests for the Dreamer — uses a scripted fake OpenAI client."""

from __future__ import annotations
from typing import Any, Callable
from pathlib import Path
from dataclasses import field, dataclass

import pytest

from reachy_mini_conversation_app.memory.dreamer import (
    DEFAULT_DREAMER_MODEL,
    Dreamer,
    run_dream_pass,
)
from reachy_mini_conversation_app.memory.memory_manager import MemoryManager


@dataclass
class _FakeResponses:
    """Scripted stand-in for ``client.responses``.

    ``on_create`` is a function called with the current ``input`` list; it
    returns the next response's ``output`` list (items as dicts). That keeps
    the fake tiny and lets each test describe exactly the tool-call pattern
    it wants to exercise.
    """

    on_create: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> Any:
        """Fake ``client.responses.create`` implementation."""
        self.calls.append(kwargs)
        output = self.on_create(kwargs["input"])

        class _Resp:
            pass

        resp = _Resp()
        resp.output = output
        return resp


@dataclass
class _FakeClient:
    responses: _FakeResponses


def _msg_item(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def _call_item(name: str, args: dict[str, Any], call_id: str = "c1") -> dict[str, Any]:
    import json as _json

    return {
        "type": "function_call",
        "name": name,
        "call_id": call_id,
        "arguments": _json.dumps(args),
    }


@pytest.fixture
def manager(tmp_path: Path) -> MemoryManager:
    """Create a fresh MemoryManager and queue one non-live log to dream on."""
    mgr = MemoryManager(tmp_path / "data")
    # Put a closed log into pending/ (the live session log is excluded automatically).
    (mgr.pending_logs_dir / "2026-04-14_09-15.log").write_text(
        "--- session 2026-04-14 09:15 UTC ---\n\n"
        "09:15:12 user: Hey Reachy, my name is Rémi.\n"
        "09:15:14 assistant: Nice to meet you, Rémi!\n"
        "09:15:20 user: I love chess.\n"
        "09:15:22 assistant: Got it.\n",
        encoding="utf-8",
    )
    return mgr


class TestDreamerSingleLog:
    """Verify the dreamer loop end-to-end with a scripted LLM."""

    def test_write_memory_flow(self, manager: MemoryManager) -> None:
        """Scripted LLM: write one memory, then stop."""
        steps: list[list[dict[str, Any]]] = [
            [
                _call_item(
                    "write_memory",
                    {
                        "id": "2026-04-14_user-name_a01",
                        "body": "User's name is Rémi.",
                        "kind": "fact",
                        "tags": ["identity"],
                        "sources": ["2026-04-14_09-15.log"],
                        "pinned": True,
                    },
                    call_id="c1",
                )
            ],
            [_msg_item("Wrote one identity memory.")],
        ]
        fake = _FakeClient(responses=_FakeResponses(on_create=lambda _i: steps.pop(0)))
        dreamer = Dreamer(manager, model="fake-model", client=fake)
        stats_list = dreamer.run()

        [stats] = stats_list
        assert stats.created == 1
        assert stats.tool_calls_count.get("write_memory") == 1
        assert len(stats.llm_durations_s) >= 1
        assert stats.tool_total_s >= 0.0
        assert stats.errors == []
        assert not (manager.pending_logs_dir / "2026-04-14_09-15.log").exists()
        assert (manager._processed_logs_dir / "2026-04-14_09-15.log").is_file()
        memory_file = manager.memories_dir / "2026-04-14_user-name_a01.md"
        assert memory_file.is_file()
        assert manager.active_memory_path.read_text(encoding="utf-8").count("[2026-04-14_user-name_a01]") == 1

    def test_overlap_update_flow(self, manager: MemoryManager) -> None:
        """Scripted LLM: consult existing, then update instead of create."""
        manager.write_memory(
            "2026-04-10_chess_aaa",
            "User plays chess.",
            kind="preference",
            tags=["chess"],
        )
        steps: list[list[dict[str, Any]]] = [
            [_call_item("list_existing_memories", {"tag": "chess"}, call_id="c1")],
            [
                _call_item(
                    "update_memory",
                    {
                        "id": "2026-04-10_chess_aaa",
                        "body": "User plays chess and prefers the Queen's Gambit.",
                        "sources": ["2026-04-14_09-15.log"],
                    },
                    call_id="c2",
                )
            ],
            [_msg_item("Enriched existing memory.")],
        ]
        fake = _FakeClient(responses=_FakeResponses(on_create=lambda _i: steps.pop(0)))
        Dreamer(manager, model="fake-model", client=fake).run()

        mem = manager.read_memory("2026-04-10_chess_aaa")
        assert "Queen's Gambit" in mem["body"]
        assert "2026-04-14_09-15.log" in mem["frontmatter"].get("sources", [])

    def test_errors_keep_log_in_pending(self, manager: MemoryManager) -> None:
        """If a tool call raises, the log must stay in pending/ for retry."""
        steps: list[list[dict[str, Any]]] = [
            [
                _call_item(
                    "write_memory",
                    {
                        "id": "invalid id!",
                        "body": "bad",
                        "kind": "fact",
                        "tags": ["t"],
                    },
                    call_id="c1",
                )
            ],
            [_msg_item("Giving up.")],
        ]
        fake = _FakeClient(responses=_FakeResponses(on_create=lambda _i: steps.pop(0)))
        [stats] = Dreamer(manager, model="fake-model", client=fake).run()

        assert stats.errors
        assert (manager.pending_logs_dir / "2026-04-14_09-15.log").is_file()

    def test_empty_pending_skips_llm(self, tmp_path: Path) -> None:
        """No pending logs → no LLM calls."""
        mgr = MemoryManager(tmp_path / "data")
        fake = _FakeClient(responses=_FakeResponses(on_create=lambda _i: []))
        stats_list = Dreamer(mgr, model="fake-model", client=fake).run()
        assert stats_list == []
        assert fake.responses.calls == []

    def test_auth_error_aborts_whole_pass(self, manager: MemoryManager) -> None:
        """A 401/auth failure aborts the pass after the first log, never re-tries each log."""
        import httpx
        from openai import AuthenticationError

        # Second pending log, to prove the pass stops early rather than retrying all logs.
        (manager.pending_logs_dir / "2026-04-15_10-00.log").write_text(
            "--- session ---\n10:00:00 user: hi\n", encoding="utf-8"
        )

        def boom(_inp: Any) -> Any:
            resp = httpx.Response(401, request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
            raise AuthenticationError("Missing scopes: api.responses.write", response=resp, body=None)

        fake = _FakeClient(responses=_FakeResponses(on_create=boom))
        # Must not raise: the dreamer swallows the auth failure and aborts cleanly.
        Dreamer(manager, model="fake-model", client=fake).run()

        # Only one LLM attempt, then abort (not one failed call per pending log).
        assert len(fake.responses.calls) == 1
        # Both logs remain in pending for a later pass with a valid key.
        assert (manager.pending_logs_dir / "2026-04-14_09-15.log").is_file()
        assert (manager.pending_logs_dir / "2026-04-15_10-00.log").is_file()


class TestRunDreamPass:
    """Verify the convenience runner's model-selection behaviour."""

    def test_defaults_to_dreamer_model_not_realtime_alias(
        self,
        manager: MemoryManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With no model arg/env, fall back to DEFAULT_DREAMER_MODEL and ignore OPENAI_MODEL_NAME."""
        monkeypatch.delenv("MEMORY_DREAMER_MODEL", raising=False)
        monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-realtime")  # realtime alias must be ignored
        fake = _FakeClient(responses=_FakeResponses(on_create=lambda _i: [_msg_item("done")]))
        run_dream_pass(manager, client=fake)
        assert fake.responses.calls[0]["model"] == DEFAULT_DREAMER_MODEL

    def test_uses_memory_dreamer_model_env(
        self,
        manager: MemoryManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """MEMORY_DREAMER_MODEL is used when set."""
        monkeypatch.setenv("MEMORY_DREAMER_MODEL", "custom-model")
        fake = _FakeClient(responses=_FakeResponses(on_create=lambda _i: [_msg_item("done")]))
        run_dream_pass(manager, client=fake)
        assert fake.responses.calls[0]["model"] == "custom-model"
