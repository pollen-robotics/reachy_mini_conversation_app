"""Tests for the memory.* JSON-RPC methods."""

from typing import Any
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.memory import add_memory_fact, list_memory_facts, format_memory_for_prompt
from reachy_mini_conversation_app.memory_routes import register_memory_methods


def _rpc_call(client: TestClient, method: str, params: dict[str, object] | None = None) -> dict[str, Any]:
    with client.websocket_connect("/rpc") as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": "1", "method": method, "params": params or {}})
        response: dict[str, Any] = websocket.receive_json()
        return response


def _client(instance_path: Path, enabled_calls: list[bool] | None = None) -> TestClient:
    app = FastAPI()
    rpc = JsonRpcServer()

    def _set_enabled(enabled: bool) -> str:
        if enabled_calls is not None:
            enabled_calls.append(enabled)
        config.MEMORY_ENABLED = enabled
        return "Saved."

    register_memory_methods(rpc, instance_path=instance_path, set_enabled=_set_enabled)
    rpc.mount(app)
    return TestClient(app)


def test_list_returns_facts_newest_first(tmp_path: Path) -> None:
    """The client sees stored facts with the cap and the enabled flag."""
    add_memory_fact(tmp_path, "Has a dog named Mochi")
    add_memory_fact(tmp_path, "Prefers replies in French")

    result = _rpc_call(_client(tmp_path), "memory.list")["result"]

    assert [fact["text"] for fact in result["facts"]] == [
        "Prefers replies in French",
        "Has a dog named Mochi",
    ]
    assert result["max_facts"] == 60
    assert result["enabled"] is True


def test_forget_by_id(tmp_path: Path) -> None:
    """A list-and-tap client removes the exact fact it displayed."""
    kept = add_memory_fact(tmp_path, "Has a dog named Mochi")
    doomed = add_memory_fact(tmp_path, "Prefers replies in French")
    assert kept is not None and doomed is not None

    result = _rpc_call(_client(tmp_path), "memory.forget", {"id": doomed.id})["result"]

    assert result["removed"]["id"] == doomed.id
    assert [fact.id for fact in list_memory_facts(tmp_path)] == [kept.id]


def test_forget_by_query_still_works(tmp_path: Path) -> None:
    """The substring path the model's forget tool uses is unchanged."""
    add_memory_fact(tmp_path, "Has a dog named Mochi")

    result = _rpc_call(_client(tmp_path), "memory.forget", {"query": "mochi"})["result"]

    assert result["removed"]["text"] == "Has a dog named Mochi"
    assert list_memory_facts(tmp_path) == []


def test_forget_without_target_is_invalid(tmp_path: Path) -> None:
    """Refusing an empty forget avoids deleting an arbitrary fact."""
    error = _rpc_call(_client(tmp_path), "memory.forget")["error"]

    assert error["data"]["reason"] == "invalid_params"


def test_clear_empties_the_store(tmp_path: Path) -> None:
    """Clearing is what the phone's 'Clear memory' button calls."""
    add_memory_fact(tmp_path, "Has a dog named Mochi")

    assert _rpc_call(_client(tmp_path), "memory.clear")["result"] == {"ok": True}
    assert list_memory_facts(tmp_path) == []


def test_set_enabled_stops_prompt_injection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Turning memory off keeps the facts but drops them from the prompt."""
    monkeypatch.setattr(config, "MEMORY_ENABLED", True)
    add_memory_fact(tmp_path, "Has a dog named Mochi")
    assert "Mochi" in format_memory_for_prompt(tmp_path)

    calls: list[bool] = []
    result = _rpc_call(_client(tmp_path, calls), "memory.set_enabled", {"enabled": False})["result"]

    assert calls == [False]
    assert result["enabled"] is False
    assert format_memory_for_prompt(tmp_path) == ""
    assert len(list_memory_facts(tmp_path)) == 1


def test_set_enabled_rejects_non_boolean(tmp_path: Path) -> None:
    """A malformed toggle must not silently disable memory."""
    error = _rpc_call(_client(tmp_path), "memory.set_enabled", {"enabled": "yes"})["error"]

    assert error["data"]["reason"] == "invalid_params"
