"""Tests for the memory.* JSON-RPC methods."""

from typing import Any
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini.apps.jsonrpc_server import JsonRpcServer
import reachy_mini_conversation_app.server_memory as server_memory_mod
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.memory import add_memory_fact, list_memory_facts, format_memory_for_prompt
from reachy_mini_conversation_app.memory_routes import register_memory_methods


def _rpc_call(client: TestClient, method: str, params: dict[str, object] | None = None) -> dict[str, Any]:
    with client.websocket_connect("/rpc") as websocket:
        websocket.send_json({"jsonrpc": "2.0", "id": "1", "method": method, "params": params or {}})
        response: dict[str, Any] = websocket.receive_json()
        return response


def _client(
    instance_path: Path,
    enabled_calls: list[bool] | None = None,
    refreshed: list[bool] | None = None,
    live: bool = True,
) -> TestClient:
    app = FastAPI()
    rpc = JsonRpcServer()

    def _set_enabled(enabled: bool) -> None:
        if enabled_calls is not None:
            enabled_calls.append(enabled)
        config.MEMORY_ENABLED = enabled

    async def _refresh_instructions() -> bool:
        if refreshed is not None:
            refreshed.append(True)
        return live

    register_memory_methods(
        rpc,
        instance_path=instance_path,
        set_enabled=_set_enabled,
        refresh_instructions=_refresh_instructions,
    )
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

    refreshed: list[bool] = []
    result = _rpc_call(_client(tmp_path, refreshed=refreshed), "memory.forget", {"id": doomed.id})["result"]

    assert result["removed"]["id"] == doomed.id
    assert [fact.id for fact in list_memory_facts(tmp_path)] == [kept.id]
    # The fact is in the live prompt too, so forgetting must refresh it.
    assert result["applied_live"] is True
    assert refreshed == [True]


def test_forget_of_nothing_leaves_the_session_alone(tmp_path: Path) -> None:
    """No fact removed means no prompt change worth pushing."""
    refreshed: list[bool] = []

    result = _rpc_call(_client(tmp_path, refreshed=refreshed), "memory.forget", {"id": "m_missing"})["result"]

    assert result == {"ok": True, "removed": None, "applied_live": False}
    assert refreshed == []


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


def test_clear_empties_the_store_and_the_live_prompt(tmp_path: Path) -> None:
    """Facts live in the prompt, so clearing must also refresh the session."""
    add_memory_fact(tmp_path, "Has a dog named Mochi")
    refreshed: list[bool] = []

    result = _rpc_call(_client(tmp_path, refreshed=refreshed), "memory.clear")["result"]

    assert result == {"ok": True, "applied_live": True}
    assert list_memory_facts(tmp_path) == []
    assert refreshed == [True]


def test_clear_reports_when_no_session_was_live(tmp_path: Path) -> None:
    """With no connection the store is still emptied; the next session rebuilds."""
    add_memory_fact(tmp_path, "Has a dog named Mochi")

    result = _rpc_call(_client(tmp_path, live=False), "memory.clear")["result"]

    assert result == {"ok": True, "applied_live": False}
    assert list_memory_facts(tmp_path) == []


def test_set_enabled_stops_prompt_injection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Turning memory off keeps the facts but drops them from the prompt."""
    monkeypatch.setattr(config, "MEMORY_ENABLED", True)
    add_memory_fact(tmp_path, "Has a dog named Mochi")
    assert "Mochi" in format_memory_for_prompt(tmp_path)

    calls: list[bool] = []
    refreshed: list[bool] = []
    result = _rpc_call(_client(tmp_path, calls, refreshed), "memory.set_enabled", {"enabled": False})["result"]

    assert calls == [False]
    # Read live, so a prompt refresh applies it without a reconnect.
    assert result == {"enabled": False, "applied_live": True}
    assert refreshed == [True]
    assert format_memory_for_prompt(tmp_path) == ""
    assert len(list_memory_facts(tmp_path)) == 1


def test_set_enabled_rejects_non_boolean(tmp_path: Path) -> None:
    """A malformed toggle must not silently disable memory."""
    error = _rpc_call(_client(tmp_path), "memory.set_enabled", {"enabled": "yes"})["error"]

    assert error["data"]["reason"] == "invalid_params"


def test_server_memory_reports_server_status_and_refuses_local_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the speech server's memory, clients see its status and cannot change the unused local store."""
    add_memory_fact(tmp_path, "Has a dog named Mochi")
    monkeypatch.setattr(config, "MEMORY_BACKEND", "server")
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://speech.test:8765/v1/realtime")
    requested: list[str] = []
    status = {"enabled": True, "profile": "read_only", "state": "ready", "deadlineMs": 450}

    def capabilities(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(200, json={"pipelines": [], "memory": status})

    async_client = httpx.AsyncClient
    monkeypatch.setattr(
        server_memory_mod.httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=httpx.MockTransport(capabilities), **kwargs),
    )
    client = _client(tmp_path)

    listed = _rpc_call(client, "memory.list")["result"]

    assert listed["backend"] == "server"
    assert listed["facts"] == []
    assert listed["server"] == {"reachable": True, **status}
    assert requested == ["http://speech.test:8765/v1/realtime/capabilities"]
    for method, params in (
        ("memory.forget", {"query": "Mochi"}),
        ("memory.clear", {}),
        ("memory.set_enabled", {"enabled": False}),
    ):
        assert _rpc_call(client, method, params)["error"]["data"]["reason"] == "memory_on_server"
    assert [fact.text for fact in list_memory_facts(tmp_path)] == ["Has a dog named Mochi"]
