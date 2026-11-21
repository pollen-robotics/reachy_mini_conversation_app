"""REACHY_MINI_MEMORY_BACKEND=auto: the speech server's memory when it reports one, the local store otherwise."""

import pytest

import reachy_mini_conversation_app.server_memory as server_memory_mod
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.memory import local_memory_in_use


@pytest.fixture(autouse=True)
def _fresh_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_memory_mod, "_resolved", "local")
    monkeypatch.setattr(config, "MEMORY_ENABLED", True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ({"reachable": True, "enabled": True, "state": "ready"}, "server"),
        ({"reachable": True, "enabled": False}, "local"),
        ({"reachable": False, "reason": "connection refused"}, "local"),
        ({"reachable": False, "reason": "no local speech server is configured"}, "local"),
    ],
)
async def test_auto_uses_the_server_memory_only_when_the_server_reports_one(
    monkeypatch: pytest.MonkeyPatch, status: dict[str, object], expected: str
) -> None:
    """Nothing to configure: a server with memory takes over, any other server leaves the local store in use."""
    monkeypatch.setattr(config, "MEMORY_BACKEND", "auto")

    async def reported(timeout_s: float = 3.0) -> dict[str, object]:
        return status

    monkeypatch.setattr(server_memory_mod, "server_memory_status", reported)

    assert await server_memory_mod.resolve_memory_backend() == expected
    assert server_memory_mod.effective_memory_backend() == expected
    assert local_memory_in_use() is (expected == "local")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["local", "server"])
async def test_an_explicit_backend_is_kept_and_the_server_is_not_asked(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """An explicit local or server is kept as configured, without asking the speech server."""
    monkeypatch.setattr(config, "MEMORY_BACKEND", backend)

    async def must_not_be_called(timeout_s: float = 3.0) -> dict[str, object]:
        raise AssertionError("an explicit backend does not ask the server")

    monkeypatch.setattr(server_memory_mod, "server_memory_status", must_not_be_called)

    assert await server_memory_mod.resolve_memory_backend() == backend
    assert server_memory_mod.effective_memory_backend() == backend


def test_auto_is_local_until_a_session_has_asked_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Before any session start, auto behaves as the local store, as the app did before."""
    monkeypatch.setattr(config, "MEMORY_BACKEND", "auto")
    assert server_memory_mod.effective_memory_backend() == "local"
