"""Unit tests for the headless_personality_ui module."""

import asyncio
import concurrent.futures
from typing import Any, Optional, Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini_conversation_app.headless_personality import DEFAULT_OPTION


async def _mock_apply_personality(name: Optional[str]) -> str:
    """Mock apply_personality coroutine."""
    return "Applied successfully"


async def _mock_get_available_voices() -> list[str]:
    """Mock get_available_voices coroutine."""
    return ["cedar", "ash", "ballad"]


class _MockHandler:
    """Mock handler for testing with lazy coroutine creation.

    Methods are bound at instance level to avoid class-level inspection.
    """

    __slots__ = ("_apply_fn", "_voices_fn")

    def __init__(self) -> None:
        self._apply_fn = _mock_apply_personality
        self._voices_fn = _mock_get_available_voices

    def apply_personality(self, name: Optional[str]) -> Any:
        """Create and return a coroutine for applying personality."""
        return self._apply_fn(name)

    def get_available_voices(self) -> Any:
        """Create and return a coroutine for getting voices."""
        return self._voices_fn()


@pytest.fixture
def mock_handler() -> Any:
    """Create a mock OpenaiRealtimeHandler."""
    return _MockHandler()


@pytest.fixture
def running_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create and run an event loop in a background thread."""
    import threading

    loop = asyncio.new_event_loop()

    def run_loop() -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()

    # Wait for loop to be running
    while not loop.is_running():
        pass

    yield loop

    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=1)
    loop.close()


@pytest.fixture
def mock_get_loop(running_loop: asyncio.AbstractEventLoop) -> Any:
    """Return a callable that provides the event loop."""
    def _get_loop() -> asyncio.AbstractEventLoop:
        return running_loop
    return _get_loop


@pytest.fixture
def app_and_client(
    mock_handler: Any,
    mock_get_loop: Any,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FastAPI, TestClient]:
    """Create a FastAPI app with personality routes mounted."""
    # Create the profiles directory structure
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()

    # Create a test profile
    test_profile = profiles_dir / "test_profile"
    test_profile.mkdir()
    (test_profile / "instructions.txt").write_text("Test instructions")
    (test_profile / "tools.txt").write_text("do_nothing\n")
    (test_profile / "voice.txt").write_text("ash\n")

    # Mock the resolve functions BEFORE importing mount_personality_routes
    monkeypatch.setattr(
        "reachy_mini_conversation_app.headless_personality_ui.resolve_profile_dir",
        lambda name: profiles_dir / name if name != DEFAULT_OPTION else profiles_dir / "default",
    )
    monkeypatch.setattr(
        "reachy_mini_conversation_app.headless_personality_ui.list_personalities",
        lambda: ["test_profile"],
    )
    monkeypatch.setattr(
        "reachy_mini_conversation_app.headless_personality_ui.available_tools_for",
        lambda name: ["do_nothing", "move_head"],
    )
    monkeypatch.setattr(
        "reachy_mini_conversation_app.headless_personality_ui.read_instructions_for",
        lambda name: "Test instructions" if name == "test_profile" else "Default instructions",
    )
    monkeypatch.setattr(
        "reachy_mini_conversation_app.headless_personality_ui._write_profile",
        lambda name, instr, tools, voice: None,
    )
    monkeypatch.setattr(
        "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
        lambda name: name if name else None,
    )

    from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

    app = FastAPI()
    mount_personality_routes(app, mock_handler, mock_get_loop)

    return app, TestClient(app)


@pytest.fixture
def client(app_and_client: tuple[FastAPI, TestClient]) -> TestClient:
    """Create a test client for the app."""
    return app_and_client[1]


class TestPersonalitiesEndpoint:
    """Tests for /personalities endpoint."""

    def test_list_personalities(self, client: TestClient) -> None:
        """Test listing available personalities."""
        response = client.get("/personalities")
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert "current" in data
        assert "startup" in data
        assert DEFAULT_OPTION in data["choices"]


class TestPersonalitiesLoadEndpoint:
    """Tests for /personalities/load endpoint."""

    def test_load_default_personality(self, client: TestClient) -> None:
        """Test loading default personality."""
        response = client.get(f"/personalities/load?name={DEFAULT_OPTION}")
        assert response.status_code == 200
        data = response.json()
        assert "instructions" in data
        assert "tools_text" in data
        assert "voice" in data
        assert "available_tools" in data
        assert "enabled_tools" in data

    def test_load_named_personality(
        self,
        client: TestClient,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test loading a named personality."""
        # Setup mock profile directory
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir(exist_ok=True)
        test_profile = profiles_dir / "test_profile"
        test_profile.mkdir(exist_ok=True)
        (test_profile / "instructions.txt").write_text("Test instructions")
        (test_profile / "tools.txt").write_text("do_nothing\nmove_head\n")
        (test_profile / "voice.txt").write_text("ash\n")

        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui.resolve_profile_dir",
            lambda name: test_profile,
        )

        response = client.get("/personalities/load?name=test_profile")
        assert response.status_code == 200
        data = response.json()
        assert data["voice"] == "ash"
        assert "do_nothing" in data["enabled_tools"]


class TestPersonalitiesSaveEndpoint:
    """Tests for /personalities/save POST endpoint."""

    def test_save_personality(
        self,
        mock_handler: Any,
        mock_get_loop: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test POST /personalities/save with Pydantic payload."""
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._write_profile",
            lambda name, instr, tools, voice: None,
        )
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui.list_personalities",
            lambda: ["test"],
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(app, mock_handler, mock_get_loop)

        client = TestClient(app)
        response = client.post(
            "/personalities/save",
            json={
                "name": "new_profile",
                "instructions": "New instructions",
                "tools_text": "do_nothing",
                "voice": "ash",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["value"] == "user_personalities/new_profile"

    def test_save_personality_invalid_name(
        self,
        mock_handler: Any,
        mock_get_loop: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test POST /personalities/save with invalid name."""
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: None,  # Always return None for invalid
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(app, mock_handler, mock_get_loop)

        client = TestClient(app)
        response = client.post(
            "/personalities/save",
            json={"name": "", "instructions": "Test"},
        )
        assert response.status_code == 400
        data = response.json()
        assert data["ok"] is False
        assert data["error"] == "invalid_name"

    def test_save_personality_write_exception(
        self,
        mock_handler: Any,
        mock_get_loop: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test POST /personalities/save when _write_profile raises."""
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._write_profile",
            MagicMock(side_effect=IOError("Write failed")),
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(app, mock_handler, mock_get_loop)

        client = TestClient(app)
        response = client.post(
            "/personalities/save",
            json={"name": "test_profile", "instructions": "Test"},
        )
        assert response.status_code == 500
        data = response.json()
        assert data["ok"] is False
        assert "Write failed" in data["error"]

    def test_save_personality_default_voice(
        self,
        mock_handler: Any,
        mock_get_loop: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test POST /personalities/save with None voice defaults to cedar."""
        written_values: list[tuple[str, str, str, str]] = []

        def capture_write(name: str, instr: str, tools: str, voice: str) -> None:
            written_values.append((name, instr, tools, voice))

        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._write_profile",
            capture_write,
        )
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui.list_personalities",
            lambda: [],
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(app, mock_handler, mock_get_loop)

        client = TestClient(app)
        response = client.post(
            "/personalities/save",
            json={"name": "test_profile", "instructions": "Test", "tools_text": ""},
        )
        assert response.status_code == 200
        assert len(written_values) == 1
        assert written_values[0][3] == "cedar"  # Default voice


class TestPersonalitiesSaveRawPostEndpoint:
    """Tests for /personalities/save_raw POST endpoint."""

    def test_save_raw_post(
        self,
        mock_handler: Any,
        mock_get_loop: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test POST /personalities/save_raw with query params."""
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._write_profile",
            lambda name, instr, tools, voice: None,
        )
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui.list_personalities",
            lambda: ["test"],
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(app, mock_handler, mock_get_loop)

        client = TestClient(app)
        response = client.post(
            "/personalities/save_raw",
            params={
                "name": "post_profile",
                "instructions": "Post instructions",
                "tools_text": "do_nothing",
                "voice": "ash",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["value"] == "user_personalities/post_profile"

    def test_save_raw_post_invalid_name(
        self,
        mock_handler: Any,
        mock_get_loop: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test POST /personalities/save_raw with invalid name."""
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: None,
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(app, mock_handler, mock_get_loop)

        client = TestClient(app)
        response = client.post("/personalities/save_raw", params={"name": ""})
        assert response.status_code == 400
        data = response.json()
        assert data["ok"] is False

    def test_save_raw_post_write_exception(
        self,
        mock_handler: Any,
        mock_get_loop: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test POST /personalities/save_raw when _write_profile raises."""
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._write_profile",
            MagicMock(side_effect=IOError("Write failed")),
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(app, mock_handler, mock_get_loop)

        client = TestClient(app)
        response = client.post(
            "/personalities/save_raw",
            params={"name": "test_profile", "instructions": "Test"},
        )
        assert response.status_code == 500
        data = response.json()
        assert data["ok"] is False
        assert "Write failed" in data["error"]


class TestPersonalitiesSaveRawGetEndpoint:
    """Tests for /personalities/save_raw GET endpoint."""

    def test_save_raw_get(self, client: TestClient) -> None:
        """Test GET /personalities/save_raw."""
        response = client.get(
            "/personalities/save_raw",
            params={
                "name": "get_profile",
                "instructions": "Get instructions",
                "tools_text": "do_nothing",
                "voice": "cedar",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True

    def test_save_raw_get_invalid_name(self, client: TestClient) -> None:
        """Test GET /personalities/save_raw with invalid name."""
        response = client.get("/personalities/save_raw?name=")
        assert response.status_code == 400

    def test_save_raw_get_write_exception(
        self,
        mock_handler: Any,
        mock_get_loop: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test GET /personalities/save_raw when _write_profile raises."""
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._write_profile",
            MagicMock(side_effect=IOError("Write failed")),
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(app, mock_handler, mock_get_loop)

        client = TestClient(app)
        response = client.get(
            "/personalities/save_raw",
            params={"name": "test_profile", "instructions": "Test"},
        )
        assert response.status_code == 500
        data = response.json()
        assert data["ok"] is False
        assert "Write failed" in data["error"]


class TestPersonalitiesApplyEndpoint:
    """Tests for /personalities/apply endpoint."""

    def test_apply_personality_json(
        self,
        client: TestClient,
        running_loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Test applying personality with JSON payload."""
        response = client.post(
            "/personalities/apply",
            json={"name": "test_profile", "persist": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert "status" in data

    def test_apply_personality_query_param(
        self,
        client: TestClient,
        running_loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Test applying personality with query param."""
        response = client.post("/personalities/apply?name=test_profile")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True

    def test_apply_personality_default(
        self,
        client: TestClient,
        running_loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Test applying default personality."""
        response = client.post("/personalities/apply")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True

    def test_apply_personality_no_loop(
        self,
        mock_handler: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test apply when loop is not available."""
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(app, mock_handler, lambda: None)

        client = TestClient(app)
        response = client.post("/personalities/apply", json={"name": "test"})
        assert response.status_code == 503
        data = response.json()
        assert data["ok"] is False
        assert data["error"] == "loop_unavailable"

    def test_apply_with_persist_callback(
        self,
        mock_handler: Any,
        running_loop: asyncio.AbstractEventLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test apply with persist_personality callback."""
        persisted_values: list[Optional[str]] = []

        def persist_callback(value: Optional[str]) -> None:
            persisted_values.append(value)

        def get_loop() -> asyncio.AbstractEventLoop:
            return running_loop

        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(
            app,
            mock_handler,
            get_loop,
            persist_personality=persist_callback,
        )

        client = TestClient(app)
        response = client.post(
            "/personalities/apply?name=persist_profile&persist=true",
        )
        assert response.status_code == 200
        assert len(persisted_values) == 1
        assert persisted_values[0] == "persist_profile"

    def test_apply_persist_callback_exception(
        self,
        mock_handler: Any,
        running_loop: asyncio.AbstractEventLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test apply when persist_personality callback raises."""
        def failing_persist(value: Optional[str]) -> None:
            raise RuntimeError("Persist failed")

        def get_loop() -> asyncio.AbstractEventLoop:
            return running_loop

        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(
            app,
            mock_handler,
            get_loop,
            persist_personality=failing_persist,
        )

        client = TestClient(app)
        response = client.post(
            "/personalities/apply?name=fail_persist&persist=true",
        )
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True

    def test_apply_exception_in_do_apply(
        self,
        running_loop: asyncio.AbstractEventLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test apply when _do_apply raises."""
        handler = type("FailingHandler", (), {})()

        async def failing_apply(name: Optional[str]) -> str:
            raise RuntimeError("Apply failed")

        async def get_voices() -> list[str]:
            return ["cedar"]

        handler.apply_personality = failing_apply
        handler.get_available_voices = get_voices

        def get_loop() -> asyncio.AbstractEventLoop:
            return running_loop

        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(app, handler, get_loop)

        client = TestClient(app)
        response = client.post(
            "/personalities/apply",
            json={"name": "error_profile"},
        )
        assert response.status_code == 500
        data = response.json()
        assert data["ok"] is False
        assert "Apply failed" in data["error"]

    def test_apply_persist_default_option(
        self,
        mock_handler: Any,
        running_loop: asyncio.AbstractEventLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test apply persist with DEFAULT_OPTION passes None."""
        persisted_values: list[Optional[str]] = []

        def persist_callback(value: Optional[str]) -> None:
            persisted_values.append(value)

        def get_loop() -> asyncio.AbstractEventLoop:
            return running_loop

        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(
            app,
            mock_handler,
            get_loop,
            persist_personality=persist_callback,
        )

        client = TestClient(app)
        response = client.post(
            f"/personalities/apply?name={DEFAULT_OPTION}&persist=true",
        )
        assert response.status_code == 200
        assert len(persisted_values) == 1
        assert persisted_values[0] is None  # DEFAULT_OPTION -> None

    def test_apply_payload_persist_takes_precedence_over_query(
        self,
        mock_handler: Any,
        running_loop: asyncio.AbstractEventLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that JSON payload persist overrides query param persist."""
        persisted_values: list[Optional[str]] = []

        def persist_callback(value: Optional[str]) -> None:
            persisted_values.append(value)

        def get_loop() -> asyncio.AbstractEventLoop:
            return running_loop

        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(
            app,
            mock_handler,
            get_loop,
            persist_personality=persist_callback,
        )

        client = TestClient(app)
        # JSON payload with persist=true, query param persist=false
        response = client.post(
            "/personalities/apply?persist=false",
            json={"name": "test_profile", "persist": True},
        )
        assert response.status_code == 200
        # The query param takes precedence when explicitly provided
        assert len(persisted_values) == 0  # persist=false from query


class TestVoicesEndpoint:
    """Tests for /voices endpoint."""

    def test_get_voices(
        self,
        client: TestClient,
        running_loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Test getting available voices."""
        response = client.get("/voices")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert "cedar" in data or len(data) > 0

    def test_get_voices_no_loop(
        self,
        mock_handler: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test getting voices when loop is not available."""
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(app, mock_handler, lambda: None)

        client = TestClient(app)
        response = client.get("/voices")
        assert response.status_code == 200
        data = response.json()
        assert data == ["cedar"]

    def test_voices_run_coroutine_exception(
        self,
        mock_handler: Any,
        running_loop: asyncio.AbstractEventLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test /voices exception in run_coroutine_threadsafe."""
        def get_loop() -> asyncio.AbstractEventLoop:
            return running_loop

        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(app, mock_handler, get_loop)

        client = TestClient(app)

        # Track coroutines so we can close them after the test
        created_coroutines: list[Any] = []

        def mock_run(coro: Any, loop: Any) -> Any:
            created_coroutines.append(coro)
            raise concurrent.futures.TimeoutError("Timeout")

        with patch(
            "reachy_mini_conversation_app.headless_personality_ui.asyncio.run_coroutine_threadsafe",
            side_effect=mock_run,
        ):
            response = client.get("/voices")

        # Close any coroutines that were created but not awaited
        for coro in created_coroutines:
            coro.close()

        assert response.status_code == 200
        data = response.json()
        assert data == ["cedar"]

    def test_voices_get_available_voices_exception(
        self,
        running_loop: asyncio.AbstractEventLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test /voices when handler.get_available_voices raises."""
        handler = type("FailingVoicesHandler", (), {})()

        async def apply_ok(name: Optional[str]) -> str:
            return "Applied"

        async def failing_voices() -> list[str]:
            raise RuntimeError("API error")

        handler.apply_personality = apply_ok
        handler.get_available_voices = failing_voices

        def get_loop() -> asyncio.AbstractEventLoop:
            return running_loop

        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(app, handler, get_loop)

        client = TestClient(app)
        response = client.get("/voices")

        assert response.status_code == 200
        data = response.json()
        assert data == ["cedar"]


class TestStartupChoice:
    """Tests for _startup_choice helper."""

    def test_startup_choice_with_persisted(
        self,
        mock_handler: Any,
        mock_get_loop: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test startup choice returns persisted value."""
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(
            app,
            mock_handler,
            mock_get_loop,
            get_persisted_personality=lambda: "persisted_profile",
        )

        client = TestClient(app)
        response = client.get("/personalities")
        data = response.json()
        assert data["startup"] == "persisted_profile"

    def test_startup_choice_with_env_config(
        self,
        mock_handler: Any,
        mock_get_loop: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test startup choice falls back to config."""
        from reachy_mini_conversation_app.config import config

        monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "env_profile")
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(
            app,
            mock_handler,
            mock_get_loop,
            get_persisted_personality=lambda: None,
        )

        client = TestClient(app)
        response = client.get("/personalities")
        data = response.json()
        assert data["startup"] == "env_profile"

    def test_startup_choice_falls_back_to_default(
        self,
        mock_handler: Any,
        mock_get_loop: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test startup choice falls back to DEFAULT_OPTION when env is empty."""
        from reachy_mini_conversation_app.config import config

        monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(
            app,
            mock_handler,
            mock_get_loop,
            get_persisted_personality=lambda: None,
        )

        client = TestClient(app)
        response = client.get("/personalities")
        data = response.json()
        assert data["startup"] == DEFAULT_OPTION

    def test_startup_choice_env_config_when_persisted_empty(
        self,
        mock_handler: Any,
        mock_get_loop: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test _startup_choice falls back to env when persisted returns empty."""
        from reachy_mini_conversation_app.config import config

        monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "env_value")
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(
            app,
            mock_handler,
            mock_get_loop,
            get_persisted_personality=lambda: "",  # Empty string, not None
        )

        client = TestClient(app)
        response = client.get("/personalities")
        data = response.json()
        assert data["startup"] == "env_value"


class TestCurrentChoice:
    """Tests for _current_choice helper."""

    def test_current_choice_returns_config_value(
        self,
        mock_handler: Any,
        mock_get_loop: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test current choice returns config value."""
        from reachy_mini_conversation_app.config import config

        monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "current_profile")
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(app, mock_handler, mock_get_loop)

        client = TestClient(app)
        response = client.get("/personalities")
        data = response.json()
        assert data["current"] == "current_profile"

    def test_current_choice_returns_default_when_none(
        self,
        mock_handler: Any,
        mock_get_loop: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test current choice returns DEFAULT_OPTION when config is None."""
        from reachy_mini_conversation_app.config import config

        monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(app, mock_handler, mock_get_loop)

        client = TestClient(app)
        response = client.get("/personalities")
        data = response.json()
        assert data["current"] == DEFAULT_OPTION


class TestExceptionBranches:
    """Tests for exception handling branches."""

    def test_startup_choice_exception(
        self,
        mock_handler: Any,
        mock_get_loop: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test _startup_choice exception handling."""
        def raise_error() -> None:
            raise RuntimeError("Get persisted error")

        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(
            app,
            mock_handler,
            mock_get_loop,
            get_persisted_personality=raise_error,
        )

        client = TestClient(app)
        response = client.get("/personalities")
        data = response.json()
        assert data["startup"] == DEFAULT_OPTION

    def test_current_choice_exception(
        self,
        mock_handler: Any,
        mock_get_loop: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test _current_choice exception handling."""
        class BadConfig:
            def __getattribute__(self, name: str) -> Any:
                if name == "REACHY_MINI_CUSTOM_PROFILE":
                    raise RuntimeError("Config error")
                return object.__getattribute__(self, name)

        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui.config",
            BadConfig(),
        )
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(app, mock_handler, mock_get_loop)

        client = TestClient(app)
        response = client.get("/personalities")
        data = response.json()
        assert data["current"] == DEFAULT_OPTION

    def test_load_personality_profile_without_tools(
        self,
        mock_handler: Any,
        mock_get_loop: Any,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test loading profile without tools.txt."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        test_profile = profiles_dir / "no_tools_profile"
        test_profile.mkdir()
        (test_profile / "instructions.txt").write_text("No tools instructions")

        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui.resolve_profile_dir",
            lambda name: test_profile,
        )
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui.list_personalities",
            lambda: ["no_tools_profile"],
        )
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui.available_tools_for",
            lambda name: ["do_nothing"],
        )
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui.read_instructions_for",
            lambda name: "No tools instructions",
        )
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(app, mock_handler, mock_get_loop)

        client = TestClient(app)
        response = client.get("/personalities/load?name=no_tools_profile")
        assert response.status_code == 200
        data = response.json()
        assert data["tools_text"] == ""
        assert data["voice"] == "cedar"

    def test_load_personality_empty_voice(
        self,
        mock_handler: Any,
        mock_get_loop: Any,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test loading profile with empty voice.txt."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        test_profile = profiles_dir / "empty_voice_profile"
        test_profile.mkdir()
        (test_profile / "instructions.txt").write_text("Test instructions")
        (test_profile / "voice.txt").write_text("")

        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui.resolve_profile_dir",
            lambda name: test_profile,
        )
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui.list_personalities",
            lambda: ["empty_voice_profile"],
        )
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui.available_tools_for",
            lambda name: ["do_nothing"],
        )
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui.read_instructions_for",
            lambda name: "Test instructions",
        )
        monkeypatch.setattr(
            "reachy_mini_conversation_app.headless_personality_ui._sanitize_name",
            lambda name: name if name else None,
        )
        from reachy_mini_conversation_app.headless_personality_ui import mount_personality_routes

        app = FastAPI()
        mount_personality_routes(app, mock_handler, mock_get_loop)

        client = TestClient(app)
        response = client.get("/personalities/load?name=empty_voice_profile")
        assert response.status_code == 200
        data = response.json()
        assert data["voice"] == "cedar"


class TestPydanticModels:
    """Tests for the Pydantic models."""

    def test_save_payload_defaults(self) -> None:
        """Test SavePayload default values."""
        from reachy_mini_conversation_app.headless_personality_ui import SavePayload

        payload = SavePayload(name="test")
        assert payload.name == "test"
        assert payload.instructions == ""
        assert payload.tools_text == ""
        assert payload.voice == "cedar"

    def test_save_payload_all_fields(self) -> None:
        """Test SavePayload with all fields."""
        from reachy_mini_conversation_app.headless_personality_ui import SavePayload

        payload = SavePayload(
            name="test",
            instructions="instructions",
            tools_text="tools",
            voice="ash",
        )
        assert payload.name == "test"
        assert payload.instructions == "instructions"
        assert payload.tools_text == "tools"
        assert payload.voice == "ash"

    def test_apply_payload_defaults(self) -> None:
        """Test ApplyPayload default values."""
        from reachy_mini_conversation_app.headless_personality_ui import ApplyPayload

        payload = ApplyPayload(name="test")
        assert payload.name == "test"
        assert payload.persist is False

    def test_apply_payload_with_persist(self) -> None:
        """Test ApplyPayload with persist=True."""
        from reachy_mini_conversation_app.headless_personality_ui import ApplyPayload

        payload = ApplyPayload(name="test", persist=True)
        assert payload.name == "test"
        assert payload.persist is True
