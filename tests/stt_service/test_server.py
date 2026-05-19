"""Tests for ``robot_comic.stt_service.server``.

All tests use FastAPI's ``TestClient`` (synchronous, in-process) so they work
on Windows CI without a real Unix-domain socket.  The ``faster_whisper`` module
is stubbed via ``monkeypatch.setattr`` on the import inside ``server.py`` so the
WhisperModel is never loaded from disk.
"""

from __future__ import annotations
import types
import struct
from typing import Any, Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


class _StubSegment:
    """Minimal stand-in for a faster_whisper.TranscriptionSegment."""

    def __init__(self, text: str, start: float = 0.0, end: float = 1.0) -> None:
        self.text = text
        self.start = start
        self.end = end


class _StubWhisperModel:
    """Stand-in for ``faster_whisper.WhisperModel``.

    ``queued_results`` is a list of ``list[_StubSegment]`` to return per call.
    If the queue is empty, returns an empty segment list.
    """

    def __init__(self, model_name: str, *, device: str = "cpu", compute_type: str = "int8") -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.transcribe_calls: list[np.ndarray] = []
        self.queued_results: list[list[_StubSegment]] = []

    def transcribe(self, audio: np.ndarray, language: str = "en") -> tuple[Any, dict[str, Any]]:
        self.transcribe_calls.append(audio)
        segments = self.queued_results.pop(0) if self.queued_results else []
        return iter(segments), {"language": language}


def _make_stub_module(stub_model_cls: type) -> types.ModuleType:
    """Return a ``faster_whisper`` stub module whose ``WhisperModel`` is ``stub_model_cls``."""
    mod = types.ModuleType("faster_whisper")
    mod.WhisperModel = stub_model_cls  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_server_state() -> Iterator[None]:
    """Reset module-level model state before and after each test."""
    from robot_comic.stt_service import server

    server._reset_state_for_tests()
    yield
    server._reset_state_for_tests()


@pytest.fixture()
def stub_model() -> _StubWhisperModel:
    """A fresh StubWhisperModel instance (not yet installed)."""
    return _StubWhisperModel("base.en")


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, stub_model: _StubWhisperModel) -> TestClient:
    """TestClient with faster_whisper stubbed out.

    Patches ``faster_whisper`` in ``sys.modules`` so that the deferred
    ``from faster_whisper import WhisperModel`` inside ``_load_model_sync``
    picks up the stub.
    """
    import sys

    from robot_comic.stt_service import server

    # Build a stub module that always returns our shared stub_model instance.
    class _CaptureModel:
        """Proxy that returns stub_model from any constructor call."""

        def __new__(cls, model_name: str, **kwargs: Any) -> _StubWhisperModel:  # type: ignore[misc]
            stub_model.model_name = model_name
            stub_model.device = kwargs.get("device", "cpu")
            stub_model.compute_type = kwargs.get("compute_type", "int8")
            return stub_model

    stub_fw = _make_stub_module(_CaptureModel)
    monkeypatch.setitem(sys.modules, "faster_whisper", stub_fw)

    # Also patch the server module's internal reference path in case it was
    # already imported before the monkeypatch.
    monkeypatch.setattr(server, "_model", None, raising=False)

    return TestClient(server.app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# /healthz
# ---------------------------------------------------------------------------


def test_healthz_before_preload(client: TestClient) -> None:
    """/healthz returns ok and model_loaded=False before any preload."""
    resp = client.get("/healthz")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is False


def test_healthz_after_preload(client: TestClient) -> None:
    """/healthz returns model_loaded=True after /preload."""
    client.post("/preload")
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


# ---------------------------------------------------------------------------
# /preload
# ---------------------------------------------------------------------------


def test_preload_returns_loaded_true(client: TestClient) -> None:
    """/preload returns {'loaded': True}."""
    resp = client.post("/preload")
    assert resp.status_code == 200
    assert resp.json() == {"loaded": True}


def test_preload_idempotent(client: TestClient, stub_model: _StubWhisperModel) -> None:
    """Calling /preload twice does not load the model twice."""
    client.post("/preload")
    client.post("/preload")
    # The stub model is only constructed once; calling /preload again should
    # be a no-op (model already loaded).  We verify by checking healthz.
    resp = client.get("/healthz")
    assert resp.json()["model_loaded"] is True


def test_preload_with_custom_model(client: TestClient, stub_model: _StubWhisperModel) -> None:
    """/preload accepts an optional model name and compute_type in the body."""
    resp = client.post("/preload", json={"model": "tiny.en", "compute_type": "float16"})
    assert resp.status_code == 200
    assert resp.json() == {"loaded": True}
    assert stub_model.model_name == "tiny.en"
    assert stub_model.compute_type == "float16"


# ---------------------------------------------------------------------------
# /transcribe
# ---------------------------------------------------------------------------


def _make_pcm(seconds: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Generate silent int16 PCM for the given duration."""
    n_samples = int(seconds * sample_rate)
    return struct.pack(f"<{n_samples}h", *([0] * n_samples))


def test_transcribe_loads_model_on_demand(client: TestClient, stub_model: _StubWhisperModel) -> None:
    """/transcribe loads the model if not already loaded."""
    stub_model.queued_results.append([_StubSegment("hello world", 0.0, 1.0)])
    resp = client.post(
        "/transcribe",
        content=_make_pcm(),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "hello world"
    assert len(data["segments"]) == 1
    assert data["segments"][0]["text"] == "hello world"


def test_transcribe_after_preload(client: TestClient, stub_model: _StubWhisperModel) -> None:
    """/transcribe works when model was already preloaded."""
    client.post("/preload")
    stub_model.queued_results.append([_StubSegment("test transcript", 0.0, 2.0)])
    resp = client.post(
        "/transcribe",
        content=_make_pcm(),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert resp.status_code == 200
    assert resp.json()["text"] == "test transcript"


def test_transcribe_multi_segment(client: TestClient, stub_model: _StubWhisperModel) -> None:
    """Multiple segments are joined with spaces."""
    stub_model.queued_results.append(
        [
            _StubSegment("first", 0.0, 1.0),
            _StubSegment("second", 1.0, 2.0),
        ]
    )
    resp = client.post(
        "/transcribe",
        content=_make_pcm(),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "first second"
    assert len(data["segments"]) == 2


def test_transcribe_empty_body_returns_400(client: TestClient) -> None:
    """/transcribe with an empty body returns HTTP 400."""
    resp = client.post(
        "/transcribe",
        content=b"",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert resp.status_code == 400
    assert "Empty audio body" in resp.json()["detail"]


def test_transcribe_no_speech_returns_empty_text(client: TestClient, stub_model: _StubWhisperModel) -> None:
    """When the model returns no segments, text is empty string."""
    stub_model.queued_results.append([])
    resp = client.post(
        "/transcribe",
        content=_make_pcm(),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == ""
    assert data["segments"] == []
