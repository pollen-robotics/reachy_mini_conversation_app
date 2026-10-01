import json
import urllib.error
import urllib.request
from io import BytesIO
from types import SimpleNamespace
from typing import Any
from http.client import HTTPMessage

import pytest

from reachy_mini_conversation_app import daemon_api


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def read(self) -> bytes:
        return self._body


def _robot() -> Any:
    return SimpleNamespace(client=SimpleNamespace(host="192.168.1.42", port=8000))


def test_get_request_has_no_body_and_decodes_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GET should carry no payload and return the decoded body."""
    captured = {}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["data"] = request.data
        captured["timeout"] = timeout
        return _FakeResponse(b'{"volume": 42}')

    monkeypatch.setattr(daemon_api.urllib.request, "urlopen", fake_urlopen)

    assert daemon_api.daemon_request(_robot(), "/api/volume/current") == {"volume": 42}
    assert captured["url"] == "http://192.168.1.42:8000/api/volume/current"
    assert captured["method"] == "GET"
    assert captured["data"] is None
    assert captured["timeout"] == daemon_api.DEFAULT_TIMEOUT_S


def test_post_sends_json_body_and_content_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """FastAPI rejects a JSON body without the header, so both must be set."""
    captured = {}

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        captured["method"] = request.get_method()
        captured["data"] = request.data
        captured["content_type"] = request.get_header("Content-type")
        return _FakeResponse(b'{"volume": 70}')

    monkeypatch.setattr(daemon_api.urllib.request, "urlopen", fake_urlopen)

    result = daemon_api.daemon_request(_robot(), "/api/volume/set", method="POST", payload={"volume": 70})

    assert result == {"volume": 70}
    assert captured["method"] == "POST"
    assert json.loads(captured["data"]) == {"volume": 70}
    assert captured["content_type"] == "application/json"


def test_none_body_decodes_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Routes returning None, such as stop-current-app, serialize to the JSON null."""
    monkeypatch.setattr(daemon_api.urllib.request, "urlopen", lambda request, timeout: _FakeResponse(b"null"))

    assert daemon_api.daemon_request(_robot(), "/api/apps/stop-current-app", method="POST") is None


def test_http_error_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """An HTTP status error must surface as DaemonApiError, with its code."""

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        raise urllib.error.HTTPError(request.full_url, 503, "Service Unavailable", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(daemon_api.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(daemon_api.DaemonApiError, match="503"):
        daemon_api.daemon_request(_robot(), "/api/volume/current")


def test_http_error_preserves_daemon_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers can distinguish daemon errors sharing an HTTP status."""

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> None:
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "Bad Request",
            HTTPMessage(),
            BytesIO(b'{"detail":"Stop cleanup failed"}'),
        )

    monkeypatch.setattr(daemon_api.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(daemon_api.DaemonApiError, match="Stop cleanup failed") as raised:
        daemon_api.daemon_request(_robot(), "/api/apps/stop-current-app", method="POST")

    assert raised.value.status_code == 400
    assert raised.value.detail == "Stop cleanup failed"


@pytest.mark.parametrize("body", [b"<html>Unavailable</html>", b"\xff", b"null", b'{"detail":42}'])
def test_http_error_without_string_detail_remains_wrapped(monkeypatch: pytest.MonkeyPatch, body: bytes) -> None:
    """Unusable error bodies retain the HTTP failure instead of escaping."""

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> None:
        raise urllib.error.HTTPError(request.full_url, 503, "Service Unavailable", HTTPMessage(), BytesIO(body))

    monkeypatch.setattr(daemon_api.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(daemon_api.DaemonApiError, match="503") as raised:
        daemon_api.daemon_request(_robot(), "/api/volume/current")

    assert raised.value.status_code == 503
    assert raised.value.detail is None


def test_transport_error_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreachable daemon must not raise a bare URLError at the caller."""

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(daemon_api.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(daemon_api.DaemonApiError):
        daemon_api.daemon_request(_robot(), "/api/volume/current")


def test_invalid_json_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-JSON body must not escape as a JSONDecodeError."""
    monkeypatch.setattr(
        daemon_api.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeResponse(b"<html>nope</html>"),
    )

    with pytest.raises(daemon_api.DaemonApiError, match="invalid JSON"):
        daemon_api.daemon_request(_robot(), "/api/volume/current")
