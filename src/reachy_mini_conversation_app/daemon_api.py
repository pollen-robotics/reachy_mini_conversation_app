"""Minimal client for the Reachy Mini daemon REST API.

The daemon runs on the same host as this app and ``robot.client`` already knows
where; only the endpoints the conversation tools need are wrapped here.
"""

import json
import urllib.error
import urllib.request
from typing import Any
from http.client import IncompleteRead

from reachy_mini import ReachyMini


DEFAULT_TIMEOUT_S = 2.0


class DaemonApiError(RuntimeError):
    """Raised when a daemon REST call fails."""

    def __init__(self, message: str, status_code: int | None = None, detail: str | None = None) -> None:
        """Store the message and, for HTTP errors, the status code and detail."""
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


def daemon_request(
    robot: ReachyMini,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Any:
    """Call a daemon REST endpoint and return the decoded JSON body.

    Raises:
        DaemonApiError: on transport failure, HTTP error or invalid JSON.

    """
    url = f"http://{robot.client.host}:{robot.client.port}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read()
    except urllib.error.HTTPError as e:
        message = f"{method} {path} failed: HTTP {e.code}"
        try:
            with e:
                error_body = e.read()
            error_response: object = json.loads(error_body) if error_body else None
        except (json.JSONDecodeError, UnicodeDecodeError, OSError, IncompleteRead) as body_error:
            raise DaemonApiError(f"{message}; could not read error detail: {body_error}", status_code=e.code) from e
        detail = error_response.get("detail") if isinstance(error_response, dict) else None
        if not isinstance(detail, str):
            detail = None
        if detail is not None:
            message = f"{message}: {detail}"
        raise DaemonApiError(message, status_code=e.code, detail=detail) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise DaemonApiError(f"{method} {path} failed: {e}") from e

    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise DaemonApiError(f"{method} {path} returned invalid JSON") from e
