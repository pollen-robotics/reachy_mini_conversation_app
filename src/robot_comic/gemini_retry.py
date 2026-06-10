"""Retry helpers for Gemini API calls.

Gemini's free tier is limited to ~10 requests/day; the app *will* hit 429
RESOURCE_EXHAUSTED. The helpers here parse the error's Retry-After hint
(either the HTTP header or the embedded RetryInfo) and produce a backoff
delay with jitter, capped to keep the UI responsive.

The functions are intentionally exception-tolerant: any unrecognised error
shape yields ``None`` for the retry-after and ``"unknown"`` for the quota,
so callers can still fall through to plain exponential backoff.
"""

from __future__ import annotations
import re
import random
import logging
from typing import Any
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


logger = logging.getLogger(__name__)


# Cap the backoff at one minute — beyond that we'd rather fail loud than
# wedge a turn waiting for daily quota to roll over.
MAX_RETRY_AFTER_S: float = 60.0


_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*s\s*$")


def _parse_retry_delay_string(value: str) -> float | None:
    """Parse a protobuf Duration string like '23s' or '1.5s' to seconds."""
    match = _DURATION_RE.match(value)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_http_retry_after(value: str) -> float | None:
    """Parse an HTTP Retry-After header value (seconds or HTTP-date) to seconds."""
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _extract_error_details(exc: Any) -> list[Any]:
    """Pull the ``error.details`` list out of a google-genai APIError, if any."""
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        err = details.get("error")
        if isinstance(err, dict):
            inner = err.get("details")
            if isinstance(inner, list):
                return inner
    return []


def extract_retry_after_seconds(exc: BaseException) -> float | None:
    """Return the suggested retry-after delay in seconds, or None.

    Order of precedence:
    1. ``Retry-After`` HTTP header on the response object.
    2. ``google.rpc.RetryInfo`` entry inside ``error.details``.
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            raw = headers.get("Retry-After") or headers.get("retry-after")
        except Exception:
            raw = None
        if isinstance(raw, str):
            seconds = _parse_http_retry_after(raw)
            if seconds is not None:
                return seconds

    for entry in _extract_error_details(exc):
        if not isinstance(entry, dict):
            continue
        type_url = str(entry.get("@type", ""))
        if not type_url.endswith("RetryInfo"):
            continue
        delay = entry.get("retryDelay")
        if isinstance(delay, str):
            seconds = _parse_retry_delay_string(delay)
            if seconds is not None:
                return seconds
    return None


def describe_quota_failure(exc: BaseException) -> str:
    """Best-effort description of which quota was hit (or 'unknown')."""
    for entry in _extract_error_details(exc):
        if not isinstance(entry, dict):
            continue
        type_url = str(entry.get("@type", ""))
        if not type_url.endswith("QuotaFailure"):
            continue
        violations = entry.get("violations") or []
        if isinstance(violations, list) and violations:
            first = violations[0]
            if isinstance(first, dict):
                metric = first.get("quotaMetric") or first.get("quotaId")
                if isinstance(metric, str) and metric:
                    return metric
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message:
        return message
    return "unknown"


def is_session_rotation_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is the Live API's scheduled session end (GoAway/1008).

    Gemini Live caps session duration. At the cap the server sends a GoAway
    notice and then closes the websocket with code 1008; google-genai surfaces
    that as an APIError out of ``session.receive()``. This is a scheduled
    rotation, not a failure — callers should reconnect, not crash.
    """
    if getattr(exc, "code", None) == 1008:
        return True
    return "goaway" in str(exc).lower()


def is_rate_limit_error(exc: BaseException) -> bool:
    """Return True if ``exc`` represents a 429 / RESOURCE_EXHAUSTED response."""
    code = getattr(exc, "code", None)
    if code == 429:
        return True
    status = getattr(exc, "status", None)
    if isinstance(status, str) and status.upper() == "RESOURCE_EXHAUSTED":
        return True
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def compute_backoff(
    attempt: int,
    base_delay: float,
    retry_after_s: float | None,
    *,
    cap_s: float = MAX_RETRY_AFTER_S,
    jitter_s: float = 0.5,
) -> float:
    """Return the next sleep duration, honouring server retry-after when present.

    - ``attempt`` is 0-indexed (0 => first retry).
    - When ``retry_after_s`` is set, sleep at least that long (clamped to ``cap_s``).
    - Otherwise, exponential backoff: ``base_delay * 2**attempt``.
    - A small uniform jitter avoids client-side thundering herds.
    """
    if retry_after_s is not None:
        delay = min(max(retry_after_s, 0.0), cap_s)
    else:
        delay = min(base_delay * (2 ** max(attempt, 0)), cap_s)
    return delay + random.uniform(0.0, jitter_s)
