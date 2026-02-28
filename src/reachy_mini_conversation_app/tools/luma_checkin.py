import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

import httpx

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedGuest:
    """Resolved Luma guest identity."""

    guest_api_id: str
    name: str | None = None
    email: str | None = None


class LumaCheckin(Tool):
    """Direct Luma check-in tool using public + internal APIs."""

    name = "luma_checkin"
    description = (
        "Check in a Luma event guest directly via API. "
        "Provide one of: checkin_url, qr_payload, guest_api_id, email, or name."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "event_api_id": {
                "type": "string",
                "description": "Luma event API ID (evt-...). Optional if embedded in checkin_url or configured in env.",
            },
            "checkin_url": {
                "type": "string",
                "description": "Full Luma check-in URL from QR (https://luma.com/check-in/evt-...?...).",
            },
            "qr_payload": {
                "type": "string",
                "description": "Raw QR payload when it contains Luma check-in URL or identifier params.",
            },
            "guest_api_id": {
                "type": "string",
                "description": "Luma guest API ID (gst-...).",
            },
            "email": {
                "type": "string",
                "description": "Guest email to resolve the attendee.",
            },
            "name": {
                "type": "string",
                "description": "Guest display name to resolve the attendee.",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Execute direct Luma check-in."""
        del deps  # Tool currently does not need robot dependencies.

        if not _env_bool("LUMA_DIRECT_CHECKIN_ENABLED", True):
            return {"error": "Luma direct check-in is disabled"}

        public_api_key = os.getenv("LUMA_API_KEY", "").strip()
        if not public_api_key:
            return {"error": "Missing LUMA_API_KEY"}

        session_cookie = _load_session_cookie()
        if not session_cookie:
            return {"error": "Missing LUMA session cookie (LUMA_SESSION_COOKIE or LUMA_SESSION_COOKIE_FILE)"}

        event_api_id = _resolve_event_api_id(kwargs)
        if not event_api_id:
            return {"error": "Could not resolve event_api_id"}

        timeout_seconds = _env_float("LUMA_DIRECT_TIMEOUT_SECONDS", 5.0)
        public_base_url = os.getenv("LUMA_PUBLIC_BASE_URL", "https://public-api.luma.com").rstrip("/")
        internal_base_url = os.getenv("LUMA_INTERNAL_BASE_URL", "https://api2.luma.com").rstrip("/")
        user_agent = os.getenv("LUMA_DIRECT_USER_AGENT", "ReachyMiniConversationApp-Luma/1.0")

        try:
            async with self._make_client(timeout=timeout_seconds) as client:
                guest = await _resolve_guest(
                    client=client,
                    kwargs=kwargs,
                    event_api_id=event_api_id,
                    public_api_key=public_api_key,
                    public_base_url=public_base_url,
                    user_agent=user_agent,
                )
                if isinstance(guest, dict):
                    return guest

                await _update_checkin(
                    client=client,
                    internal_base_url=internal_base_url,
                    event_api_id=event_api_id,
                    guest_api_id=guest.guest_api_id,
                    session_cookie=session_cookie,
                    user_agent=user_agent,
                )
                last_checked_in_at = await _verify_checkin(
                    client=client,
                    internal_base_url=internal_base_url,
                    event_api_id=event_api_id,
                    guest_api_id=guest.guest_api_id,
                    session_cookie=session_cookie,
                    user_agent=user_agent,
                )
        except httpx.TimeoutException:
            return {"error": "Luma check-in timed out"}
        except httpx.HTTPError as err:
            return {"error": f"Luma API request failed: {err}"}
        except Exception as err:  # noqa: BLE001
            logger.exception("Unexpected luma_checkin error")
            return {"error": f"Unexpected luma_checkin error: {type(err).__name__}: {err}"}

        if not last_checked_in_at:
            return {"error": "Check-in request succeeded, but last_checked_in_at verification failed"}

        return {
            "status": "checked_in",
            "verified": True,
            "event_api_id": event_api_id,
            "guest_api_id": guest.guest_api_id,
            "name": guest.name,
            "email": guest.email,
            "last_checked_in_at": last_checked_in_at,
        }

    def _make_client(self, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value <= 0:
        return default
    return value


def _load_session_cookie() -> str:
    inline_cookie = os.getenv("LUMA_SESSION_COOKIE", "").strip()
    if inline_cookie:
        return inline_cookie

    cookie_file_raw = os.getenv("LUMA_SESSION_COOKIE_FILE", "").strip()
    if not cookie_file_raw:
        return ""
    cookie_path = Path(cookie_file_raw).expanduser()
    if not cookie_path.exists():
        return ""
    try:
        return cookie_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _extract_qr_payload(kwargs: Dict[str, Any]) -> str | None:
    checkin_url = str(kwargs.get("checkin_url") or "").strip()
    if checkin_url:
        return checkin_url

    qr_payload = str(kwargs.get("qr_payload") or "").strip()
    if qr_payload:
        return qr_payload
    return None


def _resolve_event_api_id(kwargs: Dict[str, Any]) -> str | None:
    explicit = str(kwargs.get("event_api_id") or "").strip()
    if explicit:
        return explicit

    qr_payload = _extract_qr_payload(kwargs)
    if qr_payload:
        event_id = _event_id_from_luma_payload(qr_payload)
        if event_id:
            return event_id

    from_env = os.getenv("LUMA_ACTIVE_EVENT_API_ID", "").strip()
    if from_env:
        return from_env

    registry_raw = os.getenv(
        "LUMA_EVENTS_REGISTRY_PATH",
        "~/.openclaw/workspace/skills/luma-check-in/runtime/events_registry.json",
    ).strip()
    if not registry_raw:
        return None
    registry_path = Path(registry_raw).expanduser()
    if not registry_path.exists():
        return None

    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    active = payload.get("active_event_api_id")
    if isinstance(active, str) and active.strip():
        return active.strip()
    return None


def _event_id_from_luma_payload(payload: str) -> str | None:
    parsed = _safe_parse_url(payload)
    if parsed is None:
        return None
    parts = [part for part in (parsed.path or "").split("/") if part]
    if len(parts) >= 2 and parts[0] == "check-in" and parts[1].startswith("evt-"):
        return parts[1]
    params = parse_qs(parsed.query)
    for key in ("event_api_id", "event_id"):
        values = params.get(key)
        if values and values[0].strip():
            return values[0].strip()
    return None


def _identifier_from_qr_payload(payload: str) -> str | None:
    parsed = _safe_parse_url(payload)
    if parsed is None:
        return None
    params = parse_qs(parsed.query)
    for key in ("pk", "proxy_key", "id"):
        values = params.get(key)
        if values and values[0].strip():
            return values[0].strip()
    return None


def _safe_parse_url(text: str):
    try:
        parsed = urlparse(text.strip())
    except Exception:
        return None
    if parsed.scheme not in {"http", "https"}:
        return None
    if not parsed.netloc:
        return None
    return parsed


async def _resolve_guest(
    *,
    client: httpx.AsyncClient,
    kwargs: Dict[str, Any],
    event_api_id: str,
    public_api_key: str,
    public_base_url: str,
    user_agent: str,
) -> ResolvedGuest | Dict[str, Any]:
    qr_payload = _extract_qr_payload(kwargs)
    if qr_payload:
        identifier = _identifier_from_qr_payload(qr_payload)
        if identifier:
            guest = await _resolve_guest_by_identifier(
                client=client,
                event_api_id=event_api_id,
                identifier=identifier,
                public_api_key=public_api_key,
                public_base_url=public_base_url,
                user_agent=user_agent,
            )
            if guest is not None:
                return guest

    guest_api_id = str(kwargs.get("guest_api_id") or "").strip()
    if guest_api_id:
        guest = await _resolve_guest_by_identifier(
            client=client,
            event_api_id=event_api_id,
            identifier=guest_api_id,
            public_api_key=public_api_key,
            public_base_url=public_base_url,
            user_agent=user_agent,
        )
        if guest is not None:
            return guest

    email = str(kwargs.get("email") or "").strip().lower()
    if email:
        guests = await _list_guests(
            client=client,
            event_api_id=event_api_id,
            public_api_key=public_api_key,
            public_base_url=public_base_url,
            user_agent=user_agent,
        )
        matched = [guest for guest in guests if (guest.email or "").lower() == email]
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            return {"error": "Multiple guests matched this email. Provide qr_payload or guest_api_id."}
        return {"error": "Guest not found for this email"}

    name = str(kwargs.get("name") or "").strip()
    if name:
        guests = await _list_guests(
            client=client,
            event_api_id=event_api_id,
            public_api_key=public_api_key,
            public_base_url=public_base_url,
            user_agent=user_agent,
        )
        matched = [guest for guest in guests if (guest.name or "").strip().casefold() == name.casefold()]
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            return {"error": "Multiple guests matched this name. Provide email, qr_payload, or guest_api_id."}
        return {"error": "Guest not found for this name"}

    return {"error": "Could not resolve guest. Provide qr_payload/checkin_url, guest_api_id, email, or name."}


async def _resolve_guest_by_identifier(
    *,
    client: httpx.AsyncClient,
    event_api_id: str,
    identifier: str,
    public_api_key: str,
    public_base_url: str,
    user_agent: str,
) -> ResolvedGuest | None:
    response = await client.get(
        f"{public_base_url}/v1/event/get-guest",
        params={"event_id": event_api_id, "id": identifier},
        headers={"accept": "application/json", "x-luma-api-key": public_api_key, "user-agent": user_agent},
    )
    if response.status_code >= 400:
        return None
    payload = response.json() if response.content else {}
    guest = payload.get("guest") if isinstance(payload, dict) and isinstance(payload.get("guest"), dict) else payload
    if not isinstance(guest, dict):
        return None
    guest_api_id = _first_nonempty_str(guest, ("api_id", "id"))
    if not guest_api_id:
        return None
    return ResolvedGuest(
        guest_api_id=guest_api_id,
        name=_first_nonempty_str(guest, ("name", "user_name")),
        email=_first_nonempty_str(guest, ("email", "user_email")),
    )


async def _list_guests(
    *,
    client: httpx.AsyncClient,
    event_api_id: str,
    public_api_key: str,
    public_base_url: str,
    user_agent: str,
) -> list[ResolvedGuest]:
    response = await client.get(
        f"{public_base_url}/v1/event/get-guests",
        params={"event_id": event_api_id, "pagination_limit": 100},
        headers={"accept": "application/json", "x-luma-api-key": public_api_key, "user-agent": user_agent},
    )
    if response.status_code >= 400:
        return []
    payload = response.json() if response.content else {}
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return []

    guests: list[ResolvedGuest] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        nested = entry.get("guest") if isinstance(entry.get("guest"), dict) else entry
        if not isinstance(nested, dict):
            continue
        guest_api_id = _first_nonempty_str(nested, ("api_id", "id")) or _first_nonempty_str(entry, ("api_id",))
        if not guest_api_id:
            continue
        guests.append(
            ResolvedGuest(
                guest_api_id=guest_api_id,
                name=_first_nonempty_str(nested, ("name", "user_name")),
                email=_first_nonempty_str(nested, ("email", "user_email")),
            )
        )
    return guests


async def _update_checkin(
    *,
    client: httpx.AsyncClient,
    internal_base_url: str,
    event_api_id: str,
    guest_api_id: str,
    session_cookie: str,
    user_agent: str,
) -> None:
    response = await client.post(
        f"{internal_base_url}/event/admin/update-check-in",
        json={
            "event_api_id": event_api_id,
            "check_in_method": "guest-list",
            "check_in_status": "checked-in",
            "type": "guest",
            "rsvp_api_id": guest_api_id,
        },
        headers={
            "accept": "application/json",
            "content-type": "application/json",
            "cookie": session_cookie,
            "x-luma-client-type": "luma-web",
            "x-luma-web-url": f"https://luma.com/check-in/{event_api_id}",
            "user-agent": user_agent,
        },
    )
    response.raise_for_status()


async def _verify_checkin(
    *,
    client: httpx.AsyncClient,
    internal_base_url: str,
    event_api_id: str,
    guest_api_id: str,
    session_cookie: str,
    user_agent: str,
) -> str | None:
    response = await client.get(
        f"{internal_base_url}/event/admin/get-guest",
        params={"event_api_id": event_api_id, "guest_api_id": guest_api_id},
        headers={
            "accept": "application/json",
            "cookie": session_cookie,
            "x-luma-client-type": "luma-web",
            "x-luma-web-url": f"https://luma.com/check-in/{event_api_id}",
            "user-agent": user_agent,
        },
    )
    response.raise_for_status()
    payload = response.json() if response.content else {}
    guest = payload.get("guest") if isinstance(payload, dict) and isinstance(payload.get("guest"), dict) else payload
    if not isinstance(guest, dict):
        return None
    return _first_nonempty_str(guest, ("last_checked_in_at",))


def _first_nonempty_str(obj: Dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
