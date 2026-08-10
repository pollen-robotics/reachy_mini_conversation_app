"""Persist instance-wide background-assistant settings."""

import os
import json
import logging
import threading
from pathlib import Path
from tempfile import NamedTemporaryFile
from dataclasses import field, dataclass
from urllib.parse import urlsplit


logger = logging.getLogger(__name__)

COMPANION_SETTINGS_FILENAME = "companion_settings.json"
TERMINAL_EXTERNAL_CONTENT_DIRECTORY = Path("external_content")
MIN_COMPANION_API_TOKEN_CHARS = 32
MAX_COMPANION_API_TOKEN_CHARS = 4_096
_SETTINGS_LOCK = threading.RLock()


def normalize_companion_api_token(value: str) -> str:
    """Validate and return an assistant API token."""
    if (
        not MIN_COMPANION_API_TOKEN_CHARS <= len(value) <= MAX_COMPANION_API_TOKEN_CHARS
        or not value.isascii()
        or any(not 0x21 <= ord(character) <= 0x7E for character in value)
    ):
        raise ValueError("The assistant API token is invalid.")
    return value


def normalize_companion_api_url(value: str, *, hosted_only: bool) -> str:
    """Validate and normalize an assistant API origin."""
    normalized = value.strip().rstrip("/")
    try:
        parsed_url = urlsplit(normalized)
        port = parsed_url.port
    except ValueError as exc:
        raise ValueError("The assistant API URL is invalid.") from exc
    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.hostname
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.path
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise ValueError("The assistant API URL is invalid.")
    if hosted_only and (
        parsed_url.scheme != "https"
        or port is not None
        or not parsed_url.hostname.endswith(".hf.space")
        or parsed_url.netloc != parsed_url.hostname
    ):
        raise ValueError("Saved assistants must use a Hugging Face Space HTTPS origin.")
    return normalized


@dataclass(frozen=True, slots=True)
class CompanionSettings:
    """Hold the saved assistant preference and optional connection."""

    enabled: bool = True
    api_url: str | None = None
    api_token: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate and normalize a complete saved connection."""
        if (self.api_url is None) != (self.api_token is None):
            raise ValueError("The saved assistant connection is incomplete.")
        if self.api_url is None or self.api_token is None:
            return
        object.__setattr__(self, "api_url", normalize_companion_api_url(self.api_url, hosted_only=True))
        normalize_companion_api_token(self.api_token)


def get_companion_settings_path(instance_path: str | Path | None) -> Path:
    """Return the background-assistant settings path for the current mode."""
    if instance_path is not None:
        return Path(instance_path) / COMPANION_SETTINGS_FILENAME
    return TERMINAL_EXTERNAL_CONTENT_DIRECTORY / COMPANION_SETTINGS_FILENAME


def read_companion_settings(instance_path: str | Path | None) -> CompanionSettings:
    """Read saved assistant settings, failing closed when they are invalid."""
    with _SETTINGS_LOCK:
        settings_path = get_companion_settings_path(instance_path)
        if not settings_path.exists():
            return CompanionSettings()
        try:
            payload: object = json.loads(settings_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("The settings document is not an object.")
            if set(payload) - {"enabled", "api_url", "api_token"}:
                raise ValueError("The settings document contains unexpected fields.")
            enabled = payload.get("enabled")
            if not isinstance(enabled, bool):
                raise ValueError("The assistant preference is invalid.")
            api_url = payload.get("api_url")
            api_token = payload.get("api_token")
            if api_url is not None and not isinstance(api_url, str):
                raise ValueError("The saved assistant URL is invalid.")
            if api_token is not None and not isinstance(api_token, str):
                raise ValueError("The saved assistant token is invalid.")
            return CompanionSettings(enabled=enabled, api_url=api_url, api_token=api_token)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Ignoring invalid companion settings from %s: %s", settings_path, exc)
            return CompanionSettings(enabled=False)


def write_companion_settings(instance_path: str | Path | None, settings: CompanionSettings) -> Path:
    """Atomically persist protected instance-wide assistant settings."""
    with _SETTINGS_LOCK:
        settings_path = get_companion_settings_path(instance_path)
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {"enabled": settings.enabled}
        if settings.api_url is not None and settings.api_token is not None:
            payload["api_url"] = settings.api_url
            payload["api_token"] = settings.api_token

        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=settings_path.parent,
                prefix=f".{settings_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(payload, temporary_file, indent=2, sort_keys=True)
                temporary_file.write("\n")
            if os.name == "posix":
                temporary_path.chmod(0o600)
            temporary_path.replace(settings_path)
            if os.name == "posix":
                settings_path.chmod(0o600)
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning("Failed to remove temporary companion settings file %s: %s", temporary_path, exc)
        return settings_path
