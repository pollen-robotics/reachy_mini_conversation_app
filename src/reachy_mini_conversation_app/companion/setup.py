"""Provision and activate a managed background assistant."""

import os
import sys
import json
import asyncio
import logging
import secrets
from enum import Enum
from pathlib import Path
from urllib.parse import quote, urlsplit

from reachy_mini_conversation_app.companion.client import CompanionClient, CompanionClientError
from reachy_mini_conversation_app.companion.settings import (
    CompanionSettings,
    read_companion_settings,
    write_companion_settings,
    normalize_companion_api_url,
)
from reachy_mini_conversation_app.companion.provisioner import (
    ASSISTANT_SPACE_NAME,
    ASSISTANT_BUCKET_NAME,
    DEFAULT_PROVISIONING_TIMEOUT,
)


logger = logging.getLogger(__name__)
PROVISIONER_MODULE = "reachy_mini_conversation_app.companion.provisioner"
MANAGED_API_HOST_SUFFIX = f"-{ASSISTANT_SPACE_NAME}.hf.space"
PROVISIONING_TIMEOUT_SECONDS = DEFAULT_PROVISIONING_TIMEOUT + 120.0
MAX_PROVISIONING_OUTPUT_BYTES = 8_192
MAX_PROVISIONING_ERROR_CHARS = 500
VERIFICATION_ATTEMPTS = 5
VERIFICATION_RETRY_SECONDS = 2.0
_CHILD_CREDENTIAL_ENVIRONMENT = {
    "HF_TOKEN",
    "HF_ENDPOINT",
    "SMOL_ASSISTANT_API_TOKEN",
}


class CompanionSetupError(RuntimeError):
    """Report a safe background-assistant setup failure."""


class CompanionSetupState(str, Enum):
    """Describe the current setup phase without exposing credentials."""

    IDLE = "idle"
    PROVISIONING = "provisioning"
    VERIFYING = "verifying"
    RESTART_REQUIRED = "restart_required"
    FAILED = "failed"
    READY = "ready"


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    try:
        await process.communicate()
    except (BrokenPipeError, ConnectionResetError):
        await process.wait()


async def _run_provisioner(request: dict[str, object]) -> dict[str, object]:
    environment = os.environ.copy()
    for name in _CHILD_CREDENTIAL_ENVIRONMENT:
        environment.pop(name, None)

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-m",
            PROVISIONER_MODULE,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
        )
    except (OSError, NotImplementedError) as exc:
        raise CompanionSetupError("The bundled assistant setup could not start.") from exc

    encoded_request = json.dumps(request, separators=(",", ":")).encode()
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(encoded_request),
            timeout=PROVISIONING_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        await _stop_process(process)
        raise CompanionSetupError("Hugging Face took too long to set up the assistant.") from exc
    except asyncio.CancelledError:
        await asyncio.shield(_stop_process(process))
        raise
    except OSError as exc:
        await _stop_process(process)
        raise CompanionSetupError("The bundled assistant setup stopped unexpectedly.") from exc

    if process.returncode != 0:
        try:
            error_lines = stderr.decode("utf-8").splitlines()
        except UnicodeDecodeError:
            error_lines = []
        credentials: set[str] = set()
        for key in ("hf_token", "api_token"):
            value = request.get(key)
            if isinstance(value, str):
                credentials.add(value)
        for line in reversed(error_lines):
            if line.startswith("ERROR: "):
                detail = line.removeprefix("ERROR: ").strip()
                if (
                    detail
                    and len(detail) <= MAX_PROVISIONING_ERROR_CHARS
                    and all(credential not in detail for credential in credentials)
                ):
                    raise CompanionSetupError(detail)
        raise CompanionSetupError("Hugging Face could not finish setting up the assistant.")
    if not stdout or len(stdout) > MAX_PROVISIONING_OUTPUT_BYTES:
        raise CompanionSetupError("The assistant setup returned an invalid response.")
    try:
        payload: object = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompanionSetupError("The assistant setup returned an invalid response.") from exc
    if not isinstance(payload, dict):
        raise CompanionSetupError("The assistant setup returned an invalid response.")
    return payload


def _validate_resource_ids(space_id: str, bucket_id: str, namespace: str) -> None:
    space_namespace, separator, space_name = space_id.partition("/")
    bucket_namespace, bucket_separator, bucket_name = bucket_id.partition("/")
    if (
        not separator
        or not bucket_separator
        or space_namespace != namespace
        or bucket_namespace != namespace
        or space_name != ASSISTANT_SPACE_NAME
        or bucket_name != ASSISTANT_BUCKET_NAME
    ):
        raise CompanionSetupError("The assistant setup returned unexpected resources.")


def _managed_resource_metadata(api_url: str) -> dict[str, str]:
    hostname = urlsplit(api_url).hostname or ""
    if not hostname.endswith(MANAGED_API_HOST_SUFFIX):
        return {}
    namespace = hostname[: -len(MANAGED_API_HOST_SUFFIX)]
    if (
        not namespace
        or not namespace.isascii()
        or namespace.startswith("-")
        or namespace.endswith("-")
        or any(not (character.isalnum() or character == "-") for character in namespace)
    ):
        return {}
    quoted_namespace = quote(namespace, safe="")
    return {
        "namespace": namespace,
        "space_url": f"https://huggingface.co/spaces/{quoted_namespace}/{ASSISTANT_SPACE_NAME}",
        "bucket_url": f"https://huggingface.co/buckets/{quoted_namespace}/{ASSISTANT_BUCKET_NAME}",
    }


async def provision_companion(
    hf_token: str,
    api_token: str,
    namespace: str,
) -> str:
    """Provision the selected namespace's canonical assistant."""
    payload = await _run_provisioner(
        {
            "hf_token": hf_token,
            "api_token": api_token,
            "namespace": namespace,
        }
    )
    if set(payload) != {"space_id", "bucket_id", "api_url"}:
        raise CompanionSetupError("The assistant setup returned an invalid response.")
    space_id = payload["space_id"]
    bucket_id = payload["bucket_id"]
    api_url = payload["api_url"]
    if not isinstance(space_id, str) or not isinstance(bucket_id, str) or not isinstance(api_url, str):
        raise CompanionSetupError("The assistant setup returned an invalid response.")
    _validate_resource_ids(space_id, bucket_id, namespace)
    try:
        normalized_api_url = normalize_companion_api_url(api_url, hosted_only=True)
    except ValueError as exc:
        raise CompanionSetupError("The assistant setup returned an invalid endpoint.") from exc
    expected_hostname = f"{namespace}-{ASSISTANT_SPACE_NAME}.hf.space".lower()
    if urlsplit(normalized_api_url).hostname != expected_hostname:
        raise CompanionSetupError("The assistant setup returned an unexpected endpoint.")
    return normalized_api_url


class CompanionSetup:
    """Own one in-process setup job and its non-secret UI state."""

    def __init__(self, instance_path: str | Path | None) -> None:
        """Initialize setup state from the protected saved connection."""
        self._instance_path = instance_path
        settings = read_companion_settings(instance_path)
        self._resource_metadata = _managed_resource_metadata(settings.api_url) if settings.api_url else {}
        if settings.api_url is None:
            self._state = CompanionSetupState.IDLE
            self._message = "Choose a Hugging Face account or organization for the private assistant."
        else:
            self._state = CompanionSetupState.FAILED
            self._message = "The saved assistant is not active. Check your Hugging Face sign-in and try setup again."
        self._task: asyncio.Task[None] | None = None

    def status(self, *, configured: bool) -> dict[str, str]:
        """Return the current non-secret setup state for the UI."""
        if configured:
            status = {
                "state": CompanionSetupState.READY.value,
                "message": "Assistant ready for every personality.",
            }
        else:
            status = {"state": self._state.value, "message": self._message}
        status.update(self._resource_metadata)
        return status

    def set_connection_available(self, available: bool) -> None:
        """Update UI state after validating the saved assistant connection."""
        if available:
            settings = read_companion_settings(self._instance_path)
            self._resource_metadata = _managed_resource_metadata(settings.api_url) if settings.api_url else {}
            return
        self._state = CompanionSetupState.FAILED
        self._message = "The saved assistant is unavailable. Choose a namespace to reconnect or replace it."
        self._resource_metadata = {}

    def start(self, hf_token: str, namespace: str) -> None:
        """Create or reconnect an assistant in the selected namespace."""
        if self._task is not None and not self._task.done():
            return
        if self._state == CompanionSetupState.RESTART_REQUIRED:
            raise CompanionSetupError("Restart the Conversation App to finish setup.")
        self._state = CompanionSetupState.PROVISIONING
        self._message = f"Preparing private assistant and storage in @{namespace}…"
        self._resource_metadata = {}
        self._task = asyncio.create_task(self._run(hf_token, namespace))

    async def _run(self, hf_token: str, namespace: str) -> None:
        try:
            api_token = secrets.token_urlsafe(32)
            api_url = await provision_companion(hf_token, api_token, namespace)
            self._state = CompanionSetupState.VERIFYING
            self._message = "Checking the private assistant connection…"
            client = CompanionClient(api_url, api_token, hf_token)
            try:
                for attempt in range(VERIFICATION_ATTEMPTS):
                    try:
                        await client.list_tasks()
                        break
                    except CompanionClientError:
                        if attempt == VERIFICATION_ATTEMPTS - 1:
                            raise
                        await asyncio.sleep(VERIFICATION_RETRY_SECONDS)
            finally:
                await client.close()
            write_companion_settings(
                self._instance_path,
                CompanionSettings(enabled=True, api_url=api_url, api_token=api_token),
            )
            self._resource_metadata = _managed_resource_metadata(api_url)
        except asyncio.CancelledError:
            raise
        except CompanionSetupError as exc:
            logger.warning("Background assistant setup failed: %s", exc)
            self._state = CompanionSetupState.FAILED
            self._message = str(exc)
        except (CompanionClientError, OSError, ValueError) as exc:
            logger.warning("Background assistant setup failed: %s", exc)
            self._state = CompanionSetupState.FAILED
            self._message = "Setup could not finish safely. Check the selected Hugging Face namespace and try again."
        except Exception:
            logger.exception("Background assistant setup failed unexpectedly")
            self._state = CompanionSetupState.FAILED
            self._message = "Setup could not finish safely. Check the selected Hugging Face namespace and try again."
        else:
            self._state = CompanionSetupState.RESTART_REQUIRED
            self._message = "Assistant set up. Restart the Conversation App to activate it."
        finally:
            self._task = None
