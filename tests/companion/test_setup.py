"""Tests for private background-assistant setup."""

import sys
import json
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import reachy_mini_conversation_app.companion.setup as companion_setup_module
from reachy_mini_conversation_app.companion.setup import (
    CompanionSetup,
    CompanionSetupError,
    provision_companion,
)
from reachy_mini_conversation_app.companion.client import CompanionClientError
from reachy_mini_conversation_app.companion.settings import (
    CompanionSettings,
    read_companion_settings,
    write_companion_settings,
)


HF_TOKEN = "hf_test_credential"
API_TOKEN = "a" * 32
API_URL = "https://alice-smolagents-assistant-reachy-mini.hf.space"
ORG_API_URL = "https://pollen-robotics-smolagents-assistant-reachy-mini.hf.space"
PROVISIONER_OUTPUT = json.dumps(
    {
        "space_id": "alice/smolagents-assistant-reachy-mini",
        "bucket_id": "alice/smolagents-assistant-reachy-mini-data",
        "api_url": API_URL,
    }
).encode()


@pytest.mark.asyncio
async def test_provisioner_receives_credentials_only_through_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provisioning scrubs credentials from the child environment."""
    monkeypatch.setenv("HF_TOKEN", "environment-hf-token")
    monkeypatch.setenv("SMOL_ASSISTANT_API_TOKEN", "environment-api-token")
    process = MagicMock(returncode=0)
    process.communicate = AsyncMock(return_value=(PROVISIONER_OUTPUT, b""))
    create_process = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)

    provisioned = await provision_companion(HF_TOKEN, API_TOKEN, "alice")

    assert provisioned == API_URL
    expected_command = (
        sys.executable,
        "-I",
        "-m",
        "reachy_mini_conversation_app.companion.provisioner",
    )
    assert create_process.await_args.args == expected_command
    child_environment = create_process.await_args.kwargs["env"]
    assert "HF_TOKEN" not in child_environment
    assert "SMOL_ASSISTANT_API_TOKEN" not in child_environment
    request = json.loads(process.communicate.await_args.args[0])
    assert request == {
        "hf_token": HF_TOKEN,
        "api_token": API_TOKEN,
        "namespace": "alice",
    }


@pytest.mark.asyncio
async def test_provisioner_rejects_another_space_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provisioner metadata cannot redirect either credential to another Space."""
    output = json.dumps(
        {
            "space_id": "alice/smolagents-assistant-reachy-mini",
            "bucket_id": "alice/smolagents-assistant-reachy-mini-data",
            "api_url": "https://mallory-smolagents-assistant-reachy-mini.hf.space",
        }
    ).encode()
    process = MagicMock(returncode=0)
    process.communicate = AsyncMock(return_value=(output, b""))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))

    with pytest.raises(CompanionSetupError, match="unexpected endpoint"):
        await provision_companion(HF_TOKEN, API_TOKEN, "alice")


@pytest.mark.asyncio
async def test_provisioner_reports_safe_hugging_face_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A controlled child error reaches the Settings view without credentials."""
    detail = "The selected namespace cannot create a private Docker Space."
    process = MagicMock(returncode=1)
    process.communicate = AsyncMock(return_value=(b"", f"ERROR: {detail}\n".encode()))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))

    with pytest.raises(CompanionSetupError, match=detail):
        await provision_companion(HF_TOKEN, API_TOKEN, "pollen-robotics")


@pytest.mark.asyncio
async def test_absent_assistant_is_created_verified_and_saved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An absent assistant is provisioned and persisted only after verification."""
    provision = AsyncMock(return_value=ORG_API_URL)
    client = MagicMock()
    client.list_tasks = AsyncMock(return_value=())
    client.close = AsyncMock()
    client_class = MagicMock(return_value=client)
    monkeypatch.setattr(companion_setup_module, "provision_companion", provision)
    monkeypatch.setattr(companion_setup_module, "CompanionClient", client_class)
    setup = CompanionSetup(tmp_path)

    setup.start(HF_TOKEN, "pollen-robotics")
    for _ in range(10):
        await asyncio.sleep(0)
        if setup.status(configured=False)["state"] == "restart_required":
            break

    provision_call = provision.await_args
    assert provision_call is not None
    generated_token = provision_call.args[1]
    provision.assert_awaited_once_with(HF_TOKEN, generated_token, "pollen-robotics")
    assert read_companion_settings(tmp_path) == CompanionSettings(
        enabled=True,
        api_url=ORG_API_URL,
        api_token=generated_token,
    )
    client_class.assert_called_once_with(ORG_API_URL, generated_token, HF_TOKEN)
    client.list_tasks.assert_awaited_once_with()
    client.close.assert_awaited_once_with()
    status = setup.status(configured=False)
    assert status["state"] == "restart_required"
    assert status["namespace"] == "pollen-robotics"
    assert generated_token not in str(status)


@pytest.mark.asyncio
async def test_failed_reconnection_does_not_overwrite_saved_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed setup retry leaves protected settings unchanged."""
    saved_settings = CompanionSettings(enabled=True, api_url=API_URL, api_token=API_TOKEN)
    write_companion_settings(tmp_path, saved_settings)
    provision = AsyncMock(return_value=ORG_API_URL)
    client = MagicMock()
    client.list_tasks = AsyncMock(side_effect=CompanionClientError("Unavailable."))
    client.close = AsyncMock()
    monkeypatch.setattr(companion_setup_module, "provision_companion", provision)
    monkeypatch.setattr(companion_setup_module, "CompanionClient", MagicMock(return_value=client))
    monkeypatch.setattr(companion_setup_module, "VERIFICATION_RETRY_SECONDS", 0)
    setup = CompanionSetup(tmp_path)

    setup.start(HF_TOKEN, "pollen-robotics")
    for _ in range(20):
        await asyncio.sleep(0)
        if setup.status(configured=False)["state"] == "failed":
            break

    provision_call = provision.await_args
    assert provision_call is not None
    replacement_token = provision_call.args[1]
    provision.assert_awaited_once_with(HF_TOKEN, replacement_token, "pollen-robotics")
    assert replacement_token != API_TOKEN
    assert read_companion_settings(tmp_path) == saved_settings
    assert setup.status(configured=False) == {
        "state": "failed",
        "message": "Setup could not finish safely. Check the selected Hugging Face namespace and try again.",
    }
