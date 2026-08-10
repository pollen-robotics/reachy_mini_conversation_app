"""Companion tasks and global settings exposed to the Conversation App UI."""

import os
import asyncio
import logging
from typing import Any, TypeVar
from pathlib import Path
from dataclasses import replace
from collections.abc import Callable, Coroutine

from huggingface_hub import get_token
from huggingface_hub.errors import HfHubHTTPError

from reachy_mini.io.jsonrpc import JsonRpcError
from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini_conversation_app.config import (
    SMOL_ASSISTANT_API_URL_ENV,
    SMOL_ASSISTANT_API_TOKEN_ENV,
    config,
)
from reachy_mini_conversation_app.tool_settings import RestartCallback, apply_tool_change
from reachy_mini_conversation_app.companion.setup import CompanionSetup, CompanionSetupError
from reachy_mini_conversation_app.companion.client import (
    CompanionTask,
    CompanionApiError,
    CompanionClientError,
    CompanionUnavailableError,
)
from reachy_mini_conversation_app.companion.settings import read_companion_settings, write_companion_settings
from reachy_mini_conversation_app.companion.coordinator import CompanionTaskCoordinator
from reachy_mini_conversation_app.companion.provisioner import (
    ProvisioningError,
    AssistantNamespace,
    list_assistant_namespaces,
)


ResultT = TypeVar("ResultT")
MAX_VISIBLE_TASKS = 20
COMPANION_RPC_TIMEOUT_SECONDS = 15.0
logger = logging.getLogger(__name__)


def _companion_error(
    reason: str,
    detail: str,
    *,
    code: int = -32000,
) -> JsonRpcError:
    return JsonRpcError(detail, reason=reason, code=code, data={"detail": detail})


async def _run_on_companion_loop(
    coroutine: Coroutine[Any, Any, ResultT],
    get_loop: Callable[[], asyncio.AbstractEventLoop | None],
) -> ResultT:
    loop = get_loop()
    if loop is None or not loop.is_running():
        coroutine.close()
        raise _companion_error("companion_starting", "The background assistant is still starting.")
    try:
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
    except RuntimeError as exc:
        coroutine.close()
        raise _companion_error(
            "companion_starting",
            "The background assistant is still starting.",
        ) from exc

    try:
        return await asyncio.wait_for(
            asyncio.wrap_future(future),
            timeout=COMPANION_RPC_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        future.cancel()
        raise _companion_error(
            "companion_timeout",
            "The background assistant took too long to respond.",
        ) from exc
    except CompanionUnavailableError as exc:
        raise _companion_error("companion_unavailable", str(exc)) from exc
    except CompanionApiError as exc:
        raise _companion_error("companion_request_failed", str(exc)) from exc
    except CompanionClientError as exc:
        raise _companion_error("companion_invalid_response", str(exc)) from exc


def register_companion_methods(
    rpc: JsonRpcServer,
    companion_tasks: CompanionTaskCoordinator | None,
    get_loop: Callable[[], asyncio.AbstractEventLoop | None],
    restart_conversation: RestartCallback,
    *,
    instance_path: str | Path | None,
) -> None:
    """Register companion task and global-settings methods."""
    setup = CompanionSetup(instance_path)
    companion_available = companion_tasks is not None
    setup_started = False
    namespace_cache: tuple[AssistantNamespace, ...] | None = None
    environment_override = SMOL_ASSISTANT_API_URL_ENV in os.environ or SMOL_ASSISTANT_API_TOKEN_ENV in os.environ

    def _config_payload() -> dict[str, object]:
        setup_status = setup.status(configured=companion_available)
        if environment_override:
            setup_status = {
                key: value
                for key, value in setup_status.items()
                if key not in {"namespace", "space_url", "bucket_url"}
            }
        return {
            "configured": companion_available,
            "enabled": companion_available and config.COMPANION_ENABLED,
            "setup": setup_status,
        }

    async def _load_tasks(*, retry_unavailable: bool = False) -> tuple[CompanionTask, ...] | None:
        nonlocal companion_available
        if companion_tasks is None or (not companion_available and not retry_unavailable):
            return None
        retry_connection = True
        while True:
            try:
                tasks = await _run_on_companion_loop(companion_tasks.list_tasks(), get_loop)
            except JsonRpcError as exc:
                if exc.reason == "companion_starting" or environment_override:
                    raise
                if retry_connection:
                    retry_connection = False
                    continue
                logger.warning("Saved background assistant is unavailable: %s", exc)
                companion_available = False
                config.COMPANION_CONFIGURED = False
                setup.set_connection_available(False)
                return None
            companion_available = True
            config.COMPANION_CONFIGURED = True
            setup.set_connection_available(True)
            return tasks

    async def _get_config(_params: dict[str, Any]) -> dict[str, object]:
        if not environment_override and not setup_started:
            await _load_tasks(retry_unavailable=True)
        return _config_payload()

    async def _available_namespaces() -> tuple[str, tuple[AssistantNamespace, ...]]:
        nonlocal namespace_cache
        hf_token = (config.HF_TOKEN or "").strip() or (get_token() or "").strip()
        if not hf_token:
            raise _companion_error(
                "companion_hf_login_required",
                "Sign in to Hugging Face on this device, then try again.",
            )
        if namespace_cache is None:
            try:
                namespaces = await asyncio.to_thread(list_assistant_namespaces, hf_token)
            except HfHubHTTPError as exc:
                logger.warning("Hugging Face namespace lookup failed: %s", exc)
                detail = (
                    "Hugging Face rejected the token. Sign in again and retry."
                    if exc.response.status_code == 401
                    else "Hugging Face could not list the available organizations."
                )
                raise _companion_error("companion_namespace_lookup_failed", detail) from exc
            except ProvisioningError as exc:
                raise _companion_error("companion_namespace_lookup_failed", str(exc)) from exc
            namespace_cache = namespaces
        return hf_token, namespace_cache

    async def _list_namespaces(_params: dict[str, Any]) -> dict[str, object]:
        if environment_override:
            raise _companion_error(
                "companion_setup_overridden",
                "Remove the developer assistant override before using automatic setup.",
            )
        _, namespaces = await _available_namespaces()
        return {"namespaces": [{"name": namespace.name, "kind": namespace.kind.value} for namespace in namespaces]}

    def _save_config(params: dict[str, Any]) -> dict[str, object]:
        enabled = params.get("enabled")
        if not isinstance(enabled, bool):
            raise _companion_error(
                "invalid_companion_setting",
                "Choose whether to use the background assistant.",
                code=-32602,
            )
        if enabled and not companion_available:
            raise _companion_error(
                "companion_not_configured",
                "Configure the background assistant before enabling it.",
            )
        try:
            settings = replace(read_companion_settings(instance_path), enabled=enabled)
            write_companion_settings(instance_path, settings)
        except OSError as exc:
            raise _companion_error(
                "companion_settings_save_failed",
                "Could not save the background-assistant setting.",
            ) from exc

        config.COMPANION_ENABLED = enabled
        apply_detail = apply_tool_change(
            instance_path,
            get_loop,
            restart_conversation,
            "companion_setting_changed",
        )
        return {
            **_config_payload(),
            "message": f"Background assistant {'enabled' if enabled else 'disabled'}. {apply_detail}",
        }

    async def _start_setup(params: dict[str, Any]) -> dict[str, object]:
        nonlocal setup_started
        if companion_available:
            return _config_payload()
        if environment_override:
            raise _companion_error(
                "companion_setup_overridden",
                "Remove the developer assistant override before using automatic setup.",
            )
        namespace = params.get("namespace")
        if set(params) != {"namespace"} or not isinstance(namespace, str) or not namespace:
            raise _companion_error(
                "invalid_companion_namespace",
                "Choose a Hugging Face account or organization.",
                code=-32602,
            )
        hf_token, available_namespaces = await _available_namespaces()
        if namespace not in {candidate.name for candidate in available_namespaces}:
            raise _companion_error(
                "invalid_companion_namespace",
                "The selected Hugging Face namespace is not available for assistant setup.",
                code=-32602,
            )
        try:
            setup.start(hf_token, namespace)
        except CompanionSetupError as exc:
            raise _companion_error("companion_setup_failed", str(exc)) from exc
        setup_started = True
        return _config_payload()

    async def _list_tasks(_params: dict[str, Any]) -> dict[str, object]:
        tasks = await _load_tasks()
        if tasks is None:
            return {**_config_payload(), "tasks": []}
        return {
            **_config_payload(),
            "tasks": [
                task.model_dump(
                    mode="json",
                    include={
                        "task_id",
                        "status",
                        "summary",
                        "question",
                        "error",
                        "created_at",
                        "updated_at",
                        "result_available",
                    },
                )
                for task in tasks[:MAX_VISIBLE_TASKS]
            ],
        }

    async def _read_result(params: dict[str, Any]) -> dict[str, object]:
        task_id = params.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise _companion_error(
                "invalid_companion_task",
                "Choose a completed task.",
                code=-32602,
            )
        if companion_tasks is None or not companion_available:
            raise _companion_error(
                "companion_not_configured",
                "Configure the background assistant to read its briefs.",
            )
        try:
            result = await _run_on_companion_loop(
                companion_tasks.result(task_id, max_chars=None),
                get_loop,
            )
        except ValueError as exc:
            raise _companion_error("companion_result_unavailable", str(exc)) from exc
        return {"markdown": result.markdown}

    async def _cancel_task(params: dict[str, Any]) -> dict[str, object]:
        task_id = params.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise _companion_error(
                "invalid_companion_task",
                "Choose a task to stop.",
                code=-32602,
            )
        if companion_tasks is None or not companion_available:
            raise _companion_error(
                "companion_not_configured",
                "Configure the background assistant to stop its tasks.",
            )
        task = await _run_on_companion_loop(companion_tasks.cancel(task_id.strip()), get_loop)
        return {"status": task.status.value}

    rpc.register("companion.config.get", _get_config)
    rpc.register("companion.config.save", _save_config)
    rpc.register("companion.setup.namespaces", _list_namespaces)
    rpc.register("companion.setup.start", _start_setup)
    rpc.register("companion.tasks.list", _list_tasks)
    rpc.register("companion.tasks.result", _read_result)
    rpc.register("companion.tasks.cancel", _cancel_task)
