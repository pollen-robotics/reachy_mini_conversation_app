"""Tests for durable companion-task coordination."""

import json
import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

import reachy_mini_conversation_app.companion.client as client_module
from reachy_mini_conversation_app.companion.client import (
    CompanionTask,
    CompanionClient,
    CompanionQuestion,
    CompanionTaskStatus,
)
from reachy_mini_conversation_app.companion.coordinator import CompanionTaskCoordinator


def _task(
    status: CompanionTaskStatus,
    *,
    version: int,
    task_id: str = "task-1",
    summary: str | None = None,
    question: CompanionQuestion | None = None,
    error: str | None = None,
    result_available: bool = False,
) -> CompanionTask:
    return CompanionTask(
        task_id=task_id,
        status=status,
        summary=summary,
        question=question,
        error=error,
        next_attempt_at=None,
        version=version,
        result_available=result_available,
        artifacts=(),
        created_at="2026-08-03T10:00:00Z",
        updated_at="2026-08-03T10:00:00Z",
    )


@pytest.mark.asyncio
async def test_start_is_authenticated_and_retries_with_the_same_idempotency_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lost creation response retries the same authenticated durable request."""
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            raise httpx.ReadError("Response lost.", request=request)
        return httpx.Response(
            202,
            json={
                "task_id": "task-1",
                "status": "queued",
                "summary": "Task queued.",
                "question": None,
                "error": None,
                "next_attempt_at": None,
                "version": 1,
                "result_available": False,
                "artifacts": [],
                "created_at": "2026-08-03T10:00:00Z",
                "updated_at": "2026-08-03T10:00:00Z",
            },
        )

    async_client = httpx.AsyncClient
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        client_module.httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=transport, **kwargs),
    )
    client = CompanionClient(
        "https://alice-smolagents-assistant-reachy-mini.hf.space",
        "a" * 32,
        "hf_test_credential",
    )
    coordinator = CompanionTaskCoordinator(client)

    try:
        task = await coordinator.start(" Plan something ")
    finally:
        await client.close()

    assert task.task_id == "task-1"
    assert len(requests) == 2
    assert [json.loads(request.content) for request in requests] == [
        {"request": "Plan something"},
        {"request": "Plan something"},
    ]
    assert requests[0].headers["Idempotency-Key"] == requests[1].headers["Idempotency-Key"]
    assert requests[0].headers["X-Smol-Assistant-Token"] == "a" * 32
    assert requests[0].headers["Authorization"] == "Bearer hf_test_credential"


@pytest.mark.asyncio
async def test_run_announces_question_completion_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tracked task lifecycle reaches the active conversation once per transition."""
    question = CompanionQuestion(
        question_id="question-1",
        text="Which area should the plan focus on?",
        options=("Latin Quarter", "Montmartre"),
    )
    client = AsyncMock()
    client.start.side_effect = [
        _task(CompanionTaskStatus.QUEUED, version=1),
        _task(CompanionTaskStatus.QUEUED, version=1, task_id="task-2"),
    ]
    client.answer.return_value = _task(CompanionTaskStatus.RUNNING, version=3)
    client.get.side_effect = [
        _task(CompanionTaskStatus.INPUT_REQUIRED, version=2, question=question),
        _task(CompanionTaskStatus.FAILED, version=2, task_id="task-2", error="Provider unavailable."),
        _task(
            CompanionTaskStatus.COMPLETED,
            version=4,
            summary="The Paris plan is ready.",
            result_available=True,
        ),
    ]
    coordinator = CompanionTaskCoordinator(client)
    await coordinator.start("Plan Paris")
    await coordinator.start("Research restaurants")
    deliver = AsyncMock(return_value=True)

    async def answer_then_stop(_delay: float) -> None:
        if not client.answer.await_count:
            await coordinator.answer("task-1", "question-1", "Latin Quarter")
            return
        raise asyncio.CancelledError

    monkeypatch.setattr("reachy_mini_conversation_app.companion.coordinator.asyncio.sleep", answer_then_stop)

    with pytest.raises(asyncio.CancelledError):
        await coordinator.run(deliver, lambda: True)

    notices = [call.args[0] for call in deliver.await_args_list]
    assert [notice.task_id for notice in notices] == ["task-1", "task-2", "task-1"]
    assert "Ask this question concisely" in notices[0].instruction
    assert "Which area should the plan focus on?" in notices[0].instruction
    assert 'task_id="task-1"' in notices[0].instruction
    assert 'question_id="question-1"' in notices[0].instruction
    assert "background task failed" in notices[1].instruction
    assert "background task finished" in notices[2].instruction
    client.answer.assert_awaited_once_with("task-1", "question-1", "Latin Quarter")


@pytest.mark.asyncio
async def test_poll_failure_does_not_stop_the_coordinator(monkeypatch: pytest.MonkeyPatch) -> None:
    """An optional polling failure cannot stop the conversation service."""
    client = AsyncMock()
    client.start.return_value = _task(CompanionTaskStatus.QUEUED, version=1)
    client.get.side_effect = [RuntimeError("unexpected failure"), _task(CompanionTaskStatus.CANCELLED, version=2)]
    coordinator = CompanionTaskCoordinator(client)
    await coordinator.start("Plan something")
    sleep_count = 0

    async def stop_after_two_cycles(_delay: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr("reachy_mini_conversation_app.companion.coordinator.asyncio.sleep", stop_after_two_cycles)

    with pytest.raises(asyncio.CancelledError):
        await coordinator.run(AsyncMock(return_value=True), lambda: True)

    assert client.get.await_count == 2
    client.close.assert_awaited_once()
