import json
from enum import Enum

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError


class CompanionClientError(RuntimeError):
    """Base error for companion API operations."""


class CompanionUnavailableError(CompanionClientError):
    """Raised when the companion API cannot be reached."""


class CompanionApiError(CompanionClientError):
    """Raised when the companion API rejects a request."""


class CompanionTaskStatus(str, Enum):
    """State of a durable companion task."""

    QUEUED = "queued"
    RUNNING = "running"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CompanionQuestion(BaseModel):
    """Question blocking a companion task."""

    model_config = ConfigDict(frozen=True)

    question_id: str
    text: str
    options: tuple[str, ...]


class CompanionArtifact(BaseModel):
    """Artifact metadata returned for a completed task."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str
    media_type: str


class CompanionTask(BaseModel):
    """Task state returned by the standalone assistant."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    status: CompanionTaskStatus
    summary: str | None
    question: CompanionQuestion | None
    error: str | None
    next_attempt_at: str | None
    version: int
    result_available: bool
    artifacts: tuple[CompanionArtifact, ...]
    created_at: str
    updated_at: str

    @property
    def terminal(self) -> bool:
        """Return whether the task has finished."""
        return self.status in {
            CompanionTaskStatus.COMPLETED,
            CompanionTaskStatus.FAILED,
            CompanionTaskStatus.CANCELLED,
        }


class CompanionTaskList(BaseModel):
    """Newest-first task list returned by the standalone assistant."""

    model_config = ConfigDict(frozen=True)

    tasks: tuple[CompanionTask, ...]


def companion_task_to_tool_result(task: CompanionTask) -> dict[str, object]:
    """Return a task response suitable for the voice model."""
    return task.model_dump(
        mode="json",
        exclude={"created_at", "updated_at"},
    )


class CompanionClient:
    """Call one authenticated standalone companion task API."""

    def __init__(self, api_url: str, api_token: str, hf_token: str | None = None) -> None:
        """Initialize the companion API connection."""
        self._api_url = api_url.rstrip("/")
        headers = {"X-Smol-Assistant-Token": api_token}
        if hf_token is not None:
            headers["Authorization"] = f"Bearer {hf_token}"
        self._http = httpx.AsyncClient(
            headers=headers,
            timeout=10.0,
            trust_env=False,
        )

    async def start(self, request: str, idempotency_key: str) -> CompanionTask:
        """Create or replay one durable task."""
        return await self._request(
            "POST",
            "/v1/tasks",
            json_body={"request": request},
            headers={"Idempotency-Key": idempotency_key},
        )

    async def get(self, task_id: str) -> CompanionTask:
        """Return the latest task state."""
        return await self._request("GET", f"/v1/tasks/{task_id}")

    async def list_tasks(self) -> tuple[CompanionTask, ...]:
        """Return tasks from newest to oldest."""
        payload = await self._request_json("GET", "/v1/tasks")
        try:
            return CompanionTaskList.model_validate(payload).tasks
        except ValidationError as exc:
            raise CompanionClientError("The background assistant returned an invalid task list.") from exc

    async def answer(self, task_id: str, question_id: str, answer: str) -> CompanionTask:
        """Answer a pending task question."""
        return await self._request(
            "POST",
            f"/v1/tasks/{task_id}/answers/{question_id}",
            json_body={"answer": answer},
        )

    async def cancel(self, task_id: str) -> CompanionTask:
        """Cancel a non-terminal task."""
        return await self._request("POST", f"/v1/tasks/{task_id}/cancel")

    async def read_artifact(self, task_id: str, artifact_id: str) -> str:
        """Read one UTF-8 Markdown task artifact."""
        try:
            response = await self._http.get(f"{self._api_url}/v1/tasks/{task_id}/artifacts/{artifact_id}")
        except httpx.HTTPError as exc:
            raise CompanionUnavailableError("The background assistant is unavailable.") from exc
        if not response.is_success:
            raise CompanionApiError("The background assistant could not return the result.")
        media_type = response.headers.get("Content-Type", "").partition(";")[0].lower()
        if media_type != "text/markdown":
            raise CompanionClientError("The background assistant result is not a Markdown document.")
        try:
            return response.content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CompanionClientError("The background assistant result is not valid UTF-8.") from exc

    async def close(self) -> None:
        """Close the HTTP connection pool."""
        await self._http.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
    ) -> CompanionTask:
        payload = await self._request_json(method, path, json_body=json_body, headers=headers)
        try:
            return CompanionTask.model_validate(payload)
        except ValidationError as exc:
            raise CompanionClientError("The background assistant returned an invalid task.") from exc

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
    ) -> object:
        try:
            response = await self._http.request(
                method,
                f"{self._api_url}{path}",
                headers=headers,
                json=json_body,
            )
        except httpx.HTTPError as exc:
            raise CompanionUnavailableError("The background assistant is unavailable.") from exc

        try:
            payload: object = response.json()
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CompanionClientError("The background assistant returned invalid JSON.") from exc
        if response.is_success:
            return payload

        message = "The background assistant rejected the request."
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            error = payload["error"]
            if isinstance(error.get("message"), str):
                message = error["message"]
        raise CompanionApiError(message)
