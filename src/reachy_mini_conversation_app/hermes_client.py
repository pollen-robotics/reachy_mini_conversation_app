"""HTTP client for delegating conversation tasks to a Hermes API server."""

from __future__ import annotations
import json
import asyncio
import logging
import urllib.error
import urllib.request
from typing import Any, Callable
from dataclasses import field, asdict, dataclass


logger = logging.getLogger(__name__)

Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, bytes]]


@dataclass(frozen=True)
class HermesDelegationResult:
    """Normalized result returned by Hermes delegation calls."""

    status: str
    answer: str
    details: dict[str, Any] = field(default_factory=dict)
    actions_taken: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    task_id: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


class HermesDelegationClient:
    """Small async wrapper around Hermes' `/v1/responses` HTTP API."""

    def __init__(
        self,
        *,
        base_url: str,
        api_token: str,
        timeout_seconds: float = 45.0,
        model: str = "hermes-agent",
        session_name: str = "reachy-mini",
        transport: Transport | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            base_url: Base Hermes API URL, for example `http://100.79.161.14:8080`.
            api_token: Bearer token for Hermes API auth. Never logged.
            timeout_seconds: Interactive request timeout.
            model: Hermes API model label to send in `/v1/responses` payloads.
            session_name: Source/session label visible to Hermes.
            transport: Optional test transport; defaults to stdlib urllib.

        """
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.timeout_seconds = timeout_seconds
        self.model = model
        self.session_name = session_name
        self._transport = transport or self._urllib_transport

    @property
    def configured(self) -> bool:
        """Return whether the client has the minimum settings needed to call Hermes."""
        return bool(self.base_url and self.api_token)

    async def delegate(
        self,
        *,
        task: str,
        why_needed: str | None = None,
        response_style: str = "brief",
        allowed_domains: list[str] | None = None,
        context: dict[str, Any] | None = None,
        urgency: str = "interactive",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Delegate a task to Hermes and return a normalized JSON dict."""
        task = task.strip()
        if not task:
            return HermesDelegationResult(
                status="error",
                answer="I need a task before I can ask Hermes for help.",
                error="empty_task",
            ).to_dict()

        if not self.configured:
            return HermesDelegationResult(
                status="unavailable",
                answer="Hermes delegation is not configured on this Reachy Mini.",
                error="missing_base_url_or_token",
            ).to_dict()

        payload = self._build_payload(
            task=task,
            why_needed=why_needed,
            response_style=response_style,
            allowed_domains=allowed_domains,
            context=context or {},
            urgency=urgency,
            session_id=session_id,
        )
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "reachy-mini-conversation-app/hermes-delegation",
        }
        if session_id:
            headers["X-Reachy-Session-Id"] = session_id

        url = f"{self.base_url}/v1/responses"
        logger.info("Delegating task to Hermes API at %s/v1/responses", self.base_url)
        try:
            status_code, response_body = await asyncio.wait_for(
                asyncio.to_thread(self._transport, url, headers, body, self.timeout_seconds),
                timeout=self.timeout_seconds,
            )
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("Hermes delegation timed out after %.1fs", self.timeout_seconds)
            return HermesDelegationResult(
                status="timeout",
                answer="Hermes did not answer before the timeout.",
                error="timeout",
            ).to_dict()
        except urllib.error.URLError as exc:
            logger.warning(
                "Hermes delegation unavailable: %s", exc.reason if hasattr(exc, "reason") else type(exc).__name__
            )
            return HermesDelegationResult(
                status="unavailable",
                answer="Hermes is unreachable right now.",
                error="network_unavailable",
            ).to_dict()
        except Exception as exc:
            logger.warning("Hermes delegation failed before receiving a response: %s", type(exc).__name__)
            return HermesDelegationResult(
                status="unavailable",
                answer="Hermes is unavailable right now.",
                error=type(exc).__name__,
            ).to_dict()

        if status_code >= 500:
            logger.warning("Hermes API returned HTTP %s", status_code)
            return HermesDelegationResult(
                status="unavailable",
                answer="Hermes returned a temporary server error.",
                details={"http_status": status_code},
                error="server_error",
            ).to_dict()

        if status_code >= 400:
            logger.warning("Hermes API rejected the request with HTTP %s", status_code)
            return HermesDelegationResult(
                status="error",
                answer="Hermes rejected the delegation request.",
                details={"http_status": status_code},
                error="http_error",
            ).to_dict()

        try:
            data = json.loads(response_body.decode("utf-8")) if response_body else {}
        except json.JSONDecodeError:
            logger.warning("Hermes API returned non-JSON response")
            return HermesDelegationResult(
                status="error",
                answer="Hermes returned a response I could not understand.",
                error="invalid_json",
            ).to_dict()

        return self._normalize_response(data).to_dict()

    def _build_payload(
        self,
        *,
        task: str,
        why_needed: str | None,
        response_style: str,
        allowed_domains: list[str] | None,
        context: dict[str, Any],
        urgency: str,
        session_id: str | None,
    ) -> dict[str, Any]:
        """Build an OpenAI Responses-compatible Hermes request payload."""
        delegation_envelope = {
            "source": "reachy_mini_conversation_app",
            "session_name": self.session_name,
            "session_id": session_id,
            "task": task,
            "why_needed": why_needed,
            "response_style": response_style,
            "allowed_domains": allowed_domains or [],
            "urgency": urgency,
            "context": context,
            "tool_policy": {
                "preferred_toolset": "hermes-api-server",
                "home_assistant": "deferred/unconfigured unless Hermes has HA credentials",
            },
        }
        prompt = (
            "You are Hermes, helping Reachy Mini's live conversation app. "
            "Complete the delegated task using your available web, Codex, MCP, or long-running tools when helpful. "
            "Home Assistant-specific work is deferred unless Hermes is explicitly configured for it. "
            "Return a concise answer suitable for Reachy to speak aloud, with citations if you used current web information.\n\n"
            f"Delegation request JSON:\n{json.dumps(delegation_envelope, ensure_ascii=False)}"
        )
        return {
            "model": self.model,
            "input": prompt,
            "store": False,
        }

    def _normalize_response(self, data: Any) -> HermesDelegationResult:
        """Normalize known Hermes/OpenAI-compatible response shapes."""
        if not isinstance(data, dict):
            return HermesDelegationResult(
                status="error",
                answer="Hermes returned an unexpected response shape.",
                error="unexpected_response_shape",
            )

        answer = self._extract_answer(data)
        citations = self._extract_citations(data)
        hermes_status = data.get("status") if isinstance(data.get("status"), str) else None
        task_id = data.get("id") if isinstance(data.get("id"), str) else None

        if not answer:
            answer = "Hermes completed the request but did not return a text answer."

        raw_details = data.get("details")
        details: dict[str, Any] = raw_details if isinstance(raw_details, dict) else {}
        raw_actions_taken = data.get("actions_taken")
        actions_taken: list[Any] = raw_actions_taken if isinstance(raw_actions_taken, list) else []

        return HermesDelegationResult(
            status="ok" if hermes_status in {None, "completed", "ok"} else "error",
            answer=answer,
            details={
                **details,
                "hermes_status": hermes_status,
                "http_api": "/v1/responses",
                "model": data.get("model") if isinstance(data.get("model"), str) else self.model,
            },
            actions_taken=[action for action in actions_taken if isinstance(action, dict)],
            citations=citations,
            task_id=task_id,
            error=None if hermes_status in {None, "completed", "ok"} else str(hermes_status),
        )

    def _extract_answer(self, data: dict[str, Any]) -> str:
        """Extract human-readable text from common response fields."""
        for key in ("answer", "output_text", "final_response", "message", "content", "text"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        choices = data.get("choices")
        if isinstance(choices, list):
            texts: list[str] = []
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    texts.append(message["content"])
                elif isinstance(choice.get("text"), str):
                    texts.append(choice["text"])
            if texts:
                return "\n".join(text.strip() for text in texts if text.strip())

        output = data.get("output")
        if isinstance(output, list):
            texts = []
            for item in output:
                texts.extend(self._extract_text_parts(item))
            if texts:
                return "\n".join(text.strip() for text in texts if text.strip())

        return ""

    def _extract_text_parts(self, item: Any) -> list[str]:
        """Extract text fragments from a nested Responses API output item."""
        if isinstance(item, str):
            return [item]
        if not isinstance(item, dict):
            return []

        texts: list[str] = []
        for key in ("text", "output_text", "content"):
            value = item.get(key)
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(value, list):
                for nested in value:
                    texts.extend(self._extract_text_parts(nested))
        return texts

    def _extract_citations(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract lightweight citation metadata when present."""
        citations = data.get("citations")
        if isinstance(citations, list):
            return [citation for citation in citations if isinstance(citation, dict)]

        extracted: list[dict[str, Any]] = []
        output = data.get("output")
        if not isinstance(output, list):
            return extracted

        for item in output:
            for part in self._iter_content_parts(item):
                annotations = part.get("annotations")
                if not isinstance(annotations, list):
                    continue
                for annotation in annotations:
                    if isinstance(annotation, dict) and annotation.get("url"):
                        extracted.append(
                            {
                                "title": annotation.get("title") or annotation.get("url"),
                                "url": annotation.get("url"),
                            }
                        )
        return extracted

    def _iter_content_parts(self, item: Any) -> list[dict[str, Any]]:
        """Return nested dict content parts from a response item."""
        if not isinstance(item, dict):
            return []
        content = item.get("content")
        if isinstance(content, list):
            return [part for part in content if isinstance(part, dict)]
        return [item]

    def _urllib_transport(
        self, url: str, headers: dict[str, str], body: bytes, timeout_seconds: float
    ) -> tuple[int, bytes]:
        """POST JSON with urllib and return `(status_code, response_body)`.

        The Authorization header may contain a secret; never log it here.
        """
        request = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                return int(response.status), response.read()
        except urllib.error.HTTPError as exc:
            return int(exc.code), exc.read()
