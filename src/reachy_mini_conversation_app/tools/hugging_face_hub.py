import asyncio
import logging
from typing import Any
from collections.abc import Callable

import httpx
from huggingface_hub import HfApi, ModelInfo, SpaceInfo, DatasetInfo
from huggingface_hub.errors import RepositoryNotFoundError

from reachy_mini_conversation_app.tool_spaces import current_hf_token
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

_ACTIONS = ("search", "trending", "details", "mine")
_KINDS = ("model", "dataset", "space")
# Spoken aloud: a handful of results is all a listener can follow.
_RESULT_LIMIT = 5

RepoInfo = ModelInfo | DatasetInfo | SpaceInfo


def _summary(repo: RepoInfo) -> dict[str, Any]:
    """Keep the few fields worth saying out loud for one repo in a list."""
    summary = {
        "id": repo.id,
        "likes": repo.likes,
        "task": getattr(repo, "pipeline_tag", None),
        "private": repo.private or None,
    }
    return {key: value for key, value in summary.items() if value is not None}


def _details(repo: RepoInfo) -> dict[str, Any]:
    """Describe one repo, dropping fields that do not apply to its kind."""
    card = repo.card_data
    runtime = getattr(repo, "runtime", None)
    details = {
        "id": repo.id,
        "author": repo.author,
        "likes": repo.likes,
        "downloads": getattr(repo, "downloads", None),
        "task": getattr(repo, "pipeline_tag", None),
        "library": getattr(repo, "library_name", None),
        "sdk": getattr(repo, "sdk", None),
        "status": runtime.stage if runtime else None,
        "license": card.get("license") if card else None,
        "summary": card.get("short_description") if card else None,
        "last_modified": repo.last_modified.date().isoformat() if repo.last_modified else None,
        "private": repo.private,
        "gated": repo.gated or None,
    }
    return {key: value for key, value in details.items() if value is not None}


def _list(api: HfApi, kind: str, **filters: str | None) -> list[RepoInfo]:
    listing = {"model": api.list_models, "dataset": api.list_datasets, "space": api.list_spaces}[kind]
    return list(listing(limit=_RESULT_LIMIT, **filters))


def _run(action: str, kind: str, query: str) -> dict[str, Any]:
    """Run one Hub request; blocking, so it is called off the event loop."""
    token = current_hf_token()
    api = HfApi(token=token)

    if action == "details":
        repo_info: dict[str, Callable[[str], RepoInfo]] = {
            "model": api.model_info,
            "dataset": api.dataset_info,
            "space": api.space_info,
        }
        return _details(repo_info[kind](query))
    if action == "mine":
        if token is None:
            return {"error": "the robot is not signed in to Hugging Face"}
        username = api.whoami()["name"]
        repos = _list(api, kind, author=username, sort="last_modified")
        return {"username": username, "results": [_summary(repo) for repo in repos]}

    search = query if action == "search" else None
    repos = _list(api, kind, search=search, sort="trending_score")
    return {"results": [_summary(repo) for repo in repos]}


class HuggingFaceHub(Tool):
    """Browse the Hugging Face Hub with the robot's own sign-in."""

    name = "hugging_face_hub"
    description = (
        "Browse the Hugging Face Hub. Actions: 'search' finds models, datasets or Spaces matching a query, "
        "'trending' lists what is popular right now, 'details' describes one repo from its full id "
        "(owner/name, as returned by search), 'mine' lists the signed-in user's own repos, private ones included."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(_ACTIONS), "description": "What to do."},
            "kind": {
                "type": "string",
                "enum": list(_KINDS),
                "description": "Which kind of repo. Defaults to model.",
            },
            "query": {
                "type": "string",
                "description": "Search words for 'search', or the full repo id (owner/name) for 'details'.",
            },
        },
        "required": ["action"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Run the requested Hub action."""
        action = str(kwargs.get("action") or "")
        kind = str(kwargs.get("kind") or "model")
        query = str(kwargs.get("query") or "").strip()
        if action not in _ACTIONS:
            return {"error": f"unknown action {action!r}"}
        if kind not in _KINDS:
            return {"error": f"unknown kind {kind!r}"}
        if action in ("search", "details") and not query:
            return {"error": f"'{action}' needs a query"}

        logger.info("Tool call: hugging_face_hub action=%s kind=%s query=%r", action, kind, query)
        try:
            result = await asyncio.to_thread(_run, action, kind, query)
        except RepositoryNotFoundError:
            result = {"error": f"no {kind} named {query!r}; search for it to get its full id"}
        except httpx.HTTPError as e:
            logger.error("hugging_face_hub(%s) failed: %s", action, e)
            result = {"error": str(e)}

        return {"action": action, "kind": kind, **result}
