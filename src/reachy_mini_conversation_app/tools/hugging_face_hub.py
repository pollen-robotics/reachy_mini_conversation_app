import re
import asyncio
import logging
from typing import Any
from datetime import datetime, timezone, timedelta
from collections.abc import Callable

import httpx
from huggingface_hub import HfApi, RepoCard, ModelInfo, SpaceInfo, DatasetInfo
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
from huggingface_hub.hf_api import PaperInfo

from reachy_mini_conversation_app.tool_spaces import current_hf_token
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

_ACTIONS = ("search", "trending", "details", "mine", "papers")
_KINDS = ("model", "dataset", "space")
# Spoken aloud: a handful of results is all a listener can follow.
_RESULT_LIMIT = 5
_README_EXCERPT_CHARS = 700
_README_MARKUP = re.compile(r"!\[[^\]]*\]\([^)]*\)|<[^>]+>")
# Covers a weekend plus a holiday.
_DAILY_PAPERS_LOOKBACK_DAYS = 4

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


def _readme_excerpt(repo_id: str, kind: str, token: str | None) -> str | None:
    """Return the start of a repo's README as plain-ish text, or None when it has none."""
    try:
        text = RepoCard.load(repo_id, repo_type=kind, token=token).text
    except EntryNotFoundError:
        return None
    # Images and HTML tags only burn the excerpt budget; markdown links read fine.
    words = _README_MARKUP.sub(" ", text).split()
    return " ".join(words)[:_README_EXCERPT_CHARS] or None


def _papers(api: HfApi, query: str) -> list[dict[str, Any]]:
    """List the latest daily papers page, most upvoted first, or papers matching a query."""
    papers: list[PaperInfo] = []
    if query:
        papers = list(api.list_papers(query=query, limit=_RESULT_LIMIT))
    else:
        # The daily page stays empty on weekends and early in the UTC day.
        today = datetime.now(timezone.utc).date()
        for days_back in range(_DAILY_PAPERS_LOOKBACK_DAYS):
            day = (today - timedelta(days=days_back)).isoformat()
            papers = list(api.list_daily_papers(date=day, limit=_RESULT_LIMIT))
            if papers:
                break
    # The abstract's first sentence; a full abstract per paper is too long to speak.
    return [
        {
            "id": paper.id,
            "title": paper.title,
            "upvotes": paper.upvotes,
            "summary": " ".join((paper.summary or "").split()).split(". ")[0],
        }
        for paper in papers
    ]


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
        details = _details(repo_info[kind](query))
        readme = _readme_excerpt(query, kind, token)
        return {**details, "readme": readme} if readme else details
    if action == "papers":
        return {"results": _papers(api, query)}
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
        "(owner/name, as returned by search) with the start of its README, and for a Space whether it is "
        "running, 'mine' lists the signed-in user's own repos, private ones included, 'papers' lists today's "
        "trending research papers, or papers matching a query."
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
                "description": (
                    "Search words for 'search' (optional for 'papers'), or the full repo id (owner/name) for 'details'."
                ),
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
