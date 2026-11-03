from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from reachy_mini_conversation_app.tools import hugging_face_hub as hub_module
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


def _repo(repo_id: str, **fields: object) -> SimpleNamespace:
    return SimpleNamespace(**{"id": repo_id, "likes": 3, "private": False, **fields})


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace HfApi, recording the token it was built with in api.built_with."""
    api = MagicMock()

    def build(token: str | None) -> MagicMock:
        api.built_with = token
        return api

    monkeypatch.setattr(hub_module, "HfApi", build)
    monkeypatch.setattr(hub_module, "current_hf_token", lambda: "daemon-token")
    return api


@pytest.mark.asyncio
async def test_search_uses_the_daemon_sign_in_and_keeps_spoken_fields(api: MagicMock) -> None:
    """A search lists the right repo kind with the robot's token, reduced to a few fields."""
    api.list_spaces.return_value = [_repo("pollen-robotics/a"), _repo("someone/b", private=True)]

    result = await hub_module.HuggingFaceHub()(_deps(), action="search", kind="space", query="reachy")

    assert api.built_with == "daemon-token"
    api.list_spaces.assert_called_once_with(limit=5, search="reachy", sort="trending_score")
    assert result == {
        "action": "search",
        "kind": "space",
        "results": [{"id": "pollen-robotics/a", "likes": 3}, {"id": "someone/b", "likes": 3, "private": True}],
    }


@pytest.mark.asyncio
async def test_mine_lists_the_signed_in_users_repos(api: MagicMock) -> None:
    """'mine' resolves the username from the token and filters on it."""
    api.whoami.return_value = {"name": "alice"}
    api.list_models.return_value = [_repo("alice/m", pipeline_tag="text-generation")]

    result = await hub_module.HuggingFaceHub()(_deps(), action="mine")

    api.list_models.assert_called_once_with(limit=5, author="alice", sort="last_modified")
    assert result["username"] == "alice"
    assert result["results"] == [{"id": "alice/m", "likes": 3, "task": "text-generation"}]


@pytest.mark.asyncio
async def test_mine_without_sign_in_never_calls_the_hub(api: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without any token there is no user to list, so the tool says so."""
    monkeypatch.setattr(hub_module, "current_hf_token", lambda: None)

    result = await hub_module.HuggingFaceHub()(_deps(), action="mine", kind="dataset")

    assert result == {"action": "mine", "kind": "dataset", "error": "the robot is not signed in to Hugging Face"}
    api.whoami.assert_not_called()


@pytest.mark.asyncio
async def test_details_of_an_unknown_repo_points_back_to_search(api: MagicMock) -> None:
    """A short name is not a repo id; the error tells the model to search first."""
    api.model_info.side_effect = RepositoryNotFoundError("404", response=MagicMock())

    result = await hub_module.HuggingFaceHub()(_deps(), action="details", query="whisper")

    assert result["error"] == "no model named 'whisper'; search for it to get its full id"


@pytest.mark.asyncio
async def test_rejects_bad_arguments_before_any_request(api: MagicMock) -> None:
    """Unknown actions and missing queries fail fast."""
    tool = hub_module.HuggingFaceHub()

    assert await tool(_deps(), action="delete") == {"error": "unknown action 'delete'"}
    assert await tool(_deps(), action="details") == {"error": "'details' needs a query"}
    api.model_info.assert_not_called()
