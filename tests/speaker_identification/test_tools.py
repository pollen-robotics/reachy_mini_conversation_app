"""Tests for speaker identification tools."""

from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.identify_speaker import (
    ListSpeakers,
    RemoveSpeaker,
    IdentifySpeaker,
)
from reachy_mini_conversation_app.tools.register_speaker import (
    RegisterSpeaker,
    CancelSpeakerRegistration,
    FinishSpeakerRegistration,
)


@pytest.fixture
def mock_deps() -> ToolDependencies:
    """Create mock ToolDependencies with a mocked speaker_id_worker."""
    deps = MagicMock(spec=ToolDependencies)
    deps.speaker_id_worker = MagicMock()
    return deps


@pytest.fixture
def deps_without_speaker_id() -> ToolDependencies:
    """Create mock ToolDependencies without speaker_id_worker."""
    deps = MagicMock(spec=ToolDependencies)
    deps.speaker_id_worker = None
    return deps


class TestRegisterSpeaker:
    """Tests for RegisterSpeaker tool."""

    @pytest.mark.asyncio
    async def test_register_speaker_success(self, mock_deps: ToolDependencies) -> None:
        """Test successful speaker registration start."""
        tool = RegisterSpeaker()
        result = await tool(mock_deps, name="Alice")

        assert result["status"] == "registration_started"
        assert "Alice" in result["message"]
        mock_deps.speaker_id_worker.start_registration.assert_called_once_with("Alice")

    @pytest.mark.asyncio
    async def test_register_speaker_no_worker(
        self, deps_without_speaker_id: ToolDependencies
    ) -> None:
        """Test error when speaker ID is not enabled."""
        tool = RegisterSpeaker()
        result = await tool(deps_without_speaker_id, name="Alice")

        assert "error" in result
        assert "not enabled" in result["error"]

    @pytest.mark.asyncio
    async def test_register_speaker_missing_name(self, mock_deps: ToolDependencies) -> None:
        """Test error when name is missing."""
        tool = RegisterSpeaker()
        result = await tool(mock_deps)

        assert "error" in result
        assert "name" in result["error"]


class TestFinishSpeakerRegistration:
    """Tests for FinishSpeakerRegistration tool."""

    @pytest.mark.asyncio
    async def test_finish_registration_success(self, mock_deps: ToolDependencies) -> None:
        """Test successful registration completion."""
        mock_deps.speaker_id_worker.is_registering.return_value = True
        mock_deps.speaker_id_worker.finish_registration.return_value = True

        tool = FinishSpeakerRegistration()
        result = await tool(mock_deps)

        assert result["status"] == "success"
        assert "registered successfully" in result["message"]
        mock_deps.speaker_id_worker.finish_registration.assert_called_once()

    @pytest.mark.asyncio
    async def test_finish_registration_failed(self, mock_deps: ToolDependencies) -> None:
        """Test registration failure (not enough audio)."""
        mock_deps.speaker_id_worker.is_registering.return_value = True
        mock_deps.speaker_id_worker.finish_registration.return_value = False

        tool = FinishSpeakerRegistration()
        result = await tool(mock_deps)

        assert result["status"] == "failed"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_finish_registration_not_started(self, mock_deps: ToolDependencies) -> None:
        """Test error when no registration is in progress."""
        mock_deps.speaker_id_worker.is_registering.return_value = False

        tool = FinishSpeakerRegistration()
        result = await tool(mock_deps)

        assert "error" in result
        assert "No registration" in result["error"]

    @pytest.mark.asyncio
    async def test_finish_registration_no_worker(
        self, deps_without_speaker_id: ToolDependencies
    ) -> None:
        """Test error when speaker ID is not enabled."""
        tool = FinishSpeakerRegistration()
        result = await tool(deps_without_speaker_id)

        assert "error" in result
        assert "not enabled" in result["error"]


class TestCancelSpeakerRegistration:
    """Tests for CancelSpeakerRegistration tool."""

    @pytest.mark.asyncio
    async def test_cancel_registration_success(self, mock_deps: ToolDependencies) -> None:
        """Test successful registration cancellation."""
        mock_deps.speaker_id_worker.is_registering.return_value = True

        tool = CancelSpeakerRegistration()
        result = await tool(mock_deps)

        assert result["status"] == "cancelled"
        mock_deps.speaker_id_worker.cancel_registration.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_registration_none_in_progress(
        self, mock_deps: ToolDependencies
    ) -> None:
        """Test when no registration is in progress."""
        mock_deps.speaker_id_worker.is_registering.return_value = False

        tool = CancelSpeakerRegistration()
        result = await tool(mock_deps)

        assert result["status"] == "no_registration"

    @pytest.mark.asyncio
    async def test_cancel_registration_no_worker(
        self, deps_without_speaker_id: ToolDependencies
    ) -> None:
        """Test error when speaker ID is not enabled."""
        tool = CancelSpeakerRegistration()
        result = await tool(deps_without_speaker_id)

        assert "error" in result
        assert "not enabled" in result["error"]


class TestIdentifySpeaker:
    """Tests for IdentifySpeaker tool."""

    @pytest.mark.asyncio
    async def test_identify_speaker_known(self, mock_deps: ToolDependencies) -> None:
        """Test identifying a known speaker."""
        mock_deps.speaker_id_worker.get_current_speaker.return_value = ("Alice", 0.85)

        tool = IdentifySpeaker()
        result = await tool(mock_deps)

        assert result["status"] == "identified"
        assert result["speaker"] == "Alice"
        assert result["confidence"] == 0.85
        assert "Alice" in result["message"]

    @pytest.mark.asyncio
    async def test_identify_speaker_unknown(self, mock_deps: ToolDependencies) -> None:
        """Test identifying an unknown speaker."""
        mock_deps.speaker_id_worker.get_current_speaker.return_value = (None, 0.0)

        tool = IdentifySpeaker()
        result = await tool(mock_deps)

        assert result["status"] == "unknown"
        assert result["speaker"] is None
        assert "don't recognize" in result["message"]

    @pytest.mark.asyncio
    async def test_identify_speaker_no_worker(
        self, deps_without_speaker_id: ToolDependencies
    ) -> None:
        """Test error when speaker ID is not enabled."""
        tool = IdentifySpeaker()
        result = await tool(deps_without_speaker_id)

        assert "error" in result
        assert "not enabled" in result["error"]


class TestListSpeakers:
    """Tests for ListSpeakers tool."""

    @pytest.mark.asyncio
    async def test_list_speakers_with_speakers(self, mock_deps: ToolDependencies) -> None:
        """Test listing speakers when some are registered."""
        mock_deps.speaker_id_worker.list_speakers.return_value = ["Alice", "Bob", "Charlie"]

        tool = ListSpeakers()
        result = await tool(mock_deps)

        assert result["status"] == "success"
        assert result["speakers"] == ["Alice", "Bob", "Charlie"]
        assert result["count"] == 3
        assert "Alice" in result["message"]

    @pytest.mark.asyncio
    async def test_list_speakers_empty(self, mock_deps: ToolDependencies) -> None:
        """Test listing speakers when none are registered."""
        mock_deps.speaker_id_worker.list_speakers.return_value = []

        tool = ListSpeakers()
        result = await tool(mock_deps)

        assert result["status"] == "empty"
        assert result["speakers"] == []
        assert result["count"] == 0
        assert "No speakers" in result["message"]

    @pytest.mark.asyncio
    async def test_list_speakers_no_worker(
        self, deps_without_speaker_id: ToolDependencies
    ) -> None:
        """Test error when speaker ID is not enabled."""
        tool = ListSpeakers()
        result = await tool(deps_without_speaker_id)

        assert "error" in result
        assert "not enabled" in result["error"]


class TestRemoveSpeaker:
    """Tests for RemoveSpeaker tool."""

    @pytest.mark.asyncio
    async def test_remove_speaker_success(self, mock_deps: ToolDependencies) -> None:
        """Test successful speaker removal."""
        mock_deps.speaker_id_worker.remove_speaker.return_value = True

        tool = RemoveSpeaker()
        result = await tool(mock_deps, name="Alice")

        assert result["status"] == "success"
        assert "Removed" in result["message"]
        mock_deps.speaker_id_worker.remove_speaker.assert_called_once_with("Alice")

    @pytest.mark.asyncio
    async def test_remove_speaker_not_found(self, mock_deps: ToolDependencies) -> None:
        """Test removing a speaker that doesn't exist."""
        mock_deps.speaker_id_worker.remove_speaker.return_value = False

        tool = RemoveSpeaker()
        result = await tool(mock_deps, name="Unknown")

        assert result["status"] == "not_found"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_remove_speaker_missing_name(self, mock_deps: ToolDependencies) -> None:
        """Test error when name is missing."""
        tool = RemoveSpeaker()
        result = await tool(mock_deps)

        assert "error" in result
        assert "name" in result["error"]

    @pytest.mark.asyncio
    async def test_remove_speaker_no_worker(
        self, deps_without_speaker_id: ToolDependencies
    ) -> None:
        """Test error when speaker ID is not enabled."""
        tool = RemoveSpeaker()
        result = await tool(deps_without_speaker_id, name="Alice")

        assert "error" in result
        assert "not enabled" in result["error"]


class TestToolSpecs:
    """Tests for tool specifications."""

    def test_register_speaker_spec(self) -> None:
        """Test RegisterSpeaker tool spec is valid."""
        tool = RegisterSpeaker()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "register_speaker"
        assert "name" in spec["parameters"]["properties"]
        assert "name" in spec["parameters"]["required"]

    def test_finish_registration_spec(self) -> None:
        """Test FinishSpeakerRegistration tool spec is valid."""
        tool = FinishSpeakerRegistration()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "finish_speaker_registration"

    def test_cancel_registration_spec(self) -> None:
        """Test CancelSpeakerRegistration tool spec is valid."""
        tool = CancelSpeakerRegistration()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "cancel_speaker_registration"

    def test_identify_speaker_spec(self) -> None:
        """Test IdentifySpeaker tool spec is valid."""
        tool = IdentifySpeaker()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "identify_speaker"

    def test_list_speakers_spec(self) -> None:
        """Test ListSpeakers tool spec is valid."""
        tool = ListSpeakers()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "list_speakers"

    def test_remove_speaker_spec(self) -> None:
        """Test RemoveSpeaker tool spec is valid."""
        tool = RemoveSpeaker()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "remove_speaker"
        assert "name" in spec["parameters"]["properties"]
        assert "name" in spec["parameters"]["required"]
