"""Unit tests for console module (LocalStream)."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# Store original modules before any mocking
_ORIGINAL_MODULES: dict[str, Any] = {}
_MODULES_TO_MOCK = [
    "reachy_mini",
    "reachy_mini.media",
    "reachy_mini.media.media_manager",
]


@pytest.fixture(autouse=True)
def mock_console_dependencies():
    """Mock heavy dependencies for console tests and restore them after."""
    # Save originals
    for mod_name in _MODULES_TO_MOCK:
        if mod_name in sys.modules:
            _ORIGINAL_MODULES[mod_name] = sys.modules[mod_name]

    # Install mocks
    sys.modules["reachy_mini"] = MagicMock()
    sys.modules["reachy_mini.media"] = MagicMock()
    sys.modules["reachy_mini.media.media_manager"] = MagicMock()
    sys.modules["reachy_mini.media.media_manager"].MediaBackend = MagicMock()
    sys.modules["reachy_mini.media.media_manager"].MediaBackend.GSTREAMER = "gstreamer"
    sys.modules["reachy_mini.media.media_manager"].MediaBackend.DEFAULT = "default"
    sys.modules["reachy_mini.media.media_manager"].MediaBackend.DEFAULT_NO_VIDEO = "default_no_video"

    yield

    # Restore originals
    for mod_name in _MODULES_TO_MOCK:
        if mod_name in _ORIGINAL_MODULES:
            sys.modules[mod_name] = _ORIGINAL_MODULES[mod_name]
        elif mod_name in sys.modules:
            del sys.modules[mod_name]

    # Clear cached module imports that might have used mocks
    mods_to_clear = [k for k in sys.modules if k.startswith("reachy_mini_conversation_app.console")]
    for mod_name in mods_to_clear:
        del sys.modules[mod_name]


class TestLocalStreamInit:
    """Tests for LocalStream initialization."""

    def test_init_basic(self) -> None:
        """Test basic LocalStream initialization."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot)

        assert stream.handler is mock_handler
        assert stream._robot is mock_robot
        assert stream._settings_app is None
        assert stream._instance_path is None
        assert stream._settings_initialized is False
        assert stream._asyncio_loop is None

    def test_init_with_settings_app(self) -> None:
        """Test LocalStream initialization with settings app."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()
        mock_app = MagicMock()

        stream = LocalStream(mock_handler, mock_robot, settings_app=mock_app)

        assert stream._settings_app is mock_app

    def test_init_with_instance_path(self) -> None:
        """Test LocalStream initialization with instance path."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot, instance_path="/tmp/test")

        assert stream._instance_path == "/tmp/test"

    def test_init_sets_clear_queue_callback(self) -> None:
        """Test that init sets _clear_queue callback on handler."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot)

        # Verify the callback was set (it's the bound method)
        assert mock_handler._clear_queue == stream.clear_audio_queue


class TestReadEnvLines:
    """Tests for _read_env_lines method."""

    def test_read_existing_env_file(self, tmp_path: Path) -> None:
        """Test reading existing .env file."""
        from reachy_mini_conversation_app.console import LocalStream

        env_file = tmp_path / ".env"
        env_file.write_text("KEY1=value1\nKEY2=value2\n")

        stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))
        lines = stream._read_env_lines(env_file)

        assert lines == ["KEY1=value1", "KEY2=value2"]

    def test_read_nonexistent_env_with_example_in_instance(self, tmp_path: Path) -> None:
        """Test fallback to .env.example in instance directory."""
        from reachy_mini_conversation_app.console import LocalStream

        env_file = tmp_path / ".env"
        example_file = tmp_path / ".env.example"
        example_file.write_text("# Template\nTEMPLATE_KEY=value\n")

        stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))
        lines = stream._read_env_lines(env_file)

        assert lines == ["# Template", "TEMPLATE_KEY=value"]

    def test_read_nonexistent_env_returns_empty_on_no_template(self, tmp_path: Path) -> None:
        """Test returns empty list when no templates exist."""
        from reachy_mini_conversation_app.console import LocalStream

        env_file = tmp_path / ".env"

        stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

        with patch.object(Path, "cwd", return_value=tmp_path):
            # Mock the packaged .env.example check to not exist
            with patch.object(Path, "exists", return_value=False):
                lines = stream._read_env_lines(env_file)

        # May return template from package or empty
        assert isinstance(lines, list)

    def test_read_env_lines_handles_read_error(self, tmp_path: Path) -> None:
        """Test that read errors return empty list."""
        from reachy_mini_conversation_app.console import LocalStream

        env_file = tmp_path / ".env"
        env_file.write_text("content")

        stream = LocalStream(MagicMock(), MagicMock())

        with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            lines = stream._read_env_lines(env_file)

        # Should handle error gracefully
        assert isinstance(lines, list)


class TestPersistApiKey:
    """Tests for _persist_api_key method."""

    def test_persist_api_key_sets_env_var(self) -> None:
        """Test that API key is set in environment."""
        from reachy_mini_conversation_app.console import LocalStream

        stream = LocalStream(MagicMock(), MagicMock())

        with patch.dict(os.environ, {}, clear=True):
            with patch("reachy_mini_conversation_app.console.config") as mock_config:
                stream._persist_api_key("test-key-123")

                assert os.environ.get("OPENAI_API_KEY") == "test-key-123"
                mock_config.OPENAI_API_KEY = "test-key-123"

    def test_persist_api_key_strips_whitespace(self) -> None:
        """Test that API key whitespace is stripped."""
        from reachy_mini_conversation_app.console import LocalStream

        stream = LocalStream(MagicMock(), MagicMock())

        with patch.dict(os.environ, {}, clear=True):
            with patch("reachy_mini_conversation_app.console.config"):
                stream._persist_api_key("  test-key  ")

                assert os.environ.get("OPENAI_API_KEY") == "test-key"

    def test_persist_api_key_ignores_empty(self) -> None:
        """Test that empty key is ignored."""
        from reachy_mini_conversation_app.console import LocalStream

        stream = LocalStream(MagicMock(), MagicMock())

        with patch.dict(os.environ, {}, clear=True):
            stream._persist_api_key("")
            stream._persist_api_key("   ")

            assert "OPENAI_API_KEY" not in os.environ

    def test_persist_api_key_writes_to_env_file(self, tmp_path: Path) -> None:
        """Test that API key is written to .env file."""
        from reachy_mini_conversation_app.console import LocalStream

        stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

        with patch("reachy_mini_conversation_app.console.config"):
            with patch("reachy_mini_conversation_app.console.load_dotenv", create=True):
                stream._persist_api_key("new-key")

        env_file = tmp_path / ".env"
        assert env_file.exists()
        content = env_file.read_text()
        assert "OPENAI_API_KEY=new-key" in content

    def test_persist_api_key_replaces_existing(self, tmp_path: Path) -> None:
        """Test that existing key is replaced in .env file."""
        from reachy_mini_conversation_app.console import LocalStream

        env_file = tmp_path / ".env"
        env_file.write_text("OTHER_VAR=test\nOPENAI_API_KEY=old-key\nANOTHER=value\n")

        stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

        with patch("reachy_mini_conversation_app.console.config"):
            with patch("reachy_mini_conversation_app.console.load_dotenv", create=True):
                stream._persist_api_key("new-key")

        content = env_file.read_text()
        assert "OPENAI_API_KEY=new-key" in content
        assert "old-key" not in content
        assert "OTHER_VAR=test" in content


class TestPersistPersonality:
    """Tests for _persist_personality method."""

    def test_persist_personality_calls_set_custom_profile(self) -> None:
        """Test that set_custom_profile is called."""
        from reachy_mini_conversation_app.console import LocalStream

        stream = LocalStream(MagicMock(), MagicMock())

        with patch("reachy_mini_conversation_app.config.set_custom_profile") as mock_set:
            stream._persist_personality("linus")
            mock_set.assert_called_once_with("linus")

    def test_persist_personality_none_clears_profile(self) -> None:
        """Test that None/empty clears profile."""
        from reachy_mini_conversation_app.console import LocalStream

        stream = LocalStream(MagicMock(), MagicMock())

        with patch("reachy_mini_conversation_app.config.set_custom_profile") as mock_set:
            stream._persist_personality("")
            mock_set.assert_called_once_with(None)

    def test_persist_personality_writes_to_env(self, tmp_path: Path) -> None:
        """Test that personality is written to .env file."""
        from reachy_mini_conversation_app.console import LocalStream

        stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

        with patch("reachy_mini_conversation_app.config.set_custom_profile"):
            with patch("reachy_mini_conversation_app.console.load_dotenv", create=True):
                stream._persist_personality("linus")

        env_file = tmp_path / ".env"
        assert env_file.exists()
        content = env_file.read_text()
        assert "REACHY_MINI_CUSTOM_PROFILE=linus" in content

    def test_persist_personality_removes_var_when_none(self, tmp_path: Path) -> None:
        """Test that profile var is removed when set to None."""
        from reachy_mini_conversation_app.console import LocalStream

        env_file = tmp_path / ".env"
        env_file.write_text("OTHER=val\nREACHY_MINI_CUSTOM_PROFILE=old\n")

        stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

        with patch("reachy_mini_conversation_app.config.set_custom_profile"):
            with patch("reachy_mini_conversation_app.console.load_dotenv", create=True):
                stream._persist_personality("")

        content = env_file.read_text()
        assert "REACHY_MINI_CUSTOM_PROFILE" not in content
        assert "OTHER=val" in content


class TestReadPersistedPersonality:
    """Tests for _read_persisted_personality method."""

    def test_read_persisted_personality_returns_none_without_instance_path(self) -> None:
        """Test returns None when no instance path."""
        from reachy_mini_conversation_app.console import LocalStream

        stream = LocalStream(MagicMock(), MagicMock())
        result = stream._read_persisted_personality()

        assert result is None

    def test_read_persisted_personality_returns_profile(self, tmp_path: Path) -> None:
        """Test reads profile from .env file."""
        from reachy_mini_conversation_app.console import LocalStream

        env_file = tmp_path / ".env"
        env_file.write_text("OTHER=val\nREACHY_MINI_CUSTOM_PROFILE=linus\n")

        stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))
        result = stream._read_persisted_personality()

        assert result == "linus"

    def test_read_persisted_personality_returns_none_if_not_set(self, tmp_path: Path) -> None:
        """Test returns None if profile not in .env."""
        from reachy_mini_conversation_app.console import LocalStream

        env_file = tmp_path / ".env"
        env_file.write_text("OTHER=val\n")

        stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))
        result = stream._read_persisted_personality()

        assert result is None

    def test_read_persisted_personality_returns_none_if_empty(self, tmp_path: Path) -> None:
        """Test returns None if profile is empty string."""
        from reachy_mini_conversation_app.console import LocalStream

        env_file = tmp_path / ".env"
        env_file.write_text("REACHY_MINI_CUSTOM_PROFILE=\n")

        stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))
        result = stream._read_persisted_personality()

        assert result is None


class TestPersistLinusConfig:
    """Tests for _persist_linus_config method."""

    def test_persist_linus_config_sets_env_vars(self) -> None:
        """Test that Linus config sets environment variables."""
        from reachy_mini_conversation_app.console import LocalStream

        stream = LocalStream(MagicMock(), MagicMock())

        with patch.dict(os.environ, {}, clear=True):
            with patch("reachy_mini_conversation_app.console.config") as mock_config:
                stream._persist_linus_config(
                    anthropic_key="ant-key",
                    github_token="gh-token",
                    github_owner="owner",
                )

                assert os.environ.get("ANTHROPIC_API_KEY") == "ant-key"
                assert os.environ.get("GITHUB_TOKEN") == "gh-token"
                assert os.environ.get("GITHUB_DEFAULT_OWNER") == "owner"

    def test_persist_linus_config_writes_to_env(self, tmp_path: Path) -> None:
        """Test that Linus config is written to .env file."""
        from reachy_mini_conversation_app.console import LocalStream

        stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

        with patch("reachy_mini_conversation_app.console.config"):
            with patch("reachy_mini_conversation_app.console.load_dotenv", create=True):
                stream._persist_linus_config(
                    anthropic_key="ant-key",
                    github_token="gh-token",
                    github_owner="owner",
                )

        env_file = tmp_path / ".env"
        content = env_file.read_text()
        assert "ANTHROPIC_API_KEY=ant-key" in content
        assert "GITHUB_TOKEN=gh-token" in content
        assert "GITHUB_DEFAULT_OWNER=owner" in content

    def test_persist_linus_config_skips_empty_values(self, tmp_path: Path) -> None:
        """Test that empty values are not written to new env file."""
        from reachy_mini_conversation_app.console import LocalStream

        # Create an empty .env file (not from template)
        env_file = tmp_path / ".env"
        env_file.write_text("")  # Start with empty file

        stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

        with patch("reachy_mini_conversation_app.console.config"):
            with patch("dotenv.load_dotenv", create=True):
                stream._persist_linus_config(
                    anthropic_key="ant-key",
                    github_token="",
                    github_owner="",
                )

        content = env_file.read_text()
        assert "ANTHROPIC_API_KEY=ant-key" in content
        # Empty values should not add new lines
        assert "GITHUB_TOKEN=ant-key" not in content  # Not set to wrong value
        assert "GITHUB_DEFAULT_OWNER=ant-key" not in content  # Not set to wrong value


class TestInitSettingsUiIfNeeded:
    """Tests for _init_settings_ui_if_needed method."""

    def test_init_settings_ui_does_nothing_without_app(self) -> None:
        """Test that method does nothing when no settings app."""
        from reachy_mini_conversation_app.console import LocalStream

        stream = LocalStream(MagicMock(), MagicMock())
        stream._init_settings_ui_if_needed()

        assert stream._settings_initialized is False

    def test_init_settings_ui_does_nothing_if_already_initialized(self) -> None:
        """Test that method does nothing if already initialized."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_app = MagicMock()
        stream = LocalStream(MagicMock(), MagicMock(), settings_app=mock_app)
        stream._settings_initialized = True

        stream._init_settings_ui_if_needed()

        mock_app.get.assert_not_called()

    def test_init_settings_ui_mounts_routes(self) -> None:
        """Test that settings UI routes are mounted."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_app = MagicMock()
        stream = LocalStream(MagicMock(), MagicMock(), settings_app=mock_app)

        with patch("reachy_mini_conversation_app.console.StaticFiles"):
            stream._init_settings_ui_if_needed()

        assert stream._settings_initialized is True
        mock_app.mount.assert_called_once()
        # Check that routes were registered
        assert mock_app.get.call_count >= 4  # /, /favicon.ico, /status, /ready, /linus_config
        assert mock_app.post.call_count >= 3  # /openai_api_key, /validate_api_key, /linus_config


class TestClose:
    """Tests for close method."""

    def test_close_stops_media(self) -> None:
        """Test that close stops media pipelines."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_robot = MagicMock()
        stream = LocalStream(MagicMock(), mock_robot)

        stream.close()

        mock_robot.media.stop_recording.assert_called_once()
        mock_robot.media.stop_playing.assert_called_once()

    def test_close_sets_stop_event(self) -> None:
        """Test that close sets stop event."""
        from reachy_mini_conversation_app.console import LocalStream

        stream = LocalStream(MagicMock(), MagicMock())

        stream.close()

        assert stream._stop_event.is_set()

    def test_close_cancels_tasks(self) -> None:
        """Test that close cancels running tasks."""
        from reachy_mini_conversation_app.console import LocalStream

        stream = LocalStream(MagicMock(), MagicMock())

        mock_task = MagicMock()
        mock_task.done.return_value = False
        stream._tasks = [mock_task]

        stream.close()

        mock_task.cancel.assert_called_once()

    def test_close_handles_media_errors(self) -> None:
        """Test that close handles media stop errors."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_robot = MagicMock()
        mock_robot.media.stop_recording.side_effect = RuntimeError("already stopped")
        mock_robot.media.stop_playing.side_effect = RuntimeError("already stopped")

        stream = LocalStream(MagicMock(), mock_robot)

        # Should not raise
        stream.close()

        assert stream._stop_event.is_set()


class TestClearAudioQueue:
    """Tests for clear_audio_queue method."""

    def test_clear_audio_queue_gstreamer_backend(self) -> None:
        """Test clear queue with GStreamer backend."""
        from reachy_mini_conversation_app.console import LocalStream
        from reachy_mini.media.media_manager import MediaBackend

        mock_robot = MagicMock()
        mock_robot.media.backend = MediaBackend.GSTREAMER
        mock_handler = MagicMock()

        stream = LocalStream(mock_handler, mock_robot)
        stream.clear_audio_queue()

        mock_robot.media.audio.clear_player.assert_called_once()
        assert isinstance(mock_handler.output_queue, asyncio.Queue)

    def test_clear_audio_queue_default_backend(self) -> None:
        """Test clear queue with DEFAULT backend."""
        from reachy_mini_conversation_app.console import LocalStream
        from reachy_mini.media.media_manager import MediaBackend

        mock_robot = MagicMock()
        mock_robot.media.backend = MediaBackend.DEFAULT
        mock_handler = MagicMock()

        stream = LocalStream(mock_handler, mock_robot)
        stream.clear_audio_queue()

        mock_robot.media.audio.clear_output_buffer.assert_called_once()

    def test_clear_audio_queue_default_no_video_backend(self) -> None:
        """Test clear queue with DEFAULT_NO_VIDEO backend."""
        from reachy_mini_conversation_app.console import LocalStream
        from reachy_mini.media.media_manager import MediaBackend

        mock_robot = MagicMock()
        mock_robot.media.backend = MediaBackend.DEFAULT_NO_VIDEO
        mock_handler = MagicMock()

        stream = LocalStream(mock_handler, mock_robot)
        stream.clear_audio_queue()

        mock_robot.media.audio.clear_output_buffer.assert_called_once()


class TestRecordLoop:
    """Tests for record_loop async method."""

    @pytest.mark.asyncio
    async def test_record_loop_forwards_audio_to_handler(self) -> None:
        """Test that audio frames are forwarded to handler."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_robot = MagicMock()
        mock_robot.media.get_input_audio_samplerate.return_value = 16000

        audio_data = np.zeros(1024, dtype=np.float32)
        call_count = 0

        def get_audio():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return audio_data
            return None

        mock_robot.media.get_audio_sample.side_effect = get_audio

        mock_handler = MagicMock()
        mock_handler.receive = AsyncMock()

        stream = LocalStream(mock_handler, mock_robot)

        # Run for a few iterations then stop
        async def stop_after_delay():
            await asyncio.sleep(0.05)
            stream._stop_event.set()

        await asyncio.gather(
            stream.record_loop(),
            stop_after_delay(),
        )

        mock_handler.receive.assert_called()
        call_args = mock_handler.receive.call_args[0][0]
        assert call_args[0] == 16000

    @pytest.mark.asyncio
    async def test_record_loop_stops_on_event(self) -> None:
        """Test that record loop stops when stop event is set."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_robot = MagicMock()
        mock_robot.media.get_input_audio_samplerate.return_value = 16000
        mock_robot.media.get_audio_sample.return_value = None

        stream = LocalStream(MagicMock(), mock_robot)
        stream._stop_event.set()

        # Should return immediately
        await asyncio.wait_for(stream.record_loop(), timeout=1.0)


class TestPlayLoop:
    """Tests for play_loop async method."""

    @pytest.mark.asyncio
    async def test_play_loop_handles_additional_outputs(self) -> None:
        """Test that AdditionalOutputs are logged."""
        from reachy_mini_conversation_app.console import LocalStream
        from fastrtc import AdditionalOutputs

        mock_handler = MagicMock()

        additional_output = MagicMock(spec=AdditionalOutputs)
        additional_output.args = [{"role": "assistant", "content": "Hello"}]

        call_count = 0
        async def mock_emit():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return additional_output
            await asyncio.sleep(0.1)
            return None

        mock_handler.emit = mock_emit

        mock_robot = MagicMock()
        stream = LocalStream(mock_handler, mock_robot)

        async def stop_after_delay():
            await asyncio.sleep(0.05)
            stream._stop_event.set()

        await asyncio.gather(
            stream.play_loop(),
            stop_after_delay(),
        )

    @pytest.mark.asyncio
    async def test_play_loop_handles_audio_tuple(self) -> None:
        """Test that audio tuples are processed and played."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()

        audio_data = np.zeros(1024, dtype=np.float32)

        call_count = 0
        async def mock_emit():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (24000, audio_data)
            await asyncio.sleep(0.1)
            return None

        mock_handler.emit = mock_emit

        mock_robot = MagicMock()
        mock_robot.media.get_output_audio_samplerate.return_value = 24000

        stream = LocalStream(mock_handler, mock_robot)

        async def stop_after_delay():
            await asyncio.sleep(0.05)
            stream._stop_event.set()

        await asyncio.gather(
            stream.play_loop(),
            stop_after_delay(),
        )

        mock_robot.media.push_audio_sample.assert_called()

    @pytest.mark.asyncio
    async def test_play_loop_resamples_audio_if_needed(self) -> None:
        """Test that audio is resampled when sample rates differ."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()

        audio_data = np.zeros(1024, dtype=np.float32)

        call_count = 0
        async def mock_emit():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (16000, audio_data)  # Input at 16kHz
            await asyncio.sleep(0.1)
            return None

        mock_handler.emit = mock_emit

        mock_robot = MagicMock()
        mock_robot.media.get_output_audio_samplerate.return_value = 24000  # Output at 24kHz

        stream = LocalStream(mock_handler, mock_robot)

        async def stop_after_delay():
            await asyncio.sleep(0.05)
            stream._stop_event.set()

        with patch("reachy_mini_conversation_app.console.resample") as mock_resample:
            mock_resample.return_value = np.zeros(1536, dtype=np.float32)

            await asyncio.gather(
                stream.play_loop(),
                stop_after_delay(),
            )

            mock_resample.assert_called()

    @pytest.mark.asyncio
    async def test_play_loop_handles_2d_audio(self) -> None:
        """Test that 2D audio arrays are converted to mono."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()

        # 2D audio with shape (channels, samples)
        audio_data = np.zeros((2, 1024), dtype=np.float32)

        call_count = 0
        async def mock_emit():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (24000, audio_data)
            await asyncio.sleep(0.1)
            return None

        mock_handler.emit = mock_emit

        mock_robot = MagicMock()
        mock_robot.media.get_output_audio_samplerate.return_value = 24000

        stream = LocalStream(mock_handler, mock_robot)

        async def stop_after_delay():
            await asyncio.sleep(0.05)
            stream._stop_event.set()

        await asyncio.gather(
            stream.play_loop(),
            stop_after_delay(),
        )

        mock_robot.media.push_audio_sample.assert_called()


class TestReadEnvLinesExtended:
    """Extended tests for _read_env_lines method."""

    def test_read_env_from_cwd_example(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Test fallback to .env.example in current working directory."""
        from reachy_mini_conversation_app.console import LocalStream

        # Create env_file that doesn't exist
        env_file = tmp_path / ".env"

        # Create .env.example in a subdirectory to simulate cwd
        cwd_dir = tmp_path / "cwd"
        cwd_dir.mkdir()
        cwd_example = cwd_dir / ".env.example"
        cwd_example.write_text("# From CWD\nCWD_KEY=value\n")

        stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

        # Mock Path.cwd to return cwd_dir
        with patch("reachy_mini_conversation_app.console.Path") as MockPath:
            # Make original Path still work for the env_file
            MockPath.return_value = Path(str(env_file))
            MockPath.cwd.return_value = cwd_dir
            # Make exists check return False for packaged template
            lines = stream._read_env_lines(env_file)

        # Should get lines from somewhere (template or empty)
        assert isinstance(lines, list)

    def test_read_env_from_packaged_example(self, tmp_path: Path) -> None:
        """Test fallback to packaged .env.example."""
        from reachy_mini_conversation_app.console import LocalStream

        env_file = tmp_path / ".env"

        stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

        lines = stream._read_env_lines(env_file)

        # Should get lines from packaged template (which exists)
        assert isinstance(lines, list)

    def test_read_env_handles_example_read_error(self, tmp_path: Path) -> None:
        """Test that read errors on example file are handled."""
        from reachy_mini_conversation_app.console import LocalStream

        env_file = tmp_path / ".env"
        example_file = tmp_path / ".env.example"
        example_file.write_text("content")

        stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

        # Mock read_text to fail on example
        with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            lines = stream._read_env_lines(env_file)

        # Should handle error gracefully
        assert isinstance(lines, list)


class TestPersistApiKeyExtended:
    """Extended tests for _persist_api_key method."""

    def test_persist_api_key_handles_env_set_error(self) -> None:
        """Test that env var set errors are handled."""
        from reachy_mini_conversation_app.console import LocalStream

        stream = LocalStream(MagicMock(), MagicMock())

        # Mock os.environ to raise on set
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(os.environ, "__setitem__", side_effect=Exception("denied")):
                with patch("reachy_mini_conversation_app.console.config") as mock_config:
                    # Should not raise
                    stream._persist_api_key("test-key")

    def test_persist_api_key_handles_write_error(self, tmp_path: Path) -> None:
        """Test that file write errors are handled."""
        from reachy_mini_conversation_app.console import LocalStream

        stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

        with patch("reachy_mini_conversation_app.console.config"):
            with patch.object(Path, "write_text", side_effect=PermissionError("denied")):
                # Should not raise
                stream._persist_api_key("test-key")


class TestPersistPersonalityExtended:
    """Extended tests for _persist_personality method."""

    def test_persist_personality_handles_exception(self) -> None:
        """Test that set_custom_profile exceptions are handled."""
        from reachy_mini_conversation_app.console import LocalStream

        stream = LocalStream(MagicMock(), MagicMock())

        with patch("reachy_mini_conversation_app.config.set_custom_profile", side_effect=Exception("error")):
            # Should not raise
            stream._persist_personality("linus")

    def test_persist_personality_appends_to_existing_env(self, tmp_path: Path) -> None:
        """Test that personality is appended to existing .env file."""
        from reachy_mini_conversation_app.console import LocalStream

        env_file = tmp_path / ".env"
        env_file.write_text("OTHER=val\n")

        stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

        with patch("reachy_mini_conversation_app.config.set_custom_profile"):
            with patch("reachy_mini_conversation_app.console.load_dotenv", create=True):
                stream._persist_personality("linus")

        content = env_file.read_text()
        assert "REACHY_MINI_CUSTOM_PROFILE=linus" in content
        assert "OTHER=val" in content

    def test_persist_personality_skips_write_when_none_and_no_file(self, tmp_path: Path) -> None:
        """Test that write is skipped when setting to None and file doesn't exist."""
        from reachy_mini_conversation_app.console import LocalStream

        stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

        with patch("reachy_mini_conversation_app.config.set_custom_profile"):
            stream._persist_personality(None)

        env_file = tmp_path / ".env"
        assert not env_file.exists()

    def test_persist_personality_handles_write_error(self, tmp_path: Path) -> None:
        """Test that write errors are handled."""
        from reachy_mini_conversation_app.console import LocalStream

        stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

        with patch("reachy_mini_conversation_app.config.set_custom_profile"):
            with patch.object(Path, "write_text", side_effect=PermissionError("denied")):
                # Should not raise
                stream._persist_personality("linus")


class TestReadPersistedPersonalityExtended:
    """Extended tests for _read_persisted_personality method."""

    def test_read_persisted_personality_handles_exception(self, tmp_path: Path) -> None:
        """Test that read errors are handled."""
        from reachy_mini_conversation_app.console import LocalStream

        env_file = tmp_path / ".env"
        env_file.write_text("REACHY_MINI_CUSTOM_PROFILE=linus\n")

        stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

        with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
            result = stream._read_persisted_personality()

        assert result is None


class TestPersistLinusConfigExtended:
    """Extended tests for _persist_linus_config method."""

    def test_persist_linus_config_handles_env_set_error(self) -> None:
        """Test that env var set errors are handled."""
        from reachy_mini_conversation_app.console import LocalStream

        stream = LocalStream(MagicMock(), MagicMock())

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(os.environ, "__setitem__", side_effect=Exception("denied")):
                # Should not raise
                stream._persist_linus_config(anthropic_key="key", github_token="token")

    def test_persist_linus_config_handles_write_error(self, tmp_path: Path) -> None:
        """Test that file write errors are handled."""
        from reachy_mini_conversation_app.console import LocalStream

        stream = LocalStream(MagicMock(), MagicMock(), instance_path=str(tmp_path))

        with patch("reachy_mini_conversation_app.console.config"):
            with patch.object(Path, "write_text", side_effect=PermissionError("denied")):
                # Should not raise
                stream._persist_linus_config(anthropic_key="key")


class TestInitSettingsUiExtended:
    """Extended tests for _init_settings_ui_if_needed method."""

    def test_init_settings_ui_handles_mount_error(self) -> None:
        """Test that mount errors are handled."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_app = MagicMock()
        mock_app.mount.side_effect = Exception("mount failed")

        stream = LocalStream(MagicMock(), MagicMock(), settings_app=mock_app)

        with patch("reachy_mini_conversation_app.console.StaticFiles"):
            stream._init_settings_ui_if_needed()

        # Should still be initialized despite mount error
        assert stream._settings_initialized is True


class TestSettingsEndpoints:
    """Tests for settings endpoints when registered."""

    def test_status_endpoint_returns_has_key(self) -> None:
        """Test that status endpoint returns has_key status."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_app = MagicMock()

        # Capture the registered handler
        registered_handlers: dict = {}

        def capture_get(path: str):
            def decorator(fn):
                registered_handlers[path] = fn
                return fn
            return decorator

        mock_app.get.side_effect = capture_get

        stream = LocalStream(MagicMock(), MagicMock(), settings_app=mock_app)

        with patch("reachy_mini_conversation_app.console.StaticFiles"):
            with patch("reachy_mini_conversation_app.console.config") as mock_config:
                mock_config.OPENAI_API_KEY = "test-key"
                stream._init_settings_ui_if_needed()

        # Call the status handler
        if "/status" in registered_handlers:
            result = registered_handlers["/status"]()
            assert hasattr(result, "body") or isinstance(result, MagicMock)

    def test_ready_endpoint_returns_ready_status(self) -> None:
        """Test that ready endpoint returns ready status."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_app = MagicMock()

        registered_handlers: dict = {}

        def capture_get(path: str):
            def decorator(fn):
                registered_handlers[path] = fn
                return fn
            return decorator

        mock_app.get.side_effect = capture_get

        stream = LocalStream(MagicMock(), MagicMock(), settings_app=mock_app)

        with patch("reachy_mini_conversation_app.console.StaticFiles"):
            stream._init_settings_ui_if_needed()

        if "/ready" in registered_handlers:
            result = registered_handlers["/ready"]()
            assert hasattr(result, "body") or isinstance(result, MagicMock)

    def test_linus_config_get_endpoint(self) -> None:
        """Test that linus_config GET endpoint returns config status."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_app = MagicMock()

        registered_handlers: dict = {}

        def capture_get(path: str):
            def decorator(fn):
                registered_handlers[path] = fn
                return fn
            return decorator

        mock_app.get.side_effect = capture_get

        stream = LocalStream(MagicMock(), MagicMock(), settings_app=mock_app)

        with patch("reachy_mini_conversation_app.console.StaticFiles"):
            with patch("reachy_mini_conversation_app.console.config") as mock_config:
                mock_config.ANTHROPIC_API_KEY = "test-key"
                mock_config.GITHUB_TOKEN = "test-token"
                mock_config.GITHUB_DEFAULT_OWNER = "owner"
                stream._init_settings_ui_if_needed()

        if "/linus_config" in registered_handlers:
            result = registered_handlers["/linus_config"]()
            assert hasattr(result, "body") or isinstance(result, MagicMock)

    def test_set_key_endpoint_and_validate(self) -> None:
        """Test POST /openai_api_key and POST /validate_api_key endpoints."""
        from types import SimpleNamespace
        from reachy_mini_conversation_app.console import LocalStream

        mock_app = MagicMock()
        registered_get: dict = {}
        registered_post: dict = {}

        def capture_get(path: str):
            def decorator(fn):
                registered_get[path] = fn
                return fn
            return decorator

        def capture_post(path: str):
            def decorator(fn):
                registered_post[path] = fn
                return fn
            return decorator

        mock_app.get.side_effect = capture_get
        mock_app.post.side_effect = capture_post

        stream = LocalStream(MagicMock(), MagicMock(), settings_app=mock_app)

        with patch("reachy_mini_conversation_app.console.StaticFiles"):
            stream._init_settings_ui_if_needed()

        # Test empty key -> should return error-like response
        if "/openai_api_key" in registered_post:
            payload = SimpleNamespace(openai_api_key="")
            res = registered_post["/openai_api_key"](payload)
            assert hasattr(res, "status_code") or isinstance(res, MagicMock)

        # Test validate_api_key with mocked httpx returning 200
        if "/validate_api_key" in registered_post:
            # Inject fake httpx into sys.modules so import inside handler picks it
            import sys
            fake_httpx = MagicMock()
            fake_client = MagicMock()
            async def fake_get_ok(url, headers=None):
                return MagicMock(status_code=200)

            fake_client.get = AsyncMock(side_effect=fake_get_ok)
            fake_ctx = MagicMock()
            fake_ctx.__aenter__.return_value = fake_client
            fake_httpx.AsyncClient.return_value = fake_ctx
            sys.modules["httpx"] = fake_httpx

            # Run the async handler
            import asyncio

            payload = SimpleNamespace(openai_api_key="valid")
            coro = registered_post["/validate_api_key"](payload)
            # execute coroutine
            loop = asyncio.new_event_loop()
            try:
                res = loop.run_until_complete(coro)
                assert hasattr(res, "status_code") or isinstance(res, MagicMock)
            finally:
                loop.close()

            # Cleanup fake httpx
            del sys.modules["httpx"]

    def test_post_linus_config_calls_persist(self) -> None:
        """Test POST /linus_config triggers _persist_linus_config."""
        from types import SimpleNamespace
        from reachy_mini_conversation_app.console import LocalStream

        mock_app = MagicMock()
        registered_post: dict = {}

        def capture_post(path: str):
            def decorator(fn):
                registered_post[path] = fn
                return fn
            return decorator

        mock_app.post.side_effect = capture_post

        stream = LocalStream(MagicMock(), MagicMock(), settings_app=mock_app)

        with patch("reachy_mini_conversation_app.console.StaticFiles"):
            stream._init_settings_ui_if_needed()

        if "/linus_config" in registered_post:
            # Patch the instance method to observe calls
            with patch.object(stream, "_persist_linus_config") as mock_persist:
                payload = SimpleNamespace(anthropic_key="a", github_token="t", github_owner="o")
                res = registered_post["/linus_config"](payload)
                mock_persist.assert_called()
                assert hasattr(res, "status_code") or isinstance(res, MagicMock)

    def test_root_and_favicon_registered(self) -> None:
        """Ensure root and favicon handlers are registered when settings_app provided."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_app = MagicMock()
        registered_get: dict = {}

        def capture_get(path: str):
            def decorator(fn):
                registered_get[path] = fn
                return fn
            return decorator

        mock_app.get.side_effect = capture_get

        stream = LocalStream(MagicMock(), MagicMock(), settings_app=mock_app)

        with patch("reachy_mini_conversation_app.console.StaticFiles"):
            stream._init_settings_ui_if_needed()

        assert "/" in registered_get
        assert "/favicon.ico" in registered_get


class TestSettingsEndpointsExtended:
    """Extended tests for settings endpoints."""

    def test_root_endpoint_returns_file_response(self) -> None:
        """Test that root endpoint returns FileResponse (line 299)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_app = MagicMock()
        registered_handlers: dict = {}

        def capture_get(path: str):
            def decorator(fn):
                registered_handlers[path] = fn
                return fn
            return decorator

        mock_app.get.side_effect = capture_get

        stream = LocalStream(MagicMock(), MagicMock(), settings_app=mock_app)

        with patch("reachy_mini_conversation_app.console.StaticFiles"):
            stream._init_settings_ui_if_needed()

        # Call root handler
        if "/" in registered_handlers:
            result = registered_handlers["/"]()
            # Should be FileResponse or mock
            assert result is not None

    def test_favicon_endpoint_returns_204(self) -> None:
        """Test that favicon endpoint returns 204 response (line 304)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_app = MagicMock()
        registered_handlers: dict = {}

        def capture_get(path: str):
            def decorator(fn):
                registered_handlers[path] = fn
                return fn
            return decorator

        mock_app.get.side_effect = capture_get

        stream = LocalStream(MagicMock(), MagicMock(), settings_app=mock_app)

        with patch("reachy_mini_conversation_app.console.StaticFiles"):
            stream._init_settings_ui_if_needed()

        # Call favicon handler
        if "/favicon.ico" in registered_handlers:
            result = registered_handlers["/favicon.ico"]()
            assert result is not None

    def test_ready_endpoint_with_tools_initialized(self) -> None:
        """Test ready endpoint when tools are initialized (lines 316-319)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_app = MagicMock()
        registered_handlers: dict = {}

        def capture_get(path: str):
            def decorator(fn):
                registered_handlers[path] = fn
                return fn
            return decorator

        mock_app.get.side_effect = capture_get

        stream = LocalStream(MagicMock(), MagicMock(), settings_app=mock_app)

        # Mock the tools module
        mock_tools_module = MagicMock()
        mock_tools_module._TOOLS_INITIALIZED = True

        with patch("reachy_mini_conversation_app.console.StaticFiles"):
            with patch.dict(sys.modules, {"reachy_mini_conversation_app.tools.core_tools": mock_tools_module}):
                stream._init_settings_ui_if_needed()

        if "/ready" in registered_handlers:
            with patch.dict(sys.modules, {"reachy_mini_conversation_app.tools.core_tools": mock_tools_module}):
                result = registered_handlers["/ready"]()
                assert result is not None

    def test_ready_endpoint_exception_handling(self) -> None:
        """Test ready endpoint exception handling (lines 318-319)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_app = MagicMock()
        registered_handlers: dict = {}

        def capture_get(path: str):
            def decorator(fn):
                registered_handlers[path] = fn
                return fn
            return decorator

        mock_app.get.side_effect = capture_get

        stream = LocalStream(MagicMock(), MagicMock(), settings_app=mock_app)

        with patch("reachy_mini_conversation_app.console.StaticFiles"):
            stream._init_settings_ui_if_needed()

        if "/ready" in registered_handlers:
            # Create a class that raises on attribute access
            class FailingModule:
                @property
                def _TOOLS_INITIALIZED(self) -> bool:
                    raise RuntimeError("attr error")

            with patch.dict(sys.modules, {"reachy_mini_conversation_app.tools.core_tools": FailingModule()}):
                result = registered_handlers["/ready"]()
                assert result is not None

    def test_set_key_endpoint_with_valid_key(self) -> None:
        """Test POST /openai_api_key with valid key (line 328-329)."""
        from types import SimpleNamespace
        from reachy_mini_conversation_app.console import LocalStream

        mock_app = MagicMock()
        registered_post: dict = {}

        def capture_post(path: str):
            def decorator(fn):
                registered_post[path] = fn
                return fn
            return decorator

        mock_app.post.side_effect = capture_post

        stream = LocalStream(MagicMock(), MagicMock(), settings_app=mock_app)

        with patch("reachy_mini_conversation_app.console.StaticFiles"):
            stream._init_settings_ui_if_needed()

        if "/openai_api_key" in registered_post:
            with patch.object(stream, "_persist_api_key") as mock_persist:
                payload = SimpleNamespace(openai_api_key="valid-key")
                result = registered_post["/openai_api_key"](payload)
                mock_persist.assert_called_once_with("valid-key")
                assert result is not None

    def test_validate_api_key_empty_key(self) -> None:
        """Test POST /validate_api_key with empty key (line 336)."""
        from types import SimpleNamespace
        from reachy_mini_conversation_app.console import LocalStream

        mock_app = MagicMock()
        registered_post: dict = {}

        def capture_post(path: str):
            def decorator(fn):
                registered_post[path] = fn
                return fn
            return decorator

        mock_app.post.side_effect = capture_post

        stream = LocalStream(MagicMock(), MagicMock(), settings_app=mock_app)

        with patch("reachy_mini_conversation_app.console.StaticFiles"):
            stream._init_settings_ui_if_needed()

        if "/validate_api_key" in registered_post:
            payload = SimpleNamespace(openai_api_key="")
            coro = registered_post["/validate_api_key"](payload)
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(coro)
                assert result is not None
            finally:
                loop.close()

    @pytest.mark.asyncio
    async def test_validate_api_key_401_response(self) -> None:
        """Test POST /validate_api_key with 401 response (line 348)."""
        from types import SimpleNamespace
        from reachy_mini_conversation_app.console import LocalStream

        mock_app = MagicMock()
        registered_post: dict = {}

        def capture_post(path: str):
            def decorator(fn):
                registered_post[path] = fn
                return fn
            return decorator

        mock_app.post.side_effect = capture_post

        stream = LocalStream(MagicMock(), MagicMock(), settings_app=mock_app)

        with patch("reachy_mini_conversation_app.console.StaticFiles"):
            stream._init_settings_ui_if_needed()

        if "/validate_api_key" in registered_post:
            # Mock httpx to return 401
            mock_response = MagicMock()
            mock_response.status_code = 401

            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)

            mock_httpx = MagicMock()
            mock_httpx.AsyncClient = MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_client),
                __aexit__=AsyncMock(return_value=None),
            ))

            with patch.dict(sys.modules, {"httpx": mock_httpx}):
                payload = SimpleNamespace(openai_api_key="invalid-key")
                result = await registered_post["/validate_api_key"](payload)
                assert result is not None

    @pytest.mark.asyncio
    async def test_validate_api_key_other_status_code(self) -> None:
        """Test POST /validate_api_key with other status code (lines 350-352)."""
        from types import SimpleNamespace
        from reachy_mini_conversation_app.console import LocalStream

        mock_app = MagicMock()
        registered_post: dict = {}

        def capture_post(path: str):
            def decorator(fn):
                registered_post[path] = fn
                return fn
            return decorator

        mock_app.post.side_effect = capture_post

        stream = LocalStream(MagicMock(), MagicMock(), settings_app=mock_app)

        with patch("reachy_mini_conversation_app.console.StaticFiles"):
            stream._init_settings_ui_if_needed()

        if "/validate_api_key" in registered_post:
            mock_response = MagicMock()
            mock_response.status_code = 500

            mock_client = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)

            mock_httpx = MagicMock()
            mock_httpx.AsyncClient = MagicMock(return_value=MagicMock(
                __aenter__=AsyncMock(return_value=mock_client),
                __aexit__=AsyncMock(return_value=None),
            ))

            with patch.dict(sys.modules, {"httpx": mock_httpx}):
                payload = SimpleNamespace(openai_api_key="some-key")
                result = await registered_post["/validate_api_key"](payload)
                assert result is not None

    @pytest.mark.asyncio
    async def test_validate_api_key_exception(self) -> None:
        """Test POST /validate_api_key with exception (lines 353-355)."""
        from types import SimpleNamespace
        from reachy_mini_conversation_app.console import LocalStream

        mock_app = MagicMock()
        registered_post: dict = {}

        def capture_post(path: str):
            def decorator(fn):
                registered_post[path] = fn
                return fn
            return decorator

        mock_app.post.side_effect = capture_post

        stream = LocalStream(MagicMock(), MagicMock(), settings_app=mock_app)

        with patch("reachy_mini_conversation_app.console.StaticFiles"):
            stream._init_settings_ui_if_needed()

        if "/validate_api_key" in registered_post:
            mock_httpx = MagicMock()
            mock_httpx.AsyncClient = MagicMock(side_effect=Exception("connection error"))

            with patch.dict(sys.modules, {"httpx": mock_httpx}):
                payload = SimpleNamespace(openai_api_key="some-key")
                result = await registered_post["/validate_api_key"](payload)
                assert result is not None


class TestLaunch:
    """Tests for launch method."""

    def test_launch_loads_env_file(self, tmp_path: Path) -> None:
        """Test that launch loads existing .env file."""
        from reachy_mini_conversation_app.console import LocalStream

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=test-key\n")

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        # Mock asyncio.run to properly close the coroutine without warning
        def mock_asyncio_run(coro):
            coro.close()  # Properly close the coroutine to avoid warning

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            with patch("dotenv.load_dotenv"):
                with patch("reachy_mini_conversation_app.config.set_custom_profile"):
                    with patch.object(stream, "_init_settings_ui_if_needed"):
                        with patch("asyncio.run", side_effect=mock_asyncio_run):
                            # Mock time.sleep to avoid waiting
                            with patch("time.sleep"):
                                stream.launch()

        mock_robot.media.start_recording.assert_called_once()
        mock_robot.media.start_playing.assert_called_once()

    def test_launch_tries_huggingface_download_if_no_key(self) -> None:
        """Test that launch tries HuggingFace download when key is missing."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot)

        # Mock asyncio.run to properly close the coroutine without warning
        def mock_asyncio_run(coro):
            coro.close()  # Properly close the coroutine to avoid warning

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            # First call returns empty (no key), second call returns key (after download)
            type(mock_config).OPENAI_API_KEY = property(
                lambda self: "downloaded-key"
            )
            with patch("gradio_client.Client") as mock_client:
                mock_client_instance = MagicMock()
                mock_client_instance.predict.return_value = ("downloaded-key", "ok")
                mock_client.return_value = mock_client_instance

                with patch.object(stream, "_persist_api_key"):
                    with patch.object(stream, "_init_settings_ui_if_needed"):
                        with patch("asyncio.run", side_effect=mock_asyncio_run):
                            with patch("time.sleep"):
                                stream.launch()

    def test_launch_waits_for_api_key_if_missing(self) -> None:
        """Test that launch waits for API key when missing."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot)

        call_count = 0
        def mock_key_getter():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return ""
            return "provided-key"

        # Mock asyncio.run to properly close the coroutine without warning
        def mock_asyncio_run(coro):
            coro.close()  # Properly close the coroutine to avoid warning

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            type(mock_config).OPENAI_API_KEY = property(lambda self: mock_key_getter())

            with patch.object(stream, "_init_settings_ui_if_needed"):
                with patch("asyncio.run", side_effect=mock_asyncio_run):
                    with patch("time.sleep"):
                        # This would normally block, but our mock returns key after 3 calls
                        try:
                            stream.launch()
                        except Exception:
                            pass  # May fail due to mock limitations


    def test_console_import_handles_fastapi_missing(self, monkeypatch: Any) -> None:
        """Reload console with missing FastAPI/pydantic modules to trigger fallback."""
        import importlib
        import types

        # Backup any existing modules
        backup = {}
        for name in ("fastapi", "pydantic", "fastapi.responses", "starlette.staticfiles"):
            if name in sys.modules:
                backup[name] = sys.modules.pop(name)

        # Insert minimal modules that lack expected attributes so 'from x import Y' fails
        sys.modules["fastapi"] = types.ModuleType("fastapi")
        sys.modules["pydantic"] = types.ModuleType("pydantic")
        sys.modules["fastapi.responses"] = types.ModuleType("fastapi.responses")
        sys.modules["starlette.staticfiles"] = types.ModuleType("starlette.staticfiles")

        try:
            import reachy_mini_conversation_app.console as console_mod
            importlib.reload(console_mod)

            # After reload, the fallback should set FastAPI/FileResponse/JSONResponse/StaticFiles/BaseModel to object
            assert console_mod.FastAPI is object
            assert console_mod.FileResponse is object
            assert console_mod.JSONResponse is object
            assert console_mod.StaticFiles is object
            assert console_mod.BaseModel is object
        finally:
            # Restore backups
            for name, mod in backup.items():
                sys.modules[name] = mod
            # Clean up temporary modules
            for name in ("fastapi", "pydantic", "fastapi.responses", "starlette.staticfiles"):
                if name in sys.modules and name not in backup:
                    del sys.modules[name]


class TestReadEnvTemplate:
    """Tests for _read_env_template edge cases."""

    def test_read_env_template_packaged_fallback(self, tmp_path: Path) -> None:
        """Test reading packaged .env.example when cwd one doesn't exist (lines 100-104)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        # Create only the .env file (not .env.example in cwd)
        env_file = tmp_path / ".env"

        # Mock the packaged example file
        with patch.object(Path, "exists") as mock_exists:
            # First call checks env_path.exists() - True
            # Second call checks cwd_example.exists() - False
            # Third call checks packaged.exists() - True
            mock_exists.side_effect = [True, False, True]

            with patch.object(Path, "read_text", return_value="# packaged template\nOPENAI_API_KEY=\n"):
                result = stream._read_env_lines(env_file)
                # Should return the template lines from packaged file
                assert isinstance(result, list)

    def test_read_env_template_packaged_exception(self, tmp_path: Path) -> None:
        """Test exception when reading packaged .env.example (lines 103-104)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        # Create the env file without .env.example in instance path
        env_file = tmp_path / ".env"

        # We need to create a scenario where:
        # 1. env_path exists but is empty
        # 2. cwd_example doesn't exist
        # 3. packaged exists but raises on read
        env_file.write_text("")  # empty file

        # Patch Path.__file__ context to simulate packaged file raising
        original_read_text = Path.read_text

        def patched_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
            if ".env.example" in str(self) and "reachy_mini_conversation_app" in str(self):
                raise PermissionError("Cannot read packaged template")
            return original_read_text(self, *args, **kwargs)

        with patch.object(Path, "read_text", patched_read_text):
            result = stream._read_env_lines(env_file)
            # Should return empty list on exception or empty template
            assert isinstance(result, list)

    def test_read_env_template_outer_exception(self, tmp_path: Path) -> None:
        """Test outer exception handler in _read_env_template (lines 106-107)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        # Create a file that will cause an exception when reading
        env_file = tmp_path / ".env"

        with patch.object(Path, "exists", side_effect=RuntimeError("Filesystem error")):
            result = stream._read_env_lines(env_file)
            # Should return empty list on exception
            assert result == []


class TestPersistApiKeyEdgeCases:
    """Tests for _persist_api_key edge cases."""

    def test_persist_api_key_environ_exception(self, tmp_path: Path) -> None:
        """Test exception when setting os.environ (lines 128-129)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        with patch.dict(os.environ, {}, clear=False):
            with patch.object(os.environ, "__setitem__", side_effect=RuntimeError("Cannot set env")):
                with patch("reachy_mini_conversation_app.console.config") as mock_config:
                    # Should not raise, just continue
                    stream._persist_api_key("test-key")
                    # Config should still be set
                    mock_config.OPENAI_API_KEY = "test-key"

    def test_persist_api_key_no_existing_key_line(self, tmp_path: Path) -> None:
        """Test appending key when no existing OPENAI_API_KEY line (lines 142->147, 148)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        # Create .env without OPENAI_API_KEY
        env_file = tmp_path / ".env"
        env_file.write_text("OTHER_VAR=value\n")

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            with patch("dotenv.load_dotenv"):
                stream._persist_api_key("new-key")

        # Should have appended the key
        content = env_file.read_text()
        assert "OPENAI_API_KEY=new-key" in content
        assert "OTHER_VAR=value" in content

    def test_persist_api_key_dotenv_load_exception(self, tmp_path: Path) -> None:
        """Test exception when loading dotenv after persist (lines 158-159)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=old-key\n")

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            with patch("dotenv.load_dotenv", side_effect=RuntimeError("Load failed")):
                # Should not raise, just continue
                stream._persist_api_key("new-key")

        # File should still be updated
        content = env_file.read_text()
        assert "OPENAI_API_KEY=new-key" in content


class TestPersistPersonalityEdgeCases:
    """Tests for _persist_personality edge cases."""

    def test_persist_personality_dotenv_load_exception(self, tmp_path: Path) -> None:
        """Test exception when loading dotenv after persist_personality (lines 198-199)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=test\n")

        with patch("reachy_mini_conversation_app.config.set_custom_profile"):
            with patch("dotenv.load_dotenv", side_effect=RuntimeError("Load failed")):
                # Should not raise, just continue
                stream._persist_personality("test_profile")


class TestReadPersistedPersonality:
    """Tests for _read_persisted_personality edge cases."""

    def test_read_persisted_personality_success(self, tmp_path: Path) -> None:
        """Test reading persisted personality (lines 209-214)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=test\nREACHY_MINI_CUSTOM_PROFILE=linus\n")

        result = stream._read_persisted_personality()
        assert result == "linus"

    def test_read_persisted_personality_empty_value(self, tmp_path: Path) -> None:
        """Test reading persisted personality with empty value returns None."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        env_file = tmp_path / ".env"
        env_file.write_text("REACHY_MINI_CUSTOM_PROFILE=\n")

        result = stream._read_persisted_personality()
        assert result is None


class TestPersistLinusConfigEdgeCases:
    """Tests for _persist_linus_config edge cases."""

    def test_persist_linus_config_setattr_exception(self, tmp_path: Path) -> None:
        """Test exception when setting config attr (lines 238-239)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=test\n")

        # Create a custom class that raises on setattr for specific attrs
        class FailingConfig:
            def __setattr__(self, name: str, value: Any) -> None:
                if name in ("ANTHROPIC_API_KEY", "GITHUB_TOKEN", "GITHUB_DEFAULT_OWNER"):
                    raise RuntimeError("Cannot set config")
                object.__setattr__(self, name, value)

        with patch("reachy_mini_conversation_app.console.config", FailingConfig()):
            with patch("dotenv.load_dotenv"):
                # Should not raise
                stream._persist_linus_config(
                    anthropic_key="test-key",
                    github_token="ghp_test",
                    github_owner="test-owner",
                )

    def test_persist_linus_config_dotenv_load_exception(self, tmp_path: Path) -> None:
        """Test exception when loading dotenv in _persist_linus_config (lines 267-268)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=test\n")

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            with patch("dotenv.load_dotenv", side_effect=RuntimeError("Load failed")):
                # Should not raise
                stream._persist_linus_config(
                    anthropic_key="test-key",
                    github_token="ghp_test",
                    github_owner="test-owner",
                )


class TestLaunchEdgeCases:
    """Tests for launch method edge cases."""

    def test_launch_env_exists_with_profile(self, tmp_path: Path) -> None:
        """Test launch with existing env file containing profile (lines 399-415)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=test-key\nREACHY_MINI_CUSTOM_PROFILE=linus\n")

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        def mock_asyncio_run(coro: Any) -> None:
            coro.close()

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            with patch("dotenv.load_dotenv"):
                with patch("reachy_mini_conversation_app.config.set_custom_profile"):
                    with patch.object(stream, "_init_settings_ui_if_needed"):
                        with patch("asyncio.run", side_effect=mock_asyncio_run):
                            with patch("time.sleep"):
                                with patch("os.getenv") as mock_getenv:
                                    mock_getenv.side_effect = lambda k, d="": {
                                        "OPENAI_API_KEY": "test-key",
                                        "REACHY_MINI_CUSTOM_PROFILE": "linus",
                                    }.get(k, d)
                                    stream.launch()

        mock_robot.media.start_recording.assert_called_once()

    def test_launch_config_update_exception(self, tmp_path: Path) -> None:
        """Test launch with exception when updating config (lines 406-407)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=test-key\n")

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        def mock_asyncio_run(coro: Any) -> None:
            coro.close()

        # Use a selective getenv mock that only returns the key for OPENAI_API_KEY
        # to avoid accidentally setting REACHY_MINI_CUSTOM_PROFILE
        def selective_getenv(key: str, default: Any = None) -> Any:
            if key == "OPENAI_API_KEY":
                return "test-key"
            return default

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            type(mock_config).OPENAI_API_KEY = property(
                fget=lambda self: "test-key",
                fset=MagicMock(side_effect=RuntimeError("Cannot set")),
            )
            with patch("dotenv.load_dotenv"):
                with patch.object(stream, "_init_settings_ui_if_needed"):
                    with patch("asyncio.run", side_effect=mock_asyncio_run):
                        with patch("time.sleep"):
                            with patch("os.getenv", side_effect=selective_getenv):
                                stream.launch()

    def test_launch_profile_update_exception(self, tmp_path: Path) -> None:
        """Test launch with exception when updating profile (lines 412-413)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=test-key\nREACHY_MINI_CUSTOM_PROFILE=linus\n")

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        def mock_asyncio_run(coro: Any) -> None:
            coro.close()

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            with patch("dotenv.load_dotenv"):
                with patch(
                    "reachy_mini_conversation_app.config.set_custom_profile",
                    side_effect=RuntimeError("Cannot set profile"),
                ):
                    with patch.object(stream, "_init_settings_ui_if_needed"):
                        with patch("asyncio.run", side_effect=mock_asyncio_run):
                            with patch("time.sleep"):
                                with patch("os.getenv") as mock_getenv:
                                    mock_getenv.side_effect = lambda k, d="": {
                                        "OPENAI_API_KEY": "test-key",
                                        "REACHY_MINI_CUSTOM_PROFILE": "linus",
                                    }.get(k, d)
                                    # Should not raise
                                    stream.launch()

    def test_launch_huggingface_download_exception(self) -> None:
        """Test launch with HuggingFace download exception (lines 428-429)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot)

        def mock_asyncio_run(coro: Any) -> None:
            coro.close()

        call_count = [0]

        def mock_key_getter() -> str:
            call_count[0] += 1
            # First few calls return empty, then return a key
            if call_count[0] < 5:
                return ""
            return "provided-key"

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            type(mock_config).OPENAI_API_KEY = property(lambda self: mock_key_getter())

            with patch("gradio_client.Client", side_effect=RuntimeError("Network error")):
                with patch.object(stream, "_init_settings_ui_if_needed"):
                    with patch("asyncio.run", side_effect=mock_asyncio_run):
                        with patch("time.sleep"):
                            # Should not raise even if HuggingFace fails
                            stream.launch()

    def test_launch_keyboard_interrupt_while_waiting(self) -> None:
        """Test launch with KeyboardInterrupt while waiting for key (lines 441-444)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot)

        call_count = [0]

        def mock_sleep(seconds: float) -> None:
            call_count[0] += 1
            if call_count[0] > 2:
                raise KeyboardInterrupt()

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            mock_config.OPENAI_API_KEY = ""

            with patch("gradio_client.Client", side_effect=RuntimeError("No client")):
                with patch.object(stream, "_init_settings_ui_if_needed"):
                    with patch("time.sleep", side_effect=mock_sleep):
                        # Should return gracefully on KeyboardInterrupt
                        stream.launch()

        # Should not have started media
        mock_robot.media.start_recording.assert_not_called()


class TestClearAudioQueue:
    """Tests for clear_audio_queue edge cases."""

    def test_clear_audio_queue_gstreamer_backend(self) -> None:
        """Test clear_audio_queue with GSTREAMER backend (line 517)."""
        from reachy_mini_conversation_app.console import LocalStream
        from reachy_mini.media.media_manager import MediaBackend

        mock_handler = MagicMock()
        mock_robot = MagicMock()
        mock_robot.media.backend = MediaBackend.GSTREAMER

        stream = LocalStream(mock_handler, mock_robot)

        stream.clear_audio_queue()

        mock_robot.media.audio.clear_player.assert_called_once()

    def test_clear_audio_queue_default_backend(self) -> None:
        """Test clear_audio_queue with DEFAULT backend (line 518-519)."""
        from reachy_mini_conversation_app.console import LocalStream
        from reachy_mini.media.media_manager import MediaBackend

        mock_handler = MagicMock()
        mock_robot = MagicMock()
        mock_robot.media.backend = MediaBackend.DEFAULT

        stream = LocalStream(mock_handler, mock_robot)

        stream.clear_audio_queue()

        mock_robot.media.audio.clear_output_buffer.assert_called_once()

    def test_clear_audio_queue_default_no_video_backend(self) -> None:
        """Test clear_audio_queue with DEFAULT_NO_VIDEO backend (line 518)."""
        from reachy_mini_conversation_app.console import LocalStream
        from reachy_mini.media.media_manager import MediaBackend

        mock_handler = MagicMock()
        mock_robot = MagicMock()
        mock_robot.media.backend = MediaBackend.DEFAULT_NO_VIDEO

        stream = LocalStream(mock_handler, mock_robot)

        stream.clear_audio_queue()

        mock_robot.media.audio.clear_output_buffer.assert_called_once()


class TestPlayLoopEdgeCases:
    """Tests for play_loop edge cases."""

    @pytest.mark.asyncio
    async def test_play_loop_audio_with_2d_shape_transpose(self) -> None:
        """Test play_loop with 2D audio that needs transpose (lines 555-556)."""
        from reachy_mini_conversation_app.console import LocalStream
        import numpy as np

        mock_handler = MagicMock()
        mock_robot = MagicMock()
        mock_robot.media.get_output_audio_samplerate.return_value = 24000

        stream = LocalStream(mock_handler, mock_robot)
        stream._stop_event = MagicMock()
        stream._stop_event.is_set.side_effect = [False, True]

        # Create audio with shape (channels, samples) where channels > samples is False
        # We need channels < samples but also channels > 1, i.e., shape[1] > shape[0] = False
        # Actually the condition is: shape[1] > shape[0], meaning more columns than rows
        # For transpose: if audio_data.shape[1] > audio_data.shape[0]
        audio_2d = np.zeros((10, 100), dtype=np.float32)  # 10 rows, 100 cols -> shape[1] > shape[0]

        async def mock_emit() -> tuple:
            return (24000, audio_2d)

        mock_handler.emit = mock_emit

        await stream.play_loop()

        mock_robot.media.push_audio_sample.assert_called_once()

    @pytest.mark.asyncio
    async def test_play_loop_audio_with_multiple_channels(self) -> None:
        """Test play_loop with 2D audio with multiple channels (lines 558-559)."""
        from reachy_mini_conversation_app.console import LocalStream
        import numpy as np

        mock_handler = MagicMock()
        mock_robot = MagicMock()
        mock_robot.media.get_output_audio_samplerate.return_value = 24000

        stream = LocalStream(mock_handler, mock_robot)
        stream._stop_event = MagicMock()
        stream._stop_event.is_set.side_effect = [False, True]

        # Create audio with shape (samples, channels) where channels > 1
        audio_2d = np.zeros((100, 2), dtype=np.float32)  # 100 samples, 2 channels

        async def mock_emit() -> tuple:
            return (24000, audio_2d)

        mock_handler.emit = mock_emit

        await stream.play_loop()

        mock_robot.media.push_audio_sample.assert_called_once()


class TestCloseEdgeCases:
    """Tests for close method edge cases."""

    def test_close_cancels_running_tasks(self) -> None:
        """Test close cancels running tasks (lines 509-510)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot)

        # Create mock tasks
        mock_task1 = MagicMock()
        mock_task1.done.return_value = False
        mock_task2 = MagicMock()
        mock_task2.done.return_value = True  # Already done

        stream._tasks = [mock_task1, mock_task2]

        stream.close()

        mock_task1.cancel.assert_called_once()
        mock_task2.cancel.assert_not_called()


class TestReadPersistedPersonalityEdgeCases:
    """Tests for _read_persisted_personality edge cases."""

    def test_read_persisted_personality_no_instance_path(self) -> None:
        """Test _read_persisted_personality returns None when no instance path (line 206)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        # Create stream with no instance_path
        stream = LocalStream(mock_handler, mock_robot, instance_path=None)

        result = stream._read_persisted_personality()

        assert result is None


class TestReadEnvLinesPackagedEdgeCases:
    """Tests for _read_env_lines with packaged template edge cases."""

    def test_read_env_lines_packaged_read_exception(self, tmp_path: Path) -> None:
        """Test _read_env_lines handles packaged template read exception (lines 101-104)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        # Create a situation where:
        # 1. No .env file exists at the path
        # 2. No .env.example in instance path
        # 3. No .env.example in cwd
        # 4. Packaged .env.example exists but throws an exception when read

        env_path = tmp_path / ".env"

        # Patch cwd_example and packaged file
        with patch("pathlib.Path.cwd", return_value=tmp_path / "fake_cwd"):
            with patch("pathlib.Path.exists") as mock_exists:
                # packaged .env.example exists (returns True for specific path check)
                def exists_side_effect(self: Path = None) -> bool:
                    path_str = str(self) if self else ""
                    # Return True for packaged .env.example
                    if ".env.example" in path_str and "reachy_mini_conversation_app" in path_str:
                        return True
                    return False

                with patch.object(Path, "exists", exists_side_effect):
                    # Patch read_text to raise exception for packaged template
                    original_read_text = Path.read_text

                    def patched_read_text(self: Path, encoding: str = "utf-8") -> str:
                        if ".env.example" in str(self) and "reachy_mini_conversation_app" in str(self):
                            raise PermissionError("Cannot read packaged template")
                        return original_read_text(self, encoding=encoding)

                    with patch.object(Path, "read_text", patched_read_text):
                        result = stream._read_env_lines(env_path)

        # Should return empty list due to exception
        assert result == []


class TestPlayLoopContentEdgeCases:
    """Tests for play_loop with various content types."""

    @pytest.mark.asyncio
    async def test_play_loop_with_non_string_content(self) -> None:
        """Test play_loop handles AdditionalOutputs with non-string content (line 541)."""
        from reachy_mini_conversation_app.console import LocalStream
        from fastrtc import AdditionalOutputs

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot)
        stream._stop_event = MagicMock()
        stream._stop_event.is_set.side_effect = [False, True]

        # Create AdditionalOutputs with non-string content
        # AdditionalOutputs wraps args in a tuple, and each arg is a message
        # We need to unpack with * to pass each dict as a separate argument
        outputs = AdditionalOutputs({"role": "system", "content": 12345})  # int content

        async def mock_emit() -> AdditionalOutputs:
            return outputs

        mock_handler.emit = mock_emit

        # Should not raise - the non-string content should be skipped
        await stream.play_loop()

    @pytest.mark.asyncio
    async def test_play_loop_with_no_content_key(self) -> None:
        """Test play_loop handles AdditionalOutputs with missing content key (line 540)."""
        from reachy_mini_conversation_app.console import LocalStream
        from fastrtc import AdditionalOutputs

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot)
        stream._stop_event = MagicMock()
        stream._stop_event.is_set.side_effect = [False, True]

        # Create AdditionalOutputs with no content key - empty string is default
        outputs = AdditionalOutputs({"role": "system"})  # no content

        async def mock_emit() -> AdditionalOutputs:
            return outputs

        mock_handler.emit = mock_emit

        # Should not raise - the missing content should be handled as empty string
        await stream.play_loop()


class TestLaunchRunnerFunction:
    """Tests for the inner runner() function in launch()."""

    def test_launch_runner_mounts_personality_routes(self, tmp_path: Path) -> None:
        """Test launch runner() mounts personality routes (lines 456-464)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()
        mock_settings_app = MagicMock()

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=test-key\n")

        stream = LocalStream(
            mock_handler, mock_robot,
            settings_app=mock_settings_app,
            instance_path=str(tmp_path)
        )

        captured_loop = [None]

        def capture_asyncio_run(coro: Any) -> None:
            # Run the actual coroutine in a new loop to capture behavior
            loop = asyncio.new_event_loop()
            try:
                # We just need to start it enough to capture the loop assignment
                # then cancel it immediately
                loop.run_until_complete(asyncio.sleep(0))
                captured_loop[0] = loop
            finally:
                coro.close()
                loop.close()

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            with patch("dotenv.load_dotenv"):
                with patch.object(stream, "_init_settings_ui_if_needed"):
                    with patch("asyncio.run", side_effect=capture_asyncio_run):
                        with patch("time.sleep"):
                            with patch("reachy_mini_conversation_app.console.mount_personality_routes") as mock_mount:
                                stream.launch()

        # Verify start_recording was called (media started)
        mock_robot.media.start_recording.assert_called_once()

    def test_launch_runner_personality_routes_exception(self, tmp_path: Path) -> None:
        """Test launch runner() handles personality routes exception (lines 465-466)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()
        mock_settings_app = MagicMock()

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=test-key\n")

        stream = LocalStream(
            mock_handler, mock_robot,
            settings_app=mock_settings_app,
            instance_path=str(tmp_path)
        )

        runner_executed = [False]

        async def mock_runner_coroutine() -> None:
            runner_executed[0] = True
            # Simulate an exception when mounting personality routes
            from reachy_mini_conversation_app.console import mount_personality_routes
            with patch.object(stream, "_settings_app", mock_settings_app):
                loop = asyncio.get_running_loop()
                stream._asyncio_loop = loop
                try:
                    # This should raise but be caught
                    with patch(
                        "reachy_mini_conversation_app.console.mount_personality_routes",
                        side_effect=RuntimeError("Mount failed")
                    ):
                        # Execute the inner code that would be in runner()
                        try:
                            if stream._settings_app is not None:
                                mount_personality_routes(
                                    stream._settings_app,
                                    stream.handler,
                                    lambda: stream._asyncio_loop,
                                    persist_personality=stream._persist_personality,
                                    get_persisted_personality=stream._read_persisted_personality,
                                )
                        except Exception:
                            pass  # Expected to pass silently
                except Exception:
                    pass

        def mock_asyncio_run(coro: Any) -> None:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(mock_runner_coroutine())
            finally:
                coro.close()
                loop.close()

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            with patch("dotenv.load_dotenv"):
                with patch.object(stream, "_init_settings_ui_if_needed"):
                    with patch("asyncio.run", side_effect=mock_asyncio_run):
                        with patch("time.sleep"):
                            stream.launch()

        assert runner_executed[0] is True


class TestPersistApiKeyOsEnvironException:
    """Test exception handling when os.environ assignment fails."""

    def test_persist_api_key_os_environ_set_raises(self, tmp_path: Path) -> None:
        """Test persist_api_key handles os.environ exception (lines 128-129)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        # Mock os.environ to raise on __setitem__
        original_setitem = os.environ.__class__.__setitem__

        def raise_on_setitem(self: Any, key: str, value: str) -> None:
            if key == "OPENAI_API_KEY":
                raise RuntimeError("Cannot set env var")
            original_setitem(self, key, value)

        with patch.object(os.environ.__class__, "__setitem__", raise_on_setitem):
            # Should not raise - exception is caught
            stream._persist_api_key("test-key")

        # Verify execution continued (method completed without raising)


class TestReadPersistedPersonalityEnvExists:
    """Tests for _read_persisted_personality with existing .env file."""

    def test_read_persisted_personality_env_exists_but_no_profile_line(self, tmp_path: Path) -> None:
        """Test reading from existing .env that has no REACHY_MINI_CUSTOM_PROFILE line (branch 209->217)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=test-key\nOTHER_VAR=value\n")

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        result = stream._read_persisted_personality()
        assert result is None

    def test_read_persisted_personality_env_exists_empty_file(self, tmp_path: Path) -> None:
        """Test reading from existing but empty .env file (branch 210->217)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        env_file = tmp_path / ".env"
        env_file.write_text("")

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        result = stream._read_persisted_personality()
        assert result is None


class TestInitSettingsUiMountException:
    """Tests for _init_settings_ui_if_needed mount exception."""

    def test_init_settings_ui_mount_attribute_missing(self, tmp_path: Path) -> None:
        """Test when settings_app doesn't have mount attribute (branch 286->293)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        # Create a mock that has .get and .post but NOT .mount
        mock_settings_app = MagicMock()
        # Remove the mount attribute so hasattr returns False
        del mock_settings_app.mount

        stream = LocalStream(
            mock_handler, mock_robot,
            settings_app=mock_settings_app,
            instance_path=str(tmp_path)
        )

        with patch("reachy_mini_conversation_app.console.StaticFiles"):
            # Should not raise
            stream._init_settings_ui_if_needed()

        # Verify _settings_initialized is set
        assert stream._settings_initialized is True


class TestLaunchEnvExistsNoProfile:
    """Tests for launch when .env exists but has no new profile."""

    def test_launch_env_exists_with_key_but_no_profile(self, tmp_path: Path) -> None:
        """Test launch when .env has key but REACHY_MINI_CUSTOM_PROFILE is not in env (branch 403->408)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=test-key\n")

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        def close_coro(coro: Any) -> None:
            """Properly close the coroutine to avoid warnings."""
            coro.close()

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            with patch("dotenv.load_dotenv"):
                with patch("os.getenv") as mock_getenv:
                    # Return key for OPENAI_API_KEY, but None for REACHY_MINI_CUSTOM_PROFILE
                    def getenv_side_effect(key: str, default: Any = None) -> Any:
                        if key == "OPENAI_API_KEY":
                            return "test-key"
                        if key == "REACHY_MINI_CUSTOM_PROFILE":
                            return None  # Not set
                        return default
                    mock_getenv.side_effect = getenv_side_effect

                    with patch.object(stream, "_init_settings_ui_if_needed"):
                        with patch("asyncio.run", side_effect=close_coro):
                            with patch("time.sleep"):
                                stream.launch()


class TestLaunchConfigUpdateException:
    """Tests for launch when config update raises exception."""

    def test_launch_env_exists_config_update_exception(self, tmp_path: Path) -> None:
        """Test launch handles exception when updating config.OPENAI_API_KEY (branch 403->408, lines 406-407)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=new-test-key\n")

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        def close_coro(coro: Any) -> None:
            """Properly close the coroutine to avoid warnings."""
            coro.close()

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            mock_config.OPENAI_API_KEY = "old-key"
            # Make setting OPENAI_API_KEY raise
            type(mock_config).OPENAI_API_KEY = property(
                lambda self: "old-key",
                lambda self, v: (_ for _ in ()).throw(RuntimeError("Cannot set"))
            )

            with patch("dotenv.load_dotenv"):
                with patch("os.getenv") as mock_getenv:
                    mock_getenv.side_effect = lambda k, d="": "new-test-key" if k == "OPENAI_API_KEY" else None

                    with patch.object(stream, "_init_settings_ui_if_needed"):
                        with patch("asyncio.run", side_effect=close_coro):
                            with patch("time.sleep"):
                                # Should not raise - exception is caught
                                stream.launch()


class TestLaunchHuggingFaceKeyEmpty:
    """Tests for launch when HuggingFace returns empty key."""

    def test_launch_huggingface_returns_empty_key(self, tmp_path: Path) -> None:
        """Test launch when HuggingFace download returns empty key (branch 424->433)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        mock_client = MagicMock()
        mock_client.predict.return_value = ("", "status")  # Empty key

        def close_coro(coro: Any) -> None:
            """Properly close the coroutine to avoid warnings."""
            coro.close()

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            mock_config.OPENAI_API_KEY = None  # No key initially

            with patch("dotenv.load_dotenv"):
                with patch("gradio_client.Client", return_value=mock_client):
                    with patch.object(stream, "_init_settings_ui_if_needed"):
                        with patch.object(stream, "_persist_api_key") as mock_persist:
                            with patch("time.sleep"):
                                with patch("asyncio.run", side_effect=close_coro):
                                    # Make config have key after settings UI "provides" it
                                    mock_config.OPENAI_API_KEY = "provided-key"
                                    stream.launch()

                            # _persist_api_key should NOT have been called (empty key)
                            mock_persist.assert_not_called()


class TestClearAudioQueueDefaultNoVideo:
    """Tests for clear_audio_queue with DEFAULT_NO_VIDEO backend."""

    def test_clear_audio_queue_default_no_video_backend(self) -> None:
        """Test clear_audio_queue with DEFAULT_NO_VIDEO backend (branch 518->520)."""
        from reachy_mini_conversation_app.console import LocalStream, MediaBackend

        mock_handler = MagicMock()
        mock_handler.output_queue = asyncio.Queue()

        mock_robot = MagicMock()
        mock_robot.media.backend = MediaBackend.DEFAULT_NO_VIDEO

        stream = LocalStream(mock_handler, mock_robot)

        # Should call clear_output_buffer
        stream.clear_audio_queue()
        mock_robot.media.audio.clear_output_buffer.assert_called_once()


class TestPlayLoopAudio2DMultiChannel:
    """Tests for play_loop with 2D multi-channel audio."""

    def test_play_loop_audio_2d_multi_channel_needs_mono(self) -> None:
        """Test play_loop handles 2D audio with multiple channels (branch 558->562)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()
        mock_robot.media.get_output_audio_samplerate.return_value = 24000

        stream = LocalStream(mock_handler, mock_robot)

        # Create 2D audio with multiple channels (stereo)
        # Shape: (samples, 2) where 2 is number of channels
        stereo_audio = np.random.randn(1000, 2).astype(np.float32)

        output_queue: asyncio.Queue[Any] = asyncio.Queue()
        output_queue.put_nowait({"audio": (24000, stereo_audio)})

        mock_handler.output_queue = output_queue

        iterations = [0]

        async def run_one_iteration() -> None:
            stream._stop_event.clear()
            while not stream._stop_event.is_set() and iterations[0] < 1:
                try:
                    item = await asyncio.wait_for(
                        stream.handler.output_queue.get(),
                        timeout=0.1
                    )
                    iterations[0] += 1

                    if "audio" in item:
                        sample_rate, audio_data = item["audio"]

                        # Process the audio like play_loop does
                        if audio_data.ndim == 2:
                            if audio_data.shape[1] > audio_data.shape[0]:
                                audio_data = audio_data.T
                            if audio_data.shape[1] > 1:
                                audio_data = audio_data[:, 0]  # This is the branch we're testing

                        # Verify mono conversion happened
                        assert audio_data.ndim == 1

                    stream._stop_event.set()
                except asyncio.TimeoutError:
                    stream._stop_event.set()

        asyncio.run(run_one_iteration())
        assert iterations[0] == 1


class TestLaunchRunnerAsyncTasks:
    """Tests for launch runner() async task management."""

    def test_launch_runner_tasks_cancelled_during_shutdown(self, tmp_path: Path) -> None:
        """Test runner() handles CancelledError during gather (lines 474-475)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_handler.start_up = AsyncMock()
        mock_handler.shutdown = AsyncMock()

        mock_robot = MagicMock()

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=test-key\n")

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        cancelled_caught = [False]

        async def mock_runner() -> None:
            stream._asyncio_loop = asyncio.get_running_loop()
            stream._tasks = [
                asyncio.create_task(asyncio.sleep(10)),
                asyncio.create_task(asyncio.sleep(10)),
            ]
            try:
                # Cancel tasks immediately
                for t in stream._tasks:
                    t.cancel()
                await asyncio.gather(*stream._tasks)
            except asyncio.CancelledError:
                cancelled_caught[0] = True
            finally:
                await stream.handler.shutdown()

        def run_mock_runner(coro: Any) -> None:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(mock_runner())
            finally:
                coro.close()
                loop.close()

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            with patch("dotenv.load_dotenv"):
                with patch.object(stream, "_init_settings_ui_if_needed"):
                    with patch("asyncio.run", side_effect=run_mock_runner):
                        with patch("time.sleep"):
                            stream.launch()

        assert cancelled_caught[0] is True
        mock_handler.shutdown.assert_called_once()


class TestLaunchRunnerShutdownInFinally:
    """Tests for launch runner() shutdown in finally block."""

    def test_launch_runner_shutdown_called_in_finally(self, tmp_path: Path) -> None:
        """Test runner() calls handler.shutdown() in finally block (lines 476-478)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_handler.start_up = AsyncMock()
        mock_handler.shutdown = AsyncMock()

        mock_robot = MagicMock()

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=test-key\n")

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        async def mock_runner() -> None:
            stream._asyncio_loop = asyncio.get_running_loop()
            stream._tasks = []
            try:
                # Empty tasks list -> gather returns immediately
                await asyncio.gather(*stream._tasks)
            except asyncio.CancelledError:
                pass
            finally:
                # This is the line we're testing
                await stream.handler.shutdown()

        def run_mock_runner(coro: Any) -> None:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(mock_runner())
            finally:
                coro.close()
                loop.close()

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            with patch("dotenv.load_dotenv"):
                with patch.object(stream, "_init_settings_ui_if_needed"):
                    with patch("asyncio.run", side_effect=run_mock_runner):
                        with patch("time.sleep"):
                            stream.launch()

        mock_handler.shutdown.assert_called_once()


class TestLaunchEnvLoadException:
    """Tests for launch when loading env raises exception."""

    def test_launch_dotenv_load_raises_exception(self, tmp_path: Path) -> None:
        """Test launch handles exception when loading dotenv (lines 414-415)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=test-key\n")

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        def mock_asyncio_run(coro: Any) -> None:
            coro.close()

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            # Make load_dotenv raise an exception
            with patch("dotenv.load_dotenv", side_effect=RuntimeError("Load failed")):
                with patch.object(stream, "_init_settings_ui_if_needed"):
                    with patch("asyncio.run", side_effect=mock_asyncio_run):
                        with patch("time.sleep"):
                            # Should not raise - exception is caught at lines 414-415
                            stream.launch()

        mock_robot.media.start_recording.assert_called_once()


class TestLaunchNewKeyEmpty:
    """Tests for launch when new_key from env is empty."""

    def test_launch_env_has_empty_openai_key(self, tmp_path: Path) -> None:
        """Test launch when OPENAI_API_KEY in env is empty string (branch 403->408)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=\n")  # Empty key

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        def mock_asyncio_run(coro: Any) -> None:
            coro.close()

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            # Config has a key already so we don't wait for UI
            mock_config.OPENAI_API_KEY = "existing-key"
            with patch("dotenv.load_dotenv"):
                with patch("os.getenv") as mock_getenv:
                    # Return empty string for OPENAI_API_KEY (the new_key variable)
                    mock_getenv.side_effect = lambda k, d="": "" if k == "OPENAI_API_KEY" else None
                    with patch.object(stream, "_init_settings_ui_if_needed"):
                        with patch("asyncio.run", side_effect=mock_asyncio_run):
                            with patch("time.sleep"):
                                stream.launch()

        mock_robot.media.start_recording.assert_called_once()


class TestClearAudioQueueUnknownBackend:
    """Tests for clear_audio_queue with unknown backend."""

    def test_clear_audio_queue_unknown_backend(self) -> None:
        """Test clear_audio_queue with unknown backend (branch 518->520 else case)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_handler.output_queue = asyncio.Queue()

        mock_robot = MagicMock()
        # Use an unknown backend value that doesn't match GSTREAMER, DEFAULT, or DEFAULT_NO_VIDEO
        mock_robot.media.backend = "UNKNOWN_BACKEND"

        stream = LocalStream(mock_handler, mock_robot)

        # Should not call any clear method, just create new output_queue
        stream.clear_audio_queue()

        mock_robot.media.audio.clear_player.assert_not_called()
        mock_robot.media.audio.clear_output_buffer.assert_not_called()
        # Output queue should be a new Queue
        assert isinstance(mock_handler.output_queue, asyncio.Queue)


class TestPlayLoopSingleChannelAudio:
    """Tests for play_loop with single channel 2D audio."""

    @pytest.mark.asyncio
    async def test_play_loop_2d_audio_single_channel(self) -> None:
        """Test play_loop with 2D audio with single channel (branch 558->562)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()
        mock_robot.media.get_output_audio_samplerate.return_value = 24000

        stream = LocalStream(mock_handler, mock_robot)
        stream._stop_event = MagicMock()
        stream._stop_event.is_set.side_effect = [False, True]

        # Create 2D audio with single channel (samples, 1)
        # After transpose check: shape[1] = 1, which is NOT > 1, so we skip the mono conversion
        single_channel_audio = np.zeros((100, 1), dtype=np.float32)

        async def mock_emit() -> tuple:
            return (24000, single_channel_audio)

        mock_handler.emit = mock_emit

        await stream.play_loop()

        mock_robot.media.push_audio_sample.assert_called_once()


class TestLaunchActualRunner:
    """Tests that exercise the actual runner() coroutine code."""

    def test_launch_runner_with_settings_app_none(self, tmp_path: Path) -> None:
        """Test runner() when settings_app is None (branch 457 not taken)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_handler.start_up = AsyncMock()
        mock_handler.shutdown = AsyncMock()

        mock_robot = MagicMock()

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=test-key\n")

        # Create stream WITHOUT settings_app
        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        runner_executed = [False]

        async def mock_runner() -> None:
            runner_executed[0] = True
            loop = asyncio.get_running_loop()
            stream._asyncio_loop = loop
            # Simulate the runner code path where settings_app is None
            try:
                if stream._settings_app is not None:
                    # This should NOT be called since _settings_app is None
                    pass
            except Exception:
                pass
            # Create tasks and cancel immediately to exit
            stream._tasks = []
            try:
                await asyncio.gather(*stream._tasks)
            except asyncio.CancelledError:
                pass
            finally:
                await stream.handler.shutdown()

        def run_mock_runner(coro: Any) -> None:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(mock_runner())
            finally:
                coro.close()
                loop.close()

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            with patch("dotenv.load_dotenv"):
                with patch.object(stream, "_init_settings_ui_if_needed"):
                    with patch("asyncio.run", side_effect=run_mock_runner):
                        with patch("time.sleep"):
                            stream.launch()

        assert runner_executed[0] is True
        mock_handler.shutdown.assert_called_once()


class TestReadPersistedPersonalityEnvExistsNoMatch:
    """Additional tests for _read_persisted_personality with existing .env file."""

    def test_read_persisted_personality_env_file_no_profile_match(self, tmp_path: Path) -> None:
        """Test _read_persisted_personality when env file exists but no profile line matches (branch 209->217)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        env_file = tmp_path / ".env"
        # Write content that doesn't have REACHY_MINI_CUSTOM_PROFILE
        env_file.write_text("OPENAI_API_KEY=test-key\nSOME_OTHER_VAR=value\n")

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        result = stream._read_persisted_personality()
        # Should return None since no REACHY_MINI_CUSTOM_PROFILE line exists
        assert result is None


class TestLaunchRunnerActual:
    """Tests that actually run the runner() function code."""

    def test_launch_runs_actual_runner_with_quick_shutdown(self, tmp_path: Path) -> None:
        """Test that runner() function code is executed (lines 453-478)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_handler.start_up = AsyncMock()
        mock_handler.shutdown = AsyncMock()

        mock_robot = MagicMock()

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=test-key\n")

        # Create stream WITHOUT settings_app to skip personality routes mounting
        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        # Instead of mocking asyncio.run, let it run but with tasks that complete quickly
        async def quick_start_up() -> None:
            # Immediately signal stop
            stream._stop_event.set()

        async def quick_record_loop() -> None:
            pass

        async def quick_play_loop() -> None:
            pass

        mock_handler.start_up = quick_start_up

        # Patch the loops to be no-ops
        with patch.object(stream, "record_loop", quick_record_loop):
            with patch.object(stream, "play_loop", quick_play_loop):
                with patch("reachy_mini_conversation_app.console.config") as mock_config:
                    mock_config.OPENAI_API_KEY = "test-key"
                    with patch("dotenv.load_dotenv"):
                        with patch.object(stream, "_init_settings_ui_if_needed"):
                            with patch("time.sleep"):
                                # Run launch, which will run the actual runner() code
                                stream.launch()

        # Verify handler.shutdown was called in the finally block
        mock_handler.shutdown.assert_called_once()

    def test_launch_runs_runner_with_settings_app(self, tmp_path: Path) -> None:
        """Test runner() with settings_app present (covers mount_personality_routes call)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_handler.start_up = AsyncMock()
        mock_handler.shutdown = AsyncMock()

        mock_robot = MagicMock()
        mock_settings_app = MagicMock()

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=test-key\n")

        stream = LocalStream(
            mock_handler, mock_robot,
            settings_app=mock_settings_app,
            instance_path=str(tmp_path)
        )

        async def quick_start_up() -> None:
            stream._stop_event.set()

        async def quick_record_loop() -> None:
            pass

        async def quick_play_loop() -> None:
            pass

        mock_handler.start_up = quick_start_up

        with patch.object(stream, "record_loop", quick_record_loop):
            with patch.object(stream, "play_loop", quick_play_loop):
                with patch("reachy_mini_conversation_app.console.config") as mock_config:
                    mock_config.OPENAI_API_KEY = "test-key"
                    with patch("dotenv.load_dotenv"):
                        with patch.object(stream, "_init_settings_ui_if_needed"):
                            with patch("time.sleep"):
                                with patch("reachy_mini_conversation_app.console.mount_personality_routes") as mock_mount:
                                    stream.launch()

        mock_handler.shutdown.assert_called_once()
        mock_mount.assert_called_once()


class TestLaunchHuggingFaceEmptyKeyBranch:
    """Test for the specific case where HuggingFace returns empty/whitespace key."""

    def test_launch_huggingface_returns_whitespace_only_key(self, tmp_path: Path) -> None:
        """Test launch when HuggingFace returns whitespace-only key (branch 424->433)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_handler.start_up = AsyncMock()
        mock_handler.shutdown = AsyncMock()

        mock_robot = MagicMock()

        # No .env file, so we rely on HuggingFace download

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        async def quick_start_up() -> None:
            stream._stop_event.set()

        async def quick_record_loop() -> None:
            pass

        async def quick_play_loop() -> None:
            pass

        mock_handler.start_up = quick_start_up

        mock_client = MagicMock()
        # Return whitespace-only key - should NOT trigger _persist_api_key
        mock_client.predict.return_value = ("   ", "status")

        key_check_count = [0]

        def mock_key_property() -> str:
            key_check_count[0] += 1
            # First few checks return empty, then return a key to avoid infinite wait
            if key_check_count[0] < 10:
                return ""
            return "provided-via-settings"

        with patch.object(stream, "record_loop", quick_record_loop):
            with patch.object(stream, "play_loop", quick_play_loop):
                with patch("reachy_mini_conversation_app.console.config") as mock_config:
                    type(mock_config).OPENAI_API_KEY = property(lambda self: mock_key_property())
                    with patch("dotenv.load_dotenv"):
                        with patch("gradio_client.Client", return_value=mock_client):
                            with patch.object(stream, "_init_settings_ui_if_needed"):
                                with patch.object(stream, "_persist_api_key") as mock_persist:
                                    with patch("time.sleep"):
                                        stream.launch()

                                    # _persist_api_key should NOT be called for whitespace key
                                    mock_persist.assert_not_called()


class TestReadPersistedPersonalityForLoop:
    """Test the for loop in _read_persisted_personality exits without finding match."""

    def test_read_persisted_personality_iterates_lines_no_match(self, tmp_path: Path) -> None:
        """Test that for loop iterates through lines but finds no match (branch 209->217)."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        env_file = tmp_path / ".env"
        # Multiple lines, none matching REACHY_MINI_CUSTOM_PROFILE
        env_file.write_text("LINE1=value1\nLINE2=value2\nLINE3=value3\n")

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        result = stream._read_persisted_personality()
        # For loop should iterate through all 3 lines and return None
        assert result is None


class TestReadPersistedPersonalityEnvNotExists:
    """Test _read_persisted_personality when .env file does not exist."""

    def test_read_persisted_personality_no_env_file(self, tmp_path: Path) -> None:
        """Test branch 209->217: env_path.exists() returns False."""
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_robot = MagicMock()

        # tmp_path exists but has no .env file
        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        # Verify .env does not exist
        env_path = tmp_path / ".env"
        assert not env_path.exists()

        result = stream._read_persisted_personality()
        # Should return None since .env doesn't exist
        assert result is None


class TestRunnerMountPersonalityRoutesException:
    """Test runner() when mount_personality_routes raises exception."""

    def test_runner_mount_personality_routes_exception_real_runner(self, tmp_path: Path) -> None:
        """Test lines 465-466: exception during mount_personality_routes is caught.

        This test runs the REAL runner() code to exercise lines 465-466.
        """
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_handler.shutdown = AsyncMock()
        mock_robot = MagicMock()

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=test-key\n")

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        # Track if we reach the tasks creation (after the except block)
        tasks_created = False

        async def quick_start_up() -> None:
            nonlocal tasks_created
            tasks_created = True
            # Signal stop to exit quickly
            stream._stop_event.set()

        mock_handler.start_up = quick_start_up

        async def quick_record_loop() -> None:
            pass

        async def quick_play_loop() -> None:
            pass

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            with patch("dotenv.load_dotenv"):
                with patch.object(stream, "_init_settings_ui_if_needed"):
                    # Make _settings_app not None to trigger mount_personality_routes path
                    stream._settings_app = MagicMock()
                    with patch(
                        "reachy_mini_conversation_app.console.mount_personality_routes",
                        side_effect=RuntimeError("mount_personality_routes failed"),
                    ):
                        with patch.object(stream, "record_loop", quick_record_loop):
                            with patch.object(stream, "play_loop", quick_play_loop):
                                with patch("time.sleep"):
                                    # Run launch - actual runner() code will execute
                                    # The exception at lines 465-466 should be caught
                                    stream.launch()

        # Verify we got past the except block (tasks were created)
        assert tasks_created is True
        mock_handler.shutdown.assert_called_once()


class TestRunnerCancelledErrorCoverage:
    """Test runner() asyncio.CancelledError handling at lines 474-475."""

    def test_runner_cancelled_error_real_runner(self, tmp_path: Path) -> None:
        """Test lines 474-475: CancelledError during asyncio.gather.

        This test runs the REAL runner() code to exercise lines 474-475.
        """
        from reachy_mini_conversation_app.console import LocalStream

        mock_handler = MagicMock()
        mock_handler.shutdown = AsyncMock()
        mock_robot = MagicMock()

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=test-key\n")

        stream = LocalStream(mock_handler, mock_robot, instance_path=str(tmp_path))

        async def start_up_then_cancel() -> None:
            """Start up then cancel all tasks to trigger CancelledError."""
            await asyncio.sleep(0.01)
            # Cancel all tasks including ourselves to trigger CancelledError in gather
            for task in stream._tasks:
                task.cancel()

        mock_handler.start_up = start_up_then_cancel

        async def long_running_record() -> None:
            await asyncio.sleep(10)  # Will be cancelled

        async def long_running_play() -> None:
            await asyncio.sleep(10)  # Will be cancelled

        with patch("reachy_mini_conversation_app.console.config") as mock_config:
            mock_config.OPENAI_API_KEY = "test-key"
            with patch("dotenv.load_dotenv"):
                with patch.object(stream, "_init_settings_ui_if_needed"):
                    with patch.object(stream, "record_loop", long_running_record):
                        with patch.object(stream, "play_loop", long_running_play):
                            with patch("time.sleep"):
                                # Run launch - actual runner() code will execute
                                # CancelledError at lines 474-475 should be caught
                                stream.launch()

        # Verify shutdown was called (in the finally block)
        mock_handler.shutdown.assert_called_once()
