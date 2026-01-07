"""Unit tests for the prompts module."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestExpandPromptIncludes:
    """Tests for _expand_prompt_includes function."""

    def test_expand_simple_placeholder(self, tmp_path: Path) -> None:
        """Test expanding a simple placeholder."""
        from reachy_mini_conversation_app import prompts

        # Create a temp prompts directory with a template
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "greeting.txt").write_text("Hello, I am Reachy!")

        # Patch the prompts library directory
        with patch.object(prompts, "PROMPTS_LIBRARY_DIRECTORY", prompts_dir):
            content = "Welcome!\n[greeting]\nGoodbye!"
            result = prompts._expand_prompt_includes(content)

        assert result == "Welcome!\nHello, I am Reachy!\nGoodbye!"

    def test_expand_multiple_placeholders(self, tmp_path: Path) -> None:
        """Test expanding multiple placeholders."""
        from reachy_mini_conversation_app import prompts

        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "intro.txt").write_text("Introduction text")
        (prompts_dir / "outro.txt").write_text("Conclusion text")

        with patch.object(prompts, "PROMPTS_LIBRARY_DIRECTORY", prompts_dir):
            content = "[intro]\nMiddle content\n[outro]"
            result = prompts._expand_prompt_includes(content)

        assert result == "Introduction text\nMiddle content\nConclusion text"

    def test_expand_with_subdirectory(self, tmp_path: Path) -> None:
        """Test expanding placeholder with subdirectory path."""
        from reachy_mini_conversation_app import prompts

        prompts_dir = tmp_path / "prompts"
        subdir = prompts_dir / "common"
        subdir.mkdir(parents=True)
        (subdir / "header.txt").write_text("Common header")

        with patch.object(prompts, "PROMPTS_LIBRARY_DIRECTORY", prompts_dir):
            content = "[common/header]\nBody text"
            result = prompts._expand_prompt_includes(content)

        assert result == "Common header\nBody text"

    def test_placeholder_not_found_keeps_original(self, tmp_path: Path) -> None:
        """Test that missing template keeps original placeholder."""
        from reachy_mini_conversation_app import prompts

        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        with patch.object(prompts, "PROMPTS_LIBRARY_DIRECTORY", prompts_dir):
            content = "Start\n[nonexistent]\nEnd"
            result = prompts._expand_prompt_includes(content)

        assert result == "Start\n[nonexistent]\nEnd"

    def test_no_placeholders_returns_unchanged(self) -> None:
        """Test content without placeholders is returned unchanged."""
        from reachy_mini_conversation_app import prompts

        content = "Just regular text\nNo placeholders here"
        result = prompts._expand_prompt_includes(content)

        assert result == content

    def test_placeholder_with_whitespace_preserved(self, tmp_path: Path) -> None:
        """Test that indented placeholders are handled."""
        from reachy_mini_conversation_app import prompts

        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "content.txt").write_text("Expanded content")

        with patch.object(prompts, "PROMPTS_LIBRARY_DIRECTORY", prompts_dir):
            # Placeholder with leading whitespace
            content = "Start\n  [content]\nEnd"
            result = prompts._expand_prompt_includes(content)

        # The placeholder is stripped, so it should match and expand
        assert result == "Start\nExpanded content\nEnd"

    def test_invalid_placeholder_format_not_expanded(self) -> None:
        """Test that invalid placeholder formats are not expanded."""
        from reachy_mini_conversation_app import prompts

        # Not on its own line
        content = "Some [invalid] text"
        result = prompts._expand_prompt_includes(content)

        assert result == content

    def test_expand_handles_read_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that read errors keep the original placeholder."""
        from reachy_mini_conversation_app import prompts

        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        template_file = prompts_dir / "error.txt"
        template_file.write_text("content")

        # Make read_text raise an exception
        def raise_error(*args: object, **kwargs: object) -> str:
            raise PermissionError("Cannot read file")

        with patch.object(prompts, "PROMPTS_LIBRARY_DIRECTORY", prompts_dir):
            with patch.object(Path, "read_text", raise_error):
                content = "[error]"
                result = prompts._expand_prompt_includes(content)

        # Should keep original due to error
        assert result == "[error]"


class TestGetSessionInstructions:
    """Tests for get_session_instructions function."""

    def test_load_default_prompt_when_no_profile(self, tmp_path: Path) -> None:
        """Test loading default prompt when no custom profile is set."""
        from reachy_mini_conversation_app import prompts
        from reachy_mini_conversation_app.config import config

        # Create temp prompts directory with default prompt
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "default_prompt.txt").write_text("Default instructions content")

        original_profile = config.REACHY_MINI_CUSTOM_PROFILE
        config.REACHY_MINI_CUSTOM_PROFILE = None

        try:
            with patch.object(prompts, "PROMPTS_LIBRARY_DIRECTORY", prompts_dir):
                result = prompts.get_session_instructions()

            assert result == "Default instructions content"
        finally:
            config.REACHY_MINI_CUSTOM_PROFILE = original_profile

    def test_load_profile_instructions(self, tmp_path: Path) -> None:
        """Test loading instructions from a custom profile."""
        from reachy_mini_conversation_app import prompts
        from reachy_mini_conversation_app.config import config

        # Create temp profile directory
        profiles_dir = tmp_path / "profiles"
        profile_dir = profiles_dir / "test_profile"
        profile_dir.mkdir(parents=True)
        (profile_dir / "instructions.txt").write_text("Test profile instructions")

        original_profile = config.REACHY_MINI_CUSTOM_PROFILE
        config.REACHY_MINI_CUSTOM_PROFILE = "test_profile"

        try:
            with patch.object(prompts, "PROFILES_DIRECTORY", profiles_dir):
                result = prompts.get_session_instructions()

            assert result == "Test profile instructions"
        finally:
            config.REACHY_MINI_CUSTOM_PROFILE = original_profile

    def test_instructions_with_placeholder_expansion(self, tmp_path: Path) -> None:
        """Test that placeholders in instructions are expanded."""
        from reachy_mini_conversation_app import prompts
        from reachy_mini_conversation_app.config import config

        # Create prompts library
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "personality.txt").write_text("I am friendly and helpful")

        # Create profile
        profiles_dir = tmp_path / "profiles"
        profile_dir = profiles_dir / "test_profile"
        profile_dir.mkdir(parents=True)
        (profile_dir / "instructions.txt").write_text("Hello!\n[personality]\nBye!")

        original_profile = config.REACHY_MINI_CUSTOM_PROFILE
        config.REACHY_MINI_CUSTOM_PROFILE = "test_profile"

        try:
            with patch.object(prompts, "PROFILES_DIRECTORY", profiles_dir):
                with patch.object(prompts, "PROMPTS_LIBRARY_DIRECTORY", prompts_dir):
                    result = prompts.get_session_instructions()

            assert result == "Hello!\nI am friendly and helpful\nBye!"
        finally:
            config.REACHY_MINI_CUSTOM_PROFILE = original_profile

    def test_exit_when_instructions_file_missing(self, tmp_path: Path) -> None:
        """Test that sys.exit is called when instructions file is missing."""
        from reachy_mini_conversation_app import prompts
        from reachy_mini_conversation_app.config import config

        profiles_dir = tmp_path / "profiles"
        profile_dir = profiles_dir / "empty_profile"
        profile_dir.mkdir(parents=True)
        # No instructions.txt file

        original_profile = config.REACHY_MINI_CUSTOM_PROFILE
        config.REACHY_MINI_CUSTOM_PROFILE = "empty_profile"

        try:
            with patch.object(prompts, "PROFILES_DIRECTORY", profiles_dir):
                with pytest.raises(SystemExit) as exc_info:
                    prompts.get_session_instructions()

            assert exc_info.value.code == 1
        finally:
            config.REACHY_MINI_CUSTOM_PROFILE = original_profile

    def test_exit_when_instructions_file_empty(self, tmp_path: Path) -> None:
        """Test that sys.exit is called when instructions file is empty."""
        from reachy_mini_conversation_app import prompts
        from reachy_mini_conversation_app.config import config

        profiles_dir = tmp_path / "profiles"
        profile_dir = profiles_dir / "empty_instructions"
        profile_dir.mkdir(parents=True)
        (profile_dir / "instructions.txt").write_text("")  # Empty file

        original_profile = config.REACHY_MINI_CUSTOM_PROFILE
        config.REACHY_MINI_CUSTOM_PROFILE = "empty_instructions"

        try:
            with patch.object(prompts, "PROFILES_DIRECTORY", profiles_dir):
                with pytest.raises(SystemExit) as exc_info:
                    prompts.get_session_instructions()

            assert exc_info.value.code == 1
        finally:
            config.REACHY_MINI_CUSTOM_PROFILE = original_profile

    def test_exit_on_read_exception(self, tmp_path: Path) -> None:
        """Test that sys.exit is called on read exception."""
        from reachy_mini_conversation_app import prompts
        from reachy_mini_conversation_app.config import config

        profiles_dir = tmp_path / "profiles"
        profile_dir = profiles_dir / "error_profile"
        profile_dir.mkdir(parents=True)
        (profile_dir / "instructions.txt").write_text("content")

        original_profile = config.REACHY_MINI_CUSTOM_PROFILE
        config.REACHY_MINI_CUSTOM_PROFILE = "error_profile"

        def raise_error(*args: object, **kwargs: object) -> str:
            raise PermissionError("Cannot read")

        try:
            with patch.object(prompts, "PROFILES_DIRECTORY", profiles_dir):
                with patch.object(Path, "read_text", raise_error):
                    with pytest.raises(SystemExit) as exc_info:
                        prompts.get_session_instructions()

            assert exc_info.value.code == 1
        finally:
            config.REACHY_MINI_CUSTOM_PROFILE = original_profile


class TestGetSessionVoice:
    """Tests for get_session_voice function."""

    def test_returns_default_when_no_profile(self) -> None:
        """Test that default voice is returned when no profile is set."""
        from reachy_mini_conversation_app import prompts
        from reachy_mini_conversation_app.config import config

        original_profile = config.REACHY_MINI_CUSTOM_PROFILE
        config.REACHY_MINI_CUSTOM_PROFILE = None

        try:
            result = prompts.get_session_voice()
            assert result == "cedar"

            result = prompts.get_session_voice(default="ash")
            assert result == "ash"
        finally:
            config.REACHY_MINI_CUSTOM_PROFILE = original_profile

    def test_returns_voice_from_profile(self, tmp_path: Path) -> None:
        """Test that voice is loaded from profile's voice.txt."""
        from reachy_mini_conversation_app import prompts
        from reachy_mini_conversation_app.config import config

        profiles_dir = tmp_path / "profiles"
        profile_dir = profiles_dir / "voice_profile"
        profile_dir.mkdir(parents=True)
        (profile_dir / "voice.txt").write_text("alloy")

        original_profile = config.REACHY_MINI_CUSTOM_PROFILE
        config.REACHY_MINI_CUSTOM_PROFILE = "voice_profile"

        try:
            with patch.object(prompts, "PROFILES_DIRECTORY", profiles_dir):
                result = prompts.get_session_voice()

            assert result == "alloy"
        finally:
            config.REACHY_MINI_CUSTOM_PROFILE = original_profile

    def test_returns_default_when_voice_file_missing(self, tmp_path: Path) -> None:
        """Test that default is returned when voice.txt is missing."""
        from reachy_mini_conversation_app import prompts
        from reachy_mini_conversation_app.config import config

        profiles_dir = tmp_path / "profiles"
        profile_dir = profiles_dir / "no_voice_profile"
        profile_dir.mkdir(parents=True)
        # No voice.txt file

        original_profile = config.REACHY_MINI_CUSTOM_PROFILE
        config.REACHY_MINI_CUSTOM_PROFILE = "no_voice_profile"

        try:
            with patch.object(prompts, "PROFILES_DIRECTORY", profiles_dir):
                result = prompts.get_session_voice(default="echo")

            assert result == "echo"
        finally:
            config.REACHY_MINI_CUSTOM_PROFILE = original_profile

    def test_returns_default_when_voice_file_empty(self, tmp_path: Path) -> None:
        """Test that default is returned when voice.txt is empty."""
        from reachy_mini_conversation_app import prompts
        from reachy_mini_conversation_app.config import config

        profiles_dir = tmp_path / "profiles"
        profile_dir = profiles_dir / "empty_voice_profile"
        profile_dir.mkdir(parents=True)
        (profile_dir / "voice.txt").write_text("")  # Empty file

        original_profile = config.REACHY_MINI_CUSTOM_PROFILE
        config.REACHY_MINI_CUSTOM_PROFILE = "empty_voice_profile"

        try:
            with patch.object(prompts, "PROFILES_DIRECTORY", profiles_dir):
                result = prompts.get_session_voice(default="nova")

            assert result == "nova"
        finally:
            config.REACHY_MINI_CUSTOM_PROFILE = original_profile

    def test_returns_default_on_exception(self, tmp_path: Path) -> None:
        """Test that default is returned on read exception."""
        from reachy_mini_conversation_app import prompts
        from reachy_mini_conversation_app.config import config

        profiles_dir = tmp_path / "profiles"
        profile_dir = profiles_dir / "error_voice_profile"
        profile_dir.mkdir(parents=True)
        (profile_dir / "voice.txt").write_text("content")

        original_profile = config.REACHY_MINI_CUSTOM_PROFILE
        config.REACHY_MINI_CUSTOM_PROFILE = "error_voice_profile"

        def raise_error(*args: object, **kwargs: object) -> str:
            raise PermissionError("Cannot read")

        try:
            with patch.object(prompts, "PROFILES_DIRECTORY", profiles_dir):
                with patch.object(Path, "read_text", raise_error):
                    result = prompts.get_session_voice(default="shimmer")

            assert result == "shimmer"
        finally:
            config.REACHY_MINI_CUSTOM_PROFILE = original_profile

    def test_voice_content_stripped(self, tmp_path: Path) -> None:
        """Test that voice content is stripped of whitespace."""
        from reachy_mini_conversation_app import prompts
        from reachy_mini_conversation_app.config import config

        profiles_dir = tmp_path / "profiles"
        profile_dir = profiles_dir / "whitespace_voice"
        profile_dir.mkdir(parents=True)
        (profile_dir / "voice.txt").write_text("  fable  \n\n")

        original_profile = config.REACHY_MINI_CUSTOM_PROFILE
        config.REACHY_MINI_CUSTOM_PROFILE = "whitespace_voice"

        try:
            with patch.object(prompts, "PROFILES_DIRECTORY", profiles_dir):
                result = prompts.get_session_voice()

            assert result == "fable"
        finally:
            config.REACHY_MINI_CUSTOM_PROFILE = original_profile
