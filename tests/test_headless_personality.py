"""Unit tests for headless_personality module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestConstants:
    """Tests for module constants."""

    def test_default_option_value(self) -> None:
        """Test DEFAULT_OPTION constant value."""
        from reachy_mini_conversation_app.headless_personality import DEFAULT_OPTION

        assert DEFAULT_OPTION == "(built-in default)"


class TestProfilesRoot:
    """Tests for _profiles_root function."""

    def test_profiles_root_returns_path(self) -> None:
        """Test that _profiles_root returns a Path."""
        from reachy_mini_conversation_app.headless_personality import _profiles_root

        result = _profiles_root()

        assert isinstance(result, Path)
        assert result.name == "profiles"


class TestPromptsDir:
    """Tests for _prompts_dir function."""

    def test_prompts_dir_returns_path(self) -> None:
        """Test that _prompts_dir returns a Path."""
        from reachy_mini_conversation_app.headless_personality import _prompts_dir

        result = _prompts_dir()

        assert isinstance(result, Path)
        assert result.name == "prompts"


class TestToolsDir:
    """Tests for _tools_dir function."""

    def test_tools_dir_returns_path(self) -> None:
        """Test that _tools_dir returns a Path."""
        from reachy_mini_conversation_app.headless_personality import _tools_dir

        result = _tools_dir()

        assert isinstance(result, Path)
        assert result.name == "tools"


class TestSanitizeName:
    """Tests for _sanitize_name function."""

    def test_sanitize_name_removes_spaces(self) -> None:
        """Test that spaces are replaced with underscores."""
        from reachy_mini_conversation_app.headless_personality import _sanitize_name

        result = _sanitize_name("my profile name")

        assert result == "my_profile_name"

    def test_sanitize_name_removes_special_chars(self) -> None:
        """Test that special characters are removed."""
        from reachy_mini_conversation_app.headless_personality import _sanitize_name

        result = _sanitize_name("my@profile!name#123")

        assert result == "myprofilename123"

    def test_sanitize_name_keeps_valid_chars(self) -> None:
        """Test that valid characters are kept."""
        from reachy_mini_conversation_app.headless_personality import _sanitize_name

        result = _sanitize_name("My_Profile-123")

        assert result == "My_Profile-123"

    def test_sanitize_name_strips_whitespace(self) -> None:
        """Test that leading/trailing whitespace is stripped."""
        from reachy_mini_conversation_app.headless_personality import _sanitize_name

        result = _sanitize_name("  my_profile  ")

        assert result == "my_profile"


class TestListPersonalities:
    """Tests for list_personalities function."""

    def test_list_personalities_returns_empty_when_no_profiles(self, tmp_path: Path) -> None:
        """Test that list returns empty when profiles dir doesn't exist."""
        from reachy_mini_conversation_app.headless_personality import list_personalities

        with patch(
            "reachy_mini_conversation_app.headless_personality._profiles_root",
            return_value=tmp_path / "nonexistent",
        ):
            result = list_personalities()

        assert result == []

    def test_list_personalities_finds_profiles_with_instructions(self, tmp_path: Path) -> None:
        """Test that list finds profiles with instructions.txt."""
        from reachy_mini_conversation_app.headless_personality import list_personalities

        # Create profile directories
        profile1 = tmp_path / "profile1"
        profile1.mkdir()
        (profile1 / "instructions.txt").write_text("Instructions 1")

        profile2 = tmp_path / "profile2"
        profile2.mkdir()
        (profile2 / "instructions.txt").write_text("Instructions 2")

        # Create profile without instructions (should be skipped)
        profile3 = tmp_path / "profile3"
        profile3.mkdir()

        with patch(
            "reachy_mini_conversation_app.headless_personality._profiles_root",
            return_value=tmp_path,
        ):
            result = list_personalities()

        assert "profile1" in result
        assert "profile2" in result
        assert "profile3" not in result

    def test_list_personalities_skips_user_personalities_folder(self, tmp_path: Path) -> None:
        """Test that user_personalities folder is handled separately."""
        from reachy_mini_conversation_app.headless_personality import list_personalities

        # Create regular profile
        profile1 = tmp_path / "profile1"
        profile1.mkdir()
        (profile1 / "instructions.txt").write_text("Instructions 1")

        # Create user_personalities with a profile
        user_dir = tmp_path / "user_personalities"
        user_dir.mkdir()
        user_profile = user_dir / "my_custom"
        user_profile.mkdir()
        (user_profile / "instructions.txt").write_text("Custom instructions")

        with patch(
            "reachy_mini_conversation_app.headless_personality._profiles_root",
            return_value=tmp_path,
        ):
            result = list_personalities()

        assert "profile1" in result
        assert "user_personalities/my_custom" in result
        assert "user_personalities" not in result

    def test_list_personalities_handles_exception(self, tmp_path: Path) -> None:
        """Test that exceptions are handled gracefully."""
        from reachy_mini_conversation_app.headless_personality import list_personalities

        # Set to a file instead of directory to cause an error
        test_file = tmp_path / "not_a_dir"
        test_file.write_text("content")

        with patch(
            "reachy_mini_conversation_app.headless_personality._profiles_root",
            return_value=test_file,
        ):
            result = list_personalities()

        assert result == []


class TestResolveProfileDir:
    """Tests for resolve_profile_dir function."""

    def test_resolve_profile_dir_returns_correct_path(self, tmp_path: Path) -> None:
        """Test that resolve returns correct path."""
        from reachy_mini_conversation_app.headless_personality import resolve_profile_dir

        with patch(
            "reachy_mini_conversation_app.headless_personality._profiles_root",
            return_value=tmp_path,
        ):
            result = resolve_profile_dir("my_profile")

        assert result == tmp_path / "my_profile"


class TestReadInstructionsFor:
    """Tests for read_instructions_for function."""

    def test_read_instructions_for_default_option(self, tmp_path: Path) -> None:
        """Test reading instructions for default option."""
        from reachy_mini_conversation_app.headless_personality import (
            DEFAULT_OPTION,
            read_instructions_for,
        )

        # Create default prompt
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "default_prompt.txt").write_text("Default instructions\n")

        with patch(
            "reachy_mini_conversation_app.headless_personality._prompts_dir",
            return_value=prompts_dir,
        ):
            result = read_instructions_for(DEFAULT_OPTION)

        assert result == "Default instructions"

    def test_read_instructions_for_custom_profile(self, tmp_path: Path) -> None:
        """Test reading instructions for custom profile."""
        from reachy_mini_conversation_app.headless_personality import read_instructions_for

        # Create profile
        profile = tmp_path / "my_profile"
        profile.mkdir()
        (profile / "instructions.txt").write_text("Custom instructions\n")

        with patch(
            "reachy_mini_conversation_app.headless_personality._profiles_root",
            return_value=tmp_path,
        ):
            result = read_instructions_for("my_profile")

        assert result == "Custom instructions"

    def test_read_instructions_for_nonexistent_profile(self, tmp_path: Path) -> None:
        """Test reading instructions for nonexistent profile."""
        from reachy_mini_conversation_app.headless_personality import read_instructions_for

        with patch(
            "reachy_mini_conversation_app.headless_personality._profiles_root",
            return_value=tmp_path,
        ):
            result = read_instructions_for("nonexistent")

        assert result == ""

    def test_read_instructions_for_handles_exception(self) -> None:
        """Test that exceptions return error message."""
        from reachy_mini_conversation_app.headless_personality import read_instructions_for

        with patch(
            "reachy_mini_conversation_app.headless_personality.resolve_profile_dir",
            side_effect=PermissionError("No access"),
        ):
            result = read_instructions_for("some_profile")

        assert "Could not load instructions" in result


class TestAvailableToolsFor:
    """Tests for available_tools_for function."""

    def test_available_tools_for_finds_shared_tools(self, tmp_path: Path) -> None:
        """Test that shared tools are found."""
        from reachy_mini_conversation_app.headless_personality import (
            DEFAULT_OPTION,
            available_tools_for,
        )

        # Create tools directory
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "tool1.py").write_text("# tool 1")
        (tools_dir / "tool2.py").write_text("# tool 2")
        (tools_dir / "__init__.py").write_text("# init")
        (tools_dir / "core_tools.py").write_text("# core")

        with patch(
            "reachy_mini_conversation_app.headless_personality._tools_dir",
            return_value=tools_dir,
        ):
            with patch(
                "reachy_mini_conversation_app.headless_personality._profiles_root",
                return_value=tmp_path,
            ):
                result = available_tools_for(DEFAULT_OPTION)

        assert "tool1" in result
        assert "tool2" in result
        assert "__init__" not in result
        assert "core_tools" not in result

    def test_available_tools_for_includes_local_tools(self, tmp_path: Path) -> None:
        """Test that local profile tools are included."""
        from reachy_mini_conversation_app.headless_personality import available_tools_for

        # Create tools directory
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "shared_tool.py").write_text("# shared")

        # Create profile with local tools
        profile = tmp_path / "profiles" / "my_profile"
        profile.mkdir(parents=True)
        (profile / "local_tool.py").write_text("# local")
        (profile / "instructions.txt").write_text("instructions")

        with patch(
            "reachy_mini_conversation_app.headless_personality._tools_dir",
            return_value=tools_dir,
        ):
            with patch(
                "reachy_mini_conversation_app.headless_personality._profiles_root",
                return_value=tmp_path / "profiles",
            ):
                result = available_tools_for("my_profile")

        assert "shared_tool" in result
        assert "local_tool" in result

    def test_available_tools_for_handles_exception(self) -> None:
        """Test that exceptions are handled gracefully."""
        from reachy_mini_conversation_app.headless_personality import (
            DEFAULT_OPTION,
            available_tools_for,
        )

        with patch(
            "reachy_mini_conversation_app.headless_personality._tools_dir",
            side_effect=PermissionError("No access"),
        ):
            result = available_tools_for(DEFAULT_OPTION)

        # Should return empty list on exception
        assert result == []


class TestWriteProfile:
    """Tests for _write_profile function."""

    def test_write_profile_creates_directory(self, tmp_path: Path) -> None:
        """Test that _write_profile creates the profile directory."""
        from reachy_mini_conversation_app.headless_personality import _write_profile

        with patch(
            "reachy_mini_conversation_app.headless_personality._profiles_root",
            return_value=tmp_path,
        ):
            _write_profile("test_profile", "My instructions", "tool1\ntool2", "ash")

        target_dir = tmp_path / "user_personalities" / "test_profile"
        assert target_dir.exists()

    def test_write_profile_writes_instructions(self, tmp_path: Path) -> None:
        """Test that _write_profile writes instructions.txt."""
        from reachy_mini_conversation_app.headless_personality import _write_profile

        with patch(
            "reachy_mini_conversation_app.headless_personality._profiles_root",
            return_value=tmp_path,
        ):
            _write_profile("test_profile", "My instructions", "", "cedar")

        instr_file = tmp_path / "user_personalities" / "test_profile" / "instructions.txt"
        assert instr_file.read_text() == "My instructions\n"

    def test_write_profile_writes_tools(self, tmp_path: Path) -> None:
        """Test that _write_profile writes tools.txt."""
        from reachy_mini_conversation_app.headless_personality import _write_profile

        with patch(
            "reachy_mini_conversation_app.headless_personality._profiles_root",
            return_value=tmp_path,
        ):
            _write_profile("test_profile", "instructions", "tool1\ntool2", "cedar")

        tools_file = tmp_path / "user_personalities" / "test_profile" / "tools.txt"
        assert tools_file.read_text() == "tool1\ntool2\n"

    def test_write_profile_writes_voice(self, tmp_path: Path) -> None:
        """Test that _write_profile writes voice.txt."""
        from reachy_mini_conversation_app.headless_personality import _write_profile

        with patch(
            "reachy_mini_conversation_app.headless_personality._profiles_root",
            return_value=tmp_path,
        ):
            _write_profile("test_profile", "instructions", "", "ash")

        voice_file = tmp_path / "user_personalities" / "test_profile" / "voice.txt"
        assert voice_file.read_text() == "ash\n"

    def test_write_profile_defaults_voice_to_cedar(self, tmp_path: Path) -> None:
        """Test that _write_profile defaults voice to cedar."""
        from reachy_mini_conversation_app.headless_personality import _write_profile

        with patch(
            "reachy_mini_conversation_app.headless_personality._profiles_root",
            return_value=tmp_path,
        ):
            _write_profile("test_profile", "instructions", "", "")

        voice_file = tmp_path / "user_personalities" / "test_profile" / "voice.txt"
        assert voice_file.read_text() == "cedar\n"

    def test_write_profile_strips_whitespace(self, tmp_path: Path) -> None:
        """Test that _write_profile strips whitespace from content."""
        from reachy_mini_conversation_app.headless_personality import _write_profile

        with patch(
            "reachy_mini_conversation_app.headless_personality._profiles_root",
            return_value=tmp_path,
        ):
            _write_profile("test_profile", "  instructions  ", "  tools  ", "  ash  ")

        target_dir = tmp_path / "user_personalities" / "test_profile"
        assert (target_dir / "instructions.txt").read_text() == "instructions\n"
        assert (target_dir / "tools.txt").read_text() == "tools\n"
        assert (target_dir / "voice.txt").read_text() == "ash\n"
