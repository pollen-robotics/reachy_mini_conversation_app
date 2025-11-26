"""Integration tests for rmscript tool creation."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

import pytest

from rmscript import compile_script


class TestCreateToolFromRMScript:
    """Test create_tool_from_rmscript function."""

    @pytest.fixture
    def temp_rmscript_file(self, tmp_path):
        """Create a temporary rmscript file for testing."""
        script_content = """DESCRIPTION Test tool for testing
look left
wait 1s
look right
"""
        script_file = tmp_path / "test_tool.rmscript"
        script_file.write_text(script_content)
        return script_file

    @pytest.fixture
    def invalid_rmscript_file(self, tmp_path):
        """Create an invalid rmscript file for testing."""
        script_content = """DESCRIPTION Invalid tool
jump up
"""
        script_file = tmp_path / "invalid_tool.rmscript"
        script_file.write_text(script_content)
        return script_file

    def test_create_tool_from_valid_script(self, temp_rmscript_file):
        """Test creating a tool from a valid rmscript file."""
        from reachy_mini_conversation_app.rmscript_tools import create_tool_from_rmscript

        # Change to temp directory to use relative path
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(temp_rmscript_file.parent)

            # Mock the frame inspection to return our test directory
            with patch("inspect.currentframe") as mock_frame:
                # Create mock frame hierarchy
                caller_frame = Mock()
                caller_frame.f_back = Mock()
                caller_frame.f_back.f_globals = {
                    "__file__": str(temp_rmscript_file.parent / "__init__.py"),
                    "__name__": "test_module",
                }
                mock_frame.return_value = caller_frame

                # Create the tool class
                ToolClass = create_tool_from_rmscript(temp_rmscript_file.name)

                # Check tool class attributes
                # Tool name comes from filename (test_tool.rmscript -> test_tool)
                assert ToolClass.name == "test_tool"
                assert "Test tool for testing" in ToolClass.description
                assert hasattr(ToolClass, "__call__")
                assert ToolClass.parameters_schema["type"] == "object"
        finally:
            os.chdir(old_cwd)

    def test_tool_execution(self, temp_rmscript_file):
        """Test executing the created tool."""
        from reachy_mini_conversation_app.rmscript_tools import create_tool_from_rmscript
        from reachy_mini_conversation_app.tools.core_tools import ToolDependencies

        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(temp_rmscript_file.parent)

            with patch("inspect.currentframe") as mock_frame:
                caller_frame = Mock()
                caller_frame.f_back = Mock()
                caller_frame.f_back.f_globals = {
                    "__file__": str(temp_rmscript_file.parent / "__init__.py"),
                    "__name__": "test_module",
                }
                mock_frame.return_value = caller_frame

                ToolClass = create_tool_from_rmscript(temp_rmscript_file.name)

                # Create mock dependencies
                mock_robot = Mock()
                mock_robot.get_current_head_pose.return_value = __import__("numpy").eye(4, dtype=__import__("numpy").float32)
                mock_robot.get_current_joint_positions.return_value = ([0.0, 0.0, 0.0], [0.0, 0.0])

                mock_manager = Mock()
                mock_manager.queue_move = Mock()
                mock_manager.set_moving_state = Mock()

                deps = ToolDependencies(
                    reachy_mini=mock_robot,
                    movement_manager=mock_manager,
                    camera_worker=None,
                )

                # Instantiate and execute tool
                tool_instance = ToolClass()
                import asyncio
                result = asyncio.run(tool_instance(deps))

                # Check that moves were queued
                assert mock_manager.queue_move.called
                assert "status" in result
                assert "total_duration" in result
        finally:
            os.chdir(old_cwd)

    def test_create_tool_from_invalid_script(self, invalid_rmscript_file):
        """Test creating a tool from an invalid rmscript file."""
        from reachy_mini_conversation_app.rmscript_tools import create_tool_from_rmscript
        from reachy_mini_conversation_app.tools.core_tools import ToolDependencies

        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(invalid_rmscript_file.parent)

            with patch("inspect.currentframe") as mock_frame:
                caller_frame = Mock()
                caller_frame.f_back = Mock()
                caller_frame.f_back.f_globals = {
                    "__file__": str(invalid_rmscript_file.parent / "__init__.py"),
                    "__name__": "test_module",
                }
                mock_frame.return_value = caller_frame

                # Should not raise, but tool will have errors
                ToolClass = create_tool_from_rmscript(invalid_rmscript_file.name)

                # Tool should have compilation result with errors
                assert hasattr(ToolClass, "_compilation_result")
                assert not ToolClass._compilation_result.success

                # Executing the tool should return errors
                deps = Mock(spec=ToolDependencies)
                tool_instance = ToolClass()
                import asyncio
                result = asyncio.run(tool_instance(deps))

                assert "error" in result
                assert "compilation failed" in result["error"].lower()
        finally:
            os.chdir(old_cwd)

    def test_tool_with_compilation_warnings(self, tmp_path):
        """Test tool creation with warnings."""
        script_content = """DESCRIPTION Tool with warnings
turn left 200
"""
        script_file = tmp_path / "warning_tool.rmscript"
        script_file.write_text(script_content)

        from reachy_mini_conversation_app.rmscript_tools import create_tool_from_rmscript

        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            with patch("inspect.currentframe") as mock_frame:
                caller_frame = Mock()
                caller_frame.f_back = Mock()
                caller_frame.f_back.f_globals = {
                    "__file__": str(tmp_path / "__init__.py"),
                    "__name__": "test_module",
                }
                mock_frame.return_value = caller_frame

                # Should create tool despite warnings
                ToolClass = create_tool_from_rmscript(script_file.name)

                # Tool should compile successfully
                assert ToolClass._compilation_result.success
                # But should have warnings
                assert len(ToolClass._compilation_result.warnings) > 0
        finally:
            os.chdir(old_cwd)

    def test_compiled_ir_is_cached(self, temp_rmscript_file):
        """Test that compiled IR is cached in the tool class."""
        from reachy_mini_conversation_app.rmscript_tools import create_tool_from_rmscript

        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(temp_rmscript_file.parent)

            with patch("inspect.currentframe") as mock_frame:
                caller_frame = Mock()
                caller_frame.f_back = Mock()
                caller_frame.f_back.f_globals = {
                    "__file__": str(temp_rmscript_file.parent / "__init__.py"),
                    "__name__": "test_module",
                }
                mock_frame.return_value = caller_frame

                ToolClass = create_tool_from_rmscript(temp_rmscript_file.name)

                # Check that compiled result is stored
                assert hasattr(ToolClass, "_compilation_result")
                result = ToolClass._compilation_result
                assert result.success
                assert len(result.ir) == 3  # look left, wait, look right
        finally:
            os.chdir(old_cwd)


class TestRMScriptCompilerIntegration:
    """Integration tests for rmscript compiler with adapters."""

    def test_simple_movement_compiles_to_ir(self):
        """Test that simple movement compiles correctly."""
        script = """DESCRIPTION Simple movement
look left
"""
        result = compile_script(script)

        assert result.success
        assert len(result.ir) == 1
        from rmscript.ir import IRAction
        assert isinstance(result.ir[0], IRAction)
        assert result.ir[0].head_pose is not None

    def test_wait_compiles_to_ir_wait_action(self):
        """Test that wait compiles to IRWaitAction."""
        script = """DESCRIPTION Wait test
wait 2s
"""
        result = compile_script(script)

        assert result.success
        assert len(result.ir) == 1
        from rmscript.ir import IRWaitAction
        assert isinstance(result.ir[0], IRWaitAction)
        assert result.ir[0].duration == 2.0

    def test_picture_compiles_to_ir_picture_action(self):
        """Test that picture compiles to IRPictureAction."""
        script = """DESCRIPTION Picture test
picture
"""
        result = compile_script(script)

        assert result.success
        assert len(result.ir) == 1
        from rmscript.ir import IRPictureAction
        assert isinstance(result.ir[0], IRPictureAction)

    def test_sound_compiles_to_ir_play_sound_action(self):
        """Test that sound compiles to IRPlaySoundAction."""
        script = """DESCRIPTION Sound test
play test_sound
"""
        result = compile_script(script)

        assert result.success
        assert len(result.ir) == 1
        from rmscript.ir import IRPlaySoundAction
        assert isinstance(result.ir[0], IRPlaySoundAction)
        assert result.ir[0].sound_name == "test_sound"

    def test_complex_script_compiles(self):
        """Test that complex script with multiple actions compiles."""
        script = """DESCRIPTION Complex test
look left
wait 0.5s
look right
picture
play happy_sound
"""
        result = compile_script(script)

        assert result.success
        assert len(result.ir) == 5

        from rmscript.ir import IRAction, IRWaitAction, IRPictureAction, IRPlaySoundAction
        assert isinstance(result.ir[0], IRAction)
        assert isinstance(result.ir[1], IRWaitAction)
        assert isinstance(result.ir[2], IRAction)
        assert isinstance(result.ir[3], IRPictureAction)
        assert isinstance(result.ir[4], IRPlaySoundAction)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
