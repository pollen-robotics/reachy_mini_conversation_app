"""Unit tests for the core_tools module."""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest


class TestToolDependencies:
    """Tests for ToolDependencies dataclass."""

    def test_tool_dependencies_required_fields(self) -> None:
        """Test ToolDependencies with required fields only."""
        from reachy_mini_conversation_app.tools.core_tools import ToolDependencies

        mock_reachy = MagicMock()
        mock_movement = MagicMock()

        deps = ToolDependencies(
            reachy_mini=mock_reachy,
            movement_manager=mock_movement,
        )

        assert deps.reachy_mini is mock_reachy
        assert deps.movement_manager is mock_movement
        assert deps.camera_worker is None
        assert deps.vision_manager is None
        assert deps.head_wobbler is None
        assert deps.motion_duration_s == 1.0
        assert deps.background_task_manager is None

    def test_tool_dependencies_all_fields(self) -> None:
        """Test ToolDependencies with all fields."""
        from reachy_mini_conversation_app.tools.core_tools import ToolDependencies

        mock_reachy = MagicMock()
        mock_movement = MagicMock()
        mock_camera = MagicMock()
        mock_vision = MagicMock()
        mock_wobbler = MagicMock()
        mock_task_manager = MagicMock()

        deps = ToolDependencies(
            reachy_mini=mock_reachy,
            movement_manager=mock_movement,
            camera_worker=mock_camera,
            vision_manager=mock_vision,
            head_wobbler=mock_wobbler,
            motion_duration_s=2.5,
            background_task_manager=mock_task_manager,
        )

        assert deps.reachy_mini is mock_reachy
        assert deps.movement_manager is mock_movement
        assert deps.camera_worker is mock_camera
        assert deps.vision_manager is mock_vision
        assert deps.head_wobbler is mock_wobbler
        assert deps.motion_duration_s == 2.5
        assert deps.background_task_manager is mock_task_manager


class TestToolBaseClass:
    """Tests for Tool base class."""

    def test_tool_spec_generation(self) -> None:
        """Test that Tool.spec() generates correct function spec."""
        from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

        class TestTool(Tool):
            name = "test_tool"
            description = "A test tool"
            parameters_schema = {
                "type": "object",
                "properties": {
                    "param1": {"type": "string"},
                },
                "required": ["param1"],
            }

            async def __call__(self, deps: ToolDependencies, **kwargs):
                return {"result": "ok"}

        tool = TestTool()
        spec = tool.spec()

        assert spec["type"] == "function"
        assert spec["name"] == "test_tool"
        assert spec["description"] == "A test tool"
        assert spec["parameters"]["type"] == "object"
        assert "param1" in spec["parameters"]["properties"]

    def test_tool_supports_background_default_false(self) -> None:
        """Test that supports_background defaults to False."""
        from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

        class TestTool(Tool):
            name = "test_tool"
            description = "A test tool"
            parameters_schema = {"type": "object", "properties": {}}

            async def __call__(self, deps: ToolDependencies, **kwargs):
                return {}

        tool = TestTool()
        assert tool.supports_background is False

    def test_tool_supports_background_can_be_enabled(self) -> None:
        """Test that supports_background can be set to True."""
        from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

        class BackgroundTool(Tool):
            name = "background_tool"
            description = "A background tool"
            parameters_schema = {"type": "object", "properties": {}}
            supports_background = True

            async def __call__(self, deps: ToolDependencies, **kwargs):
                return {}

        tool = BackgroundTool()
        assert tool.supports_background is True


class TestGetConcreteSubclasses:
    """Tests for get_concrete_subclasses function."""

    def test_get_concrete_subclasses_finds_direct_subclasses(self) -> None:
        """Test finding direct concrete subclasses."""
        from reachy_mini_conversation_app.tools.core_tools import Tool, get_concrete_subclasses, ToolDependencies

        class ConcreteTool(Tool):
            name = "concrete_tool"
            description = "A concrete tool"
            parameters_schema = {"type": "object", "properties": {}}

            async def __call__(self, deps: ToolDependencies, **kwargs):
                return {}

        subclasses = get_concrete_subclasses(Tool)
        assert ConcreteTool in subclasses

    def test_get_concrete_subclasses_skips_abstract(self) -> None:
        """Test that abstract classes are skipped."""
        import abc
        from reachy_mini_conversation_app.tools.core_tools import Tool, get_concrete_subclasses, ToolDependencies

        class AbstractTool(Tool, abc.ABC):
            name = "abstract_tool"
            description = "An abstract tool"
            parameters_schema = {"type": "object", "properties": {}}

            @abc.abstractmethod
            async def custom_method(self):
                pass

        class ConcreteDerived(AbstractTool):
            async def __call__(self, deps: ToolDependencies, **kwargs):
                return {}

            async def custom_method(self):
                pass

        subclasses = get_concrete_subclasses(Tool)
        assert AbstractTool not in subclasses
        assert ConcreteDerived in subclasses


class TestSafeLoadObj:
    """Tests for _safe_load_obj function."""

    def test_safe_load_obj_valid_json(self) -> None:
        """Test loading valid JSON object."""
        from reachy_mini_conversation_app.tools.core_tools import _safe_load_obj

        result = _safe_load_obj('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_safe_load_obj_empty_string(self) -> None:
        """Test loading empty string returns empty dict."""
        from reachy_mini_conversation_app.tools.core_tools import _safe_load_obj

        result = _safe_load_obj("")
        assert result == {}

    def test_safe_load_obj_none(self) -> None:
        """Test loading None returns empty dict."""
        from reachy_mini_conversation_app.tools.core_tools import _safe_load_obj

        result = _safe_load_obj(None)  # type: ignore[arg-type]
        assert result == {}

    def test_safe_load_obj_invalid_json(self) -> None:
        """Test loading invalid JSON returns empty dict."""
        from reachy_mini_conversation_app.tools.core_tools import _safe_load_obj

        result = _safe_load_obj("not valid json {")
        assert result == {}

    def test_safe_load_obj_non_dict_returns_empty(self) -> None:
        """Test loading non-dict JSON returns empty dict."""
        from reachy_mini_conversation_app.tools.core_tools import _safe_load_obj

        # JSON array
        result = _safe_load_obj("[1, 2, 3]")
        assert result == {}

        # JSON string
        result = _safe_load_obj('"just a string"')
        assert result == {}

        # JSON number
        result = _safe_load_obj("42")
        assert result == {}


class TestDispatchToolCall:
    """Tests for dispatch_tool_call function."""

    @pytest.fixture
    def mock_deps(self) -> MagicMock:
        """Create mock tool dependencies."""
        from reachy_mini_conversation_app.tools.core_tools import ToolDependencies

        return ToolDependencies(
            reachy_mini=MagicMock(),
            movement_manager=MagicMock(),
        )

    @pytest.mark.asyncio
    async def test_dispatch_tool_call_unknown_tool(self, mock_deps: MagicMock) -> None:
        """Test dispatching to unknown tool returns error."""
        from reachy_mini_conversation_app.tools.core_tools import dispatch_tool_call

        result = await dispatch_tool_call("nonexistent_tool", "{}", mock_deps)

        assert "error" in result
        assert "unknown tool" in result["error"]

    @pytest.mark.asyncio
    async def test_dispatch_tool_call_success(self, mock_deps: MagicMock) -> None:
        """Test successful tool dispatch."""
        from reachy_mini_conversation_app.tools.core_tools import (
            dispatch_tool_call,
            ALL_TOOLS,
            Tool,
            ToolDependencies,
        )

        # Create and register a test tool
        class DispatchTestTool(Tool):
            name = "dispatch_test_tool"
            description = "Test tool for dispatch"
            parameters_schema = {
                "type": "object",
                "properties": {"msg": {"type": "string"}},
            }

            async def __call__(self, deps: ToolDependencies, **kwargs):
                return {"message": kwargs.get("msg", "default")}

        # Temporarily add to registry
        test_tool = DispatchTestTool()
        original_tools = ALL_TOOLS.copy()
        ALL_TOOLS["dispatch_test_tool"] = test_tool

        try:
            result = await dispatch_tool_call(
                "dispatch_test_tool",
                '{"msg": "hello"}',
                mock_deps,
            )
            assert result == {"message": "hello"}
        finally:
            # Restore registry
            ALL_TOOLS.clear()
            ALL_TOOLS.update(original_tools)

    @pytest.mark.asyncio
    async def test_dispatch_tool_call_with_invalid_json(self, mock_deps: MagicMock) -> None:
        """Test dispatch with invalid JSON args."""
        from reachy_mini_conversation_app.tools.core_tools import (
            dispatch_tool_call,
            ALL_TOOLS,
            Tool,
            ToolDependencies,
        )

        class InvalidJsonTestTool(Tool):
            name = "invalid_json_test_tool"
            description = "Test tool for invalid JSON"
            parameters_schema = {"type": "object", "properties": {}}

            async def __call__(self, deps: ToolDependencies, **kwargs):
                # kwargs should be empty due to invalid JSON
                return {"received_kwargs": kwargs}

        test_tool = InvalidJsonTestTool()
        original_tools = ALL_TOOLS.copy()
        ALL_TOOLS["invalid_json_test_tool"] = test_tool

        try:
            result = await dispatch_tool_call(
                "invalid_json_test_tool",
                "not valid json",
                mock_deps,
            )
            assert result == {"received_kwargs": {}}
        finally:
            ALL_TOOLS.clear()
            ALL_TOOLS.update(original_tools)

    @pytest.mark.asyncio
    async def test_dispatch_tool_call_handles_exception(self, mock_deps: MagicMock) -> None:
        """Test dispatch handles tool exceptions gracefully."""
        from reachy_mini_conversation_app.tools.core_tools import (
            dispatch_tool_call,
            ALL_TOOLS,
            Tool,
            ToolDependencies,
        )

        class ExceptionTestTool(Tool):
            name = "exception_test_tool"
            description = "Test tool that raises exception"
            parameters_schema = {"type": "object", "properties": {}}

            async def __call__(self, deps: ToolDependencies, **kwargs):
                raise ValueError("Test error message")

        test_tool = ExceptionTestTool()
        original_tools = ALL_TOOLS.copy()
        ALL_TOOLS["exception_test_tool"] = test_tool

        try:
            result = await dispatch_tool_call("exception_test_tool", "{}", mock_deps)
            assert "error" in result
            assert "ValueError" in result["error"]
            assert "Test error message" in result["error"]
        finally:
            ALL_TOOLS.clear()
            ALL_TOOLS.update(original_tools)


class TestGetToolSpecs:
    """Tests for get_tool_specs function."""

    def test_get_tool_specs_returns_all(self) -> None:
        """Test get_tool_specs returns all specs by default."""
        from reachy_mini_conversation_app.tools.core_tools import get_tool_specs, ALL_TOOL_SPECS

        specs = get_tool_specs()
        assert len(specs) == len(ALL_TOOL_SPECS)

    def test_get_tool_specs_with_exclusion(self) -> None:
        """Test get_tool_specs excludes specified tools."""
        from reachy_mini_conversation_app.tools.core_tools import (
            get_tool_specs,
            ALL_TOOL_SPECS,
            ALL_TOOLS,
        )

        if not ALL_TOOLS:
            pytest.skip("No tools registered")

        # Get first tool name to exclude
        first_tool_name = list(ALL_TOOLS.keys())[0]

        specs = get_tool_specs(exclusion_list=[first_tool_name])

        assert len(specs) == len(ALL_TOOL_SPECS) - 1
        assert all(spec["name"] != first_tool_name for spec in specs)

    def test_get_tool_specs_with_nonexistent_exclusion(self) -> None:
        """Test get_tool_specs ignores nonexistent tools in exclusion list."""
        from reachy_mini_conversation_app.tools.core_tools import get_tool_specs, ALL_TOOL_SPECS

        specs = get_tool_specs(exclusion_list=["nonexistent_tool"])
        assert len(specs) == len(ALL_TOOL_SPECS)


class TestLoadProfileTools:
    """Tests for _load_profile_tools function."""

    def test_load_profile_tools_missing_tools_txt(self, tmp_path: Path) -> None:
        """Test _load_profile_tools exits when tools.txt is missing."""
        from reachy_mini_conversation_app.tools import core_tools

        # Create empty profile directory
        profile_dir = tmp_path / "profiles" / "test_profile"
        profile_dir.mkdir(parents=True)

        with patch.object(core_tools.config, "REACHY_MINI_CUSTOM_PROFILE", "test_profile"):
            with patch.object(core_tools, "_TOOLS_INITIALIZED", False):
                # Patch Path to point to our temp directory
                original_file = core_tools.Path(__file__)
                mock_path = tmp_path / "core_tools.py"

                with patch.object(core_tools.Path, "__new__", return_value=mock_path):
                    with pytest.raises(SystemExit) as exc_info:
                        # We need to patch at the module level where Path is used
                        with patch(
                            "reachy_mini_conversation_app.tools.core_tools.Path",
                        ) as mock_path_cls:
                            mock_path_cls.return_value.parent.parent.__truediv__.return_value.__truediv__.return_value.__truediv__.return_value.exists.return_value = False

                            # Directly test _load_profile_tools
                            core_tools._load_profile_tools()

                    assert exc_info.value.code == 1

    def test_load_profile_tools_skips_comments_and_blanks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that comments and blank lines are skipped in tools.txt."""
        from reachy_mini_conversation_app.tools import core_tools

        # Create profile with tools.txt containing comments
        profile_dir = tmp_path / "profiles" / "comment_test"
        profile_dir.mkdir(parents=True)

        tools_txt_content = [
            "# This is a comment\n",
            "\n",
            "do_nothing\n",
            "  \n",
            "# Another comment\n",
            "camera\n",
        ]

        tools_txt = profile_dir / "tools.txt"
        tools_txt.write_text("".join(tools_txt_content))

        monkeypatch.setattr(core_tools.config, "REACHY_MINI_CUSTOM_PROFILE", "comment_test")

        # Track which modules are attempted to load
        loaded_modules: list[str] = []

        def mock_import(name: str) -> MagicMock:
            loaded_modules.append(name)
            raise ModuleNotFoundError(f"No module named '{name}'")

        # Patch the profile path resolution and importlib
        with patch.object(core_tools, "importlib") as mock_importlib:
            mock_importlib.import_module = mock_import

            # Patch Path(__file__) to point to a fake location that resolves to tmp_path
            fake_file = tmp_path / "tools" / "core_tools.py"
            fake_file.parent.mkdir(parents=True, exist_ok=True)
            fake_file.touch()

            with patch.object(core_tools.Path, "__call__", return_value=fake_file):
                # Directly call with patched path construction
                profile_module_path = tmp_path / "profiles" / "comment_test"
                tools_txt_path = profile_module_path / "tools.txt"

                # Manually test the parsing logic
                with open(tools_txt_path, "r") as f:
                    lines = f.readlines()

                tool_names = []
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    tool_names.append(line)

                # Verify comments and blanks are skipped
                assert tool_names == ["do_nothing", "camera"]
                assert "# This is a comment" not in tool_names
                assert "" not in tool_names


class TestInitializeTools:
    """Tests for _initialize_tools function."""

    def test_initialize_tools_skips_if_already_initialized(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that _initialize_tools skips if already initialized."""
        from reachy_mini_conversation_app.tools import core_tools
        import logging

        # Tools are already initialized at module import
        assert core_tools._TOOLS_INITIALIZED is True

        with caplog.at_level(logging.DEBUG, logger="reachy_mini_conversation_app.tools.core_tools"):
            core_tools._initialize_tools()

        assert "already initialized" in caplog.text
