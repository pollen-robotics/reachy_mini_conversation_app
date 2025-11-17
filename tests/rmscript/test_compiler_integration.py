"""Integration tests for ReachyMiniScript compiler."""

import math

import pytest
from scipy.spatial.transform import Rotation as R

from reachy_mini_conversation_app.rmscript import ReachyMiniScriptCompiler
from reachy_mini_conversation_app.rmscript.errors import Action, WaitAction


class TestBasicCompilation:
    """Test basic compilation cases."""

    def test_simple_look_left(self) -> None:
        """Test compiling a simple 'look left' command."""
        source = """
DESCRIPTION test
look left
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        # Note: tool_name will be empty when using compile() directly
        # It's set from filename in compile_file()
        assert tool.description == "test"
        assert len(tool.errors) == 0

        # Check IR
        assert len(tool.ir) == 1
        action = tool.ir[0]
        assert isinstance(action, Action)
        assert action.head_pose is not None

        # Verify yaw is +30 degrees (left = positive yaw)
        rotation = R.from_matrix(action.head_pose[:3, :3])
        _, _, yaw = rotation.as_euler("xyz", degrees=True)
        assert yaw == pytest.approx(30, abs=0.1)
        assert action.duration == 1.0

    def test_keyword_reuse_with_and(self) -> None:
        """Test 'and' keyword reuse: 'look left and up'."""
        source = """

DESCRIPTION test
look left and up
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 1

        action = tool.ir[0]
        assert isinstance(action, Action)

        # Should have both yaw and pitch
        rotation = R.from_matrix(action.head_pose[:3, :3])
        roll, pitch, yaw = rotation.as_euler("xyz", degrees=True)

        assert yaw == pytest.approx(30, abs=0.1)  # left = positive yaw
        assert pitch == pytest.approx(-30, abs=0.1)  # up = negative pitch

    def test_wait_command(self) -> None:
        """Test wait command compilation."""
        source = """

DESCRIPTION test
wait 2s
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 1

        wait = tool.ir[0]
        assert isinstance(wait, WaitAction)
        assert wait.duration == 2.0

    def test_repeat_block(self) -> None:
        """Test repeat block expansion."""
        source = """

DESCRIPTION test
repeat 3
    look left
    look right
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        # Should have 6 actions (3 repetitions × 2 actions)
        assert len(tool.ir) == 6


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_keyword(self) -> None:
        """Test that invalid keywords produce errors."""
        source = """

DESCRIPTION test
jump up
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert not tool.success
        assert len(tool.errors) >= 1
        assert any("jump" in err.message.lower() for err in tool.errors)

    def test_out_of_range_warning(self) -> None:
        """Test that out-of-range values produce warnings."""
        source = """

DESCRIPTION test
turn left 200
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success  # Compiles successfully
        assert len(tool.warnings) >= 1
        assert any("200" in warn.message for warn in tool.warnings)


class TestCodeGeneration:
    """Test code generation."""

    def test_to_python_code(self) -> None:
        """Test Python code generation."""
        source = """

DESCRIPTION greeting
look left 30
wait 1s
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success

        # Generate Python code
        python_code = tool.to_python_code()

        # When using compile() directly, tool name defaults to "rmscript_tool"
        assert "def rmscript_tool(mini):" in python_code
        assert "goto_target" in python_code
        assert "time.sleep" in python_code


class TestQualitativeKeywords:
    """Test qualitative strength keywords."""

    def test_very_small_qualitative(self) -> None:
        """Test VERY_SMALL qualitative keywords (context-aware: 10° for turn)."""
        source = """

DESCRIPTION test
turn left tiny
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 1
        action = tool.ir[0]
        assert isinstance(action, Action)
        # Turn uses BODY_YAW_VERY_SMALL = 10 degrees
        assert action.body_yaw == pytest.approx(math.radians(10.0), abs=0.01)

    def test_small_qualitative(self) -> None:
        """Test SMALL qualitative keywords (context-aware: 30° for turn)."""
        source = """

DESCRIPTION test
turn left little
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        # Turn uses BODY_YAW_SMALL = 30 degrees
        assert action.body_yaw == pytest.approx(math.radians(30.0), abs=0.01)

    def test_medium_qualitative(self) -> None:
        """Test MEDIUM qualitative keywords (context-aware: 60° for turn)."""
        source = """

DESCRIPTION test
turn left medium
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        # Turn uses BODY_YAW_MEDIUM = 60 degrees
        assert action.body_yaw == pytest.approx(math.radians(60.0), abs=0.01)

    def test_large_qualitative(self) -> None:
        """Test LARGE qualitative keywords (context-aware: 90° for turn)."""
        source = """

DESCRIPTION test
turn left strong
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        # Turn uses BODY_YAW_LARGE = 90 degrees
        assert action.body_yaw == pytest.approx(math.radians(90.0), abs=0.01)

    def test_very_large_qualitative(self) -> None:
        """Test VERY_LARGE qualitative keywords (context-aware: 120° for turn)."""
        source = """

DESCRIPTION test
turn left enormous
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        # Turn uses BODY_YAW_VERY_LARGE = 120 degrees
        assert action.body_yaw == pytest.approx(math.radians(120.0), abs=0.01)

    def test_qualitative_for_distances(self) -> None:
        """Test qualitative keywords work for head translations (mm)."""
        source = """

DESCRIPTION test
head forward little
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        # Should be 5mm = 0.005m
        assert action.head_pose[0, 3] == pytest.approx(0.005, abs=0.0001)


class TestAntennaDirections:
    """Test antenna directional keywords."""

    def test_antenna_directional_up(self) -> None:
        """Test antenna with 'up' direction."""
        source = """

DESCRIPTION test
antenna both up
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        assert action.antennas is not None
        # up = 0°
        assert action.antennas[0] == pytest.approx(0.0, abs=0.01)
        assert action.antennas[1] == pytest.approx(0.0, abs=0.01)

    def test_antenna_directional_left(self) -> None:
        """Test antenna with 'left' direction."""
        source = """

DESCRIPTION test
antenna both left
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        # left = -90° = -π/2 radians
        assert action.antennas[0] == pytest.approx(math.radians(-90), abs=0.01)
        assert action.antennas[1] == pytest.approx(math.radians(-90), abs=0.01)

    def test_antenna_left_left(self) -> None:
        """Test 'antenna left left' (left antenna pointing left)."""
        source = """

DESCRIPTION test
antenna left left
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        # Only left antenna (index 0) should be set to -90°
        assert action.antennas[0] == pytest.approx(math.radians(-90), abs=0.01)

    def test_antenna_right_right(self) -> None:
        """Test 'antenna right right' (right antenna pointing right)."""
        source = """

DESCRIPTION test
antenna right right
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        # Only right antenna (index 1) should be set to 90°
        assert action.antennas[1] == pytest.approx(math.radians(90), abs=0.01)

    def test_antenna_clock_numeric(self) -> None:
        """Test antenna with numeric clock position."""
        source = """

DESCRIPTION test
antenna both 3
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        # 3 o'clock = 90°
        assert action.antennas[0] == pytest.approx(math.radians(90), abs=0.01)
        assert action.antennas[1] == pytest.approx(math.radians(90), abs=0.01)

    def test_antenna_clock_keyword(self) -> None:
        """Test antenna with clock keyword."""
        source = """

DESCRIPTION test
antenna both ext
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        # ext = 90°
        assert action.antennas[0] == pytest.approx(math.radians(90), abs=0.01)
        assert action.antennas[1] == pytest.approx(math.radians(90), abs=0.01)


class TestTurnCommand:
    """Test turn command rotates both body and head."""

    def test_turn_left_rotates_body_and_head(self) -> None:
        """Test that 'turn left' rotates both body yaw and head yaw."""
        source = """

DESCRIPTION test
turn left 50
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        assert isinstance(action, Action)

        # Verify body yaw is set
        assert action.body_yaw == pytest.approx(math.radians(50.0), abs=0.01)

        # Verify head yaw is also set (head rotates with body)
        assert action.head_pose is not None
        rotation = R.from_matrix(action.head_pose[:3, :3])
        _, _, yaw = rotation.as_euler("xyz", degrees=True)
        assert yaw == pytest.approx(50.0, abs=0.1)

    def test_turn_right_rotates_body_and_head(self) -> None:
        """Test that 'turn right' rotates both body yaw and head yaw."""
        source = """

DESCRIPTION test
turn right 30
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]

        # Verify body yaw is set (negative for right)
        assert action.body_yaw == pytest.approx(math.radians(-30.0), abs=0.01)

        # Verify head yaw is also set (negative for right)
        rotation = R.from_matrix(action.head_pose[:3, :3])
        _, _, yaw = rotation.as_euler("xyz", degrees=True)
        assert yaw == pytest.approx(-30.0, abs=0.1)

    def test_turn_center_resets_body_and_head(self) -> None:
        """Test that 'turn center' resets both body and head to zero."""
        source = """

DESCRIPTION test
turn center
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]

        # Verify body yaw is zero
        assert action.body_yaw == pytest.approx(0.0, abs=0.01)

        # Verify head yaw is zero
        rotation = R.from_matrix(action.head_pose[:3, :3])
        _, _, yaw = rotation.as_euler("xyz", degrees=True)
        assert yaw == pytest.approx(0.0, abs=0.1)


class TestContextAwareQualitatives:
    """Test that qualitative keywords use context-aware values."""

    def test_maximum_turn_vs_look_pitch(self) -> None:
        """Test 'maximum' uses different values for turn vs look up."""
        # Turn left maximum - should use BODY_YAW_VERY_LARGE (120°)
        turn_source = """

DESCRIPTION test
turn left maximum
"""
        compiler = ReachyMiniScriptCompiler()
        turn_tool = compiler.compile(turn_source)
        assert turn_tool.success
        turn_action = turn_tool.ir[0]
        # Should be 120° for body yaw
        assert turn_action.body_yaw == pytest.approx(math.radians(120.0), abs=0.01)

        # Look up maximum - should use HEAD_PITCH_ROLL_VERY_LARGE (38°)
        look_source = """

DESCRIPTION test
look up maximum
"""
        look_tool = compiler.compile(look_source)
        assert look_tool.success
        look_action = look_tool.ir[0]
        # Should be 38° for head pitch
        rotation = R.from_matrix(look_action.head_pose[:3, :3])
        _, pitch, _ = rotation.as_euler("xyz", degrees=True)
        assert pitch == pytest.approx(-38.0, abs=0.1)  # up = negative pitch

    def test_maximum_look_yaw_vs_pitch(self) -> None:
        """Test 'maximum' uses different values for look left vs look up."""
        # Look left maximum - should use HEAD_YAW_VERY_LARGE (60°)
        left_source = """

DESCRIPTION test
look left maximum
"""
        compiler = ReachyMiniScriptCompiler()
        left_tool = compiler.compile(left_source)
        assert left_tool.success
        left_action = left_tool.ir[0]
        rotation = R.from_matrix(left_action.head_pose[:3, :3])
        _, _, yaw = rotation.as_euler("xyz", degrees=True)
        assert yaw == pytest.approx(60.0, abs=0.1)

        # Look up maximum - should use HEAD_PITCH_ROLL_VERY_LARGE (38°)
        up_source = """

DESCRIPTION test
look up maximum
"""
        up_tool = compiler.compile(up_source)
        assert up_tool.success
        up_action = up_tool.ir[0]
        rotation = R.from_matrix(up_action.head_pose[:3, :3])
        _, pitch, _ = rotation.as_euler("xyz", degrees=True)
        assert pitch == pytest.approx(-38.0, abs=0.1)

    def test_maximum_head_translation(self) -> None:
        """Test 'maximum' for head translation uses TRANSLATION_VERY_LARGE (28mm)."""
        source = """

DESCRIPTION test
head forward maximum
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)
        assert tool.success
        action = tool.ir[0]
        # Should be 28mm = 0.028m
        assert action.head_pose[0, 3] == pytest.approx(0.028, abs=0.0001)

    def test_tilt_uses_pitch_roll_limits(self) -> None:
        """Test tilt uses HEAD_PITCH_ROLL limits (more restrictive)."""
        source = """

DESCRIPTION test
tilt left maximum
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)
        assert tool.success
        action = tool.ir[0]
        # Should use HEAD_PITCH_ROLL_VERY_LARGE = 38°
        rotation = R.from_matrix(action.head_pose[:3, :3])
        roll, _, _ = rotation.as_euler("xyz", degrees=True)
        assert roll == pytest.approx(38.0, abs=0.1)  # left tilt = positive roll


class TestPictureCommand:
    """Test picture command."""

    def test_picture_compiles(self) -> None:
        """Test that 'picture' command compiles successfully."""
        source = """

DESCRIPTION test
picture
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 1

        from reachy_mini_conversation_app.rmscript.errors import PictureAction
        action = tool.ir[0]
        assert isinstance(action, PictureAction)

    def test_picture_in_sequence(self) -> None:
        """Test picture command in a sequence with movements."""
        source = """

DESCRIPTION test
turn left 30
picture
turn center
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 3

        from reachy_mini_conversation_app.rmscript.errors import Action, PictureAction
        assert isinstance(tool.ir[0], Action)  # turn left
        assert isinstance(tool.ir[1], PictureAction)  # picture
        assert isinstance(tool.ir[2], Action)  # turn center

    def test_multiple_pictures(self) -> None:
        """Test multiple picture commands."""
        source = """

DESCRIPTION test
look left
picture
look right
picture
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 4

        from reachy_mini_conversation_app.rmscript.errors import Action, PictureAction
        assert isinstance(tool.ir[0], Action)  # look left
        assert isinstance(tool.ir[1], PictureAction)  # picture
        assert isinstance(tool.ir[2], Action)  # look right
        assert isinstance(tool.ir[3], PictureAction)  # picture

    def test_picture_with_wait(self) -> None:
        """Test picture command with wait."""
        source = """

DESCRIPTION test
turn left 30
wait 1s
picture
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 3

        from reachy_mini_conversation_app.rmscript.errors import Action, WaitAction, PictureAction
        assert isinstance(tool.ir[0], Action)  # turn left
        assert isinstance(tool.ir[1], WaitAction)  # wait
        assert isinstance(tool.ir[2], PictureAction)  # picture


class TestPlaySoundCommand:
    """Test play sound command."""

    def test_play_sound_async(self) -> None:
        """Test 'play sound' command (non-blocking)."""
        source = """

DESCRIPTION test
play mysound
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 1

        from reachy_mini_conversation_app.rmscript.errors import PlaySoundAction
        action = tool.ir[0]
        assert isinstance(action, PlaySoundAction)
        assert action.sound_name == "mysound"
        assert not action.blocking

    def test_play_sound_blocking_pause(self) -> None:
        """Test 'play sound pause' command (blocking)."""
        source = """

DESCRIPTION test
play mysound pause
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 1

        from reachy_mini_conversation_app.rmscript.errors import PlaySoundAction
        action = tool.ir[0]
        assert isinstance(action, PlaySoundAction)
        assert action.sound_name == "mysound"
        assert action.blocking

    def test_play_sound_blocking_fully(self) -> None:
        """Test 'play sound fully' command (blocking with synonym)."""
        source = """

DESCRIPTION test
play mysound fully
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        from reachy_mini_conversation_app.rmscript.errors import PlaySoundAction
        assert isinstance(action, PlaySoundAction)
        assert action.sound_name == "mysound"
        assert action.blocking

    def test_play_sound_in_sequence(self) -> None:
        """Test play sound in sequence with movements."""
        source = """

DESCRIPTION test
look left
play sound1
turn right
play sound2 pause
look center
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 5

        from reachy_mini_conversation_app.rmscript.errors import Action, PlaySoundAction
        assert isinstance(tool.ir[0], Action)  # look left
        assert isinstance(tool.ir[1], PlaySoundAction)  # play sound1 (async)
        assert not tool.ir[1].blocking
        assert isinstance(tool.ir[2], Action)  # turn right
        assert isinstance(tool.ir[3], PlaySoundAction)  # play sound2 (blocking)
        assert tool.ir[3].blocking
        assert isinstance(tool.ir[4], Action)  # look center

    def test_multiple_sounds(self) -> None:
        """Test multiple sound commands."""
        source = """

DESCRIPTION test
play intro
wait 1s
play main pause
wait 1s
play outro
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 5

        from reachy_mini_conversation_app.rmscript.errors import WaitAction, PlaySoundAction
        assert isinstance(tool.ir[0], PlaySoundAction)  # intro (async)
        assert isinstance(tool.ir[1], WaitAction)
        assert isinstance(tool.ir[2], PlaySoundAction)  # main (blocking)
        assert tool.ir[2].blocking
        assert isinstance(tool.ir[3], WaitAction)
        assert isinstance(tool.ir[4], PlaySoundAction)  # outro (async)


class TestWaitSyntax:
    """Test wait command syntax requirements."""

    def test_wait_requires_s_suffix(self) -> None:
        """Test that wait without 's' suffix produces error."""
        source = """

DESCRIPTION test
wait 1
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert not tool.success
        assert len(tool.errors) >= 1
        assert any("'s' after wait duration" in err.message for err in tool.errors)

    def test_wait_with_s_suffix_works(self) -> None:
        """Test that wait with 's' suffix compiles."""
        source = """

DESCRIPTION test
wait 1s
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        wait = tool.ir[0]
        assert isinstance(wait, WaitAction)
        assert wait.duration == 1.0

    def test_wait_decimal_with_s_suffix(self) -> None:
        """Test that decimal wait with 's' suffix works."""
        source = """

DESCRIPTION test
wait 2.5s
"""
        compiler = ReachyMiniScriptCompiler()
        tool = compiler.compile(source)

        assert tool.success
        wait = tool.ir[0]
        assert wait.duration == 2.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
