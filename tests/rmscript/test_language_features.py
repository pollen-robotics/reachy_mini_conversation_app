"""Integration tests for ReachyMiniScript language features."""

import math

import pytest
from scipy.spatial.transform import Rotation as R

from reachy_mini_conversation_app.rmscript import ReachyMiniScriptCompiler, verify_rmscript
from reachy_mini_conversation_app.rmscript.errors import Action, PlaySoundAction
from reachy_mini_conversation_app.rmscript.constants import (
    DEFAULT_ANGLE,
    BODY_YAW_LARGE,
    BODY_YAW_SMALL,
    BODY_YAW_MEDIUM,
    DEFAULT_DURATION,
    BACKWARD_SYNONYMS,
    DURATION_KEYWORDS,
    TRANSLATION_SMALL,
    BODY_YAW_VERY_LARGE,
    BODY_YAW_VERY_SMALL,
    TRANSLATION_VERY_LARGE,
    HEAD_PITCH_ROLL_VERY_LARGE,
)


@pytest.fixture
def compiler():
    """Reusable compiler instance."""
    return ReachyMiniScriptCompiler()


class TestBasicMovements:
    """Test turn/look/center commands."""

    def test_simple_look_left(self, compiler):
        """Test compiling a simple 'look left' command."""
        source = """DESCRIPTION test
look left"""
        tool = compiler.compile(source)

        assert tool.success
        assert tool.description == "test"
        assert len(tool.ir) == 1

        action = tool.ir[0]
        assert isinstance(action, Action)
        assert action.head_pose is not None

        rotation = R.from_matrix(action.head_pose[:3, :3])
        _, _, yaw = rotation.as_euler("xyz", degrees=True)
        assert yaw == pytest.approx(DEFAULT_ANGLE, abs=0.1)
        assert action.duration == DEFAULT_DURATION

    def test_look_right(self, compiler):
        """Test 'look right' command."""
        source = """DESCRIPTION test
look right"""
        tool = compiler.compile(source)

        assert tool.success
        rotation = R.from_matrix(tool.ir[0].head_pose[:3, :3])
        _, _, yaw = rotation.as_euler("xyz", degrees=True)
        assert yaw == pytest.approx(-DEFAULT_ANGLE, abs=0.1)

    def test_look_up(self, compiler):
        """Test 'look up' command."""
        source = """DESCRIPTION test
look up"""
        tool = compiler.compile(source)

        assert tool.success
        rotation = R.from_matrix(tool.ir[0].head_pose[:3, :3])
        _, pitch, _ = rotation.as_euler("xyz", degrees=True)
        assert pitch == pytest.approx(-DEFAULT_ANGLE, abs=0.1)  # up = negative pitch

    def test_look_down(self, compiler):
        """Test 'look down' command."""
        source = """DESCRIPTION test
look down"""
        tool = compiler.compile(source)

        assert tool.success
        rotation = R.from_matrix(tool.ir[0].head_pose[:3, :3])
        _, pitch, _ = rotation.as_euler("xyz", degrees=True)
        assert pitch == pytest.approx(DEFAULT_ANGLE, abs=0.1)  # down = positive pitch

    def test_look_center(self, compiler):
        """Test 'look center' command resets head."""
        source = """DESCRIPTION test
look center"""
        tool = compiler.compile(source)

        assert tool.success
        rotation = R.from_matrix(tool.ir[0].head_pose[:3, :3])
        roll, pitch, yaw = rotation.as_euler("xyz", degrees=True)
        assert roll == pytest.approx(0.0, abs=0.1)
        assert pitch == pytest.approx(0.0, abs=0.1)
        assert yaw == pytest.approx(0.0, abs=0.1)

    def test_turn_left_rotates_body_and_head(self, compiler):
        """Test that 'turn left' rotates both body yaw and head yaw."""
        source = """DESCRIPTION test
turn left 50"""
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        assert action.body_yaw == pytest.approx(math.radians(50.0), abs=0.01)

        rotation = R.from_matrix(action.head_pose[:3, :3])
        _, _, yaw = rotation.as_euler("xyz", degrees=True)
        assert yaw == pytest.approx(50.0, abs=0.1)

    def test_turn_right_rotates_body_and_head(self, compiler):
        """Test that 'turn right' rotates both body yaw and head yaw."""
        source = """DESCRIPTION test
turn right 30"""
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        assert action.body_yaw == pytest.approx(math.radians(-30.0), abs=0.01)

        rotation = R.from_matrix(action.head_pose[:3, :3])
        _, _, yaw = rotation.as_euler("xyz", degrees=True)
        assert yaw == pytest.approx(-30.0, abs=0.1)

    def test_turn_center_resets_body_and_head(self, compiler):
        """Test that 'turn center' resets both body and head to zero."""
        source = """DESCRIPTION test
turn center"""
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        assert action.body_yaw == pytest.approx(0.0, abs=0.01)

        rotation = R.from_matrix(action.head_pose[:3, :3])
        _, _, yaw = rotation.as_euler("xyz", degrees=True)
        assert yaw == pytest.approx(0.0, abs=0.1)


class TestHeadTranslation:
    """Test head left/right/up/down commands (translation)."""

    def test_head_left_positive_y(self, compiler):
        """Test 'head left' moves head in positive Y direction."""
        source = """DESCRIPTION test
head left 10"""
        tool = compiler.compile(source)

        assert tool.success
        assert tool.ir[0].head_pose is not None
        # left = positive Y translation
        assert tool.ir[0].head_pose[1, 3] == pytest.approx(0.010, abs=0.0001)  # 10mm

    def test_head_right_negative_y(self, compiler):
        """Test 'head right' moves head in negative Y direction."""
        source = """DESCRIPTION test
head right 10"""
        tool = compiler.compile(source)

        assert tool.success
        assert tool.ir[0].head_pose is not None
        # right = negative Y translation
        assert tool.ir[0].head_pose[1, 3] == pytest.approx(-0.010, abs=0.0001)  # -10mm

    def test_head_up_positive_z(self, compiler):
        """Test 'head up' moves head in positive Z direction."""
        source = """DESCRIPTION test
head up 15"""
        tool = compiler.compile(source)

        assert tool.success
        assert tool.ir[0].head_pose is not None
        # up = positive Z translation
        assert tool.ir[0].head_pose[2, 3] == pytest.approx(0.015, abs=0.0001)  # 15mm

    def test_head_down_negative_z(self, compiler):
        """Test 'head down' moves head in negative Z direction."""
        source = """DESCRIPTION test
head down 15"""
        tool = compiler.compile(source)

        assert tool.success
        assert tool.ir[0].head_pose is not None
        # down = negative Z translation
        assert tool.ir[0].head_pose[2, 3] == pytest.approx(-0.015, abs=0.0001)  # -15mm

    def test_head_forward_backward(self, compiler):
        """Test 'head forward' and backward synonyms."""
        # Forward
        forward_source = """DESCRIPTION test
head forward 10"""
        tool = compiler.compile(forward_source)
        assert tool.success
        assert tool.ir[0].head_pose[0, 3] == pytest.approx(0.010, abs=0.0001)  # +10mm

        # Backward
        backward_source = """DESCRIPTION test
head backward 10"""
        tool = compiler.compile(backward_source)
        assert tool.success
        assert tool.ir[0].head_pose[0, 3] == pytest.approx(-0.010, abs=0.0001)  # -10mm

    @pytest.mark.parametrize("direction", BACKWARD_SYNONYMS)
    def test_backward_synonyms_all_work(self, compiler, direction):
        """Test that all backward synonyms work for head movement."""
        source = f"""DESCRIPTION test
head {direction} 10"""
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 1
        action = tool.ir[0]
        assert isinstance(action, Action)
        # All backward synonyms should move head in negative X direction
        assert action.head_pose[0, 3] == pytest.approx(-0.010, abs=0.0001)  # -10mm


class TestTiltCommands:
    """Test tilt left/right commands."""

    def test_tilt_left(self, compiler):
        """Test 'tilt left' command."""
        source = """DESCRIPTION test
tilt left"""
        tool = compiler.compile(source)

        assert tool.success
        rotation = R.from_matrix(tool.ir[0].head_pose[:3, :3])
        roll, _, _ = rotation.as_euler("xyz", degrees=True)
        assert roll == pytest.approx(DEFAULT_ANGLE, abs=0.1)

    def test_tilt_right(self, compiler):
        """Test 'tilt right' command."""
        source = """DESCRIPTION test
tilt right"""
        tool = compiler.compile(source)

        assert tool.success
        rotation = R.from_matrix(tool.ir[0].head_pose[:3, :3])
        roll, _, _ = rotation.as_euler("xyz", degrees=True)
        assert roll == pytest.approx(-DEFAULT_ANGLE, abs=0.1)

    def test_tilt_center(self, compiler):
        """Test 'tilt center' command resets roll."""
        source = """DESCRIPTION test
tilt center"""
        tool = compiler.compile(source)

        assert tool.success
        rotation = R.from_matrix(tool.ir[0].head_pose[:3, :3])
        roll, _, _ = rotation.as_euler("xyz", degrees=True)
        assert roll == pytest.approx(0.0, abs=0.1)

    def test_tilt_uses_pitch_roll_limits(self, compiler):
        """Test that tilt commands use HEAD_PITCH_ROLL limits."""
        source = """DESCRIPTION test
tilt left maximum"""
        tool = compiler.compile(source)

        assert tool.success
        rotation = R.from_matrix(tool.ir[0].head_pose[:3, :3])
        roll, _, _ = rotation.as_euler("xyz", degrees=True)
        assert roll == pytest.approx(HEAD_PITCH_ROLL_VERY_LARGE, abs=0.1)


class TestCompoundMovements:
    """Test 'and' keyword for combining movements."""

    def test_keyword_reuse_with_and(self, compiler):
        """Test 'and' keyword reuse: 'look left and up'."""
        source = """DESCRIPTION test
look left and up"""
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 1

        action = tool.ir[0]
        rotation = R.from_matrix(action.head_pose[:3, :3])
        roll, pitch, yaw = rotation.as_euler("xyz", degrees=True)

        assert yaw == pytest.approx(DEFAULT_ANGLE, abs=0.1)  # left
        assert pitch == pytest.approx(-DEFAULT_ANGLE, abs=0.1)  # up

    def test_and_picture_error(self, compiler):
        """Test that 'look left and picture' produces error."""
        source = """DESCRIPTION test
look left and picture"""
        tool = compiler.compile(source)

        assert not tool.success
        assert len(tool.errors) >= 1
        assert any("cannot combine" in err.message.lower() for err in tool.errors)
        assert any("picture" in err.message.lower() for err in tool.errors)

    def test_and_play_error(self, compiler):
        """Test that 'turn left and play sound' produces error."""
        source = """DESCRIPTION test
turn left and play mysound"""
        tool = compiler.compile(source)

        assert not tool.success
        assert any("play" in err.message.lower() for err in tool.errors)

    def test_and_loop_error(self, compiler):
        """Test that 'look up and loop sound' produces error."""
        source = """DESCRIPTION test
look up and loop mysound"""
        tool = compiler.compile(source)

        assert not tool.success
        assert any("loop" in err.message.lower() for err in tool.errors)

    def test_and_wait_error(self, compiler):
        """Test that 'antenna both up and wait 1s' produces error."""
        source = """DESCRIPTION test
antenna both up and wait 1s"""
        tool = compiler.compile(source)

        assert not tool.success
        assert any("wait" in err.message.lower() for err in tool.errors)


class TestQualitativeStrengths:
    """Test context-aware qualitative keywords."""

    def test_very_small_qualitative_turn(self, compiler):
        """Test VERY_SMALL qualitative for turn."""
        source = """DESCRIPTION test
turn left tiny"""
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        assert action.body_yaw == pytest.approx(math.radians(BODY_YAW_VERY_SMALL), abs=0.01)

    def test_small_qualitative_turn(self, compiler):
        """Test SMALL qualitative for turn."""
        source = """DESCRIPTION test
turn left little"""
        tool = compiler.compile(source)

        assert tool.success
        assert tool.ir[0].body_yaw == pytest.approx(math.radians(BODY_YAW_SMALL), abs=0.01)

    def test_medium_qualitative_turn(self, compiler):
        """Test MEDIUM qualitative for turn."""
        source = """DESCRIPTION test
turn left medium"""
        tool = compiler.compile(source)

        assert tool.success
        assert tool.ir[0].body_yaw == pytest.approx(math.radians(BODY_YAW_MEDIUM), abs=0.01)

    def test_large_qualitative_turn(self, compiler):
        """Test LARGE qualitative for turn."""
        source = """DESCRIPTION test
turn left strong"""
        tool = compiler.compile(source)

        assert tool.success
        assert tool.ir[0].body_yaw == pytest.approx(math.radians(BODY_YAW_LARGE), abs=0.01)

    def test_very_large_qualitative_turn(self, compiler):
        """Test VERY_LARGE qualitative for turn."""
        source = """DESCRIPTION test
turn left enormous"""
        tool = compiler.compile(source)

        assert tool.success
        assert tool.ir[0].body_yaw == pytest.approx(math.radians(BODY_YAW_VERY_LARGE), abs=0.01)

    def test_qualitative_for_head_translation(self, compiler):
        """Test qualitative keywords for head translations (mm)."""
        source = """DESCRIPTION test
head forward little"""
        tool = compiler.compile(source)

        assert tool.success
        assert tool.ir[0].head_pose[0, 3] == pytest.approx(TRANSLATION_SMALL / 1000.0, abs=0.0001)

    def test_maximum_turn_vs_look_pitch(self, compiler):
        """Test 'maximum' uses different values for turn vs look up."""
        # Turn
        turn_source = """DESCRIPTION test
turn left maximum"""
        turn_tool = compiler.compile(turn_source)
        assert turn_tool.success
        assert turn_tool.ir[0].body_yaw == pytest.approx(math.radians(BODY_YAW_VERY_LARGE), abs=0.01)

        # Look up
        look_source = """DESCRIPTION test
look up maximum"""
        look_tool = compiler.compile(look_source)
        assert look_tool.success
        rotation = R.from_matrix(look_tool.ir[0].head_pose[:3, :3])
        _, pitch, _ = rotation.as_euler("xyz", degrees=True)
        assert pitch == pytest.approx(-HEAD_PITCH_ROLL_VERY_LARGE, abs=0.1)

    def test_maximum_head_translation(self, compiler):
        """Test 'maximum' for head translation uses TRANSLATION_VERY_LARGE."""
        source = """DESCRIPTION test
head forward maximum"""
        tool = compiler.compile(source)

        assert tool.success
        assert tool.ir[0].head_pose[0, 3] == pytest.approx(TRANSLATION_VERY_LARGE / 1000.0, abs=0.0001)


class TestAntennaControl:
    """Test antenna commands."""

    def test_antenna_directional_up(self, compiler):
        """Test antenna with 'up' direction."""
        source = """DESCRIPTION test
antenna both up"""
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        assert action.antennas is not None
        assert action.antennas[0] == pytest.approx(0.0, abs=0.01)
        assert action.antennas[1] == pytest.approx(0.0, abs=0.01)

    def test_antenna_directional_left(self, compiler):
        """Test antenna with 'left' direction."""
        source = """DESCRIPTION test
antenna both left"""
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        assert action.antennas[0] == pytest.approx(math.radians(-90), abs=0.01)
        assert action.antennas[1] == pytest.approx(math.radians(-90), abs=0.01)

    def test_antenna_directional_right(self, compiler):
        """Test antenna with 'right' direction."""
        source = """DESCRIPTION test
antenna both right"""
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        assert action.antennas[0] == pytest.approx(math.radians(90), abs=0.01)
        assert action.antennas[1] == pytest.approx(math.radians(90), abs=0.01)

    def test_antenna_directional_down(self, compiler):
        """Test antenna with 'down' direction."""
        source = """DESCRIPTION test
antenna both down"""
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        assert action.antennas[0] == pytest.approx(math.radians(180), abs=0.01)
        assert action.antennas[1] == pytest.approx(math.radians(180), abs=0.01)

    def test_antenna_left_modifier(self, compiler):
        """Test 'antenna left left' (left antenna pointing left)."""
        source = """DESCRIPTION test
antenna left left"""
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        assert action.antennas[0] == pytest.approx(math.radians(-90), abs=0.01)

    def test_antenna_right_modifier(self, compiler):
        """Test 'antenna right right' (right antenna pointing right)."""
        source = """DESCRIPTION test
antenna right right"""
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        assert action.antennas[1] == pytest.approx(math.radians(90), abs=0.01)

    def test_antenna_clock_numeric(self, compiler):
        """Test antenna with numeric clock position."""
        source = """DESCRIPTION test
antenna both 3"""
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        assert action.antennas[0] == pytest.approx(math.radians(90), abs=0.01)
        assert action.antennas[1] == pytest.approx(math.radians(90), abs=0.01)

    def test_antenna_clock_keyword(self, compiler):
        """Test antenna with clock keyword (ext/int/high/low)."""
        source = """DESCRIPTION test
antenna both ext"""
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        assert action.antennas[0] == pytest.approx(math.radians(90), abs=0.01)
        assert action.antennas[1] == pytest.approx(math.radians(90), abs=0.01)


class TestDurationControl:
    """Test timing and duration."""

    def test_default_duration(self, compiler):
        """Test that default duration is applied."""
        source = """DESCRIPTION test
look left"""
        tool = compiler.compile(source)

        assert tool.success
        assert tool.ir[0].duration == DEFAULT_DURATION

    def test_explicit_duration(self, compiler):
        """Test explicit duration with 's' suffix."""
        source = """DESCRIPTION test
look up 2s"""
        tool = compiler.compile(source)

        assert tool.success
        assert tool.ir[0].duration == 2.0

    def test_decimal_duration(self, compiler):
        """Test decimal duration values."""
        source = """DESCRIPTION test
wait 1.5s"""
        tool = compiler.compile(source)

        assert tool.success
        assert tool.ir[0].duration == 1.5

    def test_duration_keyword_fast(self, compiler):
        """Test 'fast' duration keyword."""
        source = """DESCRIPTION test
look left fast"""
        tool = compiler.compile(source)

        assert tool.success
        assert tool.ir[0].duration == DURATION_KEYWORDS["fast"]

    def test_duration_keyword_slow(self, compiler):
        """Test 'slow' duration keyword."""
        source = """DESCRIPTION test
look left slow"""
        tool = compiler.compile(source)

        assert tool.success
        assert tool.ir[0].duration == DURATION_KEYWORDS["slow"]

    @pytest.mark.parametrize("keyword,expected_duration", [
        ("slow", DURATION_KEYWORDS["slow"]),
        ("slowly", DURATION_KEYWORDS["slowly"]),
    ])
    def test_slowly_synonym(self, compiler, keyword, expected_duration):
        """Test that 'slowly' works as synonym for 'slow'."""
        source = f"""DESCRIPTION test
look left {keyword}"""
        tool = compiler.compile(source)

        assert tool.success
        assert tool.ir[0].duration == expected_duration

    def test_wait_requires_s_suffix(self, compiler):
        """Test that wait without 's' suffix produces an error."""
        source = """DESCRIPTION test
wait 2"""
        tool = compiler.compile(source)

        assert not tool.success
        assert any("s" in err.message.lower() for err in tool.errors)


class TestSoundPlayback:
    """Test play/loop sound commands."""

    def test_play_sound_async(self, compiler):
        """Test async sound playback."""
        source = """DESCRIPTION test
play mysound"""
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        assert isinstance(action, PlaySoundAction)
        assert action.sound_name == "mysound"
        assert not action.blocking
        assert action.duration is None

    def test_play_sound_blocking_pause(self, compiler):
        """Test blocking sound with 'pause' modifier."""
        source = """DESCRIPTION test
play mysound pause"""
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        assert action.blocking

    def test_play_sound_blocking_fully(self, compiler):
        """Test blocking sound with 'fully' modifier."""
        source = """DESCRIPTION test
play mysound fully"""
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        assert action.blocking

    def test_play_sound_with_duration(self, compiler):
        """Test 'play sound 5s' command."""
        source = """DESCRIPTION test
play mysound 5s"""
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        assert isinstance(action, PlaySoundAction)
        assert action.sound_name == "mysound"
        assert action.blocking
        assert action.duration == 5.0
        assert not action.loop

    def test_play_sound_in_sequence(self, compiler):
        """Test multiple sound commands in sequence."""
        source = """DESCRIPTION test
play sound1
play sound2
play sound3"""
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 3
        assert all(isinstance(a, PlaySoundAction) for a in tool.ir)

    def test_loop_sound_default_duration(self, compiler):
        """Test 'loop sound' uses default 10s duration."""
        source = """DESCRIPTION test
loop mysound"""
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        assert isinstance(action, PlaySoundAction)
        assert action.sound_name == "mysound"
        assert action.loop
        assert action.blocking
        assert action.duration == 10.0

    def test_loop_sound_custom_duration(self, compiler):
        """Test 'loop sound 30s' uses custom duration."""
        source = """DESCRIPTION test
loop mysound 30s"""
        tool = compiler.compile(source)

        assert tool.success
        action = tool.ir[0]
        assert action.loop
        assert action.duration == 30.0

    def test_loop_in_sequence(self, compiler):
        """Test loop sound in sequence with movements."""
        source = """DESCRIPTION test
look left
loop background 15s
look right"""
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 3
        assert isinstance(tool.ir[0], Action)  # look left
        assert isinstance(tool.ir[1], PlaySoundAction)  # loop
        assert tool.ir[1].loop
        assert tool.ir[1].duration == 15.0
        assert isinstance(tool.ir[2], Action)  # look right


class TestPictureCapture:
    """Test picture command."""

    def test_picture_compiles(self, compiler):
        """Test that 'picture' command compiles."""
        source = """DESCRIPTION test
picture"""
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 1

    def test_picture_in_sequence(self, compiler):
        """Test picture in sequence with movements."""
        source = """DESCRIPTION test
look left
picture
look right"""
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 3

    def test_multiple_pictures(self, compiler):
        """Test multiple picture commands."""
        source = """DESCRIPTION test
picture
wait 1s
picture"""
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 3

    def test_picture_with_wait(self, compiler):
        """Test picture with wait for positioning."""
        source = """DESCRIPTION test
look up
wait 1s
picture"""
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 3


class TestRepeatBlocks:
    """Test repeat/end blocks."""

    def test_repeat_block_basic(self, compiler):
        """Test basic repeat block expansion."""
        source = """DESCRIPTION test
repeat 3
    look left
    look right"""
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 6  # 3 repetitions × 2 actions

    def test_repeat_block_with_wait(self, compiler):
        """Test repeat block with wait commands."""
        source = """DESCRIPTION test
repeat 2
    look left
    wait 1s"""
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 4  # 2 repetitions × 2 actions

    def test_repeat_with_mixed_actions(self, compiler):
        """Test repeat block with mixed action types."""
        source = """DESCRIPTION test
repeat 2
    turn left
    antenna both up
    wait 0.5s"""
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 6  # 2 × 3


class TestErrorMessages:
    """Test error message quality."""

    def test_error_invalid_keyword_clear_message(self, compiler):
        """Test that invalid keywords produce clear errors."""
        source = """DESCRIPTION test
jump up"""
        tool = compiler.compile(source)

        assert not tool.success
        assert len(tool.errors) >= 1
        assert any("jump" in err.message.lower() for err in tool.errors)

    def test_error_invalid_direction_for_command(self, compiler):
        """Test error for invalid direction with specific command."""
        source = """DESCRIPTION test
turn up"""
        tool = compiler.compile(source)

        assert not tool.success
        assert any("turn" in err.message.lower() for err in tool.errors)
        assert any("up" in err.message.lower() or "direction" in err.message.lower() for err in tool.errors)

    def test_error_missing_antenna_parameters(self, compiler):
        """Test error when antenna parameters are missing."""
        source = """DESCRIPTION test
antenna"""
        tool = compiler.compile(source)

        assert not tool.success

    def test_error_malformed_duration(self, compiler):
        """Test error for malformed duration."""
        source = """DESCRIPTION test
wait abc"""
        tool = compiler.compile(source)

        assert not tool.success

    def test_error_unclosed_repeat_block(self, compiler):
        """Test error for missing indentation in repeat block."""
        source = """DESCRIPTION test
repeat 3
look left"""
        tool = compiler.compile(source)

        # Should fail due to missing indent
        assert not tool.success

    def test_error_missing_sound_name(self, compiler):
        """Test error when sound name is missing."""
        source = """DESCRIPTION test
play"""
        tool = compiler.compile(source)

        assert not tool.success
        assert any("sound" in err.message.lower() or "expected" in err.message.lower() for err in tool.errors)

    def test_warning_out_of_range_clear_message(self, compiler):
        """Test that out-of-range values produce clear warnings."""
        source = """DESCRIPTION test
turn left 200"""
        tool = compiler.compile(source)

        assert tool.success  # Compiles successfully
        assert len(tool.warnings) >= 1
        assert any("200" in warn.message for warn in tool.warnings)

    def test_warning_antenna_out_of_range(self, compiler):
        """Test warning for antenna position out of range."""
        source = """DESCRIPTION test
antenna both 15"""
        tool = compiler.compile(source)

        # Should compile but may warn (depends on implementation)
        # At minimum, shouldn't crash
        assert len(tool.errors) == 0 or "15" in str(tool.errors[0])

    def test_error_and_keyword_with_control_helpful(self, compiler):
        """Test helpful error for 'and' with control commands."""
        source = """DESCRIPTION test
look left and wait 1s"""
        tool = compiler.compile(source)

        assert not tool.success
        assert any("cannot combine" in err.message.lower() or "separate" in err.message.lower()
                   for err in tool.errors)

    def test_error_repeat_count_not_number(self, compiler):
        """Test error when repeat count is not a number."""
        source = """DESCRIPTION test
repeat abc
    look left"""
        tool = compiler.compile(source)

        assert not tool.success


class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_empty_program_after_description(self, compiler):
        """Test that program with only description is valid."""
        source = """DESCRIPTION test"""
        tool = compiler.compile(source)

        # Empty program should compile successfully
        assert tool.success
        assert len(tool.ir) == 0

    def test_comment_only_lines(self, compiler):
        """Test that comment-only programs work."""
        source = """DESCRIPTION test
# This is a comment
# Another comment"""
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 0

    def test_blank_lines_ignored(self, compiler):
        """Test that blank lines are properly ignored."""
        source = """DESCRIPTION test

look left

look right

"""
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 2

    def test_zero_duration_wait(self, compiler):
        """Test wait with zero duration."""
        source = """DESCRIPTION test
wait 0s"""
        tool = compiler.compile(source)

        assert tool.success
        assert tool.ir[0].duration == 0.0

    def test_very_large_repeat_count(self, compiler):
        """Test repeat with large count."""
        source = """DESCRIPTION test
repeat 100
    look left"""
        tool = compiler.compile(source)

        assert tool.success
        assert len(tool.ir) == 100


class TestCompilerAPI:
    """Test public API functions."""

    def test_compile_returns_compiled_tool(self, compiler):
        """Test that compile() returns CompiledTool."""
        source = """DESCRIPTION test
look left"""
        tool = compiler.compile(source)

        assert hasattr(tool, "success")
        assert hasattr(tool, "errors")
        assert hasattr(tool, "warnings")
        assert hasattr(tool, "ir")
        assert hasattr(tool, "executable")

    def test_to_python_code_generation(self, compiler):
        """Test Python code generation."""
        source = """DESCRIPTION greeting
look left 30
wait 1s"""
        tool = compiler.compile(source)

        assert tool.success
        python_code = tool.to_python_code()

        assert "def " in python_code
        assert "goto_target" in python_code
        assert "time.sleep" in python_code

    def test_verify_rmscript_valid(self):
        """Test verify_rmscript returns True for valid script."""
        source = """DESCRIPTION test
look left
wait 1s"""
        is_valid, errors = verify_rmscript(source)

        assert is_valid
        assert len(errors) == 0

    def test_verify_rmscript_invalid(self):
        """Test verify_rmscript returns False for invalid script."""
        source = """DESCRIPTION test
look left and picture"""
        is_valid, errors = verify_rmscript(source)

        assert not is_valid
        assert len(errors) >= 1
        assert any("cannot combine" in err.lower() for err in errors)

    def test_verify_rmscript_with_warnings(self):
        """Test verify_rmscript includes warnings."""
        source = """DESCRIPTION test
turn left 200"""
        is_valid, errors = verify_rmscript(source)

        assert is_valid  # Warnings don't prevent compilation
        assert len(errors) >= 1  # But warnings are included
        assert any("200" in err for err in errors)

    def test_verify_rmscript_syntax_error(self):
        """Test verify_rmscript catches syntax errors."""
        source = """DESCRIPTION test
jump up"""
        is_valid, errors = verify_rmscript(source)

        assert not is_valid
        assert len(errors) >= 1
        assert any("jump" in err.lower() for err in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
