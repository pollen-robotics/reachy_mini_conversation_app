"""Semantic analysis for ReachyMiniScript - validation, defaults, and IR generation."""

from typing import List

import numpy as np

from reachy_mini.utils import create_head_pose
from reachy_mini_conversation_app.rmscript.errors import (
    Action,
    WaitAction,
    PictureAction,
    PlaySoundAction,
    CompilationError,
)
from reachy_mini_conversation_app.rmscript.ast_nodes import (
    Program,
    Statement,
    ActionChain,
    RepeatBlock,
    SingleAction,
    WaitStatement,
    PictureStatement,
    PlaySoundStatement,
)
from reachy_mini_conversation_app.rmscript.constants import (
    # Legacy angle/distance constants
    ANTENNA_LARGE,
    ANTENNA_SMALL,
    DEFAULT_ANGLE,
    ANTENNA_MEDIUM,
    BODY_YAW_LARGE,
    BODY_YAW_SMALL,
    HEAD_YAW_LARGE,
    HEAD_YAW_SMALL,
    LARGE_KEYWORDS,
    SMALL_KEYWORDS,
    BODY_YAW_MEDIUM,
    CENTER_SYNONYMS,
    HEAD_YAW_MEDIUM,
    MEDIUM_KEYWORDS,
    DEFAULT_DISTANCE,
    # Other constants
    DEFAULT_DURATION,
    MAX_BODY_YAW_DEG,
    DURATION_KEYWORDS,
    MAX_HEAD_ROLL_DEG,
    TRANSLATION_LARGE,
    TRANSLATION_SMALL,
    ANTENNA_VERY_LARGE,
    ANTENNA_VERY_SMALL,
    MAX_HEAD_PITCH_DEG,
    TRANSLATION_MEDIUM,
    BODY_YAW_VERY_LARGE,
    # Context-aware qualitative constants
    BODY_YAW_VERY_SMALL,
    HEAD_YAW_VERY_LARGE,
    HEAD_YAW_VERY_SMALL,
    VERY_LARGE_KEYWORDS,
    VERY_SMALL_KEYWORDS,
    DEFAULT_ANTENNA_ANGLE,
    HEAD_PITCH_ROLL_LARGE,
    HEAD_PITCH_ROLL_SMALL,
    MAX_ANTENNA_ANGLE_DEG,
    ANTENNA_CLOCK_KEYWORDS,
    HEAD_PITCH_ROLL_MEDIUM,
    SAFE_ANTENNA_ANGLE_DEG,
    TRANSLATION_VERY_LARGE,
    TRANSLATION_VERY_SMALL,
    MAX_HEAD_TRANSLATION_X_MM,
    MAX_HEAD_TRANSLATION_Y_MM,
    MAX_HEAD_TRANSLATION_Z_MM,
    MIN_HEAD_TRANSLATION_Z_MM,
    ANTENNA_DIRECTION_KEYWORDS,
    HEAD_PITCH_ROLL_VERY_LARGE,
    HEAD_PITCH_ROLL_VERY_SMALL,
)


class SemanticAnalyzer:
    """Analyzes AST and generates intermediate representation."""

    def __init__(self) -> None:
        """Initialize semantic analyzer."""
        self.errors: List[CompilationError] = []
        self.warnings: List[CompilationError] = []

    @staticmethod
    def clock_to_angle(clock_position: float) -> float:
        """Convert clock position (0-12) to angle in degrees.

        Args:
            clock_position: Clock position from 0 to 12

        Returns:
            Angle in degrees, normalized to ±180°

        Examples:
            0 -> 0° (straight up)
            3 -> 90° (to the right)
            6 -> 180° (straight down)
            9 -> -90° (to the left)

        """
        # Convert clock position to angle: position * 30° (360/12)
        angle = clock_position * 30.0

        # Normalize to ±180° range
        if angle > 180:
            angle = angle - 360

        return angle

    def error(self, line: int, message: str) -> None:
        """Add an error."""
        self.errors.append(CompilationError(line=line, message=message))

    def warn(self, line: int, message: str) -> None:
        """Add a warning."""
        self.warnings.append(
            CompilationError(line=line, message=message, severity="warning")
        )

    def analyze(self, program: Program) -> List[Action | WaitAction | PictureAction | PlaySoundAction]:
        """Analyze program and generate IR."""
        ir: List[Action | WaitAction | PictureAction | PlaySoundAction] = []

        for statement in program.statements:
            ir.extend(self.analyze_statement(statement))

        return ir

    def analyze_statement(self, stmt: Statement) -> List[Action | WaitAction | PictureAction | PlaySoundAction]:
        """Analyze a single statement."""
        if isinstance(stmt, WaitStatement):
            return [self.analyze_wait(stmt)]
        elif isinstance(stmt, PictureStatement):
            return [self.analyze_picture(stmt)]
        elif isinstance(stmt, PlaySoundStatement):
            return [self.analyze_play_sound(stmt)]
        elif isinstance(stmt, ActionChain):
            return [self.analyze_action_chain(stmt)]
        elif isinstance(stmt, RepeatBlock):
            return self.analyze_repeat(stmt)
        else:
            self.error(0, f"Unknown statement type: {type(stmt)}")
            return []

    def analyze_wait(self, wait: WaitStatement) -> WaitAction:
        """Analyze wait statement."""
        # Resolve duration
        duration = DEFAULT_DURATION
        if wait.duration is not None:
            duration = wait.duration
        elif wait.duration_keyword is not None:
            duration = DURATION_KEYWORDS.get(wait.duration_keyword, DEFAULT_DURATION)

        return WaitAction(
            duration=duration, source_line=wait.line, original_text="wait"
        )

    def analyze_picture(self, picture: PictureStatement) -> PictureAction:
        """Analyze picture statement."""
        return PictureAction(
            source_line=picture.line, original_text="picture"
        )

    def analyze_play_sound(self, play: PlaySoundStatement) -> PlaySoundAction:
        """Analyze play sound statement."""
        return PlaySoundAction(
            sound_name=play.sound_name,
            blocking=play.blocking,
            source_line=play.line,
            original_text=f"play {play.sound_name}{'pause' if play.blocking else ''}"
        )

    def analyze_repeat(self, repeat: RepeatBlock) -> List[Action | WaitAction | PictureAction | PlaySoundAction]:
        """Analyze repeat block - expand it into IR."""
        if repeat.count < 0:
            self.error(repeat.line, f"Repeat count cannot be negative: {repeat.count}")
            return []

        if repeat.count == 0:
            self.warn(
                repeat.line,
                "Repeat count is 0, block will not execute",
            )
            return []

        # Analyze body once
        body_ir = []
        for stmt in repeat.body:
            body_ir.extend(self.analyze_statement(stmt))

        # Expand repeat
        expanded = []
        for _ in range(repeat.count):
            expanded.extend(body_ir)

        return expanded

    def analyze_action_chain(self, chain: ActionChain) -> Action:
        """Analyze action chain and merge into single Action."""
        # Resolve each action
        resolved_actions = [self.resolve_action(action) for action in chain.actions]

        # Merge all actions into a single Action
        merged = self.merge_actions(resolved_actions, chain.line)

        return merged

    def resolve_action(self, action: SingleAction) -> "ResolvedAction":
        """Resolve a single action - apply defaults and convert qualitative."""
        # Determine strength
        strength = self.resolve_strength(action)

        # Determine duration
        duration = self.resolve_duration(action)

        # Check ranges and issue warnings
        self.validate_ranges(action, strength)

        return ResolvedAction(
            keyword=action.keyword,
            direction=action.direction or "",
            antenna_modifier=action.antenna_modifier or "both",
            strength=strength,
            duration=duration,
            line=action.line,
        )

    def _get_qualitative_values(self, action: SingleAction) -> tuple[float, float, float, float, float, float]:
        """Get context-aware qualitative values based on movement type.

        Returns tuple of (very_small, small, medium, large, very_large, default).
        """
        if action.keyword == "head":
            # Head translations (distances in mm)
            return (TRANSLATION_VERY_SMALL, TRANSLATION_SMALL, TRANSLATION_MEDIUM,
                    TRANSLATION_LARGE, TRANSLATION_VERY_LARGE, DEFAULT_DISTANCE)

        elif action.keyword == "turn":
            # Body yaw rotation
            return (BODY_YAW_VERY_SMALL, BODY_YAW_SMALL, BODY_YAW_MEDIUM,
                    BODY_YAW_LARGE, BODY_YAW_VERY_LARGE, DEFAULT_ANGLE)

        elif action.keyword == "antenna":
            # Antenna rotations
            return (ANTENNA_VERY_SMALL, ANTENNA_SMALL, ANTENNA_MEDIUM,
                    ANTENNA_LARGE, ANTENNA_VERY_LARGE, DEFAULT_ANTENNA_ANGLE)

        elif action.keyword == "tilt":
            # Head roll - limited by cone constraint
            return (HEAD_PITCH_ROLL_VERY_SMALL, HEAD_PITCH_ROLL_SMALL, HEAD_PITCH_ROLL_MEDIUM,
                    HEAD_PITCH_ROLL_LARGE, HEAD_PITCH_ROLL_VERY_LARGE, DEFAULT_ANGLE)

        elif action.keyword == "look":
            # Head pitch/yaw - depends on direction
            if action.direction in ["up", "down"]:
                # Pitch - limited by cone constraint
                return (HEAD_PITCH_ROLL_VERY_SMALL, HEAD_PITCH_ROLL_SMALL, HEAD_PITCH_ROLL_MEDIUM,
                        HEAD_PITCH_ROLL_LARGE, HEAD_PITCH_ROLL_VERY_LARGE, DEFAULT_ANGLE)
            else:
                # Yaw (left/right) - larger range
                return (HEAD_YAW_VERY_SMALL, HEAD_YAW_SMALL, HEAD_YAW_MEDIUM,
                        HEAD_YAW_LARGE, HEAD_YAW_VERY_LARGE, DEFAULT_ANGLE)

        else:
            # Fallback to head yaw values
            return (HEAD_YAW_VERY_SMALL, HEAD_YAW_SMALL, HEAD_YAW_MEDIUM,
                    HEAD_YAW_LARGE, HEAD_YAW_VERY_LARGE, DEFAULT_ANGLE)

    def resolve_strength(self, action: SingleAction) -> float:
        """Resolve strength parameter with context-aware qualitative mapping."""
        # If quantitative specified, use it
        if action.strength is not None:
            if action.strength_qualitative is not None:
                self.warn(
                    action.line,
                    f"Both qualitative '{action.strength_qualitative}' and quantitative '{action.strength}' strength specified, using {action.strength}",
                )
            return action.strength

        # Get context-aware values
        very_small, small, medium, large, very_large, default = self._get_qualitative_values(action)

        # If qualitative specified, convert it using context-aware values
        if action.strength_qualitative is not None:
            if action.strength_qualitative in VERY_SMALL_KEYWORDS:
                return float(very_small)
            elif action.strength_qualitative in SMALL_KEYWORDS:
                return float(small)
            elif action.strength_qualitative in MEDIUM_KEYWORDS:
                return float(medium)
            elif action.strength_qualitative in LARGE_KEYWORDS:
                return float(large)
            elif action.strength_qualitative in VERY_LARGE_KEYWORDS:
                return float(very_large)

        # Use context-aware default
        return float(default)

    def resolve_duration(self, action: SingleAction) -> float:
        """Resolve duration parameter."""
        if action.duration is not None:
            if action.duration < 0.1:
                self.warn(
                    action.line,
                    f"Very short duration ({action.duration}s) may cause jerky motion",
                )
            return action.duration

        if action.duration_keyword is not None:
            return DURATION_KEYWORDS.get(action.duration_keyword, DEFAULT_DURATION)

        return DEFAULT_DURATION

    def validate_ranges(self, action: SingleAction, strength: float) -> None:
        """Validate ranges and issue warnings."""
        if action.keyword == "turn":
            if abs(strength) > MAX_BODY_YAW_DEG:
                self.warn(
                    action.line,
                    f"Body yaw {strength}° exceeds safe range (±{MAX_BODY_YAW_DEG}°), will be clamped",
                )

        elif action.keyword == "look":
            if action.direction in ["up", "down"]:  # pitch
                if abs(strength) > MAX_HEAD_PITCH_DEG:
                    self.warn(
                        action.line,
                        f"Head pitch {strength}° exceeds limit (±{MAX_HEAD_PITCH_DEG}°), will be clamped",
                    )

        elif action.keyword == "tilt":
            if abs(strength) > MAX_HEAD_ROLL_DEG:
                self.warn(
                    action.line,
                    f"Head roll {strength}° exceeds limit (±{MAX_HEAD_ROLL_DEG}°), will be clamped",
                )

        elif action.keyword == "head":
            if action.direction in ["forward", "back"]:
                if abs(strength) > MAX_HEAD_TRANSLATION_X_MM:
                    self.warn(
                        action.line,
                        f"Head X translation {strength}mm exceeds typical range (±{MAX_HEAD_TRANSLATION_X_MM}mm)",
                    )
            elif action.direction in ["left", "right"]:
                if abs(strength) > MAX_HEAD_TRANSLATION_Y_MM:
                    self.warn(
                        action.line,
                        f"Head Y translation {strength}mm exceeds typical range (±{MAX_HEAD_TRANSLATION_Y_MM}mm)",
                    )
            elif action.direction == "up":
                if strength > MAX_HEAD_TRANSLATION_Z_MM:
                    self.warn(
                        action.line,
                        f"Head Z translation {strength}mm exceeds typical range ({MAX_HEAD_TRANSLATION_Z_MM}mm max)",
                    )
            elif action.direction == "down":
                if strength > abs(MIN_HEAD_TRANSLATION_Z_MM):
                    self.warn(
                        action.line,
                        f"Head Z translation {strength}mm exceeds typical range ({MIN_HEAD_TRANSLATION_Z_MM}mm min)",
                    )

        elif action.keyword == "antenna":
            if abs(strength) > MAX_ANTENNA_ANGLE_DEG:
                self.warn(
                    action.line,
                    f"Antenna angle {strength}° exceeds maximum (±{MAX_ANTENNA_ANGLE_DEG}°), will be clamped",
                )
            elif abs(strength) > SAFE_ANTENNA_ANGLE_DEG:
                self.warn(
                    action.line,
                    f"Antenna angle {strength}° exceeds recommended safe range (±{SAFE_ANTENNA_ANGLE_DEG}°), may cause collision",
                )

    def merge_actions(
        self, actions: List["ResolvedAction"], line: int
    ) -> Action:  # noqa: F821
        """Merge multiple actions into a single Action IR node."""
        # Extract parameters
        head_pose_params = {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        has_head_movement = False

        antennas = [0.0, 0.0]  # [right, left]
        has_antenna_movement = False

        body_yaw = 0.0
        has_body_yaw = False

        # Use the maximum duration
        max_duration = max((a.duration for a in actions), default=DEFAULT_DURATION)

        for action in actions:
            if action.keyword == "turn":
                has_body_yaw = True
                has_head_movement = True  # Head rotates with body
                if action.direction in CENTER_SYNONYMS:
                    body_yaw = 0.0
                    head_pose_params["yaw"] = 0.0
                elif action.direction == "left":
                    body_yaw += action.strength  # LEFT = positive yaw
                    head_pose_params["yaw"] += action.strength  # Head follows body
                elif action.direction == "right":
                    body_yaw -= action.strength  # RIGHT = negative yaw
                    head_pose_params["yaw"] -= action.strength  # Head follows body

            elif action.keyword == "look":
                has_head_movement = True
                if action.direction in CENTER_SYNONYMS:
                    head_pose_params["yaw"] = 0.0
                    head_pose_params["pitch"] = 0.0
                elif action.direction == "left":
                    head_pose_params["yaw"] += action.strength  # LEFT = positive yaw
                elif action.direction == "right":
                    head_pose_params["yaw"] -= action.strength  # RIGHT = negative yaw
                elif action.direction == "up":
                    head_pose_params["pitch"] -= action.strength  # UP = negative pitch
                elif action.direction == "down":
                    head_pose_params["pitch"] += action.strength  # DOWN = positive pitch

            elif action.keyword == "head":
                has_head_movement = True
                if action.direction == "forward":
                    head_pose_params["x"] += action.strength
                elif action.direction == "back":
                    head_pose_params["x"] -= action.strength
                elif action.direction == "left":
                    head_pose_params["y"] += action.strength
                elif action.direction == "right":
                    head_pose_params["y"] -= action.strength
                elif action.direction == "up":
                    head_pose_params["z"] += action.strength
                elif action.direction == "down":
                    head_pose_params["z"] -= action.strength

            elif action.keyword == "tilt":
                has_head_movement = True
                if action.direction in CENTER_SYNONYMS:
                    head_pose_params["roll"] = 0.0
                elif action.direction == "left":
                    head_pose_params["roll"] += action.strength
                elif action.direction == "right":
                    head_pose_params["roll"] -= action.strength

            elif action.keyword == "antenna":
                has_antenna_movement = True

                # Determine which antenna(s) to move
                # Note: antennas array is [left_antenna, right_antenna]
                if action.antenna_modifier == "both":
                    left_idx, right_idx = 0, 1
                elif action.antenna_modifier == "right":
                    left_idx, right_idx = None, 1
                elif action.antenna_modifier == "left":
                    left_idx, right_idx = 0, None
                else:
                    left_idx, right_idx = 0, 1  # default to both

                # Convert clock position to angle
                # If direction is "clock", the clock position is stored in strength
                # If direction is a clock keyword (high/low/ext/int), look it up
                if action.direction == "clock":
                    # Clock position as number (already in strength)
                    clock_pos = action.strength
                elif action.direction in ANTENNA_CLOCK_KEYWORDS:
                    # Clock keyword - look up the position
                    clock_pos = ANTENNA_CLOCK_KEYWORDS[action.direction]
                elif action.direction in ANTENNA_DIRECTION_KEYWORDS:
                    # Directional keyword (up/down/left/right) - look up the position
                    clock_pos = ANTENNA_DIRECTION_KEYWORDS[action.direction]
                else:
                    # Shouldn't happen due to parser validation, but fallback
                    clock_pos = 0.0

                # Convert clock position to angle
                angle = self.clock_to_angle(clock_pos)

                # Set antenna positions (same angle for specified antenna(s))
                if right_idx is not None:
                    antennas[right_idx] = angle
                if left_idx is not None:
                    antennas[left_idx] = angle

        # Build Action IR node
        result = Action(
            source_line=line,
            duration=max_duration,
        )

        if has_head_movement:
            result.head_pose = create_head_pose(
                x=head_pose_params["x"],
                y=head_pose_params["y"],
                z=head_pose_params["z"],
                roll=head_pose_params["roll"],
                pitch=head_pose_params["pitch"],
                yaw=head_pose_params["yaw"],
                mm=True,
                degrees=True,
            )

        if has_antenna_movement:
            result.antennas = [np.deg2rad(antennas[0]), np.deg2rad(antennas[1])]

        if has_body_yaw:
            result.body_yaw = np.deg2rad(body_yaw)

        return result


class ResolvedAction:
    """Temporary structure for resolved action parameters."""

    def __init__(
        self,
        keyword: str,
        direction: str,
        antenna_modifier: str,
        strength: float,
        duration: float,
        line: int,
    ):
        """Initialize resolved action."""
        self.keyword = keyword
        self.direction = direction
        self.antenna_modifier = antenna_modifier
        self.strength = strength
        self.duration = duration
        self.line = line
