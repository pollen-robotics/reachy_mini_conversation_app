"""Constants and default values for ReachyMiniScript."""

from typing import Dict, List


# Default Strengths
DEFAULT_ANGLE = 30  # degrees (for turn, look, tilt)
DEFAULT_DISTANCE = 10  # mm (for head translation)
DEFAULT_ANTENNA_ANGLE = 45  # degrees

# Qualitative Strength Mappings - Context-Aware
# Different movement types have different physical limits

# For BODY YAW (turn command) - can handle larger angles
BODY_YAW_VERY_SMALL = 10  # degrees
BODY_YAW_SMALL = 30  # degrees
BODY_YAW_MEDIUM = 60  # degrees
BODY_YAW_LARGE = 90  # degrees
BODY_YAW_VERY_LARGE = 120  # degrees (safe limit, max is ±160°)

# For HEAD PITCH/ROLL (look up/down, tilt) - limited by cone constraint
HEAD_PITCH_ROLL_VERY_SMALL = 5  # degrees
HEAD_PITCH_ROLL_SMALL = 10  # degrees
HEAD_PITCH_ROLL_MEDIUM = 20  # degrees
HEAD_PITCH_ROLL_LARGE = 30  # degrees
HEAD_PITCH_ROLL_VERY_LARGE = 38  # degrees (just under ±40° limit)

# For HEAD YAW (look left/right) - larger range possible
HEAD_YAW_VERY_SMALL = 5  # degrees
HEAD_YAW_SMALL = 15  # degrees
HEAD_YAW_MEDIUM = 30  # degrees
HEAD_YAW_LARGE = 45  # degrees
HEAD_YAW_VERY_LARGE = 60  # degrees (respecting ±65° body-head differential)

# For ANTENNAS - safe operating range
ANTENNA_VERY_SMALL = 10  # degrees
ANTENNA_SMALL = 30  # degrees
ANTENNA_MEDIUM = 60  # degrees
ANTENNA_LARGE = 90  # degrees
ANTENNA_VERY_LARGE = 110  # degrees (safe limit, under ±120°)

# For HEAD TRANSLATIONS (distances in mm)
TRANSLATION_VERY_SMALL = 2  # mm
TRANSLATION_SMALL = 5  # mm
TRANSLATION_MEDIUM = 10  # mm
TRANSLATION_LARGE = 20  # mm
TRANSLATION_VERY_LARGE = 28  # mm (just under ±30mm limit)

# Legacy constants (deprecated - use context-aware versions above)
VERY_SMALL_ANGLE = HEAD_YAW_VERY_SMALL
VERY_SMALL_DISTANCE = TRANSLATION_VERY_SMALL
SMALL_ANGLE = HEAD_YAW_SMALL
SMALL_DISTANCE = TRANSLATION_SMALL
MEDIUM_ANGLE = HEAD_YAW_MEDIUM
MEDIUM_DISTANCE = TRANSLATION_MEDIUM
LARGE_ANGLE = HEAD_YAW_LARGE
LARGE_DISTANCE = TRANSLATION_LARGE
VERY_LARGE_ANGLE = HEAD_YAW_VERY_LARGE
VERY_LARGE_DISTANCE = TRANSLATION_VERY_LARGE

# Duration Mappings
DEFAULT_DURATION = 1.0  # seconds

DURATION_KEYWORDS: Dict[str, float] = {
    "superfast": 0.2,
    "veryfast": 0.2,
    "fast": 0.5,
    "slow": 2.0,
    "veryslow": 3.0,
    "superslow": 3.0,
}

# Qualitative keywords
SMALL_KEYWORDS: List[str] = ["little", "slightly", "small", "alittle"]
MEDIUM_KEYWORDS: List[str] = ["medium", "normal", "regular", "standard", "normally"]
LARGE_KEYWORDS: List[str] = ["lot", "big", "large", "very", "alot", "huge", "strong", "strongly"]
VERY_SMALL_KEYWORDS: List[str] = ["minuscule", "mini", "verysmall", "tiny"]
VERY_LARGE_KEYWORDS: List[str] = ["verybig", "enormous","verylarge","maximum"]

# Direction Synonyms
CENTER_SYNONYMS: List[str] = ["center", "straight", "forward", "neutral"]
INWARD_SYNONYMS: List[str] = ["in", "inside", "inward"]
OUTWARD_SYNONYMS: List[str] = ["out", "outside", "outward"]

# Physical Limits (from robot constraints)
MAX_BODY_YAW_DEG = 160.0  # degrees
MAX_HEAD_PITCH_DEG = 40.0  # degrees (cone constraint)
MAX_HEAD_ROLL_DEG = 40.0  # degrees (cone constraint)
MAX_HEAD_YAW_DEG = 180.0  # degrees (absolute)
MAX_HEAD_BODY_YAW_DIFF_DEG = 65.0  # degrees (relative constraint)

MAX_HEAD_TRANSLATION_X_MM = 30.0  # mm
MAX_HEAD_TRANSLATION_Y_MM = 30.0  # mm
MAX_HEAD_TRANSLATION_Z_MM = 40.0  # mm (positive)
MIN_HEAD_TRANSLATION_Z_MM = -20.0  # mm (negative)

MAX_ANTENNA_ANGLE_DEG = 180.0  # degrees
SAFE_ANTENNA_ANGLE_DEG = 120.0  # recommended maximum for safety

# Movement Keywords
MOVEMENT_KEYWORDS: List[str] = ["turn", "look", "head", "tilt", "antenna"]

# Control Keywords
CONTROL_KEYWORDS: List[str] = ["wait", "repeat", "end", "play", "picture"]

# Sound Playback Mode Keywords
SOUND_BLOCKING_KEYWORDS: List[str] = ["pause", "fully", "wait", "block", "complete"]

# Valid Directions per Keyword
TURN_DIRECTIONS: List[str] = ["left", "right"] + CENTER_SYNONYMS
LOOK_DIRECTIONS: List[str] = ["left", "right", "up", "down"] + CENTER_SYNONYMS
HEAD_DIRECTIONS: List[str] = ["forward", "back", "left", "right", "up", "down"]
TILT_DIRECTIONS: List[str] = ["left", "right"] + CENTER_SYNONYMS

# Antenna movements (OLD SYSTEM - kept for backward compatibility if needed)
ANTENNA_MOVEMENTS: List[str] = ["up", "down"] + INWARD_SYNONYMS + OUTWARD_SYNONYMS
ANTENNA_MODIFIERS: List[str] = ["left", "right", "both"]

# Antenna clock system (NEW)
ANTENNA_CLOCK_KEYWORDS: Dict[str, float] = {
    "high": 0.0,  # 0 o'clock = 0°
    "ext": 3.0,   # 3 o'clock = +90° (external)
    "low": 6.0,   # 6 o'clock = ±180°
    "int": 9.0,   # 9 o'clock = -90° (internal)
}

# Antenna directional keywords (map to clock positions)
ANTENNA_DIRECTION_KEYWORDS: Dict[str, float] = {
    "up": 0.0,     # Same as high: 0°
    "right": 3.0,  # Same as ext: +90° (external/right)
    "down": 6.0,   # Same as low: ±180°
    "left": 9.0,   # Same as int: -90° (internal/left)
}

# Direction Sign Conventions (for coordinate system)
# Note: These match the robot's actual coordinate system
# Positive values
POSITIVE_DIRECTIONS = {
    "left": "yaw",  # Positive yaw (verified from robot)
    "down": "pitch",  # Positive pitch (verified from robot)
    "forward": "x",  # Positive X
}

# Negative values
NEGATIVE_DIRECTIONS = {
    "right": "yaw",  # Negative yaw (verified from robot)
    "up": "pitch",  # Negative pitch (verified from robot)
    "back": "x",  # Negative X
}
