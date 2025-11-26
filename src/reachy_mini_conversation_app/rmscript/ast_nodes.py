"""Abstract Syntax Tree node definitions for ReachyMiniScript."""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ASTNode:
    """Base class for all AST nodes."""

    line: int = 0
    column: int = 0


@dataclass
class Program(ASTNode):
    """Root node of the AST."""

    tool_name: str = ""
    description: str = ""
    statements: List["Statement"] = None  # type: ignore

    def __post_init__(self) -> None:
        """Initialize mutable defaults."""
        if self.statements is None:
            self.statements = []


@dataclass
class Statement(ASTNode):
    """Base class for statements."""

    pass


@dataclass
class ActionChain(Statement):
    """Chain of actions connected by 'and' keyword."""

    actions: List["SingleAction"] = None  # type: ignore

    def __post_init__(self) -> None:
        """Initialize mutable defaults."""
        if self.actions is None:
            self.actions = []


@dataclass
class SingleAction(ASTNode):
    """Single action (may be part of a chain)."""

    keyword: str = ""  # turn, look, head, tilt, antenna
    direction: Optional[str] = None  # left, right, up, down, etc.
    antenna_modifier: Optional[str] = None  # left, right, both (for antenna only)
    strength: Optional[float] = None  # Numerical value (degrees or mm)
    strength_qualitative: Optional[str] = None  # little, lot, etc.
    duration: Optional[float] = None  # seconds
    duration_keyword: Optional[str] = None  # fast, slow, etc.


@dataclass
class RepeatBlock(Statement):
    """Repeat block with nested statements."""

    count: int = 0
    body: List[Statement] = None  # type: ignore

    def __post_init__(self) -> None:
        """Initialize mutable defaults."""
        if self.body is None:
            self.body = []


@dataclass
class WaitStatement(Statement):
    """Wait/pause statement."""

    duration: Optional[float] = None
    duration_keyword: Optional[str] = None


@dataclass
class PictureStatement(Statement):
    """Take a picture statement."""

    pass


@dataclass
class PlaySoundStatement(Statement):
    """Play a sound statement."""

    sound_name: str = ""  # Name of the sound file (without extension)
    blocking: bool = False  # True = wait for sound to finish
    loop: bool = False  # True = loop the sound
    duration: Optional[float] = None  # Duration for looping (or limited playback)
