"""Lexer for ReachyMiniScript - tokenization and indentation handling."""

from enum import Enum, auto
from typing import List, Optional
from dataclasses import dataclass


class TokenType(Enum):
    """Token types for ReachyMiniScript."""

    # Keywords
    KEYWORD_TURN = auto()
    KEYWORD_LOOK = auto()
    KEYWORD_HEAD = auto()
    KEYWORD_TILT = auto()
    KEYWORD_ANTENNA = auto()
    KEYWORD_WAIT = auto()
    KEYWORD_PICTURE = auto()
    KEYWORD_PLAY = auto()
    KEYWORD_LOOP = auto()
    KEYWORD_REPEAT = auto()
    KEYWORD_END = auto()
    KEYWORD_DESCRIPTION = auto()

    # Directions
    DIRECTION = auto()  # left, right, up, down, center, etc.

    # Values
    NUMBER = auto()  # 30, 2.5
    DURATION = auto()  # 2s, 0.5s
    DURATION_KEYWORD = auto()  # fast, slow, superfast, superslow
    QUALITATIVE = auto()  # little, lot, etc.
    ANTENNA_CLOCK = auto()  # high, low, ext, int (clock position keywords)
    SOUND_BLOCKING = auto()  # pause, fully, wait, block, complete (for sound playback)

    # Operators
    AND = auto()  # and

    # Structure
    INDENT = auto()
    DEDENT = auto()
    NEWLINE = auto()
    EOF = auto()

    # Identifiers
    IDENTIFIER = auto()  # For tool names and other identifiers


@dataclass
class Token:
    """Represents a lexical token."""

    type: TokenType
    value: str
    line: int
    column: int

    def __repr__(self) -> str:
        """Turn as string representation for debugging."""
        return f"Token({self.type.name}, {self.value!r}, L{self.line}:C{self.column})"


class Lexer:
    """Tokenizer for ReachyMiniScript with indentation handling."""

    def __init__(self, source: str):
        """Initialize lexer with source code."""
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []
        self.indent_stack: List[int] = [0]  # Track indentation levels

        # Keywords (case-insensitive)
        self.keywords = {
            "turn": TokenType.KEYWORD_TURN,
            "look": TokenType.KEYWORD_LOOK,
            "head": TokenType.KEYWORD_HEAD,
            "tilt": TokenType.KEYWORD_TILT,
            "antenna": TokenType.KEYWORD_ANTENNA,
            "wait": TokenType.KEYWORD_WAIT,
            "picture": TokenType.KEYWORD_PICTURE,
            "play": TokenType.KEYWORD_PLAY,
            "loop": TokenType.KEYWORD_LOOP,
            "repeat": TokenType.KEYWORD_REPEAT,
            "end": TokenType.KEYWORD_END,
            "description": TokenType.KEYWORD_DESCRIPTION,
        }

        # Directions - import from constants
        from reachy_mini_conversation_app.rmscript.constants import ALL_DIRECTIONS
        self.directions = set(ALL_DIRECTIONS)

        # Duration keywords - import from constants
        from reachy_mini_conversation_app.rmscript.constants import DURATION_KEYWORDS
        self.duration_keywords = set(DURATION_KEYWORDS.keys())

        # Qualitative strength - import from constants
        from reachy_mini_conversation_app.rmscript.constants import (
            LARGE_KEYWORDS,
            SMALL_KEYWORDS,
            MEDIUM_KEYWORDS,
            VERY_LARGE_KEYWORDS,
            VERY_SMALL_KEYWORDS,
        )
        self.qualitative_keywords = (
            set(SMALL_KEYWORDS)
            | set(MEDIUM_KEYWORDS)
            | set(LARGE_KEYWORDS)
            | set(VERY_SMALL_KEYWORDS)
            | set(VERY_LARGE_KEYWORDS)
        )

        # Antenna clock keywords
        self.antenna_clock_keywords = {"high", "low", "ext", "int"}

        # Sound blocking keywords (for "play sound pause/fully")
        from reachy_mini_conversation_app.rmscript.constants import SOUND_BLOCKING_KEYWORDS
        self.sound_blocking_keywords = set(SOUND_BLOCKING_KEYWORDS)

    def error(self, message: str) -> Exception:
        """Create a lexer error with position information."""
        return SyntaxError(f"Line {self.line}, Column {self.column}: {message}")

    def peek(self, offset: int = 0) -> Optional[str]:
        """Peek at character at current position + offset."""
        pos = self.pos + offset
        if pos < len(self.source):
            return self.source[pos]
        return None

    def advance(self) -> Optional[str]:
        """Advance position and return current character."""
        if self.pos >= len(self.source):
            return None

        char = self.source[self.pos]
        self.pos += 1

        if char == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1

        return char

    def skip_whitespace_inline(self) -> None:
        """Skip spaces and tabs but not newlines."""
        while self.peek() in (" ", "\t"):
            self.advance()

    def skip_comment(self) -> None:
        """Skip comment (from # to end of line)."""
        if self.peek() == "#":
            while self.peek() and self.peek() != "\n":
                self.advance()

    def read_number(self) -> Token:
        """Read a number token (integer or float)."""
        start_col = self.column
        num_str = ""

        while (ch := self.peek()) is not None and (ch.isdigit() or ch == "."):
            num_str += self.advance()  # type: ignore

        # Check if followed by 's' for duration
        if self.peek() == "s":
            self.advance()
            return Token(
                TokenType.DURATION, num_str + "s", self.line, start_col
            )  # Keep 's' in value

        return Token(TokenType.NUMBER, num_str, self.line, start_col)

    def read_identifier(self) -> Token:
        """Read an identifier or keyword."""
        start_col = self.column
        ident = ""

        while (ch := self.peek()) is not None and (ch.isalnum() or ch == "_"):
            ident += self.advance()  # type: ignore

        ident_lower = ident.lower()

        # Check if it's a keyword
        if ident_lower in self.keywords:
            return Token(self.keywords[ident_lower], ident, self.line, start_col)

        # Check if it's a direction
        if ident_lower in self.directions:
            return Token(TokenType.DIRECTION, ident_lower, self.line, start_col)

        # Check if it's a duration keyword
        if ident_lower in self.duration_keywords:
            return Token(TokenType.DURATION_KEYWORD, ident_lower, self.line, start_col)

        # Check if it's a qualitative keyword
        if ident_lower in self.qualitative_keywords:
            return Token(TokenType.QUALITATIVE, ident_lower, self.line, start_col)

        # Check if it's an antenna clock keyword
        if ident_lower in self.antenna_clock_keywords:
            return Token(TokenType.ANTENNA_CLOCK, ident_lower, self.line, start_col)

        # Check if it's a sound blocking keyword
        if ident_lower in self.sound_blocking_keywords:
            return Token(TokenType.SOUND_BLOCKING, ident_lower, self.line, start_col)

        # Check for 'and'
        if ident_lower == "and":
            return Token(TokenType.AND, ident_lower, self.line, start_col)

        # Otherwise it's a generic identifier
        return Token(TokenType.IDENTIFIER, ident, self.line, start_col)

    def handle_indentation(self, indent_level: int) -> List[Token]:
        """Generate INDENT/DEDENT tokens based on indentation level."""
        tokens = []

        if indent_level > self.indent_stack[-1]:
            # Indent
            self.indent_stack.append(indent_level)
            tokens.append(Token(TokenType.INDENT, "", self.line, 1))

        elif indent_level < self.indent_stack[-1]:
            # Dedent (possibly multiple levels)
            while self.indent_stack and indent_level < self.indent_stack[-1]:
                self.indent_stack.pop()
                tokens.append(Token(TokenType.DEDENT, "", self.line, 1))

            # Check if we dedented to a valid level
            if not self.indent_stack or indent_level != self.indent_stack[-1]:
                raise self.error(f"Inconsistent indentation (level {indent_level})")

        return tokens

    def tokenize(self) -> List[Token]:
        """Tokenize the entire source code."""
        tokens = []
        at_line_start = True

        while self.pos < len(self.source):
            # Handle indentation at start of line BEFORE skipping whitespace
            if at_line_start:
                # Count indentation
                indent_level = 0
                temp_pos = self.pos
                while temp_pos < len(self.source) and self.source[temp_pos] in (
                    " ",
                    "\t",
                ):
                    if self.source[temp_pos] == " ":
                        indent_level += 1
                    else:  # tab
                        indent_level += 4  # Treat tab as 4 spaces
                    temp_pos += 1

                # Check if this is a blank line or comment-only line
                if (
                    temp_pos >= len(self.source)
                    or self.source[temp_pos] == "\n"
                    or self.source[temp_pos] == "#"
                ):
                    # Skip blank/comment lines - consume the line
                    while self.peek() and self.peek() != "\n":
                        self.advance()
                    if self.peek() == "\n":
                        self.advance()
                    continue

                # Generate INDENT/DEDENT tokens
                indent_tokens = self.handle_indentation(indent_level)
                tokens.extend(indent_tokens)

                at_line_start = False

            # Skip inline whitespace
            self.skip_whitespace_inline()

            # Skip comments
            self.skip_comment()

            # Check for EOF
            if self.pos >= len(self.source):
                break

            # Handle newlines
            if self.peek() == "\n":
                self.advance()
                tokens.append(Token(TokenType.NEWLINE, "\\n", self.line - 1, 1))
                at_line_start = True
                continue

            # Read numbers
            if (ch := self.peek()) is not None and ch.isdigit():
                tokens.append(self.read_number())
                continue

            # Read identifiers/keywords
            if (ch := self.peek()) is not None and (ch.isalpha() or ch == "_"):
                tokens.append(self.read_identifier())
                continue

            # Unknown character
            raise self.error(f"Unexpected character: {self.peek()!r}")

        # Close any remaining indentation levels
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            tokens.append(Token(TokenType.DEDENT, "", self.line, 1))

        # Add EOF token
        tokens.append(Token(TokenType.EOF, "", self.line, self.column))

        return tokens


def tokenize(source: str) -> List[Token]:
    """Tokenize source code, convenience function."""
    lexer = Lexer(source)
    return lexer.tokenize()
