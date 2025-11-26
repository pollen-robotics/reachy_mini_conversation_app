"""Parser for ReachyMiniScript - builds Abstract Syntax Tree from tokens."""

from typing import List, Optional

from reachy_mini_conversation_app.rmscript.lexer import Token, TokenType
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
    HEAD_DIRECTIONS,
    LOOK_DIRECTIONS,
    TILT_DIRECTIONS,
    TURN_DIRECTIONS,
    ANTENNA_MODIFIERS,
    ANTENNA_DIRECTION_KEYWORDS,
)


class ParseError(Exception):
    """Parser error with position information."""

    def __init__(self, message: str, token: Token):
        """Initialize parse error."""
        self.message = message
        self.token = token
        super().__init__(f"Line {token.line}: {message}")


class Parser:
    """Parser for ReachyMiniScript."""

    def __init__(self, tokens: List[Token]):
        """Initialize parser with token stream."""
        self.tokens = tokens
        self.pos = 0

    def error(self, message: str) -> ParseError:
        """Create parse error at current position."""
        return ParseError(message, self.current())

    def current(self) -> Token:
        """Get current token."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # Return EOF token

    def peek(self, offset: int = 0) -> Token:
        """Peek at token at current position + offset."""
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return self.tokens[-1]  # Return EOF token

    def advance(self) -> Token:
        """Consume and return current token."""
        token = self.current()
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token

    def expect(self, token_type: TokenType) -> Token:
        """Consume token of expected type or raise error."""
        token = self.current()
        if token.type != token_type:
            raise self.error(
                f"Expected {token_type.name}, got {token.type.name} ({token.value!r})"
            )
        return self.advance()

    def skip_newlines(self) -> None:
        """Skip any newline tokens."""
        while self.current().type == TokenType.NEWLINE:
            self.advance()

    def parse(self) -> Program:
        """Parse entire program."""
        program = Program()

        self.skip_newlines()

        # Parse optional DESCRIPTION header
        if self.current().type == TokenType.KEYWORD_DESCRIPTION:
            self.advance()  # Skip DESCRIPTION keyword

            # Read the rest of the line as the description
            description_parts = []
            while self.current().type not in (TokenType.NEWLINE, TokenType.EOF):
                description_parts.append(self.current().value)
                self.advance()
            program.description = " ".join(description_parts)
            self.skip_newlines()

        # Note: tool_name will be set from filename by the compiler
        program.tool_name = ""

        # Parse statements
        program.statements = self.parse_statements()

        return program

    def parse_statements(self) -> List[Statement]:
        """Parse a list of statements."""
        statements = []

        while self.current().type not in (
            TokenType.EOF,
            TokenType.DEDENT,
            TokenType.KEYWORD_END,
        ):
            self.skip_newlines()

            if self.current().type in (
                TokenType.EOF,
                TokenType.DEDENT,
                TokenType.KEYWORD_END,
            ):
                break

            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)

            self.skip_newlines()

        return statements

    def parse_statement(self) -> Optional[Statement]:
        """Parse a single statement."""
        token = self.current()

        # Check for repeat block
        if token.type == TokenType.KEYWORD_REPEAT:
            return self.parse_repeat_block()

        # Check for wait command
        if token.type == TokenType.KEYWORD_WAIT:
            return self.parse_wait()

        # Check for picture command
        if token.type == TokenType.KEYWORD_PICTURE:
            return self.parse_picture()

        # Check for play command
        if token.type == TokenType.KEYWORD_PLAY:
            return self.parse_play()

        # Check for loop command
        if token.type == TokenType.KEYWORD_LOOP:
            return self.parse_loop()

        # Check for movement keywords
        if token.type in (
            TokenType.KEYWORD_TURN,
            TokenType.KEYWORD_LOOK,
            TokenType.KEYWORD_HEAD,
            TokenType.KEYWORD_TILT,
            TokenType.KEYWORD_ANTENNA,
        ):
            return self.parse_action_chain()

        # Unknown statement
        if token.type != TokenType.NEWLINE:
            raise self.error(f"Unexpected token: {token.value!r}")

        return None

    def parse_wait(self) -> WaitStatement:
        """Parse wait command."""
        token = self.current()
        self.advance()  # consume 'wait'

        wait = WaitStatement(line=token.line, column=token.column)

        # Parse duration
        if self.current().type == TokenType.DURATION:
            duration_str = self.current().value.rstrip("s")
            wait.duration = float(duration_str)
            self.advance()
        elif self.current().type == TokenType.DURATION_KEYWORD:
            wait.duration_keyword = self.current().value
            self.advance()
        elif self.current().type == TokenType.NUMBER:
            wait.duration = float(self.current().value)
            self.advance()
            # Require 's' suffix for consistency
            if self.current().value != "s":
                raise self.error(
                    f"Expected 's' after wait duration (e.g., 'wait {wait.duration}s')"
                )
            self.advance()
        else:
            raise self.error("Expected duration after 'wait' (e.g., 'wait 1s')")

        return wait

    def parse_picture(self) -> PictureStatement:
        """Parse picture command."""
        token = self.current()
        self.advance()  # consume 'picture'

        return PictureStatement(line=token.line, column=token.column)

    def parse_play(self) -> PlaySoundStatement:
        """Parse play command: 'play soundname', 'play soundname pause', or 'play soundname 10s'."""
        token = self.current()
        self.advance()  # consume 'play'

        play = PlaySoundStatement(line=token.line, column=token.column)

        # Expect sound name (identifier)
        if self.current().type != TokenType.IDENTIFIER:
            raise self.error(f"Expected sound name after 'play', got '{self.current().value}'")

        play.sound_name = self.current().value
        self.advance()

        # Check for optional duration or blocking modifier
        if self.current().type == TokenType.DURATION:
            # Duration specified (e.g., "play mysound 10s")
            duration_str = self.current().value.rstrip("s")
            play.duration = float(duration_str)
            play.blocking = True  # Duration implies blocking
            self.advance()
        elif self.current().type == TokenType.SOUND_BLOCKING:
            # Blocking modifier (pause, fully, wait, block, complete)
            play.blocking = True
            self.advance()

        return play

    def parse_loop(self) -> PlaySoundStatement:
        """Parse loop command: 'loop soundname' or 'loop soundname 10s'."""
        token = self.current()
        self.advance()  # consume 'loop'

        play = PlaySoundStatement(line=token.line, column=token.column, loop=True, blocking=True)

        # Expect sound name (identifier)
        if self.current().type != TokenType.IDENTIFIER:
            raise self.error(f"Expected sound name after 'loop', got '{self.current().value}'")

        play.sound_name = self.current().value
        self.advance()

        # Check for optional duration (default to 10s if not specified)
        if self.current().type == TokenType.DURATION:
            duration_str = self.current().value.rstrip("s")
            play.duration = float(duration_str)
            self.advance()
        else:
            # Default duration for loop is 10s
            play.duration = 10.0

        return play

    def parse_repeat_block(self) -> RepeatBlock:
        """Parse repeat block."""
        token = self.current()
        self.advance()  # consume 'repeat'

        # Parse count
        if self.current().type != TokenType.NUMBER:
            raise self.error("Expected number after 'repeat'")

        count = int(float(self.current().value))
        self.advance()

        self.skip_newlines()

        # Expect indent
        if self.current().type != TokenType.INDENT:
            raise self.error("Expected indented block after 'repeat'")
        self.advance()

        # Parse body
        body = self.parse_statements()

        # Expect dedent (this closes the block)
        if self.current().type == TokenType.DEDENT:
            self.advance()
        # Note: 'end' keyword is no longer required - indentation defines the block

        return RepeatBlock(count=count, body=body, line=token.line, column=token.column)

    def parse_action_chain(self) -> ActionChain:
        """Parse action chain (one or more actions connected by 'and')."""
        chain = ActionChain()
        chain.line = self.current().line
        chain.column = self.current().column

        # Parse first action
        first_action = self.parse_single_action()
        chain.actions.append(first_action)

        # Parse additional actions connected by 'and'
        while self.current().type == TokenType.AND:
            self.advance()  # consume 'and'

            # Check if next token is a keyword
            next_action = self.parse_single_action(
                previous_keyword=first_action.keyword
            )
            chain.actions.append(next_action)

        return chain

    def parse_single_action(
        self, previous_keyword: Optional[str] = None
    ) -> SingleAction:
        """Parse a single action.

        Args:
            previous_keyword: If provided and no keyword is present, reuse this keyword.

        """
        token = self.current()
        action = SingleAction(keyword="", line=token.line, column=token.column)

        # Check if we have a movement keyword
        if token.type in (
            TokenType.KEYWORD_TURN,
            TokenType.KEYWORD_LOOK,
            TokenType.KEYWORD_HEAD,
            TokenType.KEYWORD_TILT,
            TokenType.KEYWORD_ANTENNA,
        ):
            action.keyword = token.value.lower()
            self.advance()
        elif previous_keyword:
            # Check if current token is a control keyword (not allowed after 'and')
            if token.type in (TokenType.KEYWORD_PICTURE, TokenType.KEYWORD_PLAY, TokenType.KEYWORD_LOOP, TokenType.KEYWORD_WAIT):
                raise self.error(
                    f"Cannot combine movement with '{token.value}' using 'and'. "
                    f"Use separate lines instead."
                )
            # Keyword reuse after 'and'
            action.keyword = previous_keyword
        else:
            raise self.error(f"Expected movement keyword, got {token.value!r}")

        # Parse antenna modifier and clock position if applicable
        if action.keyword == "antenna":
            # Antenna modifier is now REQUIRED (left, right, both)
            if (
                self.current().type == TokenType.DIRECTION
                and self.current().value in ANTENNA_MODIFIERS
            ):
                action.antenna_modifier = self.current().value
                self.advance()
            else:
                raise self.error(
                    f"Antenna command requires a modifier (left/right/both), got '{self.current().value}'"
                )

            # Now parse clock position (number, clock keyword, or directional keyword)
            if self.current().type == TokenType.NUMBER:
                # Clock position as number (0-12)
                clock_pos = float(self.current().value)
                if clock_pos < 0 or clock_pos > 12:
                    raise self.error(
                        f"Antenna clock position must be between 0 and 12, got {clock_pos}"
                    )
                action.strength = clock_pos  # Store as strength for now
                action.direction = "clock"  # Special marker
                self.advance()
            elif self.current().type == TokenType.ANTENNA_CLOCK:
                # Clock keyword (high/low/ext/int)
                action.direction = self.current().value  # Store keyword
                self.advance()
            elif (
                self.current().type == TokenType.DIRECTION
                and self.current().value in ANTENNA_DIRECTION_KEYWORDS
            ):
                # Directional keyword (up/down/left/right)
                action.direction = self.current().value  # Store keyword
                self.advance()
            else:
                raise self.error(
                    f"Antenna command requires a position (0-12, high/low/ext/int, or up/down/left/right), got '{self.current().value}'"
                )

        # Parse direction for non-antenna keywords
        elif self.current().type == TokenType.DIRECTION:
            direction = self.current().value

            # Validate direction for keyword
            if action.keyword == "turn" and direction not in TURN_DIRECTIONS:
                raise self.error(
                    f"Invalid direction '{direction}' for 'turn' (use left/right/center)"
                )
            elif action.keyword == "look" and direction not in LOOK_DIRECTIONS:
                raise self.error(
                    f"Invalid direction '{direction}' for 'look' (use left/right/up/down/center)"
                )
            elif action.keyword == "head" and direction not in HEAD_DIRECTIONS:
                raise self.error(
                    f"Invalid direction '{direction}' for 'head' (use forward/back/left/right/up/down)"
                )
            elif action.keyword == "tilt" and direction not in TILT_DIRECTIONS:
                raise self.error(
                    f"Invalid direction '{direction}' for 'tilt' (use left/right/center)"
                )

            action.direction = direction
            self.advance()

        # Parse optional parameters (strength and/or duration, in any order)
        while self.current().type in (
            TokenType.NUMBER,
            TokenType.DURATION,
            TokenType.DURATION_KEYWORD,
            TokenType.QUALITATIVE,
        ):
            if self.current().type == TokenType.NUMBER:
                action.strength = float(self.current().value)
                self.advance()
            elif self.current().type == TokenType.DURATION:
                duration_str = self.current().value.rstrip("s")
                action.duration = float(duration_str)
                self.advance()
            elif self.current().type == TokenType.DURATION_KEYWORD:
                action.duration_keyword = self.current().value
                self.advance()
            elif self.current().type == TokenType.QUALITATIVE:
                action.strength_qualitative = self.current().value
                self.advance()

        return action


def parse(tokens: List[Token]) -> Program:
    """Parse tokens into AST, convenience function."""
    parser = Parser(tokens)
    return parser.parse()
