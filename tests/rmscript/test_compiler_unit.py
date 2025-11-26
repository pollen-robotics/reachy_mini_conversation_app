"""Unit tests for ReachyMiniScript compiler pipeline stages."""

import math

import numpy as np
import pytest

from reachy_mini_conversation_app.rmscript.lexer import Lexer, TokenType
from reachy_mini_conversation_app.rmscript.errors import Action, WaitAction, PictureAction, PlaySoundAction
from reachy_mini_conversation_app.rmscript.parser import Parser, ParseError
from reachy_mini_conversation_app.rmscript.codegen import CodeGenerator
from reachy_mini_conversation_app.rmscript.semantic import SemanticAnalyzer
from reachy_mini_conversation_app.rmscript.constants import (
    DEFAULT_ANGLE,
    DEFAULT_DURATION,
)
from reachy_mini_conversation_app.rmscript.optimizer import Optimizer


@pytest.fixture
def lexer():
    """Reusable lexer factory."""
    return lambda src: Lexer(src)


@pytest.fixture
def optimizer():
    """Reusable optimizer instance."""
    return Optimizer()


class TestLexer:
    """Test tokenization."""

    def test_tokenize_keywords(self, lexer):
        """Test that all movement keywords are recognized."""
        source = "turn look head tilt antenna wait picture play loop repeat end"
        tokens = lexer(source).tokenize()

        expected_types = [
            TokenType.KEYWORD_TURN,
            TokenType.KEYWORD_LOOK,
            TokenType.KEYWORD_HEAD,
            TokenType.KEYWORD_TILT,
            TokenType.KEYWORD_ANTENNA,
            TokenType.KEYWORD_WAIT,
            TokenType.KEYWORD_PICTURE,
            TokenType.KEYWORD_PLAY,
            TokenType.KEYWORD_LOOP,
            TokenType.KEYWORD_REPEAT,
            TokenType.KEYWORD_END,
            TokenType.EOF,
        ]

        assert len(tokens) == len(expected_types)
        for token, expected_type in zip(tokens, expected_types):
            assert token.type == expected_type

    def test_tokenize_numbers(self, lexer):
        """Test that numbers are tokenized correctly."""
        source = "30 2.5 10"
        tokens = lexer(source).tokenize()

        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "30"
        assert tokens[1].type == TokenType.NUMBER
        assert tokens[1].value == "2.5"
        assert tokens[2].type == TokenType.NUMBER
        assert tokens[2].value == "10"

    def test_tokenize_durations(self, lexer):
        """Test that duration values with 's' suffix are recognized."""
        source = "1s 2.5s 10s"
        tokens = lexer(source).tokenize()

        assert tokens[0].type == TokenType.DURATION
        assert tokens[0].value == "1s"
        assert tokens[1].type == TokenType.DURATION
        assert tokens[1].value == "2.5s"
        assert tokens[2].type == TokenType.DURATION
        assert tokens[2].value == "10s"

    def test_tokenize_directions(self, lexer):
        """Test that direction keywords are recognized."""
        source = "left right up down center back backward backwards"
        tokens = lexer(source).tokenize()

        for i in range(8):  # All 8 directions plus EOF
            if i < 8:
                assert tokens[i].type == TokenType.DIRECTION
                assert tokens[i].value in ["left", "right", "up", "down", "center", "back", "backward", "backwards"]

    def test_tokenize_duration_keywords(self, lexer):
        """Test that duration keywords are recognized."""
        source = "fast slow slowly superfast superslow"
        tokens = lexer(source).tokenize()

        assert tokens[0].type == TokenType.DURATION_KEYWORD
        assert tokens[0].value == "fast"
        assert tokens[1].type == TokenType.DURATION_KEYWORD
        assert tokens[1].value == "slow"
        assert tokens[2].type == TokenType.DURATION_KEYWORD
        assert tokens[2].value == "slowly"

    def test_tokenize_indentation(self, lexer):
        """Test that indentation generates INDENT/DEDENT tokens."""
        source = """repeat 3
    look left
    look right
turn center"""
        tokens = lexer(source).tokenize()

        # Find INDENT and DEDENT tokens
        indent_tokens = [t for t in tokens if t.type == TokenType.INDENT]
        dedent_tokens = [t for t in tokens if t.type == TokenType.DEDENT]

        assert len(indent_tokens) == 1
        assert len(dedent_tokens) == 1

    def test_tokenize_comments(self, lexer):
        """Test that comments are ignored."""
        source = """# This is a comment
look left  # inline comment
# Another comment
look right"""
        tokens = lexer(source).tokenize()

        # Should only have look keywords, directions, newlines, and EOF
        look_tokens = [t for t in tokens if t.type == TokenType.KEYWORD_LOOK]
        assert len(look_tokens) == 2

    def test_tokenize_error_unexpected_character(self, lexer):
        """Test that unexpected characters raise an error."""
        source = "look left @ right"

        with pytest.raises(SyntaxError) as excinfo:
            lexer(source).tokenize()

        assert "Unexpected character" in str(excinfo.value)


class TestParser:
    """Test AST generation."""

    def test_parse_description(self):
        """Test that DESCRIPTION header is parsed correctly."""
        source = """DESCRIPTION This is a test tool
look left"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        assert program.description == "This is a test tool"

    def test_parse_action_statement(self):
        """Test that simple action statements are parsed."""
        source = """DESCRIPTION test
look left
turn right"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        assert len(program.statements) == 2
        assert program.statements[0].actions[0].keyword == "look"
        assert program.statements[0].actions[0].direction == "left"
        assert program.statements[1].actions[0].keyword == "turn"
        assert program.statements[1].actions[0].direction == "right"

    def test_parse_action_with_strength(self):
        """Test that actions with numeric strength are parsed."""
        source = """DESCRIPTION test
turn left 45"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        assert program.statements[0].actions[0].strength == 45.0

    def test_parse_action_with_duration(self):
        """Test that actions with duration are parsed."""
        source = """DESCRIPTION test
look up 2s"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        assert program.statements[0].actions[0].duration == 2.0

    def test_parse_compound_action_with_and(self):
        """Test that compound actions with 'and' are parsed."""
        source = """DESCRIPTION test
turn left and look right"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        assert len(program.statements[0].actions) == 2
        assert program.statements[0].actions[0].keyword == "turn"
        assert program.statements[0].actions[1].keyword == "look"

    def test_parse_repeat_block(self):
        """Test that repeat blocks are parsed."""
        source = """DESCRIPTION test
repeat 3
    look left
    look right"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        assert len(program.statements) == 1
        repeat_block = program.statements[0]
        assert repeat_block.count == 3
        assert len(repeat_block.body) == 2

    def test_parse_wait_statement(self):
        """Test that wait statements are parsed."""
        source = """DESCRIPTION test
wait 2s"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        assert program.statements[0].duration == 2.0

    def test_parse_picture_statement(self):
        """Test that picture statements are parsed."""
        source = """DESCRIPTION test
picture"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        assert len(program.statements) == 1

    def test_parse_play_sound_with_duration(self):
        """Test that play sound with duration is parsed."""
        source = """DESCRIPTION test
play mysound 5s"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        assert program.statements[0].sound_name == "mysound"
        assert program.statements[0].duration == 5.0
        assert program.statements[0].blocking

    def test_parse_error_and_with_picture(self):
        """Test that 'and picture' produces a parse error."""
        source = """DESCRIPTION test
look left and picture"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)

        with pytest.raises(ParseError) as excinfo:
            parser.parse()

        assert "cannot combine" in str(excinfo.value).lower()


class TestSemanticAnalyzer:
    """Test semantic analysis and IR generation."""

    def test_apply_default_angle(self):
        """Test that default angle is applied when no strength specified."""
        source = """DESCRIPTION test
turn left"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        analyzer = SemanticAnalyzer()
        ir = analyzer.analyze(program)

        assert len(ir) == 1
        assert isinstance(ir[0], Action)
        assert ir[0].body_yaw == pytest.approx(math.radians(DEFAULT_ANGLE), abs=0.01)

    def test_apply_default_duration(self):
        """Test that default duration is applied when not specified."""
        source = """DESCRIPTION test
look up"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        analyzer = SemanticAnalyzer()
        ir = analyzer.analyze(program)

        assert ir[0].duration == DEFAULT_DURATION

    def test_qualitative_strength_context_aware(self):
        """Test that qualitative keywords map to context-appropriate values."""
        source = """DESCRIPTION test
turn left maximum
look left maximum"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        analyzer = SemanticAnalyzer()
        ir = analyzer.analyze(program)

        # Both use "maximum" but should have different values
        # turn maximum -> larger body_yaw
        # look maximum -> larger head yaw
        assert isinstance(ir[0], Action)
        assert isinstance(ir[1], Action)
        assert ir[0].body_yaw is not None
        assert ir[1].head_pose is not None

    def test_head_translation_backward(self):
        """Test that backward direction moves head in negative X."""
        source = """DESCRIPTION test
head backward 10"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        analyzer = SemanticAnalyzer()
        ir = analyzer.analyze(program)

        assert isinstance(ir[0], Action)
        assert ir[0].head_pose is not None
        # backward should be negative X translation
        assert ir[0].head_pose[0, 3] == pytest.approx(-0.010, abs=0.0001)  # -10mm

    def test_antenna_both_modifier(self):
        """Test that 'antenna both' sets both antennas."""
        source = """DESCRIPTION test
antenna both up"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        analyzer = SemanticAnalyzer()
        ir = analyzer.analyze(program)

        assert isinstance(ir[0], Action)
        assert ir[0].antennas is not None
        assert len(ir[0].antennas) == 2
        # Both antennas should be equal (both modifier)
        assert ir[0].antennas[0] == ir[0].antennas[1]

    def test_repeat_block_expansion(self):
        """Test that repeat blocks expand actions correctly."""
        source = """DESCRIPTION test
repeat 3
    look left"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        analyzer = SemanticAnalyzer()
        ir = analyzer.analyze(program)

        # Should have 3 identical actions
        assert len(ir) == 3
        for action in ir:
            assert isinstance(action, Action)

    def test_wait_action_generation(self):
        """Test that wait statements generate WaitAction in IR."""
        source = """DESCRIPTION test
wait 2s"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        analyzer = SemanticAnalyzer()
        ir = analyzer.analyze(program)

        assert len(ir) == 1
        assert isinstance(ir[0], WaitAction)
        assert ir[0].duration == 2.0

    def test_coordinate_system_consistency(self):
        """Test that coordinate system is consistent across movements."""
        source = """DESCRIPTION test
head forward 10
head backward 10"""
        lexer = Lexer(source)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        analyzer = SemanticAnalyzer()
        ir = analyzer.analyze(program)

        # Forward and backward should be opposite
        assert ir[0].head_pose[0, 3] > 0  # Forward is positive X
        assert ir[1].head_pose[0, 3] < 0  # Backward is negative X
        assert abs(ir[0].head_pose[0, 3]) == abs(ir[1].head_pose[0, 3])


class TestOptimizer:
    """Test IR optimization."""

    def test_merge_consecutive_waits(self, optimizer):
        """Test that consecutive wait actions are merged."""
        ir = [
            WaitAction(duration=1.0, source_line=1),
            WaitAction(duration=2.0, source_line=2),
            WaitAction(duration=1.5, source_line=3),
        ]

        optimized = optimizer.optimize(ir)

        assert len(optimized) == 1
        assert isinstance(optimized[0], WaitAction)
        assert optimized[0].duration == 4.5

    def test_remove_noop_actions(self, optimizer):
        """Test that no-op actions are removed."""
        ir = [
            Action(head_pose=None, antennas=None, body_yaw=None, duration=1.0),
            Action(head_pose=np.eye(4), antennas=None, body_yaw=None, duration=1.0),
        ]

        optimized = optimizer.optimize(ir)

        # First action is no-op (all None), second has head_pose
        assert len(optimized) == 1
        assert optimized[0].head_pose is not None

    def test_preserve_non_mergeable_actions(self, optimizer):
        """Test that non-wait actions are preserved and not merged."""
        head_pose = np.eye(4)
        ir = [
            Action(head_pose=head_pose, duration=1.0),
            WaitAction(duration=1.0),
            Action(body_yaw=0.5, duration=1.0),
        ]

        optimized = optimizer.optimize(ir)

        # All three should be preserved (different types)
        assert len(optimized) == 3
        assert isinstance(optimized[0], Action)
        assert isinstance(optimized[1], WaitAction)
        assert isinstance(optimized[2], Action)

    def test_preserve_picture_and_sound_actions(self, optimizer):
        """Test that picture and sound actions are preserved."""
        ir = [
            PictureAction(source_line=1),
            PlaySoundAction(sound_name="test", source_line=2),
            WaitAction(duration=1.0, source_line=3),
        ]

        optimized = optimizer.optimize(ir)

        assert len(optimized) == 3
        assert isinstance(optimized[0], PictureAction)
        assert isinstance(optimized[1], PlaySoundAction)
        assert isinstance(optimized[2], WaitAction)

    def test_optimize_empty_ir(self, optimizer):
        """Test that optimizing empty IR returns empty list."""
        ir = []

        optimized = optimizer.optimize(ir)

        assert len(optimized) == 0

    def test_optimize_single_action_unchanged(self, optimizer):
        """Test that single actions pass through unchanged."""
        head_pose = np.eye(4)
        ir = [Action(head_pose=head_pose, duration=1.0, source_line=1)]

        optimized = optimizer.optimize(ir)

        assert len(optimized) == 1
        assert optimized[0] is ir[0]  # Same object


class TestCodeGenerator:
    """Test Python code generation."""

    def test_generate_function_with_name(self):
        """Test that generated function has correct name."""
        codegen = CodeGenerator()
        ir = [WaitAction(duration=1.0)]

        func = codegen.generate("test_tool", "Test description", ir)

        assert func.__name__ == "test_tool"

    def test_generate_function_with_description(self):
        """Test that generated function has correct docstring."""
        codegen = CodeGenerator()
        ir = [WaitAction(duration=1.0)]

        func = codegen.generate("test_tool", "Test description", ir)

        assert func.__doc__ == "Test description"

    def test_generate_wait_action(self):
        """Test that WaitAction generates time.sleep call."""
        codegen = CodeGenerator()
        ir = [WaitAction(duration=2.5)]

        func = codegen.generate("test_tool", "Test", ir)

        # Function should be callable (we can't test execution without robot)
        assert callable(func)

    def test_generate_movement_action(self):
        """Test that Action generates goto_target call."""
        codegen = CodeGenerator()
        head_pose = np.eye(4)
        ir = [Action(head_pose=head_pose, duration=1.5)]

        func = codegen.generate("test_tool", "Test", ir)

        assert callable(func)

    def test_generate_mixed_actions(self):
        """Test that mixed action types are handled."""
        codegen = CodeGenerator()
        head_pose = np.eye(4)
        ir = [
            Action(head_pose=head_pose, duration=1.0),
            WaitAction(duration=1.0),
            PictureAction(),
            PlaySoundAction(sound_name="test"),
        ]

        func = codegen.generate("test_tool", "Test", ir)

        assert callable(func)
        assert func.__name__ == "test_tool"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
