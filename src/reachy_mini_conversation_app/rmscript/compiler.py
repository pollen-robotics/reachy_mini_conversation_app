"""Main ReachyMiniScript compiler - orchestrates lexing, parsing, analysis, and code generation."""

import logging

from reachy_mini_conversation_app.rmscript.lexer import Lexer
from reachy_mini_conversation_app.rmscript.errors import CompiledTool, CompilationError
from reachy_mini_conversation_app.rmscript.parser import Parser, ParseError
from reachy_mini_conversation_app.rmscript.codegen import CodeGenerator
from reachy_mini_conversation_app.rmscript.semantic import SemanticAnalyzer
from reachy_mini_conversation_app.rmscript.optimizer import Optimizer


class ReachyMiniScriptCompiler:
    """Compiler for ReachyMiniScript language."""

    def __init__(self, log_level: str = "WARNING"):
        """Initialize compiler."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

    def compile(self, source: str) -> CompiledTool:
        """Compile ReachyMiniScript source code into an executable tool."""
        result = CompiledTool(
            name="",
            description="",
            source_code=source,
        )

        try:
            # Stage 1: Lexical Analysis
            self.logger.info("Stage 1: Lexical analysis...")
            lexer = Lexer(source)
            tokens = lexer.tokenize()
            self.logger.debug(f"Generated {len(tokens)} tokens")

            # Stage 2: Parsing
            self.logger.info("Stage 2: Parsing...")
            parser = Parser(tokens)
            ast = parser.parse()

            # Use tool_name from AST, or default to "rmscript_tool" if empty
            result.name = ast.tool_name if ast.tool_name else "rmscript_tool"
            result.description = ast.description

            self.logger.debug(f"Parsed tool '{result.name}'")

            # Stage 3: Semantic Analysis
            self.logger.info("Stage 3: Semantic analysis...")
            analyzer = SemanticAnalyzer()
            ir = analyzer.analyze(ast)

            result.ir = ir
            result.errors.extend(analyzer.errors)
            result.warnings.extend(analyzer.warnings)

            if result.errors:
                self.logger.error(
                    f"Compilation failed with {len(result.errors)} error(s)"
                )
                result.success = False
                return result

            self.logger.debug(f"Generated {len(ir)} IR actions")

            # Stage 4: Optimization
            self.logger.info("Stage 4: Optimization...")
            optimizer = Optimizer()
            optimized_ir = optimizer.optimize(ir)

            result.ir = optimized_ir
            self.logger.debug(
                f"Optimized to {len(optimized_ir)} actions (from {len(ir)})"
            )

            # Stage 5: Code Generation
            self.logger.info("Stage 5: Code generation...")
            codegen = CodeGenerator()
            executable = codegen.generate(result.name, result.description, optimized_ir)

            result.executable = executable
            result.success = True

            self.logger.info(f"✓ Successfully compiled tool '{result.name}'")

            # Print warnings if any
            if result.warnings:
                self.logger.warning(
                    f"Compilation succeeded with {len(result.warnings)} warning(s)"
                )

        except ParseError as e:
            # Parser error
            result.success = False
            result.errors.append(
                CompilationError(
                    line=e.token.line, column=e.token.column, message=e.message
                )
            )
            self.logger.error(f"Parse error: {e}")

        except SyntaxError as e:
            # Lexer error
            result.success = False
            result.errors.append(CompilationError(line=1, column=1, message=str(e)))
            self.logger.error(f"Syntax error: {e}")

        except Exception as e:
            # Unexpected error
            result.success = False
            result.errors.append(
                CompilationError(line=0, column=0, message=f"Internal error: {e}")
            )
            self.logger.exception("Unexpected error during compilation")

        return result

    def compile_file(self, filepath: str) -> CompiledTool:
        """Compile a rmscript file.

        The tool name is derived from the filename (without .rmscript extension),
        with spaces replaced by underscores.
        """
        from pathlib import Path

        try:
            # Extract tool name from filename
            filename = Path(filepath).stem  # Remove .rmscript extension
            tool_name = filename.replace(" ", "_")  # Replace spaces with underscores

            # Read and compile source
            with open(filepath, "r") as f:
                source = f.read()

            result = self.compile(source)

            # Override tool name with filename (from compile(), name is "rmscript_tool")
            if result.name == "rmscript_tool":
                result.name = tool_name

            # Store absolute path to source file for sound file discovery
            result.source_file_path = str(Path(filepath).resolve())

            return result
        except FileNotFoundError:
            result = CompiledTool(name="", description="", source_code="")
            result.success = False
            result.errors.append(
                CompilationError(
                    line=0, column=0, message=f"File not found: {filepath}"
                )
            )
            return result
        except Exception as e:
            result = CompiledTool(name="", description="", source_code="")
            result.success = False
            result.errors.append(
                CompilationError(
                    line=0, column=0, message=f"Error reading file: {e}"
                )
            )
            return result


def compile_rmscript(source: str, verbose: bool = False) -> CompiledTool:
    """Compile ReachyMiniScript source, convenience function."""
    log_level = "INFO" if verbose else "WARNING"
    compiler = ReachyMiniScriptCompiler(log_level=log_level)
    return compiler.compile(source)


def verify_rmscript(source: str) -> tuple[bool, list[str]]:
    """Verify ReachyMiniScript source without generating executable code.

    Args:
        source: ReachyMiniScript source code to verify

    Returns:
        Tuple of (is_valid, error_messages)
        - is_valid: True if compilation succeeds, False otherwise
        - error_messages: List of error and warning messages (empty if no errors/warnings)

    Example:
        >>> is_valid, errors = verify_rmscript("look left\\nwait 1s")
        >>> if not is_valid:
        ...     for error in errors:
        ...         print(error)

    """
    compiler = ReachyMiniScriptCompiler(log_level="ERROR")
    result = compiler.compile(source)

    error_messages = []

    # Collect errors
    for error in result.errors:
        error_messages.append(str(error))

    # Collect warnings
    for warning in result.warnings:
        error_messages.append(str(warning))

    return result.success, error_messages
