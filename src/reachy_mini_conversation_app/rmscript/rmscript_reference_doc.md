# rmscript - Kid-Friendly Robot Programming

rmscript is a natural language-inspired programming language designed to make robot programming accessible and fun for children. It compiles to Python code that controls the Reachy Mini robot.

## Example Usage

Create a file `hello.rmscript`:

```rmscript
DESCRIPTION Wave hello to someone
antenna up
wait 1s
antenna down
look left
look right
look center
```

Test the script using the provided runner (after starting the reachy-mini-daemon):

```
python src/reachy_mini_conversation_app/rmscript/run_rmscript.py path/to/hello.rmscript
```


## Language Syntax

### File Structure

Every rmscript file has a simple structure:

```rmscript
DESCRIPTION Wave hello to someone
# Your commands here
look left
wait 1s
```

**Key features:**
- **Tool name**: Automatically derived from the filename (e.g., `wave_hello.rmscript` → tool name is `wave_hello`)
- **DESCRIPTION** (optional): One-line description used for LLM tool registration

### Basic Commands

```rmscript
# Comments start with #

# Movement commands
look left
turn right
antenna up
head forward 10

# Wait command
wait 2s
wait 0.5s

# Camera command
picture

# Sound playback
play mysound
play othersound pause
```

### Case Insensitivity

rmscript is case-insensitive for keywords:

```rmscript
LOOK left    # Same as "look left"
Look Left    # Same as "look left"
WAIT 1s      # Same as "wait 1s"
```

## Movement Commands

### Look (Head Orientation)

Control the robot's head orientation (pitch and yaw):

```rmscript
look left          # Turn head left (30° default)
look right 45      # Turn head right 45°
look up           # Tilt head up (30° default)
look down 20      # Tilt head down 20°
look center       # Return to center position

# Synonyms
look straight     # Same as center
look neutral      # Same as center
```

**Physical Limits:**
- Pitch (up/down): ±40°
- Yaw (left/right): ±180° absolute, ±65° relative to body

### Turn (Body Rotation)

Rotate the robot's body (the head rotates together with the body):

```rmscript
turn left         # Rotate body and head left (30° default)
turn right 90     # Rotate body and head right 90°
turn center       # Face forward
```

**Note:** The `turn` command rotates both the body yaw and the head yaw together, since the body carries the head.

**Physical Limits:**
- Body yaw: ±160° (safe range)

### Antenna

Control the antenna positions using multiple syntaxes:

**Clock Position (Numeric 0-12):**
```rmscript
antenna both 0       # 0 o'clock = 0° (straight up)
antenna both 3       # 3 o'clock = 90° (external/right)
antenna both 6       # 6 o'clock = 180° (straight down)
antenna both 9       # 9 o'clock = -90° (internal/left)
antenna left 4.5     # Left antenna to 4.5 o'clock (135°)
```

**Clock Keywords:**
```rmscript
antenna both high    # 0° (high position)
antenna both ext     # 90° (external)
antenna both low     # 180° (low position)
antenna both int     # -90° (internal)
```

**Directional Keywords (Natural Language):**
```rmscript
antenna both up      # 0° (up)
antenna both right   # 90° (right/external)
antenna both down    # 180° (down)
antenna both left    # -90° (left/internal)

# Individual antenna control
antenna left up      # Left antenna pointing up
antenna right down   # Right antenna pointing down

# Even in potentially confusing cases, it works naturally:
antenna left left    # Left antenna pointing left (-90°)
antenna right right  # Right antenna pointing right (90°)
```

**Physical Limits:**
- Antenna angle: ±180° (safe recommended: ±120°)

### Head Translation

Move the head forward/back/left/right/up/down in space:

```rmscript
head forward 10    # Move head forward 10mm
head back 5        # Move head back 5mm
head left 8        # Move head left 8mm
head right 8       # Move head right 8mm
head up 5          # Move head up 5mm
head down 3        # Move head down 3mm
```

**Physical Limits:**
- X (forward/back): ±50mm typical
- Y (left/right): ±50mm typical
- Z (up/down): +20mm / -30mm typical

### Tilt (Head Roll)

Tilt the head side-to-side:

```rmscript
tilt left 15       # Tilt head left
tilt right 15      # Tilt head right
tilt center        # Return to level
```

**Physical Limits:**
- Roll: ±40°

### Wait

Pause between movements:

```rmscript
wait 1s           # Wait 1 second
wait 0.5s         # Wait 0.5 seconds
wait 2.5s         # Wait 2.5 seconds
```

**Important:** The `s` suffix is **required** for consistency. `wait 1` will produce a compilation error.

### Picture

Take a picture with the robot's camera and add it to the conversation:

```rmscript
picture           # Take a picture
```

The picture command captures a frame from the camera and returns it as a base64-encoded image.

**How it works:**

1. **During execution**: The `picture` command is queued as a movement (instant duration: 0.01s)
2. **Capture timing**: Picture is captured at the right moment in the movement sequence
3. **File storage**: Pictures are automatically saved to `/tmp/reachy_picture_YYYYMMDD_HHMMSS_microseconds.jpg`
4. **Color handling**: Images are captured in RGB format with proper color conversion
5. **LLM integration**: When called by the LLM, pictures are automatically added to the conversation

**Single picture (LLM-compatible):**
```rmscript
DESCRIPTION Check behind by taking a picture
turn left maximum
wait 0.5s
picture
wait 0.5s
turn center
```

This script:
- Turns the robot to look behind (120° left)
- Waits for movement to stabilize
- Captures a picture at the right moment
- Returns to center position
- The LLM receives the picture and can describe what it sees

**Important timing notes:**
- Add `wait` commands before `picture` to let movements complete
- The tool automatically waits for all movements + picture capture before returning
- Pictures are captured when the movement queue reaches them (not immediately)

**Multiple pictures:**
```rmscript
DESCRIPTION Look around and take pictures
look left
wait 0.5s
picture
look right
wait 0.5s
picture
look center
```

Each `picture` command captures a separate image. For single-picture scripts, the image is returned in Camera-compatible format (`b64_im`) for LLM integration. For multi-picture scripts, images are returned as an array.

**Use with `run_rmscript.py`:**

When testing scripts with `run_rmscript.py`, pictures are:
- Saved to `/tmp` with timestamped filenames
- Logged with full paths for easy access
- Displayed in the execution summary

**Use with LLM conversation:**

When called as a tool by the LLM:
- Single picture: Automatically fed to the LLM conversation (like Camera tool)
- Multiple pictures: Available as base64 data (future enhancement for multi-image display)
- Gradio UI: Picture thumbnails displayed in the conversation interface

**Example conversation:**
```
User: "Can you check what's behind you?"
→ LLM calls: check_behind()
→ Robot: turns, captures picture, returns to center
→ LLM sees the picture and responds: "I can see a desk with a laptop and some books!"
```

### Play Sound

Play sound files (.wav, .mp3, .ogg, .flac) during script execution:

```rmscript
play soundname         # Play sound in background (async, continues immediately)
play soundname pause   # Wait for sound to finish before continuing (blocking)
```

The script automatically searches for sound files in this order:
1. **Directory containing the .rmscript file** (highest priority - co-located sounds)
2. Current working directory
3. `sounds/` subdirectory

**Asynchronous playback (default):**
```rmscript
play intro            # Starts playing, script continues immediately
look left
wait 1s
```
The sound plays in the background while movements execute.

**Blocking playback (wait for sound to finish):**
```rmscript
play mysound pause    # Waits for sound to finish
# OR use synonyms:
play mysound fully
play mysound wait
play mysound block
play mysound complete
```

The script pauses until the sound finishes playing, then continues.

**Example use case:**
```rmscript
DESCRIPTION Greet with sound and movement
play hello pause      # Play greeting sound fully
wait 0.5s
antenna both up       # Wave antennas
wait 1s
antenna both down
```

**Sound files:**
- File extensions: `.wav`, `.mp3`, `.ogg`, `.flac`
- Example: `play hello` looks for `hello.wav`, `hello.mp3`, etc.
- **Best practice**: Place sound files in the same directory as your .rmscript file
- Alternative locations: working directory or `sounds/` subdirectory

## Advanced Features

### Keyword Reuse with `and`

Chain multiple directions with the same action keyword:

```rmscript
# Reuse "look" keyword
look left and up 25
# → Equivalent to: look left + look up 25
# → Merged into single command

# Different keywords - no reuse
turn left and look right
# → turn left + look right
# → Two separate commands merged
```

This creates more natural, flowing descriptions while optimizing execution.

### Qualitative Strength

Use descriptive words instead of numbers - works for both angles and distances:

```rmscript
# 5 levels of strength - values are context-aware!

# Examples:
turn left tiny          # 10° (body yaw can handle larger movements)
look up tiny            # 5° (head pitch limited by cone constraint)
head forward tiny       # 2mm

turn left maximum       # 120° (body yaw safe maximum)
look up maximum         # 38° (respects ±40° pitch limit)
look left maximum       # 60° (respects ±65° body-head differential)
head forward maximum    # 28mm (under 30mm limit)
```

**Available Qualitative Keywords:**
- **VERY_SMALL**: minuscule, mini, verysmall, tiny
- **SMALL**: little, slightly, small, alittle
- **MEDIUM**: medium, normal, regular, standard, normally
- **LARGE**: lot, big, large, very, alot, huge, strong, strongly
- **VERY_LARGE**: verybig, enormous, verylarge, maximum

**Context-Aware Values:**

The same qualitative keyword maps to different values depending on the movement type, respecting physical limits:

| Keyword | VERY_SMALL | SMALL | MEDIUM | LARGE | VERY_LARGE |
|---------|------------|-------|--------|-------|------------|
| **Body Yaw (turn)** | 10° | 30° | 60° | 90° | 120° |
| **Head Pitch/Roll (look up/down, tilt)** | 5° | 10° | 20° | 30° | 38° |
| **Head Yaw (look left/right)** | 5° | 15° | 30° | 45° | 60° |
| **Head Translation (head forward/back/etc)** | 2mm | 5mm | 10mm | 20mm | 28mm |
| **Antennas** | 10° | 30° | 60° | 90° | 110° |

**Note:** These values are carefully chosen to stay within the robot's physical limits while maximizing safe range of motion.

### Duration Keywords

Use descriptive speed words:

```rmscript
look left superfast    # 0.2 seconds
look right fast        # 0.5 seconds
look up slow           # 2.0 seconds
look down superslow    # 3.0 seconds
```

Combine with 'and':

```rmscript
turn left and look right fast
# Both movements complete in 0.5 seconds
```

### Repeat Blocks

Repeat a sequence of commands:

```rmscript
repeat 3
    look left
    wait 0.5s
    look right
    wait 0.5s

# Nested repeat blocks work too
repeat 2
    antenna up
    repeat 3
        look left
        look right
    antenna down
```

**Indentation:** Use spaces or tabs consistently (tabs = 4 spaces).

### Combining Movements

Combine multiple movements into a single smooth motion:

```rmscript
# All happen simultaneously
antenna both up and look up 25 and turn left 30
```

This merges into a single `goto_target()` call with all parameters.

## Compiler Architecture

### Overview

The compiler uses a 5-stage pipeline:

```
Source Code → Lexer → Parser → Semantic Analyzer → Optimizer → Code Generator
```

### Stage 1: Lexer (Tokenization)

Converts source text into tokens with indentation tracking:

```python
from reachy_mini.rmscript.lexer import Lexer

source = "look left\nwait 1s"
lexer = Lexer(source)
tokens = lexer.tokenize()
# → [Token(KEYWORD_LOOK, 'look', L1:C1), Token(DIRECTION, 'left', L1:C6), ...]
```

Features:
- Python-like INDENT/DEDENT tokens
- Case-insensitive keyword matching
- Comment stripping
- Line/column tracking for error messages

### Stage 2: Parser (AST Construction)

Builds an Abstract Syntax Tree from tokens:

```python
from reachy_mini.rmscript.parser import Parser

ast = Parser(tokens).parse()
# → Program(tool_name='...', description='...', statements=[...])
```

Features:
- Keyword reuse detection for `and` chains
- Direction validation per keyword
- Repeat block nesting
- Syntax error reporting

### Stage 3: Semantic Analyzer

Applies defaults, validates ranges, and generates IR:

```python
from reachy_mini.rmscript.semantic import SemanticAnalyzer

ir, errors, warnings = SemanticAnalyzer().analyze(ast)
# → List[Action | WaitAction], List[CompilationError], List[CompilationError]
```

Features:
- Default value resolution (30° for angles, 1s for duration)
- Qualitative → quantitative conversion
- Duration keyword mapping
- Physical limit validation
- Action merging (multiple SingleActions → one Action)

### Stage 4: Optimizer

Optimizes the IR:

```python
from reachy_mini.rmscript.optimizer import Optimizer

optimized_ir = Optimizer().optimize(ir)
```

Features:
- Consecutive wait merging
- No-op action removal

### Stage 5: Code Generator

Creates executable Python functions:

```python
from reachy_mini.rmscript.codegen import CodeGenerator

executable = CodeGenerator().generate(tool_name, description, ir)
# → Callable that executes the behavior
```

Features:
- Direct execution: `executable(mini)`
- Readable code generation: `to_python_code()`
- Proper imports and function signatures

## API Reference

### rmscriptCompiler

Main compiler class.

```python
from reachy_mini.rmscript import rmscriptCompiler

compiler = rmscriptCompiler(log_level="INFO")
tool = compiler.compile(source_code)
```

**Parameters:**
- `log_level` (str): Logging level - "DEBUG", "INFO", "WARNING", "ERROR"

**Returns:**
- `CompiledTool`: Compilation result object

### CompiledTool

Result of compilation.

```python
class CompiledTool:
    name: str                        # Tool name
    description: str                      # Tool description
    executable: Callable             # Function to execute on robot
    success: bool                    # Compilation succeeded?
    errors: List[CompilationError]   # Compilation errors
    warnings: List[CompilationError] # Compilation warnings
    source_code: str                 # Original source
    ir: List[Action | WaitAction]    # Intermediate representation

    def execute(self, mini: ReachyMini) -> None:
        """Execute the compiled behavior on the robot."""

    def to_python_code(self) -> str:
        """Generate readable Python code."""
```

**Usage:**

```python
if tool.success:
    # Execute on robot
    with ReachyMini() as mini:
        tool.execute(mini)

    # Or get Python code
    print(tool.to_python_code())
else:
    # Handle errors
    for error in tool.errors:
        print(f"{error.severity} Line {error.line}: {error.message}")
```

### CompilationError

Error or warning from compilation.

```python
@dataclass
class CompilationError:
    line: int          # Line number (1-indexed)
    column: int        # Column number (1-indexed)
    message: str       # Error/warning message
    severity: str      # "ERROR" or "WARNING"
```

### Action (IR)

Intermediate representation of a movement.

```python
@dataclass
class Action:
    head_pose: Optional[np.ndarray]  # 4x4 transformation matrix
    antennas: Optional[np.ndarray]   # [left, right] angles in radians
    body_yaw: Optional[float]        # Body rotation in radians
    duration: float                  # Movement duration in seconds
    interpolation: str               # "minjerk", "linear", "ease", "cartoon"
```

### WaitAction (IR)

Intermediate representation of a wait/pause.

```python
@dataclass
class WaitAction:
    duration: float  # Wait duration in seconds
```

## Examples

### Example 1: Simple Greeting

```rmscript

DESCRIPTION Greet someone warmly
# Wave with antennas
antenna up
wait 0.5s
antenna down
wait 0.5s

# Nod
look down 20
wait 0.3s
look up 20
wait 0.3s
look center
```

### Example 2: Search Pattern

```rmscript

DESCRIPTION Look around to search for something
# Scan left to right
look left 45
wait 1s
look center
wait 0.5s
look right 45
wait 1s
look center

# Check up and down
look up 30
wait 0.5s
look down 30
wait 0.5s
look center
```

### Example 3: Dance Choreography

```rmscript

DESCRIPTION Perform a fun dance routine
# Opening pose
antenna both up and look up 25
wait 1s

# Main sequence
repeat 2
    # Move left
    turn left 30 and look right 20
    wait 0.5s

    # Move right
    turn right 30 and look left 20
    wait 0.5s

# Ending pose
turn center and look center and antenna down
```

### Example 4: Using Qualitative Strength

```rmscript

DESCRIPTION Wave shyly at someone
# Small movements
look down slightly
antenna left up a little
wait 1s
antenna left down
look center
```

### Example 5: Complex Combination

```rmscript

DESCRIPTION Show excitement when greeting
# Quick movements
repeat 3
    antenna both up superfast
    antenna both down superfast

# Energetic looking around
look left and up fast
wait 0.2s
look right and up fast
wait 0.2s
look center slow
```

## Error Handling

### Compilation Errors

Errors prevent code generation:

```rmscript
# Invalid keyword
jump up
```
**Output:** `❌ Line 1: Unexpected token: 'jump'`

```rmscript
# Invalid direction for keyword
turn up
```
**Output:** `❌ Line 1: Invalid direction 'up' for keyword 'turn'`

```rmscript
# Missing indentation
repeat 3
look left
```
**Output:** `❌ Line 2: Expected indented block after 'repeat'`

### Compilation Warnings

Warnings allow compilation but alert to potential issues:

```rmscript
# Exceeds safe range
turn left 200
```
**Output:** `⚠️  Line 1: Body yaw 200.0° exceeds safe range (±160.0°), will be clamped`

```rmscript
# Exceeds physical limit
look up 50
```
**Output:** `⚠️  Line 1: Head pitch 50.0° exceeds limit (±40.0°), will be clamped`

### Handling Errors in Code

```python
tool = compiler.compile(source)

if not tool.success:
    print("Compilation failed!")
    for error in tool.errors:
        print(f"  ❌ Line {error.line}: {error.message}")
    exit(1)

if tool.warnings:
    print("Compilation succeeded with warnings:")
    for warning in tool.warnings:
        print(f"  ⚠️  Line {warning.line}: {warning.message}")

# Safe to execute
tool.execute(mini)
```

## Tips and Best Practices

### 1. Use Comments

```rmscript
# Good: explain what you're doing
# Opening sequence: get attention
antenna up
wait 1s

# Main behavior: scan the room
look left 45
wait 1s
look right 45
```

### 2. Use Descriptive Tool Names

```python
# Good




# Avoid



```

### 3. Break Complex Behaviors into Steps

```rmscript
# Good: readable steps
antenna up
wait 1s
look left
wait 0.5s
look right
wait 0.5s
look center

# Avoid: too compressed
antenna up and look left and wait 1s and look right
```

### 4. Use Repeat for Patterns

```rmscript
# Good: use repeat
repeat 3
    antenna up
    antenna down

# Avoid: repetitive
antenna up
antenna down
antenna up
antenna down
antenna up
antenna down
```

### 5. Test with Small Values First

```rmscript
# Start conservative
look left 10
turn right 15

# Then increase if needed
look left 45
turn right 90
```

## Physical Safety Limits

rmscript validates all movements against these limits:

| Movement | Limit | Warning Threshold |
|----------|-------|------------------|
| Body yaw (turn) | ±180° | ±160° |
| Head yaw (look left/right) | ±180° absolute | ±65° relative to body |
| Head pitch (look up/down) | ±40° | ±40° |
| Head roll (tilt) | ±40° | ±40° |
| Antenna | ±90° | ±65° |
| Head X (forward/back) | ±50mm typical | - |
| Head Y (left/right) | ±50mm typical | - |
| Head Z (up/down) | +20mm / -30mm typical | - |

Exceeding these limits generates **warnings** but still compiles. The robot's hardware will clamp values to safe ranges.

## Default Values

When values aren't specified:

| Parameter | Default |
|-----------|---------|
| Angle (look, turn, tilt) | 30° |
| Distance (head translation) | 10mm |
| Antenna angle | 45° |
| Duration | 1.0s |

**Qualitative Strength Values:**

| Level | Keywords | Angle | Distance |
|-------|----------|-------|----------|
| VERY_SMALL | tiny, minuscule, mini, verysmall | 5° | 2mm |
| SMALL | little, slightly, small, alittle | 15° | 5mm |
| MEDIUM | medium, normal, regular, standard, normally | 30° | 10mm |
| LARGE | strong, lot, big, large, very, alot, huge, strongly | 45° | 20mm |
| VERY_LARGE | enormous, verybig, verylarge, maximum | 60° | 30mm |

## Troubleshooting

### Issue: "Inconsistent indentation"

**Cause:** Mixing tabs and spaces, or inconsistent indentation levels.

**Solution:** Use only spaces or only tabs (not mixed). Ensure nested blocks are indented consistently.

```rmscript
# Bad: mixed tabs and spaces
repeat 3
    look left      # 4 spaces
→   look right     # 1 tab (shows as →)

# Good: consistent spaces
repeat 3
    look left      # 4 spaces
    look right     # 4 spaces
```

### Issue: "Expected indented block after 'repeat'"

**Cause:** No indentation after `repeat`.

**Solution:** Indent the repeat block body.

```rmscript
# Bad
repeat 3
look left

# Good
repeat 3
    look left
```

### Issue: Generated code has unexpected angles

**Cause:** Sign convention confusion (left/right).

**Solution:** Remember: `left` = positive yaw, `right` = negative yaw in the generated code. `up` = negative pitch, `down` = positive pitch. This matches the robot's actual coordinate system.


## Contributing

To extend rmscript:

1. **Add new keywords**: Update `lexer.py`, `parser.py`, and `semantic.py`
2. **Add new features**: Modify the appropriate compiler stage
3. **Add tests**: Create integration tests in `tests/test_rmscript/`
4. **Update docs**: Keep this README and examples current

## License

Same as Reachy Mini SDK.
