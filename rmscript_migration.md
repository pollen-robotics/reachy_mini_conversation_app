# RMScript Migration: Standalone Library with Adapter Pattern

**Status**: In Progress
**Started**: 2025-01-26
**Target Completion**: 4 weeks

---

## Executive Summary

Decoupling rmscript from the conversation app into a standalone Python package with adapter-based execution. This enables:
1. **Conversation app**: LLM → rmscript → queue-based robot execution (existing)
2. **Web demo**: Browser → rmscript → WebSocket pose streaming → robot (new)

**Key Architectural Decision**: Extract to standalone library, use adapter pattern for execution, refactor conversation app immediately.

---

## Architecture Overview

### Three-Layer Design

```
┌─────────────────────────────────────────┐
│  rmscript (standalone package)          │
│  - Compiler (Lexer → Parser → IR)       │
│  - IR definitions (no execution)         │
│  - Adapter protocol                      │
└─────────────────────────────────────────┘
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
┌──────────────────┐   ┌──────────────────┐
│ Conversation App │   │   Web Demo       │
│ QueueAdapter     │   │ WebSocketAdapter │
│ (MovementManager)│   │ (Pose streaming) │
└──────────────────┘   └──────────────────┘
```

### Key Changes

**Compiler Pipeline Change:**
- **Before**: Source → Lexer → Parser → Semantic → Optimizer → **CodeGen** → Executable
- **After**: Source → Lexer → Parser → Semantic → Optimizer → **IR** (adapters consume IR)

**Separation of Concerns:**
- **Core Library**: Language parsing and IR generation (reusable)
- **Adapters**: Execution strategies (conversation app, web demo, etc.)
- **No Breaking Changes**: All existing .rmscript files work identically

---

## Package Structure

### New: rmscript (Standalone Package)

Location: Separate repository/package
```
rmscript/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── rmscript/
│       ├── __init__.py          # Public API
│       ├── compiler.py           # Orchestrator (stops at IR)
│       ├── lexer.py              # Tokenization
│       ├── parser.py             # AST generation
│       ├── semantic.py           # Semantic analysis & IR generation
│       ├── optimizer.py          # IR optimization
│       ├── ir.py                 # IR definitions (NEW, from errors.py)
│       ├── ast_nodes.py          # AST node definitions
│       ├── constants.py          # Language constants
│       └── adapters/
│           ├── __init__.py       # ExecutionAdapter protocol
│           └── base.py           # Base adapter class
├── tests/
│   ├── test_compiler_unit.py    # Moved from conversation app
│   ├── test_language_features.py # Moved from conversation app
│   └── fixtures/
│       └── *.rmscript
└── examples/
    └── basic_usage.py
```

### Updated: reachy_mini_conversation_app

```
src/reachy_mini_conversation_app/
├── rmscript_adapters/           # NEW: Conversation-specific adapters
│   ├── __init__.py
│   ├── queue_adapter.py         # QueueExecutionAdapter
│   └── queue_moves.py           # GotoQueueMove, SoundQueueMove, PictureQueueMove
├── rmscript_tools.py            # NEW: create_tool_from_rmscript
├── tools/
│   └── core_tools.py            # Updated imports
├── pyproject.toml               # Add rmscript dependency
└── profiles/
    └── */
        └── *.rmscript           # Kept (no changes)
```

### Deleted from Conversation App

```
src/reachy_mini_conversation_app/rmscript/  # ENTIRE MODULE DELETED
├── lexer.py                     → Moved to rmscript package
├── parser.py                    → Moved to rmscript package
├── semantic.py                  → Moved to rmscript package
├── optimizer.py                 → Moved to rmscript package
├── codegen.py                   → Removed (no longer needed)
├── compiler.py                  → Moved to rmscript package
├── errors.py                    → Split: IR → rmscript/ir.py, execution removed
├── ast_nodes.py                 → Moved to rmscript package
├── constants.py                 → Moved to rmscript package
├── __init__.py                  → Replaced by rmscript_tools.py
├── sound_player.py              → Moved to queue_moves.py
└── picture_capture.py           → Moved to queue_moves.py
```

---

## Implementation Phases

### Phase 1: Extract Core (Week 1) ✅ IN PROGRESS

**Objective**: Create standalone rmscript package with core compiler.

**Tasks**:
- [x] Create rmscript repository structure
- [ ] Extract core files: lexer.py, parser.py, semantic.py, optimizer.py, ast_nodes.py, constants.py
- [ ] Create ir.py (from errors.py, IR classes only)
- [ ] Create adapters/base.py (ExecutionAdapter protocol)
- [ ] Update compiler.py (remove codegen stage)
- [ ] Create public API in __init__.py
- [ ] Setup pyproject.toml with minimal dependencies
- [ ] Move tests from conversation app
- [ ] Verify all tests pass

**Deliverable**: Standalone rmscript package that compiles .rmscript → IR

---

### Phase 2: Create Adapters (Week 1-2)

**Objective**: Implement QueueExecutionAdapter in conversation app.

**Tasks**:
- [ ] Create `src/reachy_mini_conversation_app/rmscript_adapters/` directory
- [ ] Extract GotoQueueMove from dance_emotion_moves.py → queue_moves.py
- [ ] Extract SoundQueueMove from rmscript/sound_player.py → queue_moves.py
- [ ] Extract PictureQueueMove from rmscript/picture_capture.py → queue_moves.py
- [ ] Implement QueueExecutionAdapter.execute()
  - Convert IRAction → GotoQueueMove
  - Convert IRWaitAction → hold move
  - Convert IRPictureAction → PictureQueueMove
  - Convert IRPlaySoundAction → SoundQueueMove
- [ ] Maintain picture collection behavior
- [ ] Test adapter in isolation with mocks

**Deliverable**: QueueExecutionAdapter that executes IR via MovementManager

---

### Phase 3: Update Conversation App (Week 2)

**Objective**: Switch conversation app to use new rmscript library.

**Tasks**:
- [ ] Add rmscript dependency to pyproject.toml
- [ ] Create src/reachy_mini_conversation_app/rmscript_tools.py
  - Implement create_tool_from_rmscript() using new architecture
  - Use QueueExecutionAdapter for execution
  - Maintain Tool class creation logic
- [ ] Update src/reachy_mini_conversation_app/tools/core_tools.py
  - Change import from rmscript to rmscript_tools
- [ ] Update run_rmscript.py to use new adapter
- [ ] Delete src/reachy_mini_conversation_app/rmscript/ (entire directory)
- [ ] Keep .rmscript files in profiles/

**Deliverable**: Conversation app using standalone rmscript library

---

### Phase 4: Testing & Validation (Week 2-3)

**Objective**: Ensure identical behavior, no regressions.

**Tasks**:
- [ ] Run all 122 rmscript tests
- [ ] Test with actual robot (simulation)
- [ ] Test with actual robot (hardware)
- [ ] Test all profile .rmscript files
- [ ] Verify picture capture works
- [ ] Verify sound playback works
- [ ] Performance benchmarks (compilation speed)
- [ ] Performance benchmarks (execution smoothness)
- [ ] Documentation updates
- [ ] Publish rmscript to PyPI (test)

**Success Criteria**: All tests pass, <5% performance degradation

---

### Phase 5: Web Demo (Week 3-4+)

**Objective**: Create web-based rmscript execution demo.

**Tasks**:
- [ ] Create rmscript-web-demo repository
- [ ] Implement WebSocketExecutionAdapter
  - Interpolate movements
  - Stream pose parameters at configurable FPS
  - Handle picture/sound actions
- [ ] Create FastAPI backend
  - POST /compile endpoint
  - WebSocket /execute endpoint
- [ ] Create frontend
  - Code editor (Monaco or CodeMirror)
  - 3D robot visualization (Three.js)
  - WebSocket client
- [ ] Deploy to HuggingFace Spaces or similar

**Deliverable**: Web demo for rmscript execution

---

## Breaking Changes

### For Conversation App Code

**Import Changes:**
```python
# OLD
from reachy_mini_conversation_app.rmscript import create_tool_from_rmscript

# NEW
from reachy_mini_conversation_app.rmscript_tools import create_tool_from_rmscript
```

**Internal Changes** (not user-facing):
- `CompiledTool.execute_queued()` removed → Use `QueueExecutionAdapter.execute()`
- `CompiledTool.executable` removed → No codegen stage
- Queue moves relocated → `rmscript_adapters.queue_moves`

### For .rmscript Files

✅ **NO BREAKING CHANGES**
- All existing .rmscript files work identically
- No syntax changes
- No behavior changes
- Tool names/descriptions preserved

---

## Dependencies

### rmscript Core

```toml
[project]
name = "rmscript"
version = "0.1.0"
description = "A kid-friendly robot programming language for Reachy Mini"
dependencies = [
    "numpy>=1.24.0",
]

[project.optional-dependencies]
reachy = ["reachy_mini>=1.0.0"]  # For create_head_pose in semantic.py
scipy = ["scipy>=1.10.0"]        # For Rotation utilities
```

### Conversation App Update

```toml
# Add to dependencies
dependencies = [
    "rmscript>=0.1.0",  # NEW
    "reachy_mini>=1.0.0.rc4",
    # ... existing deps
]
```

---

## Code Examples

### 1. Public API Usage

```python
from rmscript import compile_script, compile_file, verify_script

# Compile source code
result = compile_script("""
DESCRIPTION Wave hello
look left
antenna both up
wait 1s
look right
""")

print(f"Success: {result.success}")
print(f"IR actions: {len(result.ir)}")

# Compile file
result = compile_file("hello.rmscript")

# Verify only (no IR generation)
is_valid, errors = verify_script("look left\nwait 1s")
```

### 2. Adapter Protocol

```python
from typing import Protocol, Any, Dict
from dataclasses import dataclass

@dataclass
class ExecutionContext:
    """Base context for all adapters."""
    script_name: str
    script_description: str
    source_file_path: Optional[str] = None

class ExecutionAdapter(Protocol):
    """Protocol for execution adapters."""

    def execute(
        self,
        ir: List[IRAction | IRWaitAction | IRPictureAction | IRPlaySoundAction],
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """Execute IR in adapter-specific way."""
        ...
```

### 3. Queue Adapter Usage

```python
from rmscript import compile_file
from reachy_mini_conversation_app.rmscript_adapters import (
    QueueExecutionAdapter,
    QueueAdapterContext
)

# Compile script
result = compile_file("wave_hello.rmscript")

# Create adapter
adapter = QueueExecutionAdapter()

# Create context
context = QueueAdapterContext(
    script_name=result.name,
    script_description=result.description,
    source_file_path=result.source_file_path,
    reachy_mini=mini,
    movement_manager=movement_manager,
    camera_worker=camera_worker,
)

# Execute
execution_result = adapter.execute(result.ir, context)
print(execution_result)  # {"status": "Queued 5 moves", "total_duration": "5.0s"}
```

---

## Migration Checklist

### Phase 1: Core Extraction
- [ ] rmscript repository created
- [ ] Core files extracted and cleaned
- [ ] ir.py created from errors.py
- [ ] Adapter protocol defined
- [ ] Public API defined
- [ ] Tests moved and passing
- [ ] pyproject.toml configured

### Phase 2: Adapter Implementation
- [ ] rmscript_adapters/ module created
- [ ] Queue moves extracted
- [ ] QueueExecutionAdapter implemented
- [ ] Adapter tests passing

### Phase 3: Conversation App Update
- [ ] rmscript dependency added
- [ ] rmscript_tools.py created
- [ ] Imports updated
- [ ] Old rmscript/ deleted
- [ ] .rmscript files preserved

### Phase 4: Validation
- [ ] All 122 tests passing
- [ ] Robot testing (sim) passed
- [ ] Robot testing (hardware) passed
- [ ] Performance validated
- [ ] Documentation updated

### Phase 5: Web Demo
- [ ] WebSocketExecutionAdapter implemented
- [ ] FastAPI backend created
- [ ] Frontend built
- [ ] Demo deployed

---

## Risk Mitigation

### Risk: Behavioral Changes
- **Mitigation**: Extensive regression testing with all existing .rmscript files
- **Validation**: Side-by-side execution comparison before/after
- **Status**: Phase 4 task

### Risk: Performance Regression
- **Mitigation**: Benchmark compilation speed and execution smoothness
- **Validation**: Measure 100 compilations before/after, ensure <5% degradation
- **Status**: Phase 4 task

### Risk: Dependency Hell
- **Mitigation**: Minimal dependencies in core, optional extras
- **Validation**: Test installation in clean virtualenv
- **Status**: Phase 1 task

### Risk: Breaking Existing Profiles
- **Mitigation**: Update all profiles in same PR, maintain backward compatibility
- **Validation**: Test all profiles before merging
- **Status**: Phase 3 task

---

## Timeline

| Week | Phase | Focus | Deliverable |
|------|-------|-------|-------------|
| 1 | Phase 1 | Core extraction | Standalone rmscript package |
| 1-2 | Phase 2 | Adapters | QueueExecutionAdapter |
| 2 | Phase 3 | Integration | Updated conversation app |
| 2-3 | Phase 4 | Testing | Validated system |
| 3-4+ | Phase 5 | Web demo | Demo deployment |

---

## Success Criteria

- [ ] rmscript package compiles scripts to IR
- [ ] All 122 existing tests pass in new architecture
- [ ] Conversation app uses QueueExecutionAdapter
- [ ] All .rmscript tools work identically
- [ ] Picture capture works
- [ ] Sound playback works
- [ ] Performance <5% degradation
- [ ] rmscript published to PyPI
- [ ] Documentation updated
- [ ] Web demo prototype functional

---

## Notes

**Important Files**:
- Plan: `/Users/david/.claude/plans/shimmying-wiggling-lantern.md`
- Migration Doc: `/Users/david/code/reachy_mini_conversation_app/rmscript_migration.md`

**References**:
- Original rmscript: `src/reachy_mini_conversation_app/rmscript/`
- Tests: `tests/rmscript/` (122 tests)
- Language reference: `src/reachy_mini_conversation_app/rmscript/rmscript_reference_doc.md`

**Next Steps**: Start Phase 1 - Create standalone rmscript package
