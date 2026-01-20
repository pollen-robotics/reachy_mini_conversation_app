# Contributing

Thank you for helping improve Reachy Mini Conversation App! 🤖

We welcome all contributions: bug fixes, new features, documentation, testing, and more. Please respect our [code of conduct](CODE_OF_CONDUCT.md).

## Quick Start

1. Fork and clone the repo:
   ```bash
   git clone https://github.com/pollen-robotics/reachy_mini_conversation_app
   cd reachy_mini_conversation_app
   ```
2. Follow the [README installation guide](README.md#installation) to set up dependencies and `.env`.
3. Run the contributor checks after your changes:
   ```bash
   uv run ruff check . --fix
   uv run ruff format .
   uv run mypy --pretty --show-error-codes .
   uv run pytest tests/ -v
   ```

## Development Workflow

### 1. Create an Issue

Open an issue first describing the bug fix, feature, or improvement you plan to work on.

### 2. Create a Branch

Create a branch using the issue number and label assigned to the issue:

```bash
fix/485-handle-camera-timeout
feat/123-add-head-tracking
docs/67-update-installation-guide
```

**Format:** `<type>/<issue-number>-<short-description>`

Common types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`

### 3. Make Your Changes

Follow the [quality checklist](#before-opening-a-pr) below to ensure your changes meet our standards.

### 4. Write Conventional Commits

**This project auto-releases based on commit messages.** Use conventional commits:

```bash
# Bug fix -> patch release (0.1.0 -> 0.1.1)
git commit -m "fix: handle camera timeout"

# New feature -> minor release (0.1.0 -> 0.2.0)  
git commit -m "feat: add head tracking tool"

# Documentation, refactor, tests -> no release
git commit -m "docs: update installation guide"
git commit -m "refactor: simplify motion loop"
git commit -m "test: add vision processor tests"
```

**Format:** `<type>: <description>` (lowercase, no period at end)

**Common types:**
- `feat:` - new feature (minor bump)
- `fix:` - bug fix (patch bump)  
- `docs:`, `test:`, `refactor:`, `style:`, `chore:` - no version bump

For a complete list of commit types and the full specification, see the [Conventional Commits specification](https://www.conventionalcommits.org/).

⚡️ **Breaking changes** (use with caution!):
```bash
# Method 1: Add ! after type
git commit -m "feat!: redesign motion API"

# Method 2: Add footer
git commit -m "feat: change camera tool

BREAKING CHANGE: camera tool now requires question parameter"
```

💡 **Not sure?** Just ask in your PR - maintainers can help with commit messages during merge.

<details>
<summary><b>📖 More commit examples and guidelines</b></summary>

### Good Commits
```bash
git commit -m "fix: handle missing camera frames"
git commit -m "feat(vision): add YOLO head tracker"
git commit -m "perf: reduce tool dispatch latency"
```

### Bad Commits
```bash
git commit -m "update stuff"  # Too vague
git commit -m "Fixed bug"     # Wrong format
git commit -m "feat: Add feature"  # Wrong capitalization
```

### Scopes (Optional)
Add context with scopes: `vision`, `motion`, `conversation`, `tools`, `config`

### Preview Your Release
```bash
uv sync --group dev # to make sure you have python-semantic-release installed
uv run semantic-release version --print
```

### Multiple Changes
Try to keep commits focused. If you must combine changes, semantic-release will still work but changelogs will be less granular.

</details>

<details>
<summary><b>🔒 What counts as a breaking change?</b></summary>

**Breaking changes:** Changes that affect how users interact with the application:
- Removing or renaming CLI flags, such as dropping `--debug`;
- Changing configuration file formats, including renamed `.env` variables or profile schema fields;
- Breaking custom tool compatibility by renaming tools in `src/reachy_mini_conversation_app/tools`;
- Changing profile file structures, like moving `profiles/default/` or renaming `instructions.txt`;
- Altering public API entrypoints, such as renaming `reachy_mini_conversation_app.main` or CLI entry points.

Internal code refactoring doesn't require a breaking change marker. For example:
- `refactor: move camera processing into camera_worker.py`
- `refactor: split headless UI helpers into headless_personality_ui.py`
- `refactor: reorganize prompts in src/reachy_mini_conversation_app/prompts/`

**When unsure:** Use `feat:` or `fix:` without the breaking marker. Ask in your PR if needed. We can make breaking changes later, but can't undo them!

</details>

### 5. Open a Pull Request

Open a PR and fill out the template. Our CI will automatically check:
- Ruff linting and formatting
- Type checking with mypy
- Test suite with pytest
- `uv.lock` validation

Maintainers may request changes during review. Once approved, your PR will be squashed into one commit following conventional format - this squashed commit message drives automated releases!

**About uv.lock validation:** our CI checks that `uv.lock` is in sync with `pyproject.toml` by validating the lockfile without modifying it. When you change dependencies in `pyproject.toml`, update the lockfile locally by running `uv lock`, then commit both `pyproject.toml` and `uv.lock` together. This avoids unintended updates and ensures reproducible builds across contributors.

## Before Opening a PR

- All tests pass locally (`uv run pytest tests/ -v`)
- Code is formatted (`uv run ruff format .`) and type-checked (`uv run mypy .`)
- Commits follow [conventional format](#4-write-conventional-commits) (this is critical!)
- Added tests for bug fixes or new features
- Updated docs if needed
- No secrets or `.env` files committed
- `uv.lock` is up to date if you changed dependencies

<details>
<summary><b>🧪 Quality checks reference</b></summary>

### Linting
```bash
uv run ruff check . --fix      # Auto-fix issues
uv run ruff format .            # Format code
```

### Type Checking
```bash
uv run mypy --pretty --show-error-codes .
```

### Testing
```bash
uv run pytest tests/ -v         # Run all tests
uv run pytest tests/ -v --cov  # With coverage
```

### All at Once
```bash
uv run mypy --pretty --show-error-codes . && uv run ruff check . --fix && uv run pytest tests/ -v
```

</details>

## Ways to Contribute

- **Bug fixes** - especially in conversation loop, vision, or motion;
- **Features** - new tools, integrations, or capabilities;
- **Profiles** - add personalities in `profiles/` directory;
- **Documentation** - improve README, docstrings, or guides;
- **Testing** - add tests or improve coverage.

**Testing guidelines:**
- Bug fixes should include a regression test;
- New features need at least one happy-path test.

🙋 Need help? Join our [Discord](https://discord.gg/5HcukpMX)!

## Filing Issues

- Search existing issues first;
- For bugs: include reproduction steps, OS, Python version, logs (use `--debug` flag);
- For features: describe the use case and expected behavior.

---

**Questions?** Open an issue or ask in your PR. We're here to help!

Thank you for contributing! 🦾
