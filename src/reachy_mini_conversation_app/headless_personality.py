"""Headless personality management (console-based).

Provides an interactive CLI to browse, preview, apply, create and edit
"personalities" (profiles) when running without Gradio.

This module is intentionally not shared with the Gradio implementation to
avoid coupling and keep responsibilities clear for headless mode.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .config import config


DEFAULT_OPTION = "(built-in default)"


def _profiles_root() -> Path:
    return Path(__file__).parent / "profiles"


def _prompts_dir() -> Path:
    return Path(__file__).parent / "prompts"


def _tools_dir() -> Path:
    return Path(__file__).parent / "tools"


def _sanitize_name(name: str) -> str:
    import re

    s = name.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_-]", "", s)
    return s


def list_personalities() -> List[str]:
    names: List[str] = []
    root = _profiles_root()
    try:
        if root.exists():
            for p in sorted(root.iterdir()):
                if p.name == "user_personalities":
                    continue
                if p.is_dir() and (p / "instructions.txt").exists():
                    names.append(p.name)
        udir = root / "user_personalities"
        if udir.exists():
            for p in sorted(udir.iterdir()):
                if p.is_dir() and (p / "instructions.txt").exists():
                    names.append(f"user_personalities/{p.name}")
    except Exception:
        pass
    return names


def resolve_profile_dir(selection: str) -> Path:
    return _profiles_root() / selection


def read_instructions_for(name: str) -> str:
    try:
        if name == DEFAULT_OPTION:
            df = _prompts_dir() / "default_prompt.txt"
            return df.read_text(encoding="utf-8").strip() if df.exists() else ""
        target = resolve_profile_dir(name) / "instructions.txt"
        return target.read_text(encoding="utf-8").strip() if target.exists() else ""
    except Exception as e:
        return f"Could not load instructions: {e}"


def available_tools_for(selected: str) -> List[str]:
    shared: List[str] = []
    try:
        for py in _tools_dir().glob("*.py"):
            if py.stem in {"__init__", "core_tools"}:
                continue
            shared.append(py.stem)
    except Exception:
        pass
    local: List[str] = []
    try:
        if selected != DEFAULT_OPTION:
            for py in resolve_profile_dir(selected).glob("*.py"):
                local.append(py.stem)
    except Exception:
        pass
    return sorted(set(shared + local))


def _write_profile(name_s: str, instructions: str, tools_text: str, voice: str = "cedar") -> None:
    target_dir = _profiles_root() / "user_personalities" / name_s
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "instructions.txt").write_text(instructions.strip() + "\n", encoding="utf-8")
    (target_dir / "tools.txt").write_text((tools_text or "").strip() + "\n", encoding="utf-8")
    (target_dir / "voice.txt").write_text((voice or "cedar").strip() + "\n", encoding="utf-8")


@dataclass
class PersonalityState:
    current: str = DEFAULT_OPTION


class HeadlessPersonalityCLI:
    """Interactive console for managing personalities in headless mode."""

    def __init__(self, handler) -> None:
        self._handler = handler
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._loop = None  # asyncio loop, set by set_loop
        self._state = PersonalityState(current=config.REACHY_MINI_CUSTOM_PROFILE or DEFAULT_OPTION)

    def set_loop(self, loop) -> None:
        self._loop = loop

    def start(self) -> None:
        if not sys.stdin or not sys.stdin.isatty():  # avoid blocking when no TTY
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="personality-cli", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ---- Core CLI loop ----
    def _run(self) -> None:
        self._print_banner()
        while not self._stop.is_set():
            try:
                cmd = input("[personality] > ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\n(exit)")
                break
            if not cmd:
                continue

            try:
                self._handle(cmd)
            except Exception as e:
                print(f"Error: {e}")

    def _print_banner(self) -> None:
        print("\nHeadless Personality Manager")
        print("Type 'help' for commands. Running alongside audio stream.")

    # ---- Commands ----
    def _handle(self, cmd: str) -> None:
        parts = cmd.split()
        if not parts:
            return
        op = parts[0].lower()

        if op in {"help", "?"}:
            self._help()
        elif op in {"list", "ls"}:
            self._list()
        elif op == "show":
            self._show(self._resolve_arg(parts[1:]))
        elif op == "apply":
            self._apply(self._resolve_arg(parts[1:]))
        elif op == "new":
            self._new()
        elif op == "edit":
            self._edit(self._resolve_arg(parts[1:]))
        elif op in {"tools"}:
            self._tools(self._resolve_arg(parts[1:]))
        elif op in {"voices", "voice"}:
            self._voices()
        elif op in {"quit", "exit"}:
            self.stop()
        else:
            print("Unknown command. Type 'help'.")

    def _help(self) -> None:
        print("""
Commands:
  list|ls              List available personalities
  show <name|#>        Show instructions preview
  apply <name|#>       Apply personality immediately
  new                  Create a new personality (guided)
  edit <name|#>        Edit an existing personality
  tools [name|#]       List available tools for a profile
  voices               List available voices from backend
  quit|exit            Close this console (audio keeps running)
        """.rstrip())

    def _list(self) -> None:
        names = [DEFAULT_OPTION, *list_personalities()]
        cur = self._state.current
        for i, n in enumerate(names):
            mark = "*" if n == cur else " "
            print(f"{i:2d}. {mark} {n}")

    def _resolve_arg(self, rest: Iterable[str]) -> str:
        if not rest:
            return self._state.current
        token = " ".join(rest).strip()
        # index?
        try:
            idx = int(token)
            names = [DEFAULT_OPTION, *list_personalities()]
            if 0 <= idx < len(names):
                return names[idx]
        except Exception:
            pass
        return token

    def _show(self, name: str) -> None:
        print(f"--- {name} ---")
        print(read_instructions_for(name))
        print("---------------")

    def _apply(self, name: str) -> None:
        # Update state immediately
        self._state.current = name
        # Schedule apply on the OpenAI handler loop if available
        if self._loop is None:
            print("(apply queued; loop not ready yet)")
            return
        try:
            import asyncio

            async def _do_apply():
                sel = None if name == DEFAULT_OPTION else name
                status = await self._handler.apply_personality(sel)
                return status

            fut = asyncio.run_coroutine_threadsafe(_do_apply(), self._loop)
            status = fut.result(timeout=10)
            print(status)
        except Exception as e:
            print(f"Failed to apply: {e}")

    def _prompt_multiline(self, prompt: str) -> str:
        print(prompt)
        print("End with a single line containing only 'EOF'.")
        lines: List[str] = []
        while True:
            try:
                ln = input()
            except EOFError:
                break
            if ln.strip() == "EOF":
                break
            lines.append(ln)
        return "\n".join(lines)

    def _new(self) -> None:
        raw = input("Name: ").strip()
        name_s = _sanitize_name(raw)
        if not name_s:
            print("Invalid name.")
            return
        instr = self._prompt_multiline("Enter instructions:")
        tools = self._prompt_multiline("Enter tools (one per line):")
        voice = input("Voice [cedar]: ").strip() or "cedar"
        try:
            _write_profile(name_s, instr, tools, voice)
            print(f"Saved personality 'user_personalities/{name_s}'.")
            self._apply(f"user_personalities/{name_s}")
        except Exception as e:
            print(f"Failed to save: {e}")

    def _edit(self, name: str) -> None:
        if name == DEFAULT_OPTION:
            print("Cannot edit built-in default.")
            return
        pdir = resolve_profile_dir(name)
        if not pdir.exists():
            print("Profile not found.")
            return
        cur_i = read_instructions_for(name)
        print("Current instructions:")
        print(cur_i)
        instr = self._prompt_multiline("New instructions (leave empty to keep):")
        if not instr.strip():
            instr = cur_i
        tools_path = pdir / "tools.txt"
        cur_tools = tools_path.read_text(encoding="utf-8") if tools_path.exists() else ""
        print("Current tools.txt:")
        print(cur_tools)
        tools = self._prompt_multiline("New tools (one per line, empty to keep):")
        if not tools.strip():
            tools = cur_tools
        voice_path = pdir / "voice.txt"
        cur_voice = voice_path.read_text(encoding="utf-8").strip() if voice_path.exists() else "cedar"
        voice = input(f"Voice [{cur_voice}]: ").strip() or cur_voice
        try:
            _write_profile(Path(name).name if "/" in name else name, instr, tools, voice)
            print("Saved.")
        except Exception as e:
            print(f"Failed to save: {e}")

    def _tools(self, name: Optional[str]) -> None:
        name = name or self._state.current
        tools = available_tools_for(name)
        for t in tools:
            print(f"- {t}")

    def _voices(self) -> None:
        if self._loop is None:
            print("Loop not ready yet.")
            return
        try:
            import asyncio

            fut = asyncio.run_coroutine_threadsafe(self._handler.get_available_voices(), self._loop)
            voices = fut.result(timeout=10)
            for v in voices:
                print(f"- {v}")
        except Exception as e:
            print(f"Failed to fetch voices: {e}")

