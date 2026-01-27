"""Gradio personality UI components and wiring.

This module encapsulates the UI elements and logic related to managing
conversation "personalities" (profiles) so that `main.py` stays lean.
"""

from __future__ import annotations
from typing import Any
from pathlib import Path

import gradio as gr

from .config import LOCKED_PROFILE, config


class PersonalityUI:
    """Container for personality-related Gradio components."""

    def __init__(self) -> None:
        """Initialize the PersonalityUI instance."""
        # Constants and paths
        self.DEFAULT_OPTION = "(built-in default)"
        self._profiles_root = Path(__file__).parent / "profiles"
        self._tools_dir = Path(__file__).parent / "tools"
        self._prompts_dir = Path(__file__).parent / "prompts"

        # Components (initialized in create_components)
        self.personalities_dropdown: gr.Dropdown
        self.apply_btn: gr.Button
        self.status_md: gr.Markdown
        self.preview_md: gr.Markdown
        self.person_name_tb: gr.Textbox
        self.person_instr_ta: gr.TextArea
        self.tools_txt_ta: gr.TextArea
        self.voice_dropdown: gr.Dropdown
        self.new_personality_btn: gr.Button
        self.available_tools_cg: gr.CheckboxGroup
        self.save_btn: gr.Button

    # ---------- Filesystem helpers ----------
    def _list_personalities(self) -> list[str]:
        names: list[str] = []
        try:
            if self._profiles_root.exists():
                for p in sorted(self._profiles_root.iterdir()):
                    if p.name == "user_personalities":
                        continue
                    if p.is_dir() and (p / "instructions.txt").exists():
                        names.append(p.name)
                user_dir = self._profiles_root / "user_personalities"
                if user_dir.exists():
                    for p in sorted(user_dir.iterdir()):
                        if p.is_dir() and (p / "instructions.txt").exists():
                            names.append(f"user_personalities/{p.name}")
        except Exception:
            pass
        return names

    def _resolve_profile_dir(self, selection: str) -> Path:
        return self._profiles_root / selection

    def _read_instructions_for(self, name: str) -> str:
        try:
            if name == self.DEFAULT_OPTION:
                default_file = self._prompts_dir / "default_prompt.txt"
                if default_file.exists():
                    return default_file.read_text(encoding="utf-8").strip()
                return ""
            target = self._resolve_profile_dir(name) / "instructions.txt"
            if target.exists():
                return target.read_text(encoding="utf-8").strip()
            return ""
        except Exception as e:
            return f"Could not load instructions: {e}"

    @staticmethod
    def _sanitize_name(name: str) -> str:
        import re

        s = name.strip()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^a-zA-Z0-9_-]", "", s)
        return s

    # ---------- Public API ----------
    def create_components(self) -> None:
        """Instantiate Gradio components for the personality UI."""
        if LOCKED_PROFILE is not None:
            is_locked = True
            current_value: str = LOCKED_PROFILE
            dropdown_label = "Select personality (locked)"
            dropdown_choices: list[str] = [LOCKED_PROFILE]
        else:
            is_locked = False
            current_value = config.REACHY_MINI_CUSTOM_PROFILE or self.DEFAULT_OPTION
            dropdown_label = "Select personality"
            dropdown_choices = [self.DEFAULT_OPTION, *(self._list_personalities())]

        self.personalities_dropdown = gr.Dropdown(
            label=dropdown_label,
            choices=dropdown_choices,
            value=current_value,
            interactive=not is_locked,
        )
        self.apply_btn = gr.Button("Apply personality", interactive=not is_locked)
        self.status_md = gr.Markdown(visible=True)
        self.preview_md = gr.Markdown(value=self._read_instructions_for(current_value))
        self.person_name_tb = gr.Textbox(label="Personality name", interactive=not is_locked)
        self.person_instr_ta = gr.TextArea(label="Personality instructions", lines=10, interactive=not is_locked)
        self.tools_txt_ta = gr.TextArea(label="tools.txt", lines=10, interactive=not is_locked)
        self.voice_dropdown = gr.Dropdown(label="Voice", choices=["cedar"], value="cedar", interactive=not is_locked)
        self.new_personality_btn = gr.Button("New personality", interactive=not is_locked)
        self.available_tools_cg = gr.CheckboxGroup(label="Available tools (helper)", choices=[], value=[], interactive=not is_locked)
        self.save_btn = gr.Button("Save personality (instructions + tools)", interactive=not is_locked)

    def additional_inputs_ordered(self) -> list[Any]:
        """Return the additional inputs in the expected order for Stream."""
        return [
            self.personalities_dropdown,
            self.apply_btn,
            self.new_personality_btn,
            self.status_md,
            self.preview_md,
            self.person_name_tb,
            self.person_instr_ta,
            self.tools_txt_ta,
            self.voice_dropdown,
            self.available_tools_cg,
            self.save_btn,
        ]

    # ---------- Event wiring ----------
    def wire_events(self, handler: Any, blocks: gr.Blocks) -> None:
        """Attach event handlers to components within a Blocks context."""

        async def _apply_personality(selected: str) -> tuple[str, str]:
            if LOCKED_PROFILE is not None and selected != LOCKED_PROFILE:
                return (
                    f"Profile is locked to '{LOCKED_PROFILE}'. Cannot change personality.",
                    self._read_instructions_for(LOCKED_PROFILE),
                )
            profile = None if selected == self.DEFAULT_OPTION else selected
            status = await handler.apply_personality(profile)
            preview = self._read_instructions_for(selected)
            return status, preview

        def _read_voice_for(name: str) -> str:
            try:
                if name == self.DEFAULT_OPTION:
                    return "cedar"
                vf = self._resolve_profile_dir(name) / "voice.txt"
                if vf.exists():
                    v = vf.read_text(encoding="utf-8").strip()
                    return v or "cedar"
            except Exception:
                pass
            return "cedar"

        async def _fetch_voices(selected: str) -> dict[str, Any]:
            try:
                voices = await handler.get_available_voices()
                current = _read_voice_for(selected)
                if current not in voices:
                    current = "cedar"
                return gr.update(choices=voices, value=current)
            except Exception:
                return gr.update(choices=["cedar"], value="cedar")

        def _available_tools_for(selected: str) -> tuple[list[str], list[str]]:
            shared: list[str] = []
            try:
                for py in self._tools_dir.glob("*.py"):
                    if py.stem in {"__init__", "core_tools"}:
                        continue
                    shared.append(py.stem)
            except Exception:
                pass
            local: list[str] = []
            try:
                if selected != self.DEFAULT_OPTION:
                    for py in (self._profiles_root / selected).glob("*.py"):
                        local.append(py.stem)
            except Exception:
                pass
            return sorted(shared), sorted(local)

        def _parse_enabled_tools(text: str) -> list[str]:
            enabled: list[str] = []
            for line in text.splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                enabled.append(s)
            return enabled

        def _load_profile_for_edit(selected: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
            instr = self._read_instructions_for(selected)
            tools_txt = ""
            if selected != self.DEFAULT_OPTION:
                tp = self._resolve_profile_dir(selected) / "tools.txt"
                if tp.exists():
                    tools_txt = tp.read_text(encoding="utf-8")
            shared, local = _available_tools_for(selected)
            all_tools = sorted(set(shared + local))
            enabled = _parse_enabled_tools(tools_txt)
            status_text = f"Loaded profile '{selected}'."
            return (
                gr.update(value=instr),
                gr.update(value=tools_txt),
                gr.update(choices=all_tools, value=enabled),
                status_text,
            )

        def _new_personality() -> tuple[
            dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str, dict[str, Any]
        ]:
            try:
                # Prefill with hints
                instr_val = """# Write your instructions here\n# e.g., Keep responses concise and friendly."""
                tools_txt_val = "# tools enabled for this profile\n"
                return (
                    gr.update(value=""),
                    gr.update(value=instr_val),
                    gr.update(value=tools_txt_val),
                    gr.update(choices=sorted(_available_tools_for(self.DEFAULT_OPTION)[0]), value=[]),
                    "Fill in a name, instructions and (optional) tools, then Save.",
                    gr.update(value="cedar"),
                )
            except Exception:
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    "Failed to initialize new personality.",
                    gr.update(),
                )

        def _save_personality(
            name: str, instructions: str, tools_text: str, voice: str
        ) -> tuple[dict[str, Any], dict[str, Any], str]:
            name_s = self._sanitize_name(name)
            if not name_s:
                return gr.update(), gr.update(), "Please enter a valid name."
            try:
                target_dir = self._profiles_root / "user_personalities" / name_s
                target_dir.mkdir(parents=True, exist_ok=True)
                (target_dir / "instructions.txt").write_text(instructions.strip() + "\n", encoding="utf-8")
                (target_dir / "tools.txt").write_text(tools_text.strip() + "\n", encoding="utf-8")
                (target_dir / "voice.txt").write_text((voice or "cedar").strip() + "\n", encoding="utf-8")

                choices = self._list_personalities()
                value = f"user_personalities/{name_s}"
                if value not in choices:
                    choices.append(value)
                return (
                    gr.update(choices=[self.DEFAULT_OPTION, *sorted(choices)], value=value),
                    gr.update(value=instructions),
                    f"Saved personality '{name_s}'.",
                )
            except Exception as e:
                return gr.update(), gr.update(), f"Failed to save personality: {e}"

        def _sync_tools_from_checks(selected: list[str], current_text: str) -> dict[str, Any]:
            comments = [ln for ln in current_text.splitlines() if ln.strip().startswith("#")]
            body = "\n".join(selected)
            out = ("\n".join(comments) + ("\n" if comments else "") + body).strip() + "\n"
            return gr.update(value=out)

        with blocks:
            self.apply_btn.click(
                fn=_apply_personality,
                inputs=[self.personalities_dropdown],
                outputs=[self.status_md, self.preview_md],
            )

            self.personalities_dropdown.change(
                fn=_load_profile_for_edit,
                inputs=[self.personalities_dropdown],
                outputs=[self.person_instr_ta, self.tools_txt_ta, self.available_tools_cg, self.status_md],
            )

            blocks.load(
                fn=_fetch_voices,
                inputs=[self.personalities_dropdown],
                outputs=[self.voice_dropdown],
            )

            self.available_tools_cg.change(
                fn=_sync_tools_from_checks,
                inputs=[self.available_tools_cg, self.tools_txt_ta],
                outputs=[self.tools_txt_ta],
            )

            self.new_personality_btn.click(
                fn=_new_personality,
                inputs=[],
                outputs=[
                    self.person_name_tb,
                    self.person_instr_ta,
                    self.tools_txt_ta,
                    self.available_tools_cg,
                    self.status_md,
                    self.voice_dropdown,
                ],
            )

            self.save_btn.click(
                fn=_save_personality,
                inputs=[self.person_name_tb, self.person_instr_ta, self.tools_txt_ta, self.voice_dropdown],
                outputs=[self.personalities_dropdown, self.person_instr_ta, self.status_md],
            ).then(
                fn=_apply_personality,
                inputs=[self.personalities_dropdown],
                outputs=[self.status_md, self.preview_md],
            )
