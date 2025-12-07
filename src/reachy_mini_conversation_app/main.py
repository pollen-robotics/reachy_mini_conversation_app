"""Entrypoint for the Reachy Mini conversation app."""

import os
import sys
import time
import asyncio
import argparse
import threading
from typing import Any, Dict, List, Optional

import gradio as gr
from fastapi import FastAPI
from fastrtc import Stream
from gradio.utils import get_space

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini_conversation_app.utils import (
    parse_args,
    setup_logger,
    handle_vision_stuff,
)
from reachy_mini_conversation_app.config import config
from pathlib import Path


def update_chatbot(chatbot: List[Dict[str, Any]], response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update the chatbot with AdditionalOutputs."""
    chatbot.append(response)
    return chatbot


def main() -> None:
    """Entrypoint for the Reachy Mini conversation app."""
    args = parse_args()
    run(args)


def run(
    args: argparse.Namespace,
    robot: ReachyMini = None,
    app_stop_event: Optional[threading.Event] = None,
    settings_app: Optional[FastAPI] = None,
) -> None:
    """Run the Reachy Mini conversation app."""
    # Putting these dependencies here makes the dashboard faster to load when the conversation app is installed
    from reachy_mini_conversation_app.moves import MovementManager
    from reachy_mini_conversation_app.console import LocalStream
    from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
    from reachy_mini_conversation_app.audio.head_wobbler import HeadWobbler

    logger = setup_logger(args.debug)
    logger.info("Starting Reachy Mini Conversation App")

    if args.no_camera and args.head_tracker is not None:
        logger.warning("Head tracking is not activated due to --no-camera.")

    if robot is None:
        robot = ReachyMini()

    # Check if running in simulation mode without --gradio
    if robot.client.get_status()["simulation_enabled"] and not args.gradio:
        logger.error(
            "Simulation mode requires Gradio interface. Please use --gradio flag when running in simulation mode.",
        )
        robot.client.disconnect()
        sys.exit(1)

    camera_worker, _, vision_manager = handle_vision_stuff(args, robot)

    movement_manager = MovementManager(
        current_robot=robot,
        camera_worker=camera_worker,
    )

    head_wobbler = HeadWobbler(set_speech_offsets=movement_manager.set_speech_offsets)

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        vision_manager=vision_manager,
        head_wobbler=head_wobbler,
    )
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    logger.debug(f"Current file absolute path: {current_file_path}")
    chatbot = gr.Chatbot(
        type="messages",
        resizable=True,
        avatar_images=(
            os.path.join(current_file_path, "images", "user_avatar.png"),
            os.path.join(current_file_path, "images", "reachymini_avatar.png"),
        ),
    )
    logger.debug(f"Chatbot avatar images: {chatbot.avatar_images}")

    handler = OpenaiRealtimeHandler(deps, gradio_mode=args.gradio)

    stream_manager: gr.Blocks | LocalStream | None = None

    if args.gradio:
        api_key_textbox = gr.Textbox(
            label="OPENAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY") if not get_space() else "",
        )
        # Helpers for personalities management (profiles folder)
        profiles_root = Path(__file__).parent / "profiles"

        def _list_personalities() -> list[str]:
            names: list[str] = []
            try:
                if profiles_root.exists():
                    # Built-in profiles (exclude holder for user profiles)
                    for p in sorted(profiles_root.iterdir()):
                        if p.name == "user_personalities":
                            continue
                        if p.is_dir() and (p / "instructions.txt").exists():
                            names.append(p.name)
                    # User-created profiles live under profiles/user_personalities/<name>
                    user_dir = profiles_root / "user_personalities"
                    if user_dir.exists():
                        for p in sorted(user_dir.iterdir()):
                            if p.is_dir() and (p / "instructions.txt").exists():
                                # Encode value as path segment so tools can resolve folder
                                names.append(f"user_personalities/{p.name}")
            except Exception:
                pass
            return names

        DEFAULT_OPTION = "(built-in default)"

        def _current_selection_value() -> str:
            return config.REACHY_MINI_CUSTOM_PROFILE or DEFAULT_OPTION

        def _resolve_profile_dir(selection: str) -> Path:
            return profiles_root / selection

        def _read_instructions_for(name: str) -> str:
            try:
                if name == DEFAULT_OPTION:
                    # Show baked-in default prompt
                    default_file = Path(__file__).parent / "prompts" / "default_prompt.txt"
                    if default_file.exists():
                        return default_file.read_text(encoding="utf-8").strip()
                    return ""
                target = _resolve_profile_dir(name) / "instructions.txt"
                if target.exists():
                    return target.read_text(encoding="utf-8").strip()
                return ""
            except Exception as e:
                return f"Could not load instructions: {e}"

        async def _apply_personality(selected: str) -> tuple[str, str]:
            profile = None if selected == DEFAULT_OPTION else selected
            status = await handler.apply_personality(profile)
            preview = _read_instructions_for(selected)
            return status, preview

        def _sanitize_name(name: str) -> str:
            import re

            s = name.strip()
            s = re.sub(r"\s+", "_", s)
            s = re.sub(r"[^a-zA-Z0-9_-]", "", s)
            return s

        def _create_personality(name: str, instructions: str, tools_text: str):  # type: ignore[no-untyped-def]
            name_s = _sanitize_name(name)
            if not name_s:
                return gr.update(), gr.update(), "Please enter a valid name."
            try:
                target_dir = profiles_root / "user_personalities" / name_s
                target_dir.mkdir(parents=True, exist_ok=False)
                # Write instructions
                (target_dir / "instructions.txt").write_text(instructions.strip() + "\n", encoding="utf-8")
                # Write tools.txt
                (target_dir / "tools.txt").write_text(tools_text.strip() + "\n", encoding="utf-8")

                choices = _list_personalities()
                value = f"user_personalities/{name_s}"
                if value not in choices:
                    choices.append(value)
                return gr.update(choices=[DEFAULT_OPTION, *sorted(choices)], value=value), gr.update(value=instructions), f"Created personality '{name_s}'."
            except FileExistsError:
                choices = _list_personalities()
                value = f"user_personalities/{name_s}"
                if value not in choices:
                    choices.append(value)
                return gr.update(choices=[DEFAULT_OPTION, *sorted(choices)], value=value), gr.update(value=instructions), f"Personality '{name_s}' already exists."
            except Exception as e:
                return gr.update(), gr.update(), f"Failed to create personality: {e}"

        # Build personality UI components to place in the side panel
        personalities_dropdown = gr.Dropdown(
            label="Select personality",
            choices=[DEFAULT_OPTION, *(_list_personalities())],
            value=_current_selection_value(),
        )
        apply_btn = gr.Button("Apply personality")
        status_md = gr.Markdown(visible=True)
        preview_md = gr.Markdown(value=_read_instructions_for(_current_selection_value()))
        person_name_tb = gr.Textbox(label="Personality name")
        person_instr_ta = gr.TextArea(label="Personality instructions", lines=10)
        tools_txt_ta = gr.TextArea(label="tools.txt", lines=10)
        new_personality_btn = gr.Button("New personality")
        # Convenience: discovered tools (shared + profile-local)
        available_tools_cg = gr.CheckboxGroup(label="Available tools (helper)", choices=[], value=[])
        load_file_dropdown = gr.Dropdown(label="Custom tool files (*.py)", choices=[], value=None)
        file_content_ta = gr.TextArea(label="Selected file content", lines=12)
        new_file_name_tb = gr.Textbox(label="New custom tool filename (e.g., my_tool.py)")
        create_file_btn = gr.Button("Create file")
        save_file_btn = gr.Button("Save file")
        delete_file_btn = gr.Button("Delete file")
        save_btn = gr.Button("Save personality (instructions + tools)")

        # Build the streaming UI first
        stream = Stream(
            handler=handler,
            mode="send-receive",
            modality="audio",
            # Keep original order for first two to preserve expected arg indices
            additional_inputs=[
                chatbot,
                api_key_textbox,
                personalities_dropdown,
                apply_btn,
                status_md,
                preview_md,
                person_name_tb,
                person_instr_ta,
                tools_txt_ta,
                new_personality_btn,
                available_tools_cg,
                load_file_dropdown,
                file_content_ta,
                new_file_name_tb,
                create_file_btn,
                save_file_btn,
                delete_file_btn,
                save_btn,
            ],
            additional_outputs=[chatbot],
            additional_outputs_handler=update_chatbot,
            ui_args={"title": "Talk with Reachy Mini"},
        )
        stream_manager = stream.ui
        if not settings_app:
            app = FastAPI()
        else:
            app = settings_app

        # Wire events for side panel controls inside the Blocks context
        with stream_manager:
            apply_btn.click(
                fn=_apply_personality,
                inputs=[personalities_dropdown],
                outputs=[status_md, preview_md],
            )

            def _available_tools_for(selected: str):  # type: ignore[no-untyped-def]
                # Shared tools list
                tools_dir = Path(__file__).parent / "tools"
                shared = []
                try:
                    for py in tools_dir.glob("*.py"):
                        if py.stem in {"__init__", "core_tools"}:
                            continue
                        shared.append(py.stem)
                except Exception:
                    pass
                # Profile-local tools
                local = []
                try:
                    if selected != DEFAULT_OPTION:
                        for py in (profiles_root / selected).glob("*.py"):
                            local.append(py.stem)
                except Exception:
                    pass
                all_tools = sorted(set(shared + local))
                return gr.update(choices=all_tools)

            def _parse_enabled_tools(text: str) -> list[str]:
                enabled: list[str] = []
                for line in text.splitlines():
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    enabled.append(s)
                return enabled

            def _load_profile_for_edit(selected: str):  # type: ignore[no-untyped-def]
                instr = _read_instructions_for(selected)
                # tools.txt
                tools_txt = ""
                if selected != DEFAULT_OPTION:
                    tp = _resolve_profile_dir(selected) / "tools.txt"
                    if tp.exists():
                        tools_txt = tp.read_text(encoding="utf-8")
                # available tools and enabled tools
                tools_dir = Path(__file__).parent / "tools"
                shared = [py.stem for py in tools_dir.glob("*.py") if py.stem not in {"__init__", "core_tools"}]
                local = []
                if selected != DEFAULT_OPTION:
                    local = [py.stem for py in (profiles_root / selected).glob("*.py")]
                all_tools = sorted(set(shared + local))
                enabled = _parse_enabled_tools(tools_txt)

                # files under profile (py only)
                files = []
                if selected != DEFAULT_OPTION:
                    files = [p.name for p in (profiles_root / selected).glob("*.py")]
                file_value = files[0] if files else None
                file_content = ""
                if file_value:
                    file_content = (profiles_root / selected / file_value).read_text(encoding="utf-8")

                # Name textbox
                from pathlib import Path as _P
                name_for_edit = "" if selected == DEFAULT_OPTION else _P(selected).name

                return (
                    instr,
                    tools_txt,
                    gr.update(choices=all_tools, value=[t for t in enabled if t in all_tools]),
                    gr.update(choices=files, value=file_value),
                    file_content,
                    name_for_edit,
                )

            personalities_dropdown.change(
                fn=_load_profile_for_edit,
                inputs=[personalities_dropdown],
                outputs=[person_instr_ta, tools_txt_ta, available_tools_cg, load_file_dropdown, file_content_ta, person_name_tb],
            )

            # Keep the name field in sync with selection (basename for user profiles)
            def _selected_name_for_edit(selected: str) -> str:
                if selected == DEFAULT_OPTION:
                    return ""
                p = Path(selected)
                return p.name

            # Initial tools choices for current selection
            personalities_dropdown.change(
                fn=_available_tools_for,
                inputs=[personalities_dropdown],
                outputs=[available_tools_cg],
            )

            # Start a new personality from scratch
            def _new_personality():  # type: ignore[no-untyped-def]
                # Empty name and instructions
                name_val = ""
                instr_val = ""
                # Blank tools.txt with a helpful header
                tools_txt_val = "# tools enabled for this profile\n"
                # Shared tools list (no local tools for a new profile yet)
                tools_dir = Path(__file__).parent / "tools"
                try:
                    shared = [py.stem for py in tools_dir.glob("*.py") if py.stem not in {"__init__", "core_tools"}]
                except Exception:
                    shared = []
                # No files yet for a new personality
                files_dd = gr.update(choices=[], value=None)
                file_content_val = ""
                # Update status to guide the user
                status_text = "Creating a new personality. Fill the fields and click 'Save'."
                return (
                    gr.update(value=name_val),
                    gr.update(value=instr_val),
                    gr.update(value=tools_txt_val),
                    gr.update(choices=sorted(shared), value=[]),
                    files_dd,
                    gr.update(value=file_content_val),
                    status_text,
                )

            new_personality_btn.click(
                fn=_new_personality,
                inputs=[],
                outputs=[person_name_tb, person_instr_ta, tools_txt_ta, available_tools_cg, load_file_dropdown, file_content_ta, status_md],
            )

            def _save_personality(name: str, instructions: str, tools_text: str):  # type: ignore[no-untyped-def]
                name_s = _sanitize_name(name)
                if not name_s:
                    return gr.update(), gr.update(), "Please enter a valid name."
                try:
                    target_dir = profiles_root / "user_personalities" / name_s
                    target_dir.mkdir(parents=True, exist_ok=True)
                    # Write instructions
                    (target_dir / "instructions.txt").write_text(instructions.strip() + "\n", encoding="utf-8")
                    # Write tools.txt
                    (target_dir / "tools.txt").write_text(tools_text.strip() + "\n", encoding="utf-8")
                    # Ensure tools.txt exists (copy from default once)
                    tools_target = target_dir / "tools.txt"
                    if not tools_target.exists():
                        tools_target.write_text("# tools enabled for this profile\n", encoding="utf-8")

                    # Refresh choices and select the saved profile
                    choices = _list_personalities()
                    value = f"user_personalities/{name_s}"
                    if value not in choices:
                        choices.append(value)
                    return gr.update(choices=[DEFAULT_OPTION, *sorted(choices)], value=value), gr.update(value=instructions), f"Saved personality '{name_s}'."
                except Exception as e:
                    return gr.update(), gr.update(), f"Failed to save personality: {e}"

            save_btn.click(
                fn=_save_personality,
                inputs=[person_name_tb, person_instr_ta, tools_txt_ta],
                outputs=[personalities_dropdown, person_instr_ta, status_md],
            ).then(
                fn=_apply_personality,
                inputs=[personalities_dropdown],
                outputs=[status_md, preview_md],
            )

            def _sync_tools_from_checks(selected: list[str], current_text: str):  # type: ignore[no-untyped-def]
                # Keep comments from current_text at the top, then list selected tools
                comments = [ln for ln in current_text.splitlines() if ln.strip().startswith("#")]
                body = "\n".join(selected)
                out = ("\n".join(comments) + ("\n" if comments else "") + body).strip() + "\n"
                return gr.update(value=out)

            available_tools_cg.change(
                fn=_sync_tools_from_checks,
                inputs=[available_tools_cg, tools_txt_ta],
                outputs=[tools_txt_ta],
            )

            def _refresh_file_content(selected_profile: str, filename: str | None):  # type: ignore[no-untyped-def]
                if not filename:
                    return ""
                path = profiles_root / selected_profile / filename
                try:
                    if path.exists():
                        return path.read_text(encoding="utf-8")
                except Exception as e:
                    return f"Error loading file: {e}"
                return ""

            load_file_dropdown.change(
                fn=_refresh_file_content,
                inputs=[personalities_dropdown, load_file_dropdown],
                outputs=[file_content_ta],
            )

            def _create_file(selected_profile: str, filename: str):  # type: ignore[no-untyped-def]
                name = filename.strip()
                if not name.endswith(".py"):
                    return gr.update(), "Filename must end with .py"
                # Always create in user_personalities
                base = Path(selected_profile).name if selected_profile != DEFAULT_OPTION else (person_name_tb.value or "new_profile")  # type: ignore[attr-defined]
                target_dir = profiles_root / "user_personalities" / base
                target_dir.mkdir(parents=True, exist_ok=True)
                path = target_dir / name
                if path.exists():
                    # do nothing
                    pass
                else:
                    path.write_text("# custom tool\n", encoding="utf-8")
                # refresh list to show new file
                files = [p.name for p in target_dir.glob("*.py")]
                return gr.update(choices=files, value=name), f"Created {name}"

            create_file_btn.click(
                fn=_create_file,
                inputs=[personalities_dropdown, new_file_name_tb],
                outputs=[load_file_dropdown, status_md],
            )

            def _save_file(selected_profile: str, filename: str | None, content: str):  # type: ignore[no-untyped-def]
                if not filename:
                    return "No file selected."
                base = Path(selected_profile).name if selected_profile != DEFAULT_OPTION else (person_name_tb.value or "new_profile")  # type: ignore[attr-defined]
                target_dir = profiles_root / "user_personalities" / base
                target_dir.mkdir(parents=True, exist_ok=True)
                path = target_dir / filename
                path.write_text(content, encoding="utf-8")
                return f"Saved {filename}"

            save_file_btn.click(
                fn=_save_file,
                inputs=[personalities_dropdown, load_file_dropdown, file_content_ta],
                outputs=[status_md],
            )

            def _delete_file(selected_profile: str, filename: str | None):  # type: ignore[no-untyped-def]
                if not filename:
                    return gr.update(), "No file selected."
                # Only delete from user profiles
                if not selected_profile.startswith("user_personalities/"):
                    return gr.update(), "Cannot delete from official profile. Save to user profile first."
                path = profiles_root / selected_profile / filename
                try:
                    if path.exists():
                        path.unlink()
                    files = [p.name for p in (profiles_root / selected_profile).glob("*.py")]
                    new_value = files[0] if files else None
                    return gr.update(choices=files, value=new_value), f"Deleted {filename}"
                except Exception as e:
                    return gr.update(), f"Failed to delete: {e}"

            delete_file_btn.click(
                fn=_delete_file,
                inputs=[personalities_dropdown, load_file_dropdown],
                outputs=[load_file_dropdown, status_md],
            )

        app = gr.mount_gradio_app(app, stream.ui, path="/")
    else:
        stream_manager = LocalStream(handler, robot)

    # Each async service → its own thread/loop
    movement_manager.start()
    head_wobbler.start()
    if camera_worker:
        camera_worker.start()
    if vision_manager:
        vision_manager.start()

    def poll_stop_event() -> None:
        """Poll the stop event to allow graceful shutdown."""
        if app_stop_event is not None:
            app_stop_event.wait()

        logger.info("App stop event detected, shutting down...")
        try:
            stream_manager.close()
        except Exception as e:
            logger.error(f"Error while closing stream manager: {e}")

    if app_stop_event:
        threading.Thread(target=poll_stop_event, daemon=True).start()

    try:
        stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
    finally:
        movement_manager.stop()
        head_wobbler.stop()
        if camera_worker:
            camera_worker.stop()
        if vision_manager:
            vision_manager.stop()

        # prevent connection to keep alive some threads
        robot.client.disconnect()
        time.sleep(1)
        logger.info("Shutdown complete.")


class ReachyMiniConversationApp(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for the conversation app."""

    custom_app_url = "http://127.0.0.1:7860/"
    dont_start_webserver = True

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the Reachy Mini conversation app."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        args = parse_args()
        args.gradio = True  # Force gradio for Reachy Mini App integration
        run(args, robot=reachy_mini, app_stop_event=stop_event, settings_app=self.settings_app)


if __name__ == "__main__":
    main()
