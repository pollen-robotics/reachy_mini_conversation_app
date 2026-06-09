"""Entrypoint for the Reachy Mini conversation app."""

import os
import sys
import time
import asyncio
import argparse
import threading
from typing import Any, Dict, List, Optional
from pathlib import Path

import httpx
import gradio as gr
from fastapi import FastAPI
from fastrtc import Stream
from gradio.utils import get_space

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini.io.protocol import SetVolumeCmd
from reachy_mini_conversation_app.utils import (
    CameraVisionInitializationError,
    parse_args,
    setup_logger,
    initialize_camera_and_vision,
    log_connection_troubleshooting,
)


def update_chatbot(chatbot: List[Dict[str, Any]], response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update the chatbot with AdditionalOutputs."""
    chatbot.append(response)
    return chatbot


def main() -> None:
    """Entrypoint for the Reachy Mini conversation app."""
    args, _ = parse_args()
    if args.command == "tool-spaces":
        from reachy_mini_conversation_app.tool_spaces import handle_tool_spaces_command

        logger = setup_logger(args.debug)
        try:
            raise SystemExit(handle_tool_spaces_command(args))
        except Exception as exc:
            logger.error("tool-spaces command failed: %s", exc)
            raise SystemExit(1) from exc
    run(args)


def run(
    args: argparse.Namespace,
    robot: ReachyMini = None,
    app_stop_event: Optional[threading.Event] = None,
    settings_app: Optional[FastAPI] = None,
    instance_path: Optional[str] = None,
) -> None:
    """Run the Reachy Mini conversation app."""
    # Putting these dependencies here makes the dashboard faster to load when the conversation app is installed
    from reachy_mini_conversation_app.moves import MovementManager
    from reachy_mini_conversation_app.config import (
        HF_BACKEND,
        GEMINI_BACKEND,
        OPENAI_BACKEND,
        HF_LOCAL_CONNECTION_MODE,
        config,
        is_gemini_model,
        get_backend_label,
        get_hf_connection_selection,
        refresh_runtime_config_from_env,
    )
    from reachy_mini_conversation_app.startup_settings import (
        StartupSettings,
        load_startup_settings_into_runtime,
    )

    logger = setup_logger(args.debug)
    logger.info("Starting Reachy Mini Conversation App")
    startup_settings = StartupSettings()

    if instance_path is not None:
        try:
            from dotenv import load_dotenv

            env_path = Path(instance_path) / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=str(env_path), override=True)
                refresh_runtime_config_from_env()
                logger.info("Loaded instance configuration from %s", env_path)
        except Exception as e:
            logger.warning("Failed to load instance configuration: %s", e)

        try:
            startup_settings = load_startup_settings_into_runtime(instance_path)
        except Exception as e:
            logger.warning("Failed to load startup settings: %s", e)

    if config.BACKEND_PROVIDER == HF_BACKEND:
        logger.info(
            "Configured backend provider: %s (%s), connection mode: %s",
            config.BACKEND_PROVIDER,
            get_backend_label(config.BACKEND_PROVIDER),
            get_hf_connection_selection().mode,
        )
    else:
        logger.info(
            "Configured backend provider: %s (%s), model: %s",
            config.BACKEND_PROVIDER,
            get_backend_label(config.BACKEND_PROVIDER),
            config.MODEL_NAME,
        )

    from reachy_mini_conversation_app.console import LocalStream
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, initialize_tools
    from reachy_mini_conversation_app.conversation_handler import ConversationHandler

    try:
        initialize_tools(instance_path=instance_path)
    except Exception as e:
        logger.error("Failed to initialize tools: %s", e)
        sys.exit(1)

    if args.no_camera and args.head_tracker is not None:
        logger.warning("Head tracking disabled: --no-camera flag is set. Remove --no-camera to enable head tracking.")

    if robot is None:
        try:
            robot_kwargs = {}
            if args.robot_name is not None:
                robot_kwargs["robot_name"] = args.robot_name

            logger.info("Initializing ReachyMini (SDK will auto-detect appropriate backend)")
            robot = ReachyMini(**robot_kwargs)

        except TimeoutError as e:
            logger.error(f"Connection timeout: Failed to connect to Reachy Mini daemon. Details: {e}")
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except ConnectionError as e:
            logger.error(f"Connection failed: Unable to establish connection to Reachy Mini. Details: {e}")
            log_connection_troubleshooting(logger, args.robot_name)
            sys.exit(1)

        except Exception as e:
            logger.error(f"Unexpected error during robot initialization: {type(e).__name__}: {e}")
            logger.error("Please check your configuration and try again.")
            sys.exit(1)

    try:
        camera_worker, vision_processor = initialize_camera_and_vision(args, robot)
    except CameraVisionInitializationError as e:
        logger.error("Failed to initialize camera/vision: %s", e)
        sys.exit(1)

    movement_manager = MovementManager(
        current_robot=robot,
        camera_worker=camera_worker,
    )

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        vision_processor=vision_processor,
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

    def build_handler(startup_voice: Optional[str] = None) -> ConversationHandler:
        """Build a realtime handler for the current runtime backend config."""
        if is_gemini_model():
            from reachy_mini_conversation_app.gemini_live import GeminiLiveHandler

            logger.info(
                "Using %s via GeminiLiveHandler",
                get_backend_label(config.BACKEND_PROVIDER),
            )
            return GeminiLiveHandler(
                deps,
                gradio_mode=args.gradio,
                instance_path=instance_path,
                startup_voice=startup_voice,
            )
        if config.BACKEND_PROVIDER == HF_BACKEND:
            from reachy_mini_conversation_app.huggingface_realtime import HuggingFaceRealtimeHandler

            hf_connection_selection = get_hf_connection_selection()
            transport_label = (
                "Hugging Face direct websocket"
                if hf_connection_selection.mode == HF_LOCAL_CONNECTION_MODE and hf_connection_selection.has_target
                else "Hugging Face session proxy"
            )
            logger.info(
                "Using %s via Hugging Face realtime handler (%s)",
                get_backend_label(config.BACKEND_PROVIDER),
                transport_label,
            )
            return HuggingFaceRealtimeHandler(
                deps,
                gradio_mode=args.gradio,
                instance_path=instance_path,
                startup_voice=startup_voice,
            )

        from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler

        logger.info(
            "Using %s via OpenAI realtime handler (OpenAI Realtime API)",
            get_backend_label(config.BACKEND_PROVIDER),
        )
        return OpenaiRealtimeHandler(
            deps,
            gradio_mode=args.gradio,
            instance_path=instance_path,
            startup_voice=startup_voice,
        )

    handler = build_handler(startup_settings.voice)

    stream_manager: gr.Blocks | LocalStream | None = None

    if args.gradio:
        from reachy_mini_conversation_app.gradio_personality import PersonalityUI

        personality_ui = PersonalityUI()
        personality_ui.create_components()
        additional_inputs: list[Any] = [chatbot, *personality_ui.additional_inputs_ordered()]

        if config.BACKEND_PROVIDER in {OPENAI_BACKEND, GEMINI_BACKEND}:
            uses_gemini_backend = is_gemini_model()
            api_key_textbox = gr.Textbox(
                label="GEMINI_API_KEY" if uses_gemini_backend else "OPENAI API Key",
                type="password",
                value=(os.getenv("GEMINI_API_KEY") if uses_gemini_backend else os.getenv("OPENAI_API_KEY"))
                if not get_space()
                else "",
            )
            additional_inputs.insert(1, api_key_textbox)

        stream = Stream(
            handler=handler,
            mode="send-receive",
            modality="audio",
            additional_inputs=additional_inputs,
            additional_outputs=[chatbot],
            additional_outputs_handler=update_chatbot,
            ui_args={"title": "Talk with Reachy Mini"},
        )
        stream_manager = stream.ui
        if not settings_app:
            app = FastAPI()
        else:
            app = settings_app

        personality_ui.wire_events(handler, stream_manager)

        app = gr.mount_gradio_app(app, stream.ui, path="/")
    else:
        # In headless mode, wire settings_app + instance_path to console LocalStream
        stream_manager = LocalStream(
            handler,
            robot,
            settings_app=settings_app,
            instance_path=instance_path,
            handler_factory=build_handler,
            startup_voice=startup_settings.voice,
        )

    # Each async service → its own thread/loop
    movement_manager.start()
    # Audio-reactive head motion is driven by the daemon's wobbler, which
    # taps the media pipeline at push_audio_sample. In headless mode the
    # console stream pushes assistant audio through that pipeline directly.
    # In Gradio mode audio plays in the browser; the handler additionally
    # taps the same call to keep the wobbler fed (see
    # BaseRealtimeHandler._tap_audio_for_daemon_wobbler) — mute the robot
    # speaker to avoid double playback.
    robot.enable_wobbling()
    saved_speaker_volume: int | None = None
    if args.gradio:
        # LocalStream.launch() starts the playback pipeline in headless mode.
        # In Gradio mode nothing else does, and push_audio_sample is a no-op
        # until the pipeline is in PLAYING state — so the wobbler stays idle.
        try:
            robot.media.start_playing()
        except Exception as exc:
            logger.warning(f"Failed to start media playback for Gradio mode: {exc}")
        # Audio plays in the browser; silence the robot speaker so we don't
        # hear it twice. GET via REST to read the current level (no side
        # effects), SET via the WS protocol so the daemon does not fire its
        # "test sound" played by POST /api/volume/set on each call.
        try:
            volume_url = f"http://{robot.client.host}:{robot.client.port}/api/volume/current"
            saved_speaker_volume = int(httpx.get(volume_url, timeout=2).json()["volume"])
            robot.client.send_command(SetVolumeCmd(volume=0))
            logger.info(f"Muted robot speaker for Gradio mode (saved volume: {saved_speaker_volume})")
        except Exception as exc:
            logger.warning(f"Could not mute robot speaker: {exc}")
            saved_speaker_volume = None
    if camera_worker:
        camera_worker.start()

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
        try:
            robot.disable_wobbling()
        except Exception as e:
            logger.debug(f"Error disabling wobbling during shutdown: {e}")
        if saved_speaker_volume is not None:
            try:
                robot.client.send_command(SetVolumeCmd(volume=saved_speaker_volume))
                logger.info(f"Restored robot speaker volume to {saved_speaker_volume}")
            except Exception as e:
                logger.debug(f"Error restoring speaker volume during shutdown: {e}")
        if camera_worker:
            camera_worker.stop()

        # Ensure media is explicitly closed before disconnecting
        try:
            robot.media.close()
        except Exception as e:
            logger.debug(f"Error closing media during shutdown: {e}")

        # prevent connection to keep alive some threads
        robot.client.disconnect()
        time.sleep(1)
        logger.info("Shutdown complete.")


class ReachyMiniConversationApp(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for the conversation app."""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = False

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Run the Reachy Mini conversation app."""
        asyncio.set_event_loop(asyncio.new_event_loop())

        args, _ = parse_args()

        instance_path = self._get_instance_path().parent
        run(
            args,
            robot=reachy_mini,
            app_stop_event=stop_event,
            settings_app=self.settings_app,
            instance_path=instance_path,
        )


if __name__ == "__main__":
    app = ReachyMiniConversationApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
