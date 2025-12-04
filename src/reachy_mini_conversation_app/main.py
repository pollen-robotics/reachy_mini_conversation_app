"""Entrypoint for the Reachy Mini conversation app."""

import os
import sys
from typing import Any, Dict, List

import gradio as gr
from fastapi import FastAPI
from fastrtc import Stream

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.moves import MovementManager
from reachy_mini_conversation_app.utils import (
    parse_args,
    setup_logger,
    handle_vision_stuff,
)
from reachy_mini_conversation_app.console import LocalStream
from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.audio.head_wobbler import HeadWobbler


def update_chatbot(chatbot: List[Dict[str, Any]], response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update the chatbot with AdditionalOutputs."""
    chatbot.append(response)
    return chatbot


def main() -> None:
    """Entrypoint for the Reachy Mini conversation app."""
    args = parse_args()

    logger = setup_logger(args.debug)
    logger.info("Starting Reachy Mini Conversation App")

    if args.no_camera and args.head_tracker is not None:
        logger.warning("Head tracking is not activated due to --no-camera.")

    # Initialize robot with appropriate backend
    # TODO: Implement dynamic robot connection detection
    # Automatically detect and connect to available Reachy Mini robot(s!)
    # Priority checks (in order):
    #   1. Reachy Lite connected directly to the host
    #   2. Reachy Mini daemon running on localhost (same device)
    #   3. Reachy Mini daemon on local network (same subnet)

    if args.wireless_version and not args.on_device:
        logger.info("Using WebRTC backend for fully remote wireless version")
        robot = ReachyMini(media_backend="webrtc", localhost_only=False)
    elif args.wireless_version and args.on_device:
        logger.info("Using GStreamer backend for on-device wireless version")
        robot = ReachyMini(media_backend="gstreamer")
    else:
        logger.info("Using default backend for lite version")
        robot = ReachyMini(media_backend="default")

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

    handler = OpenaiRealtimeHandler(deps)

    stream_manager: gr.Blocks | LocalStream | None = None

    if args.gradio:
        stream = Stream(
            handler=handler,
            mode="send-receive",
            modality="audio",
            additional_inputs=[chatbot],
            additional_outputs=[chatbot],
            additional_outputs_handler=update_chatbot,
            ui_args={"title": "Talk with Reachy Mini"},
        )
        stream_manager = stream.ui
        app = FastAPI()
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

    try:
        stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
    finally:
        # Stop the stream manager and its pipelines
        stream_manager.close()

        # Stop other services
        movement_manager.stop()
        head_wobbler.stop()
        if camera_worker:
            camera_worker.stop()
        if vision_manager:
            vision_manager.stop()

        # prevent connection to keep alive some threads
        robot.client.disconnect()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
