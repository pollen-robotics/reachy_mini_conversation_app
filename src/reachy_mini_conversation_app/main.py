"""Entrypoint for the Reachy Mini conversation app."""

import os
import sys
from typing import Any, Dict, List

import gradio as gr
from fastapi import FastAPI
from fastrtc import Stream

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.moves import MovementManager
from reachy_mini_conversation_app.tools import ToolDependencies
from reachy_mini_conversation_app.utils import (
    parse_args,
    setup_logger,
    handle_vision_stuff,
)
from reachy_mini_conversation_app.console import LocalStream
from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler
from reachy_mini_conversation_app.audio.head_wobbler import HeadWobbler
from reachy_mini_conversation_app.config import config


def update_chatbot(chatbot: List[Dict[str, Any]], response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Update the chatbot with AdditionalOutputs."""
    chatbot.append(response)
    return chatbot


def main() -> None:
    """Entrypoint for the Reachy Mini conversation app."""
    args = parse_args()

    logger = setup_logger(args.debug)
    logger.info(f"Starting Reachy Mini Conversation App with {args.agent.upper()} agent")

    # Validate agent-specific requirements
    if args.agent == "elevenlabs":
        if args.gradio:
            logger.error("ElevenLabs agent does not support Gradio mode yet. Please run without --gradio flag.")
            sys.exit(1)
        
        # Check for ElevenLabs configuration
        if not config.ELEVENLABS_AGENT_ID:
            logger.error(
                "ELEVENLABS_AGENT_ID is not set in .env file. Please add it:\n"
                "  ELEVENLABS_AGENT_ID=your_agent_id_here"
            )
            sys.exit(1)

    if args.no_camera and args.head_tracker is not None:
        logger.warning("Head tracking is not activated due to --no-camera.")

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

    stream_manager = None

    # Initialize based on agent provider
    if args.agent == "openai":
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
    
    elif args.agent == "elevenlabs":
        from reachy_mini_conversation_app.elevenlabs_agent import (
            ElevenLabsStream,
            ElevenLabsConfig,
        )
        
        elevenlabs_config = ElevenLabsConfig(
            agent_id=config.ELEVENLABS_AGENT_ID,
            api_key=config.ELEVENLABS_API_KEY,
            requires_auth=bool(config.ELEVENLABS_API_KEY),
        )
        
        stream_manager = ElevenLabsStream(elevenlabs_config, robot, deps)

    # Each async service → its own thread/loop
    movement_manager.start()
    
    # Head wobbler only needed for OpenAI (audio-reactive motion)
    if args.agent == "openai":
        head_wobbler.start()
    
    if camera_worker:
        camera_worker.start()
    if vision_manager:
        vision_manager.start()

    try:
        if stream_manager:
            stream_manager.launch()
        else:
            logger.error("No stream manager initialized")
    except KeyboardInterrupt:
        logger.info("Keyboard interruption in main thread... closing server.")
    finally:
        # Stop the stream manager and its pipelines
        if stream_manager:
            stream_manager.close()

        # Stop other services
        movement_manager.stop()
        if args.agent == "openai":
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
