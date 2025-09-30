"""Entrypoint for the Reachy Mini conversation demo."""

import os

import gradio as gr
from fastapi import FastAPI
from fastrtc import Stream
from reachy_mini import ReachyMini

from reachy_mini_conversation_demo.audio.head_wobbler import HeadWobbler
from reachy_mini_conversation_demo.moves import MovementManager
from reachy_mini_conversation_demo.openai_realtime import OpenaiRealtimeHandler
from reachy_mini_conversation_demo.tools import ToolDependencies
from reachy_mini_conversation_demo.utils import (
    handle_vision_stuff,
    parse_args,
    setup_logger,
)


def update_chatbot(chatbot: list[dict], response: dict):
    """Update the chatbot with AdditionalOutputs."""
    chatbot.append(response)
    return chatbot


def main():
    """Entrypoint for the Reachy Mini conversation demo."""
    args = parse_args()

    logger = setup_logger(args.debug)
    logger.info("Starting Reachy Mini Conversation Demo")

    robot = ReachyMini()

    camera_worker, head_tracker, vision_manager = handle_vision_stuff(args, robot)

    movement_manager = MovementManager(
        current_robot=robot,
        head_tracker=head_tracker,
        camera_worker=camera_worker,
    )

    head_wobbler = HeadWobbler(set_offsets=movement_manager.set_offsets)

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        vision_manager=vision_manager,
        head_wobbler=head_wobbler,
    )
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Current file absolute path: {current_file_path}")
    chatbot = gr.Chatbot(
        type="messages",
        resizable=True,
        avatar_images=(
            os.path.join(current_file_path, "images", "user_avatar.png"),
            os.path.join(current_file_path, "images", "reachymini_avatar.png"),
        ),
    )
    logger.info(f"Chatbot avatar images: {chatbot.avatar_images}")

    handler = OpenaiRealtimeHandler(deps)
    stream = Stream(
        handler=handler,
        mode="send-receive",
        modality="audio",
        additional_inputs=[chatbot],
        additional_outputs=[chatbot],
        additional_outputs_handler=update_chatbot,
        ui_args={"title": "Talk with Reachy Mini"},
    )

    app = FastAPI()
    app = gr.mount_gradio_app(app, stream.ui, path="/")

    # Each async service → its own thread/loop
    movement_manager.start()
    head_wobbler.start()
    if camera_worker:
        camera_worker.start()

    try:
        stream.ui.launch()
    except KeyboardInterrupt:
        logger.info("Exiting...")

    finally:
        movement_manager.stop()
        head_wobbler.stop()
        if camera_worker:
            camera_worker.stop()

        # prevent connection to keep alive some threads
        robot.client.disconnect()


if __name__ == "__main__":
    main()
