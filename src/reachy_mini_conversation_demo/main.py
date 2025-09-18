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
    AioTaskThread,
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

    camera, camera_worker, head_tracker, vision_manager = handle_vision_stuff(
        args, robot
    )

    movement_manager = MovementManager(
        current_robot=robot,
        head_tracker=head_tracker,
        camera=camera,
        camera_worker=camera_worker,
    )

    head_wobbler = HeadWobbler(set_offsets=movement_manager.set_offsets)

    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera=camera,
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
    move_thread = AioTaskThread(movement_manager.enable)  # loop A
    wobbler_thread = AioTaskThread(head_wobbler.enable)  # loop B
    cam_thread = AioTaskThread(camera_worker.enable) if camera_worker else None

    move_thread.start()
    wobbler_thread.start()
    if cam_thread:
        cam_thread.start()

    # Link the loops for thread-safe communication
    head_wobbler.bind_loops(
        consumer_loop=wobbler_thread.loop,
        movement_loop=move_thread.loop,
    )

    try:
        stream.ui.launch()
    except KeyboardInterrupt:
        logger.info("Exiting...")

    finally:
        move_thread.request_stop()
        wobbler_thread.request_stop()
        if cam_thread:
            cam_thread.request_stop()

        move_thread.join()
        wobbler_thread.join()
        if cam_thread:
            cam_thread.join()

        # prevent connection to keep alive some threads
        robot.client.disconnect()


if __name__ == "__main__":
    main()
