from threading import Thread  # noqa: D100

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


def main():
    """Entrypoint for the Reachy Mini conversation demo."""
    args = parse_args()

    setup_logger(args.debug)
    
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

    handler = OpenaiRealtimeHandler(deps)
    stream = Stream(handler=handler, mode="send-receive", modality="audio")

    # UI bloquante → thread standard
    ui_thread = Thread(target=stream.ui.launch, daemon=True)
    ui_thread.start()

    # Chaque service async → son propre thread/loop
    move_thread = AioTaskThread(movement_manager.enable)  # loop A
    wobbler_thread = AioTaskThread(head_wobbler.enable)  # loop B
    cam_thread = AioTaskThread(camera_worker.enable) if camera_worker else None

    move_thread.start()
    wobbler_thread.start()
    if cam_thread:
        cam_thread.start()

    # lier les loops pour la communication thread-safe
    head_wobbler.bind_loops(
        consumer_loop=wobbler_thread.loop,
        movement_loop=move_thread.loop,
    )

    try:
        ui_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        move_thread.request_stop()
        wobbler_thread.request_stop()
        if cam_thread:
            cam_thread.request_stop()

        move_thread.join()
        wobbler_thread.join()
        if cam_thread:
            cam_thread.join()


if __name__ == "__main__":
    main()
